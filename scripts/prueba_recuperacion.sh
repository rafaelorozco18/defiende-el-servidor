#!/usr/bin/env bash
# Prueba obligatoria de recuperación en una carpeta y un entorno nuevos.
#
#   bash scripts/prueba_recuperacion.sh [REVISION_GIT] [DESTINO_EVIDENCIAS]
#
# - Clona el repo de GitHub en una carpeta temporal nueva y hace checkout de REVISION_GIT (default: main).
# - Crea un entorno virtual nuevo con caché de uv vacía (uv sync --frozen: dependencias fijadas en uv.lock).
# - dvc pull desde el remote HTTP (Hugging Face) con caché DVC vacía y verifica los SHA-256.
# - Descarga v1 y v2 desde Hugging Face por SHA completo con HF_HOME vacío, predice las 25 filas
#   fijas de validación y compara con las salidas de referencia.
# - Copia log, salidas, hashes y reportes a DESTINO_EVIDENCIAS (default: evidencias/recuperacion/prueba).
set -euo pipefail

REPO_URL="https://github.com/rafaelorozco18/defiende-el-servidor.git"
REV="${1:-main}"
ORIGEN="$(pwd)"
DESTINO="$(realpath -m "${2:-evidencias/recuperacion/prueba}")"
TRABAJO="$(mktemp -d -t defiende-recuperacion-XXXXXX)"
mkdir -p "$DESTINO"
LOG="$DESTINO/log.txt"
exec > >(tee "$LOG") 2>&1

export UV_CACHE_DIR="$TRABAJO/uv-cache"
export HF_HOME="$TRABAJO/hf-home"
export MLFLOW_DISABLE_AGENT_HINT=1
unset VIRTUAL_ENV

paso() { echo; echo "=== $(date -u +%FT%TZ) · $*"; }

paso "Entorno"
echo "carpeta nueva: $TRABAJO"
uname -srm; uv --version; git --version

paso "1. Código: clonar y fijar revisión $REV"
git clone --quiet "$REPO_URL" "$TRABAJO/repo"
cd "$TRABAJO/repo"
git checkout --quiet "$REV"
echo "commit: $(git rev-parse HEAD)"
time uv sync --frozen --quiet
uv run python --version

paso "2. Datos: dvc pull desde el remote entregado (caché vacía)"
uv run dvc remote list
test ! -d .dvc/cache && echo "caché DVC vacía antes del pull"
time uv run dvc pull
uv run python -m defiende.recuperacion huellas

leer() { uv run python -c "import json;print(json.load(open('evidencias/versiones.json'))['$1']['huggingface']['$2'])"; }
HF_REPO="$(leer v1 repo)"
V1_SHA="$(leer v1 revision_sha)"
V2_SHA="$(leer v2 revision_sha)"
SALIDAS="$TRABAJO/salidas"
mkdir -p "$SALIDAS"

for V in v1 v2; do
  SHA="$V1_SHA"; [ "$V" = v2 ] && SHA="$V2_SHA"
  paso "3-4. Modelo $V: descarga de $HF_REPO@$SHA y predicción"
  uv run python -m defiende.predecir --repo "$HF_REPO" --revision "$SHA" \
    --entrada evidencias/recuperacion/entrada_val.csv --salida "$SALIDAS/recuperada_$V.csv"
  paso "5. Comparación $V con la salida de referencia"
  uv run python -m defiende.recuperacion comparar --version "$V" \
    --salida "$SALIDAS/recuperada_$V.csv" --reporte "../salidas/comparacion_$V.json"
done

paso "Hashes"
sha256sum evidencias/recuperacion/entrada_val.csv evidencias/recuperacion/referencia_v*.csv "$SALIDAS"/* \
  | sed "s#$TRABAJO/##" | tee "$SALIDAS/hashes.txt"
find "$HF_HOME" -name model.joblib -exec sha256sum {} \; | sed "s#$TRABAJO/##"

cp "$SALIDAS"/* "$DESTINO/"
paso "RESULTADO: recuperación correcta para v1 y v2 (revisión de código $(git rev-parse HEAD))"
cd "$ORIGEN"
