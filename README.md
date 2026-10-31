# Defiende el servidor

Clasificación binaria **normal (0) / ataque (1)** con las particiones publicadas de UNSW-NB15, con
versionado de código (Git/GitHub), datos (DVC), experimentos (MLflow) y modelos (Hugging Face Hub).
Taller evaluable de MLOps 2026-II, Universidad Externado de Colombia.

| Pieza | Dónde |
|---|---|
| Código | https://github.com/rafaelorozco18/defiende-el-servidor (tags `v1`, `v2`) |
| Datos (remote DVC) | https://huggingface.co/datasets/Rafoniloni/defiende-el-servidor-dvc (público, remote HTTP `hfstore`) |
| Modelos | https://huggingface.co/Rafoniloni/defiende-el-servidor-unsw (tags `v1`, `v2`) |
| Experimentos | MLflow local exportado en `evidencias/mlflow/` y `evidencias/experimentos.csv` |
| Índice de la entrega | [entrega.md](entrega.md) |

Cadena de trazabilidad: **commit Git → md5 DVC → run ID MLflow → revisión (SHA) de Hugging Face**.

## Requisitos

- Python **3.12** (probado con 3.12.14), Git y [uv](https://docs.astral.sh/uv/) ≥ 0.12 (probado con 0.12.21).
- Dependencias fijadas en `uv.lock` (también exportadas a `requirements.txt`).
- CPU; ~2 GB de RAM libres. No se necesitan credenciales: todos los repositorios son públicos.

## Instalación

```bash
git clone https://github.com/rafaelorozco18/defiende-el-servidor.git
cd defiende-el-servidor
uv sync --frozen
```

Sin uv: `python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pip install -e . --no-deps`
(después, quitar el prefijo `uv run` de los comandos).

## Recuperar los datos (DVC)

```bash
uv run dvc pull                              # data/raw (CSV recibidos) y data/prepared (particiones)
uv run python -m defiende.recuperacion huellas   # verifica los SHA-256 contra configs/ y auditoria.json
```

- Versión 1 de los datos: `data/raw.dvc` (los dos CSV publicados, sin modificar).
- Versión 2: salida de la etapa `preparar` en `dvc.lock` (`data/prepared/`: train, val, test con `row_id`,
  y `asignacion.csv` con el split de cada fila). Se regenera con `uv run dvc repro`.

## Reproducir un experimento

El entrenamiento exige un árbol Git limpio, para que el commit registrado en MLflow sea el código real.

```bash
uv run python -m defiende.entrenar --exp E4-hgb     # o E1-logreg-base, E2-logreg-log1p, E3-rf
uv run python -m defiende.exportar_mlflow           # regenera evidencias/mlflow y experimentos.csv
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # opcional, para explorar los runs
```

Las configuraciones, hipótesis y la regla de selección están en `configs/experimentos.yaml`.
El entrenamiento es determinista (semilla 42): se obtienen las mismas métricas de validación.

## Descargar cada modelo y predecir

Cada revisión contiene la solución completa: `model.joblib` (pipeline de preprocesamiento + estimador),
`config.json` (umbral y variables), `requirements.txt`, `predecir.py` y la model card.

```bash
# v1: regresión logística, umbral 0.5
uv run python -m defiende.predecir --repo Rafoniloni/defiende-el-servidor-unsw \
  --revision 264e987803bcae3aaa09c719df77fe8d726eb96d \
  --entrada evidencias/recuperacion/entrada_val.csv --salida salidas/v1.csv

# v2: HistGradientBoosting, umbral 0.64
uv run python -m defiende.predecir --repo Rafoniloni/defiende-el-servidor-unsw \
  --revision 64fc9774c0779005d65f79c769a4def98258065a \
  --entrada evidencias/recuperacion/entrada_val.csv --salida salidas/v2.csv
```

`predecir` verifica que el sha256 de `model.joblib` coincida con `config.json`. La salida tiene
`row_id, score_attack, y_pred`, con `y_pred = 1` si `score_attack >= umbral`.

## Prueba de recuperación en un entorno nuevo

```bash
bash scripts/prueba_recuperacion.sh 2f82c1bbf6f9afac4037a0175b12be102b8241c0
```

Clona el repo en una carpeta temporal y crea un entorno con cachés de uv, DVC y Hugging Face vacías.
Luego hace `dvc pull`, verifica las huellas, descarga v1 y v2 por SHA, predice 25 filas fijas de
validación y las compara con `evidencias/recuperacion/referencia_v*.csv`. El log de la ejecución
entregada está en `evidencias/recuperacion/prueba/`.

## Flujo completo (cómo se produjo la entrega)

```bash
uv run dvc repro                                    # auditoría + particiones
uv run python -m defiende.entrenar --exp <ID>       # x4, con commit limpio
uv run python -m defiende.exportar_mlflow
uv run python -m defiende.seleccionar v1            # luego v2, solo con validación
uv run python -m defiende.empaquetar --version v2 --repo Rafoniloni/defiende-el-servidor-unsw \
  --github https://github.com/rafaelorozco18/defiende-el-servidor
uv run python -m defiende.publicar_hf --version v2 --repo Rafoniloni/defiende-el-servidor-unsw   # requiere hf auth login
uv run python -m defiende.dvc_remote_hf --repo Rafoniloni/defiende-el-servidor-dvc               # "push" del remote DVC
git commit evidencias/seleccion.json                # congelación
uv run python -m defiende.evaluar_test              # una sola vez; se niega a repetir
```

## Estructura

```
configs/      datos.yaml (fuente, huellas, split), experimentos.yaml, conclusiones.yaml
src/defiende/ preparar, entrenar, modelos, exportar_mlflow, seleccionar, empaquetar, publicar_hf,
              predecir, evaluar_test, diagnostico_test, recuperacion, dvc_remote_hf
data/         raw.dvc (+ data/prepared vía dvc.yaml/dvc.lock)
evidencias/   auditoria.json, experimentos.csv, seleccion.json, versiones.json, umbral_v2.csv,
              metricas-test.json, predicciones-test.csv, diagnostico-test.json, mlflow/, recuperacion/
scripts/      prueba_recuperacion.sh
```

## Datos y atribución

Moustafa, N. y Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems*.
MilCIS 2015. Fuente oficial: https://research.unsw.edu.au/projects/unsw-nb15-dataset. Los CSV se descargaron
de un mirror de terceros fijado por commit (`configs/datos.yaml`), y sus SHA-256 coinciden con el anexo de la guía.
