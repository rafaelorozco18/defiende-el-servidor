"""Descarga una versión del modelo desde Hugging Face por SHA y predice sobre un CSV.

Uso:
  python -m defiende.predecir --repo USUARIO/REPO --revision <SHA completo> \
      --entrada evidencias/recuperacion/entrada_val.csv --salida salida.csv

También funciona con una carpeta local (--local hf/v1) para generar salidas de referencia.
Este archivo se copia tal cual a cada revisión del repo de modelos (predecir.py), así que solo
depende de pandas, joblib, scikit-learn y huggingface_hub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def cargar_solucion(repo: str | None = None, revision: str | None = None, local: str | None = None):
    """Devuelve (pipeline, config). Verifica que el sha256 del modelo coincide con config.json."""
    if local:
        base = Path(local)
        ruta_cfg, ruta_modelo = base / "config.json", base / "model.joblib"
    else:
        from huggingface_hub import hf_hub_download

        if not revision or len(revision) != 40:
            raise SystemExit("Indica la revisión con el SHA completo de 40 caracteres")
        ruta_cfg = Path(hf_hub_download(repo, "config.json", revision=revision))
        ruta_modelo = Path(hf_hub_download(repo, "model.joblib", revision=revision))
    config = json.loads(ruta_cfg.read_text(encoding="utf-8"))
    sha = sha256_archivo(ruta_modelo)
    if sha != config["model_sha256"]:
        raise SystemExit(f"sha256 del modelo {sha} no coincide con config.json {config['model_sha256']}")
    return joblib.load(ruta_modelo), config


def predecir(pipe, config: dict, df: pd.DataFrame) -> pd.DataFrame:
    scores = pipe.predict_proba(df[config["variables_entrada"]])[:, 1]
    return pd.DataFrame({
        "row_id": df["row_id"],
        "score_attack": scores,
        # Regla de decisión: ataque (1) si score_attack >= umbral.
        "y_pred": (scores >= config["umbral"]).astype(int),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo")
    ap.add_argument("--revision")
    ap.add_argument("--local")
    ap.add_argument("--entrada", required=True)
    ap.add_argument("--salida", required=True)
    args = ap.parse_args()

    pipe, config = cargar_solucion(args.repo, args.revision, args.local)
    df = pd.read_csv(args.entrada, encoding="utf-8-sig")
    if "row_id" not in df.columns:
        df.insert(0, "row_id", [f"entrada:{i}" for i in range(1, len(df) + 1)])
    salida = predecir(pipe, config, df)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(args.salida, index=False, float_format="%.12f")
    print(f"{config['version']} ({config['exp_id']}, umbral {config['umbral']}, sha256 {config['model_sha256'][:12]}…): "
          f"{len(salida)} filas, {int(salida['y_pred'].sum())} ataques -> {args.salida}")


if __name__ == "__main__":
    main()
