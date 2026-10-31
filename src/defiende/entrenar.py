"""Entrena una configuración de configs/experimentos.yaml y la registra en MLflow.

Uso: python -m defiende.entrenar --exp E1-logreg-base

Exige un árbol Git limpio para que el commit registrado sea el código que produjo el modelo.
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn

from .comun import (
    CONFIG_DATOS,
    CONFIG_EXP,
    EXPERIMENTO_MLFLOW,
    MODELOS,
    PREPARADOS,
    TRACKING_URI,
    aplicar_umbral,
    cargar_yaml,
    columnas_predictoras,
    git_commit,
    git_sucio,
    guardar_json,
    metricas_binarias,
    referencias_dvc,
    sha256_archivo,
)
from .modelos import construir


def cargar_split(nombre: str, excluidas: list[str]):
    df = pd.read_csv(PREPARADOS / f"{nombre}.csv")
    features = columnas_predictoras(df, excluidas)
    return df, df[features], df["label"].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--permitir-sucio", action="store_true", help="solo para pruebas; no usar en runs entregados")
    args = ap.parse_args()

    if git_sucio() and not args.permitir_sucio:
        raise SystemExit("Hay cambios sin commit: haz commit antes de entrenar para que el run sea trazable.")

    cdatos, cexp = cargar_yaml(CONFIG_DATOS), cargar_yaml(CONFIG_EXP)
    exp = cexp["experimentos"][args.exp]
    semilla = cexp["semilla"]
    excluidas = cdatos["excluidas"]
    categoricas = cdatos["categoricas"]

    _, X_tr, y_tr = cargar_split("train", excluidas)
    df_val, X_val, y_val = cargar_split("val", excluidas)
    numericas = [c for c in X_tr.columns if c not in categoricas]

    pipe = construir(exp, categoricas, numericas, semilla)
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    t_fit = time.perf_counter() - t0
    scores = pipe.predict_proba(X_val)[:, 1]
    umbral = 0.5
    met = metricas_binarias(y_val, aplicar_umbral(scores, umbral))

    salida = MODELOS / args.exp
    salida.mkdir(parents=True, exist_ok=True)
    ruta_modelo = salida / "model.joblib"
    joblib.dump(pipe, ruta_modelo, compress=3)
    sha_modelo = sha256_archivo(ruta_modelo)
    pd.DataFrame({"row_id": df_val["row_id"], "y_true": y_val, "score_attack": scores}).to_csv(
        salida / "val_scores.csv", index=False, float_format="%.10f")

    refs = referencias_dvc()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENTO_MLFLOW)
    with mlflow.start_run(run_name=args.exp) as run:
        mlflow.set_tags({
            "exp_id": args.exp,
            "familia": exp["familia"],
            "hipotesis": exp["hipotesis"].strip(),
            "cambio": exp["cambio"],
            "git_commit": git_commit(),
            "git_sucio": str(git_sucio()),
            **refs,
            "model_sha256": sha_modelo,
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
        })
        mlflow.log_params({
            "modelo": exp["modelo"], "preproc": exp["preproc"], "semilla": semilla, "umbral": umbral,
            "n_train": len(y_tr), "n_val": len(y_val),
            **{f"modelo__{k}": v for k, v in exp["params"].items()},
        })
        mlflow.log_metrics({
            **{f"val_{k}": v for k, v in met.items() if isinstance(v, (int, float)) and v is not None and k != "n"},
            **{f"val_{k}": v for k, v in met["matriz_confusion"].items()},
            "tiempo_entrenamiento_s": t_fit,
            "tamano_modelo_mb": ruta_modelo.stat().st_size / 1e6,
        })
        guardar_json({"umbral": umbral, **met}, salida / "metricas_val.json")
        mlflow.log_artifact(str(salida / "metricas_val.json"))
        mlflow.log_artifact(str(salida / "val_scores.csv"))
        mlflow.log_artifact(str(ruta_modelo), artifact_path="joblib")
        mlflow.sklearn.log_model(pipe, name="modelo", serialization_format="cloudpickle")
        run_id = run.info.run_id

    guardar_json({"exp_id": args.exp, "run_id": run_id, "model_sha256": sha_modelo, "git_commit": git_commit(),
                  **refs, "umbral": umbral, "tiempo_entrenamiento_s": round(t_fit, 2)}, salida / "run.json")
    print(json.dumps({"exp": args.exp, "run_id": run_id, "t_fit_s": round(t_fit, 1),
                      **{k: met[k] for k in ("recall_ataque", "precision_ataque", "fpr", "macro_f1")}}, indent=1))


if __name__ == "__main__":
    main()
