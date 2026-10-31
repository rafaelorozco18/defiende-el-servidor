"""Exporta los runs de MLflow a evidencias/mlflow/ y construye evidencias/experimentos.csv.

La exportación es consultable sin el servidor local: un JSON por run (info, params, métricas, tags)
y sus artefactos ligeros (metricas_val.json, val_scores.csv). Los binarios de v1/v2 viven en Hugging Face.
"""

from __future__ import annotations

import shutil

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from .comun import CONFIG_EXP, EVIDENCIAS, EXPERIMENTO_MLFLOW, RAIZ, TRACKING_URI, cargar_yaml, guardar_json

ARTEFACTOS_EXPORTADOS = ("metricas_val.json", "val_scores.csv")


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    cliente = MlflowClient()
    experimento = cliente.get_experiment_by_name(EXPERIMENTO_MLFLOW)
    cexp = cargar_yaml(CONFIG_EXP)
    conclusiones = cargar_yaml(RAIZ / "configs" / "conclusiones.yaml") or {}

    # Último run de cada configuración declarada.
    runs = {}
    for r in cliente.search_runs([experimento.experiment_id], order_by=["attributes.start_time ASC"]):
        if r.data.tags.get("exp_id") in cexp["experimentos"]:
            runs[r.data.tags["exp_id"]] = r

    destino = EVIDENCIAS / "mlflow"
    shutil.rmtree(destino, ignore_errors=True)
    filas = []
    for exp_id in cexp["experimentos"]:
        r = runs[exp_id]
        if exp_id in conclusiones:
            cliente.set_tag(r.info.run_id, "conclusion", conclusiones[exp_id].strip())
            r = cliente.get_run(r.info.run_id)
        carpeta = destino / f"{exp_id}__{r.info.run_id}"
        guardar_json({
            "run_id": r.info.run_id,
            "run_name": r.info.run_name,
            "experiment_id": r.info.experiment_id,
            "status": r.info.status,
            "start_time_ms": r.info.start_time,
            "end_time_ms": r.info.end_time,
            "artifact_uri": r.info.artifact_uri.replace(str(RAIZ), "<repo>"),
            "params": r.data.params,
            "metrics": r.data.metrics,
            "tags": {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.log-model")},
            "artefactos": [a.path for a in cliente.list_artifacts(r.info.run_id)]
                          + [f"joblib/{a.path.split('/')[-1]}" for a in cliente.list_artifacts(r.info.run_id, "joblib")],
        }, carpeta / "run.json")
        for nombre in ARTEFACTOS_EXPORTADOS:
            shutil.copy(cliente.download_artifacts(r.info.run_id, nombre), carpeta / nombre)

        m, t, p = r.data.metrics, r.data.tags, r.data.params
        filas.append({
            "exp_id": exp_id,
            "familia": t["familia"],
            "hipotesis": t["hipotesis"],
            "cambio": t["cambio"],
            "modelo": p["modelo"],
            "preproc": p["preproc"],
            "parametros": "; ".join(f"{k.removeprefix('modelo__')}={v}" for k, v in sorted(p.items()) if k.startswith("modelo__")),
            "config": f"configs/experimentos.yaml#{exp_id}",
            "semilla": p["semilla"],
            "git_commit": t["git_commit"],
            "dvc_commit": t["dvc_commit"],
            "dvc_raw_md5": t["dvc_raw_md5"],
            "dvc_prepared_md5": t["dvc_prepared_md5"],
            "run_id": r.info.run_id,
            "umbral": float(p["umbral"]),
            "n_val": int(p["n_val"]),
            "recall_ataque": m["val_recall_ataque"],
            "precision_ataque": m["val_precision_ataque"],
            "fpr": m["val_fpr"],
            "macro_f1": m["val_macro_f1"],
            "accuracy": m["val_accuracy"],
            "tn": int(m["val_tn"]), "fp": int(m["val_fp"]), "fn": int(m["val_fn"]), "tp": int(m["val_tp"]),
            "model_sha256": t["model_sha256"],
            "tamano_modelo_mb": round(m["tamano_modelo_mb"], 3),
            "tiempo_entrenamiento_s": round(m["tiempo_entrenamiento_s"], 1),
            "conclusion": t.get("conclusion", ""),
        })
    pd.DataFrame(filas).to_csv(EVIDENCIAS / "experimentos.csv", index=False, float_format="%.6f")
    print(pd.DataFrame(filas)[["exp_id", "run_id", "recall_ataque", "precision_ataque", "fpr", "macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
