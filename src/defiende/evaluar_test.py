"""Evaluación final ÚNICA de la solución congelada (v2) sobre todo el archivo de test.

Requisitos que el script verifica antes de leer test:
  - evidencias/seleccion.json está commiteado y sin cambios (la decisión es anterior al test);
  - v2 tiene revisión de Hugging Face; el modelo se descarga por ese SHA y se verifica su sha256;
  - no existe ya evidencias/metricas-test.json (no se re-evalúa).

Uso: python -m defiende.evaluar_test
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from .comun import EVIDENCIAS, PREPARADOS, RAIZ, git, git_commit, guardar_json, metricas_binarias, sha256_archivo
from .predecir import cargar_solucion, predecir


def main() -> None:
    ruta_sel = EVIDENCIAS / "seleccion.json"
    salida_met = EVIDENCIAS / "metricas-test.json"
    if salida_met.exists():
        raise SystemExit("El test ya fue evaluado; no se repite la evaluación.")
    if git("status", "--porcelain", "--", str(ruta_sel)):
        raise SystemExit("seleccion.json tiene cambios sin commit: congela la decisión antes de evaluar.")
    commit_congelado = git("log", "-1", "--format=%H", "--", str(ruta_sel))
    fecha_congelado = git("log", "-1", "--format=%cI", "--", str(ruta_sel))

    sel = json.loads(ruta_sel.read_text(encoding="utf-8"))
    v2 = sel["v2"]
    hf = v2["huggingface"]
    pipe, config = cargar_solucion(hf["repo"], hf["revision_sha"])
    if config["model_sha256"] != v2["model_sha256"] or config["umbral"] != v2["umbral"]:
        raise SystemExit("La revisión descargada no es la solución seleccionada")

    ruta_test = PREPARADOS / "test.csv"
    test = pd.read_csv(ruta_test)
    pred = predecir(pipe, config, test)
    pred.insert(1, "y_true", test["label"].to_numpy())
    pred = pred[["row_id", "y_true", "score_attack", "y_pred"]]
    ruta_pred = EVIDENCIAS / "predicciones-test.csv"
    pred.to_csv(ruta_pred, index=False, float_format="%.12f")

    met = metricas_binarias(pred["y_true"], pred["y_pred"])
    guardar_json({
        "version": "v2",
        "exp_id": v2["exp_id"],
        "mlflow_run_id": v2["run_id"],
        "huggingface_repo": hf["repo"],
        "huggingface_revision_evaluada": hf["revision_sha"],
        "model_sha256": config["model_sha256"],
        "umbral": config["umbral"],
        "regla_decision": "y_pred = 1 (ataque) si score_attack >= umbral; score_attack = predict_proba[:, 1]",
        "clase_positiva": "1 = ataque",
        "dataset": {
            "archivo_original": "data/raw/UNSW_NB15_testing-set.csv",
            "sha256_original": sha256_archivo(RAIZ / "data" / "raw" / "UNSW_NB15_testing-set.csv"),
            "archivo_evaluado": "data/prepared/test.csv",
            "sha256_evaluado": sha256_archivo(ruta_test),
            "filas_evaluadas": int(len(test)),
            "filas_eliminadas": 0,
        },
        "seleccion_congelada": {"commit": commit_congelado, "fecha": fecha_congelado},
        "evaluado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "codigo_evaluacion_commit": git_commit(),
        "predicciones": {"archivo": "evidencias/predicciones-test.csv", "sha256": sha256_archivo(ruta_pred)},
        "metricas": met,
        "metricas_val_misma_solucion": v2["metricas_val"],
    }, salida_met)
    print(json.dumps({k: met[k] for k in ("recall_ataque", "precision_ataque", "fpr", "macro_f1", "matriz_confusion")}, indent=1))


if __name__ == "__main__":
    main()
