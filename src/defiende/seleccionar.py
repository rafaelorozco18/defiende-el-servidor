"""Define las versiones de la solución usando SOLO validación.

  python -m defiende.seleccionar v1   # v1 = configuración `v1.exp_id` con su umbral fijo
  python -m defiende.seleccionar v2   # aplica la regla `seleccion` declarada en configs/experimentos.yaml

Escribe evidencias/versiones.json (ficha de cada versión) y, para v2, evidencias/seleccion.json y
evidencias/umbral_v2.csv (curva de umbrales en validación).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .comun import CONFIG_EXP, EVIDENCIAS, aplicar_umbral, cargar_yaml, guardar_json, metricas_binarias

RUTA_VERSIONES = EVIDENCIAS / "versiones.json"


def val_scores(exp_id: str, run_id: str) -> pd.DataFrame:
    return pd.read_csv(EVIDENCIAS / "mlflow" / f"{exp_id}__{run_id}" / "val_scores.csv")


def resumen(m: dict) -> dict:
    return {k: m[k] for k in ("recall_ataque", "precision_ataque", "fpr", "macro_f1", "accuracy", "matriz_confusion",
                              "n", "denominadores_cero")}


def ficha(fila, umbral: float, razon: str) -> dict:
    vs = val_scores(fila["exp_id"], fila["run_id"])
    met = metricas_binarias(vs["y_true"], aplicar_umbral(vs["score_attack"], umbral))
    return {
        "exp_id": fila["exp_id"], "run_id": fila["run_id"], "familia": fila["familia"],
        "modelo": fila["modelo"], "preproc": fila["preproc"],
        "git_commit": fila["git_commit"], "dvc_commit": fila["dvc_commit"],
        "dvc_raw_md5": fila["dvc_raw_md5"], "dvc_prepared_md5": fila["dvc_prepared_md5"],
        "model_sha256": fila["model_sha256"], "umbral": umbral, "regla_decision": "ataque si score_attack >= umbral",
        "metricas_val": resumen(met), "razon": razon,
        "huggingface": {"repo": None, "revision_sha": None, "tag": None},
    }


def guardar_version(nombre: str, datos: dict) -> None:
    versiones = json.loads(RUTA_VERSIONES.read_text(encoding="utf-8")) if RUTA_VERSIONES.exists() else {}
    versiones[nombre] = datos
    guardar_json(versiones, RUTA_VERSIONES)


def seleccionar_v1(cfg: dict, tabla: pd.DataFrame) -> None:
    v1 = cfg["v1"]
    fila = tabla[tabla["exp_id"] == v1["exp_id"]].iloc[0]
    datos = ficha(fila, float(v1["umbral"]), v1["razon"])
    guardar_version("v1", datos)
    print("v1", datos["exp_id"], datos["metricas_val"])


def seleccionar_v2(cfg: dict, tabla: pd.DataFrame) -> None:
    regla = cfg["seleccion"]
    v1 = json.loads(RUTA_VERSIONES.read_text(encoding="utf-8"))["v1"]
    fila = tabla.sort_values(regla["metrica_modelo"], ascending=False).iloc[0]

    vs = val_scores(fila["exp_id"], fila["run_id"])
    g = regla["umbral"]["rejilla"]
    curva = []
    for u in np.round(np.arange(g["inicio"], g["fin"] + 1e-9, g["paso"]), 4):
        m = metricas_binarias(vs["y_true"], aplicar_umbral(vs["score_attack"], u))
        curva.append({"umbral": u, **{k: m[k] for k in ("recall_ataque", "precision_ataque", "fpr", "macro_f1")},
                      **m["matriz_confusion"]})
    curva = pd.DataFrame(curva)
    curva.to_csv(EVIDENCIAS / "umbral_v2.csv", index=False, float_format="%.6f")

    fpr_max = regla["umbral"]["fpr_max"]
    factibles = curva[curva["fpr"] <= fpr_max]
    # Mayor recall con FPR <= fpr_max; en empate, el umbral más alto (menos alertas).
    umbral = float(factibles.sort_values(["recall_ataque", "umbral"], ascending=[False, False]).iloc[0]["umbral"])
    met_05 = metricas_binarias(vs["y_true"], aplicar_umbral(vs["score_attack"], 0.5))

    v2 = ficha(fila, umbral,
               f"{fila['exp_id']} tiene el mayor macro-F1 en validación a umbral 0.5 ({fila['macro_f1']:.4f}). "
               f"Su FPR a 0.5 ({met_05['fpr']:.4f}) supera el máximo operativo de {fpr_max}; el umbral {umbral} "
               f"es el que da mayor recall de ataque con FPR <= {fpr_max} en validación.")
    guardar_version("v2", v2)
    seleccion = {
        "version_elegida": "v2",
        "fecha_decision_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datos_usados_para_decidir": "solo validación (data/prepared/val.csv); test no se ha leído",
        "regla": regla,
        "candidatas": tabla[["exp_id", "run_id", "macro_f1", "recall_ataque", "precision_ataque", "fpr"]].to_dict("records"),
        "v2": v2,
        "v2_a_umbral_0_5": resumen(met_05),
        "v1_referencia": {k: v1[k] for k in ("exp_id", "run_id", "umbral", "model_sha256", "metricas_val")},
        "cambio_v1_a_v2": {
            "modelo": f"{v1['modelo']} -> {v2['modelo']}",
            "preprocesamiento": f"{v1['preproc']} -> {v2['preproc']}",
            "umbral": f"{v1['umbral']} -> {umbral}",
            "delta_val": {k: round(v2["metricas_val"][k] - v1["metricas_val"][k], 6)
                          for k in ("recall_ataque", "precision_ataque", "fpr", "macro_f1")},
        },
        "congelacion": "el commit Git que fija este archivo (con la revisión de Hugging Face) es anterior a la "
                       "evaluación en test; evaluar_test.py lo verifica y lo registra en metricas-test.json",
    }
    guardar_json(seleccion, EVIDENCIAS / "seleccion.json")
    print("v2", v2["exp_id"], "umbral", umbral, v2["metricas_val"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("version", choices=["v1", "v2"])
    args = ap.parse_args()
    cfg = cargar_yaml(CONFIG_EXP)
    tabla = pd.read_csv(EVIDENCIAS / "experimentos.csv")
    seleccionar_v1(cfg, tabla) if args.version == "v1" else seleccionar_v2(cfg, tabla)


if __name__ == "__main__":
    main()
