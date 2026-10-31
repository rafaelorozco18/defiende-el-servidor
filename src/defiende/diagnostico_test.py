"""Diagnóstico POSTERIOR a la evaluación final: explica dónde cambian los errores entre validación y test.

No modifica el modelo, el umbral ni las métricas oficiales (metricas-test.json); solo describe.

Uso: python -m defiende.diagnostico_test
"""

from __future__ import annotations

import json

import pandas as pd

from .comun import EVIDENCIAS, PREPARADOS, guardar_json, metricas_binarias


def fpr_por(df: pd.DataFrame, col: str, top: int = 6) -> dict:
    normales = df[df["label"] == 0]
    g = normales.groupby(col)["y_pred"].agg(["size", "mean"]).sort_values("size", ascending=False).head(top)
    return {str(k): {"normales": int(r["size"]), "fpr": round(float(r["mean"]), 4)} for k, r in g.iterrows()}


def main() -> None:
    sel = json.loads((EVIDENCIAS / "seleccion.json").read_text(encoding="utf-8"))["v2"]
    umbral = sel["umbral"]

    val = pd.read_csv(PREPARADOS / "val.csv")
    vs = pd.read_csv(EVIDENCIAS / "mlflow" / f"{sel['exp_id']}__{sel['run_id']}" / "val_scores.csv")
    val = val.merge(vs[["row_id", "score_attack"]], on="row_id")
    val["y_pred"] = (val["score_attack"] >= umbral).astype(int)

    test = pd.read_csv(PREPARADOS / "test.csv").merge(pd.read_csv(EVIDENCIAS / "predicciones-test.csv"), on="row_id")
    vistos = set(pd.read_csv(PREPARADOS / "train.csv", usecols=["grupo"])["grupo"]) | set(val["grupo"])
    test["solapa_train_val"] = test["grupo"].isin(vistos)

    def mezcla(df, col):
        return df[df["label"] == 0][col].value_counts(normalize=True).head(5).round(4).to_dict()

    def corto(m):
        return {k: m[k] for k in ("recall_ataque", "precision_ataque", "fpr", "macro_f1", "n")}

    guardar_json({
        "nota": "Análisis posterior a la evaluación única de test; no se usó para elegir modelo ni umbral.",
        "umbral": umbral,
        "fpr_por_state": {"val": fpr_por(val, "state"), "test": fpr_por(test, "state")},
        "fpr_por_service": {"val": fpr_por(val, "service", 8), "test": fpr_por(test, "service", 8)},
        "mezcla_trafico_normal": {
            "state": {"val": mezcla(val, "state"), "test": mezcla(test, "state")},
            "service": {"val": mezcla(val, "service"), "test": mezcla(test, "service")},
        },
        "test_por_solapamiento": {
            "con_entradas_vistas_en_train_val": corto(metricas_binarias(
                test.loc[test["solapa_train_val"], "y_true"], test.loc[test["solapa_train_val"], "y_pred"])),
            "sin_solapamiento": corto(metricas_binarias(
                test.loc[~test["solapa_train_val"], "y_true"], test.loc[~test["solapa_train_val"], "y_pred"])),
        },
        "score_normales_cuantiles": {
            "val": val.loc[val["label"] == 0, "score_attack"].quantile([0.5, 0.8, 0.9, 0.95]).round(4).to_dict(),
            "test": test.loc[test["label"] == 0, "score_attack"].quantile([0.5, 0.8, 0.9, 0.95]).round(4).to_dict(),
        },
    }, EVIDENCIAS / "diagnostico-test.json")
    print((EVIDENCIAS / "diagnostico-test.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
