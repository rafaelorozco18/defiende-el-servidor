"""Construcción de la solución completa: preprocesamiento + estimador en un único Pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler


def preprocesador(tipo: str, categoricas: list[str], numericas: list[str]) -> ColumnTransformer:
    if tipo == "base":
        num = StandardScaler()
        cat = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
    elif tipo == "log1p":
        # Las numéricas de UNSW-NB15 son >= 0 (verificado en la auditoría). Se usa np.log1p y no una
        # función propia para que el modelo serializado solo dependa de numpy/scikit-learn.
        num = make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler())
        cat = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
    elif tipo == "ordinal":
        num = "passthrough"
        cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        raise ValueError(f"preproc desconocido: {tipo}")
    return ColumnTransformer([("num", num, numericas), ("cat", cat, categoricas)], sparse_threshold=0)


def estimador(modelo: str, params: dict, semilla: int):
    if modelo == "logreg":
        return LogisticRegression(random_state=semilla, **params)
    if modelo == "random_forest":
        return RandomForestClassifier(random_state=semilla, **params)
    if modelo == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=semilla, **params)
    raise ValueError(f"modelo desconocido: {modelo}")


def construir(exp: dict, categoricas: list[str], numericas: list[str], semilla: int) -> Pipeline:
    return Pipeline([
        ("preproc", preprocesador(exp["preproc"], categoricas, numericas)),
        ("modelo", estimador(exp["modelo"], exp["params"], semilla)),
    ])
