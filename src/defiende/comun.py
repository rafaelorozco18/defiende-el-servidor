"""Utilidades compartidas: rutas, lectura de datos, huellas, métricas y referencias Git/DVC."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[2]
CONFIG_DATOS = RAIZ / "configs" / "datos.yaml"
CONFIG_EXP = RAIZ / "configs" / "experimentos.yaml"
PREPARADOS = RAIZ / "data" / "prepared"
EVIDENCIAS = RAIZ / "evidencias"
MODELOS = RAIZ / "models"
TRACKING_URI = f"sqlite:///{RAIZ / 'mlflow.db'}"
EXPERIMENTO_MLFLOW = "defiende-el-servidor"


def cargar_yaml(ruta: Path) -> dict:
    return yaml.safe_load(ruta.read_text(encoding="utf-8"))


def guardar_json(obj, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_csv(ruta: Path) -> pd.DataFrame:
    # Los CSV publicados traen BOM UTF-8 en la cabecera ("﻿id").
    return pd.read_csv(ruta, encoding="utf-8-sig")


def columnas_predictoras(df: pd.DataFrame, excluidas: list[str]) -> list[str]:
    return [c for c in df.columns if c not in excluidas and c not in ("row_id", "grupo")]


def huella_grupo(df: pd.DataFrame, features: list[str]) -> pd.Series:
    """sha256 de los valores predictores de cada fila: filas con la misma huella son entradas idénticas."""
    texto = df[features].astype(str).agg("|".join, axis=1)
    return texto.map(lambda s: hashlib.sha256(s.encode()).hexdigest()[:16])


# --------------------------------------------------------------------------- métricas
def metricas_binarias(y_true, y_pred) -> dict:
    """Métricas con ataque (1) como clase positiva.

    Denominadores cero: precision sin predicciones positivas, recall sin positivos reales o
    FPR sin negativos reales se reportan como null y se listan en `denominadores_cero`.
    Para macro-F1 una clase con F1 indefinido aporta 0.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    ceros = []

    def div(num, den, nombre):
        if den == 0:
            ceros.append(nombre)
            return None
        return num / den

    precision = div(tp, tp + fp, "precision_ataque")
    recall = div(tp, tp + fn, "recall_ataque")
    fpr = div(fp, fp + tn, "fpr")
    prec_norm = div(tn, tn + fn, "precision_normal")
    rec_norm = div(tn, tn + fp, "recall_normal")

    def f1(p, r):
        if p is None or r is None or p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    f1_ataque = f1(precision, recall)
    f1_normal = f1(prec_norm, rec_norm)
    return {
        "recall_ataque": recall,
        "precision_ataque": precision,
        "fpr": fpr,
        "f1_ataque": f1_ataque,
        "f1_normal": f1_normal,
        "macro_f1": (f1_ataque + f1_normal) / 2,
        "accuracy": (tp + tn) / len(y_true) if len(y_true) else None,
        "matriz_confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "n": int(len(y_true)),
        "denominadores_cero": ceros,
    }


def aplicar_umbral(scores, umbral: float) -> np.ndarray:
    """Regla de decisión: ataque si score_attack >= umbral."""
    return (np.asarray(scores) >= umbral).astype(int)


# --------------------------------------------------------------------------- referencias
def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True, check=True).stdout.strip()


def git_commit() -> str:
    return git("rev-parse", "HEAD")


def git_sucio() -> bool:
    return bool(git("status", "--porcelain", "--untracked-files=no"))


def referencias_dvc() -> dict:
    """md5 DVC de los datos crudos (data/raw.dvc) y preparados (dvc.lock), más el commit que los fija."""
    raw = cargar_yaml(RAIZ / "data" / "raw.dvc")["outs"][0]
    lock = cargar_yaml(RAIZ / "dvc.lock")
    prep = next(o for o in lock["stages"]["preparar"]["outs"] if o["path"] == "data/prepared")
    return {
        "dvc_raw_md5": raw["md5"],
        "dvc_prepared_md5": prep["md5"],
        "dvc_commit": git("log", "-1", "--format=%H", "--", "dvc.lock", "data/raw.dvc"),
    }
