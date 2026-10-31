"""Prueba de recuperación.

  python -m defiende.recuperacion muestra      # fija 25 filas de validación y guarda las salidas de
                                               # referencia de v1/v2 con los artefactos originales (hf/v1, hf/v2)
  python -m defiende.recuperacion huellas      # verifica los SHA-256 de los datos recuperados con DVC
  python -m defiende.recuperacion comparar --version v1 --salida <csv recuperado>
                                               # compara contra la referencia y escribe un reporte JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd

from .comun import CONFIG_DATOS, EVIDENCIAS, PREPARADOS, RAIZ, cargar_yaml, guardar_json, sha256_archivo
from .predecir import cargar_solucion, predecir

DIR = EVIDENCIAS / "recuperacion"
ENTRADA = DIR / "entrada_val.csv"
N_FILAS = 25
TOLERANCIA = 1e-9  # diferencia absoluta máxima aceptada en score_attack


def muestra() -> None:
    val = pd.read_csv(PREPARADOS / "val.csv")
    # Selección determinista: las N filas con menor sha256(row_id), sin depender de generadores aleatorios.
    orden = val["row_id"].map(lambda r: hashlib.sha256(r.encode()).hexdigest())
    entrada = val.loc[orden.sort_values().index[:N_FILAS]].drop(columns="grupo")
    entrada.to_csv(ENTRADA, index=False)
    hashes = {"entrada_val.csv": sha256_archivo(ENTRADA)}
    for version in ("v1", "v2"):
        pipe, config = cargar_solucion(local=str(RAIZ / "hf" / version))
        ruta = DIR / f"referencia_{version}.csv"
        predecir(pipe, config, entrada).to_csv(ruta, index=False, float_format="%.12f")
        hashes[ruta.name] = sha256_archivo(ruta)
        hashes[f"model_sha256_{version}"] = config["model_sha256"]
    guardar_json({"filas": len(entrada), "criterio": f"{N_FILAS} filas de val con menor sha256(row_id)",
                  "clases": {str(k): int(v) for k, v in entrada["label"].value_counts().sort_index().items()},
                  "hashes": hashes}, DIR / "referencia.json")
    print(f"{ENTRADA} ({len(entrada)} filas) y referencias v1/v2 escritas")


def huellas() -> bool:
    esperado = {m["ruta"]: m["sha256_esperado"] for m in cargar_yaml(CONFIG_DATOS)["archivos"].values()}
    auditoria = json.loads((EVIDENCIAS / "auditoria.json").read_text(encoding="utf-8"))
    esperado |= {f"data/prepared/{n}": h for n, h in auditoria["particiones"]["huellas_sha256"].items()}
    ok = True
    for ruta, sha in esperado.items():
        real = sha256_archivo(RAIZ / ruta)
        ok &= real == sha
        print(f"{'OK ' if real == sha else 'MAL'} {ruta} {real}")
    return ok


def comparar(version: str, salida: str) -> dict:
    ref = pd.read_csv(DIR / f"referencia_{version}.csv")
    rec = pd.read_csv(salida)
    if list(ref["row_id"]) != list(rec["row_id"]):
        raise SystemExit("Las filas no coinciden")
    dif = float((ref["score_attack"] - rec["score_attack"]).abs().max())
    reporte = {
        "version": version,
        "filas": len(ref),
        "etiquetas_iguales": bool((ref["y_pred"] == rec["y_pred"]).all()),
        "diferencia_max_score": dif,
        "tolerancia": TOLERANCIA,
        "coincide": bool((ref["y_pred"] == rec["y_pred"]).all() and dif <= TOLERANCIA),
        "sha256_referencia": sha256_archivo(DIR / f"referencia_{version}.csv"),
        "sha256_recuperada": sha256_archivo(salida),
    }
    print(json.dumps(reporte, indent=1))
    return reporte


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("muestra")
    sub.add_parser("huellas")
    c = sub.add_parser("comparar")
    c.add_argument("--version", choices=["v1", "v2"], required=True)
    c.add_argument("--salida", required=True)
    c.add_argument("--reporte")
    args = ap.parse_args()
    if args.cmd == "muestra":
        muestra()
    elif args.cmd == "huellas":
        if not huellas():
            raise SystemExit(1)
    else:
        rep = comparar(args.version, args.salida)
        if args.reporte:
            guardar_json(rep, RAIZ / args.reporte)
        if not rep["coincide"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
