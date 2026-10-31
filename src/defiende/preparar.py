"""Etapa DVC `preparar`: verifica los CSV crudos, audita, limpia y crea particiones fijas.

Salidas:
  data/prepared/train.csv, val.csv, test.csv   (con row_id y grupo; test sin filas eliminadas)
  data/prepared/asignacion.csv                 (row_id -> split, para el archivo de entrenamiento)
  evidencias/auditoria.json
"""

from __future__ import annotations

import hashlib

import pandas as pd

from .comun import (
    CONFIG_DATOS,
    EVIDENCIAS,
    PREPARADOS,
    RAIZ,
    cargar_yaml,
    columnas_predictoras,
    guardar_json,
    huella_grupo,
    leer_csv,
    sha256_archivo,
)


def perfil_archivo(nombre: str, meta: dict, df: pd.DataFrame, sha: str, objetivo: str) -> dict:
    return {
        "archivo": meta["ruta"],
        "url_descarga": meta["url"],
        "sha256": sha,
        "sha256_esperado": meta["sha256_esperado"],
        "sha256_coincide": sha == meta["sha256_esperado"],
        "filas": int(len(df)),
        "filas_esperadas": meta["filas_esperadas"],
        "columnas": int(df.shape[1]),
        "nombres_columnas": list(df.columns),
        "clases": {str(k): int(v) for k, v in df[objetivo].value_counts().sort_index().items()},
        "faltantes_total": int(df.isna().sum().sum()),
        "faltantes_por_columna": {c: int(v) for c, v in df.isna().sum().items() if v},
        "valor_guion_en_service": int((df["service"] == "-").sum()),
        "minimo_variables_numericas": float(df.select_dtypes("number").drop(columns=["id", objetivo]).min().min()),
        "tipos": {c: str(t) for c, t in df.dtypes.items()},
        "attack_cat": {str(k): int(v) for k, v in df["attack_cat"].value_counts().items()},
    }


def duplicados(df: pd.DataFrame, objetivo: str) -> dict:
    g = df.groupby("grupo")[objetivo].agg(["size", "nunique"])
    contra = g[g["nunique"] > 1]
    return {
        "grupos_entradas_unicas": int(len(g)),
        "grupos_con_duplicados": int((g["size"] > 1).sum()),
        "filas_duplicadas_extra": int((g["size"] - 1).sum()),
        "filas_en_grupos_duplicados": int(g.loc[g["size"] > 1, "size"].sum()),
        "grupos_contradictorios": int(len(contra)),
        "filas_en_grupos_contradictorios": int(contra["size"].sum()),
    }


def asignar_split(grupos: pd.Series, semilla: int, fraccion: float) -> pd.Series:
    corte = int(round(fraccion * 10_000))

    def destino(g: str) -> str:
        n = int(hashlib.sha256(f"{semilla}|{g}".encode()).hexdigest(), 16) % 10_000
        return "val" if n < corte else "train"

    return grupos.map(destino)


def main() -> None:
    cfg = cargar_yaml(CONFIG_DATOS)
    objetivo = cfg["objetivo"]
    excluidas = cfg["excluidas"]
    semilla = cfg["particion"]["semilla"]
    fraccion = cfg["particion"]["fraccion_validacion"]

    dfs, perfiles = {}, {}
    for nombre, meta in cfg["archivos"].items():
        ruta = RAIZ / meta["ruta"]
        sha = sha256_archivo(ruta)
        df = leer_csv(ruta)
        perfiles[nombre] = perfil_archivo(nombre, meta, df, sha, objetivo)
        if sha != meta["sha256_esperado"] or len(df) != meta["filas_esperadas"]:
            raise SystemExit(f"{ruta}: huella o filas no coinciden; no es la partición esperada")
        prefijo = "training-set" if nombre == "train" else "testing-set"
        df.insert(0, "row_id", [f"{prefijo}:{i}" for i in range(1, len(df) + 1)])
        dfs[nombre] = df

    tr, te = dfs["train"], dfs["test"]
    if list(tr.columns) != list(te.columns):
        raise SystemExit("Los esquemas de train y test no coinciden")
    features = columnas_predictoras(tr, excluidas)
    tr["grupo"] = huella_grupo(tr, features)
    te["grupo"] = huella_grupo(te, features)

    dup_train = duplicados(tr, objetivo)
    dup_test = duplicados(te, objetivo)

    # Política: los grupos contradictorios (mismas entradas, etiquetas distintas) son ambiguos;
    # se eliminan SOLO de entrenamiento/validación. Los duplicados no contradictorios se conservan
    # (la frecuencia es información) y cada grupo va completo a un único split.
    contradictorios = tr.groupby("grupo")[objetivo].transform("nunique") > 1
    limpio = tr[~contradictorios].copy()
    limpio["split"] = asignar_split(limpio["grupo"], semilla, fraccion)

    PREPARADOS.mkdir(parents=True, exist_ok=True)
    train = limpio[limpio["split"] == "train"].drop(columns="split")
    val = limpio[limpio["split"] == "val"].drop(columns="split")
    train.to_csv(PREPARADOS / "train.csv", index=False)
    val.to_csv(PREPARADOS / "val.csv", index=False)
    te.to_csv(PREPARADOS / "test.csv", index=False)
    asignacion = tr[["row_id", "grupo"]].copy()
    asignacion["split"] = "excluida_contradictoria"
    asignacion.loc[limpio.index, "split"] = limpio["split"]
    asignacion.to_csv(PREPARADOS / "asignacion.csv", index=False)

    g_train, g_val, g_test = set(train["grupo"]), set(val["grupo"]), set(te["grupo"])
    test_en_train = te["grupo"].isin(g_train)
    test_en_val = te["grupo"].isin(g_val)
    etiqueta_train = tr.groupby("grupo")[objetivo].agg(lambda s: s.mode().iloc[0])
    cruzados = te[te["grupo"].isin(etiqueta_train.index)]
    discrepan = int((cruzados[objetivo].values != etiqueta_train.loc[cruzados["grupo"]].values).sum())

    def resumen(df):
        return {"filas": int(len(df)), "grupos": int(df["grupo"].nunique()),
                "clases": {str(k): int(v) for k, v in df[objetivo].value_counts().sort_index().items()}}

    auditoria = {
        "fuente": {k: cfg[k] for k in ("fuente_oficial", "cita", "mirror", "fecha_descarga")},
        "archivos": perfiles,
        "objetivo": {"columna": objetivo, "clase_positiva": "1 = ataque", "clase_negativa": "0 = normal"},
        "variables_excluidas": {
            "id": "identificador de fila; se conserva solo como auditoría (row_id)",
            "attack_cat": "categoría del ataque: revela la respuesta (fuga)",
            "label": "variable objetivo",
        },
        "variables_usadas": features,
        "variables_categoricas": cfg["categoricas"],
        "duplicados_por_entradas": {
            "definicion": "filas con valores idénticos en todas las variables predictoras (sin id, attack_cat ni label)",
            "train_file": dup_train,
            "test_file": dup_test,
        },
        "politica_limpieza": {
            "contradictorios": "se eliminan de train/val todas las filas de grupos con etiquetas distintas",
            "filas_eliminadas_train_file": int(contradictorios.sum()),
            "duplicados": "se conservan; los grupos no se reparten entre train y val",
            "test": "no se elimina ninguna fila; se usa completo y solo para la evaluación final",
            "faltantes": "no hay NaN; '-' en service es una categoría ('sin servicio identificado')",
        },
        "particiones": {
            "metodo": "asignación por grupo: sha256(semilla|grupo) mod 10000 < fraccion*10000 -> val",
            "semilla": semilla,
            "fraccion_validacion": fraccion,
            "train": resumen(train),
            "val": resumen(val),
            "test": resumen(te),
            "huellas_sha256": {f.name: sha256_archivo(f) for f in sorted(PREPARADOS.glob("*.csv"))},
        },
        "cruces_entre_particiones": {
            "grupos_train_y_val": len(g_train & g_val),
            "filas_test_con_entradas_en_train": int(test_en_train.sum()),
            "filas_test_con_entradas_en_val": int(test_en_val.sum()),
            "filas_test_con_entradas_en_train_o_val": int((test_en_train | test_en_val).sum()),
            "proporcion_test_solapada": round(float((test_en_train | test_en_val).mean()), 4),
            "grupos_test_vistos_en_train_file": len(g_test & set(tr["grupo"])),
            "filas_test_solapadas_con_etiqueta_distinta": discrepan,
            "nota": "El solapamiento es una propiedad del dataset publicado; las métricas de test sobre "
                    "esas filas miden en parte memoria del patrón, no generalización.",
        },
    }
    guardar_json(auditoria, EVIDENCIAS / "auditoria.json")
    print(f"train={len(train)} val={len(val)} test={len(te)} eliminadas={int(contradictorios.sum())}")


if __name__ == "__main__":
    main()
