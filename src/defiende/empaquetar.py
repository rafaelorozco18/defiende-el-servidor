"""Arma la carpeta hf/<version>/ que se publica como una revisión del repo de modelos.

Contenido: model.joblib (pipeline completo), config.json (umbral, variables, referencias),
requirements.txt (versiones fijadas), predecir.py y README.md (model card).

Uso: python -m defiende.empaquetar --version v1 --repo USUARIO/REPO [--github URL]
"""

from __future__ import annotations

import argparse
import json
import shutil
from importlib.metadata import version as version_paquete

from .comun import CONFIG_DATOS, EVIDENCIAS, MODELOS, RAIZ, cargar_yaml, guardar_json, sha256_archivo

DEPENDENCIAS = ("scikit-learn", "numpy", "scipy", "pandas", "joblib", "huggingface_hub")


def tabla_metricas(m: dict) -> str:
    mc = m["matriz_confusion"]
    return (
        "| Métrica (ataque = positivo) | Valor |\n|---|---|\n"
        f"| Recall de ataque | {m['recall_ataque']:.4f} |\n"
        f"| Precision de ataque | {m['precision_ataque']:.4f} |\n"
        f"| Tasa de falsos positivos (FPR) | {m['fpr']:.4f} |\n"
        f"| Macro-F1 | {m['macro_f1']:.4f} |\n"
        f"| Accuracy (complementaria) | {m['accuracy']:.4f} |\n"
        f"| Filas | {m['n']} |\n\n"
        "Matriz de confusión:\n\n"
        "| | pred. normal | pred. ataque |\n|---|---|---|\n"
        f"| real normal | TN = {mc['tn']} | FP = {mc['fp']} |\n"
        f"| real ataque | FN = {mc['fn']} | TP = {mc['tp']} |\n"
    )


def model_card(version: str, v: dict, sel: dict, repo: str, github: str, auditoria: dict) -> str:
    p = auditoria["particiones"]
    otra = "v2" if version == "v1" else "v1"
    cambio = ""
    if version == "v2":
        c = sel["cambio_v1_a_v2"]
        d = c["delta_val"]
        cambio = (
            "\n## Cambio respecto a v1\n\n"
            f"- Modelo: {c['modelo']}\n- Preprocesamiento: {c['preprocesamiento']}\n- Umbral: {c['umbral']}\n\n"
            f"En validación: FPR {d['fpr']:+.4f}, precision {d['precision_ataque']:+.4f}, macro-F1 {d['macro_f1']:+.4f}, "
            f"recall de ataque {d['recall_ataque']:+.4f}.\n\n"
            "**Qué mejora:** muchas menos falsas alarmas sobre tráfico normal y mayor precisión. "
            "**Qué empeora:** se escapan más ataques (más falsos negativos). **Error aceptado:** con la regla "
            "declarada (FPR ≤ 5 %) se prioriza que el equipo de seguridad no se sature de alertas, a costa de "
            f"dejar pasar {-d['recall_ataque'] * 100:.1f} de cada 100 ataques más que v1 en validación.\n\n"
            "La regla de selección (mayor macro-F1 a umbral 0.5 entre las cuatro configuraciones y, después, "
            "mayor recall con FPR ≤ 0.05) se declaró en `configs/experimentos.yaml` antes de ver resultados y "
            "solo usa validación. Las métricas de validación de v2 son algo optimistas porque el umbral se "
            "eligió sobre esa misma partición.\n"
        )
    ruta_test = EVIDENCIAS / "metricas-test.json"
    if version == "v1":
        seccion_test = "v1 no se evalúa en test: la comparación v1/v2 se hace solo con validación."
    elif ruta_test.exists():
        mt = json.loads(ruta_test.read_text(encoding="utf-8"))
        seccion_test = (
            f"Evaluación única sobre las {mt['dataset']['filas_evaluadas']} filas de UNSW_NB15_testing-set.csv "
            f"(sin eliminar filas), con la solución congelada en el commit `{mt['seleccion_congelada']['commit']}`.\n\n"
            f"- Revisión evaluada: `{mt['huggingface_revision_evaluada']}`\n"
            f"- sha256 de `model.joblib` evaluado: `{mt['model_sha256']}` (esta revisión de documentación no lo cambia)\n"
            f"- sha256 del archivo de test: `{mt['dataset']['sha256_original']}`\n\n"
            f"{tabla_metricas(mt['metricas'])}\n"
            "**El resultado empeora frente a validación** (FPR de 0.049 a 0.203): el tráfico normal del archivo de "
            "test recibe puntuaciones más altas que el de validación (percentil 90 del score de normales: 0.39 en "
            "validación, 0.85 en test), sobre todo en conexiones con estado FIN e INT. Las filas de test sin "
            "solapamiento con train/val tienen la misma FPR, así que no se explica por fuga. Es un cambio de "
            "distribución entre las particiones publicadas; se conserva el resultado sin reajustar el umbral. "
            "Detalle en `evidencias/diagnostico-test.json` del repo de código."
        )
    else:
        seccion_test = ("Pendiente. La selección se congela antes de evaluar; el resultado se agregará aquí en una "
                        "revisión de solo documentación, sin cambiar `model.joblib`.")
    return f"""---
library_name: sklearn
tags:
- tabular-classification
- intrusion-detection
- unsw-nb15
- mlops-curso
pipeline_tag: tabular-classification
---

# Defiende el servidor — {version} ({v['exp_id']})

Clasificador binario de conexiones de red: **0 = normal, 1 = ataque** (clase positiva), entrenado con las
particiones publicadas de UNSW-NB15. Taller académico de MLOps (Universidad Externado de Colombia, 2026-II):
el objetivo es demostrar versionado y recuperación, no desplegar un detector.

## Propósito

Separar tráfico normal de posibles ataques en registros de flujo con las 42 variables de UNSW-NB15.
**No** está pensado para producción ni demuestra eficacia sobre una red actual.

## Solución

- Pipeline de scikit-learn completo (`model.joblib`): preprocesamiento (`{v['exp_id']}`) + estimador.
- Familia: {v['familia']}. Umbral: **{v['umbral']}**. Regla: `ataque si score_attack >= {v['umbral']}`,
  donde `score_attack = predict_proba(X)[:, 1]`.
- sha256 de `model.joblib`: `{v['model_sha256']}`
- Configuración completa en `config.json`; dependencias fijadas en `requirements.txt`.

## Trazabilidad

| Pieza | Referencia |
|---|---|
| Código (commit Git que entrenó el modelo) | [`{v['git_commit']}`]({github}/tree/{v['git_commit']}) |
| Datos (commit con `data/raw.dvc` y `dvc.lock`) | `{v['dvc_commit']}` |
| DVC md5 datos crudos | `{v['dvc_raw_md5']}` |
| DVC md5 datos preparados | `{v['dvc_prepared_md5']}` |
| Run de MLflow | `{v['run_id']}` (exportado en `evidencias/mlflow/` del repo de código) |
| Otra versión | `{otra}` en este mismo repo (ver tags) |

## Datos

- UNSW_NB15_training-set.csv (175 341 filas) para entrenamiento/validación; UNSW_NB15_testing-set.csv
  (82 332 filas) reservado para la evaluación final.
- Variables excluidas: `id` (identificador), `attack_cat` (revela la respuesta), `label` (objetivo).
- Se eliminaron de train/val {auditoria['politica_limpieza']['filas_eliminadas_train_file']} filas de grupos con
  entradas idénticas y etiquetas contradictorias. Los duplicados se conservan, pero cada grupo de entradas
  idénticas cae completo en train o en val (semilla {p['semilla']}).
- Train: {p['train']['filas']} filas · Validación: {p['val']['filas']} filas.

## Métricas de validación (umbral {v['umbral']})

{tabla_metricas(v['metricas_val'])}
Denominadores cero: si una métrica no tiene denominador se reporta `null`; en macro-F1 aporta 0.
{cambio}
## Evaluación final en test

{seccion_test}

## Limitaciones

- Dataset histórico (2015) generado en laboratorio: no representa el tráfico de una red actual.
- El {auditoria['cruces_entre_particiones']['proporcion_test_solapada'] * 100:.1f} % de las filas de test tiene entradas
  idénticas a filas del archivo de entrenamiento; parte del desempeño en test es memoria de patrones.
- Los duplicados pesan en las métricas (se conservan como frecuencia observada).
- Pickle de scikit-learn: requiere las versiones de `requirements.txt` y solo debe cargarse desde fuentes confiables.

## Uso

```bash
pip install -r requirements.txt
```

```python
from huggingface_hub import hf_hub_download
import joblib, json, pandas as pd

REPO = "{repo}"
REV = "<SHA completo de la revisión {version}>"
pipe = joblib.load(hf_hub_download(REPO, "model.joblib", revision=REV))
cfg = json.load(open(hf_hub_download(REPO, "config.json", revision=REV)))
df = pd.read_csv("conexiones.csv")          # mismas columnas que UNSW-NB15
score = pipe.predict_proba(df[cfg["variables_entrada"]])[:, 1]
y_pred = (score >= cfg["umbral"]).astype(int)
```

O con el script incluido: `python predecir.py --repo {repo} --revision <SHA> --entrada conexiones.csv --salida pred.csv`.

## Referencias

Moustafa, N. y Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems.
MilCIS 2015. Fuente: https://research.unsw.edu.au/projects/unsw-nb15-dataset
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], required=True)
    ap.add_argument("--repo", required=True, help="repo de modelos en Hugging Face (usuario/nombre)")
    ap.add_argument("--github", required=True, help="URL del repo de código")
    args = ap.parse_args()

    v = json.loads((EVIDENCIAS / "versiones.json").read_text(encoding="utf-8"))[args.version]
    ruta_sel = EVIDENCIAS / "seleccion.json"
    sel = json.loads(ruta_sel.read_text(encoding="utf-8")) if args.version == "v2" else {}
    auditoria = json.loads((EVIDENCIAS / "auditoria.json").read_text(encoding="utf-8"))
    origen = MODELOS / v["exp_id"] / "model.joblib"
    if sha256_archivo(origen) != v["model_sha256"]:
        raise SystemExit(f"{origen} no coincide con el sha256 registrado en MLflow")

    destino = RAIZ / "hf" / args.version
    shutil.rmtree(destino, ignore_errors=True)
    destino.mkdir(parents=True)
    shutil.copy(origen, destino / "model.joblib")
    shutil.copy(RAIZ / "src" / "defiende" / "predecir.py", destino / "predecir.py")
    cdatos = cargar_yaml(CONFIG_DATOS)
    guardar_json({
        "version": args.version,
        "exp_id": v["exp_id"],
        "familia": v["familia"],
        "umbral": v["umbral"],
        "regla_decision": v["regla_decision"],
        "clase_positiva": "1 = ataque",
        "variables_entrada": auditoria["variables_usadas"],
        "variables_categoricas": cdatos["categoricas"],
        "variables_excluidas": cdatos["excluidas"],
        "model_sha256": v["model_sha256"],
        "mlflow_run_id": v["run_id"],
        "git_repo": args.github,
        "git_commit": v["git_commit"],
        "dvc_commit": v["dvc_commit"],
        "dvc_raw_md5": v["dvc_raw_md5"],
        "dvc_prepared_md5": v["dvc_prepared_md5"],
        "metricas_val": v["metricas_val"],
    }, destino / "config.json")
    (destino / "requirements.txt").write_text(
        "".join(f"{d}=={version_paquete(d)}\n" for d in DEPENDENCIAS), encoding="utf-8")
    (destino / "README.md").write_text(model_card(args.version, v, sel, args.repo, args.github, auditoria), encoding="utf-8")
    print(f"{destino}: {sorted(p.name for p in destino.iterdir())}")


if __name__ == "__main__":
    main()
