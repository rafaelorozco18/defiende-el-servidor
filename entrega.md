# Entrega — MLOps · Defiende el servidor

## Integrantes y repositorio

- **Integrante:** Rafael Orozco, Grupo B (trabajo individual).
- **Repositorio de código:** https://github.com/rafaelorozco18/defiende-el-servidor
- **Commit final de la entrega:** el que apunta el tag `entrega` (SHA completo en el mensaje de entrega;
  `git rev-parse entrega^{commit}`).

## Versiones de la solución

| Campo | v1 | v2 |
|---|---|---|
| Configuración | E1-logreg-base (regresión logística, one-hot + escalado) | E4-hgb (HistGradientBoosting, codificación ordinal) |
| Umbral | 0.5 | 0.64 |
| Tag Git | `v1` → `485eddf6367a37ed32e976f6f6e783190abba490` | `v2` → `2f82c1bbf6f9afac4037a0175b12be102b8241c0` (congelación) |
| Commit que entrenó el modelo | `d7011d2272daf4fab8555744bbcaeebe49f5cf08` | `cd74ed860fc4c812e886272a6de16a82809ecfc9` |
| Datos DVC (commit con `data/raw.dvc` + `dvc.lock`) | `d7011d2272daf4fab8555744bbcaeebe49f5cf08` | igual |
| md5 DVC crudos / preparados | `fe93051ae209ffb0128a31e7a478c009.dir` / `731ffba210f38c4b5fed86beb0d6eff7.dir` | igual |
| Run ID MLflow | `8a9c17ff6fb94677aee381b05b8e4827` | `6bb2e44a2de843f8ac5dfff5c77cc865` |
| Repo Hugging Face | https://huggingface.co/Rafoniloni/defiende-el-servidor-unsw | igual |
| SHA Hugging Face (modelo) | `264e987803bcae3aaa09c719df77fe8d726eb96d` (tag `v1`) | `64fc9774c0779005d65f79c769a4def98258065a` (tag `v2`, evaluada en test) |
| SHA HF solo documentación | — | `abdccefcf0b2cc6bf5950ba9b8accd8c6ea23090` (model card con test; mismo `model.joblib`) |
| sha256 `model.joblib` | `68b0defa2b8ad78e682489b3cadb03fd04368c06accacaf3fcd544d864a5cdfa` | `b32254d6bf876d341287690214a015e463a9186e3186b8e28d4c5df12310ce1b` |
| Validación: recall / precision / FPR / macro-F1 | 0.9876 / 0.9273 / 0.1639 / 0.9272 | 0.9597 / 0.9766 / 0.0486 / 0.9512 |

## Evidencias

| Requisito | Archivo |
|---|---|
| auditoria.json | `evidencias/auditoria.json` (generado por la etapa DVC `preparar`) |
| experimentos.csv | `evidencias/experimentos.csv` |
| seleccion.json | `evidencias/seleccion.json` (+ `evidencias/versiones.json` con la ficha de v1 y v2, `evidencias/umbral_v2.csv`) |
| Congelación anterior al test | commit `2f82c1b…` (31-10-2026 08:50 -05:00); test evaluado a las 08:51 -05:00 (`metricas-test.json` → `seleccion_congelada`, `evaluado_utc`) |
| metricas-test.json | `evidencias/metricas-test.json` |
| predicciones-test.csv | `evidencias/predicciones-test.csv` (82 332 filas; `row_id` = `testing-set:<n.º de fila de datos>`, regla `y_pred = score_attack >= 0.64`) |
| mlflow/ | `evidencias/mlflow/<exp_id>__<run_id>/run.json` (params, métricas, tags, artefactos) + `metricas_val.json`, `val_scores.csv` |
| recuperacion/ | `evidencias/recuperacion/`: `entrada_val.csv` (25 filas), `referencia_v1/v2.csv` (antes), `referencia.json` (hashes), `prueba/` (log, salidas recuperadas, comparaciones, hashes) |
| Diagnóstico posterior al test | `evidencias/diagnostico-test.json` (no se usó para decidir nada) |
| Informe | `informe.pdf` (adjunto aparte) |

## Almacenamiento DVC y acceso

- Remote `hfstore` (por defecto en `.dvc/config`): https://huggingface.co/datasets/Rafoniloni/defiende-el-servidor-dvc ,
  leído por HTTP en `…/resolve/main`. Es **público**: `uv run dvc pull` funciona sin credenciales.
- Como Hugging Face no acepta el `PUT` de `dvc push`, la subida se hace con `python -m defiende.dvc_remote_hf`,
  que copia `.dvc/cache/files/md5` con la misma estructura que espera DVC.
- GitHub, el repo de modelos y el remote DVC son públicos; no hace falta invitación ni token.

## Comandos

```bash
git clone https://github.com/rafaelorozco18/defiende-el-servidor.git && cd defiende-el-servidor
uv sync --frozen                                       # instalación
uv run dvc pull && uv run python -m defiende.recuperacion huellas      # datos
uv run python -m defiende.entrenar --exp E4-hgb        # entrenamiento (árbol Git limpio)
uv run python -m defiende.predecir --repo Rafoniloni/defiende-el-servidor-unsw \
  --revision 64fc9774c0779005d65f79c769a4def98258065a \
  --entrada evidencias/recuperacion/entrada_val.csv --salida salidas/v2.csv   # predicción
bash scripts/prueba_recuperacion.sh 2f82c1bbf6f9afac4037a0175b12be102b8241c0  # prueba completa
```

**Tiempo y recursos medidos** (Intel Core i5-1235U, 12 hilos, 5 GB de RAM en WSL2, sin GPU):
etapa `preparar` 68 s (1.1 GB de RAM pico; regenera bytes idénticos); entrenamiento E1 17 s, E2 19 s, E3 11 s, E4 10 s; prueba de recuperación completa
≈ 51 s (`uv sync` 10 s, `dvc pull` 23 s, descarga y predicción de v1/v2 ≈ 12 s).

## Limitaciones conocidas

- La FPR de v2 sube de 0.049 en validación a 0.203 en test: el tráfico normal del archivo de test tiene otra
  distribución (ver informe y `diagnostico-test.json`). El umbral no se reajustó.
- El umbral de v2 se eligió sobre la misma validación con la que se reporta, así que esas cifras son algo optimistas.
- El 10.1 % de las filas de test repite entradas de train/val. Los duplicados se conservan con su frecuencia.
- Dataset de laboratorio de 2015: no demuestra eficacia sobre una red actual.
- `model.joblib` es un pickle de scikit-learn 1.9.1: requiere las versiones fijadas y solo debe cargarse desde fuentes confiables.
- MLflow se usó en local (SQLite). `mlflow.db` no se entrega; la exportación en `evidencias/mlflow/` lo sustituye.

## Uso de IA, código externo y contribución

- **Contribución:** trabajo individual de Rafael Orozco.
- **Asistente de IA:** Claude Code (modelo Claude Opus 5.5, Anthropic). Se usó para diseñar y escribir el
  código de `src/defiende/`, los scripts, la configuración de DVC/MLflow/Hugging Face, el README, este
  índice y el borrador del informe, y para ejecutar el flujo. Los commits generados con el asistente
  incluyen la línea `Co-Authored-By`.
- **Qué se verificó:**
  - Los SHA-256 de los CSV coinciden con el anexo de la guía.
  - La auditoría muestra 0 grupos compartidos entre train y val.
  - El entrenamiento es determinista: dos ejecuciones dieron las mismas métricas.
  - La prueba de recuperación en entorno nuevo reproduce byte a byte las salidas de v1 y v2.
  - Las métricas de test se recalculan desde `predicciones-test.csv`.
  - La revisión de documentación en HF conserva el mismo sha256 del modelo.
- **Historial:** antes del primer `push` se reescribió una vez la historia local para separar el commit de v1
  (solo E1) del de E2–E4 y la regla de selección. Los runs se reentrenaron sobre los commits definitivos
  con métricas idénticas. La regla de selección de v2 ya estaba escrita antes del primer entrenamiento de
  E2–E4. Nada publicado se modificó después.
- **Código externo:** no se copió código de terceros; solo se usan las bibliotecas de `uv.lock`
  (scikit-learn BSD-3, pandas BSD-3, MLflow Apache-2.0, DVC Apache-2.0, huggingface_hub Apache-2.0).
