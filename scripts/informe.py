"""Genera informe.pdf leyendo TODAS las cifras de evidencias/ (coherencia con los archivos entregados).

uv run --with reportlab --with matplotlib python scripts/informe.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RAIZ = Path(__file__).resolve().parents[1]
EV = RAIZ / "evidencias"
FUENTES = Path("/usr/share/fonts/truetype/dejavu")


def cargar(n):
    return json.loads((EV / n).read_text(encoding="utf-8"))


aud, sel, ver = cargar("auditoria.json"), cargar("seleccion.json"), cargar("versiones.json")
mt, diag = cargar("metricas-test.json"), cargar("diagnostico-test.json")
rec1 = cargar("recuperacion/prueba/comparacion_v1.json")
rec2 = cargar("recuperacion/prueba/comparacion_v2.json")
exp = pd.read_csv(EV / "experimentos.csv")
curva = pd.read_csv(EV / "umbral_v2.csv")
v1, v2 = ver["v1"], ver["v2"]

pdfmetrics.registerFont(TTFont("DV", str(FUENTES / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(FUENTES / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(FUENTES / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DV", boldItalic="DVB")
base = getSampleStyleSheet()
P = ParagraphStyle("p", parent=base["Normal"], fontName="DV", fontSize=8.6, leading=11.4, alignment=TA_JUSTIFY,
                   spaceAfter=4)
H1 = ParagraphStyle("h1", parent=P, fontName="DVB", fontSize=15, leading=18, spaceAfter=2, alignment=0)
H2 = ParagraphStyle("h2", parent=P, fontName="DVB", fontSize=10.5, leading=13, spaceBefore=7, spaceAfter=3,
                    textColor=colors.HexColor("#1b4d3e"), alignment=0)
SUB = ParagraphStyle("sub", parent=P, fontSize=8, textColor=colors.HexColor("#555555"), alignment=0)
CELDA = ParagraphStyle("c", parent=P, fontSize=7.2, leading=8.8, alignment=0, spaceAfter=0)
CAP = ParagraphStyle("cap", parent=P, fontSize=7.4, leading=9, textColor=colors.HexColor("#444444"), alignment=0)


def f(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def mil(n):
    return f"{int(n):,}".replace(",", ".")


def tabla(filas, anchos, cabecera=True):
    datos = [[c if isinstance(c, Paragraph) else Paragraph(str(c), CELDA) for c in fila] for fila in filas]
    t = Table(datos, colWidths=[a * cm for a in anchos], repeatRows=1 if cabecera else 0)
    estilo = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b5c4bd")),
              ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
    if cabecera:
        estilo.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3ece8")))
    t.setStyle(TableStyle(estilo))
    return t


def figura(fig, ancho_cm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    img.drawWidth = ancho_cm * cm
    img.drawHeight = img.imageHeight * ancho_cm * cm / img.imageWidth
    return img


def fig_umbral():
    fig, ax = plt.subplots(figsize=(5.2, 2.3))
    ax.plot(curva["umbral"], curva["recall_ataque"], label="recall de ataque", color="#1b6e53")
    ax.plot(curva["umbral"], curva["fpr"], label="FPR", color="#c0504d")
    ax.plot(curva["umbral"], curva["macro_f1"], label="macro-F1", color="#555555", ls="--", lw=1)
    ax.axhline(sel["regla"]["umbral"]["fpr_max"], color="#c0504d", lw=0.6, ls=":")
    ax.axvline(v2["umbral"], color="black", lw=0.8)
    ax.text(v2["umbral"] + 0.01, 0.55, f"umbral {f(v2['umbral'], 2)}", fontsize=7)
    ax.set_xlabel("umbral sobre score_attack (E4-hgb, validación)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="center left")
    ax.grid(alpha=0.25)
    return fig


def fig_scores():
    q = diag["score_normales_cuantiles"]
    ks = list(q["val"].keys())
    fig, ax = plt.subplots(figsize=(3.2, 2.1))
    x = range(len(ks))
    ax.bar([i - 0.2 for i in x], [q["val"][k] for k in ks], 0.4, label="validación", color="#1b6e53")
    ax.bar([i + 0.2 for i in x], [q["test"][k] for k in ks], 0.4, label="test", color="#c0504d")
    ax.axhline(v2["umbral"], color="black", lw=0.7, ls=":")
    ax.set_xticks(list(x), [f"p{int(float(k) * 100)}" for k in ks], fontsize=7)
    ax.set_title("score_attack de conexiones NORMALES", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5)
    return fig


def main():
    hf = v1["huggingface"]["repo"]
    tr, va, te = (aud["particiones"][k] for k in ("train", "val", "test"))
    dup, dupt = aud["duplicados_por_entradas"]["train_file"], aud["duplicados_por_entradas"]["test_file"]
    cruce = aud["cruces_entre_particiones"]
    m1, m2, m2t = v1["metricas_val"], v2["metricas_val"], mt["metricas"]
    mc1, mc2, mct = m1["matriz_confusion"], m2["matriz_confusion"], m2t["matriz_confusion"]
    m05 = sel["v2_a_umbral_0_5"]
    s = []

    s += [Paragraph("Defiende el servidor: versionado y recuperación de un detector de ataques", H1),
          Paragraph("MLOps · Visualización 2 · 2026-II · Grupo B · Rafael Orozco (individual) · "
                    "Universidad Externado de Colombia · 31 de octubre de 2026", SUB), Spacer(1, 5)]

    s += [Paragraph("1. Objetivo y trazabilidad", H2),
          Paragraph("Se construyó un clasificador binario normal (0) / ataque (1, clase positiva) con las particiones "
                    "publicadas de UNSW-NB15. El foco es que cada versión pueda identificarse y recuperarse: el código "
                    "vive en Git/GitHub, los datos en DVC (remote HTTP en un repo dataset público de Hugging Face), "
                    "los experimentos en MLflow (exportados a <i>evidencias/mlflow</i>) y los modelos en Hugging Face, "
                    "referenciados por SHA completo. Cada pieza guarda el identificador de la anterior.", P),
          tabla([["", "v1", "v2"],
                 ["Configuración / umbral", f"{v1['exp_id']} / {v1['umbral']}", f"{v2['exp_id']} / {v2['umbral']}"],
                 ["Commit Git que entrenó", v1["git_commit"], v2["git_commit"]],
                 ["Datos DVC (crudos · preparados)", f"{v1['dvc_raw_md5']} · {v1['dvc_prepared_md5']}", "iguales a v1"],
                 ["Run ID MLflow", v1["run_id"], v2["run_id"]],
                 [f"Revisión HF ({hf})", v1["huggingface"]["revision_sha"], v2["huggingface"]["revision_sha"]],
                 ["sha256 model.joblib", v1["model_sha256"], v2["model_sha256"]]],
                [3.6, 6.9, 6.9])]

    s += [Paragraph("2. Datos, auditoría y limpieza", H2),
          Paragraph(f"Se usaron <i>UNSW_NB15_training-set.csv</i> ({mil(aud['archivos']['train']['filas'])} filas) y "
                    f"<i>UNSW_NB15_testing-set.csv</i> ({mil(aud['archivos']['test']['filas'])} filas), descargados el "
                    f"{aud['fuente']['fecha_descarga']} de un mirror fijado por commit. Sus SHA-256 coinciden con el anexo "
                    "de la guía, y también las filas y el esquema (45 columnas). No hay valores faltantes; en <i>service</i>, "
                    "'-' es una categoría. Las variables numéricas son ≥ 0. Se excluyen <i>id</i> (identificador; se conserva "
                    "como <i>row_id</i> solo para auditoría), <i>attack_cat</i> (revela la respuesta) y <i>label</i>, "
                    "de modo que quedan 42 predictoras (3 categóricas).", P),
          Paragraph(f"<b>Duplicados por entradas.</b> En el archivo de entrenamiento hay {mil(dup['grupos_entradas_unicas'])} "
                    f"combinaciones únicas de predictoras; {mil(dup['grupos_con_duplicados'])} grupos se repiten y suman "
                    f"{mil(dup['filas_duplicadas_extra'])} filas extra. {dup['grupos_contradictorios']} grupos "
                    f"({dup['filas_en_grupos_contradictorios']} filas) tienen las mismas entradas con etiquetas distintas. "
                    f"En test: {mil(dupt['filas_duplicadas_extra'])} filas extra y {dupt['grupos_contradictorios']} grupos "
                    "contradictorios. <b>Política:</b> los grupos contradictorios se eliminan solo de train/val, porque no "
                    "son aprendibles. Los demás duplicados se conservan como frecuencia observada, y cada grupo va "
                    "completo a un solo split. El test no se toca.", P),
          Paragraph(f"<b>Particiones.</b> Validación = grupos con sha256(semilla|grupo) mod 10 000 &lt; 2 000 (semilla "
                    f"{aud['particiones']['semilla']}). La asignación es determinista y no depende de la versión de la "
                    f"librería; queda guardada fila a fila en <i>asignacion.csv</i>. Train: {mil(tr['filas'])} filas "
                    f"({mil(tr['clases']['1'])} ataques); validación: {mil(va['filas'])} ({mil(va['clases']['1'])} ataques); "
                    f"test: {mil(te['filas'])} ({mil(te['clases']['1'])} ataques). Grupos compartidos entre train y val: "
                    f"<b>{cruce['grupos_train_y_val']}</b>. El {f(cruce['proporcion_test_solapada'] * 100, 1)} % de las filas "
                    f"de test ({mil(cruce['filas_test_con_entradas_en_train_o_val'])}) repite entradas de train/val, y "
                    f"{cruce['filas_test_solapadas_con_etiqueta_distinta']} de ellas con otra etiqueta. Codificadores, "
                    "escalado y modelo se ajustan solo con train dentro de un único Pipeline de scikit-learn. Los CSV "
                    "recibidos (<i>data/raw.dvc</i>) y los preparados (<i>dvc.lock</i>) son las dos versiones de datos en DVC.", P)]

    filas = [["ID / familia", "Hipótesis y cambio", "Recall", "Prec.", "FPR", "Macro-F1", "FP / FN", "Run ID"]]
    for _, r in exp.iterrows():
        filas.append([f"{r['exp_id']}<br/>{r['familia']}", f"{r['cambio']} {r['hipotesis']}",
                      f(r["recall_ataque"]), f(r["precision_ataque"]), f(r["fpr"]), f(r["macro_f1"]),
                      f"{mil(r['fp'])} / {mil(r['fn'])}", r["run_id"][:10] + "…"])
    s += [Paragraph("3. Cuatro configuraciones en dos familias", H2),
          Paragraph("Todas usan la misma partición, la semilla 42 y umbral 0,5 para compararse. Cada run registra en MLflow "
                    "la hipótesis, el cambio, los parámetros, el commit, los md5 de DVC, las métricas de validación, el "
                    "pipeline completo y su sha256. Las métricas toman ataque como positivo: FPR = FP/(FP+TN). Si un "
                    "denominador es 0, la métrica se reporta como <i>null</i> y aporta 0 al macro-F1; no ocurrió en "
                    f"ningún run. Validación: n = {mil(va['filas'])}.", P),
          tabla(filas, [2.5, 6.4, 1.2, 1.2, 1.2, 1.4, 1.6, 1.9]), Spacer(1, 3),
          Paragraph("<b>Interpretación.</b> Las cuatro hipótesis se confirmaron. En la familia lineal, log1p reduce las falsas "
                    "alarmas un 23 % (1.822 → 1.404), pero pierde recall: la frontera lineal no separa bien el tráfico "
                    "normal voluminoso. Los árboles capturan interacciones (TTL × estado, bytes × servicio) y reducen la "
                    "FPR a la mitad. E4 (boosting) iguala o supera a E3 (bagging) con un artefacto 18 veces menor (1 MB "
                    "frente a 17 MB). En todos los modelos, el error dominante son conexiones normales marcadas como "
                    "ataque, no ataques perdidos: el ataque es la clase mayoritaria (68 %).", P)]

    d = sel["cambio_v1_a_v2"]["delta_val"]
    s += [Paragraph("4. De v1 a v2 y elección del umbral", H2),
          Paragraph(f"<b>v1</b> es la primera solución funcional: E1 con umbral 0,5. <b>v2</b> sale de una regla declarada en "
                    "<i>configs/experimentos.yaml</i> antes de entrenar E2–E4 y aplicada solo sobre validación: (1) elegir la "
                    f"configuración con mayor macro-F1 a 0,5, que es {v2['exp_id']}; (2) elegir el umbral con mayor recall de "
                    f"ataque sujeto a FPR ≤ {f(sel['regla']['umbral']['fpr_max'], 2)}, en una rejilla de 0,01 a 0,99 con paso 0,01. "
                    f"A 0,5, E4 tenía FPR {f(m05['fpr'])}, por encima del límite; el primer umbral que lo cumple es "
                    f"<b>{f(v2['umbral'], 2)}</b>. Para un equipo de seguridad, una FPR de {f(m1['fpr'])} (1 de cada 6 conexiones "
                    "normales alertada) satura la operación; por eso se priorizó acotarla.", P),
          KeepTogether([figura(fig_umbral(), 11.5),
                        Paragraph("Figura 1. Curva de umbral de E4 en validación (evidencias/umbral_v2.csv). Línea punteada: "
                                  "FPR máxima; vertical: umbral elegido.", CAP)]),
          tabla([["Validación", "Recall", "Precision", "FPR", "Macro-F1", "TN", "FP", "FN", "TP"],
                 ["v1 (E1, 0,5)", f(m1["recall_ataque"]), f(m1["precision_ataque"]), f(m1["fpr"]), f(m1["macro_f1"]),
                  mil(mc1["tn"]), mil(mc1["fp"]), mil(mc1["fn"]), mil(mc1["tp"])],
                 [f"v2 (E4, {f(v2['umbral'], 2)})", f(m2["recall_ataque"]), f(m2["precision_ataque"]), f(m2["fpr"]),
                  f(m2["macro_f1"]), mil(mc2["tn"]), mil(mc2["fp"]), mil(mc2["fn"]), mil(mc2["tp"])]],
                [3.0, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6]), Spacer(1, 3),
          Paragraph(f"<b>Qué mejora:</b> la FPR baja {f(-d['fpr'])} (las falsas alarmas pasan de {mil(mc1['fp'])} a "
                    f"{mil(mc2['fp'])}), la precisión sube {f(d['precision_ataque'])} y el macro-F1 {f(d['macro_f1'])}. "
                    f"<b>Qué empeora:</b> el recall baja {f(-d['recall_ataque'])} y los ataques no detectados pasan de "
                    f"{mil(mc1['fn'])} a {mil(mc2['fn'])}. <b>Error aceptado:</b> perder unos 3 de cada 100 ataques a cambio "
                    "de unas 3,4 veces menos alertas falsas. Como el umbral se eligió sobre la misma validación, estas "
                    "cifras de v2 son algo optimistas. La comparación v1/v2 se hace solo en validación.", P)]

    sv, st = diag["fpr_por_state"]["val"], diag["fpr_por_state"]["test"]
    sol = diag["test_por_solapamiento"]
    s += [Paragraph("5. Evaluación final en test", H2),
          Paragraph(f"La selección se congeló en el commit {mt['seleccion_congelada']['commit'][:10]}… "
                    f"({mt['seleccion_congelada']['fecha']}) con la revisión de Hugging Face ya fijada. Después, "
                    f"<i>evaluar_test.py</i> descargó v2 por SHA ({mt['huggingface_revision_evaluada'][:10]}…), verificó "
                    f"el sha256 del modelo y predijo una sola vez las {mil(mt['dataset']['filas_evaluadas'])} filas del test, "
                    "sin eliminar ninguna. El script se niega a re-evaluar. <i>predicciones-test.csv</i> tiene una fila por "
                    "registro (row_id = testing-set:&lt;fila&gt;, y_true, score_attack, y_pred = score ≥ 0,64); las métricas "
                    "se recalcularon desde ese archivo y coinciden.", P),
          tabla([["Test (v2)", "Recall", "Precision", "FPR", "Macro-F1", "Accuracy", "TN", "FP", "FN", "TP"],
                 [f"n = {mil(m2t['n'])}", f(m2t["recall_ataque"]), f(m2t["precision_ataque"]), f(m2t["fpr"]),
                  f(m2t["macro_f1"]), f(m2t["accuracy"]), mil(mct["tn"]), mil(mct["fp"]), mil(mct["fn"]), mil(mct["tp"])]],
                [2.2, 1.5, 1.8, 1.4, 1.7, 1.8, 1.5, 1.5, 1.5, 1.5]), Spacer(1, 3),
          Paragraph(f"<b>El resultado empeora y se conserva.</b> El recall se mantiene ({f(m2t['recall_ataque'])}), pero la "
                    f"FPR sube de {f(m2['fpr'])} a {f(m2t['fpr'])}. Un análisis posterior (<i>diagnostico-test.json</i>, que "
                    "no cambia nada de lo decidido) muestra que las conexiones normales del archivo de test reciben "
                    "puntuaciones mucho más altas (Figura 2). Por estado de conexión, la FPR pasa de "
                    f"{f(sv['FIN']['fpr'])} a {f(st['FIN']['fpr'])} en FIN y de {f(sv['INT']['fpr'])} a "
                    f"{f(st['INT']['fpr'])} en INT. No se explica por el solapamiento: las filas de test no vistas en "
                    f"train/val tienen una FPR de {f(sol['sin_solapamiento']['fpr'])}, y las vistas, de "
                    f"{f(sol['con_entradas_vistas_en_train_val']['fpr'])}. Es un cambio de distribución entre las "
                    "particiones publicadas: la validación, tomada del mismo archivo que train, no lo anticipaba. "
                    "Ajustar el umbral con el test lo invalidaría como evaluación final, así que no se hizo.", P),
          KeepTogether([figura(fig_scores(), 7.0),
                        Paragraph("Figura 2. Cuantiles del score de las conexiones normales; la línea punteada es el umbral 0,64.", CAP)])]

    s += [Paragraph("6. Prueba de recuperación", H2),
          Paragraph("<i>scripts/prueba_recuperacion.sh</i> clonó GitHub en una carpeta temporal nueva, con cachés de uv, DVC "
                    "y Hugging Face vacías. Instaló las dependencias fijadas (<i>uv sync --frozen</i>), ejecutó "
                    "<i>dvc pull</i> desde el remote público y verificó los 6 SHA-256 de los datos crudos y preparados. "
                    "Luego descargó v1 y v2 por su SHA completo, sin credenciales, y predijo las mismas 25 filas fijas de "
                    f"validación. Diferencia máxima de score: v1 = {rec1['diferencia_max_score']}, v2 = "
                    f"{rec2['diferencia_max_score']} (tolerancia {rec1['tolerancia']}); etiquetas iguales y CSV con "
                    "hashes idénticos a los de referencia. El log y las salidas están en <i>evidencias/recuperacion/prueba</i>.", P)]

    s += [Paragraph("7. Limitaciones", H2),
          Paragraph("(i) UNSW-NB15 es tráfico de laboratorio de 2015 y no demuestra eficacia en una red actual. (ii) Hay un "
                    "cambio de distribución entre train y test: la FPR operativa real sería mayor que la estimada en "
                    "validación, y una validación del mismo archivo resulta optimista. (iii) El umbral de v2 se eligió y "
                    "reportó sobre la misma validación. (iv) Los duplicados se conservan y pesan en las métricas; el "
                    "10 % del test repite entradas vistas. (v) El modelo es un pickle de scikit-learn 1.9.1 y solo debe "
                    "cargarse con esas versiones y desde fuentes confiables. (vi) MLflow corrió en local; la evidencia "
                    "consultable es la exportación en JSON/CSV.", P)]

    s += [Paragraph("Referencias", H2),
          Paragraph("Moustafa, N. y Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection "
                    "systems. MilCIS 2015, IEEE. https://research.unsw.edu.au/projects/unsw-nb15-dataset · Documentación de "
                    "Git, DVC, MLflow Tracking, Hugging Face Hub (descarga por revisión, model cards) y scikit-learn (common "
                    "pitfalls). Uso de IA: Claude Code (Claude Opus 5.5) apoyó el código y la redacción; ver entrega.md.", P)]

    def pie(canvas, doc):
        canvas.setFont("DV", 7)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(1.8 * cm, 1.0 * cm, "MLOps · Defiende el servidor · Grupo B · Rafael Orozco")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"{doc.page}")

    doc = SimpleDocTemplate(str(RAIZ / "informe.pdf"), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.6 * cm, title="Defiende el servidor — informe",
                            author="Rafael Orozco")
    doc.build(s, onFirstPage=pie, onLaterPages=pie)


if __name__ == "__main__":
    main()
