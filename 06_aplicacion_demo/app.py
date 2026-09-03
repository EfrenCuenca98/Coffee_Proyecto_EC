# =============================================================================
# CAFELEAF — Clasificador foliar de variedades de Coffea arabica
#
# Aplicación local que ejecuta el modelo YOLOv8n-seg ya entrenado, segmenta
# hojas de café, extrae sus descriptores morfológicos y permite exportarlos.
# No requiere inicio de sesión: se abre directamente lista para usar.
#
# Ejecutar:  streamlit run app.py
# =============================================================================

import io
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="CafeLeaf · Clasificador foliar",
                   page_icon="☕", layout="wide",
                   initial_sidebar_state="expanded")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.environ.get("CAFELEAF_MODELO",
                             os.path.join(BASE_DIR, "modelo", "best_cafe_yolov8_seg.pt"))
RUTA_CSV_BASE = os.environ.get("CAFELEAF_CSV",
                               os.path.join(BASE_DIR, "datos", "descriptores_por_hoja_v2.csv"))

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Un color fijo por variedad para que la misma clase se vea igual siempre.
# Se define en RGB y se invierte al dibujar, porque OpenCV trabaja en BGR.
PALETA = [(214, 40, 40), (244, 140, 6), (240, 200, 8), (0, 158, 115),
          (0, 114, 178), (140, 81, 195), (204, 63, 130)]

# =============================================================================
# ESTILO
# =============================================================================

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

  .stApp { background: #FBFAF6; }
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important;
               color: #1F3D2B !important; letter-spacing: -0.015em; }

  section[data-testid="stSidebar"] { background: #F3F1E9;
                                     border-right: 1px solid #E3E0D6; }

  .marca { font-family:'Fraunces',Georgia,serif; font-size:1.6rem; font-weight:600;
           color:#1F3D2B; line-height:1.1; margin-bottom:2px; }
  .marca-sub { font-size:0.78rem; color:#7A7566; letter-spacing:0.06em;
               text-transform:uppercase; margin-bottom:18px; }

  .tarjeta { background:#FFFFFF; border:1px solid #E3E0D6; border-left:3px solid #2E5E42;
             border-radius:3px; padding:0.85rem 1rem; margin-bottom:0.6rem; }
  .t-label { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;
             color:#7A7566; margin-bottom:0.25rem; }
  .t-valor { font-size:1.5rem; font-weight:600; color:#1F3D2B; line-height:1.1; }
  .t-apoyo { font-size:0.78rem; color:#7A7566; margin-top:2px; }

  .chip { display:inline-block; padding:3px 10px; border-radius:12px;
          font-size:0.75rem; font-weight:500; background:#E8EFE9; color:#1F3D2B;
          margin-right:6px; }

  .aviso { background:#FDF4E3; border-left:3px solid #B8860B; padding:0.8rem 1rem;
           border-radius:3px; font-size:0.88rem; color:#5C4A17; }
  .rechazo { background:#FCEDEC; border-left:3px solid #C1121F; padding:0.8rem 1rem;
             border-radius:3px; font-size:0.88rem; color:#7A1015; margin-bottom:0.7rem; }
  .tarjeta-no { border-left-color:#C1121F !important; }

  .stButton>button { font-weight:500; border-radius:4px; }
  div[data-testid="stForm"] { border:1px solid #E3E0D6; border-radius:4px;
                              padding:1.4rem; background:#FFFFFF; }
  footer, #MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CARGA DE RECURSOS (una sola vez por sesión)
# =============================================================================

@st.cache_resource(show_spinner="Cargando el modelo entrenado...")
def cargar_modelo(ruta):
    if not os.path.exists(ruta):
        return None, None
    from ultralytics import YOLO
    modelo = YOLO(ruta)
    return modelo, modelo.names

DESCRIPTORES_REFERENCIA = [
    "H", "S", "V", "H_sd", "S_sd", "V_sd",
    "aspect_ratio", "solidity", "circularity", "rugosidad_borde",
    "homogeneity", "contrast", "energy", "correlation", "dissimilarity",
    "laplaciano_var", "desviacion_local", "rango_iq_brillo",
]

@st.cache_resource(show_spinner="Preparando la referencia morfológica...")
def cargar_plausibilidad(_df_base, percentil):
    cols = [c for c in DESCRIPTORES_REFERENCIA if c in _df_base.columns]
    X = _df_base[cols].dropna().values
    if len(X) < 50:
        return None
    mu = X.mean(axis=0)
    cov_inv = np.linalg.pinv(np.cov(X.T))
    dif = X - mu
    d2 = np.einsum("ij,jk,ik->i", dif, cov_inv, dif)
    return {"cols": cols, "mu": mu, "cov_inv": cov_inv,
            "umbral": float(np.percentile(d2, percentil)),
            "n": len(X)}

@st.cache_resource(show_spinner="Ajustando los modelos de mezcla gaussiana...")
def cargar_referencia(ruta_csv):
    if not os.path.exists(ruta_csv):
        return None, None, None
    sys.path.insert(0, BASE_DIR)
    from descriptores_v2 import ajustar_gmms, verificar_coherencia
    df = pd.read_csv(ruta_csv)
    ok, mensaje = verificar_coherencia(df)
    if not ok:
        return None, None, mensaje
    return ajustar_gmms(df), df, mensaje

modelo, CLASES = cargar_modelo(RUTA_MODELO)
GMMS, DF_BASE, MSG_BASE = cargar_referencia(RUTA_CSV_BASE)

if modelo is None:
    st.error(f"No se encontró el modelo en `{RUTA_MODELO}`.\n\n"
             "Copia `best_cafe_yolov8_seg.pt` a la carpeta `modelo/` y recarga la página.")
    st.stop()

# =============================================================================
# MOTOR
# =============================================================================

def dibujar(img_bgr, mascaras, cajas, clases, confs, pintar_mascara):
    salida = img_bgr.copy()
    alto, ancho = salida.shape[:2]
    grosor = max(2, int(round(min(alto, ancho) / 380)))
    escala = max(0.55, min(alto, ancho) / 950)
    capa = salida.copy()

    for i, m in enumerate(mascaras):
        r, g, b = PALETA[int(clases[i]) % len(PALETA)]
        color = (b, g, r)
        binaria = (cv2.resize(m, (ancho, alto)) > 0.5).astype(np.uint8)
        if pintar_mascara:
            capa[binaria == 1] = color
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(salida, contornos, -1, color, grosor)

    if pintar_mascara:
        salida = cv2.addWeighted(capa, 0.32, salida, 0.68, 0)

    ocupados = []

    def choca(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    for i, caja in enumerate(cajas):
        r, g, b = PALETA[int(clases[i]) % len(PALETA)]
        color = (b, g, r)
        x1, y1, x2, y2 = [int(v) for v in caja]
        cv2.rectangle(salida, (x1, y1), (x2, y2), color, grosor)

        texto = f"{CLASES[int(clases[i])]} {confs[i]:.2f} ({i + 1})"
        (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX,
                                         escala, max(1, grosor - 1))
        alto_etq = th + base + 8

        top = max(0, y1 - alto_etq)
        for _ in range(12):
            rect = (x1, top, x1 + tw + 10, top + alto_etq)
            if not any(choca(rect, o) for o in ocupados):
                break
            top += alto_etq + 3
        top = min(top, alto - alto_etq - 1)
        rect = (x1, top, x1 + tw + 10, top + alto_etq)
        ocupados.append(rect)

        cv2.rectangle(salida, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
        cv2.putText(salida, texto, (x1 + 5, top + th + 4), cv2.FONT_HERSHEY_DUPLEX,
                    escala, (255, 255, 255), max(1, grosor - 1), cv2.LINE_AA)
    return salida

def analizar(img_bgr, nombre, conf, pintar_mascara, nms_agnostico=True,
             iou=0.45, ref_plaus=None):
    from descriptores_v2 import (calcular_color, calcular_forma,
                                 calcular_textura, clasificar_nivel)
    filas, anotada = [], None

    for r in modelo.predict(source=img_bgr, conf=conf, iou=iou,
                            agnostic_nms=nms_agnostico, verbose=False):
        original = r.orig_img
        if r.masks is None:
            anotada = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            continue

        mascaras = r.masks.data.cpu().numpy()
        cajas = r.boxes.xyxy.cpu().numpy()
        clases = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        anotada = cv2.cvtColor(
            dibujar(original, mascaras, cajas, clases, confs, pintar_mascara),
            cv2.COLOR_BGR2RGB)

        for i, m in enumerate(mascaras):
            binaria = (cv2.resize(m, (original.shape[1], original.shape[0]))
                       > 0.5).astype(np.uint8)
            color = calcular_color(original, binaria)
            forma = calcular_forma(binaria)
            textura = calcular_textura(original, binaria)

            fila = {"Imagen": nombre, "Hoja_ID": i + 1,
                    "Variedad": CLASES[clases[i]],
                    "Probabilidad (%)": round(float(confs[i]) * 100, 1)}

            if None in (color, forma, textura) or GMMS is None:
                fila.update({"Color": "No calculable", 
                             "Forma": "No calculable", 
                             "Textura": "No calculable"})
                filas.append(fila)
                continue

            desc = {**color, **forma, **textura}
            nc, _ = clasificar_nivel(desc["H"], GMMS["color"])
            nf, _ = clasificar_nivel(desc["aspect_ratio"], GMMS["forma"])
            nt, _ = clasificar_nivel(desc["contrast"], GMMS["textura"])
            
            fila.update({"Color": nc,
                         "Forma": nf,
                         "Textura": nt})

            d2 = distancia_al_cafe(desc, ref_plaus)
            if d2 is None:
                fila["Detección típica"] = "Sin evaluar"
                fila["Distancia morfológica"] = None
            else:
                compatible = d2 <= ref_plaus["umbral"]
                fila["Detección típica"] = "Sí" if compatible else "No"
                fila["Distancia morfológica"] = round(d2, 1)

            for k, v in desc.items():
                fila[k] = round(float(v), 3)
            filas.append(fila)

    return filas, anotada

def distancia_al_cafe(descriptores, ref):
    if ref is None:
        return None
    try:
        x = np.array([float(descriptores[c]) for c in ref["cols"]])
    except (KeyError, TypeError, ValueError):
        return None
    if np.isnan(x).any():
        return None
    dif = x - ref["mu"]
    return float(dif @ ref["cov_inv"] @ dif)

def tarjeta(label, valor, apoyo=""):
    st.markdown(f"<div class='tarjeta'><div class='t-label'>{label}</div>"
                f"<div class='t-valor'>{valor}</div>"
                f"<div class='t-apoyo'>{apoyo}</div></div>", unsafe_allow_html=True)

# =============================================================================
# ESTADO
# =============================================================================

# Definimos el orden estricto de las columnas principales
COLUMNAS_TABLA = ["Imagen", "Hoja_ID", "Variedad", "Probabilidad (%)",
                  "Color", "Forma", "Textura", 
                  "Detección típica", "Distancia morfológica"]

st.session_state.setdefault("resultados", [])
st.session_state.setdefault("tabla", pd.DataFrame())
st.session_state.setdefault("ajustes_usados", None)
st.session_state.setdefault("cancelado", False)

# =============================================================================
# BARRA LATERAL
# =============================================================================

with st.sidebar:
    st.markdown("<div class='marca'>CafeLeaf</div>"
                "<div class='marca-sub'>Clasificación foliar</div>",
                unsafe_allow_html=True)

    st.subheader("Ajustes")

    conf = st.slider("Confianza mínima", 0.05, 0.95, 0.40, 0.05,
                     help="Detecciones por debajo de este valor se descartan. "
                          "Bájalo si el modelo no encuentra hojas que sí están "
                          "en la fotografía; súbelo si propone detecciones "
                          "sobre el fondo.")

    pintar = st.toggle("Superponer máscara de color", value=True,
                       help="Desactivado, deja visible el color real de la "
                            "hoja, lo que permite verificar a simple vista la "
                            "clasificación cromática.")

    nms_agnostico = False
    iou = 0.45
    verificar = True
    percentil = 100

    with st.expander("Opciones avanzadas"):
        verificar = st.toggle(
            "Avisar ante detecciones atípicas", value=True,
            help="Contrasta los descriptores de cada detección con los de las "
                 "hojas del conjunto de referencia y advierte cuando la "
                 "morfología se aparta de lo observado. Sirve para señalar "
                 "fotografías que no contienen hojas de café.")
        percentil = st.slider(
            "Sensibilidad del aviso", 95, 100, 100, 1,
            help="Percentil del conjunto de referencia que se acepta sin "
                 "advertencia. Con 100 no se marca ninguna hoja de café real; "
                 "valores menores son más estrictos y pueden señalar hojas "
                 "legítimas poco frecuentes.",
            disabled=not verificar)

    st.divider()
    st.caption(f"Variedades reconocidas · {len(CLASES)}")
    st.markdown(" ".join(f"<span class='chip'>{v}</span>" for v in CLASES.values()),
                unsafe_allow_html=True)

    if GMMS is None:
        st.warning("Sin CSV de referencia: solo se reportan descriptores numéricos, "
                   "sin las categorías morfológicas.")

# =============================================================================
# CABECERA
# =============================================================================

REF_PLAUS = (cargar_plausibilidad(DF_BASE, percentil)
             if (verificar and DF_BASE is not None) else None)

st.title("Clasificación foliar de variedades de café")
st.caption("Segmentación con YOLOv8n-seg y caracterización morfológica de color, "
           "forma y textura · Universidad Nacional de Loja")

tab_analisis, tab_resultados, tab_ayuda = st.tabs(
    ["Analizar", "Resultados", "Cómo usar"])

# ----------------------------------------------------------------- ANALIZAR
with tab_analisis:
    modo = st.radio("Origen de las imágenes",
                    ["Subir archivos", "Carpeta del equipo"],
                    horizontal=True, label_visibility="collapsed")

    rutas, archivos = [], []

    if modo == "Subir archivos":
        archivos = st.file_uploader(
            "Arrastra aquí una o varias fotografías de hojas",
            type=[e.strip(".") for e in EXTS], accept_multiple_files=True,
            help="Solo se aceptan imágenes: JPG, JPEG, PNG, BMP y WEBP.")

        if archivos:
            validos, invalidos = [], []
            for a in archivos:
                try:
                    Image.open(a).verify()
                    a.seek(0)
                    validos.append(a)
                except Exception:
                    invalidos.append(a.name)
            archivos = validos
            if invalidos:
                st.error("No son imágenes válidas y se descartaron: "
                         + ", ".join(invalidos))
            if archivos:
                st.caption(f"{len(archivos)} imagen(es) lista(s) para analizar.")
    else:
        carpeta = st.text_input("Ruta de la carpeta",
                                placeholder=r"C:\Users\alex_\Escritorio\hojas_nuevas")
        if carpeta:
            if os.path.isdir(carpeta):
                rutas = sorted(os.path.join(carpeta, f) for f in os.listdir(carpeta)
                               if f.lower().endswith(EXTS))
                if rutas:
                    st.caption(f"{len(rutas)} imagen(es) encontrada(s). "
                               "Los archivos que no sean imágenes se ignoran.")
                else:
                    st.warning("La carpeta existe pero no contiene imágenes.")
            else:
                st.error("Esa carpeta no existe. Revisa la ruta.")

    acumular = st.checkbox(
        "Acumular con los resultados anteriores", value=False,
        help="Desactivado, cada segmentación reemplaza los resultados previos. "
             "Actívalo para ir sumando lotes distintos en una sola tabla. En "
             "ambos casos, volver a analizar una imagen ya procesada sustituye "
             "su resultado en lugar de duplicarlo.")

    ajustes_actuales = (round(conf, 2), bool(pintar), bool(nms_agnostico),
                        round(iou, 2), bool(verificar), int(percentil))
    hay_entrada = bool(archivos or rutas)

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        lanzar = st.button("Segmentar", type="primary", use_container_width=True,
                           disabled=not hay_entrada)
    with col_b:
        reprocesar = st.button(
            "Reprocesar", use_container_width=True,
            disabled=not st.session_state.resultados,
            help="Vuelve a analizar las imágenes ya cargadas aplicando los "
                 "ajustes actuales de la barra lateral.")
    with col_c:
        if st.session_state.resultados:
            if st.button("Limpiar resultados"):
                st.session_state.resultados = []
                st.session_state.tabla = pd.DataFrame()
                st.session_state.ajustes_usados = None
                st.rerun()

    if (st.session_state.resultados
            and st.session_state.ajustes_usados
            and st.session_state.ajustes_usados != ajustes_actuales):
        st.markdown("<div class='aviso'><b>Los ajustes cambiaron.</b> "
                    "Los resultados en pantalla se generaron con la "
                    "configuración anterior. Pulsa <b>Reprocesar</b> para "
                    "aplicar los valores actuales.</div>",
                    unsafe_allow_html=True)

    # ------------------------------------------------------------ EJECUCIÓN
    entradas = []
    if lanzar:
        for a in (archivos or []):
            entradas.append((a.name, Image.open(a).convert("RGB")))
        for r in rutas:
            entradas.append((os.path.basename(r), Image.open(r).convert("RGB")))
    elif reprocesar:
        entradas = [(r["nombre"], r["pil"]) for r in st.session_state.resultados
                    if r.get("pil") is not None]
        if not entradas:
            st.warning("No se conservan las imágenes originales de esa sesión. "
                       "Vuelve a cargarlas para reprocesarlas.")

    if entradas:
        zona_cancelar = st.empty()
        if zona_cancelar.button("Cancelar", key="cancelar_proceso"):
            st.warning("Segmentación cancelada.")
            st.stop()

        barra = st.progress(0.0, text="Segmentando...")
        nuevos, filas_todas = [], []

        for i, (nombre, pil) in enumerate(entradas, start=1):
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            filas, anotada = analizar(bgr, nombre, conf, pintar,
                                      nms_agnostico=nms_agnostico, iou=iou,
                                      ref_plaus=REF_PLAUS)
            nuevos.append({"nombre": nombre, "pil": pil,
                           "imagen": anotada if anotada is not None else np.array(pil),
                           "filas": filas})
            filas_todas.extend(filas)
            barra.progress(i / len(entradas),
                           text=f"Segmentando {i} de {len(entradas)}...")

        barra.empty()
        zona_cancelar.empty()

        nombres_nuevos = {n["nombre"] for n in nuevos}
        if not acumular or reprocesar:
            previos, tabla_previa = [], pd.DataFrame()
        else:
            previos = [r for r in st.session_state.resultados
                       if r["nombre"] not in nombres_nuevos]
            tabla_previa = st.session_state.tabla
            if not tabla_previa.empty:
                tabla_previa = tabla_previa[~tabla_previa["Imagen"].isin(nombres_nuevos)]

        st.session_state.resultados = previos + nuevos
        st.session_state.tabla = (
            pd.concat([tabla_previa, pd.DataFrame(filas_todas)], ignore_index=True)
            if filas_todas else tabla_previa)
        st.session_state.ajustes_usados = ajustes_actuales

        detectadas = len(filas_todas)
        sin_det = sum(1 for n in nuevos if not n["filas"])
        if detectadas:
            st.success(f"{detectadas} hoja(s) detectada(s) en {len(entradas)} imagen(es).")
            st.info("Revisa la pestaña **Resultados** para ver las hojas segmentadas.")
        if sin_det:
            st.markdown(f"<div class='aviso'><b>{sin_det} imagen(es) sin detección.</b> "
                        f"El umbral actual es {conf:.2f}. Bájalo desde la barra "
                        f"lateral y pulsa Reprocesar.</div>", unsafe_allow_html=True)

# --------------------------------------------------------------- RESULTADOS
with tab_resultados:
    df = st.session_state.tabla
    resultados = st.session_state.resultados

    if not resultados:
        st.info("Todavía no hay resultados. Analiza imágenes desde la primera pestaña.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            tarjeta("Imágenes", len(resultados))
        with c2:
            tarjeta("Hojas detectadas", len(df))
        with c3:
            if not df.empty:
                top = df["Variedad"].value_counts()
                tarjeta("Variedad predominante", top.idxmax(),
                        f"{top.max()} de {len(df)} hojas")
            else:
                tarjeta("Variedad predominante", "—")
        with c4:
            if not df.empty and "Detección típica" in df.columns:
                rechazadas = int((df["Detección típica"] == "No").sum())
                if rechazadas:
                    tarjeta("Detecciones atípicas", rechazadas,
                            f"de {len(df)} detecciones")
                else:
                    tarjeta("Confianza media",
                            f'{df["Probabilidad (%)"].mean():.1f} %',
                            "Ninguna detección atípica")
            else:
                tarjeta("Confianza media",
                        f'{df["Probabilidad (%)"].mean():.1f} %' if not df.empty else "—")

        st.divider()
        st.subheader("Distribución por variedad")

        if df.empty:
            st.info("Todavía no hay hojas detectadas.")
        else:
            conteo = df["Variedad"].value_counts()
            porcentaje = (conteo / conteo.sum() * 100).round(1)

            g_izq, g_der = st.columns([1.4, 1])

            with g_izq:
                import matplotlib.pyplot as plt

                indices = {v: k for k, v in CLASES.items()}
                colores = [tuple(c / 255 for c in PALETA[indices.get(v, 0) % len(PALETA)])
                           for v in conteo.index]

                fig, ax = plt.subplots(figsize=(7, 0.55 * len(conteo) + 1.6))
                barras = ax.barh(conteo.index[::-1], conteo.values[::-1],
                                 color=colores[::-1], height=0.62,
                                 edgecolor="#1A1A1A", linewidth=0.6)

                for barra, valor, pct in zip(barras, conteo.values[::-1],
                                             porcentaje.values[::-1]):
                    ax.text(barra.get_width() + max(conteo.values) * 0.015,
                            barra.get_y() + barra.get_height() / 2,
                            f"{valor}  ·  {pct:.1f} %",
                            va="center", fontsize=10, color="#3A3A3A")

                ax.set_xlim(0, max(conteo.values) * 1.28)
                ax.set_xlabel("Hojas detectadas", fontsize=10)
                ax.grid(axis="x", linestyle="--", alpha=0.3)
                ax.set_axisbelow(True)
                for lado in ["top", "right", "left"]:
                    ax.spines[lado].set_visible(False)
                ax.tick_params(axis="y", length=0, labelsize=11)
                fig.patch.set_facecolor("#FBFAF6")
                ax.set_facecolor("#FBFAF6")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with g_der:
                resumen = pd.DataFrame({
                    "Variedad": conteo.index,
                    "Hojas": conteo.values,
                    "Porcentaje (%)": porcentaje.values,
                    "Confianza media (%)": [
                        round(df[df["Variedad"] == v]["Probabilidad (%)"].mean(), 1)
                        for v in conteo.index],
                })
                st.dataframe(resumen, use_container_width=True, hide_index=True)

                if len(conteo) == 1:
                    st.caption("Todas las hojas se asignaron a una sola variedad.")
                else:
                    st.caption(f"{len(conteo)} variedades presentes en "
                               f"{len(resultados)} imagen(es).")

        st.divider()
        st.subheader("Revisión por imagen")

        nombres = [r["nombre"] for r in resultados]
        elegida = st.selectbox("Imagen", nombres, label_visibility="collapsed")
        actual = next(r for r in resultados if r["nombre"] == elegida)

        izq, der = st.columns([1.15, 1])
        with izq:
            st.image(actual["imagen"], use_container_width=True)
            st.caption("El número entre paréntesis en cada etiqueta corresponde "
                       "al Hoja_ID de la tabla.")

        with der:
            if not actual["filas"]:
                st.warning("No se detectó ninguna hoja por encima del umbral.")
            else:
                for f in actual["filas"]:
                    compatible = f.get("Detección típica", "Sin evaluar")
                    marca = "" if compatible != "No" else "  ·  detección atípica"
                    st.markdown(f"**Hoja {f['Hoja_ID']} · {f['Variedad']}**{marca}")

                    if compatible == "No":
                        st.markdown(
                            "<div class='rechazo'><b>Detección atípica.</b><br>"
                            "Los descriptores de color, forma y textura de esta "
                            "detección se apartan de los observados en las hojas "
                            f"de referencia (distancia {f['Distancia morfológica']}, "
                            f"límite {REF_PLAUS['umbral']:.1f}). Conviene "
                            "verificar la fotografía antes de dar la variedad "
                            "por válida.</div>",
                            unsafe_allow_html=True)
                    
                    tarjeta("Variedad Predicha", f["Variedad"],
                            f'Confianza del modelo (YOLO): {f["Probabilidad (%)"]} %')
                    
                    st.caption("Estimaciones Morfológicas (Extraídas de la máscara)")
                    
                    if f.get("Color") != "No calculable":
                        # Sin st.columns(), las tarjetas se apilan verticalmente ocupando todo el ancho
                        tarjeta("Color", f["Color"], f'Matiz (H): {f.get("H", "")}')
                        tarjeta("Forma", f["Forma"], f'Rel. Aspecto: {f.get("aspect_ratio", "")}')
                        tarjeta("Textura", f["Textura"], f'Contraste: {f.get("contrast", "")}')

                    with st.expander(f"Descriptores numéricos · hoja {f['Hoja_ID']}"):
                        nums = {k: v for k, v in f.items()
                                if isinstance(v, (int, float))
                                and k not in ("Hoja_ID", "Probabilidad (%)")}
                        st.dataframe(pd.DataFrame(nums.items(),
                                                  columns=["Descriptor", "Valor"]),
                                     use_container_width=True, hide_index=True)

        if not df.empty:
            st.divider()
            st.subheader("Tabla de resultados")

            f1, f2 = st.columns([1, 1])
            with f1:
                variedades = ["Todas"] + sorted(df["Variedad"].unique())
                filtro = st.selectbox("Filtrar por variedad", variedades)
            with f2:
                ver_numericos = st.checkbox(
                    "Mostrar descriptores numéricos", value=False,
                    help="Valores crudos de color, forma y textura calculados "
                         "con OpenCV y scikit-image. Son el insumo del que se "
                         "derivan las categorías, útiles para verificación y "
                         "para el informe, no para la lectura habitual.")

            vista = df if filtro == "Todas" else df[df["Variedad"] == filtro]

            # LÓGICA MODIFICADA: Asegurar que Color, Forma y Textura siempre queden juntas
            cols_principales = [c for c in COLUMNAS_TABLA if c in vista.columns]
            
            if ver_numericos:
                # Si se piden los números, se ponen AL FINAL de las columnas principales
                cols_numericas = [c for c in vista.columns if c not in cols_principales]
                tabla_vista = vista[cols_principales + cols_numericas]
            else:
                # Si no, solo se muestran las principales
                tabla_vista = vista[cols_principales]

            st.dataframe(tabla_vista, use_container_width=True, hide_index=True)
            st.caption(f"{len(vista)} hoja(s) en la selección actual. "
                       "La descarga incluye siempre todos los descriptores.")

            marca = datetime.now().strftime("%Y%m%d_%H%M%S")
            e1, e2 = st.columns(2)
            with e1:
                st.download_button("Descargar CSV",
                                   vista.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"morfologia_{marca}.csv",
                                   mime="text/csv", use_container_width=True)
            with e2:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as w:
                    vista.to_excel(w, index=False, sheet_name="Morfologia")
                st.download_button("Descargar Excel", buffer.getvalue(),
                                   file_name=f"morfologia_{marca}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument."
                                        "spreadsheetml.sheet",
                                   use_container_width=True)

# -------------------------------------------------------------------- AYUDA
with tab_ayuda:
    st.subheader("Cómo obtener buenos resultados")
    st.markdown("""
    - Encuadra la hoja de modo que ocupe buena parte del cuadro.
    - Usa luz natural y evita las sombras duras sobre el limbo.
    - Prefiere un fondo despejado: el follaje superpuesto es la causa principal
      de que el modelo no detecte la hoja o proponga detecciones falsas.
    - Si no aparece ninguna detección, baja la confianza mínima antes de
      descartar la fotografía.
    """)

    st.subheader("Confianza del Modelo vs. Estimaciones Morfológicas")
    st.markdown("""
    Para garantizar el rigor técnico del análisis, la interfaz separa estrictamente 
    la predicción algorítmica de la medición física:
    
    - **Confianza del modelo (YOLO):** Es el único porcentaje mostrado. Indica qué tan segura 
      está la red neuronal convolucional de su clasificación (ej. Castillo 92%).
    - **Estimaciones Morfológicas (Color, Forma, Textura):** No son probabilidades, 
      sino **mediciones numéricas reales** extraídas directamente de los píxeles de la 
      máscara foliar utilizando OpenCV y scikit-image (ej. Matiz H: 38.4, o Relación de Aspecto: 2.53). 
    """)

    st.subheader("Cómo se asignan las categorías descriptivas")
    st.markdown("""
    Para facilitar la lectura agronómica, los valores numéricos crudos se traducen a 
    categorías verbales (ej. "Verde claro", "Elíptica", "Lisa"). 
    
    Esta traducción no es arbitraria: se realiza mediante un Modelo de Mezcla Gaussiana (GMM) 
    ajustado sobre el conjunto de datos original. De esta forma, cada etiqueta cualitativa en 
    la interfaz está respaldada por una estimación matemática auditable y calculable.
    """)

    st.subheader("Verificación de plausibilidad")
    st.markdown("""
    Cada detección se contrasta con la distribución conjunta de los
    descriptores de las hojas del conjunto base mediante la distancia de
    Mahalanobis. Cuando una detección queda fuera del rango observado, la
    aplicación lo advierte en lugar de reportar una variedad sin más.

    El criterio evalúa el conjunto de descriptores y no cada uno por separado:
    un objeto puede tener un matiz dentro del rango del café y una forma dentro
    del rango del café, y aun así presentar una combinación de rasgos que
    ninguna hoja real exhibe. Esto permite detectar objetos intrusos (falsos positivos).
    """)

    if DF_BASE is not None:
        st.subheader("Referencia del conjunto base")
        st.markdown("""
        Promedios de las estimaciones morfológicas calculadas sobre el dataset original 
        previo al aumento de datos.
        """)
        resumen = (DF_BASE.groupby("Variedad")
                   .agg(Hojas=("H", "count"), H=("H", "mean"), S=("S", "mean"),
                        V=("V", "mean"),
                        **{"Rel. aspecto": ("aspect_ratio", "mean"),
                           "Contraste": ("contrast", "mean")})
                   .round(2).reset_index())
        st.dataframe(resumen, use_container_width=True, hide_index=True)