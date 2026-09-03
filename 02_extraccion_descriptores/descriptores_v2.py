# =============================================================================
# descriptores_v2.py — Módulo compartido de extracción morfológica
#
# Contiene EXACTAMENTE las mismas funciones que Extraer_morfologia_v2.py, el
# script que generó descriptores_por_hoja_v2.csv.
#
# Es indispensable que el runtime (notebook y aplicación Streamlit) use estas
# funciones y no las de la versión anterior: los modelos de mezcla gaussiana
# se ajustan sobre el CSV v2, y si una hoja nueva se midiera con el cálculo
# antiguo caería siempre en el componente más bajo, porque las escalas de los
# descriptores texturales no son comparables entre ambas versiones.
#
# Uso:
#     from descriptores_v2 import calcular_color, calcular_forma, calcular_textura
# =============================================================================

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

# --- Parámetros: deben coincidir con los de Extraer_morfologia_v2.py ---------
NIVELES_GRIS = 32
DISTANCIAS = [1, 3, 5, 10]
ANGULOS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
APLICAR_CLAHE = True


def calcular_color(img_bgr, mask):
    """Medias y desviaciones de los canales HSV sobre los píxeles de la hoja."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    px = hsv[mask == 1]
    if len(px) == 0:
        return None
    return {
        "H": float(px[:, 0].mean()), "S": float(px[:, 1].mean()),
        "V": float(px[:, 2].mean()),
        "H_sd": float(px[:, 0].std()), "S_sd": float(px[:, 1].std()),
        "V_sd": float(px[:, 2].std()),
    }


def calcular_forma(mask):
    """Descriptores de forma a partir del contorno de la máscara."""
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    c = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimetro = cv2.arcLength(c, True)
    if area == 0 or perimetro == 0:
        return None

    # Rectángulo de área mínima: invariante a la orientación de la fotografía
    (_, _), (w_rot, h_rot), _ = cv2.minAreaRect(c)
    largo, ancho = max(w_rot, h_rot), min(w_rot, h_rot)

    x, y, w, h = cv2.boundingRect(c)
    hull_area = cv2.contourArea(cv2.convexHull(c))

    return {
        "aspect_ratio": float(largo / max(ancho, 1e-6)),
        "aspect_ratio_ejes": float(max(w, h) / max(min(w, h), 1)),
        "solidity": float(area / hull_area) if hull_area > 0 else 0.0,
        "circularity": float((4 * np.pi * area) / (perimetro ** 2)),
        "rugosidad_borde": float(perimetro / (2 * np.sqrt(np.pi * area))),
        "area_px": float(area),
    }


def calcular_textura(img_bgr, mask):
    """Descriptores texturales con la GLCM restringida a los píxeles de la hoja."""
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if APLICAR_CLAHE:
        gris = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)

    x, y, w, h = cv2.boundingRect(mask)
    if w < 8 or h < 8:
        return None

    recorte_gris = gris[y:y + h, x:x + w]
    recorte_mask = mask[y:y + h, x:x + w]

    pix_hoja = recorte_gris[recorte_mask == 1]
    if len(pix_hoja) < 50:
        return None

    # Se cuantiza la hoja a los niveles 1..N y se reserva el 0 para el fondo
    lo, hi = np.percentile(pix_hoja, [1, 99])
    if hi <= lo:
        hi = lo + 1
    escalado = np.clip((recorte_gris.astype(np.float32) - lo) / (hi - lo), 0, 1)
    cuantizado = (escalado * (NIVELES_GRIS - 1)).astype(np.uint8) + 1
    cuantizado[recorte_mask == 0] = 0

    glcm = graycomatrix(cuantizado, distances=DISTANCIAS, angles=ANGULOS,
                        levels=NIVELES_GRIS + 1, symmetric=True,
                        normed=False).astype(np.float64)

    # Se descartan los pares que involucran fondo y se renormaliza
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0
    sumas = glcm.sum(axis=(0, 1), keepdims=True)
    sumas[sumas == 0] = 1
    glcm /= sumas

    props = {n: float(graycoprops(glcm, n).mean())
             for n in ["homogeneity", "contrast", "energy",
                       "correlation", "dissimilarity"]}

    # Medidas directas de relieve superficial
    lap = cv2.Laplacian(recorte_gris, cv2.CV_32F, ksize=3)
    props["laplaciano_var"] = float(np.nanvar(
        np.where(recorte_mask == 1, lap, np.nan)))

    media_local = cv2.blur(recorte_gris.astype(np.float32), (9, 9))
    desv_local = np.abs(recorte_gris.astype(np.float32) - media_local)
    props["desviacion_local"] = float(np.nanmean(
        np.where(recorte_mask == 1, desv_local, np.nan)))

    props["rango_iq_brillo"] = float(np.percentile(pix_hoja, 75) -
                                     np.percentile(pix_hoja, 25))
    return props


# =============================================================================
# CATEGORIZACIÓN CUALITATIVA
# =============================================================================

def ajustar_gmms(df_base):
    """Ajusta los tres modelos de mezcla sobre el CSV base v2.

    Devuelve un diccionario listo para pasar a clasificar_nivel().
    """
    from sklearn.mixture import GaussianMixture

    definiciones = {
    "color":   ("H",            ["Verde amarillento", "Verde claro", "Verde oscuro"]),
    "forma":   ("aspect_ratio", ["Redondeada", "Elíptica", "Alargada"]),
    "textura": ("contrast",     ["Lisa", "Coriácea", "Rugosa"]),
    }

    modelos = {}
    for clave, (columna, niveles) in definiciones.items():
        if columna not in df_base.columns:
            raise KeyError(
                f"El CSV base no contiene la columna '{columna}'. "
                "Verifica que estés usando descriptores_por_hoja_v2.csv.")
        X = df_base[[columna]].dropna().values
        gmm = GaussianMixture(n_components=3, random_state=42, n_init=10,
                              covariance_type="full").fit(X)
        modelos[clave] = {"gmm": gmm,
                          "orden": np.argsort(gmm.means_.flatten()),
                          "niveles": niveles,
                          "columna": columna}
    return modelos


def clasificar_nivel(valor, modelo):
    """Devuelve (nombre_del_nivel, probabilidad_en_porcentaje)."""
    probs = modelo["gmm"].predict_proba(np.array([[valor]]))[0][modelo["orden"]]
    ganador = int(np.argmax(probs))
    return modelo["niveles"][ganador], round(float(probs[ganador]) * 100, 1)


def verificar_coherencia(df_base):
    """Comprueba que el CSV cargado corresponde a la versión v2.

    El contraste calculado con ocho niveles de gris y fondo incluido se mueve
    por debajo de 3; el de la versión v2 supera ampliamente ese valor. Una
    discrepancia aquí significa que el CSV y el código de extracción no
    coinciden, y que las categorías resultantes no serían fiables.
    """
    if "contrast" not in df_base.columns:
        return False, "El CSV no contiene la columna 'contrast'."
    mediana = float(df_base["contrast"].median())
    if mediana < 3:
        return False, (f"El contraste mediano del CSV es {mediana:.2f}, propio de la "
                       "versión antigua (8 niveles de gris, fondo incluido). Este "
                       "módulo calcula la textura con 32 niveles y fondo excluido, "
                       "por lo que las escalas no coinciden. Regenera el CSV con "
                       "Extraer_morfologia_v2.py.")
    faltantes = [c for c in ["desviacion_local", "dissimilarity", "correlation"]
                 if c not in df_base.columns]
    if faltantes:
        return False, f"Faltan columnas de la versión v2: {', '.join(faltantes)}."
    return True, f"CSV v2 verificado: {len(df_base)} hojas, contraste mediano {mediana:.2f}."
