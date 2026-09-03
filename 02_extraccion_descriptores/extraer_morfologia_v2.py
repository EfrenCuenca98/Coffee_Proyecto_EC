# =============================================================================
# EXTRACCIÓN DE DESCRIPTORES MORFOLÓGICOS 
# =============================================================================

import os
import glob
import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops

# =============================================================================
# PARÁMETROS AJUSTABLES — aquí es donde conviene experimentar
# =============================================================================

DATASET_DIR = r"C:\Users\alex_\OneDrive\Escritorio\9no\Tesis\Dataset YOLOv8"
SPLITS = ["train", "valid", "test"]
CLASES = ['Bourbon', 'Castillo', 'Geisha', 'Mundonovo','SanSalvador', 'Sarchimor', 'Typica']

# --- TEXTURA -----------------------------------------------------------------

# Niveles de gris para la GLCM.
# El original usaba 8. Con 8 niveles, un relieve foliar sutil (la bulosidad
# típica de Bourbon) queda por debajo de la resolución del descriptor: dos
# hojas muy distintas caen en el mismo nivel. Subir a 32 o 64 recupera esa
# información. Coste: la GLCM crece al cuadrado (64 niveles = 4096 celdas),
# así que el cálculo es más lento.
NIVELES_GRIS = 32

# Distancias de co-ocurrencia, en píxeles.
# El original usaba solo distancia 1, que mide variación entre píxeles
# contiguos, es decir, ruido del sensor. La bulosidad de una hoja tiene una
# escala espacial de varios milímetros: en una imagen de 640 px equivale a
# decenas de píxeles. Incluir distancias mayores captura el relieve real.
DISTANCIAS = [1, 3, 5, 10]

# Ángulos en radianes.
# El original usaba solo 0° (horizontal). Una hoja fotografiada en diagonal
# presenta su nervadura en otra orientación, de modo que el descriptor
# dependía del ángulo de la foto. Promediar sobre cuatro ángulos lo vuelve
# invariante a la rotación.
ANGULOS = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Ecualización adaptativa del contraste antes de calcular la GLCM.
# Reduce el efecto de que una foto esté más iluminada que otra, de modo que
# el contraste medido refleje el relieve de la hoja y no la hora del día.
APLICAR_CLAHE = True

# --- SALIDA ------------------------------------------------------------------

CSV_SALIDA = os.path.join(DATASET_DIR, "descriptores_por_hoja_v2.csv")
CSV_RESUMEN = os.path.join(DATASET_DIR, "resumen_morfologia_por_variedad_v2.csv")


# =============================================================================
# COLOR
# =============================================================================

def calcular_color(img_bgr, mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    px = hsv[mask == 1]
    if len(px) == 0:
        return None
    # Se añaden las desviaciones: una hoja con moteado o clorosis tiene
    # media de matiz normal pero desviación alta. La media sola lo oculta.
    return {
        "H": px[:, 0].mean(), "S": px[:, 1].mean(), "V": px[:, 2].mean(),
        "H_sd": px[:, 0].std(), "S_sd": px[:, 1].std(), "V_sd": px[:, 2].std(),
    }


# =============================================================================
# FORMA
# =============================================================================

def calcular_forma(mask):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    c = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimetro = cv2.arcLength(c, True)
    if area == 0 or perimetro == 0:
        return None

    # [FIX-1] RELACIÓN DE ASPECTO ORIENTADA
    # El original usaba cv2.boundingRect(), que devuelve un rectángulo
    # ALINEADO A LOS EJES de la imagen. Una hoja alargada fotografiada en
    # diagonal produce un rectángulo casi cuadrado, y su relación de aspecto
    # cae hacia 1,0 aunque la hoja sea estrecha. El resultado es que la forma
    # medía en parte el ángulo de la foto, no la morfología de la hoja.
    #
    # cv2.minAreaRect() devuelve el rectángulo de MENOR ÁREA, rotado para
    # ajustarse a la hoja. Es invariante a la orientación de la captura.
    (_, _), (w_rot, h_rot), _ = cv2.minAreaRect(c)
    largo = max(w_rot, h_rot)
    ancho = min(w_rot, h_rot)
    aspect_ratio = largo / max(ancho, 1e-6)

    # Se conserva la versión antigua para poder comparar ambas en el informe
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio_ejes = max(w, h) / max(min(w, h), 1)

    hull_area = cv2.contourArea(cv2.convexHull(c))
    solidity = area / hull_area if hull_area > 0 else 0
    circularity = (4 * np.pi * area) / (perimetro ** 2)

    # Descriptor adicional: cuán "ondulado" es el borde del limbo.
    # Un margen foliar sinuoso eleva el perímetro sin elevar el área.
    rugosidad_borde = perimetro / (2 * np.sqrt(np.pi * area)) if area > 0 else 0

    return {
        "aspect_ratio": aspect_ratio,
        "aspect_ratio_ejes": aspect_ratio_ejes,
        "solidity": solidity,
        "circularity": circularity,
        "rugosidad_borde": rugosidad_borde,
        "area_px": area,
    }


# =============================================================================
# TEXTURA
# =============================================================================

def calcular_textura(img_bgr, mask):
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if APLICAR_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gris = clahe.apply(gris)

    x, y, w, h = cv2.boundingRect(mask)
    if w < 8 or h < 8:
        return None

    recorte_gris = gris[y:y + h, x:x + w]
    recorte_mask = mask[y:y + h, x:x + w]

    # [FIX-2] EXCLUIR EL FONDO DE LA GLCM
    # El original ponía el fondo a 0 y calculaba la GLCM sobre todo el
    # rectángulo. Como una hoja es aproximadamente elíptica, cerca del 20 %
    # de ese rectángulo es fondo negro. Esos píxeles generaban una enorme
    # cantidad de pares (0,0) idénticos, que la GLCM interpreta como
    # "superficie perfectamente homogénea". Ese es el motivo por el cual la
    # homogeneidad salía entre 0,94 y 0,97 en TODAS las variedades: no medía
    # la hoja, medía el fondo.
    #
    # Solución estándar: se cuantiza la hoja a los niveles 1..N y se reserva
    # el nivel 0 para el fondo. Después se elimina la fila y la columna 0 de
    # la GLCM y se renormaliza, de modo que solo quedan pares hoja-hoja.
    cuantizado = np.zeros_like(recorte_gris, dtype=np.uint8)
    pix_hoja = recorte_gris[recorte_mask == 1]
    if len(pix_hoja) < 50:
        return None

    lo, hi = np.percentile(pix_hoja, [1, 99])
    if hi <= lo:
        hi = lo + 1
    escalado = np.clip((recorte_gris.astype(np.float32) - lo) / (hi - lo), 0, 1)
    cuantizado = (escalado * (NIVELES_GRIS - 1)).astype(np.uint8) + 1
    cuantizado[recorte_mask == 0] = 0

    glcm = graycomatrix(cuantizado, distances=DISTANCIAS, angles=ANGULOS,
                        levels=NIVELES_GRIS + 1, symmetric=True, normed=False)

    glcm = glcm.astype(np.float64)
    glcm[0, :, :, :] = 0          # elimina pares que involucran fondo
    glcm[:, 0, :, :] = 0
    sumas = glcm.sum(axis=(0, 1), keepdims=True)
    sumas[sumas == 0] = 1
    glcm /= sumas                 # renormaliza solo sobre píxeles de hoja

    # Se promedia sobre distancias y ángulos
    props = {}
    for nombre in ["homogeneity", "contrast", "energy", "correlation", "dissimilarity"]:
        props[nombre] = float(graycoprops(glcm, nombre).mean())

    # [FIX-3] DESCRIPTORES DIRECTOS DE RELIEVE
    # La GLCM es un descriptor indirecto. Para la bulosidad conviene medir
    # el relieve de forma explícita. Estos tres son sensibles a la
    # ondulación de la superficie foliar y complementan al contraste.
    hoja = recorte_gris.astype(np.float32)
    hoja[recorte_mask == 0] = np.nan

    # Varianza del laplaciano: responde a los cambios bruscos de relieve
    lap = cv2.Laplacian(recorte_gris, cv2.CV_32F, ksize=3)
    props["laplaciano_var"] = float(np.nanvar(np.where(recorte_mask == 1, lap, np.nan)))

    # Desviación local: cuánto varía el brillo dentro de vecindades pequeñas
    media_local = cv2.blur(recorte_gris.astype(np.float32), (9, 9))
    desv_local = np.abs(recorte_gris.astype(np.float32) - media_local)
    props["desviacion_local"] = float(np.nanmean(
        np.where(recorte_mask == 1, desv_local, np.nan)))

    # Rango intercuartílico del brillo dentro de la hoja
    props["rango_iq_brillo"] = float(np.percentile(pix_hoja, 75) -
                                     np.percentile(pix_hoja, 25))
    return props


# =============================================================================
# RECORRIDO DEL DATASET
# =============================================================================

def poligono_a_mascara(coords, alto, ancho):
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= ancho
    pts[:, 1] *= alto
    mask = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


filas = []
for split in SPLITS:
    dir_img = os.path.join(DATASET_DIR, split, "images")
    dir_lbl = os.path.join(DATASET_DIR, split, "labels")
    if not os.path.isdir(dir_img):
        print(f"[AVISO] No existe {dir_img}, se omite.")
        continue

    archivos = sorted(glob.glob(os.path.join(dir_img, "*.jpg")) +
                      glob.glob(os.path.join(dir_img, "*.png")))
    print(f"\n[{split}] {len(archivos)} imágenes")

    for n, ruta_img in enumerate(archivos, 1):
        if n % 100 == 0:
            print(f"   {n}/{len(archivos)}")

        base = os.path.splitext(os.path.basename(ruta_img))[0]
        ruta_lbl = os.path.join(dir_lbl, base + ".txt")
        if not os.path.exists(ruta_lbl):
            continue

        img = cv2.imread(ruta_img)
        if img is None:
            continue
        alto, ancho = img.shape[:2]

        with open(ruta_lbl) as f:
            lineas = [l.strip().split() for l in f if l.strip()]

        for i, partes in enumerate(lineas, 1):
            if len(partes) < 7:
                continue
            cid = int(partes[0])
            coords = list(map(float, partes[1:]))
            mask = poligono_a_mascara(coords, alto, ancho)
            if mask.sum() < 100:
                continue

            color = calcular_color(img, mask)
            forma = calcular_forma(mask)
            textura = calcular_textura(img, mask)
            if None in (color, forma, textura):
                continue

            fila = {"Imagen": os.path.basename(ruta_img), "Split": split,
                    "Variedad": CLASES[cid] if cid < len(CLASES) else str(cid),
                    "Instancia": i}
            fila.update(color)
            fila.update(forma)
            fila.update(textura)
            filas.append(fila)

df = pd.DataFrame(filas).round(4)
df.to_csv(CSV_SALIDA, index=False, encoding="utf-8-sig")
print(f"\nHojas procesadas: {len(df)}")
print(f"Guardado en: {CSV_SALIDA}")


# =============================================================================
# COMPARACIÓN: ¿mejoró el poder discriminante?
# =============================================================================

def eta_cuadrado(datos, col):
    grupos = [g[col].dropna().values for _, g in datos.groupby("Variedad")]
    grupos = [g for g in grupos if len(g) > 1]
    if len(grupos) < 2:
        return 0.0
    todos = np.concatenate(grupos)
    mg = todos.mean()
    ss_tot = ((todos - mg) ** 2).sum()
    ss_ent = sum(len(g) * (g.mean() - mg) ** 2 for g in grupos)
    return ss_ent / ss_tot if ss_tot > 0 else 0.0


numericas = [c for c in df.columns if df[c].dtype.kind in "fi" and c != "Instancia"]
ranking = sorted(((c, eta_cuadrado(df, c)) for c in numericas),
                 key=lambda x: x[1], reverse=True)

print("\n" + "=" * 58)
print("PODER DISCRIMINANTE POR DESCRIPTOR (eta cuadrado)")
print("Referencia: con el script original, el máximo era 0,17")
print("=" * 58)
for c, v in ranking:
    marca = "  <-- mejora" if v > 0.17 else ""
    print(f"  {c:<24} {v:.3f}{marca}")
print("=" * 58)

resumen = df.groupby("Variedad")[numericas].mean().round(3).reset_index()
resumen.insert(1, "n_hojas", df.groupby("Variedad").size().values)
resumen.to_csv(CSV_RESUMEN, index=False, encoding="utf-8-sig")
print(f"\nResumen por variedad: {CSV_RESUMEN}")