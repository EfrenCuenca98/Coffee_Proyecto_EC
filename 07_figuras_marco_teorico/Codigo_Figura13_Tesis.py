"""
Genera la Figura 13 del Marco Teórico: cálculo de la matriz de
co-ocurrencia de niveles de gris (GLCM) sobre la lámina foliar
enmascarada, usando los MISMOS parámetros que el script de
extracción v2 (descriptores_por_hoja_v2.csv), más un panel
didáctico con un ejemplo pequeño de 4x4 píxeles.

Requisitos:
    pip install opencv-python matplotlib numpy scikit-image

Uso:
    python generar_figura13_glcm.py
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# =============================================================================
# PARÁMETROS — idénticos a los de tu script de extracción v2
# =============================================================================
NIVELES_GRIS = 32
DISTANCIAS = [1, 3, 5, 10]
ANGULOS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
APLICAR_CLAHE = True

# =============================================================================
# Funciones IDÉNTICAS a las de tu script de extracción v2
# =============================================================================

def poligono_a_mascara(coords, alto, ancho):
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= ancho
    pts[:, 1] *= alto
    mask = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def calcular_textura_y_glcm(img_bgr, mask):
    """
    Misma lógica que calcular_textura() de tu script v2, pero además
    devuelve el recorte en escala de grises, la máscara recortada y la
    GLCM promedio (sin la fila/columna de fondo) para poder graficarla.
    """
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if APLICAR_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gris = clahe.apply(gris)

    x, y, w, h = cv2.boundingRect(mask)
    if w < 8 or h < 8:
        return None

    recorte_gris = gris[y:y + h, x:x + w]
    recorte_mask = mask[y:y + h, x:x + w]

    pix_hoja = recorte_gris[recorte_mask == 1]
    if len(pix_hoja) < 50:
        return None

    # [FIX-2] Excluir el fondo: se reserva el nivel 0 para el fondo
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

    props = {}
    for nombre in ["homogeneity", "contrast", "energy"]:
        props[nombre] = float(graycoprops(glcm, nombre).mean())

    # Promedio sobre distancias y ángulos, quitando la fila/columna 0 (fondo)
    # solo para visualizar la matriz de forma legible
    glcm_visual = glcm[1:, 1:, :, :].mean(axis=(2, 3))

    imagen_enmascarada = recorte_gris.copy()
    imagen_enmascarada[recorte_mask == 0] = 0

    return imagen_enmascarada, glcm_visual, props


# =============================================================================
# 1. Ruta — usa la MISMA imagen de las Figuras 11 y 12 (valid o test)
# =============================================================================
RUTA_IMAGEN   = r"C:\Users\alex_\OneDrive\Escritorio\9no\Tesis\Dataset YOLOv8\test\images\20260728_ARA_BOU_HOJA_052_jpg.rf.fe95085a09f5158c0dd5448b703315c0.jpg"
RUTA_ETIQUETA = r"C:\Users\alex_\OneDrive\Escritorio\9no\Tesis\Dataset YOLOv8\test\labels\20260728_ARA_BOU_HOJA_052_jpg.rf.fe95085a09f5158c0dd5448b703315c0.txt"
INDICE_INSTANCIA = 0

# =============================================================================
# 2. Cargar imagen, generar máscara y calcular la GLCM real
# =============================================================================
img_bgr = cv2.imread(RUTA_IMAGEN)
if img_bgr is None:
    raise FileNotFoundError(f"No se pudo abrir la imagen: {RUTA_IMAGEN}")
alto, ancho = img_bgr.shape[:2]

with open(RUTA_ETIQUETA, "r") as f:
    lineas = [l.strip().split() for l in f if l.strip()]

partes = lineas[INDICE_INSTANCIA]
coords = list(map(float, partes[1:]))
mask = poligono_a_mascara(coords, alto, ancho)

resultado = calcular_textura_y_glcm(img_bgr, mask)
if resultado is None:
    raise ValueError("No se pudo calcular la GLCM para esta hoja (máscara demasiado pequeña).")

imagen_enmascarada, glcm_visual, props = resultado

print(f"Homogeneidad: {props['homogeneity']:.3f}")
print(f"Contraste:    {props['contrast']:.3f}")
print(f"Energía:      {props['energy']:.3f}")

# =============================================================================
# 3. Ejemplo didáctico: matriz 4x4 y su GLCM "contada a mano"
#    (distancia 1, orientación horizontal 0°, 4 niveles de gris)
# =============================================================================
ejemplo_img = np.array([
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [0, 2, 2, 2],
    [2, 2, 3, 3],
], dtype=np.uint8)

glcm_ejemplo = graycomatrix(ejemplo_img, distances=[1], angles=[0],
                            levels=4, symmetric=True, normed=False)[:, :, 0, 0]

# =============================================================================
# 4. Figura con cuatro paneles: (a) hoja, (b) GLCM real, (c) ejemplo, (d) GLCM ejemplo
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(9, 9))

# (a) Hoja enmascarada en escala de grises
ax = axes[0, 0]
ax.imshow(imagen_enmascarada, cmap="gray")
ax.set_title("(a) Hoja enmascarada\n(escala de grises, fondo = 0)", fontsize=10)
ax.axis("off")

# (b) Heatmap de la GLCM real (promedio de distancias y ángulos)
ax = axes[0, 1]
im = ax.imshow(glcm_visual, cmap="viridis")
ax.set_title("(b) Matriz de co-ocurrencia\n(GLCM, hoja real)", fontsize=10)
ax.set_xlabel("Nivel de gris j")
ax.set_ylabel("Nivel de gris i")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
texto = f"Homogeneidad: {props['homogeneity']:.3f}\nContraste: {props['contrast']:.2f}\nEnergía: {props['energy']:.3f}"
ax.text(0.02, -0.18, texto, transform=ax.transAxes, fontsize=9, va="top")

# (c) Ejemplo didáctico: imagen 4x4 con los valores anotados
ax = axes[1, 0]
ax.imshow(ejemplo_img, cmap="gray", vmin=0, vmax=3)
for i in range(4):
    for j in range(4):
        ax.text(j, i, str(ejemplo_img[i, j]), ha="center", va="center",
                color="red", fontsize=14, fontweight="bold")
ax.set_title("(c) Ejemplo ilustrativo 4x4\n(niveles de gris 0-3)", fontsize=10)
ax.set_xticks(range(4)); ax.set_yticks(range(4))

# (d) GLCM del ejemplo, contada a mano (distancia 1, 0 grados, simétrica)
ax = axes[1, 1]
im2 = ax.imshow(glcm_ejemplo, cmap="viridis")
for i in range(4):
    for j in range(4):
        ax.text(j, i, int(glcm_ejemplo[i, j]), ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
ax.set_title("(d) GLCM del ejemplo\n(distancia=1, 0°, simétrica)", fontsize=10)
ax.set_xlabel("Nivel de gris j")
ax.set_ylabel("Nivel de gris i")
fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("figura13_glcm.png", dpi=300, bbox_inches="tight")
plt.show()

print("Figura guardada como figura13_glcm.png")