"""
Genera la Figura 12 del Marco Teórico: rectángulo de área mínima,
envolvente convexa y contorno real, superpuestos sobre una hoja
segmentada, con los valores de relación de aspecto, solidez y
circularidad calculados EXACTAMENTE con la misma función que usa
el script de extracción (descriptores_por_hoja_v2.csv).

Requisitos:
    pip install opencv-python matplotlib numpy

Uso:
    python generar_figura12_descriptores_forma.py
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# =============================================================================
# Funciones IDÉNTICAS a las de tu script de extracción v2
# (copiadas literalmente para garantizar que los valores de esta figura
#  coincidan exactamente con los de descriptores_por_hoja_v2.csv)
# =============================================================================

def poligono_a_mascara(coords, alto, ancho):
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= ancho
    pts[:, 1] *= alto
    mask = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def calcular_forma(mask):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None, None
    c = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimetro = cv2.arcLength(c, True)
    if area == 0 or perimetro == 0:
        return None, None

    # Relación de aspecto orientada (rectángulo de área mínima, [FIX-1])
    (_, _), (w_rot, h_rot), _ = cv2.minAreaRect(c)
    largo = max(w_rot, h_rot)
    ancho_r = min(w_rot, h_rot)
    aspect_ratio = largo / max(ancho_r, 1e-6)

    hull_area = cv2.contourArea(cv2.convexHull(c))
    solidity = area / hull_area if hull_area > 0 else 0
    circularity = (4 * np.pi * area) / (perimetro ** 2)

    descriptores = {
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "circularity": circularity,
    }
    return c, descriptores


# =============================================================================
# 1. Rutas — usa una imagen de 'valid' o 'test' (NO 'train', evita el aumento)
# =============================================================================
RUTA_IMAGEN   = r"C:\Users\alex_\OneDrive\Escritorio\9no\Tesis\Dataset YOLOv8\test\images\20260728_ARA_BOU_HOJA_052_jpg.rf.fe95085a09f5158c0dd5448b703315c0.jpg"
RUTA_ETIQUETA = r"C:\Users\alex_\OneDrive\Escritorio\9no\Tesis\Dataset YOLOv8\test\labels\20260728_ARA_BOU_HOJA_052_jpg.rf.fe95085a09f5158c0dd5448b703315c0.txt"
INDICE_INSTANCIA = 0  # si hay varias hojas anotadas en la imagen, elige la línea

# =============================================================================
# 2. Cargar imagen y polígono, generar la máscara (misma función que el v2)
# =============================================================================
img_bgr = cv2.imread(RUTA_IMAGEN)
if img_bgr is None:
    raise FileNotFoundError(f"No se pudo abrir la imagen: {RUTA_IMAGEN}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
alto, ancho = img_bgr.shape[:2]

with open(RUTA_ETIQUETA, "r") as f:
    lineas = [l.strip().split() for l in f if l.strip()]

partes = lineas[INDICE_INSTANCIA]
coords = list(map(float, partes[1:]))
mask = poligono_a_mascara(coords, alto, ancho)

# =============================================================================
# 3. Calcular descriptores con la MISMA función que tu extracción real
# =============================================================================
contorno, descriptores = calcular_forma(mask)
if contorno is None:
    raise ValueError("No se pudo extraer un contorno válido de esta hoja.")

print(f"Relación de aspecto: {descriptores['aspect_ratio']:.3f}")
print(f"Solidez:             {descriptores['solidity']:.3f}")
print(f"Circularidad:        {descriptores['circularity']:.3f}")

# =============================================================================
# 4. Geometrías auxiliares, solo para el dibujo (mismos parámetros que arriba)
# =============================================================================
rect = cv2.minAreaRect(contorno)
box = np.intp(cv2.boxPoints(rect))
hull = cv2.convexHull(contorno)

# =============================================================================
# 5. Dibujar los tres trazados sobre la imagen
# =============================================================================
img_dibujo = img_rgb.copy()
cv2.drawContours(img_dibujo, [contorno], -1, (255, 0, 0), 3)   # contorno real: rojo
cv2.drawContours(img_dibujo, [hull], -1, (0, 255, 0), 3)       # envolvente convexa: verde
cv2.drawContours(img_dibujo, [box], -1, (0, 0, 255), 3)        # rectángulo de área mínima: azul

# =============================================================================
# 6. Mostrar con leyenda y recuadro de valores
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img_dibujo)
ax.axis("off")

leyenda_elementos = [
    plt.Line2D([0], [0], color="red",   lw=3, label="Contorno real"),
    plt.Line2D([0], [0], color="green", lw=3, label="Envolvente convexa"),
    plt.Line2D([0], [0], color="blue",  lw=3, label="Rectángulo de área mínima"),
]
ax.legend(handles=leyenda_elementos, loc="upper right", fontsize=9, framealpha=0.9)

texto_valores = (
    f"Relación de aspecto: {descriptores['aspect_ratio']:.2f}\n"
    f"Solidez: {descriptores['solidity']:.3f}\n"
    f"Circularidad: {descriptores['circularity']:.3f}"
)
ax.text(
    0.02, 0.02, texto_valores, transform=ax.transAxes,
    fontsize=10, va="bottom", ha="left", color="white",
    bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.4")
)

plt.tight_layout()
plt.savefig("figura12_descriptores_forma.png", dpi=300, bbox_inches="tight")
plt.show()

print("Figura guardada como figura12_descriptores_forma.png")