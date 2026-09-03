"""
Genera la Figura 11 del Marco Teórico: la misma hoja representada
en distintos espacios de color (RGB, HSV, CIELAB y escala de grises).

Requisitos:
    pip install opencv-python matplotlib

Uso:
    python generar_figura11_espacios_color.py
"""

import cv2
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Carga tu propia imagen (una hoja de tu conjunto de datos)
# ---------------------------------------------------------------
RUTA_IMAGEN = "C:/Users/alex_/OneDrive/Escritorio/9no/Tesis/Dataset YOLOv8/test/images/20260728_ARA_BOU_HOJA_052_jpg.rf.fe95085a09f5158c0dd5448b703315c0.jpg"   # <-- cambia esto por tu archivo real

img_bgr = cv2.imread(RUTA_IMAGEN)
if img_bgr is None:
    raise FileNotFoundError(f"No se pudo abrir la imagen: {RUTA_IMAGEN}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ---------------------------------------------------------------
# 2. Conversión a los distintos espacios de color
# ---------------------------------------------------------------
img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Si prefieres reproducir el estilo de tu imagen de ejemplo
# (RGB / HSV / Lab / YCbCr) en vez de escala de grises, descomenta:
# img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

# ---------------------------------------------------------------
# 3. Montaje en cuadrícula 2x2 con matplotlib
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

paneles = [
    (img_rgb,  "RGB",              None),
    (img_hsv,  "HSV",              None),
    (img_lab,  "CIELAB",           None),
    (img_gray, "Escala de grises", "gray"),
]

for ax, (imagen, titulo, cmap) in zip(axes.flat, paneles):
    ax.imshow(imagen, cmap=cmap)
    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.savefig("figura11_espacios_color.png", dpi=300, bbox_inches="tight")
plt.show()

print("Figura guardada como figura11_espacios_color.png")