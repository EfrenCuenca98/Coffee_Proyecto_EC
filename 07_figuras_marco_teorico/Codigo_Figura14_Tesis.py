"""
Genera la Figura 14 del Marco Teórico: histograma de un descriptor
morfológico con las curvas gaussianas ajustadas superpuestas, más
la curva de la mezcla total, y las fronteras entre categorías.

A diferencia de las Figuras 11-13, esta NO se genera a partir de una
sola imagen: usa la columna completa de un descriptor calculada
sobre TODAS las hojas del CSV de salida de tu script de extracción
(descriptores_por_hoja_v2.csv).

Requisitos:
    pip install pandas numpy scipy scikit-learn matplotlib

Uso:
    python generar_figura14_gmm.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

# =============================================================================
# PARÁMETROS AJUSTABLES
# =============================================================================

# Ruta al CSV generado por tu script de extracción v2
CSV_DESCRIPTORES = r"C:\Users\alex_\OneDrive\Escritorio\9no\Tesis\Dataset YOLOv8\descriptores_por_hoja_v2.csv"

# Descriptor a modelar. Opciones ya usadas en tu Metodología/Resultados:
#   "H"            -> tonalidad (matiz)
#   "aspect_ratio" -> forma (relación de aspecto)
#   "contrast"     -> textura (contraste)
COLUMNA = "contrast"

# Nombres botánicos de las categorías, en orden de MENOR a MAYOR valor.
# Ajusta según el descriptor elegido (para "contrast" tiene sentido
# Lisa/Media/Rugosa; para "H" podría ser algo como
# "Verde amarillento"/"Verde medio"/"Verde intenso", etc.)
ETIQUETAS = ["Lisa", "Coriácea", "Rugosa"]

N_COMPONENTES = 3

# =============================================================================
# 1. Cargar los datos y ajustar el modelo de mezcla gaussiana
# =============================================================================
df = pd.read_csv(CSV_DESCRIPTORES)

if COLUMNA not in df.columns:
    raise ValueError(f"La columna '{COLUMNA}' no existe en el CSV. "
                      f"Columnas disponibles: {list(df.columns)}")

datos = df[COLUMNA].dropna().values.reshape(-1, 1)
print(f"Hojas usadas para el ajuste: {len(datos)}")

gmm = GaussianMixture(n_components=N_COMPONENTES, random_state=42, n_init=10)
gmm.fit(datos)

medias = gmm.means_.flatten()
varianzas = gmm.covariances_.flatten()
pesos = gmm.weights_.flatten()

# Ordenar los componentes de menor a mayor media, para que el
# nombramiento botánico (Lisa/Media/Rugosa) tenga sentido consistente
orden = np.argsort(medias)
medias, varianzas, pesos = medias[orden], varianzas[orden], pesos[orden]

for k in range(N_COMPONENTES):
    print(f"  {ETIQUETAS[k]:<10} media={medias[k]:.3f}  "
          f"sd={np.sqrt(varianzas[k]):.3f}  peso={pesos[k]:.3f}")

# =============================================================================
# 2. Construir las curvas: cada componente ponderada + la mezcla total
# =============================================================================
x = np.linspace(datos.min(), datos.max(), 1000)

curvas = np.zeros((N_COMPONENTES, len(x)))
for k in range(N_COMPONENTES):
    curvas[k] = pesos[k] * norm.pdf(x, medias[k], np.sqrt(varianzas[k]))

mezcla_total = curvas.sum(axis=0)

# Fronteras entre categorías: puntos donde cambia cuál componente domina
asignacion = np.argmax(curvas, axis=0)
cambios = np.where(np.diff(asignacion) != 0)[0]
fronteras = x[cambios]

# =============================================================================
# 3. Graficar
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(datos, bins=40, density=True, color="lightgray",
        edgecolor="white", alpha=0.85, label="Datos observados")

colores = ["#2ca02c", "#ff7f0e", "#d62728"]  # verde, naranja, rojo
for k in range(N_COMPONENTES):
    ax.plot(x, curvas[k], color=colores[k % len(colores)], lw=2,
            label=f"{ETIQUETAS[k]} (μ={medias[k]:.2f}, w={pesos[k]:.2f})")

ax.plot(x, mezcla_total, color="black", lw=2.5, linestyle="--",
        label="Mezcla total")

for f in fronteras:
    ax.axvline(f, color="gray", linestyle=":", lw=1)

ax.set_xlabel(COLUMNA)
ax.set_ylabel("Densidad")
ax.set_title(f"Ajuste de un modelo de mezcla gaussiana sobre el descriptor '{COLUMNA}'")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("figura14_gmm.png", dpi=300, bbox_inches="tight")
plt.show()

print("Figura guardada como figura14_gmm.png")