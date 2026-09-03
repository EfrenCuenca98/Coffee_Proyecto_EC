# Clasificación Automática de variedades de café mediante procesamiento de imágenes y modelos de Machine Learning

Prototipo desarrollado como trabajo de integración curricular en la Carrera de
Telecomunicaciones de la Universidad Nacional de Loja. Clasifica siete
variedades de café (Bourbon, Castillo, Geisha, Mundo Novo, San Salvador,
Sarchimor, Typica) a partir de fotografías de hojas capturadas in situ,
mediante segmentación de instancias (YOLOv8n-seg) y una caracterización
morfológica explícita de color, forma y textura.

**Autor:** [Efren Alexander Cuenca Jiménez]
**Director:** Ing. Marcelo Valdiviezo
**Institución:** Universidad Nacional de Loja — Grupo de Investigación (Proyecto Dataset de Café)

---

## Resumen de resultados

| Métrica (conjunto de prueba) | Valor |
|---|---|
| Precisión | 0,9410 |
| Exhaustividad | 0,9190 |
| mAP50 (máscara) | 0,9668 |
| mAP50-95 (máscara) | 0,9575 |
| Tasa global de acierto varietal | 92,8 % |

## Mapa del pipeline

El desarrollo siguió la metodología CRISP-DM. La siguiente tabla conecta cada
script del repositorio con su función, sus entradas/salidas y la sección
correspondiente del informe de tesis.

| Carpeta / Script | Qué hace | Entrada | Salida | Sección de la tesis |
|---|---|---|---|---|

| `01_captura_y_etiquetado/renombrar_fotos.py` | Renombra las fotografías capturadas en campo al formato estándar `FECHA_ARA_VARIEDAD_###.jpg` | Fotos crudas de la cámara/teléfono | Fotos renombradas | 6.1.3 (Tarea 3) |
| `02_extraccion_descriptores/descriptores_v2.py` | Módulo compartido: funciones de color (HSV), forma (contorno) y textura (GLCM), más el ajuste y clasificación con Modelos de Mezcla Gaussiana | — (es un módulo, se importa) | — | 5.2.4, 6.2.4, 6.2.5 |
| `02_extraccion_descriptores/extraer_morfologia_v2.py` | Recorre el dataset (train/valid/test), calcula los 15 descriptores morfológicos de cada hoja anotada y exporta el CSV | Imágenes + etiquetas `.txt` (polígonos YOLO) | `descriptores_por_hoja_v2.csv`, `resumen_morfologia_por_variedad_v2.csv` | 6.2.4 |
| `03_aumento_datos/aumento_balanceado_pil.py` | Aplica aumento de datos balanceado (rotación, volteo, brillo/contraste/saturación) solo al subconjunto de entrenamiento, e incorpora las imágenes negativas | Dataset particionado (train/valid/test) | Dataset aumentado + `data.yaml` | 6.2.6 |
| `04_entrenamiento/entrenamiento_yolov8_seg.ipynb` | Entrena YOLOv8n-seg mediante transferencia de aprendizaje sobre el dataset aumentado. Contiene únicamente la rutina de entrenamiento (`model.train(...)`) | `data.yaml`, pesos preentrenados COCO | `best_cafe_yolov8_seg.pt` | 6.3.1 – 6.3.3 |
| `05_evaluacion/evaluacion_metricas_matrices.ipynb` | Carga el modelo ya entrenado y ejecuta `model.val(...)` para obtener las métricas y las gráficas de evaluación (matrices de confusión, curvas F1 y precisión-exhaustividad) | `best_cafe_yolov8_seg.pt`, `data.yaml` | Métricas (Tablas 14-15), matrices de confusión (Figuras 23-24) | 6.3.4 – 6.3.6 |
| `06_aplicacion_demo/app.py` | Aplicación Streamlit de demostración: segmenta imágenes nuevas, extrae descriptores, clasifica y aplica el filtro de plausibilidad morfológica (Mahalanobis) | Imágenes nuevas, modelo entrenado, CSV base | Reporte visual + CSV/Excel exportable | 6.3.7, 6.3.8 |
| `06_aplicacion_demo/descriptores_v2.py` | Misma función que en `02_extraccion_descriptores/`, importada aquí porque la app la necesita para calcular los descriptores de cada hoja nueva en tiempo real | — (es un módulo, se importa) | — | 5.2.4 |
| `07_figuras_marco_teorico/` | Scripts independientes para regenerar las figuras del Marco Teórico a partir de una hoja del dataset | Una imagen + su `.txt` | Figuras `.png` del capítulo 4 | 4.5.2, 4.5.4.1, 4.5.4.3, 4.6.1 |

## ¿Solo quieres usar el clasificador, sin entrenar ni reproducir nada?

Ve directo a la carpeta `06_aplicacion_demo/` — ahí está todo lo necesario para
correr la interfaz, sin tocar el resto del repositorio. Esa carpeta tiene su
propio `README.md` con instrucciones más detalladas (incluido un instalador
automático para Windows, `iniciar_interfaz.bat`) y su propio `requirements.txt`,
más pequeño que el de esta raíz porque solo incluye lo que la aplicación
necesita.

Esto abrirá la aplicación en tu navegador. Sube una fotografía de una hoja de
café y obtén su variedad, junto con una descripción de su color, forma y
textura. No necesitas saber nada sobre el resto del pipeline (entrenamiento,
extracción de descriptores, etc.) para usarla — el resto de este README
documenta cómo se construyó el modelo, no cómo se usa la app.

## Si requieres Entrenar y obtener un nuevo modelo con tu porpio Dataset antes de ejecutar configura tus propias rutas

**Ningún script funcionará tal cual si solo lo clonas y lo corres.** Todas las rutas
(carpetas de Google Drive, rutas de Windows) están escritas con la organización de
carpetas del autor original. Antes de correr cualquier archivo, busca el bloque marcado
como `CONFIGURACIÓN` al inicio de cada script o notebook, y cambia esas rutas por las
tuyas. No es necesario que crees carpetas con los mismos nombres — solo que las rutas
apunten a donde tú tengas tus propios archivos.

Checklist de archivos con rutas que debes revisar:

| Archivo | Qué rutas debes cambiar |
|---|---|
| `02_extraccion_descriptores/extraer_morfologia_v2.py` | `DATASET_DIR` |
| `03_aumento_datos/aumento_balanceado_pil.py` | Carpeta de origen y destino del dataset |
| `04_entrenamiento/entrenamiento_yolov8_seg.ipynb` | `RUTA_MODELO`, `RUTA_YAML` |
| `05_evaluacion/evaluacion_metricas_matrices.ipynb` | `BEST_MODEL`, `RUTA_CSV_BASE`, `CARPETA_MODULOS`, `DATA_YAML`, `CARPETA_RESULTADOS` |
| `06_aplicacion_demo/app.py` | Ninguna — ya resuelve las rutas automáticamente respecto a su propia ubicación. Solo asegúrate de que existan las carpetas `modelo/` y `datos/` junto a `app.py`, con esos nombres exactos |
| `figuras_marco_teorico/*.py` | `RUTA_IMAGEN`, `RUTA_ETIQUETA` en cada script |

## Cómo reproducir el pipeline completo

```bash
# 1. Clonar y preparar el entorno
git clone https://github.com/EfrenCuenca98/Coffee_Proyecto_EC.git
cd tesis-clasificacion-cafe
pip install -r requirements.txt

# 2. Etiquetado (si partes de fotos crudas)
python 01_captura_y_etiquetado/renombrar_fotos.py

# 3. Extracción de descriptores morfológicos (requiere dataset ya anotado en Roboflow)
python 02_extraccion_descriptores/extraer_morfologia_v2.py

# 4. Aumento de datos balanceado
python 03_aumento_datos/aumento_balanceado_pil.py

# 5. Entrenamiento -> abrir 04_entrenamiento/entrenamiento_yolov8_seg.ipynb en Google Colab

# 6. Evaluación -> abrir 05_evaluacion/evaluacion_metricas_matrices.ipynb en Google Colab

# 7. Aplicación de demostración
streamlit run 06_aplicacion_demo/app.py
```

## Dataset y pesos del modelo

El conjunto de datos de imágenes (2 310 fotografías)  **no se incluyen en este repositorio** por su
tamaño. Están disponibles en:

- Dataset: [https://drive.google.com/drive/folders/1hgqTZRoYqFYhpWWHC8FN7zyP-37acvzR?usp=sharing]

Dataset fue transferido como activo intangible al Grupo de Investigación..
..(Proyecto Dataset de Café) de la Universidad Nacional de Loja.

## Requisitos

Ver `requirements.txt`. Resumen de las bibliotecas principales:

- `ultralytics` (YOLOv8)
- `opencv-python`, `scikit-image` (extracción de descriptores)
- `scikit-learn` (modelos de mezcla gaussiana)
- `streamlit` (aplicación de demostración)
- `pandas`, `numpy`, `matplotlib`

## Citar este trabajo

[Efrén Alexander Cuenca Jiménez]. (2026). *Clasificación Automática de variedades de 
café mediante procesamiento de imágenes y modelos de Machine Learning* [Trabajo de integración
curricular, Universidad Nacional de Loja].
