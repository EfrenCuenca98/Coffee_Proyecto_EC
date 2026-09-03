import os
import random
from collections import Counter
from PIL import Image, ImageEnhance

# =========================================================================
# CONFIGURACIÓN DE RUTAS Y CLASES
# =========================================================================
carpeta_origen  = r"C:\Users\alex_\OneDrive\Escritorio\9no\Tesis\Dataset YOLOv8"
carpeta_destino = r"C:\Users\alex_\OneDrive\Escritorio\9no\Tesis\Dataset_Augmentation_Balanceado"

# Variedades de café en el orden exacto de tus IDs de etiquetas
clases = ['Bourbon', 'Castillo', 'Geisha', 'Mundonovo', 'SanSalvador', 'Sarchimor', 'Typica']

# --- Parámetros de balanceo ---
# Límite de variaciones por imagen para no sobre-repetir las clases muy chicas
# (evita que una sola foto original termine representando el 30-40% de una clase)
MIN_VARIACIONES = 2
MAX_VARIACIONES = 8

# =========================================================================
# FUNCIONES DE TRANSFORMACIÓN QUE PRESERVAN POLÍGONOS
# (idénticas al script original, no se tocaron)
# =========================================================================

def leer_label_poligono(ruta_txt):
    """
    Lee un archivo .txt de segmentación YOLOv8.
    Formato por línea: class_id x1 y1 x2 y2 x3 y3 ... (coordenadas normalizadas 0-1)
    Retorna lista de (class_id, [(x1,y1), (x2,y2), ...])
    """
    anotaciones = []
    if not os.path.exists(ruta_txt):
        return anotaciones
    with open(ruta_txt, 'r') as f:
        for linea in f:
            parts = linea.strip().split()
            if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                puntos = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                anotaciones.append((class_id, puntos))
    return anotaciones

def guardar_label_poligono(ruta_txt, anotaciones):
    """Guarda anotaciones de polígono en formato YOLOv8 segmentación."""
    with open(ruta_txt, 'w') as f:
        for class_id, puntos in anotaciones:
            coords_str = ' '.join(f'{x:.6f} {y:.6f}' for x, y in puntos)
            f.write(f'{class_id} {coords_str}\n')

def clip_puntos(puntos):
    """Asegura que todos los puntos estén en el rango [0, 1]."""
    return [(max(0.0, min(1.0, x)), max(0.0, min(1.0, y))) for x, y in puntos]

def flip_horizontal(img, anotaciones):
    img_out = img.transpose(Image.FLIP_LEFT_RIGHT)
    anot_out = []
    for class_id, puntos in anotaciones:
        nuevos = [(1.0 - x, y) for x, y in puntos]
        anot_out.append((class_id, clip_puntos(nuevos)))
    return img_out, anot_out

def flip_vertical(img, anotaciones):
    img_out = img.transpose(Image.FLIP_TOP_BOTTOM)
    anot_out = []
    for class_id, puntos in anotaciones:
        nuevos = [(x, 1.0 - y) for x, y in puntos]
        anot_out.append((class_id, clip_puntos(nuevos)))
    return img_out, anot_out

def rotar_90(img, anotaciones, veces=1):
    """
    Rota la imagen 90° en sentido antihorario 'veces' veces.
    x_nuevo = y_viejo ; y_nuevo = 1 - x_viejo
    """
    anot_actual = anotaciones
    img_actual  = img
    for _ in range(veces % 4):
        img_actual = img_actual.rotate(90, expand=True)
        anot_nueva = []
        for class_id, puntos in anot_actual:
            nuevos = [(y, 1.0 - x) for x, y in puntos]
            anot_nueva.append((class_id, clip_puntos(nuevos)))
        anot_actual = anot_nueva
    return img_actual, anot_actual

def color_jitter(img, brightness=0.3, contrast=0.2, saturation=0.3):
    """Ajuste de brillo, contraste y saturación (solo afecta píxeles, no coordenadas)."""
    if brightness > 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        img = ImageEnhance.Brightness(img).enhance(max(0.1, factor))
    if contrast > 0:
        factor = 1.0 + random.uniform(-contrast, contrast)
        img = ImageEnhance.Contrast(img).enhance(max(0.1, factor))
    if saturation > 0:
        factor = 1.0 + random.uniform(-saturation, saturation)
        img = ImageEnhance.Color(img).enhance(max(0.1, factor))
    return img

def resize_imagen(img, size=(640, 640)):
    """Resize de imagen. Las coordenadas normalizadas NO cambian con el resize."""
    return img.resize(size, Image.BILINEAR)

def aplicar_pipeline_train(img, anotaciones):
    """Pipeline completo para TRAIN con augmentation (geométrico + color)."""
    img = resize_imagen(img, (640, 640))
    if random.random() < 0.5:
        img, anotaciones = flip_horizontal(img, anotaciones)
    if random.random() < 0.5:
        img, anotaciones = flip_vertical(img, anotaciones)
    veces = random.randint(0, 3)
    if veces > 0:
        img, anotaciones = rotar_90(img, anotaciones, veces)
    img = color_jitter(img, brightness=0.3, contrast=0.2, saturation=0.3)
    return img, anotaciones

def aplicar_pipeline_val_test(img, anotaciones):
    """Pipeline para VALID y TEST: solo resize, SIN augmentation (no se toca)."""
    img = resize_imagen(img, (640, 640))
    return img, anotaciones

# =========================================================================
# PASO 1: CONTAR IMÁGENES POR CLASE EN TRAIN (para calcular el balanceo)
# =========================================================================
def obtener_clase_principal(anotaciones):
    """
    Determina la clase 'dueña' de la imagen. Si por algún motivo hay más de
    un polígono/clase en la misma imagen, se toma la clase mayoritaria.
    """
    if not anotaciones:
        return None
    conteo = Counter(class_id for class_id, _ in anotaciones)
    return conteo.most_common(1)[0][0]

train_img_dir = os.path.join(carpeta_origen, 'train', 'images')
train_lbl_dir = os.path.join(carpeta_origen, 'train', 'labels')

conteo_por_clase = Counter()
clase_de_cada_imagen = {}  # nombre_imagen -> class_id

if os.path.exists(train_img_dir):
    archivos_train = [f for f in os.listdir(train_img_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for nombre_img in archivos_train:
        nombre_base, _ = os.path.splitext(nombre_img)
        ruta_lbl = os.path.join(train_lbl_dir, nombre_base + '.txt')
        anot = leer_label_poligono(ruta_lbl)
        clase_id = obtener_clase_principal(anot)
        clase_de_cada_imagen[nombre_img] = clase_id
        if clase_id is not None:
            conteo_por_clase[clase_id] += 1
else:
    print(f"⚠️  No existe la carpeta: {train_img_dir}")

# =========================================================================
# PASO 2: CALCULAR VARIACIONES POR CLASE (balanceo)
# =========================================================================
# Objetivo: que todas las clases terminen con un número de imágenes
# aumentadas lo más parecido posible al de la clase mayoritaria,
# limitando el multiplicador entre MIN_VARIACIONES y MAX_VARIACIONES
# para no sobre-repetir demasiado las clases con muy pocas fotos.

variaciones_por_clase = {}

if conteo_por_clase:
    max_count = max(conteo_por_clase.values())
    print("\n" + "=" * 60)
    print("CONTEO ORIGINAL DE IMÁGENES POR CLASE (train, antes de aumentar)")
    print("=" * 60)
    for idx, nombre_clase in enumerate(clases):
        count = conteo_por_clase.get(idx, 0)
        if count == 0:
            variaciones = 0
        else:
            variaciones = round(max_count / count)
            variaciones = max(MIN_VARIACIONES, min(MAX_VARIACIONES, variaciones))
        variaciones_por_clase[idx] = variaciones
        total_estimado = count * variaciones
        print(f"  {nombre_clase:<12} | originales: {count:>4} | "
              f"variaciones/imagen: {variaciones:>2} | total aumentado ≈ {total_estimado:>4}")
    print("=" * 60)
else:
    print("⚠️  No se encontraron etiquetas en train, no se puede calcular el balanceo.")

# =========================================================================
# PASO 3: PROCESAMIENTO ESTRUCTURADO DEL DATASET
# =========================================================================
splits = ['train', 'valid', 'test']

for split in splits:
    orig_img_dir = os.path.join(carpeta_origen,  split, 'images')
    orig_lbl_dir = os.path.join(carpeta_origen,  split, 'labels')
    dest_img_dir = os.path.join(carpeta_destino, split, 'images')
    dest_lbl_dir = os.path.join(carpeta_destino, split, 'labels')

    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_lbl_dir, exist_ok=True)

    if not os.path.exists(orig_img_dir):
        print(f"\n⚠️  No existe la carpeta: {orig_img_dir}, saltando...")
        continue

    archivos_img = [f for f in os.listdir(orig_img_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"\n--- Procesando Carpeta: [{split.upper()}] ---")
    print(f"Se encontraron {len(archivos_img)} imágenes originales.")

    is_train  = (split == 'train')
    contador  = 0
    sin_label = 0
    con_label = 0

    for nombre_img in archivos_img:
        nombre_base, ext_img = os.path.splitext(nombre_img)
        ruta_img = os.path.join(orig_img_dir, nombre_img)
        ruta_lbl = os.path.join(orig_lbl_dir, nombre_base + '.txt')

        try:
            with Image.open(ruta_img).convert("RGB") as img:

                anotaciones = leer_label_poligono(ruta_lbl)

                if anotaciones:
                    con_label += 1
                else:
                    sin_label += 1

                if is_train:
                    clase_id = obtener_clase_principal(anotaciones)
                    # Si la imagen no tiene etiqueta reconocida, se usa 1 variación
                    # (no se descarta, pero tampoco se le da peso de balanceo)
                    loops = variaciones_por_clase.get(clase_id, 1) if clase_id is not None else 1
                    loops = max(loops, 1)
                else:
                    loops = 1  # valid y test: SIEMPRE 1, sin augmentation

                for i in range(loops):
                    if is_train:
                        img_trans, anot_trans = aplicar_pipeline_train(img.copy(), anotaciones)
                    else:
                        img_trans, anot_trans = aplicar_pipeline_val_test(img.copy(), anotaciones)

                    nuevo_nombre = f"{nombre_base}_aug_{i}" if is_train else nombre_base

                    ruta_img_out = os.path.join(dest_img_dir, nuevo_nombre + '.jpg')
                    img_trans.save(ruta_img_out, "JPEG", quality=95)

                    ruta_lbl_out = os.path.join(dest_lbl_dir, nuevo_nombre + '.txt')
                    guardar_label_poligono(ruta_lbl_out, anot_trans)

                    contador += 1

        except Exception as e:
            print(f"  ❌ No se pudo procesar {nombre_img}. Error: {e}")

    print(f"Finalizado [{split.upper()}].")
    print(f"  Imágenes con polígonos : {con_label}")
    print(f"  Imágenes sin polígonos : {sin_label}")
    print(f"  Archivos creados       : {contador}")

# =========================================================================
# PASO 4: CONTEO FINAL POR CLASE EN TRAIN (para comprobar el balanceo)
# =========================================================================
dest_train_lbl_dir = os.path.join(carpeta_destino, 'train', 'labels')
conteo_final = Counter()

if os.path.exists(dest_train_lbl_dir):
    for archivo in os.listdir(dest_train_lbl_dir):
        if archivo.endswith('.txt'):
            anot = leer_label_poligono(os.path.join(dest_train_lbl_dir, archivo))
            clase_id = obtener_clase_principal(anot)
            if clase_id is not None:
                conteo_final[clase_id] += 1

    print("\n" + "=" * 60)
    print("CONTEO FINAL DE IMÁGENES POR CLASE (train, después de aumentar)")
    print("=" * 60)
    for idx, nombre_clase in enumerate(clases):
        print(f"  {nombre_clase:<12} | total aumentado: {conteo_final.get(idx, 0)}")
    print("=" * 60)

# =========================================================================
# PASO 5: GENERACIÓN AUTOMÁTICA DEL ARCHIVO DATA.YAML
# =========================================================================
ruta_yaml = os.path.join(carpeta_destino, 'data.yaml')

lineas_clases = "\n".join([f"  {idx}: {nombre}" for idx, nombre in enumerate(clases)])

contenido_yaml = f"""# Archivo de configuracion para YOLOv8 Segmentacion - Tesis (dataset balanceado)
path: /content/drive/MyDrive/Dataset_Augmentation_Balanceado
train: train/images
val: valid/images
test: test/images

nc: {len(clases)}
names:
{lineas_clases}
"""

with open(ruta_yaml, 'w', encoding='utf-8') as f_yaml:
    f_yaml.write(contenido_yaml)

print(f"\n✅ Archivo 'data.yaml' generado en: {ruta_yaml}")
print("\n¡PROCESO DE DATA AUGMENTATION BALANCEADO COMPLETADO!")
print("\nPróximo paso: subir la carpeta 'Dataset_Augmentation_Balanceado' a Google Drive")
print("y en Colab entrena de nuevo con:  yolov8n-seg.pt")