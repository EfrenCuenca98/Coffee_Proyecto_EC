# CafeLeaf — Clasificador foliar de variedades de café

Aplicación de escritorio que identifica la variedad de una hoja de café a
partir de una fotografía. Reconoce siete variedades de *Coffea arabica*:
Bourbon, Castillo, Geisha, Mundo Novo, San Salvador, Sarchimor y Typica.

No requiere conexión a internet una vez instalada, ni conocimientos de
programación para usarla — solo seguir los pasos de instalación una sola vez.

---

## ¿Qué hace esta aplicación?

1. Recibe una o varias fotografías de hojas de café.
2. Localiza y recorta automáticamente cada hoja en la imagen (segmentación).
3. Identifica a qué variedad pertenece cada hoja, con un porcentaje de confianza.
4. Describe cada hoja en términos de color, forma y textura, con categorías
   comprensibles ("Verde medio", "Elíptica", "Lisa", etc.).
5. Avisa cuando lo que detectó no se parece morfológicamente a ninguna hoja
   de café conocida (por ejemplo, si se fotografía por error una hoja de
   otra planta).
6. Permite exportar todos los resultados a Excel o CSV.

---

## Contenido de esta carpeta

```
06_aplicacion_demo/
├── README.md                  ← este archivo
├── iniciar_interfaz.bat       ← instalador y lanzador (Windows)
├── requirements.txt           ← lista de librerías necesarias
├── app.py                     ← la aplicación
├── descriptores_v2.py         ← módulo de cálculo (no se ejecuta solo)
├── modelo/
│   └── best_cafe_yolov8_seg.pt
└── datos/
    └── descriptores_por_hoja_v2.csv
```

No muevas ni renombres ningún archivo o carpeta — `app.py` los busca por esos
nombres exactos.

---

## Instalación

### Opción A — Automática, para Windows (recomendada)

1. Haz doble clic en **`iniciar_interfaz.bat`**.
2. Espera a que termine — la primera vez tarda varios minutos porque
   descarga las librerías necesarias. La aplicación se abrirá sola en tu
   navegador al finalizar.

Las próximas veces que quieras usar la aplicación, **vuelve a hacer doble
clic en `iniciar_interfaz.bat`**: como ya tendrás las librerías instaladas
en la carpeta `.venv`, el proceso será casi instantáneo.

Las próximas veces que quieras usar la aplicación, **vuelve a hacer doble
clic en `iniciar_interfaz.bat`**: como ya tendrás las librerías instaladas
en la carpeta `.venv`, el proceso es inmedianto.

### Opción B — Manual, para Windows, Mac o Linux

Si prefieres no usar el instalador automático, o estás en Mac/Linux, abre
una terminal dentro de esta carpeta y ejecuta:

```
pip install -r requirements.txt
streamlit run app.py
```

---

## Cómo usar la aplicación

Se abrirá automáticamente una pestaña en tu navegador. Si no se abre sola,
copia la dirección que aparece en la ventana negra de instalación (algo como
`http://localhost:8501`) y pégala en tu navegador.

Para cerrar la aplicación, vuelve a esa ventana negra y presiona `Ctrl + C`
(o simplemente ciérrala).

### Analizar fotografías

1. En la pestaña **"Analizar"**, sube una o varias fotos de hojas (o indica
   la ruta de una carpeta del equipo con el modo "Carpeta del equipo").
2. Pulsa **"Segmentar"**.
3. Ve a la pestaña **"Resultados"** para ver las hojas detectadas, su
   variedad, y sus características de color, forma y textura.
4. Descarga la tabla completa en CSV o Excel desde el botón correspondiente.

### Ajustes disponibles (barra lateral izquierda)

- **Confianza mínima:** qué tan segura debe estar la aplicación para aceptar
  una detección. Si el modelo no encuentra hojas que sí están en la foto,
  baja este valor. Si detecta cosas donde no hay hojas, súbelo.
- **Avisar ante detecciones atípicas:** mantenlo activado — es lo que hace
  que la aplicación te advierta cuando fotografiaste algo que no se parece
  a una hoja de café real.

---

## Recomendaciones para obtener buenos resultados

- Encuadra la hoja de modo que ocupe buena parte de la fotografía.
- Usa luz natural, evitando sombras muy marcadas sobre la hoja.
- Prefiere un fondo despejado; el follaje superpuesto es la causa más común
  de que la aplicación no detecte la hoja correctamente.

---

## Qué NO hace esta aplicación

- No reconoce variedades distintas a las siete con las que fue entrenada.
  Una hoja de otra variedad o de otra especie será clasificada de todas
  formas dentro de esas siete, normalmente con una confianza baja o marcada
  como "detección atípica".
---

## Solución de problemas

| Problema | Posible causa |
|---|---|
| El `.bat` se cierra de inmediato al abrirlo | Python no está instalado, o no se agregó al PATH del sistema durante su instalación. Instálalo desde [python.org/downloads](https://www.python.org/downloads/) marcando "Add Python to PATH". |
| "No se encontró el modelo en..." | Falta el archivo `.pt` dentro de la carpeta `modelo/`, o tiene otro nombre. |
| La aplicación no abre el navegador sola | Copia manualmente la dirección `http://localhost:8501` que aparece en la ventana de instalación y pégala en tu navegador. |

---

## Soporte

Ante cualquier duda o comportamiento inesperado, contactar a:

**[Efren Cuenca]** — [efren.cuenca@unl.edu.ec]
Desarrollado para el Grupo de Investigación (Proyecto Dataset de Café),
Universidad Nacional de Loja.
