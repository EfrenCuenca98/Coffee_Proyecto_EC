@echo off
cls
echo ==============================================================
echo   SISTEMA DE EVALUACION - ALGORITMO PARA HOJAS DE CAFE
echo ==============================================================
echo.

:: Verifica si el entorno virtual no existe para crearlo
if not exist ".venv\" (
    echo Creando entorno virtual por primera vez, esto tomara un momento...
    python -m venv .venv
    echo.
)

echo Activando entorno...
call .venv\Scripts\activate.bat

echo.
echo Instalando las librerias necesarias...
echo La primera vez puede tardar varios minutos, las siguientes veces sera casi instantaneo.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ==============================================================
echo Abriendo la interfaz en tu navegador...
echo ==============================================================
streamlit run app.py
pause
