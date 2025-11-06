@echo off
setlocal

rem --- Définition des constantes du projet ---
set "ENV_NAME=ragcdl"
set "MODEL_DIR=models"
set "MODEL_NAME=Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
set "MODEL_PATH=%MODEL_DIR%\%MODEL_NAME%"
set "MODEL_URL=https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

echo === Etape 1/4 : Verification des dependances systeme (Windows) ===
echo Ce script utilise Chocolatey pour installer LibreOffice et Tesseract.
where choco >nul 2>nul || (
    echo ATTENTION : Chocolatey n'est pas installe.
    echo Veuillez l'installer depuis https://chocolatey.org/install et relancer ce script.
    pause & exit /b
)
where libreoffice >nul 2>nul || (echo Installation de LibreOffice... & choco install libreoffice -y)
where tesseract >nul 2>nul || (echo Installation de Tesseract-OCR... & choco install tesseract-ocr --params "/Lang=fra" -y)
echo ✓ Dependances systeme verifiees.

echo.
echo === Etape 2/4 : Verification du modele LLM ===
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
if not exist "%MODEL_PATH%" (
    echo Modele LLM non trouve. Telechargement en cours (~4.7 Go)...
    powershell -Command "Invoke-WebRequest -Uri %MODEL_URL% -OutFile %MODEL_PATH%"
    echo ✓ Modele telecharge.
) else (
    echo ✓ Le modele LLM est deja present.
)

echo.
echo === Etape 3/4 : Verification de l'environnement Conda ===
conda env list | findstr /C:"%ENV_NAME%" >nul
if %errorlevel% neq 0 (
    echo Creation de l'environnement Conda '%ENV_NAME%'...
    conda env create -f environment.yml
    echo ✓ Environnement cree.
) else (
    echo ✓ L'environnement Conda '%ENV_NAME%' existe deja.
)

echo.
echo === Etape 4/4 : Lancement du serveur LLM ===
echo Demarrage du serveur dans l'environnement '%ENV_NAME%'...
echo Utilisation du modele : %MODEL_PATH%
echo Pour arreter le serveur, appuyez sur CTRL+C.

rem Utilise 'conda run' pour lancer la commande dans le bon environnement
conda run -n %ENV_NAME% python -m llama_cpp.server --model "%MODEL_PATH%" --host 127.0.0.1 --port 8080 --n_threads 12 --n_ctx 4096

