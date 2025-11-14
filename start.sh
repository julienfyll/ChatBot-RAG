#!/bin/bash
# Script de configuration ET de lancement pour le projet RAG (Linux/macOS)

# Arrête le script si une commande échoue
set -e

# --- Définition des constantes du projet ---
ENV_NAME="ragcdl"
MODEL_DIR="models"
MODEL_NAME="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
MODEL_URL="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

echo "=== Étape 1/4 : Vérification des dépendances système ==="
OS="$(uname -s)"
case "$OS" in
    Linux*)
        if [ -f /etc/debian_version ]; then
            PACKAGES="libreoffice tesseract-ocr tesseract-ocr-fra wget poppler-utils"
            for pkg in $PACKAGES; do
                if ! dpkg -l | grep -q "ii  $pkg "; then
                    echo "Installation de $pkg..."
                    sudo apt-get update && sudo apt-get install -y $pkg
                else
                    echo "✓ $pkg est déjà installé."
                fi
            done
        else
            echo "AVERTISSEMENT : Ce script ne gère que Debian/Ubuntu. Veuillez installer manuellement les dépendances."
        fi
        ;;
    Darwin*)
        echo "macOS détecté. Vérification avec Homebrew..."
        if ! command -v brew &> /dev/null; then echo "Homebrew n'est pas installé. Veuillez l'installer." >&2; exit 1; fi
        PACKAGES="libreoffice tesseract tesseract-lang wget"
        for pkg in $PACKAGES; do
            if ! brew list --formula | grep -q "^$pkg$"; then
                echo "Installation de $pkg..." && brew install $pkg
            else
                echo "✓ $pkg est déjà installé."
            fi
        done
        ;;
    *)
        echo "Système d'exploitation non supporté : $OS. Pour Windows, utilisez start.bat." >&2; exit 1;;
esac

echo -e "\n=== Étape 2/4 : Vérification du modèle LLM ==="
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Modèle LLM non trouvé. Téléchargement en cours (~4.7 Go)..."
    wget -O "$MODEL_PATH" "$MODEL_URL"
    echo "✓ Modèle téléchargé."
else
    echo "✓ Le modèle LLM est déjà présent."
fi

echo -e "\n=== Étape 3/4 : Vérification de l'environnement Conda ==="
# Vérifie si l'environnement existe. Si non, le crée.
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Création de l'environnement Conda '$ENV_NAME'..."
    conda env create -f environment.yml
    echo "✓ Environnement créé."
else
    echo "✓ L'environnement Conda '$ENV_NAME' existe déjà."
fi

echo -e "\n=== Étape 4/4 : Lancement du serveur LLM ==="
echo "Démarrage du serveur dans l'environnement '$ENV_NAME'..."
echo "Utilisation du modèle : $MODEL_PATH"
echo "Pour arrêter le serveur, appuyez sur CTRL+C."

# Utilise 'conda run' pour exécuter la commande dans le bon environnement
# C'est plus propre que 'source' et 'activate' dans un script.
# Les chemins sont maintenant relatifs au projet, pas au dossier personnel.
conda run -n "$ENV_NAME" python -m llama_cpp.server \
  --model "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8080 \
  --n_threads 12 \
  --n_ctx 4096

