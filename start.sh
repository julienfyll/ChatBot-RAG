# #!/bin/bash
# # Script de configuration ET de lancement pour le projet RAG (Linux/macOS)

# # Arrête le script si une commande échoue
# set -e

# # --- Définition des constantes du projet ---
# ENV_NAME="ragcdl"
# MODEL_DIR="models"
# MODEL_NAME="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
# MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
# MODEL_URL="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
# BINARY_PATH="/home/fayolle/workspace/llama.cpp/build/bin/llama-server"

# echo "=== Étape 1/4 : Vérification des dépendances système ==="
# OS="$(uname -s)"
# case "$OS" in
#     Linux*)
#         if [ -f /etc/debian_version ]; then
#             PACKAGES="libreoffice tesseract-ocr tesseract-ocr-fra wget poppler-utils"
#             for pkg in $PACKAGES; do
#                 if ! dpkg -l | grep -q "ii  $pkg "; then
#                     echo "Installation de $pkg..."
#                     sudo apt-get update && sudo apt-get install -y $pkg
#                 else
#                     echo "✓ $pkg est déjà installé."
#                 fi
#             done
#         else
#             echo "AVERTISSEMENT : Ce script ne gère que Debian/Ubuntu. Veuillez installer manuellement les dépendances."
#         fi
#         ;;
#     Darwin*)
#         echo "macOS détecté. Vérification avec Homebrew..."
#         if ! command -v brew &> /dev/null; then echo "Homebrew n'est pas installé. Veuillez l'installer." >&2; exit 1; fi
#         PACKAGES="libreoffice tesseract tesseract-lang wget"
#         for pkg in $PACKAGES; do
#             if ! brew list --formula | grep -q "^$pkg$"; then
#                 echo "Installation de $pkg..." && brew install $pkg
#             else
#                 echo "✓ $pkg est déjà installé."
#             fi
#         done
#         ;;
#     *)
#         echo "Système d'exploitation non supporté : $OS. Pour Windows, utilisez start.bat." >&2; exit 1;;
# esac

# echo -e "\n=== Étape 2/4 : Vérification du modèle LLM ==="
# mkdir -p "$MODEL_DIR"
# if [ ! -f "$MODEL_PATH" ]; then
#     echo "Modèle LLM non trouvé. Téléchargement en cours (~4.7 Go)..."
#     wget -O "$MODEL_PATH" "$MODEL_URL"
#     echo "✓ Modèle téléchargé."
# else
#     echo "✓ Le modèle LLM est déjà présent."
# fi

# echo -e "\n=== Étape 3/4 : Vérification de l'environnement Conda ==="
# # Vérifie si l'environnement existe. Si non, le crée.
# if ! conda env list | grep -q "$ENV_NAME"; then
#     echo "Création de l'environnement Conda '$ENV_NAME'..."
#     conda env create -f environment.yml
#     echo "✓ Environnement créé."
# else
#     echo "✓ L'environnement Conda '$ENV_NAME' existe déjà."
# fi

# echo -e "\n=== Étape 4/4 : Lancement du serveur LLM ==="
# echo "Utilisation du modèle : $MODEL_PATH"
# echo "Pour arrêter le serveur, appuyez sur CTRL+C."

# echo "=================================================="
# echo " Lancement du serveur Llama.cpp"
# echo "=================================================="
# echo "Modèle : $MODEL_PATH"
# echo "URL    : http://127.0.0.1:8080"
# echo "Note   : Gardez ce terminal ouvert !"
# echo "=================================================="

# # Explication des flags :
# # -m : chemin du modèle
# # -c : taille du contexte (8192 est bien pour du RAG, augmente si nécessaire)
# # -ngl : nombre de couches sur le GPU (99 = max)
# # --host : adresse d'écoute
# # --port : port d'écoute
# # --n-predict : nombre max de tokens en réponse (-1 = infini/limite contexte)

# $BINARY_PATH \
#     -m "$MODEL_PATH" \
#     -ngl 99 \
#     --host 127.0.0.1 \
#     --port 8080 \
#     --n-predict -1 \
#     --ctx-size 25000 \

#!/bin/bash
# Script de configuration ET de lancement pour le projet RAG (Linux/macOS)

# Arrête le script si une commande échoue
set -e

# --- 1. Chargement de la configuration depuis .env ---
if [ -f .env ]; then
    # 'set -a' exporte automatiquement les variables vers les sous-processus
    set -a
    source .env
    set +a
    echo "✅ Configuration chargée depuis .env"
else
    echo "❌ ERREUR : Fichier .env introuvable à la racine."
    echo "   Veuillez copier .env_example vers .env et ajuster les chemins."
    exit 1
fi

# --- Définition des constantes complémentaires ---
ENV_NAME="ragcdl"
# URL de téléchargement (On la garde ici car elle ne change pas souvent, 
# ou tu peux l'ajouter au .env si tu veux changer de modèle souvent)
MODEL_URL="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"


echo "=== Étape 1/4 : Vérification des dépendances système ==="
OS="$(uname -s)"
case "$OS" in
    Linux*)
        if [ -f /etc/debian_version ]; then
            # Ajout de 'curl' qui est souvent utile
            PACKAGES="libreoffice tesseract-ocr tesseract-ocr-fra wget poppler-utils curl"
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
        PACKAGES="libreoffice tesseract tesseract-lang wget curl"
        for pkg in $PACKAGES; do
            if ! brew list --formula | grep -q "^$pkg$"; then
                echo "Installation de $pkg..." && brew install $pkg
            else
                echo "✓ $pkg est déjà installé."
            fi
        done
        ;;
    *)
        echo "Système d'exploitation non supporté : $OS." >&2; exit 1;;
esac

echo -e "\n=== Étape 2/4 : Vérification du modèle LLM ==="
# On utilise le chemin défini dans le .env
MODEL_DIR=$(dirname "$LLM_MODEL_PATH")
mkdir -p "$MODEL_DIR"

if [ ! -f "$LLM_MODEL_PATH" ]; then
    echo "Modèle LLM non trouvé à : $LLM_MODEL_PATH"
    echo "Téléchargement en cours (~4.7 Go)..."
    wget -O "$LLM_MODEL_PATH" "$MODEL_URL"
    echo "✓ Modèle téléchargé."
else
    echo "✓ Le modèle LLM est déjà présent : $LLM_MODEL_PATH"
fi

echo -e "\n=== Étape 3/4 : Vérification de l'environnement Conda ==="
# On vérifie si conda est disponible avant de lancer la commande
if command -v conda &> /dev/null; then
    if ! conda env list | grep -q "$ENV_NAME"; then
        echo "Création de l'environnement Conda '$ENV_NAME'..."
        conda env create -f environment.yml
        echo "✓ Environnement créé."
    else
        echo "✓ L'environnement Conda '$ENV_NAME' existe déjà."
    fi
else
    echo "⚠️  Conda n'est pas détecté, on suppose que vous gérez votre venv manuellement."
fi

echo -e "\n=== Étape 4/4 : Lancement du serveur LLM ==="


# On récupère le dossier où se trouve l'exécutable llama-server
BIN_DIR=$(dirname "$LLM_BINARY_PATH")
# On ajoute ce dossier à la liste des endroits où Linux cherche les librairies (.so)
export LD_LIBRARY_PATH="$BIN_DIR:$LD_LIBRARY_PATH"



echo "=================================================="
echo "🚀 Lancement du serveur Llama.cpp"
echo "=================================================="
echo "Exécutable : $LLM_BINARY_PATH"
echo "Modèle     : $LLM_MODEL_PATH"
echo "Adresse    : http://$LLM_HOST:$LLM_PORT"
echo "Contexte   : $LLM_CONTEXT_SIZE tokens"
echo "Note       : Gardez ce terminal ouvert !"
echo "=================================================="

# Vérification finale de l'exécutable
if [ ! -f "$LLM_BINARY_PATH" ]; then
    echo "❌ ERREUR : L'exécutable llama-server est introuvable à l'adresse :"
    echo "   $LLM_BINARY_PATH"
    echo "   Vérifiez la variable LLM_BINARY_PATH dans votre fichier .env"
    exit 1
fi

# Lancement avec les variables du .env
"$LLM_BINARY_PATH" \
    -m "$LLM_MODEL_PATH" \
    -ngl "$LLM_GPU_LAYERS" \
    --host "$LLM_HOST" \
    --port "$LLM_PORT" \
    --n-predict -1 \
    --ctx-size "$LLM_CONTEXT_SIZE"