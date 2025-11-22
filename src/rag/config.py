# # config.py
# from pathlib import Path

# # CHEMINS RELATIFS À LA RACINE DU PROJET
# # On part du principe que les scripts sont lancés depuis la racine du projet (RAG_CDL25/).

# # Chemin racine des données sources.
# ROOT_DATA_PATH = Path("data/raw")

# # Dossier où le cache des textes extraits sera stocké.
# PROCESSED_TEXTS_DIR = Path("data/processed_texts")

# # Dossier où la base de données ChromaDB est persistée.
# CHROMA_PERSIST_DIR = Path("chroma_db_local")

# # ... (le reste de la configuration des modèles ne change pas) ...
# DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# DEFAULT_LLM_MODEL = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
# LLM_BASE_URL = "http://127.0.0.1:8080/v1"


import os
from pathlib import Path
from dotenv import load_dotenv

# 1. LOCALISATION DE LA RACINE DU PROJET
# On part de ce fichier (src/rag/config.py)
# .parent = src/rag
# .parent.parent = src
# .parent.parent.parent = RACINE DU PROJET (RAG_CGT/)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# 2. CONNEXION : Chargement du fichier .env
env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ [Config] Variables chargées depuis {env_path}")
else:
    print(f"⚠️ [Config] Fichier .env introuvable à {env_path}. Utilisation des valeurs par défaut.")

# ==========================================
# CONSTANTES INTERNES (Structure du projet)
# ==========================================
# Ces chemins sont relatifs au projet, ils ne changent pas d'une machine à l'autre
ROOT_DATA_PATH = Path("data/raw")
PROCESSED_TEXTS_DIR = Path("data/processed_texts")
CHROMA_PERSIST_DIR = Path("chroma_db_local")

# ==========================================
# VARIABLES D'ENVIRONNEMENT (Récupérées du .env)
# ==========================================

# os.getenv("NOM_VAR", "VALEUR_PAR_DEFAUT")
# La valeur par défaut sert de filet de sécurité si le .env est mal configuré

LLM_BINARY_PATH = os.getenv("LLM_BINARY_PATH", "llama-server")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/default_model.gguf")

LLM_HOST = os.getenv("LLM_HOST", "127.0.0.1")

# Transformation de type : Le port doit être un entier (int)
try:
    LLM_PORT = int(os.getenv("LLM_PORT", "8080"))
except ValueError:
    LLM_PORT = 8080 # Fallback si quelqu'un a écrit du texte dans le port

# Construction de l'URL complète pour Python
LLM_BASE_URL = f"http://{LLM_HOST}:{LLM_PORT}/v1"

# Paramètres de performance
try:
    LLM_CONTEXT_SIZE = int(os.getenv("LLM_CONTEXT_SIZE", "8192"))
    LLM_GPU_LAYERS = int(os.getenv("LLM_GPU_LAYERS", "99"))
except ValueError:
    LLM_CONTEXT_SIZE = 8192
    LLM_GPU_LAYERS = 99

# Nom du modèle pour l'affichage (juste le nom du fichier)
DEFAULT_LLM_MODEL = Path(LLM_MODEL_PATH).name

# Nom du modèle d'embedding
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL_NAME", 
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)