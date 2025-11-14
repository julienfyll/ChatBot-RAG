# config.py
from pathlib import Path

# CHEMINS RELATIFS À LA RACINE DU PROJET
# On part du principe que les scripts sont lancés depuis la racine du projet (RAG_CDL25/).

# Chemin racine des données sources.
ROOT_DATA_PATH = Path("data/raw")

# Dossier où le cache des textes extraits sera stocké.
PROCESSED_TEXTS_DIR = Path("data/processed_texts")

# Dossier où la base de données ChromaDB est persistée.
CHROMA_PERSIST_DIR = Path("chroma_db_local")

# ... (le reste de la configuration des modèles ne change pas) ...
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_LLM_MODEL = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
LLM_BASE_URL = "http://127.0.0.1:8080/v1"
