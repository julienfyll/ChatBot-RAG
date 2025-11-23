import os
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

# --- DÉFINITION DES STRUCTURES DE DONNÉES ---

class PathsSettings(BaseModel):
    """
    Définit où sont rangés les fichiers.
    default = La valeur utilisée sur TON PC (Local).
    """
    docs: Path = Field(default=Path("data/raw"))
    cache: Path = Field(default=Path("data/processed_texts"))
    chroma_dir: Path = Field(default=Path("chroma_db_local"))

class RetrievalSettings(BaseModel):
    """
    Paramètres techniques du moteur de recherche.
    """
    chunk_size: int = Field(default=1000, ge=50) # ge=50 : Interdit d'avoir moins de 50 car.
    overlap: int = Field(default=200)
    collection_name: str = Field(default="documents_sensibles")
    batch_encode: int = Field(default=32)
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        description="Nom du modèle d'embedding (HuggingFace)"
    )

class RagSettings(BaseModel):
    """
    Configuration principale du RAG.
    """
    model: str = Field(default="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    base_url: str = Field(default="http://127.0.0.1:8080/v1") # Localhost par défaut pour tes tests locaux
    api_key: str = Field(default="pas_de_clef")
    
    # On imbrique les sous-sections
    paths: PathsSettings = Field(default_factory=PathsSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)

class GlobalConfig(BaseModel):
    """
    La classe racine qui contient tout.
    """
    rag: RagSettings = Field(default_factory=RagSettings)

    @classmethod
    def load_config(cls, json_path: str = "config.json"):
        """
        Charge la configuration intelligemment.
        - Si on est dans Docker : Force la lecture du JSON.
        - Si on est en Local : Ignore le JSON (chemins faux) et utilise les défauts.
        """
        # On regarde si la variable d'environnement Docker est présente
        is_docker = os.getenv("RAG_ENVIRONMENT") == "docker"
        file_exists = os.path.exists(json_path)

        if is_docker and file_exists:
            print(f" [Settings] Mode Docker détecté : Chargement de {json_path}")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Pydantic valide que le JSON correspond bien aux classes ci-dessus
            return cls(**data)
        
        elif not is_docker and file_exists:
            print(f" [Settings] Mode Local détecté : {json_path} ignoré (chemins Docker).")
            print("   -> Utilisation des valeurs par défaut (chemins relatifs).")
            return cls() # Retourne les valeurs par défaut définies dans les classes
            
        else:
            print(" [Settings] Pas de fichier config. Utilisation des valeurs par défaut.")
            return cls()


