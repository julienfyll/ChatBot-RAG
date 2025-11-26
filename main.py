import uvicorn
from fastapi import FastAPI, Request
import traceback
from pydantic import BaseModel, Field  
from typing import Optional, List, Any

from src.rag.settings import GlobalConfig
from src.rag.rag import Rag 

app = FastAPI()

# Variable globale
rag_instance = None

# --- Modèles de données (Contrat d'Interface) ---

class LegacyQueryRequest(BaseModel):
    """
    Reflète la requête envoyée par le Backend Node.js.
    Le backend envoie toujours : { "question": "...", "history": [...] }
    Note: 'history' est accepté pour ne pas casser la validation, mais ignoré par la logique.
    """
    # Node.js envoie souvent "query" ou "question", j'utilise un alias pour accepter "query"
    query: str 
    history: Optional[List[Any]] = Field(default=None)

class LegacyQueryResponse(BaseModel):
    """
    Reflète la réponse attendue par le Backend Node.js.
    Le backend attend : data.response
    """
    response: str


@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    global rag_instance
    print(" Démarrage du RAG Container...")
    try:
        # Chargement intelligent (Docker vs Local)
        config = GlobalConfig.load_config("config.json")
        
        rag_instance = Rag(
            model=config.rag.model,
            base_url=config.rag.base_url,
            api_key=config.rag.api_key,
            path_doc=config.rag.paths.docs,
            chroma_persist_dir=config.rag.paths.chroma_dir,
            processed_texts_dir=config.rag.paths.cache
        )
        print(" RAG Initialisé (Global)")
    except Exception as e:
        print(f" Erreur d'init globale : {e}")
        traceback.print_exc()

@app.post("/query", response_model=LegacyQueryResponse)
async def handle_query(payload: LegacyQueryRequest):
    """
    Endpoint query adapté.
    Reçoit : JSON strict (query, history)
    Renvoie : JSON strict (response)
    """
    # Extraction propre via Pydantic
    question = payload.query
    
    # Validation
    if not question:
        return LegacyQueryResponse(response="Erreur : Question vide")

    try:
        config = GlobalConfig.load_config("config.json")

        current_rag = Rag(
            model=config.rag.model,
            base_url=config.rag.base_url,
            api_key=config.rag.api_key,
            path_doc=config.rag.paths.docs,
            chroma_persist_dir=config.rag.paths.chroma_dir,
            processed_texts_dir=config.rag.paths.cache
        )
        
        # Activation Collection
        col_name = config.rag.retrieval.collection_name
        print(f" Query sur la collection : {col_name}")
        current_rag.retrieval.chroma_storage.switch_collection(col_name)
        
        response_text = current_rag.respond(question)
        
        # Formatage pour le Node.js (champ 'response')
        return LegacyQueryResponse(response=response_text)
            
    except Exception as e:
        traceback.print_exc() 
        return LegacyQueryResponse(response=f"Error processing request: {str(e)}")

@app.post("/ingest")
async def trigger_ingest():
    """
    Endpoint pour déclencher la vectorisation AVANCÉE.
    Utilise vectorize_with_config pour gérer les métadonnées et la création propre.
    """
    try:
        # Chargement de la configuration fraîche
        config = GlobalConfig.load_config("config.json")
        
        # Création de l'instance RAG temporaire
        ingest_rag = Rag(
            model=config.rag.model,
            base_url=config.rag.base_url,
            api_key=config.rag.api_key,
            path_doc=config.rag.paths.docs,
            chroma_persist_dir=config.rag.paths.chroma_dir,
            processed_texts_dir=config.rag.paths.cache
        )
        
        col_name = config.rag.retrieval.collection_name
        print(f" Début ingestion vers : {col_name} (via vectorize_with_config)")
        

        success = ingest_rag.retrieval.vectorize_with_config(
            chunk_size=config.rag.retrieval.chunk_size,
            overlap=config.rag.retrieval.overlap,
            collection_name=col_name,
            source_folder=str(config.rag.paths.docs), 
            model_name=config.rag.retrieval.embedding_model, 
        )
        
        if success:
            return {
                "status": "Succès",
                "collection": col_name,
                "chunk_size": config.rag.retrieval.chunk_size,
                "nb_files": "Voir logs" 
            }
        else:
            return {"error": "Echec de la vectorisation (voir logs Docker)"}
            
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)