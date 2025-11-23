import os
import json
import uvicorn
from fastapi import FastAPI, Request
from pathlib import Path
import traceback

# Import de ta classe RAG
from src.rag.rag import Rag 

app = FastAPI()

# Variable globale (sert de fallback ou pour le démarrage)
rag_instance = None

def load_config():
    """Charge la configuration depuis le fichier JSON monté"""
    config_path = Path("config.json")
    if not config_path.exists():
        raise FileNotFoundError("Le fichier config.json est manquant !")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage (Optionnel avec le Hot-Reload, mais bon pour tester)"""
    global rag_instance
    print(" Démarrage du RAG Container...")
    try:
        config = load_config()
        params = config["rag"]
        
        # Instance globale de démarrage
        rag_instance = Rag(
            model=params["model"],
            base_url=params["base_url"],
            api_key=params.get("api_key", "pas_de_clef")
        )
        print(" RAG Initialisé (Global)")
    except Exception as e:
        print(f" Erreur d'init globale (non critique si Hot-Reload actif) : {e}")

@app.post("/query")
async def handle_query(request: Request):
    """
    Endpoint d'interrogation.
    Utilise une variable LOCALE pour garantir que la config est fraîche 
    et éviter les conflits entre requêtes simultanées.
    """
    data = await request.json()
    question = data.get("query", "")
    
    if not question:
        return {"error": "Question vide"}

    try:
        # 1. Chargement Dynamique de la config
        config = load_config()
        params = config["rag"]
        
        # 2. Création d'une instance LOCALE (Thread-safe)
        current_rag = Rag(
            model=params["model"],
            base_url=params["base_url"],
            api_key=params.get("api_key", "pas_de_clef")
        )
        
        # 3. ACTIVATION DE LA COLLECTION (Le Correctif Vital)
        # Récupère le nom dans le JSON, ou utilise 'documents_sensibles' par défaut
        col_name = params["retrieval"].get("collection_name", "documents_sensibles")
        
        print(f" Query sur la collection : {col_name}")
        current_rag.retrieval.chroma_storage.switch_collection(col_name)
        
        # 4. Réponse
        response = current_rag.respond(question)
        
        return {
            "query": question, 
            "response": response
        }
            
    except Exception as e:
        traceback.print_exc() # Affiche l'erreur dans les logs Docker
        return {"error": str(e)}

@app.post("/ingest")
async def trigger_ingest():
    """Endpoint pour déclencher la vectorisation"""
    try:
        config = load_config()
        # On lit la collection cible depuis la config
        col_name = config["rag"]["retrieval"].get("collection_name", "documents_sensibles")
        
        # On crée une instance temporaire dédiée à l'ingestion
        ingest_rag = Rag(
            base_url=config["rag"]["base_url"]
        )
        
        print(f"📥 Début ingestion vers : {col_name}")
        
        # Appel de la méthode d'ajout
        success = ingest_rag.retrieval.add_documents(
            collection_name=col_name,
            source_path="/app/data/raw", 
            overwrite_duplicates=False
        )
        
        if success:
            return {"status": f"Succès : Documents ajoutés à '{col_name}'"}
        else:
            return {"error": "Echec de l'ingestion (voir logs)"}
            
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)