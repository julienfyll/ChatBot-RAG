import uvicorn
from fastapi import FastAPI, Request
import traceback

# Import des Settings
from src.rag.settings import GlobalConfig
# Import de la classe RAG
from src.rag.rag import Rag 

app = FastAPI()

# Variable globale
rag_instance = None

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    global rag_instance
    print(" Démarrage du RAG Container...")
    try:
        # 1. Chargement intelligent (Docker vs Local)
        config = GlobalConfig.load_config("config.json")
        
        # On a maintenant des objets typés, pas des dictionnaires !
        # config.rag.paths.docs  au lieu de  config["rag"]["paths"]["docs"]
        
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

@app.post("/query")
async def handle_query(request: Request):
    data = await request.json()
    question = data.get("query", "")
    
    if not question:
        return {"error": "Question vide"}

    try:
        # 1. Chargement Config
        config = GlobalConfig.load_config("config.json")
        
        # 2. Création Instance
        current_rag = Rag(
            model=config.rag.model,
            base_url=config.rag.base_url,
            api_key=config.rag.api_key,
            path_doc=config.rag.paths.docs,
            chroma_persist_dir=config.rag.paths.chroma_dir,
            processed_texts_dir=config.rag.paths.cache
        )
        
        # 3. Activation Collection
        col_name = config.rag.retrieval.collection_name
        print(f" Query sur la collection : {col_name}")
        current_rag.retrieval.chroma_storage.switch_collection(col_name)
        
        # 4. Réponse
        response = current_rag.respond(question)
        
        return {"query": question, "response": response}
            
    except Exception as e:
        traceback.print_exc() 
        return {"error": str(e)}

@app.post("/ingest")
async def trigger_ingest():
    """
    Endpoint pour déclencher la vectorisation AVANCÉE.
    Utilise vectorize_with_config pour gérer les métadonnées et la création propre.
    """
    try:
        # 1. Chargement de la configuration fraîche
        config = GlobalConfig.load_config("config.json")
        
        # 2. Création de l'instance RAG temporaire
        # Elle est configurée avec les bons chemins (paths)
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
        
        # 3. Appel de la méthode RICHE (vectorize_with_config)
        # On passe tous les paramètres issus du fichier de config
        success = ingest_rag.retrieval.vectorize_with_config(
            chunk_size=config.rag.retrieval.chunk_size,
            overlap=config.rag.retrieval.overlap,
            collection_name=col_name,
            # On force le chemin source défini dans les paths (converti en string)
            source_folder=str(config.rag.paths.docs), 
            # On utilise le modèle défini dans la config
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