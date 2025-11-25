# test_rendering.py
from src.rag.rag import Rag
from pathlib import Path

def test_prompt_generation():
    print(">>> 1. Initialisation du RAG (Mode Test)...")
    # On mock les chemins pour le local
    rag = Rag(
        path_doc="data/raw",
        chroma_persist_dir="./chroma_db_local",
        processed_texts_dir="data/processed_texts"
    )

    print(">>> 2. Simulation de données contextuelles...")
    # On force des données bidons pour tester UNIQUEMENT le rendu Jinja
    # Pas besoin d'interroger ChromaDB ou le LLM pour ce test
    contextes = ["Contenu du document A", "Contenu du document B"]
    sources = ["data/raw/dossier/doc_A.pdf", "data/raw/doc_B.txt"]
    scores = [0.85, 0.72]
    query = "Quel est le sens de la vie ?"

    print(">>> 3. Génération du Prompt via Jinja2...")
    # On accède "illégalement" à la méthode interne pour tester le rendu
    # (Ou on refait la logique de préparation data ici pour valider)
    
    documents_context = []
    for i, (ctx, src, score) in enumerate(zip(contextes, sources, scores), start=1):
        documents_context.append({
            "id": i,
            "source_name": Path(src).name,
            "score": score,
            "content": ctx
        })

    prompt_final = rag.template.render(
        documents=documents_context,
        query=query
    )

    print("\n=== RÉSULTAT DU RENDU JINJA ===")
    print(prompt_final)
    print("===============================")
    
    if "doc_A.pdf" in prompt_final and "<|start_header_id|>" in prompt_final:
        print("✅ SUCCÈS : Le template et les variables sont corrects.")
    else:
        print("❌ ÉCHEC : Le rendu ne contient pas les éléments attendus.")

if __name__ == "__main__":
    test_prompt_generation()