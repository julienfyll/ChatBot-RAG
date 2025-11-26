import numpy as np
from .llm import LLM
from .retrieval import Retrieval
from pathlib import Path
from jinja2 import Template

class Rag:
    # personnalisation des paramètres d'initialisation, les valeurs par défaut sont fournies
    # toutes les personnalisations sont à fournir lors de l'appel de la classe rag
    def __init__(
        self, 
        model="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", 
        base_url="http://127.0.0.1:8080/v1", 
        api_key="pas_de_clef",
        # Valeurs par défaut locales
        path_doc="data/raw",
        chroma_persist_dir="./chroma_db_local",
        processed_texts_dir="data/processed_texts"
    ):
        # Initialisation des composants LLM et Retrieval avec paramètres personnalisés
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

        self.path_doc = Path(path_doc)
        self.chroma_persist_dir = Path(chroma_persist_dir)
        self.processed_texts_dir = Path(processed_texts_dir)

        try:
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            
            prompt_path = project_root / "prompts" / "rag_template.j2"
            
            # Sécurité : Si on est dans Docker, le WORKDIR est /app, donc prompts est souvent à /app/prompts
            if not prompt_path.exists():
                prompt_path = Path("/app/prompts/rag_template.j2")

            if not prompt_path.exists():
                 raise FileNotFoundError(f"Template introuvable à : {project_root / 'prompts'}")

            with open(prompt_path, "r", encoding="utf-8") as f:
                self.template = Template(f.read())
            print(f"✓ Template Jinja2 chargé depuis : {prompt_path}")
            
        except Exception as e:
            print(f" ERREUR CRITIQUE : Impossible de charger le template Jinja2 ({e})")
            # Fallback de secours (très basique) pour éviter le crash
            self.template = Template("CONTEXTE:\n{% for d in documents %}{{ d.content }}\n{% endfor %}\nQUESTION:\n{{ query }}")

        self.llm = LLM(model=self.model, base_url=self.base_url, api_key=self.api_key)

        self.retrieval = Retrieval(
            path_doc=self.path_doc,
            chroma_persist_dir=self.chroma_persist_dir,
            processed_texts_dir=self.processed_texts_dir,
        )

        return

    def respond(self, query: str) -> str:
        top_k = 5 

        # 1) Garde-fou minimal
        if not query or not isinstance(query, str) or not query.strip():
            return "Merci de préciser votre question."

        # 2) Appel au retriever
        try:
            contextes, sources, scores = self.retrieval.query(query, n=top_k)
        except FileNotFoundError:
            return "La base documentaire n'est pas prête."
        except Exception as e:
            return f"Une erreur est survenue pendant la recherche du contexte : {e}"

        # 3) PRÉPARATION DES DONNÉES (ViewModel pour Jinja2)
        # C'EST ICI QUE CA CHANGE : On ne concatène plus de texte manuellement.
        documents_context = []
        
        # On itère sur les résultats du retriever
        for i, (ctx, src, sc) in enumerate(zip(contextes, sources, scores), start=1):
            documents_context.append({
                "id": i,
                "source_name": Path(src).name, # Juste le nom du fichier (plus propre)
                "score": float(sc) if sc is not None else 0.0,
                "content": ctx.strip()
            })

        # 4) RENDU DU PROMPT VIA JINJA
        # On passe la liste d'objets au template
        prompt = self.template.render(
            documents=documents_context,
            query=query
        )

        # 5) Appel du LLM
        try:
            reponse = self.llm.infere(prompt)
        except Exception as e:
            return f"Une erreur est survenue pendant l'inférence du LLM : {e}"

        # 6) Affichage des sources en bas de réponse
        sources_lines = []
        for doc in documents_context:
            sources_lines.append(f"[{doc['id']}] {doc['source_name']} (score: {doc['score']:.3f})")
            
        if sources_lines:
            reponse = f"{reponse}\n\nSources (Top-{len(sources_lines)}):\n" + "\n".join(sources_lines)

        return reponse

    def update(self):
        return

    def reload(self):
        return
