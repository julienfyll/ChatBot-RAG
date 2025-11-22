import numpy as np
from .llm import LLM
from .retrieval import Retrieval
from pathlib import Path
from .config import ROOT_DATA_PATH, CHROMA_PERSIST_DIR, DEFAULT_LLM_MODEL, LLM_BASE_URL


class Rag:
    # personnalisation des paramètres d'initialisation, les valeurs par défaut sont fournies
    # toutes les personnalisations sont à fournir lors de l'appel de la classe rag
    def __init__(
        self, model=DEFAULT_LLM_MODEL, base_url=LLM_BASE_URL, api_key="pas_de_clef"
    ):
        # Initialisation des composants LLM et Retrieval avec paramètres personnalisés
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.path_doc = ROOT_DATA_PATH
        self.chroma_persist_dir = CHROMA_PERSIST_DIR

        try:
            project_root = Path(__file__).parent.parent  # src/rag/ -> src/
            prompt_path = project_root / "prompts" / "rag_template.txt"
            self.prompt_template = prompt_path.read_text(encoding="utf-8")
            print(f"✓ Prompt chargé depuis : {prompt_path}")
        except FileNotFoundError:
            print(f" ERREUR : Le fichier de prompt '{prompt_path}' est introuvable.")
            self.prompt_template = "CONTEXTE:\n{context_concat}\n\nQUESTION:\n{query}"

        self.llm = LLM(model=self.model, base_url=self.base_url, api_key=self.api_key)

        self.retrieval = Retrieval(
            path_doc=self.path_doc,
            chroma_persist_dir=self.chroma_persist_dir,
        )

        return

    def respond(self, query: str) -> str:
        top_k = 5  # Nombre de contextes à récupérer

        # 1) Garde-fou minimal
        if not query or not isinstance(query, str) or query.strip() == "":
            return "Merci de préciser votre question."

        # 2) Appel au retriever (avec les scores)
        try:
            contextes, sources, scores = self.retrieval.query(
                query, n=top_k
            )  # top_k est le nombre de contextes à récupérer
        except FileNotFoundError:
            # Cas où la base vectorielle n'existe pas encore
            print(
                "La base documentaire n'est pas prête. Lance d'abord la vectorisation (update) pour créer la BDD."
            )
            return (
                "La base documentaire n'est pas prête. "
                "Lance d'abord la vectorisation (update) pour créer la BDD."
            )
        except Exception as e:
            print(f"Une erreur est survenue pendant la recherche du contexte : {e}")
            return f"Une erreur est survenue pendant la recherche du contexte : {e}"

        # 3) Construit le CONTEXTE avec les sources intégrées et numérotées
        blocks = []
        for i, (ctx, src, sc) in enumerate(zip(contextes, sources, scores), start=1):
            label_src = str(src)
            score_txt = f"(score: {sc:.3f})" if sc is not None else ""
            blocks.append(f"[{i}] Source: {label_src} {score_txt}\n{ctx}")

        context_concat = "\n\n---\n\n".join(blocks)

        # 4) Prépare le prompt
        prompt = self.prompt_template.format(context_concat=context_concat, query=query)

        print(prompt)
        print("=== PROMPT ENVOYÉ AU LLM ===")

        # 5) Appel du LLM
        try:
            reponse = self.llm.infere(prompt)
        except Exception as e:
            return f"Une erreur est survenue pendant l'inférence du LLM : {e}"

        # 6) Affichage clair des sources + scores

        sources_lines = []
        for i, (chemin, score) in enumerate(zip(sources, scores), start=1):
            if score is not None:
                sources_lines.append(f"[{i}] {chemin} (score: {score:.3f})")
            else:
                sources_lines.append(f"[{i}] {chemin}")
        if sources_lines:
            reponse = f"{reponse}\n\nSources (Top-{len(sources_lines)}):\n" + "\n".join(
                sources_lines
            )

        return reponse

    def update(self):
        return

    def reload(self):
        return
