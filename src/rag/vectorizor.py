import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class Vectorizor:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ):
        """
        Initialise le vectorizor avec un modèle par défaut.

        Args:
            model_name (str): Nom du modèle HuggingFace à charger
        """
        self.model_name = model_name
        self.model = None
        self._model_cache = {}  # Cache des modèles chargés
        self._load_model(model_name)
        return

    def _load_model(self, model_name: str):
        """
        Charge un modèle d'embedding avec gestion d'erreurs robuste.

        Args:
            model_name (str): Nom du modèle HuggingFace
        """

        # Vérifier le cache d'abord
        if model_name in self._model_cache:
            print(f"⚡ Modèle récupéré du cache : {model_name}")
            self.model = self._model_cache[model_name]
            self.model_name = model_name
            return

        print(f" Chargement du modèle : {model_name}")

        try:
            # Cas spécial : Qwen nécessite trust_remote_code
            if "Qwen" in model_name or "qwen" in model_name:
                try:
                    model = SentenceTransformer(
                        model_name,
                        tokenizer_kwargs={"padding_side": "left"},
                        device="cuda",
                        trust_remote_code=True,  #  CRUCIAL pour Qwen3
                    )
                except Exception as qwen_error:
                    print(f"  Impossible de charger Qwen : {qwen_error}")
                    print("   → Fallback vers MPNet")

                    # Fallback vers MPNet
                    fallback_model = (
                        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                    )
                    model = SentenceTransformer(fallback_model, device="cuda")
                    model_name = fallback_model  # Mettre à jour le nom
                    print(f" Fallback réussi : {fallback_model}")

                    import torch
                    print(f"🎮 GPU disponible : {torch.cuda.is_available()}")
                    print(f"🎮 GPU utilisé : {next(model.parameters()).device}")


            else:
                # Autres modèles (BERT, MPNet, MiniLM, etc.)
                model = SentenceTransformer(model_name, device="cuda")

            # Stocker dans le cache
            self._model_cache[model_name] = model
            self.model = model
            self.model_name = model_name

            print(f" Modèle chargé : {model_name}")

            import torch
            print(f"🎮 GPU disponible : {torch.cuda.is_available()}")
            print(f"🎮 GPU utilisé : {next(model.parameters()).device}")


        except Exception as e:
            print(f" ERREUR CRITIQUE lors du chargement de {model_name} : {e}")

            # Dernier fallback : MPNet par défaut
            fallback_model = (
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            )

            if model_name != fallback_model:
                print(f"   → Fallback final vers {fallback_model}")
                try:
                    model = SentenceTransformer(fallback_model, device="cuda")
                    self._model_cache[fallback_model] = model
                    self.model = model
                    self.model_name = fallback_model
                    print(f" Fallback final réussi")
                except Exception as final_error:
                    print(f" FALLBACK ÉCHOUÉ : {final_error}")
                    raise RuntimeError("Impossible de charger un modèle d'embedding")
            else:
                raise

    def switch_to_model_for_collection(self, collection_metadata: dict):
        """
         SWITCH DYNAMIQUE : Change le modèle selon les métadonnées de la collection.
        ROBUSTE : Gère les métadonnées manquantes et les modèles non chargeables.

        Args:
            collection_metadata (dict): Métadonnées de la collection ChromaDB
        """

        # Récupérer le nom du modèle depuis les métadonnées
        original_model = collection_metadata.get("model", "unknown")

        #  MAPPING : Nom court → Chemin HuggingFace complet
        model_map = {
            # Qwen
            "Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B",
            # MiniLM
            "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            # MPNet
            "paraphrase-multilingual-mpnet-base-v2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            #  NOUVEAU : Gestion des métadonnées manquantes
            "unknown": None,  # Pas de switch si inconnu
            "N/A": None,
        }

        required_model = model_map.get(original_model)

        #  CAS 1 : Métadonnées manquantes ou invalides
        if required_model is None:
            print(f"\n  Métadonnées manquantes (model={original_model})")
            print(f"   → Conservation du modèle actuel : {self.model_name}")
            print(f"    Conseil : Recréez cette collection avec manage_collections.py")
            return  # Ne rien changer

        #  CAS 2 : Modèle identique, pas de rechargement
        if required_model == self.model_name:
            return  # Rien à faire

        #  CAS 3 : Changement nécessaire
        print(f"\n Changement de modèle détecté :")
        print(f"   Actuel : {self.model_name}")
        print(f"   Requis : {required_model}")

        # Recharger (avec gestion d'erreurs intégrée)
        self._load_model(required_model)

    def get_model_dimension(self) -> int:
        """
        Retourne la dimension du modèle actuel.

        Returns:
            int: Dimension des embeddings (ex: 768, 384, 1024)
        """
        test_emb = self.model.encode("test")
        return len(test_emb)

    def vectorize(self):
        """Méthode legacy (non utilisée)"""
        return

    def encode(self, text_series: pd.Series) -> pd.Series:
        """
        Encode une série de textes en embeddings.

        Args:
            text_series (pd.Series): Série de textes à encoder

        Returns:
            pd.Series: Série d'embeddings (numpy arrays)
        """
        tqdm.pandas(desc="Génération des embeddings")
        embeddings = text_series.progress_apply(lambda x: self.model.encode(x))
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode une requête unique.

        Args:
            query (str): Requête à encoder

        Returns:
            np.ndarray: Embedding de la requête
        """
        # Certains modèles supportent prompt_name, d'autres non
        try:
            query_embeddings = self.model.encode(query, prompt_name="query").astype(
                float
            )
        except (TypeError, AttributeError):
            query_embeddings = self.model.encode(query).astype(float)

        return query_embeddings

    def similarity(self, target_embeddings, db_embeddings):
        """
        Calcule la similarité entre embeddings.

        Args:
            target_embeddings: Embedding cible
            db_embeddings: Embeddings de la base

        Returns:
            Scores de similarité
        """
        return self.model.similarity(target_embeddings, db_embeddings)
