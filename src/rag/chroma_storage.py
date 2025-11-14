import os
import chromadb
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple
from collections import Counter


#  DÉSACTIVER LA TÉLÉMÉTRIE **AVANT** l'import ChromaDB
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# DÉSACTIVER LES LOGS CHROMADB (capture les ERROR logs)
import logging

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


class ChromaStorage:
    def __init__(self, persist_directory="./chroma_db_local"):
        self.persist_directory = persist_directory
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        self.collection_name = None
        self.collection = None

    def switch_collection(self, collection_name: str):
        """
        Change la collection ChromaDB active.

        Args:
            collection_name (str): Nom de la nouvelle collection à utiliser
        """
        self.collection_name = collection_name
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": f"Collection {collection_name}"},
        )
        print(
            f" Collection active : {collection_name} ({self.collection.count()} docs)"
        )

    def delete_collection(self):
        """Supprime complètement la collection actuelle"""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f" Collection '{self.collection_name}' supprimée")
            return True
        except Exception as e:
            print(f" Erreur suppression : {e}")
            return False

    def rename_collection(self, old_name: str, new_name: str) -> bool:
        """
        "Renomme" une collection en copiant tous ses documents et métadonnées
        vers une nouvelle collection, puis en supprimant l'ancienne.
        Cette méthode contient la logique métier et est agnostique de l'interface utilisateur.

        Args:
            old_name (str): Nom actuel de la collection.
            new_name (str): Nouveau nom souhaité.

        Returns:
            bool: True si le renommage a réussi, False sinon.
        """
        import uuid  # Import local pour cette méthode

        print(f"\n[ChromaStorage] Début du renommage : '{old_name}' -> '{new_name}'")

        # Validation en amont
        if not old_name or not new_name or old_name == new_name:
            print("  /!\\ Erreur : Noms de collection invalides ou identiques.")
            return False

        try:
            # 1. Récupérer l'ancienne collection
            old_col = self.chroma_client.get_collection(old_name)
            count = old_col.count()

            # 2. Créer la nouvelle collection avec les mêmes métadonnées
            new_col = self.chroma_client.create_collection(
                name=new_name, metadata=old_col.metadata
            )

            # 3. Si la collection n'est pas vide, copier les données
            if count > 0:
                print(f"  Copie de {count} documents...")
                # On récupère tout d'un coup
                data = old_col.get(include=["documents", "metadatas", "embeddings"])

                # ChromaDB peut retourner plus de documents que son .count(), on s'assure de ne prendre que le bon nombre
                if len(data["ids"]) > count:
                    for key in data:
                        if data[key]:
                            data[key] = data[key][:count]

                # On doit générer de nouveaux IDs car ils doivent être uniques
                new_ids = [str(uuid.uuid4()) for _ in range(count)]

                # Ajouter les données à la nouvelle collection
                new_col.add(
                    ids=new_ids,
                    documents=data["documents"],
                    metadatas=data["metadatas"],
                    embeddings=data["embeddings"],
                )

                # Vérification de sécurité
                if new_col.count() != count:
                    raise ValueError(
                        f"La copie a échoué. Attendu: {count}, Obtenu: {new_col.count()}."
                    )

            # 4. Si tout s'est bien passé, supprimer l'ancienne collection
            print(
                f"  Copie terminée. Suppression de l'ancienne collection '{old_name}'..."
            )
            self.chroma_client.delete_collection(name=old_name)

            print("✓ [ChromaStorage] Renommage terminé avec succès.")
            return True

        except Exception as e:
            print(f"  /!\\ Erreur critique lors du renommage : {e}")
            # Nettoyage : si la nouvelle collection a été créée mais que l'opération a échoué,
            # on la supprime pour ne pas laisser un état incohérent.
            try:
                self.chroma_client.delete_collection(name=new_name)
                print(
                    f"  Nettoyage : La collection intermédiaire '{new_name}' a été supprimée."
                )
            except:
                pass  # Ignorer les erreurs de nettoyage
            return False

    def create_collection_with_metadata(self, collection_name: str, metadata: dict):
        """
        Crée ou récupère une collection avec des métadonnées enrichies.

        Args:
            collection_name (str): Nom de la collection
            metadata (dict): Métadonnées à stocker (chunk_size, overlap, etc.)
        """
        # Métadonnées par défaut
        base_metadata = {
            "description": f"Collection {collection_name}",
            "created_at": None,
            "chunk_size": None,
            "overlap": None,
            "source_folder": None,
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "reranking_enabled": None,
            "created_by": None,
            "version": "3.0",
        }

        # Fusionner avec les métadonnées fournies
        base_metadata.update(metadata)

        # Ajouter la clé hnsw:space dans base_metadata
        base_metadata["hnsw:space"] = "cosine"

        # Créer/récupérer la collection
        self.collection_name = collection_name
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, metadata=base_metadata
        )

        print(f" Collection '{collection_name}' chargée/créée")
        return self.collection

    def migrate_from_json(self, json_path: str) -> bool:
        print(f" Migration depuis {json_path}...")

        if not Path(json_path).exists():
            print(f" Fichier non trouvé : {json_path}")
            return False

        if self.collection.count() > 0:
            print(f" Collection déjà remplie ({self.collection.count()} docs)")
            return True

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f" {len(data)} embeddings trouvés")
        except Exception as e:
            print(f" Erreur de lecture JSON : {e}")
            return False

        batch_size = 100
        total_migrated = 0

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            documents = []
            metadatas = []
            embeddings = []
            ids = []

            for j, item in enumerate(batch):
                doc_id = f"doc_{i + j}"
                ids.append(doc_id)
                documents.append(item["batch"])

                metadatas.append(
                    {
                        "chemin": item["chemin"],
                        "position_debut": item.get("position_debut", 0),
                        "taille_texte": len(item["batch"]),
                    }
                )

                embedding = item["embeddings"]
                if isinstance(embedding, list):
                    embeddings.append(embedding)
                else:
                    embeddings.append(embedding.tolist())

            try:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    ids=ids,
                )
                total_migrated += len(batch)
                print(
                    f" Batch {i // batch_size + 1}: {total_migrated}/{len(data)} migrés"
                )

            except Exception as e:
                print(f" Erreur batch {i}: {e}")
                continue

        print(f" Migration terminée ! {total_migrated} documents dans ChromaDB")
        return True

    def query_similar(
        self, query_embedding: np.ndarray, n_results: int = 3
    ) -> Tuple[List[str], List[str], List[float]]:
        try:
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=n_results
            )

            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            similarities = [1.0 - dist for dist in distances]
            sources = [meta.get("chemin", "unknown") for meta in metadatas]

            return documents, sources, similarities

        except Exception as e:
            print(f" Erreur de requête ChromaDB : {e}")
            return [], [], []

    def add_document(
        self, document: str, chemin: str, embedding: np.ndarray, position_debut: int = 0
    ) -> bool:
        import uuid

        doc_id = str(uuid.uuid4())

        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        try:
            self.collection.add(
                documents=[document],
                metadatas=[
                    {
                        "chemin": chemin,
                        "position_debut": position_debut,
                        "taille_texte": len(document),
                    }
                ],
                embeddings=[embedding],
                ids=[doc_id],
            )
            return True
        except Exception as e:
            print(f" Erreur ajout document : {e}")
            return False

    def count_documents(self) -> int:
        return self.collection.count()

    def get_stats(self) -> dict:
        """
        Retourne des statistiques complètes sur la collection active.
        Combine le comptage global et le résumé par fichier source.

        Returns:
            dict: Un dictionnaire contenant :
                - 'total_documents' (int): Nombre total de chunks.
                - 'total_fichiers' (int): Nombre de fichiers sources uniques.
                - 'sources_summary' (List[dict]): Une liste détaillée pour chaque fichier source,
                  contenant 'chemin', 'filename', 'nb_chunks', etc.
        """
        print(
            f"[ChromaStorage] Calcul des statistiques pour '{self.collection_name}'..."
        )

        count = self.collection.count()

        # Si la collection est vide, retourner des stats vides.
        if count == 0:
            return {"total_documents": 0, "total_fichiers": 0, "sources_summary": []}

        # Récupérer TOUS les documents pour analyser les métadonnées.
        # C'est une opération potentiellement coûteuse sur de très grandes collections.
        results = self.collection.get(include=["metadatas"])

        # --- Logique de get_sources_with_stats déplacée ici ---
        sources_count = Counter()
        if results and results["metadatas"]:
            for metadata in results["metadatas"]:
                chemin = metadata.get("chemin")
                if chemin:
                    sources_count[chemin] += 1

        # Construire la liste structurée
        sources_summary = []
        for chemin, nb_chunks in sources_count.items():
            sources_summary.append(
                {
                    "chemin": chemin,
                    "filename": Path(chemin).name,
                    "nb_chunks": nb_chunks,
                    "folder": str(Path(chemin).parent),
                }
            )

        # Trier pour un affichage cohérent
        sources_summary.sort(key=lambda x: (-x["nb_chunks"], x["filename"]))

        # Construire le dictionnaire de statistiques final
        stats = {
            "total_documents": count,
            "total_fichiers": len(sources_summary),
            "sources_summary": sources_summary,
        }

        return stats

    def delete_by_source(self, chemin: str) -> bool:
        """
        Supprime tous les documents associés à un fichier source.

        Args:
            chemin (str): Chemin du fichier source à supprimer

        Returns:
            bool: True si la suppression a réussi
        """
        try:
            # ChromaDB : suppression par filtre de métadonnées
            self.collection.delete(where={"chemin": chemin})
            print(f"Supprimé de ChromaDB : {chemin}")
            return True
        except Exception as e:
            print(f" Erreur suppression : {e}")
            return False

    def list_collection_names(self) -> List[str]:
        """
        Retourne la liste des noms de toutes les collections dans ChromaDB.
        Cette méthode gère les différences potentielles entre les OS.

        Returns:
            List[str]: Une liste des noms de collections.
        """
        try:
            collections_raw = self.chroma_client.list_collections()
            # La méthode .name est la méthode moderne et multi-plateforme.
            return [col.name for col in collections_raw]
        except Exception as e:
            print(f"Erreur lors de la récupération des collections : {e}")
            return []

    # chroma_storage.py [11]
    def migrate_paths_to_relative(self, root_data_path: Path) -> Tuple[int, int]:
        """
        Parcourt tous les documents et s'assure que la métadonnée 'chemin'
        est un chemin relatif par rapport à root_data_path, en traitant par lots.
        """
        print(
            f"\n[ChromaStorage] Migration des chemins vers un format relatif pour '{self.collection_name}'..."
        )
        resolved_root = root_data_path.resolve()
        print(f"  Chemin racine de référence : {resolved_root}")

        count = self.collection.count()
        if count == 0:
            return 0, 0

        data = self.collection.get(include=["metadatas"])

        ids_to_update, metadatas_to_update = [], []
        total_modified_count = 0

        #  Définir une taille de lot
        batch_size = 1000

        for i in range(len(data["ids"])):
            doc_id, metadata = data["ids"][i], data["metadatas"][i]
            original_path_str = metadata.get("chemin")

            if not original_path_str:
                continue

            try:
                current_absolute_path = (Path.cwd() / original_path_str).resolve()
                relative_path = current_absolute_path.relative_to(
                    resolved_root
                ).as_posix()

                if original_path_str != relative_path:
                    total_modified_count += 1
                    new_metadata = metadata.copy()
                    new_metadata["chemin"] = relative_path

                    ids_to_update.append(doc_id)
                    metadatas_to_update.append(new_metadata)

                    #  Si le lot est plein, on l'envoie et on vide les listes
                    if len(ids_to_update) >= batch_size:
                        print(
                            f"  Traitement d'un lot de {len(ids_to_update)} documents..."
                        )
                        self.collection.update(
                            ids=ids_to_update, metadatas=metadatas_to_update
                        )
                        ids_to_update.clear()
                        metadatas_to_update.clear()

            except ValueError:
                continue

        # Traiter le dernier lot s'il en reste
        if ids_to_update:
            print(f"  Traitement du dernier lot de {len(ids_to_update)} documents...")
            self.collection.update(ids=ids_to_update, metadatas=metadatas_to_update)

        if total_modified_count == 0:
            print(
                "✓ Tous les chemins sont déjà correctement relatifs à la racine des données."
            )
            return 0, count

        print(
            f"\n✓ Migration terminée. {total_modified_count} document(s) mis à jour au total."
        )
        return total_modified_count, count
