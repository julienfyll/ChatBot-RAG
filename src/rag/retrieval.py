import numpy as np
import pandas as pd
from .vectorizor import Vectorizor
from typing import Tuple, List, Optional
from pathlib import Path
from .rerank import Reranker
from .chroma_storage import ChromaStorage
import json
from datetime import datetime
import getpass
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from .config import PROCESSED_TEXTS_DIR, ROOT_DATA_PATH
from .document_processor import DocumentProcessor

class Retrieval :
    def __init__(self, 
             path_doc = ROOT_DATA_PATH, 
             chroma_persist_dir: str = "./chroma_db_local",
             processed_texts_dir: str = PROCESSED_TEXTS_DIR  
            ):
        
        self.vectorizor = Vectorizor()
        self.reranker = Reranker(enabled=True, alpha=0.5)  # moyenne pondérée 50/50
        self.chroma_storage = ChromaStorage(persist_directory=chroma_persist_dir)     
        self.path_doc = Path(path_doc)
        self.document_processor = DocumentProcessor(
            path_doc=self.path_doc,
            processed_texts_dir=Path(processed_texts_dir)
        )
        return

    
    def _vectorize_from_scratch(
            self, 
            chunk_size: int = 1000,      # ⚠️ Anciennement 200 MOTS, maintenant 1000 CARACTÈRES
            overlap: int = 200,    # ⚠️ Anciennement 0.15 (15%), maintenant 200 CARACTÈRES
            source_folder: str = None
        ):
        """
        Vectorise les documents depuis un dossier source.
        
        Args:
            chunk_size (int, optionnel): Taille des chunks en mots (défaut: 200)
            overlap (float, optionnel): Chevauchement entre chunks (défaut: 0.15)
            source_folder (str, optionnel): Dossier source des documents (défaut: self.path_doc)
        
        Returns:
            bool: True si succès
        """
        
        #  Si source_folder n'est pas spécifié, utiliser self.path_doc
        if source_folder is None:
            source_folder = str(self.path_doc)

        textes, chemins = self.document_processor.process_documents(source=source_folder)
        
        if not textes:
            print(f" Aucun document trouvé dans {source_folder}")

            return False
        
        df = self.decouper_en_batches(textes, chemins, chunk_size, overlap)
        
        print(f" Vectorisation de {len(df)} chunks (taille={chunk_size} caractères, overlap={overlap} caractères)")
             
        batch_size = 200
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size].copy()
            
            embeddings = self.vectorizor.encode(batch_df["batch"])
            
            for idx, (_, row) in enumerate(batch_df.iterrows()):
                self.chroma_storage.add_document(
                    document=row["batch"],
                    chemin=row["chemin"], 
                    embedding=embeddings.iloc[idx],
                    position_debut=row["position_debut"]
                )
            
            print(f" Batch {i//batch_size + 1}: {len(batch_df)} embeddings ajoutés à ChromaDB")
        
        return True
    
    def vectorize_with_config(
            
            self, chunk_size: int, 
            overlap: int, 
            collection_name: str, 
            source_folder: str = "code/base_test/DATA_Test", 
            model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            
            ) -> bool:
        
        """
    Vectorise les documents avec une configuration spécifique et les stocke
    dans une collection dédiée.
    
    Args:
        chunk_size (int): Taille des chunks en mots
        overlap (float): Chevauchement entre chunks (0-1)
        collection_name (str): Nom de la collection ChromaDB à créer/utiliser
        source_folder (str): Dossier source des documents

    Returns:
        bool: True si succès
        """


        print(f"\n Vectorisation avec config: chunk_size={chunk_size} caractères, overlap={overlap} caractères")
        print(f" Collection cible: {collection_name}")
        print(f" Modèle: {model_name}")
        print(f" Dossier source: {source_folder}")
        
        #  TRAÇABILITÉ
        metadata = {
        "chunk_size": chunk_size,
        "chunk_size_unit": "caractères",  # Clarifier l'unité
        
        "overlap": overlap,
        "overlap_unit": "caractères",      # 
        
        "splitter": "RecursiveCharacterTextSplitter",  #  Tracer le splitter
        "splitter_separators": json.dumps(["\n\n", "\n", ". ", " ", ""]),  # String JSON valide        
        "source_folder": source_folder,
        "created_at": datetime.now().isoformat(),
        "model": model_name,
        "reranking_enabled": self.reranker.enabled,
        "reranking_alpha": self.reranker.alpha,
        "created_by": getpass.getuser(),
        "version": "3.0"  #  Incrémenter car changement majeur
        }
        
        # Créer la collection avec métadonnées
        self.chroma_storage.create_collection_with_metadata(collection_name, metadata)

        # Charger le modèle correspondant
        print(f" Configuration du vectorizor pour {model_name}...")
        self.vectorizor._load_model(model_name)

        
        # POINT 4 : DÉTECTION DE CONFLITS
        if self.chroma_storage.count_documents() > 0:
            existing_metadata = self.chroma_storage.collection.metadata
            
            conflicts = []
            
            #  Comparer avec prise en compte de l'unité
            if existing_metadata.get("chunk_size") != chunk_size:
                old_unit = existing_metadata.get("chunk_size_unit", "mots")  # Ancien = mots
                conflicts.append(
                    f"chunk_size ({existing_metadata.get('chunk_size')} {old_unit} → {chunk_size} caractères)"
                )
            
            if existing_metadata.get("overlap") != overlap:
                old_unit = existing_metadata.get("overlap_unit", "%")
                conflicts.append(
                    f"overlap ({existing_metadata.get('overlap')} {old_unit} → {overlap} caractères)"
                )
            
            #  Détecter changement de splitter
            old_splitter = existing_metadata.get("splitter", "manuel")
            if old_splitter != "RecursiveCharacterTextSplitter":
                conflicts.append(
                    f"splitter ({old_splitter} → RecursiveCharacterTextSplitter)"
                )
            
            if conflicts:
                print(f"\n  CONFLIT DÉTECTÉ dans '{collection_name}' :")
                for conflict in conflicts:
                    print(f"   • {conflict}")

                print(f"\n   Collection existante : {self.chroma_storage.count_documents()} docs")
                print(f"   Créée le : {existing_metadata.get('created_at')}")
                print(f"   Créée par : {existing_metadata.get('created_by')}")
                
                choix = input("\n Voulez-vous ÉCRASER la collection ? (o/n) : ").strip().lower()
                
                if choix != 'o':
                    print(" Opération annulée, collection existante conservée")
                    return True
                
                # L'utilisateur veut écraser : supprimer et recréer
                print(" Suppression de l'ancienne collection...")
                self.chroma_storage.delete_collection()
                self.chroma_storage.create_collection_with_metadata(collection_name, metadata)
                # Continuer vers la vectorisation ci-dessous
            else:
                # Pas de conflit, paramètres identiques
                print(f"Collection existante avec paramètres identiques ({self.chroma_storage.count_documents()} docs)")
                return True
        
        #  VECTORISER (si collection vide OU si on a écrasé)
        print("\n Début de la vectorisation...")
        return self._vectorize_from_scratch(
            chunk_size=chunk_size,
            overlap=overlap,
            source_folder=source_folder
        )

    def clone_collection(self, source_collection: str, new_collection_name: str):
        """
        REPRODUCTIBILITÉ
        Clone une collection existante avec ses paramètres exacts.
        
        Args:
            source_collection (str): Nom de la collection à cloner
            new_collection_name (str): Nom de la nouvelle collection

        Fonctionnement (code d'exemple) :
        
        rag = retrival()
        # Cloner config_150_15 (la meilleure) vers une nouvelle collection
        rag.clone_collection("config_150_15", "config_optimale_v2")

        """
        # Récupérer la collection source
        client = self.chroma_storage.chroma_client
        source_col = client.get_collection(source_collection)
        
        # Récupérer les métadonnées
        source_metadata = source_col.metadata
        
        print(f" Clonage de '{source_collection}' → '{new_collection_name}'")
        print(f"   Paramètres source :")
        print(f"   • chunk_size: {source_metadata.get('chunk_size')}")
        print(f"   • overlap: {source_metadata.get('overlap')}")
        print(f"   • source_folder: {source_metadata.get('source_folder')}")
        
        # Vectoriser avec les mêmes paramètres
        return self.vectorize_with_config(
            chunk_size=source_metadata.get('chunk_size', 150),
            overlap=source_metadata.get('overlap', 0.15),
            collection_name=new_collection_name,
            source_folder=source_metadata.get('source_folder', str(self.path_doc))
        )



    def fusionner_checkpoints(self):

        """
     MÉTHODE DE MIGRATION UNIQUEMENT
    Utilisée une seule fois pour migrer les anciens checkpoints JSON vers ChromaDB.
    Sera automatiquement appelée par vectorize_folder() si nécessaire.
        """        
        
        checkpoint_files = glob.glob(self.checkpoint_glob)
        
        # Tri numérique correct
        def extract_number(filename):
            match = re.search(r'checkpoint_(\d+)\.json', filename)
            return int(match.group(1)) if match else 0
        
        checkpoint_files.sort(key=extract_number)
        
        if not checkpoint_files:
            print("Aucun fichier checkpoint trouvé")
            return
        
        print(f"Fusion streaming de {len(checkpoint_files)} fichiers...")
        
        final_file = str(self.base_json)
        total_count = 0
        
        # Écriture directe streaming vers le fichier final
        with open(final_file, 'w') as outfile:
            outfile.write('[')  # Début array JSON
            
            first_item = True
            
            for i, file_path in enumerate(checkpoint_files):
                if i % 100 == 0:  # Affichage moins fréquent
                    print(f"Traitement {i+1}/{len(checkpoint_files)}: {file_path}")
                
                try:
                    # Charger UN SEUL fichier à la fois
                    with open(file_path, 'r') as infile:
                        data = json.load(infile)
                    
                    # Écrire chaque item directement dans le fichier final
                    for item in data:
                        if not first_item:
                            outfile.write(',')
                        
                        # Écriture compacte sans indent
                        json.dump(item, outfile, separators=(',', ':'))
                        total_count += 1
                        first_item = False
                        
                        # Affichage périodique du progrès
                        if total_count % 10000 == 0:
                            print(f"   {total_count} embeddings traités...")
                    
                except Exception as e:
                    print(f" Erreur avec {file_path}: {e}")
                    continue
            
            outfile.write(']')  # Fin array JSON
        
        print(f" Fusion streaming terminée ! {total_count} embeddings dans {final_file}")
        print(f" Taille du fichier: {Path(final_file).stat().st_size / 1024 / 1024:.1f} MB")

    
    def query(self, query, n):
  
        try:
            # Adapter le modèle à la collection active
            collection_metadata = self.chroma_storage.collection.metadata
            self.vectorizor.switch_to_model_for_collection(collection_metadata)
            
            # 1. Génération de l'embedding de la requête
            query_embeddings = self.vectorizor.encode_query(query)
            
            # 2. Recherche dans ChromaDB
            contexts, sources, scores = self.chroma_storage.query_similar(
                query_embeddings, n_results=n
            )
            
            # 3. Si reranking activé ET plusieurs résultats, l'appliquer
            if self.reranker.enabled and len(contexts) > 1:
                candidates = []
                for i, (ctx, src, score) in enumerate(zip(contexts, sources, scores)):
                    candidates.append({
                        "batch": ctx,
                        "chemin": src,
                        "score_retrieval": score,
                    })

                # Appel du reranker
                ranked = self.reranker.rescore(query, candidates)

                # Extraction des résultats rerankés
                contexts = [c["batch"] for c in ranked]
                sources = [c["chemin"] for c in ranked]  
                scores = [c["score_final"] for c in ranked]
            
            # 4. Retour des résultats (avec ou sans reranking)
            return contexts, sources, scores
            
        except Exception as e:
            print(f" Erreur de requête : {e}")
            return [], [], []
        
    def add_documents(self, collection_name: str, source_path: str, overwrite_duplicates: bool = False) -> bool:
        """
        Ajoute des documents (fichier ou dossier) à une collection, en gérant les doublons.
        C'est la méthode métier qui orchestre tout le processus.

        Args:
            collection_name (str): Nom de la collection cible.
            source_path (str): Chemin vers le fichier ou le dossier à ajouter.
            overwrite_duplicates (bool): Si True, les fichiers déjà existants seront remplacés.
                                         Si False, ils seront ignorés.

        Returns:
            bool: True si l'opération a réussi.
        """
        print(f"\n[Retriever] Ajout de documents depuis '{source_path}' vers '{collection_name}'...")
        
        try:
            # 1. Préparation : se connecter à la bonne collection et aligner le modèle
            self.chroma_storage.switch_collection(collection_name)
            collection_metadata = self.chroma_storage.collection.metadata
            self.vectorizor.switch_to_model_for_collection(collection_metadata)
            
            # 2. Lister les fichiers à ajouter
            path = Path(source_path)
            if path.is_file():
                files_to_add = [str(path)]
            else:
                # Recherche récursive de tous les types de fichiers supportés
                files_to_add = [str(f) for f in path.rglob("*") if f.suffix.lower() in self.document_processor._handlers]

            if not files_to_add:
                print("  Aucun fichier supporté trouvé dans le chemin spécifié.")
                return True

            # 3. Gérer les doublons
            print("  Vérification des doublons...")
            stats = self.chroma_storage.get_stats()
            # existing_files contient déjà des chemins relatifs et propres
            existing_files = {s['chemin'] for s in stats.get('sources_summary', [])}
            
            # On résout le chemin racine une seule fois pour la performance
            root_path_resolved = self.path_doc.resolve()

            duplicates = [
                f for f in files_to_add 
                if Path(f).resolve().relative_to(root_path_resolved).as_posix() in existing_files
            ]
            
            if duplicates:
                print(f"  {len(duplicates)} doublon(s) détecté(s).")
                if overwrite_duplicates:
                    print("  Option 'Remplacer' activée : suppression des anciennes versions...")
                    for dup_path in duplicates:
                        # On convertit le chemin en format relatif à la base de données
                        relative_dup_path = Path(dup_path).resolve().relative_to(root_path_resolved).as_posix()
                        self.chroma_storage.delete_by_source(relative_dup_path)
                else:
                    print("  Option 'Ignorer' activée : les doublons ne seront pas traités.")
                    # On filtre la liste pour ne garder que les nouveaux fichiers
                    files_to_add = [
                        f for f in files_to_add 
                        if Path(f).resolve().relative_to(root_path_resolved).as_posix() not in existing_files
                    ]

            if not files_to_add:
                print("  Aucun nouveau fichier à traiter.")
                return True

            # 4. Lancer le traitement et la vectorisation
            print(f"  Traitement de {len(files_to_add)} fichier(s)...")
            chunk_size = collection_metadata.get("chunk_size", 1000)
            overlap = collection_metadata.get("overlap", 200)

            # A. Extraction des textes (utilise le cache de DocumentProcessor)
            textes, chemins = self.document_processor.process_documents(fichiers_specifiques=files_to_add)
            if not textes:
                print("  Aucun texte n'a pu être extrait des fichiers.")
                return False

            # B. Découpage en chunks
            df = self.decouper_en_batches(textes, chemins, chunk_size, overlap)
            if df.empty:
                print("  Aucun chunk généré.")
                return True

            # C. Vectorisation et ajout à ChromaDB
            print(f"  Vectorisation et ajout de {len(df)} chunks...")
            batch_size = 200
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size].copy()
                embeddings = self.vectorizor.encode(batch_df["batch"])
                for idx, (_, row) in enumerate(batch_df.iterrows()):
                    self.chroma_storage.add_document(
                        document=row["batch"],
                        chemin=row["chemin"],
                        embedding=embeddings.iloc[idx],
                        position_debut=row["position_debut"]
                    )
                print(f"    Batch {i//batch_size + 1}: {len(batch_df)} chunks ajoutés.")

            print("\n✓ [Retriever] Ajout de documents terminé avec succès.")
            return True

        except Exception as e:
            print(f"\n✗ [Retriever] Erreur critique lors de l'ajout de documents : {e}")
            import traceback
            traceback.print_exc()
            return False


    def get_stats(self):
            """
            NOUVELLE MÉTHODE - Statistiques ChromaDB
            """
            return self.chroma_storage.get_stats()

    
    def decouper_en_batches(
        self,
        textes: list[str],
        chemins: list[str],
        chunk_size: int = 1000,        # EN CARACTÈRES maintenant
        overlap: int = 200       # EN CARACTÈRES aussi
    ) -> pd.DataFrame:
        """
        Découpe avec RecursiveCharacterTextSplitter de LangChain.
        
        Args:
            chunk_size (int): Taille en CARACTÈRES (pas en mots)
            chunk_overlap (int): Chevauchement en CARACTÈRES
        """
        
        # Séparateurs optimisés pour le français [[1]]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # Priorité aux paragraphes
            length_function=len,
            is_separator_regex=False
        )
        
        data = []
        
        for texte, chemin in zip(textes, chemins):
            # Découpe avec LangChain [[2]]
            chunks = text_splitter.split_text(texte)
            
            # Reconstituer position_debut pour compatibilité
            position = 0
            for chunk in chunks:
                data.append({
                    "batch": chunk,
                    "chemin": chemin,
                    "position_debut": position
                })
                position += len(chunk)  # Position en caractères maintenant
        
        return pd.DataFrame(data)
    
    