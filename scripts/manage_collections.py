import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent  # Remonte de launchers/ à RAG_CGT/
sys.path.insert(0, str(ROOT_DIR))

import uuid
import time
import re
from src.rag.config import ROOT_DATA_PATH
from src.rag import Retrieval, Vectorizor


def add_collection_interactive(client):
    """
    Crée une nouvelle collection de manière interactive avec métadonnées.
    Combine le menu interactif de add_new_domain.py avec les métadonnées de test_metadata.py.

    Args:
        client: Client ChromaDB
    """

    print("\n" + "=" * 100)
    print(" CRÉATION D'UNE NOUVELLE COLLECTION")
    print("=" * 100)

    # Instance retrieval
    r = Retrieval()

    # 1. Afficher les collections existantes
    collections = r.chroma_storage.list_collection_names()

    if collections:
        print(f"\n Collections existantes ({len(collections)}) :")
        for col_name in collections:
            col = client.get_collection(col_name)
            count = col.count()
            metadata = col.metadata
            chunk_size = metadata.get("chunk_size", "N/A")
            overlap = metadata.get("overlap")
            model_name = metadata.get("model", "N/A")

            print(f"   • {col_name} : {count} docs", end="")
            if chunk_size != "N/A":
                overlap_str = f"{overlap} caractères" if overlap else "N/A"
                print(f" ({chunk_size} caractères, {overlap_str} overlap)")
            else:
                print()
    else:
        print("\n Aucune collection existante")

    # 2. Demander le nom de la nouvelle collection
    print("\n" + "=" * 100)
    nom_collection = input(" Nom de la nouvelle collection : ").strip()

    # Règle 1 : Vérifier la longueur
    if not 3 <= len(nom_collection) <= 512:
        print(
            f"\n Erreur : Le nom de la collection doit contenir entre 3 et 512 caractères. (Vous avez entré {len(nom_collection)} caractères)"
        )
        return False

    # Règle 2 : Vérifier les caractères autorisés et les caractères de début/fin
    # On utilise une expression régulière pour valider le nom
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$", nom_collection):
        # On vérifie aussi le cas d'un nom court de 3 caractères qui serait valide
        if not re.match(r"^[a-zA-Z0-9]{3}$", nom_collection):
            print(
                "\n Erreur : Le nom de la collection contient des caractères invalides ou ne commence/finit pas par une lettre/chiffre."
            )
            print("   Caractères autorisés : a-z, A-Z, 0-9, ., _, -")
            return False

    # Vérifier si le nom existe déjà
    if nom_collection in collections:
        print(
            f" Une collection '{nom_collection}' existe déjà ({client.get_collection(nom_collection).count()} docs)"
        )
        choix = input("Voulez-vous l'écraser ? (o/n) : ").strip().lower()
        if choix != "o":
            print(" Opération annulée")
            return False

    #  3. NOUVEAU : Choix du modèle d'embedding
    print("\n Modèle d'embedding :")
    print("   1. Qwen3-Embedding-0.6B (1024 dim, qualité maximale, lent)")
    print("   2. MPNet (768 dim, bon compromis qualité/vitesse)")
    print("   3. MiniLM (384 dim, rapide mais qualité inférieure)")

    choix_modele = input("\nVotre choix (1-3, défaut: 2) : ").strip()

    model_map = {
        "1": "Qwen/Qwen3-Embedding-0.6B",
        "2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "3": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    }

    model_name = model_map.get(choix_modele, model_map["2"])  # Défaut = MPNet
    model_short = {
        "Qwen/Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "paraphrase-multilingual-mpnet-base-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "paraphrase-multilingual-MiniLM-L12-v2",
    }[model_name]

    print(f"    Modèle sélectionné : {model_short}")

    # 3. Demander les paramètres de chunking
    print("\n Paramètres de chunking :")

    chunk_size_input = input(
        "   Taille des chunks EN CARACTÈRES (défaut: 1000) : "
    ).strip()
    chunk_size = int(chunk_size_input) if chunk_size_input else 1000

    overlap_input = input("   Chevauchement EN CARACTÈRES (défaut: 200) : ").strip()
    overlap = int(overlap_input) if overlap_input else 200

    #  Affichage mis à jour
    print(f"   • Paramètres : {chunk_size} caractères, {overlap} caractères overlap")

    # 4. Demander le chemin source
    print("\n Source des documents :")
    chemin_source = input(
        "   Chemin vers le dossier (ex: data/raw/DATA_Test) : "
    ).strip()

    if not chemin_source:
        print(" Chemin invalide")
        return False

    # Vérifier que le dossier existe
    if not Path(chemin_source).exists():
        print(f" Le dossier '{chemin_source}' n'existe pas")
        print(f"   Répertoire courant : {Path.cwd()}")
        return False

    # 5. Confirmation avant création
    print("\n" + "=" * 100)
    print(" RÉCAPITULATIF :")
    print(f"   • Nom : {nom_collection}")
    print(f"   • Paramètres : {chunk_size} caractères, {overlap} caractères overlap")
    print(f"   • Source : {chemin_source}")
    print("=" * 100)

    confirmation = input("\n Confirmer la création ? (o/n) : ").strip().lower()

    if confirmation != "o":
        print(" Création annulée")
        return False

    #  7. MODIFIER vectorizor.py temporairement
    print(f"\n Configuration du vectorizor pour {model_short}...")
    r.vectorizor = Vectorizor()  # Recharger avec le modèle par défaut
    r.vectorizor._load_model(model_name)  # Charger le modèle choisi

    # 6. Lancer la vectorisation avec métadonnées
    print("\n Vectorisation en cours...\n")

    try:
        success = r.vectorize_with_config(
            chunk_size=chunk_size,
            overlap=overlap,
            collection_name=nom_collection,
            source_folder=chemin_source,
            model_name=model_name,
        )

        if success:
            print(f"\n Collection '{nom_collection}' créée avec succès !")

            # Afficher les métadonnées créées
            col = client.get_collection(nom_collection)
            print("\n Métadonnées stockées :")
            print("=" * 100)
            for key, value in col.metadata.items():
                print(f"   • {key:<20} : {value}")

            # Afficher les stats
            r.chroma_storage.switch_collection(nom_collection)
            stats = r.chroma_storage.get_stats()
            print(f"\n Statistiques :")
            print(f"   • Total chunks : {stats['total_documents']}")
            print(f"   • Total fichiers : {stats['total_fichiers']}")

            return True
        else:
            print(f"\n Échec de la création de '{nom_collection}'")
            return False

    except Exception as e:
        print(f"\n Erreur lors de la création : {e}")
        return False


def rename_collection(old_name: str, new_name: str, client) -> bool:
    """
    "Renomme" une collection en copiant ses documents vers une nouvelle collection.

     LIMITATION ChromaDB : Pas de renommage natif, on doit copier puis supprimer.

    Args:
        old_name (str): Nom actuel de la collection
        new_name (str): Nouveau nom souhaité
        client: Client ChromaDB

    Returns:
        bool: True si succès
    """

    print("\n" + "=" * 100)
    print(f" PSEUDO-RENOMMAGE : '{old_name}' → '{new_name}'")
    print("=" * 100)

    # 1. Vérifier que l'ancienne collection existe
    try:
        old_col = client.get_collection(old_name)
    except Exception as e:
        print(f" Collection '{old_name}' introuvable : {e}")
        return False

    old_count = old_col.count()
    old_metadata = old_col.metadata

    print(f"\n Collection source :")
    print(f"   Nom : {old_name}")
    print(f"   Documents : {old_count}")

    if old_metadata.get("chunk_size"):
        overlap = old_metadata.get("overlap")
        overlap_str = f"{overlap} caractères" if overlap else "N/A"
        print(
            f"   Paramètres : {old_metadata.get('chunk_size')} mots, {overlap_str} overlap"
        )

    if old_count == 0:
        print("\n Collection vide, renommage simple...")
        try:
            client.delete_collection(old_name)
            client.create_collection(name=new_name, metadata=old_metadata)
            print(f" Collection vide renommée : '{old_name}' → '{new_name}'")
            return True
        except Exception as e:
            print(f" Erreur : {e}")
            return False

    # 2. Vérifier que le nouveau nom n'existe pas
    try:
        existing = client.get_collection(new_name)
        print(f"\n ATTENTION : '{new_name}' existe déjà ({existing.count()} docs)")
        choix = input("Voulez-vous l'écraser ? (o/n) : ").strip().lower()
        if choix != "o":
            print(" Opération annulée")
            return False
        client.delete_collection(new_name)
    except:
        pass  # Le nouveau nom n'existe pas (OK)

    # 3. Créer la nouvelle collection avec les mêmes métadonnées
    print(f"\n Création de '{new_name}'...")
    new_col = client.create_collection(name=new_name, metadata=old_metadata)

    # 4. Copier TOUS les documents par batch
    print(f"\n Copie de {old_count} documents...")

    batch_size = 500
    total_copied = 0

    for i in range(0, old_count, batch_size):
        try:
            # Récupérer un batch
            results = old_col.get(
                limit=batch_size,
                offset=i,
                include=["documents", "metadatas", "embeddings"],
            )

            if not results or not results["documents"]:
                break

            # Générer de nouveaux IDs
            new_ids = [str(uuid.uuid4()) for _ in results["documents"]]

            # Ajouter à la nouvelle collection
            new_col.add(
                documents=results["documents"],
                metadatas=results["metadatas"],
                embeddings=results["embeddings"],
                ids=new_ids,
            )

            total_copied += len(results["documents"])
            progress = (total_copied / old_count) * 100
            print(
                f"    Batch {i // batch_size + 1}: {total_copied}/{old_count} docs ({progress:.1f}%)"
            )

        except Exception as e:
            print(f"    Erreur batch {i // batch_size + 1} : {e}")
            continue

    # 5. Vérifier le résultat
    new_count = new_col.count()

    print(f"\n Vérification :")
    print(f"   Ancienne : {old_count} docs")
    print(f"   Nouvelle : {new_count} docs")

    if new_count != old_count:
        print(f"\n DIFFÉRENCE de {abs(new_count - old_count)} documents !")
        choix = (
            input("Supprimer quand même l'ancienne collection ? (o/n) : ")
            .strip()
            .lower()
        )
        if choix != "o":
            print(" Ancienne collection CONSERVÉE")
            return False

    # 6. Supprimer l'ancienne collection
    print(f"\n Suppression de '{old_name}'...")
    try:
        client.delete_collection(old_name)
        print(f" Renommage terminé : '{old_name}' → '{new_name}'")
        return True
    except Exception as e:
        print(f" Erreur suppression : {e}")
        return False


def add_documents_to_collection_interactive(r: Retrieval, client):
    """
    Menu interactif pour ajouter de nouveaux documents à une collection existante.
    Délègue la logique métier à la classe retrieval.
    """
    print("\n" + "=" * 100)
    print(" AJOUT DE DOCUMENTS À UNE COLLECTION EXISTANTE")
    print("=" * 100)

    # 1. Sélection de la collection (logique d'interface)
    collections = r.chroma_storage.list_collection_names()
    if not collections:
        print("\n Aucune collection disponible.")
        return False

    choix = input(" Nom ou numéro de la collection : ").strip()
    if choix.isdigit():
        # ...
        collection_name = collections[int(choix) - 1]
    else:
        collection_name = choix

    if collection_name not in collections:
        print(f" Collection '{collection_name}' introuvable.")
        return False

    # 2. Demander le chemin source (logique d'interface)
    chemin_source = input(
        "\n Chemin vers le fichier ou le dossier à ajouter : "
    ).strip()
    if not chemin_source or not Path(chemin_source).exists():
        print(f" Chemin invalide ou inexistant : {chemin_source}")
        return False

    # 3. Gérer l'interaction pour les doublons (logique d'interface)
    # On vérifie les doublons ici UNIQUEMENT pour pouvoir poser la question à l'utilisateur.
    print("\n Vérification des doublons...")
    r.chroma_storage.switch_collection(collection_name)
    stats = r.chroma_storage.get_stats()
    existing_files = {s["chemin"] for s in stats.get("sources_summary", [])}

    path = Path(chemin_source)
    if path.is_file():
        files_to_check = [str(path)]
    else:
        files_to_check = [
            str(f)
            for f in path.rglob("*")
            if f.suffix.lower() in r.document_processor._handlers
        ]

    duplicates = [
        f
        for f in files_to_check
        if Path(f).resolve().relative_to(r.path_doc.resolve()).as_posix()
        in existing_files
    ]
    overwrite_flag = False  # Par défaut, on ignore les doublons

    if duplicates:
        print(f"\n {len(duplicates)} doublon(s) détecté(s).")
        print(
            "   1. Remplacer les doublons (supprimer les anciens + ajouter les nouveaux)"
        )
        print(
            "   2. Ignorer les doublons (ajouter uniquement les nouveaux fichiers) [Défaut]"
        )
        print("   3. Annuler l'opération")

        choix_doublon = input("\nVotre choix (1-3) : ").strip()

        if choix_doublon == "1":
            overwrite_flag = True
        elif choix_doublon == "3":
            print(" Opération annulée.")
            return False
        # Si 2 ou autre, on garde overwrite_flag = False

    # 4. Appel de la méthode métier (logique déléguée)
    print("\n" + "=" * 100)
    print(" Lancement du processus d'ajout...")
    print("=" * 100)

    # On passe simplement la commande au "maître d'hôtel"
    success = r.add_documents(
        collection_name=collection_name,
        source_path=chemin_source,
        overwrite_duplicates=overwrite_flag,
    )

    # 5. Afficher le résultat (logique d'interface)
    if success:
        print("\n✓ Opération d'ajout terminée avec succès.")
        # Afficher les nouvelles stats
        r.chroma_storage.switch_collection(collection_name)
        final_stats = r.chroma_storage.get_stats()
        print(
            f"  Statistiques de '{collection_name}' : {final_stats['total_documents']} chunks, {final_stats['total_fichiers']} fichiers."
        )
    else:
        print(
            "\n✗ Échec de l'opération d'ajout. Consultez les logs pour plus de détails."
        )

    return success


def search_sources_by_keyword(
    sources: list[dict], keyword: str, max_results: int = 10
) -> list[dict]:
    """
    Recherche fuzzy dans les noms de fichiers.

    Args:
        sources: Liste retournée par get_sources_with_stats()
        keyword: Mot-clé de recherche
        max_results: Nombre max de résultats

    Returns:
        list[dict]: Sources correspondantes, triées par pertinence
    """
    from difflib import SequenceMatcher

    keyword_lower = keyword.lower()

    # Calculer un score de similarité pour chaque source
    scored_sources = []
    for source in sources:
        filename_lower = source["filename"].lower()

        # Score 1 : Correspondance exacte (substring)
        if keyword_lower in filename_lower:
            score_exact = 1.0
        else:
            score_exact = 0.0

        # Score 2 : Similarité fuzzy (Levenshtein-like)
        score_fuzzy = SequenceMatcher(None, keyword_lower, filename_lower).ratio()

        # Score final : pondération 70% exact, 30% fuzzy
        score_final = (score_exact * 0.7) + (score_fuzzy * 0.3)

        if score_final > 0.2:  # Seuil minimum
            scored_sources.append({**source, "score": score_final})

    # Trier par score décroissant
    scored_sources.sort(key=lambda x: -x["score"])

    return scored_sources[:max_results]


def display_sources_paginated(sources: list[dict], page_size: int = 20) -> str:
    """
    Affiche les sources de manière paginée.

    Returns:
        str: Chemin du fichier sélectionné ou None
    """

    total_pages = (len(sources) + page_size - 1) // page_size
    current_page = 0

    while True:
        # Calculer les indices
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, len(sources))
        page_sources = sources[start_idx:end_idx]

        # Affichage
        print("\n" + "=" * 100)
        print(f" SOURCES (Page {current_page + 1}/{total_pages})")
        print("=" * 100)

        for i, source in enumerate(page_sources, start=start_idx + 1):
            print(f"   [{i}] {source['filename']}")
            print(f"       Chemin : {source['chemin']}")
            print(f"       Chunks : {source['nb_chunks']}")
            print()

        # Menu navigation
        print("=" * 100)
        print("Actions :")
        print("   • Tapez un numéro pour sélectionner")
        if current_page < total_pages - 1:
            print("   • 'n' : Page suivante")
        if current_page > 0:
            print("   • 'p' : Page précédente")
        print("   • 'r' : Rechercher")
        print("   • 'q' : Retour")

        choix = input("\nVotre choix : ").strip()

        if choix.lower() == "q":
            return None
        elif choix.lower() == "n" and current_page < total_pages - 1:
            current_page += 1
        elif choix.lower() == "p" and current_page > 0:
            current_page -= 1
        elif choix.lower() == "r":
            keyword = input(" Mot-clé : ").strip()
            if keyword:
                results = search_sources_by_keyword(sources, keyword)
                if results:
                    # Afficher résultats de recherche (logique similaire)
                    pass
        elif choix.isdigit():
            num = int(choix)
            if 1 <= num <= len(sources):
                return sources[num - 1]["chemin"]
            else:
                print(f" Numéro {num} invalide")


def delete_source_interactive(client):
    """
    Menu interactif pour supprimer un document source.
    Détecte automatiquement la taille de la collection.
    """

    print("\n" + "=" * 100)
    print(" SUPPRESSION DE DOCUMENTS SOURCES")
    print("=" * 100)

    r = Retrieval()

    # 1. Lister les collections disponibles
    collections = r.chroma_storage.list_collection_names()

    if not collections:
        print("\n Aucune collection disponible")
        return False

    print(f"\n {len(collections)} collection(s) disponible(s) :\n")

    # Afficher chaque collection
    for i, col_name in enumerate(collections, 1):
        col = client.get_collection(col_name)
        count = col.count()
        metadata = col.metadata

        chunk_size = metadata.get("chunk_size", "N/A")
        overlap = metadata.get("overlap")

        print(f"   [{i}] {col_name}")
        print(f"       Documents : {count}")

        if chunk_size != "N/A":
            overlap_str = f"{overlap} caractères" if overlap else "N/A"
            print(f"       Paramètres : {chunk_size} caractères, {overlap_str} overlap")

        print()

    #  SÉLECTION DE LA COLLECTION (code complet)
    print("=" * 100)
    choix = input(" Nom ou numéro de la collection : ").strip()

    # Parser le choix (numéro ou nom)
    if choix.isdigit():
        num = int(choix)
        if 1 <= num <= len(collections):
            collection_name = collections[
                num - 1
            ]  # ← VOILÀ collection_name est défini !
        else:
            print(f" Numéro {num} hors limite")
            return False
    elif choix in collections:
        collection_name = choix  # ← OU ICI collection_name est défini !
    else:
        print(f"Collection '{choix}' introuvable")
        return False

    #  MAINTENANT collection_name est défini, on récupère l'objet collection
    collection = client.get_collection(collection_name)

    # 2. Récupérer les sources avec stats
    print("\n Analyse de la collection...")

    r.chroma_storage.switch_collection(collection_name)
    stats = r.chroma_storage.get_stats()

    sources = stats["sources_summary"]
    total_chunks = stats["total_documents"]
    total_sources = stats["total_fichiers"]

    print(f"\n Collection : {collection_name}")
    print(f"   • {total_chunks} chunks")
    print(f"   • {total_sources} fichiers sources")

    # 3. DÉTECTION AUTOMATIQUE DU MODE
    if total_sources < 50:
        # PETITE COLLECTION : Liste complète
        print("\n Collection de taille raisonnable, affichage de la liste complète")
        chemin_a_supprimer = display_sources_paginated(sources)

    elif total_sources < 200:
        # COLLECTION MOYENNE : Hybride
        print("\n Collection moyennement grande")
        print("Options :")
        print("   1. Rechercher par mot-clé (recommandé)")
        print("   2. Afficher la liste complète (paginée)")

        choix = input("\nVotre choix (1-2) : ").strip()

        if choix == "1":
            keyword = input("\n Mot-clé : ").strip()
            results = search_sources_by_keyword(sources, keyword)
            # (Afficher résultats et sélection)
        else:
            chemin_a_supprimer = display_sources_paginated(sources)

    else:
        # GRANDE COLLECTION : Recherche obligatoire
        print("\nCollection GRANDE (>200 fichiers), recherche obligatoire")
        keyword = input("Mot-clé : ").strip()

        results = search_sources_by_keyword(sources, keyword)

        if not results:
            print(" Aucun résultat")
            return False

        print(f"\n {len(results)} résultat(s) trouvé(s) :\n")
        for i, result in enumerate(results, 1):
            print(f"   [{i}] {result['filename']}")
            print(f"       Chemin : {result['chemin']}")
            print(f"       Chunks : {result['nb_chunks']}")
            print(f"       Score : {result['score']:.2f}")
            print()

        choix = input("Numéro du fichier (0 pour annuler) : ").strip()

        if not choix.isdigit() or int(choix) == 0:
            return False

        num = int(choix)
        if 1 <= num <= len(results):
            chemin_a_supprimer = results[num - 1]["chemin"]
        else:
            print(" Numéro invalide")
            return False

    # 4. CONFIRMATION avec preview
    source_info = next((s for s in sources if s["chemin"] == chemin_a_supprimer), None)

    if not source_info:
        print(" Erreur interne")
        return False

    print("\n" + "=" * 100)
    print(" CONFIRMATION DE SUPPRESSION")
    print("=" * 100)
    print(f"\n Fichier : {source_info['filename']}")
    print(f"   Chemin complet : {source_info['chemin']}")
    print(f"   Nombre de chunks : {source_info['nb_chunks']}")
    print(f"   Taille estimée : ~{source_info['nb_chunks'] * 0.5:.1f} KB")
    print("\nCette action est IRRÉVERSIBLE")

    confirmation = input("\nConfirmer la suppression ? (tapez 'oui') : ").strip()

    if confirmation != "oui":
        print("Suppression annulée")
        return False

    # 5. SUPPRESSION
    print("\nSuppression en cours...")

    try:
        success = r.chroma_storage.delete_by_source(chemin_a_supprimer)

        if success:
            new_count = collection.count()
            deleted = total_chunks - new_count

            print(f"\n Suppression réussie !")
            print(f"   • Chunks supprimés : {deleted}")
            print(f"   • Collection : {total_chunks} → {new_count} chunks")
            return True
        else:
            print("\n Échec de la suppression")
            return False

    except Exception as e:
        print(f"\nErreur : {e}")
        return False


def batch_create_collections(client):
    """
    Crée plusieurs collections successivement avec configuration interactive.
     NOUVEAU : Demande TOUTES les infos AVANT le lancement de la première collection.

    Args:
        client: Client ChromaDB
    """

    print("\n" + "=" * 100)
    print(" CRÉATION EN BATCH DE COLLECTIONS")
    print("=" * 100)
    print("\nCe module vous permet de créer plusieurs collections successivement")
    print("avec des paramètres différents pour chaque collection.\n")

    # Instance retrieval
    r = Retrieval()

    # ===== ÉTAPE 1 : DEMANDER LE NOMBRE DE COLLECTIONS =====
    print("=" * 100)
    nb_collections_input = input(
        " Combien de collections voulez-vous créer ? : "
    ).strip()

    try:
        nb_collections = int(nb_collections_input)
        if nb_collections <= 0:
            print(" Nombre invalide")
            return
    except ValueError:
        print(" Veuillez entrer un nombre valide")
        return

    print(f"\n Vous allez configurer {nb_collections} collection(s)\n")

    # ===== ÉTAPE 2 : COLLECTER TOUTES LES CONFIGURATIONS =====
    collections_configs = []

    for i in range(nb_collections):
        print("\n" + "=" * 100)
        print(f" CONFIGURATION COLLECTION #{i + 1}/{nb_collections}")
        print("=" * 100)

        # ----- Nom -----
        nom_collection = input(f"\n Nom de la collection #{i + 1} : ").strip()

        if not nom_collection:
            print(" Nom invalide, collection ignorée")
            continue

        # Vérifier doublons
        collections = r.chroma_storage.list_collection_names()

        if nom_collection in collections:
            print(f"  Une collection '{nom_collection}' existe déjà")
            choix = input("Voulez-vous l'écraser ? (o/n) : ").strip().lower()
            if choix != "o":
                print(" Nom déjà utilisé, collection ignorée")
                continue
            ecraser = True
        else:
            ecraser = False

        # ----- Source -----
        print(f"\n Base de données source pour '{nom_collection}' :")
        print("   Exemples :")
        print("   • data/raw/DATA_Test")
        print("   • data/raw/DATA_Production")

        source_folder = input("\n   Chemin : ").strip()

        if not source_folder or not Path(source_folder).exists():
            print(f"Le dossier '{source_folder}' n'existe pas")
            continue

        # Compter les fichiers
        source_path = Path(source_folder)
        nb_pdf = len(list(source_path.rglob("*.pdf")))
        nb_txt = len(list(source_path.rglob("*.txt")))
        nb_docx = len(list(source_path.rglob("*.docx")))
        nb_doc = len(list(source_path.rglob("*.doc")))
        nb_md = len(list(source_path.rglob("*.md"))) + len(
            list(source_path.rglob("*.markdown"))
        )
        total_files = nb_pdf + nb_txt + nb_docx + nb_doc + nb_md

        print(f"\n    Source valide : {total_files} fichier(s) détecté(s)")

        # ----- Modèle -----
        print(f"\n Modèle d'embedding pour '{nom_collection}' :")
        print("   1. Qwen3-Embedding-0.6B (1024 dim, qualité maximale, lent)")
        print("   2. MPNet (768 dim, bon compromis qualité/vitesse) [DÉFAUT]")
        print("   3. MiniLM (384 dim, rapide mais qualité inférieure)")

        choix_modele = input("\n   Votre choix (1-3, défaut: 2) : ").strip()

        model_map = {
            "1": "Qwen/Qwen3-Embedding-0.6B",
            "2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "3": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        }

        model_name = model_map.get(choix_modele, model_map["2"])

        model_short = {
            "Qwen/Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "MPNet",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "MiniLM",
        }[model_name]

        print(f"    Modèle : {model_short}")

        # ----- Chunking -----
        print(f"\n Paramètres de chunking pour '{nom_collection}' :")

        chunk_size_input = input(
            "   Taille des chunks EN CARACTÈRES (défaut: 1000) : "
        ).strip()
        chunk_size = int(chunk_size_input) if chunk_size_input else 1000

        overlap_input = input("   Chevauchement EN CARACTÈRES (défaut: 200) : ").strip()
        overlap = int(overlap_input) if overlap_input else 200

        print(f"    Paramètres : {chunk_size} caractères, {overlap} caractères overlap")

        # ----- Stocker la configuration -----
        collections_configs.append(
            {
                "nom": nom_collection,
                "source": source_folder,
                "total_files": total_files,
                "model_name": model_name,
                "model_short": model_short,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "ecraser": ecraser,
            }
        )

        print(f"\n Configuration #{i + 1} enregistrée")

    # ===== ÉTAPE 3 : RÉCAPITULATIF COMPLET =====
    if not collections_configs:
        print("\n Aucune collection configurée")
        return

    print("\n" + "=" * 100)
    print(f"RÉCAPITULATIF COMPLET - {len(collections_configs)} COLLECTION(S)")
    print("=" * 100)

    for i, config in enumerate(collections_configs, 1):
        print(f"\n Collection #{i} : {config['nom']}")
        print(f"   • Source : {config['source']} ({config['total_files']} fichiers)")
        print(f"   • Modèle : {config['model_short']}")
        print(
            f"   • Chunking : {config['chunk_size']} caractères, {config['overlap']} caractères overlap"
        )
        if config["ecraser"]:
            print(f"     ÉCRASERA la collection existante")

    # Estimation du temps
    total_files = sum(c["total_files"] for c in collections_configs)
    estimated_time_min = total_files * 0.5  # 30s par fichier en moyenne

    print("\n" + "=" * 100)
    print(f" Statistiques globales :")
    print(f"   • Collections à créer : {len(collections_configs)}")
    print(f"   • Fichiers à traiter : {total_files}")
    print(f"   • Temps estimé : ~{estimated_time_min:.0f} minutes")
    print("=" * 100)

    # ===== ÉTAPE 4 : CONFIRMATION FINALE =====
    print("\n  ATTENTION : Ce processus peut prendre du temps")
    confirmation = input(
        "\n Lancer la création de TOUTES les collections ? (tapez 'OUI' en majuscules) : "
    ).strip()

    if confirmation != "OUI":
        print("Création annulée")
        return

    # ===== ÉTAPE 5 : CRÉATION DE TOUTES LES COLLECTIONS =====
    print("\n" + "=" * 100)
    print(" LANCEMENT DE LA CRÉATION")
    print("=" * 100)

    collections_created = []
    collections_failed = []

    for i, config in enumerate(collections_configs, 1):
        print(f"\n{'=' * 100}")
        print(f" COLLECTION #{i}/{len(collections_configs)} : {config['nom']}")
        print(f"{'=' * 100}")

        try:
            # Configurer le vectorizor
            print(f"🔧 Configuration du vectorizor pour {config['model_short']}...")
            r.vectorizor = Vectorizor()
            r.vectorizor._load_model(config["model_name"])

            # Lancer la vectorisation
            start_time = time.time()

            success = r.vectorize_with_config(
                chunk_size=config["chunk_size"],
                overlap=config["overlap"],
                collection_name=config["nom"],
                source_folder=config["source"],
                model_name=config["model_name"],
            )

            elapsed_time = time.time() - start_time

            if success:
                print(f"\n Collection '{config['nom']}' créée avec succès !")
                print(
                    f"   ⏱  Temps : {elapsed_time:.1f} secondes ({elapsed_time / 60:.1f} minutes)"
                )

                collections_created.append(
                    {"nom": config["nom"], "temps": elapsed_time}
                )

                # Afficher les stats
                r.chroma_storage.switch_collection(config["nom"])
                stats = r.chroma_storage.get_stats()
                print(f"    Statistiques :")
                print(f"      • Total chunks : {stats['total_documents']}")
                print(f"      • Total fichiers : {stats['total_fichiers']}")
            else:
                print(f"\n Échec de la création de '{config['nom']}'")
                collections_failed.append(config["nom"])

        except Exception as e:
            print(f"\n Erreur lors de la création de '{config['nom']}' : {e}")
            collections_failed.append(config["nom"])

        # Pause entre collections (optionnel)
        if i < len(collections_configs):
            print(f"\n⏸ Pause de 2 secondes avant la prochaine collection...")
            time.sleep(2)

    # ===== ÉTAPE 6 : RAPPORT FINAL =====
    print("\n" + "=" * 100)
    print(" RAPPORT FINAL - CRÉATION EN BATCH")
    print("=" * 100)

    if collections_created:
        total_time = sum(c["temps"] for c in collections_created)

        print(f"\n Collections créées avec succès ({len(collections_created)}) :")
        for col_info in collections_created:
            col = client.get_collection(col_info["nom"])
            count = col.count()
            metadata = col.metadata

            chunk_size = metadata.get("chunk_size", "N/A")
            overlap = metadata.get("overlap", "N/A")
            model = metadata.get("model", "N/A")

            print(f"\n    {col_info['nom']}")
            print(f"      • Documents : {count}")
            print(
                f"      • Paramètres : {chunk_size} caractères, {overlap} caractères overlap"
            )
            print(f"      • Modèle : {model}")
            print(
                f"      • Temps : {col_info['temps']:.1f}s ({col_info['temps'] / 60:.1f} min)"
            )

        print(f"\n Temps total : {total_time:.1f}s ({total_time / 60:.1f} minutes)")

    if collections_failed:
        print(f"\n Collections échouées ({len(collections_failed)}) :")
        for col_name in collections_failed:
            print(f"   • {col_name}")

    if not collections_created and not collections_failed:
        print("\n  Aucune collection créée")

    print("\n" + "=" * 100)

    input("\nAppuyez sur Entrée pour revenir au menu principal...")


def manage_cache_menu(r: Retrieval):
    """Menu interactif pour gérer le cache des documents."""
    while True:
        print("\n" + "=" * 100)
        print(" GESTION DU CACHE DES DOCUMENTS")
        print("=" * 100)
        print("   1. Lister le contenu du cache")
        print("   2. Vider le cache d'une base de données")
        print("   3. Vider TOUT le cache (irréversible)")
        print("   4. Forcer le retraitement d'un fichier")
        print("   5. Retour au menu principal")

        choix = input("\nVotre choix (1-5) : ").strip()

        if choix == "1":
            # Appel de la méthode list_cache via l'instance de retrieval
            r.document_processor.list_cache()

        elif choix == "2":
            db_name = input(
                "Nom de la base de données dont le cache est à vider (ex: DATA_Test) : "
            ).strip()
            if db_name:
                r.document_processor.clear_cache(database=db_name)

        elif choix == "3":
            confirm = input(
                "Voulez-vous vraiment vider TOUT le cache des textes extraits ? (tapez 'OUI') : "
            ).strip()
            if confirm == "OUI":
                r.document_processor.clear_cache()
            else:
                print("Opération annulée.")

        elif choix == "4":
            file_path = input("Chemin complet du fichier à retraiter : ").strip()
            if file_path and Path(file_path).exists():
                r.document_processor.rebuild_cache_for_file(source_file=file_path)
            else:
                print("Chemin invalide ou fichier inexistant.")

        elif choix == "5":
            break
        else:
            print("Choix invalide.")

        input("\nAppuyez sur Entrée pour continuer...")


def manage_collections():
    """
    Script interactif pour gérer les collections ChromaDB.
    - Affiche toutes les collections existantes
    - Permet de supprimer une collection
    - Permet de renommer une collection (pseudo-renommage)
    """

    r = Retrieval()
    client = r.chroma_storage.chroma_client

    while True:
        print("\n" + "=" * 100)
        print(" GESTION DES COLLECTIONS CHROMADB")
        print("=" * 100)

        # Lister toutes les collections
        collections = r.chroma_storage.list_collection_names()

        if not collections:
            print("\n Aucune collection trouvée dans ChromaDB")
            print(
                "   Utilisez 'add_new_domain.py' pour créer votre première collection."
            )
            break

        print(f"\n {len(collections)} collection(s) existante(s) :\n")

        # Afficher chaque collection avec ses stats
        for col_name in collections:
            col = client.get_collection(col_name)
            count = col.count()
            metadata = col.metadata

            # Récupérer les infos clés
            chunk_size = metadata.get("chunk_size", "N/A")
            overlap = metadata.get("overlap")
            created_at = metadata.get("created_at", "N/A")
            created_by = metadata.get("created_by", "N/A")
            source_folder = metadata.get("source_folder", "N/A")
            model_name = metadata.get("model", "N/A")

            # Affichage
            print(f"   • {col_name}")
            print(f"      Documents : {count}")

            if source_folder != "N/A":
                print(f"      Source : {source_folder}")

            if model_name != "N/A":
                model_short = {
                    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B",
                    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "MPNet",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "MiniLM",
                }.get(model_name, model_name)  # Si inconnu, afficher le nom complet

                print(f"      Modèle : {model_short}")

            if chunk_size != "N/A":
                overlap_str = f"{overlap} caractères" if overlap else "N/A"
                print(
                    f"      Paramètres : {chunk_size} caractères, {overlap_str} overlap"
                )

            if created_at != "N/A":
                date_only = (
                    str(created_at).split("T")[0]
                    if "T" in str(created_at)
                    else created_at
                )
                print(f"      Créée le : {date_only}")

            if created_by != "N/A":
                print(f"      Créée par : {created_by}")

            print()

        # Menu d'actions
        print("=" * 100)
        print("Actions disponibles :")
        print("   1. Supprimer une collection")
        print("   2. Supprimer plusieurs collections")
        print("   3. Créer une nouvelle collection")
        print("   4. Créer plusieurs collections successivement (BATCH)")
        print("   5. Renommer une collection (pseudo-renommage)")
        print("   6. Ajouter des documents à une collection existante")
        print("   7. Supprimer un document source")
        print("   8. Afficher les détails d'une collection")
        print("   9. Gérer le cache des documents")
        print("  10. [MAINTENANCE] Migrer les chemins vers un format relatif")
        print("  11. [MAINTENANCE] Migrer les chemins du CACHE (.json)")
        print("  12. Quitter")

        choix = input("\nVotre choix (1-12) : ").strip()

        # Option 12 : Quitter
        if choix == "12" or choix.lower() == "quitter" or choix.lower() == "q":
            print("\n Au revoir !")
            break

        # Option 11 : Migrer les chemins du cache (.metadata.json)
        elif choix == "11":  # NOUVEAU BLOC
            print("\nLancement de la migration des fichiers de cache .metadata.json...")
            # On utilise l'instance 'r' qui contient le DocumentProcessor correctement initialisé
            r.document_processor.migrate_cache_paths_to_relative()
            print("Opération terminée.")
            input("\nAppuyez sur Entrée pour continuer...")

        # Option 10 : Migrer les chemins vers un format relatif
        elif choix == "10":
            nom_col = input("Nom de la collection à migrer : ").strip()
            if nom_col in collections:
                r.chroma_storage.switch_collection(nom_col)
                # On passe le chemin racine de la config à la fonction de migration
                r.chroma_storage.migrate_paths_to_relative(ROOT_DATA_PATH)
            else:
                print(f"Collection '{nom_col}' introuvable.")
            input("\nAppuyez sur Entrée pour continuer...")

        # Option 9 : Gérer le cache des documents
        elif choix == "9":
            manage_cache_menu(r)
            # Pas besoin de input, il est déjà dans le sous-menu

        # Option 8 : Détails d'une collection
        elif choix == "8":
            nom = input("Nom de la collection : ").strip()

            if nom not in collections:
                print(f"\n Collection '{nom}' introuvable")
                continue

            # Afficher les détails complets
            col = client.get_collection(nom)
            metadata = col.metadata

            print(f"\n DÉTAILS DE LA COLLECTION '{nom}'")
            print("=" * 100)
            print(f"Documents : {col.count()}")
            print("\nMétadonnées :")
            for key, value in metadata.items():
                print(f"   • {key:<20} : {value}")

            input("\nAppuyez sur Entrée pour continuer...")

        # Option 7 : Supprimer un document source
        elif choix == "7":
            success = delete_source_interactive(client)
            if success:
                input("\n Appuyez sur Entrée pour continuer...")
            else:
                input("\n Appuyez sur Entrée pour continuer...")

        # Option 6 : Ajouter des documents à une collection existante
        elif choix == "6":
            success = add_documents_to_collection_interactive(r, client)
            input("\n Appuyez sur Entrée pour continuer...")

        # Option 5 : Renommer une collection
        elif choix == "5":
            print("\n" + "=" * 100)
            print(" RENOMMAGE DE COLLECTION")
            print("=" * 100)

            old_name = input("Nom actuel de la collection : ").strip()
            if old_name not in collections:
                print(f"\n Collection '{old_name}' introuvable.")
                input("\nAppuyez sur Entrée pour continuer...")
                continue

            new_name = input(f"Nouveau nom pour '{old_name}' : ").strip()

            # On ajoute les mêmes garde-fous que dans l'interface précédente
            if not new_name or new_name == old_name:
                print(" Nom invalide ou identique à l'ancien.")
                input("\nAppuyez sur Entrée pour continuer...")
                continue

            if new_name in collections:
                print(
                    f" Le nom '{new_name}' est déjà utilisé par une autre collection."
                )
                input("\nAppuyez sur Entrée pour continuer...")
                continue

            # Confirmation
            confirmation = (
                input(
                    f"Confirmer le renommage de '{old_name}' en '{new_name}' ? (o/n) : "
                )
                .strip()
                .lower()
            )

            if confirmation == "o":
                # Appel simple à la méthode métier
                # Le script d'interface ne sait pas comment ça marche, il délègue.
                success = r.chroma_storage.rename_collection(old_name, new_name)

                if success:
                    print("\n✓ Collection renommée avec succès !")
                else:
                    print(
                        "\n✗ Échec du renommage. Consultez les logs pour plus de détails."
                    )
            else:
                print("Renommage annulé.")

            input("\nAppuyez sur Entrée pour continuer...")

        # Option 4 : Créer plusieurs collections successivement (BATCH)
        elif choix == "4":
            batch_create_collections(client)
            input("\n Appuyez sur Entrée pour continuer...")

        # Option 3 : Créer une nouvelle collection
        elif choix == "3":
            success = add_collection_interactive(client)

            if success:
                input("\n Appuyez sur Entrée pour continuer...")
            else:
                input("\n Appuyez sur Entrée pour continuer...")

        # Option 2 : Supprimer plusieurs collections
        elif choix == "2":
            noms = input("Noms des collections (séparés par des virgules) : ").strip()
            noms_list = [n.strip() for n in noms.split(",") if n.strip()]

            if not noms_list:
                print("\n Aucune collection spécifiée")
                continue

            # Vérifier que toutes existent
            non_existantes = [n for n in noms_list if n not in collections]
            if non_existantes:
                print(f"\n Collections introuvables : {', '.join(non_existantes)}")
                continuer = (
                    input("Continuer avec les existantes ? (o/n) : ").strip().lower()
                )
                if continuer != "o":
                    continue
                noms_list = [n for n in noms_list if n in collections]

            # Afficher ce qui va être supprimé
            total_docs = 0
            print(f"\n Collections à supprimer :")
            for nom in noms_list:
                col = client.get_collection(nom)
                count = col.count()
                total_docs += count
                print(f"   • {nom} ({count} documents)")

            print(f"\nTotal : {len(noms_list)} collection(s), {total_docs} documents")

            confirmation = input("\n Confirmer ? (tapez 'SUPPRIMER') : ").strip()

            if confirmation != "SUPPRIMER":
                print(" Suppression annulée")
                continue

            # Suppression
            success_count = 0
            for nom in noms_list:
                try:
                    client.delete_collection(nom)
                    print(f" {nom} supprimée")
                    success_count += 1
                except Exception as e:
                    print(f"Erreur avec {nom} : {e}")

            print(f"\n Résultat : {success_count}/{len(noms_list)} supprimée(s)")
            input("\nAppuyez sur Entrée pour continuer...")

        # Option 1 : Supprimer une collection
        elif choix == "1":
            nom = input("Nom de la collection à supprimer : ").strip()

            if nom not in collections:
                print(f"\n Collection '{nom}' introuvable")
                print(f"   Collections disponibles : {', '.join(collections)}")
                continue

            # Récupérer les infos
            col = client.get_collection(nom)
            count = col.count()
            metadata = col.metadata

            # Afficher les détails
            print(f"\n Vous allez supprimer '{nom}' :")
            print(f"   • Documents : {count}")

            chunk_size = metadata.get("chunk_size", "N/A")
            if chunk_size != "N/A":
                overlap = metadata.get("overlap")
                overlap_str = f"{overlap} caractères" if overlap else "N/A"
                print(
                    f"   • Paramètres : {chunk_size} caractères, {overlap_str} overlap"
                )

            created_at = metadata.get("created_at", "N/A")
            if created_at != "N/A":
                print(f"   • Créée le : {created_at}")

            # Confirmation
            confirmation = (
                input(f"\n Confirmer la suppression ? (o/n) : ").strip().lower()
            )

            if confirmation != "o":
                print(" Suppression annulée")
                continue

            # Suppression
            try:
                client.delete_collection(nom)
                print(f"\n Collection '{nom}' supprimée ({count} documents)")
            except Exception as e:
                print(f"\n Erreur : {e}")

            input("\nAppuyez sur Entrée pour continuer...")

        else:
            print("\n Choix invalide")
            input("Appuyez sur Entrée pour continuer...")


if __name__ == "__main__":
    manage_collections()
