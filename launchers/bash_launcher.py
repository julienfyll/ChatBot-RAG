import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent  # Remonte de launchers/ à RAG_CGT/
sys.path.insert(0, str(ROOT_DIR))

from src.rag import Rag


def select_collection_menu(rag_instance):
    """
    Menu interactif pour sélectionner la collection à tester.

    Args:
        rag_instance: Instance du RAG

    Returns:
        str: Nom de la collection sélectionnée
    """

    print("\n" + "=" * 100)
    print(" SÉLECTION DE LA COLLECTION À TESTER")
    print("=" * 100)

    client = rag_instance.retrieval.chroma_storage.chroma_client
    collections = rag_instance.retrieval.chroma_storage.list_collection_names()

    if not collections:
        print("\n Aucune collection trouvée dans ChromaDB")
        return None

    print(f"\n{len(collections)} collection(s) disponible(s) :\n")

    # Afficher chaque collection avec ses infos
    for i, col_name in enumerate(collections, 1):
        col = client.get_collection(col_name)
        count = col.count()
        metadata = col.metadata

        model_name = metadata.get("model", "N/A")
        chunk_size = metadata.get("chunk_size", "N/A")
        overlap = metadata.get("overlap")
        source_folder = metadata.get("source_folder", "N/A")

        print(f"   [{i}] {col_name}")
        print(f"       Documents : {count}")

        if model_name != "N/A":
            # Afficher le nom court pour plus de lisibilité
            model_short = {
                "Qwen/Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B",
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "MPNet",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "MiniLM",
            }.get(model_name, model_name)  # Si inconnu, afficher le nom complet

            print(f"       Modèle : {model_short}")

        if chunk_size != "N/A":
            overlap_str = f"{overlap * 100:.0f}%" if overlap else "N/A"
            print(f"       Paramètres : {chunk_size} mots, {overlap_str} overlap")

        if source_folder != "N/A":
            print(f"       Source : {source_folder}")

        print()

    # Menu de sélection
    DEFAULT_COLLECTION = (
        "CGT_MPNet4"  # ← Changez ce nom selon votre collection préférée
    )

    print("=" * 100)
    print("Sélection :")
    print("   • Tapez le numéro de la collection (ex: 1)")
    print("   • Tapez le nom de la collection (ex: CGT)")
    print(f"   • Appuyez sur Entrée pour utiliser : {DEFAULT_COLLECTION}")

    choix = input("\nVotre choix : ").strip()

    # Traiter le choix
    if choix.lower() == "default":
        default_collection = rag_instance.retrieval.chroma_storage.collection_name
        print(f"\n Collection par défaut utilisée : {default_collection}")
        return default_collection

    elif choix.isdigit():
        # Sélection par numéro
        num = int(choix)
        if 1 <= num <= len(collections):
            selected = collections[num - 1]
            print(f"\n Collection sélectionnée : {selected}")
            return selected
        else:
            print(f"\n Numéro {num} hors limite")
            return None

    elif choix in collections:
        # Sélection par nom
        print(f"\n Collection sélectionnée : {choix}")
        return choix

    else:
        print(f"\n Collection '{choix}' introuvable")
        return None


def test_complete():
    """Test complet du système RAG avec sélection de collection"""

    print("=" * 100)
    print("=== ÉTAPE 1: Initialisation du système RAG ===")
    print("=" * 100)

    try:
        rag_instance = Rag()
        print(" RAG initialisé avec succès")
    except Exception as e:
        print(f" Erreur lors de l'initialisation : {e}")
        return

    print("\n" + "=" * 100)
    print("=== ÉTAPE 2: Vérification ChromaDB/Migration ===")
    print("=" * 100)

    # Vérifier que ChromaDB a des collections
    client = rag_instance.retrieval.chroma_storage.chroma_client
    collections = rag_instance.retrieval.chroma_storage.list_collection_names()
    if not collections:
        print(" Aucune collection ChromaDB trouvée")
        print("   Utilisez 'manage_collections.py' pour créer une collection")
        return

    # Compter le total de documents
    total_docs = sum(client.get_collection(col).count() for col in collections)

    if total_docs == 0:
        print(" ChromaDB existe mais toutes les collections sont vides")
        print("   Utilisez 'manage_collections.py' pour créer une collection")
        return

    print(f" ChromaDB prête : {len(collections)} collection(s), {total_docs} documents")

    print("\n" + "=" * 100)
    print("=== ÉTAPE 2.5: Sélection de la collection ===")
    print("=" * 100)

    # Sélection de la collection
    selected_collection = select_collection_menu(rag_instance)

    if not selected_collection:
        print(" Aucune collection sélectionnée, arrêt du test")
        return

    # Basculer vers la collection sélectionnée
    rag_instance.retrieval.chroma_storage.switch_collection(selected_collection)

    # Afficher les statistiques de la collection
    stats = rag_instance.retrieval.get_stats()
    print(f"\n Statistiques de '{selected_collection}' :")
    print(f"   • Total documents : {stats['total_documents']}")
    print(f"   • Total fichiers : {stats['total_fichiers']}")

    # Afficher les métadonnées si disponibles
    collection = rag_instance.retrieval.chroma_storage.collection
    metadata = collection.metadata

    if metadata.get("chunk_size"):
        print(
            f"   • Paramètres : {metadata.get('chunk_size')} mots, {metadata.get('overlap', 0) * 100:.0f}% overlap"
        )
    if metadata.get("created_at"):
        print(f"   • Créée le : {metadata.get('created_at')}")
    if metadata.get("created_by"):
        print(f"   • Créée par : {metadata.get('created_by')}")

    print("\n" + "=" * 100)
    print("=== ÉTAPE 3: Test de requête(s) ===")
    print("=" * 100)

    # Menu de test : une ou plusieurs questions
    print("\nOptions de test :")
    print("   1. Tester UNE question")
    print("   2. Tester PLUSIEURS questions prédéfinies")
    print("   3. Mode interactif (chat)")

    mode = input("\nVotre choix (1-3) : ").strip()

    if mode == "1":
        # Une seule question
        query = input("\nVotre question : ").strip()
        if not query:
            query = "Quels sont les dangers de l'intelligence artificielle ?"
            print(f"Question par défaut : {query}")

        print(f"\n Question: {query}")

        try:
            response = rag_instance.respond(query)
            print("\n RÉPONSE:")
            print(response)
            print("\n Test terminé avec succès!")
        except Exception as e:
            print(f" Erreur lors de la requête: {e}")

    elif mode == "2":
        # Plusieurs questions prédéfinies
        questions = [
            "Quels sont les droits des agents ?",
            "Comment fonctionne le télétravail ?",
            "Quelles sont les règles de congés ?",
            "Qu'est-ce que le CERH ?",
        ]

        print(f"\n Test de {len(questions)} question(s) prédéfinies\n")

        for i, query in enumerate(questions, 1):
            print("=" * 100)
            print(f" QUESTION {i}/{len(questions)} : {query}")
            print("=" * 100)

            try:
                response = rag_instance.respond(query)
                print("\n RÉPONSE:")
                print(response)
                print("\n Réponse générée")

                if i < len(questions):
                    input("\nAppuyez sur Entrée pour continuer...")

            except Exception as e:
                print(f" Erreur : {e}")

        print("\n" + "=" * 100)
        print(" Tous les tests terminés !")
        print("=" * 100)

    elif mode == "3":
        # Mode interactif
        print("\n Mode interactif activé (tapez 'quit' pour quitter)")

        while True:
            query = input("\nVous : ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print(" Au revoir !")
                break

            if not query:
                continue

            try:
                response = rag_instance.respond(query)
                print(f"\n Assistant : {response}")
            except Exception as e:
                print(f" Erreur : {e}")

    else:
        print("\n Choix invalide")


if __name__ == "__main__":
    test_complete()
