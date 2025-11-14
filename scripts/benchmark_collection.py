import sys
from pathlib import Path
from src.rag import Retrieval
import time
from typing import List, Dict, Any

ROOT_DIR = Path(__file__).parent.parent  # Remonte de launchers/ à RAG_CGT/
sys.path.insert(0, str(ROOT_DIR))


def get_collection_metadata(collection) -> Dict[str, Any]:
    """
    Récupère les métadonnées directement depuis ChromaDB.

    Args:
        collection: Objet collection ChromaDB

    Returns:
        Dict: Métadonnées complètes ou valeurs par défaut
    """
    try:
        metadata = collection.metadata

        return {
            "chunk_size": metadata.get("chunk_size"),
            "overlap": metadata.get("overlap"),
            "source_folder": metadata.get("source_folder"),
            "created_at": metadata.get("created_at"),
            "model": metadata.get("model", "N/A"),
            "reranking_enabled": metadata.get("reranking_enabled"),
            "created_by": metadata.get("created_by", "N/A"),
            "version": metadata.get("version", "N/A"),
        }
    except Exception as e:
        print(f"Impossible de lire les métadonnées : {e}")
        return {
            "chunk_size": None,
            "overlap": None,
            "source_folder": None,
            "created_at": None,
            "model": "N/A",
            "reranking_enabled": None,
            "created_by": "N/A",
            "version": "N/A",
        }


def select_collections_menu() -> List[str]:
    """
    Menu interactif pour sélectionner les collections à benchmarker.

    Returns:
        List[str]: Liste des noms de collections sélectionnées
    """
    r = Retrieval()
    client = r.chroma_storage.chroma_client
    collections = r.chroma_storage.list_collection_names()

    if not collections:
        print("\nAucune collection trouvée dans ChromaDB")
        return []

    print("\n" + "=" * 100)
    print("SÉLECTION DES COLLECTIONS À BENCHMARKER")
    print("=" * 100)
    print(f"\n{len(collections)} collection(s) disponible(s) :\n")

    # Afficher chaque collection avec ses infos
    collections_info = []
    for i, col_name in enumerate(collections, 1):
        col = client.get_collection(col_name)
        count = col.count()
        metadata = get_collection_metadata(col)

        collections_info.append(
            {"num": i, "name": col_name, "count": count, "metadata": metadata}
        )

        # Affichage formaté
        print(f"   [{i}] {col_name}")
        print(f"       Documents : {count}")

        if metadata["chunk_size"]:
            overlap_str = (
                f"{metadata['overlap'] * 100:.0f}%" if metadata["overlap"] else "N/A"
            )
            print(
                f"       Paramètres : {metadata['chunk_size']} mots, {overlap_str} overlap"
            )

        if metadata["source_folder"] and metadata["source_folder"] != "N/A":
            print(f"       Source : {metadata['source_folder']}")

        if metadata["created_by"] != "N/A":
            print(f"       Créée par : {metadata['created_by']}")

        print()

    # Menu de sélection
    print("=" * 100)
    print("Options de sélection :")
    print("   • Tapez 'all' pour toutes les collections")
    print("   • Tapez les numéros séparés par des virgules (ex: 1,3,5)")
    print("   • Tapez les noms séparés par des virgules (ex: CGT, config_150_15)")

    choix = input("\nVotre choix : ").strip()

    # Traiter le choix
    if choix.lower() == "all":
        selected = collections
        print(f"\nToutes les {len(collections)} collection(s) sélectionnées")

    elif choix.replace(",", "").replace(" ", "").isdigit():
        # Sélection par numéros
        nums = [int(n.strip()) for n in choix.split(",") if n.strip().isdigit()]
        selected = []

        for num in nums:
            if 1 <= num <= len(collections):
                selected.append(collections[num - 1])
            else:
                print(f"Numéro {num} hors limite (ignoré)")

        print(
            f"\n{len(selected)} collection(s) sélectionnée(s) : {', '.join(selected)}"
        )

    else:
        # Sélection par noms
        noms = [n.strip() for n in choix.split(",")]
        selected = []

        for nom in noms:
            if nom in collections:
                selected.append(nom)
            else:
                print(f"Collection '{nom}' introuvable (ignorée)")

        if selected:
            print(
                f"\n{len(selected)} collection(s) sélectionnée(s) : {', '.join(selected)}"
            )
        else:
            print("\nAucune collection valide sélectionnée")

    return selected


def benchmark_collections(
    collections_to_test: List[str], query: str, n_results: int = 5, verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Benchmark les collections spécifiées avec UNE requête.

    Args:
        collections_to_test (List[str]): Noms des collections à tester
        query (str): Requête de test
        n_results (int): Nombre de résultats à récupérer
        verbose (bool): Afficher les détails

    Returns:
        List[Dict]: Résultats du benchmark
    """

    if verbose:
        print("\n" + "=" * 100)
        print(f" BENCHMARK DE {len(collections_to_test)} COLLECTION(S)")
        print("=" * 100)
        print(f" Requête : {query}\n")

    rag_instance = Retrieval()
    client = rag_instance.chroma_storage.chroma_client

    results = []

    for col_name in collections_to_test:
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"📊 Test : {col_name}")
            print(f"{'=' * 70}")

        try:
            # Switch vers la collection
            rag_instance.chroma_storage.switch_collection(col_name)

            # Récupérer les stats
            stats = rag_instance.chroma_storage.get_stats()
            total_chunks = stats["total_documents"]

            # Récupérer métadonnées
            collection = client.get_collection(col_name)
            params = get_collection_metadata(collection)

            if verbose:
                print(f"   • Total chunks : {total_chunks}")
                if params["chunk_size"]:
                    print(
                        f"   • Paramètres : {params['chunk_size']} mots, {params['overlap'] * 100:.0f}% overlap"
                    )
                    if params["source_folder"]:
                        print(f"   • Source : {params['source_folder']}")

            # Skip si collection vide
            if total_chunks == 0:
                if verbose:
                    print("  Collection vide, skip")
                continue

            # Test de recherche
            start_time = time.time()
            contexts, sources, scores = rag_instance.query(query, n=n_results)
            query_time = time.time() - start_time

            result = {
                "config_name": col_name,
                "chunk_size": params["chunk_size"],
                "overlap": params["overlap"],
                "source_folder": params["source_folder"],
                "total_chunks": total_chunks,
                "total_files": stats["total_fichiers"],
                "query_time": query_time,
                "best_score": scores[0] if scores else 0,
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "top_context": contexts[0][:150] + "..." if contexts else "",
                "top_source": sources[0] if sources else "N/A",
                "created_by": params.get("created_by", "N/A"),
                "created_at": params.get("created_at", "N/A"),
                "version": params.get("version", "N/A"),
                "query": query,
            }

            results.append(result)

            if verbose:
                print(f"   • Temps recherche : {query_time:.3f}s")
                print(f"   • Score top-1 : {result['best_score']:.3f}")
                print(f"   • Score moyen : {result['avg_score']:.3f}")

        except Exception as e:
            if verbose:
                print(f"   Erreur : {e}")
            continue

    return results


def benchmark_single_query(
    query: str = "Quels sont les dangers de l'intelligence artificielle ?",
    n_results: int = 5,
    collections: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Benchmark avec UNE SEULE requête.

    Args:
        query (str): Requête de test
        n_results (int): Nombre de résultats
        collections (List[str]): Collections à tester (None = menu de sélection)

    Returns:
        List[Dict]: Résultats du benchmark
    """

    # Sélection des collections si non spécifiées
    if collections is None:
        collections = select_collections_menu()

    if not collections:
        print("Aucune collection sélectionnée")
        return []

    # Lancer le benchmark
    return benchmark_collections(collections, query, n_results, verbose=True)


def benchmark_multiple_queries(
    queries: List[str] = None, n_results: int = 5, collections: List[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Benchmark avec PLUSIEURS requêtes.

    Args:
        queries (List[str]): Liste des requêtes (None = défaut)
        n_results (int): Nombre de résultats par requête
        collections (List[str]): Collections à tester (None = menu)

    Returns:
        Dict: Résultats groupés par requête
    """

    # Requêtes par défaut
    if queries is None:
        queries = [
            "Quels sont les dangers de l'intelligence artificielle ?",
            "Histoire de l'intelligence artificielle",
            "Applications pratiques de l'IA",
            "Qu'est-ce que l'apprentissage automatique ?",
            "Éthique de l'intelligence artificielle",
        ]

    # Sélection des collections
    if collections is None:
        collections = select_collections_menu()

    if not collections:
        print("Aucune collection sélectionnée")
        return {}

    print("\n" + "=" * 100)
    print(" BENCHMARK MULTI-REQUÊTES")
    print("=" * 100)
    print(
        f"\n{len(queries)} requête(s) × {len(collections)} collection(s) = {len(queries) * len(collections)} tests\n"
    )

    all_results = {}

    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 100}")
        print(f"REQUÊTE {i}/{len(queries)} : {query}")
        print("=" * 100)

        # Benchmark avec cette requête
        results = benchmark_collections(collections, query, n_results, verbose=True)
        all_results[query] = results

    return all_results


def print_comparison_table(results: List[Dict[str, Any]], title: str = "RÉSULTATS"):
    """
    Affiche le tableau comparatif formaté.

    Args:
        results (List[Dict]): Liste des résultats
        title (str): Titre du tableau
    """

    if not results:
        print("\nAucun résultat à afficher")
        return

    print("\n\n" + "=" * 110)
    print(f"=== TABLEAU COMPARATIF - {title} ===")
    print("=" * 110)

    # En-tête
    print(
        f"{'Collection':<25} {'Size':<8} {'Overlap':<10} {'Chunks':<10} "
        f"{'Files':<8} {'Temps (s)':<12} {'Score':<8}"
    )
    print("-" * 110)

    # Lignes
    for r in results:
        size_str = str(r["chunk_size"]) if r["chunk_size"] else "N/A"
        overlap_str = (
            f"{r['overlap'] * 100:.0f}%" if r["overlap"] is not None else "N/A"
        )

        print(
            f"{r['config_name']:<25} {size_str:<8} {overlap_str:<10} "
            f"{r['total_chunks']:<10} {r['total_files']:<8} "
            f"{r['query_time']:<12.3f} {r['best_score']:<8.3f}"
        )

    # Recommandations
    print("\n" + "=" * 110)
    print("MEILLEURES CONFIGURATIONS :")

    best_score_config = max(results, key=lambda x: x["best_score"])
    fastest_config = min(results, key=lambda x: x["query_time"])
    smallest_config = min(results, key=lambda x: x["total_chunks"])

    print(
        f"    Meilleur score : {best_score_config['config_name']} ({best_score_config['best_score']:.3f})"
    )
    print(
        f"   ⚡ Plus rapide : {fastest_config['config_name']} ({fastest_config['query_time']:.3f}s)"
    )
    print(
        f"   Plus compacte : {smallest_config['config_name']} ({smallest_config['total_chunks']} chunks)"
    )

    # Recommandation équilibrée
    balanced = max(
        results, key=lambda x: (x["best_score"] * 0.6) + (1 - x["query_time"] / 5) * 0.4
    )
    print(f"\n RECOMMANDATION (équilibre score + vitesse) : {balanced['config_name']}")
    if balanced["chunk_size"]:
        print(
            f"   Paramètres : {balanced['chunk_size']} mots, {balanced['overlap'] * 100:.0f}% overlap"
        )

    print("=" * 110)


def print_multi_query_summary(all_results: Dict[str, List[Dict[str, Any]]]):
    """
    Affiche un résumé comparatif multi-requêtes.

    Args:
        all_results (Dict): Résultats groupés par requête
    """

    print("\n\n" + "=" * 100)
    print("=== RÉSUMÉ MULTI-REQUÊTES ===")
    print("=" * 100)

    # Pour chaque requête, meilleure collection
    for query, results in all_results.items():
        print(f"\nRequête : {query}")

        if not results:
            print("   Aucun résultat")
            continue

        best = max(results, key=lambda x: x["best_score"])
        print(
            f"   Meilleure : {best['config_name']} (score: {best['best_score']:.3f}, temps: {best['query_time']:.3f}s)"
        )

    # Analyse globale
    print("\n" + "=" * 100)
    print("ANALYSE GLOBALE (moyenne sur toutes les requêtes)")
    print("=" * 100)

    # Scores moyens par collection
    collection_scores = {}

    for query, results in all_results.items():
        for result in results:
            col_name = result["config_name"]
            if col_name not in collection_scores:
                collection_scores[col_name] = []
            collection_scores[col_name].append(result["best_score"])

    # Moyennes
    collection_averages = {
        col: sum(scores) / len(scores) for col, scores in collection_scores.items()
    }

    # Tri décroissant
    sorted_collections = sorted(
        collection_averages.items(), key=lambda x: x[1], reverse=True
    )

    print("\nClassement (score moyen) :")
    for i, (col_name, avg_score) in enumerate(sorted_collections, 1):
        print(f"   {i}. {col_name} : {avg_score:.3f}")

    # Meilleure globale
    if sorted_collections:
        best_overall = sorted_collections[0]
        print(f"\nMEILLEURE COLLECTION GLOBALE : {best_overall[0]}")
        print(f"   Score moyen : {best_overall[1]:.3f}")
        print(f"   Testé sur {len(all_results)} requête(s)")

    print("=" * 100)


def export_results(
    results: List[Dict[str, Any]], filename: str = "benchmark_results.txt"
):
    """Exporte les résultats mono-requête"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("BENCHMARK CHROMADB - RÉSULTATS\n")
        f.write("=" * 100 + "\n\n")

        if results:
            f.write(f"Requête : {results[0].get('query', 'N/A')}\n\n")

        for r in results:
            f.write(f"Collection : {r['config_name']}\n")
            f.write(f"   Chunk size : {r['chunk_size'] or 'N/A'}\n")
            f.write(f"   Overlap : {r['overlap'] * 100 if r['overlap'] else 'N/A'}%\n")
            f.write(f"   Score : {r['best_score']:.3f}\n")
            f.write(f"   Temps : {r['query_time']:.3f}s\n\n")

    print(f"\nRésultats exportés : {filename}")


def export_multi_query_results(
    all_results: Dict[str, List[Dict[str, Any]]],
    filename: str = "benchmark_multi_queries.txt",
):
    """Exporte les résultats multi-requêtes"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("BENCHMARK MULTI-REQUÊTES\n")
        f.write("=" * 100 + "\n\n")

        for i, (query, results) in enumerate(all_results.items(), 1):
            f.write(f"\nREQUÊTE {i} : {query}\n")
            f.write("-" * 100 + "\n")

            for r in results:
                f.write(f"{r['config_name']:<30} Score: {r['best_score']:.3f}\n")

            if results:
                best = max(results, key=lambda x: x["best_score"])
                f.write(f"\nMeilleure : {best['config_name']}\n\n")

    print(f"\nRésultats multi-requêtes exportés : {filename}")


def interactive_menu():
    """Menu interactif principal"""

    print("\n" + "=" * 100)
    print("BENCHMARK COLLECTIONS CHROMADB")
    print("=" * 100)
    print("\nOptions :")
    print("   1. Benchmark avec UNE requête (rapide)")
    print("   2. Benchmark avec PLUSIEURS requêtes par défaut (thème IA)")
    print("   3. Benchmark personnalisé")
    print("   4. Quitter")

    choix = input("\nVotre choix (1-4) : ").strip()

    if choix == "1":
        # Mono-requête
        query = input(
            "\n    Requête de test (Entrée pour requête par défaut) \n(Quels sont les dangers de l'intelligence artificielle ?)   :\n"
        ).strip()
        if not query:
            query = "Quels sont les dangers de l'intelligence artificielle ?"

        results = benchmark_single_query(query=query)

        if results:
            print_comparison_table(results, title=f"REQUÊTE: {query}")

            export = input("\nExporter ? (o/n) : ").strip().lower()
            if export == "o":
                export_results(results)

    elif choix == "2":
        # Multi-requêtes défaut
        all_results = benchmark_multiple_queries()

        if all_results:
            print_multi_query_summary(all_results)

            for query, results in all_results.items():
                print_comparison_table(results, title=f"REQUÊTE: {query[:50]}...")

            export = input("\nExporter ? (o/n) : ").strip().lower()
            if export == "o":
                export_multi_query_results(all_results)

    elif choix == "3":
        # Personnalisé
        print("\nEntrez vos requêtes (une par ligne, ligne vide pour terminer) :")
        queries = []
        while True:
            q = input(f"Requête {len(queries) + 1} : ").strip()
            if not q:
                break
            queries.append(q)

        if queries:
            all_results = benchmark_multiple_queries(queries=queries)

            if all_results:
                print_multi_query_summary(all_results)

                for query, results in all_results.items():
                    print_comparison_table(results, title=f"REQUÊTE: {query}")

                export = input("\nExporter ? (o/n) : ").strip().lower()
                if export == "o":
                    export_multi_query_results(all_results)

    elif choix == "4":
        print("\nAu revoir !")

    else:
        print("\nChoix invalide")


if __name__ == "__main__":
    interactive_menu()
