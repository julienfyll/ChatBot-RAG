import re
import numpy as np
from typing import List, Dict, Any


class Reranker:
    """
    Reranker simple et stable :
    combine le score d'embedding et un score lexical heuristique
    par une moyenne pondérée directe (sans normalisation).
    """

    def __init__(
        self,
        enabled: bool = True,
        alpha: float = 0.2,
        method="heuristic",
        jaccard_weight: float = 0.5,
        density_weight: float = 0.3,
        exact_weight: float = 0.2,
    ):
        """
        Args:
            enabled (bool): active/désactive le reranking.
            alpha (float): poids du score d'embedding (entre 0 et 1).
                           Exemple : 0.5 => pondération égale embedding / heuristique.

            method (str): méthode de scoring lexical à utiliser. Options :
                          - "heuristic" : score heuristique combiné (par défaut)
                            - "bm25" : score BM25 (non implémenté ici)
                            - "llm" : score basé sur un LLM (non implémenté ici)
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"Le paramètre 'alpha' doit être compris entre 0.0 et 1.0, mais a reçu {alpha}"
            )
        self.enabled = enabled
        self.alpha = float(alpha)
        self.method = method

        total_weight = jaccard_weight + density_weight + exact_weight
        if not np.isclose(total_weight, 1.0):
            # On utilise np.isclose pour gérer les imprécisions des floats
            print(
                f"AVERTISSEMENT: La somme des poids heuristiques n'est pas égale à 1.0 (total={total_weight:.2f})"
            )

        self.jaccard_weight = jaccard_weight
        self.density_weight = density_weight
        self.exact_weight = exact_weight

    # --- Fonctions internes de similarité heuristique ---
    def _normalize(self, text: str) -> str:
        return (text or "").lower()

    def _tokens(self, text: str):
        return re.findall(r"[a-z0-9]+", self._normalize(text))

    def _score_kw_jaccard(self, q: str, p: str) -> float:
        """Similarité Jaccard entre tokens de la requête et du passage."""
        tq, tp = set(self._tokens(q)), set(self._tokens(p))
        if not tq or not tp:
            return 0.0
        inter, union = len(tq & tp), len(tq | tp)
        return inter / union if union else 0.0

    def _score_density(self, q: str, p: str) -> float:
        """Densité des mots de la requête présents dans le passage."""
        tq, tp = self._tokens(q), self._tokens(p)
        if not tq or not tp:
            return 0.0
        qset = set(tq)
        hits = sum(1 for t in tp if t in qset)
        return hits / max(1, len(tp))

    def _score_exact(self, q: str, p: str) -> float:
        """Détection d'une sous-chaîne exacte de la requête dans le passage."""
        qn, pn = self._normalize(q).strip(), self._normalize(p)
        if not qn or not pn:
            return 0.0
        if qn in pn:
            return 1.0
        qt = self._tokens(q)
        if len(qt) >= 4 and " ".join(qt[:4]) in pn:
            return 0.6
        return 0.0

    def _local_score_heuristic(self, q: str, passage: str) -> float:
        """Score heuristique global combinant plusieurs signaux lexicaux."""
        return (
            self.jaccard_weight * self._score_kw_jaccard(q, passage)
            + self.density_weight * self._score_density(q, passage)
            + self.exact_weight * self._score_exact(q, passage)
        )

    # --- Fonction principale : réévaluation et tri des candidats ---
    def rescore(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Recalcule un score final pour chaque passage.
        Combine la similarité sémantique (retrieval) et lexicale (heuristique).

        Args:
            query (str): la requête utilisateur.
            candidates (List[Dict]): liste de dicts contenant au moins :
                - "batch" (texte)
                - "chemin" (source)
                - "score_retrieval" (float)
        Returns:
            List[Dict]: candidats enrichis avec :
                - "score_local"
                - "score_final"
            triés par score_final décroissant.
        """

        if not self.enabled or len(candidates) <= 1:
            out = []
            for c in candidates:
                c2 = dict(c)
                c2["score_local"] = 0.0
                c2["score_final"] = float(c.get("score_retrieval", 0.0))
                out.append(c2)
            out.sort(key=lambda x: x["score_final"], reverse=True)
            return out

        rescored = []
        for c in candidates:
            score_emb = float(c.get("score_retrieval", 0.0)) / 2.0  # sur 0.5
            # 🔹 Ici on choisit la bonne fonction selon la méthode
            if self.method == "heuristic":
                score_loc = self._local_score_heuristic(query, c.get("batch", ""))
            elif self.method == "bm25":
                score_loc = self._local_score_bm25(query, c.get("batch", ""))
            elif self.method == "llm":
                score_loc = self._local_score_llm(query, c.get("batch", ""))
            else:
                score_loc = 0.0

            score_loc = score_loc / 2.0  # sur 0.5

            score_final = score_emb + score_loc  # somme = score total sur 1.0

            c2 = dict(c)
            c2["score_local"] = score_loc
            c2["score_final"] = score_final
            rescored.append(c2)

        rescored.sort(key=lambda x: x["score_final"], reverse=True)
        print(score_loc, score_emb, score_final)
        return rescored
