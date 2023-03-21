from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cosine


class SimilarItems:
    """Class that shows your similarity of embeddings and k-nearest items"""


    @staticmethod
    def similarity(embeddings: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], float]:
        """Calculate pairwise similarities between each item
        in embedding.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.

        Returns:
            Tuple[List[str], Dict[Tuple[int, int], float]]:
            List of all items + Pairwise similarities dict
            Keys are in form of (i, j) - combinations pairs of item_ids
            with i < j.
            Round each value to 8 decimal places.
        """
        pair_sims = {}

        emb_keys = list(embeddings.keys())

        for idx, i in enumerate(emb_keys[:-1]):
            for a in emb_keys[idx + 1:]:
                score = 1 - cosine(embeddings[i], embeddings[a])
                pair_sims[(i, a)] = round(score, 8)

        return pair_sims


    @staticmethod
    def knn(
            sim: Dict[Tuple[int, int], float], top: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Return closest neighbors for each item.

        Args:
            sim (Dict[Tuple[int, int], float]): <similarity> method output.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, List[Tuple[int, float]]]: Dict with top closest neighbors
            for each item.
        """

        keys = set([item for sublist in list(sim.keys()) for item in sublist])

        knn_dict = {}

        for i in keys:
            knn_dict[i] = {}
            dict_neighbors = {}

            for a in list(sim.keys()):
                if i in a:
                    neighbor = list(a)
                    neighbor.remove(i)
                    dict_neighbors[neighbor[0]] = sim[a]
                knn_dict[i] = dict_neighbors

        for i in knn_dict:
            knn_dict[i] = dict(sorted(knn_dict[i].items(), key=lambda x: x[1], reverse=True)[:top])
            knn_dict[i] = list(knn_dict[i].items())

        return knn_dict


    @staticmethod
    def knn_price(
            knn_dict: Dict[int, List[Tuple[int, float]]],
            prices: Dict[int, float],
    ) -> Dict[int, float]:
        """Calculate weighted average prices for each item.
        Weights should be positive numbers in [0, 2] interval.

        Args:
            knn_dict (Dict[int, List[Tuple[int, float]]]): <knn> method output.
            prices (Dict[int, float]): Price dict for each item.

        Returns:
            Dict[int, float]: New prices dict, rounded to 2 decimal places.
        """
        knn_price_dict = {}

        for item in knn_dict:

            weights = []
            neighbor_prices = []
            for a in knn_dict[item]:
                if prices[list(a)[0]] != np.nan:
                    weights.append(list(a)[1] + 1)
                    neighbor_prices.append(prices[list(a)[0]])

            vector_norm = np.linalg.norm(weights, 1)
            price = sum(neighbor_prices * (weights / vector_norm))

            knn_price_dict[item] = round(price, 2)

        return knn_price_dict


    @staticmethod
    def transform(
            embeddings: Dict[int, np.ndarray],
            prices: Dict[int, float],
            top: int,
    ) -> Dict[int, float]:
        """Transforming input embeddings into a dictionary
        with weighted average prices for each item.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.
            prices (Dict[int, float]): Price dict for each item.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, float]: Dict with weighted average prices for each item.
        """
        sim = SimilarItems.similarity(embeddings)
        knn_dict = SimilarItems.knn(sim, top)
        knn_price_dict = SimilarItems.knn_price(knn_dict, prices)

        return knn_price_dict
