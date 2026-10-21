"""Distances entre observations et prototypes, en dimension quelconque."""

import numpy as np


def euclidean_distances(X, P):
    """Retourne D[i, k] = ||X[i] - P[k]||₂, de forme (N, K).

    X et P sont des matrices réelles non vides de formes (N, d) et (K, d).
    Les listes imbriquées sont acceptées. Le calcul et la sortie utilisent
    float64 ; les entiers sont convertis avant toute soustraction.

    Raises
    ------
    ValueError
        Si les entrées ne sont pas des matrices numériques réelles, sont
        vides, non finies ou ont des nombres de coordonnées différents.

    Notes
    -----
    La boucle sur les prototypes évite un intermédiaire (N, K, d).
    Les opérations sur les N observations et d coordonnées sont vectorisées :
    coût O(N*K*d), mémoire O(N*K + N*d + K*d), entrées converties comprises.
    Comme tout calcul float64, la précision des grands entiers est limitée
    et une distance dépassant la plage représentable peut déborder.
    """
    matrices = []
    for name, values in (("X", X), ("P", P)):
        matrix = np.asarray(values)
        if matrix.ndim != 2 or matrix.size == 0:
            raise ValueError(f"{name} doit être une matrice 2D non vide.")
        if matrix.dtype.kind not in "biuf":
            raise ValueError(f"{name} doit contenir des nombres réels.")
        matrix = np.asarray(matrix, dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} doit contenir uniquement des valeurs finies.")
        matrices.append(matrix)

    X, P = matrices
    if X.shape[1] != P.shape[1]:
        raise ValueError("X et P doivent avoir le même nombre de coordonnées.")

    distances = np.empty((X.shape[0], P.shape[0]), dtype=np.float64)
    for k, prototype in enumerate(P):
        # hypot réduit les coordonnées : sqrt(somme(diff**2)), sans former
        # des carrés susceptibles de déborder ou de s'arrondir à zéro.
        distances[:, k] = np.hypot.reduce(X - prototype, axis=1)
    return distances
