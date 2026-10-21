"""Génération de groupes gaussiens indépendants à covariance diagonale."""

import numpy as np


def generate_gaussian_clouds(centers, n_samples, std=1.0, *, rng=None, shuffle=True):
    """Retourne les observations X (N, d) et leurs labels d'origine y (N,).

    Parameters
    ----------
    centers : array-like de forme (K, d)
        Moyennes réelles finies des K groupes, avec K >= 1 et d >= 1.
    n_samples : entier ou array-like de forme (K,)
        Effectif strictement positif par groupe ; un entier est répété K fois.
        N est la somme de ces effectifs. Les booléens sont refusés.
    std : réel ou array-like de forme (K,) ou (K, d), optional
        Écart-type fini non négatif, commun, par groupe ou par coordonnée de
        chaque groupe. Un vecteur (K,) désigne toujours les groupes, même si
        K == d. Les coordonnées sont indépendantes : covariance diag(std[k]**2).
        Zéro est accepté comme cas dégénéré (coordonnée constante).
    rng : None, entier non négatif ou numpy.random.Generator, optional
        None crée un générateur local non déterministe ; un entier recrée un
        générateur local reproductible ; un Generator fourni est utilisé et
        son état avance. Aucun appel ne modifie l'état global de np.random.
        Pour reproduire les tirages, conserver aussi les paramètres, l'ordre
        des appels et l'environnement NumPy (la graine seule ne garantit pas
        une identité entre versions).
    shuffle : bool, optional
        Mélange conjointement X et y après les tirages (défaut : True).
        Sinon, les lignes sont regroupées dans l'ordre des centres.

    Returns
    -------
    X : ndarray float64, forme (N, d)
        Pour chaque groupe k, X | y=k suit N(centers[k], diag(std[k]**2)).
    y : ndarray int64, forme (N,)
        Indice du groupe générateur, entre 0 et K-1 ; ce n'est pas une
        affectation calculée par K-means, même si les nuages se recouvrent.

    Raises
    ------
    ValueError
        Paramètres de forme, type ou valeur incorrects, ou tirages non finis
        dus à un débordement float64. Les entrées ne sont jamais modifiées.

    Notes
    -----
    Temps et mémoire O(N*d + K*d). Les très grandes valeurs peuvent perdre
    leur dispersion par arrondi float64 ; les allocations restent limitées
    par la mémoire disponible. Aucune covariance corrélée n'est prise en charge.
    """
    centers = np.asarray(centers)
    if centers.ndim != 2 or centers.size == 0 or centers.dtype.kind not in "iuf":
        raise ValueError("centers doit être une matrice réelle 2D non vide.")
    centers = np.asarray(centers, dtype=np.float64)
    if not np.isfinite(centers).all():
        raise ValueError("centers doit contenir uniquement des valeurs finies.")
    k, dimension = centers.shape

    # Conserver les types des listes mixtes : [True, 2] ne doit pas devenir [1, 2].
    counts = np.asarray(n_samples, dtype=object)
    if counts.shape not in ((), (k,)) or any(
        not isinstance(count, (int, np.integer)) or isinstance(count, (bool, np.bool_))
        for count in counts.flat
    ):
        raise ValueError("n_samples doit être un entier ou un vecteur de K entiers.")
    if np.any(counts <= 0):
        raise ValueError("n_samples doit être strictement positif pour chaque groupe.")
    counts = np.broadcast_to(counts, (k,))
    total = sum(int(count) for count in counts)
    if total > np.iinfo(np.intp).max // dimension:
        raise ValueError("Le nombre de coordonnées dépasse la capacité d'indexation.")

    scales = np.asarray(std)
    if scales.dtype.kind not in "iuf" or scales.shape not in ((), (k,), (k, dimension)):
        raise ValueError("std doit être un réel, un vecteur (K,) ou une matrice (K, d).")
    scales = np.asarray(scales, dtype=np.float64)
    if not np.isfinite(scales).all() or np.any(scales < 0):
        raise ValueError("std doit contenir des valeurs finies non négatives.")
    if scales.shape == (k,):
        scales = scales[:, None]
    scales = np.broadcast_to(scales, centers.shape)

    if not isinstance(shuffle, (bool, np.bool_)):
        raise ValueError("shuffle doit être un booléen.")
    if not isinstance(rng, np.random.Generator):
        if rng is not None and (
            not isinstance(rng, (int, np.integer))
            or isinstance(rng, (bool, np.bool_))
            or rng < 0
        ):
            raise ValueError("rng doit être None, un entier non négatif ou un Generator.")
        rng = np.random.default_rng(rng)

    X = np.empty((total, dimension), dtype=np.float64)
    y = np.empty(total, dtype=np.int64)
    start = 0
    for label, count in enumerate(counts):
        stop = start + int(count)
        with np.errstate(over="ignore", invalid="ignore"):
            X[start:stop] = rng.normal(centers[label], scales[label], (int(count), dimension))
        y[start:stop] = label
        start = stop
    if not np.isfinite(X).all():
        raise ValueError("Les tirages dépassent la plage finie de float64.")
    if shuffle:
        order = rng.permutation(total)
        X, y = X[order], y[order]
    return X, y
