"""K-means de Lloyd, en dimension quelconque, sans bibliothèque de clustering."""

from dataclasses import dataclass
from math import fsum

import numpy as np

from lloyd_kmeans.distance import euclidean_distances


@dataclass
class KMeansResult:
    """Partition et diagnostic d'un appel à :func:`kmeans`.

    centers : ndarray float64 (K, d)
        Prototypes finaux, dans l'ordre de l'initialisation.
    labels : ndarray int64 (N,)
        Affectation unique, entre 0 et K-1, au prototype final le plus proche.
        En cas d'égalité des distances calculées par euclidean_distances, le plus petit indice
        gagne. Deux distances égales en arithmétique exacte peuvent différer
        d'un ULP, car np.hypot.reduce dépend de l'ordre des coordonnées.
    clusters : tuple de K ndarrays float64 (n_k, d)
        Observations de chaque groupe, dans l'ordre de X ; groupes vides admis.
        Équivalent à tuple(X[labels == k] for k in range(K)).
    inertia : float
        Somme (non normalisée) des distances AU CARRÉ aux prototypes affectés.
    inertia_history : ndarray float64 (n_iter + 1,)
        Inertie initiale, puis après chaque recentrage et réaffectation.
    n_iter : int
        Nombre de recentrages effectués (au moins un).
    converged : bool
        True si le critère choisi est satisfait, même à max_iter. Le mode
        center_shift ne garantit pas des labels stationnaires.
    labels_stable : bool
        Stabilité exacte des labels lors du dernier recentrage/réaffectation.
        Si True, les centres non vides sont les moyennes des groupes retournés.
        Sinon les labels restent aux plus proches centres, mais ceux-ci ne
        sont pas nécessairement les moyennes des nouveaux groupes.
    stop_reason : str
        "labels", "center_shift" si le critère choisi est satisfait ; sinon
        "max_iter". Des labels stables peuvent précéder le seuil de déplacement.
    center_shift_history : ndarray float64 (n_iter,) ou None
        Déplacements maximaux réellement mesurés, en mode center_shift seul.
        None en mode labels : aucun déplacement fictif n'est enregistré.

    Les tableaux sont des copies indépendantes des entrées. Pour la structure
    booléenne des affectations : Index = result.labels[:, None] == np.arange(K).
    """

    centers: np.ndarray
    labels: np.ndarray
    clusters: tuple[np.ndarray, ...]
    inertia: float
    inertia_history: np.ndarray
    n_iter: int
    converged: bool
    labels_stable: bool = False
    stop_reason: str = "max_iter"
    center_shift_history: np.ndarray | None = None


def _real_matrix(values, name):
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.size == 0 or matrix.dtype.kind not in "biuf":
        raise ValueError(f"{name} doit être une matrice réelle 2D non vide.")
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} doit contenir uniquement des valeurs finies float64.")
    return matrix


def _assign(X, centers):
    # euclidean_distances calcule des distances euclidiennes robustes, sans intermédiaire (N,K,d).
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        distances = euclidean_distances(X, centers)
    if not np.isfinite(distances).all():
        raise ValueError("Les distances dépassent la plage finie de float64.")
    labels = np.argmin(distances, axis=1).astype(np.int64, copy=False)
    nearest = distances[np.arange(len(X)), labels]
    # La norme avant le carré évite de perdre chaque petit carré séparément.
    with np.errstate(over="ignore", under="ignore"):
        residual_norm = np.hypot.reduce(nearest)
        inertia = float(residual_norm * residual_norm)
    if not np.isfinite(inertia) or (residual_norm > 0 and inertia == 0):
        raise ValueError("L'inertie n'est pas représentable en float64 ; redimensionner X.")
    return labels, inertia


def kmeans(X, K, *, init=None, rng=None, max_iter=300,
           stop_criterion="labels", tol=0.0):
    """Partitionne X (N, d) en K groupes par l'algorithme de Lloyd.

    Parameters
    ----------
    X : array-like réel de forme (N, d), N >= 1, d >= 1
        Observations finies, converties en float64. Les entrées ne sont jamais
        modifiées ; aucune standardisation implicite n'est appliquée.
    K : entier, 1 <= K <= N
        Nombre de prototypes demandés, pas nécessairement de groupes occupés.
        Les booléens sont refusés, comme pour max_iter et les graines.
    init : None ou array-like réel de forme (K, d), optional
        None choisit K indices distincts de X, uniformément sans remise.
        Des lignes identiques de X peuvent donc donner des prototypes égaux.
        Une matrice fixe les prototypes (pas nécessairement pris dans X) et
        permet des comparaisons à initialisation identique.
    rng : None, entier non négatif ou numpy.random.Generator, optional
        Utilise un générateur local, état global inchangé.
        Un Generator fourni avance seulement si init est None. La graine ne
        suffit à reproduire les résultats qu'à données et environnement fixés.
    max_iter : entier strictement positif, optional
        Nombre maximal de recentrages, quel que soit le critère choisi.
    stop_criterion : {"labels", "center_shift"}, optional
        "labels" conserve exactement l'arrêt par stabilité des labels après réaffectation.
        "center_shift" exige max_k ||Pm[k] - Pc[k]||_2 <= tol.
    tol : réel fini non négatif, optional
        Seuil absolu dans les unités de X, utilisé seulement en center_shift.
        Zéro exige des prototypes inchangés en float64. Les booléens sont
        refusés. Le seuil est validé même en mode labels, où il est sans effet.

    Returns
    -------
    KMeansResult
        Clusters, prototypes, labels et diagnostic (voir la classe ci-dessus).

    Notes
    -----
    Pc désigne les prototypes courants et Pm les nouvelles moyennes.
    Chaque étape calcule les distances, affecte par argmin, puis recalcule les moyennes.
    Une classe vide conserve Pc[k] : le critère ne dépend pas de son prototype
    à partition fixée. On ne force pas artificiellement K classes non vides.

    Par défaut, l'arrêt est la stabilité exacte des labels,
    point fixe de Lloyd en arithmétique machine. En mode center_shift,
    chaque centre se déplace d'au plus tol, via le maximum
    des normes euclidiennes calculées avec hypot, sans carré intermédiaire.
    La réaffectation est toujours effectuée, sans recentrage supplémentaire
    silencieux : labels, clusters et inertie correspondent aux centres rendus.
    Un petit déplacement ne prouve pas la stationnarité des labels.
    Aucune optimalité globale, ni optimalité locale stricte en cas d'ex aequo
    ou de classes vides, n'est garantie. Une seule initialisation est exécutée.

    En arithmétique exacte, affectation et recentrage ne peuvent augmenter
    l'inertie. En float64, des écarts d'arrondi sont possibles. Les sommes des
    coordonnées utilisent fsum (sommation compensée). Si la somme déborde,
    on calcule la moyenne après translation par une observation du groupe.
    Les grandes valeurs peuvent perdre leur dispersion dès la conversion
    des entrées et les très petites coordonnées leur précision relative.
    Une distance non finie ou une inertie débordée/arrondie à zéro alors que
    des résidus sont non nuls provoque ValueError ; redimensionner les données.

    Coût O(T*N*K*d), mémoire O(N*K + N*d + K*d + T) pour T recentrages.
    Les copies des clusters sont créées uniquement au retour. Pas de matrice
    booléenne (N,K) conservée ni d'historique des positions des prototypes.

    Raises
    ------
    ValueError
        Paramètre incorrect ou calcul non représentable selon le contrat
        numérique ci-dessus. Les allocations dépendent de la mémoire disponible.
    """
    X = _real_matrix(X, "X")
    if (
        not isinstance(K, (int, np.integer))
        or isinstance(K, (bool, np.bool_))
        or not 1 <= K <= len(X)
    ):
        raise ValueError("K doit être un entier entre 1 et N.")
    if (
        not isinstance(max_iter, (int, np.integer))
        or isinstance(max_iter, (bool, np.bool_))
        or max_iter < 1
    ):
        raise ValueError("max_iter doit être un entier strictement positif.")
    if not isinstance(stop_criterion, str) or stop_criterion not in ("labels", "center_shift"):
        raise ValueError('stop_criterion doit être "labels" ou "center_shift".')
    if (
        not isinstance(tol, (int, float, np.integer, np.floating))
        or isinstance(tol, (bool, np.bool_))
    ):
        raise ValueError("tol doit être un réel fini non négatif, hors booléens.")
    try:
        tol = float(tol)
    except (OverflowError, ValueError) as exc:
        raise ValueError("tol doit être représentable en float64.") from exc
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol doit être un réel fini non négatif.")
    K, max_iter = int(K), int(max_iter)
    if not isinstance(rng, np.random.Generator):
        if rng is not None and (
            not isinstance(rng, (int, np.integer))
            or isinstance(rng, (bool, np.bool_))
            or rng < 0
        ):
            raise ValueError("rng doit être None, un entier non négatif ou un Generator.")
        rng = np.random.default_rng(rng)
    if init is None:
        Pc = X[rng.choice(len(X), size=K, replace=False)].copy()
    else:
        Pc = _real_matrix(init, "init").copy()
        if Pc.shape != (K, X.shape[1]):
            raise ValueError("init doit avoir la forme (K, d).")

    labels, inertia = _assign(X, Pc)
    history = [inertia]
    shifts = [] if stop_criterion == "center_shift" else None
    for n_iter in range(1, max_iter + 1):
        Pm = Pc.copy()
        for k in range(K):
            group = X[labels == k]
            if len(group):
                for coordinate in range(X.shape[1]):
                    column = group[:, coordinate]
                    try:
                        Pm[k, coordinate] = fsum(column) / len(group)
                    except OverflowError:
                        # L'inertie finie de l'affectation précédente borne les
                        # écarts intra-groupe ; une grande origine peut, elle,
                        # faire déborder la somme des coordonnées brutes.
                        origin = column[0]
                        Pm[k, coordinate] = origin + fsum(column - origin) / len(group)
        new_labels, inertia = _assign(X, Pm)
        history.append(inertia)
        labels_stable = np.array_equal(labels, new_labels)
        if shifts is not None:
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                shift = float(np.max(np.hypot.reduce(Pm - Pc, axis=1)))
            if not np.isfinite(shift):
                raise ValueError("Le déplacement des centres dépasse float64.")
            shifts.append(shift)
            converged = shift <= tol
        else:
            converged = labels_stable
        Pc, labels = Pm, new_labels
        if converged:
            break

    return KMeansResult(
        centers=Pc,
        labels=labels,
        clusters=tuple(X[labels == k] for k in range(K)),
        inertia=inertia,
        inertia_history=np.asarray(history, dtype=np.float64),
        n_iter=n_iter,
        converged=converged,
        labels_stable=labels_stable,
        stop_reason=stop_criterion if converged else "max_iter",
        center_shift_history=None if shifts is None else np.asarray(shifts, dtype=np.float64),
    )
