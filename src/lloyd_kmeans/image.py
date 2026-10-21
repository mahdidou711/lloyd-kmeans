"""Quantification des couleurs d'une image RGB uint8 par Lloyd."""

import numpy as np

from lloyd_kmeans.kmeans import kmeans


def _rgb_centers(centers):
    centers = np.asarray(centers)
    if (centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0
            or centers.dtype.kind not in "biuf" or not np.isfinite(centers).all()
            or np.any(centers < 0) or np.any(centers > 255)):
        raise ValueError("centers doit être une matrice RGB finie (K, 3) sur [0, 255].")
    return centers


def image_to_pixels(image):
    """Copie une image uint8 (H, W, 3) en float64 (H*W, 3), ordre C.

    Le pixel (i, j) devient la ligne i*W+j. Les intensités restent sur [0, 255],
    sans normalisation, correction gamma, coordonnées spatiales ni sous-échantillon.
    Les images grises, RGBA, uint16 et flottantes sont refusées pour éviter toute
    interprétation implicite de leur échelle. Les vues non contiguës sont admises.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image doit avoir la forme non vide (H, W, 3).")
    if image.dtype != np.uint8:
        raise ValueError("image doit être de type uint8, avec des canaux RGB sur [0, 255].")
    return image.reshape(-1, 3, order="C").astype(np.float64)


def reconstruct_image(labels, centers, image_shape):
    """Reconstruit (H, W, 3) depuis les labels en ordre C et les centres RGB.

    Chaque canal du prototype est arrondi au plus proche avec np.rint (demi-entiers
    vers l'entier pair), puis converti en uint8. Aucune nouvelle affectation après
    cet arrondi : le pixel conserve son label de cluster. Au plus K couleurs sont présentes,
    parfois moins si des groupes sont vides ou des couleurs arrondies coïncident.
    L'inertie de Lloyd porte sur les centres flottants, pas sur cette image entière.
    """
    if (len(image_shape) != 3 or image_shape[2] != 3
            or any(not isinstance(n, (int, np.integer))
                   or isinstance(n, (bool, np.bool_)) or n < 1 for n in image_shape)):
        raise ValueError("image_shape doit être une forme non vide (H, W, 3).")
    centers = _rgb_centers(centers)
    labels = np.asarray(labels)
    if (labels.shape != (image_shape[0] * image_shape[1],)
            or labels.dtype.kind not in "iu" or np.any(labels < 0)
            or np.any(labels >= len(centers))):
        raise ValueError("labels doit contenir H*W indices entiers entre 0 et K-1.")
    palette = np.rint(centers).astype(np.uint8)
    return palette[labels].reshape(tuple(image_shape), order="C")


def segment_image(image, K, *, init=None, rng=None, max_iter=300):
    """Retourne (image segmentée uint8, KMeansResult), sans modifier les entrées.

    K, rng et max_iter suivent exactement le contrat de kmeans. init doit en plus
    contenir des couleurs RGB sur [0, 255], y compris pour les groupes vides.
    Cette contrainte est vérifiée avant le calcul de Lloyd. Une sortie non
    convergée reste accessible et doit être identifiée comme telle par l'appelant.
    Tous les pixels, y compris les couleurs répétées, participent au calcul.
    """
    pixels = image_to_pixels(image)
    if init is not None:
        init = _rgb_centers(init)
    result = kmeans(pixels, K, init=init, rng=rng, max_iter=max_iter)
    segmented = reconstruct_image(result.labels, result.centers, np.shape(image))
    return segmented, result
