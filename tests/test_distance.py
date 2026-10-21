"""Validation des distances euclidiennes et comparaison à un oracle SciPy."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import cdist

from lloyd_kmeans.distance import euclidean_distances


def test_manual_distances():
    distances = euclidean_distances([[0, 0], [3, 4]], [[0, 0], [0, 4], [3, 0]])
    assert distances.shape == (2, 3)
    assert_allclose(distances, [[0, 4, 3], [5, 3, 4]], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dimension", [1, 2, 3, 10])
@pytest.mark.parametrize("n, k", [(1, 1), (7, 3), (2, 5)])
def test_matches_scipy(dimension, n, k):
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(n, dimension))
    P = rng.normal(size=(k, dimension))
    distances = euclidean_distances(X, P)
    assert distances.shape == (n, k)
    assert distances.dtype == np.float64
    assert_allclose(distances, cdist(X, P), rtol=1e-12, atol=1e-12)


def test_identical_points_symmetry_and_nonnegativity():
    X = np.random.default_rng(42).normal(size=(6, 10))
    distances = euclidean_distances(X, X)
    assert_array_equal(np.diag(distances), np.zeros(6))
    assert_allclose(distances, distances.T, rtol=1e-12, atol=1e-12)
    assert np.all(distances >= 0)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32, np.float64])
def test_numeric_dtypes_and_inputs_unchanged(dtype):
    X = np.array([[0, 0], [255, 0]], dtype=dtype)
    P = np.array([[255, 0]], dtype=dtype)
    X_before, P_before = X.copy(), P.copy()
    assert_array_equal(euclidean_distances(X, P), [[255], [0]])
    assert_array_equal(X, X_before)
    assert_array_equal(P, P_before)


def test_noncontiguous_inputs():
    X = np.arange(60.0).reshape(6, 10)[:, ::2]
    P = X[::-2]
    assert_allclose(euclidean_distances(X, P), cdist(X, P), rtol=1e-12)


def test_nearby_points_with_large_common_offset():
    X = np.array([[1e12, 1e12]])
    P = X + [3, 4]
    assert_array_equal(euclidean_distances(X, P), [[5]])


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_representable_distances_at_extreme_scales(scale):
    distances = euclidean_distances([[0, 0]], [[3 * scale, 4 * scale]])
    assert_allclose(distances, [[5 * scale]], rtol=1e-12, atol=0)


@pytest.mark.parametrize("bad", [
    1, [1, 2], np.zeros((1, 1, 1)),
    [], np.empty((0, 2)), np.empty((2, 0)),
    [[np.nan, 0]], [[np.inf, 0]], [[-np.inf, 0]],
    [[1j, 0]], [["a", "b"]], [[None, 0]],
    [[1, 2], [3]],
])
@pytest.mark.parametrize("argument", ["X", "P"])
def test_invalid_matrices(bad, argument):
    inputs = {"X": [[0, 0]], "P": [[0, 0]]}
    inputs[argument] = bad
    with pytest.raises(ValueError):
        euclidean_distances(**inputs)


def test_incompatible_dimensions():
    with pytest.raises(ValueError, match="même nombre de coordonnées"):
        euclidean_distances(np.zeros((3, 2)), np.zeros((4, 3)))
