"""Contrats, reproductibilité et contrôles statistiques de génération gaussienne."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal
from scipy.stats import kstest

from lloyd_kmeans.generation import generate_gaussian_clouds


@pytest.mark.parametrize("dimension", [1, 2, 3, 10])
@pytest.mark.parametrize("k", [1, 2, 6])
def test_shapes_types_and_group_sizes(dimension, k):
    counts = np.arange(1, k + 1)
    X, y = generate_gaussian_clouds(np.zeros((k, dimension)), counts, rng=2026)
    assert X.shape == (sum(counts), dimension)
    assert y.shape == (sum(counts),)
    assert X.dtype == np.float64
    assert y.dtype == np.int64
    assert np.isfinite(X).all()
    assert_array_equal(np.bincount(y), counts)


def test_common_count_and_order_without_shuffle():
    X, y = generate_gaussian_clouds([[1, 2], [3, 4]], 3, std=0, rng=0, shuffle=False)
    assert_array_equal(y, [0, 0, 0, 1, 1, 1])
    assert_array_equal(X, [[1, 2]] * 3 + [[3, 4]] * 3)


def test_shuffle_preserves_observations_and_labels():
    params = dict(centers=[[0, 1], [4, 5], [-4, 2]], n_samples=[11, 13, 17], rng=7)
    X, y = generate_gaussian_clouds(**params, shuffle=False)
    mixed_X, mixed_y = generate_gaussian_clouds(**params)
    # Une coordonnée tirée est unique ici : on retrouve la permutation exacte.
    original_order = np.argsort(X[:, 0])
    mixed_order = np.argsort(mixed_X[:, 0])
    assert_array_equal(mixed_X[mixed_order], X[original_order])
    assert_array_equal(mixed_y[mixed_order], y[original_order])
    assert not np.array_equal(mixed_y, y)


@pytest.mark.parametrize("shuffle", [False, True])
def test_seed_reproducibility_and_different_seeds(shuffle):
    params = dict(centers=[[0, 1], [5, 6]], n_samples=[15, 20], shuffle=shuffle)
    first = generate_gaussian_clouds(**params, rng=42)
    repeat = generate_gaussian_clouds(**params, rng=np.int64(42))
    other = generate_gaussian_clouds(**params, rng=43)
    for actual, expected in zip(first, repeat):
        assert_array_equal(actual, expected)
    assert not np.array_equal(first[0], other[0])


def test_generator_advances_and_can_be_replayed():
    rng = np.random.default_rng(23)
    replay = np.random.default_rng(23)
    first = generate_gaussian_clouds([[0, 0]], 20, rng=rng)
    second = generate_gaussian_clouds([[0, 0]], 20, rng=rng)
    assert not np.array_equal(first[0], second[0])
    for expected in (first, second):
        actual = generate_gaussian_clouds([[0, 0]], 20, rng=replay)
        assert_array_equal(actual[0], expected[0])
        assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize("rng", [None, 42, np.random.default_rng(42)])
def test_legacy_global_random_state_unchanged(rng):
    before = np.random.get_state()
    generate_gaussian_clouds([[0, 0], [2, 3]], 10, rng=rng)
    assert_equal(np.random.get_state(), before)


@pytest.mark.parametrize("std", [0.7, [0.5, 1.5], [[0.5, 1.0], [1.5, 0.3]]])
def test_gaussian_moments_independence_and_distribution(std):
    centers = np.array([[1.5, -2.0], [5.0, 3.0]])
    n = 30_000
    X, y = generate_gaussian_clouds(centers, n, std, rng=2026)
    scales = np.asarray(std)
    if scales.shape == (2,):
        scales = scales[:, None]
    scales = np.broadcast_to(scales, centers.shape)
    for label in range(2):
        # Standardisation : toutes les coordonnées doivent suivre N(0, 1).
        z = (X[y == label] - centers[label]) / scales[label]
        assert np.all(np.abs(z.mean(axis=0)) < 6 / np.sqrt(n))
        covariance = np.cov(z, rowvar=False, ddof=1)
        assert np.all(np.abs(np.diag(covariance) - 1) < 6 * np.sqrt(2 / (n - 1)))
        assert abs(covariance[0, 1]) < 6 / np.sqrt(n - 1)
        # Complète les moments par un contrôle de la forme de la distribution.
        # Seuil très conservateur, graine fixe ; ce test ne « prouve » pas la loi.
        for coordinate in z.T:
            assert kstest(coordinate, "norm").pvalue > 1e-6


def test_zero_scales_and_vector_means_groups_even_when_k_equals_d():
    centers = np.array([[0, 1], [5, 6]])
    X, y = generate_gaussian_clouds(centers, 10, [0, 0.5], rng=4)
    assert_array_equal(X[y == 0], np.tile(centers[0], (10, 1)))
    assert np.all(np.std(X[y == 1], axis=0) > 0)


def test_constant_coordinates_with_matrix_scales():
    centers = np.array([[1, 2], [3, 4]])
    X, y = generate_gaussian_clouds(centers, 5, [[0, 1], [2, 0]], rng=2)
    assert_array_equal(X[y == 0, 0], np.ones(5))
    assert_array_equal(X[y == 1, 1], np.full(5, 4))


def test_input_arrays_are_unchanged_and_noncontiguous_inputs_work():
    centers = np.arange(12, dtype=np.int16).reshape(2, 6)[:, ::2]
    counts = np.array([5, 7], dtype=np.uint8)
    scales = np.arange(12, dtype=np.float32).reshape(2, 6)[:, ::2]
    before = [array.copy() for array in (centers, counts, scales)]
    X, _ = generate_gaussian_clouds(centers, counts, scales, rng=0)
    assert X.shape == (12, 3)
    for actual, expected in zip((centers, counts, scales), before):
        assert_array_equal(actual, expected)


@pytest.mark.parametrize("bad", [
    1, [0, 1], [], [[], []], np.zeros((2, 1, 1)),
    [[0, 1], [2]], [[np.nan]], [[np.inf]], [[-np.inf]],
    [[1j]], [["1"]], [[None]], [[True]],
])
def test_invalid_centers(bad):
    with pytest.raises(ValueError):
        generate_gaussian_clouds(bad, 3, rng=0)


@pytest.mark.parametrize("bad", [
    0, -1, 2.0, True, [2], [2, 0], [2, -1], [2.0, 3.0],
    [[2, 3]], [True, False], [True, 2], [2, np.bool_(True)],
    "2", None, np.nan, np.inf, 2**100,
    np.array([np.iinfo(np.uint64).max] * 2, dtype=np.uint64),
])
def test_invalid_counts(bad):
    with pytest.raises(ValueError):
        generate_gaussian_clouds([[0, 0], [1, 1]], bad, rng=0)


@pytest.mark.parametrize("bad", [
    -1, np.nan, np.inf, -np.inf, True, "1", 1j, None, [],
    [1], [1, -1], [1, np.nan], [1, np.inf], [[1], [2]],
    [[1, 2, 3], [4, 5, 6]], np.ones((2, 2, 1)),
])
def test_invalid_scales(bad):
    with pytest.raises(ValueError):
        generate_gaussian_clouds([[0, 0], [1, 1]], 3, bad, rng=0)


@pytest.mark.parametrize("bad", [-1, 1.5, True, np.bool_(False), "42", [42], np.random.RandomState(0)])
def test_invalid_rng(bad):
    with pytest.raises(ValueError, match="rng"):
        generate_gaussian_clouds([[0, 0]], 3, rng=bad)


@pytest.mark.parametrize("bad", [0, 1, "yes", None])
def test_invalid_shuffle_does_not_advance_generator(bad):
    rng = np.random.default_rng(42)
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="shuffle"):
        generate_gaussian_clouds([[0, 0]], 3, rng=rng, shuffle=bad)
    assert rng.bit_generator.state == before


def test_overflow_in_draws_is_reported():
    with pytest.raises(ValueError, match="float64"):
        generate_gaussian_clouds([[1e308]], 100, 1e308, rng=0)
