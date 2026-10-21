"""Critères d'arrêt : mesure indépendante et cohérence des sorties anticipées."""
from copy import deepcopy
from math import dist

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
from scipy.spatial.distance import cdist

from lloyd_kmeans.kmeans import kmeans


def check_outputs(X, result):
    X = np.asarray(X, dtype=float)
    distances = cdist(X, result.centers)
    assert_array_equal(result.labels, distances.argmin(axis=1))
    assert_allclose(result.inertia, np.sum((X - result.centers[result.labels])**2), rtol=2e-13)
    for k, group in enumerate(result.clusters):
        assert_array_equal(group, X[result.labels == k])
    assert result.inertia == result.inertia_history[-1]


@pytest.mark.parametrize('seed', range(8))
def test_default_equals_explicit_historical_criterion(seed):
    X = np.random.default_rng(seed).normal(size=(51, 4))
    implicit = kmeans(X, 4, rng=seed)
    explicit = kmeans(X, 4, rng=seed, stop_criterion='labels', tol=1e200)
    for name in ('centers', 'labels', 'inertia_history', 'inertia', 'n_iter', 'converged'):
        assert_array_equal(getattr(implicit, name), getattr(explicit, name))
    for a, b in zip(implicit.clusters, explicit.clusters):
        assert_array_equal(a, b)
    assert implicit.labels_stable and implicit.stop_reason == 'labels'
    assert implicit.center_shift_history is None


def test_reviewer_counterexample_distinguishes_stability_and_movement():
    X, init = [[0], [2]], [[100]]
    historical = kmeans(X, 1, init=init)
    first = kmeans(X, 1, init=init, stop_criterion='center_shift', tol=0, max_iter=1)
    complete = kmeans(X, 1, init=init, stop_criterion='center_shift', tol=0)
    assert historical.n_iter == 1 and historical.converged
    assert first.labels_stable and not first.converged
    assert first.stop_reason == 'max_iter'
    assert_array_equal(first.center_shift_history, [99])
    assert complete.n_iter == 2 and complete.converged and complete.labels_stable
    assert complete.stop_reason == 'center_shift'
    assert_array_equal(complete.center_shift_history, [99, 0])
    assert_array_equal(complete.centers, [[1]])
    check_outputs(X, complete)


@pytest.mark.parametrize('tol, iterations, converged', [(98.0, 1, False), (99.0, 1, True), (100.0, 1, True), (0.0, 1, False)])
def test_inclusive_threshold_at_iteration_limit(tol, iterations, converged):
    result = kmeans([[0], [2]], 1, init=[[100]], max_iter=iterations,
                    stop_criterion='center_shift', tol=tol)
    assert result.converged == converged
    assert result.stop_reason == ('center_shift' if converged else 'max_iter')
    assert_array_equal(result.center_shift_history, [99])


def test_large_threshold_can_stop_before_stationary_labels():
    X = [[0], [2], [3], [10]]
    result = kmeans(X, 2, init=[[0], [2]], stop_criterion='center_shift', tol=3)
    assert result.n_iter == 1 and result.converged and not result.labels_stable
    assert_array_equal(result.center_shift_history, [3])
    assert_array_equal(result.centers, [[0], [5]])
    assert result.centers[0, 0] != result.clusters[0].mean()
    check_outputs(X, result)


@pytest.mark.parametrize('dimension', [1, 2, 3, 10])
@pytest.mark.parametrize('K', [1, 2, 5])
def test_measured_successive_shifts_against_independent_prefixes(dimension, K):
    X = np.random.default_rng(782).normal(size=(35, dimension))
    init = X[:K].copy()
    full = kmeans(X, K, init=init, stop_criterion='center_shift', tol=0)
    previous = init
    shifts = []
    for limit in range(1, full.n_iter + 1):
        prefix = kmeans(X, K, init=init, stop_criterion='center_shift', tol=0, max_iter=limit)
        shifts.append(max(dist(a, b) for a, b in zip(previous, prefix.centers)))
        check_outputs(X, prefix)
        assert prefix.converged == (shifts[-1] == 0)
        previous = prefix.centers
    assert_allclose(full.center_shift_history, shifts, rtol=3e-15, atol=0)
    assert full.center_shift_history[-1] == 0
    assert full.labels_stable


def test_maximum_of_euclidean_norms_not_coordinate_max_or_frobenius():
    # Les deux centres se déplacent de 5 : ni 4 (max coordonnée), ni sqrt(50).
    X = [[3, 4], [3, 4], [103, 104], [103, 104]]
    init = [[0, 0], [100, 100]]
    result = kmeans(X, 2, init=init, stop_criterion='center_shift', tol=5)
    assert result.n_iter == 1
    assert_array_equal(result.center_shift_history, [5])
    below = kmeans(X, 2, init=init, stop_criterion='center_shift', tol=4)
    assert below.n_iter == 2


@pytest.mark.parametrize('K', [1, 2, 5])
def test_identical_points_empty_clusters_and_ties(K):
    X = np.tile([3., -2.], (5, 1))
    result = kmeans(X, K, init=X[:K], stop_criterion='center_shift', tol=0)
    assert result.n_iter == 1 and result.converged
    assert_array_equal(result.labels, np.zeros(5, dtype=int))
    assert_array_equal(result.center_shift_history, [0])
    check_outputs(X, result)


def test_empty_cluster_preserves_its_prototype_and_final_tie():
    empty = kmeans([[0], [2], [10], [12]], 3, init=[[0], [100], [10]],
                   stop_criterion='center_shift', tol=1)
    assert_array_equal(empty.centers, [[1], [100], [11]])
    assert empty.clusters[1].shape == (0, 1)
    assert_array_equal(empty.center_shift_history, [1])
    tied = kmeans([[-2], [0], [1], [1]], 2, init=[[-1], [1]],
                  stop_criterion='center_shift', tol=0)
    assert_array_equal(tied.labels, [0, 0, 1, 1])


@pytest.mark.parametrize('tol', [-1, True, np.bool_(False), np.nan, np.inf, -np.inf,
                                  '1', None, [1], np.array(1), 1j, 10**1000])
@pytest.mark.parametrize('criterion', ['labels', 'center_shift'])
def test_invalid_tolerance_does_not_advance_rng(tol, criterion):
    rng = np.random.default_rng(3)
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match='tol'):
        kmeans([[0], [2]], 1, rng=rng, stop_criterion=criterion, tol=tol)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize('bad', ['shift', '', None, True, 2, ['labels'], np.array(['labels'])])
def test_invalid_criterion(bad):
    with pytest.raises(ValueError, match='stop_criterion'):
        kmeans([[0], [2]], 1, stop_criterion=bad)


@pytest.mark.parametrize('factor', [1e-140, 1e-5, 1, 1e140])
def test_shift_scaling_without_squaring(factor):
    X = np.array([[3., 4], [3, 4]]) * factor
    result = kmeans(X, 1, init=[[0, 0]], stop_criterion='center_shift', tol=6 * factor)
    assert result.n_iter == 1 and result.converged
    assert_allclose(result.center_shift_history, [5 * factor], rtol=2e-15)
    assert_array_equal(result.centers, X[:1])


def test_huge_common_offset_and_tiny_shift_without_square_underflow():
    X = np.tile([1.6e308, -1.6e308, 0], (10, 1))
    constant = kmeans(X, 1, init=X[:1], stop_criterion='center_shift', tol=0)
    assert_array_equal(constant.center_shift_history, [0])
    assert constant.converged
    # Inertie initiale subnormale mais représentable, déplacement sans carré.
    tiny = kmeans([[1e-161], [1e-161]], 1, init=[[0]],
                  stop_criterion='center_shift', tol=0, max_iter=1)
    assert_array_equal(tiny.center_shift_history, [1e-161])
    assert not tiny.converged


@pytest.mark.parametrize('tol', [0, np.int64(0), np.float32(0.5), np.float64(1)])
def test_real_scalar_tolerances(tol):
    result = kmeans([[0], [2]], 1, init=[[1]], stop_criterion='center_shift', tol=tol)
    assert result.converged and result.n_iter == 1
