"""Solutions connues, invariants de Lloyd, limites et oracle indépendant."""

from copy import deepcopy
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from lloyd_kmeans.generation import generate_gaussian_clouds
from lloyd_kmeans.kmeans import KMeansResult, kmeans


def assert_partition(X, result, K, *, stationary=True):
    """Contrôle les sorties avec SciPy, sans appeler les fonctions internes de Lloyd."""
    X = np.asarray(X, dtype=np.float64)
    assert isinstance(result, KMeansResult)
    assert result.centers.shape == (K, X.shape[1])
    assert result.centers.dtype == np.float64
    assert result.labels.shape == (len(X),)
    assert result.labels.dtype == np.int64
    assert np.isfinite(result.centers).all()
    distances = cdist(X, result.centers)
    assert_array_equal(result.labels, distances.argmin(axis=1))
    index = result.labels[:, None] == np.arange(K)
    assert_array_equal(index.sum(axis=1), np.ones(len(X), dtype=int))
    assert len(result.clusters) == K
    assert sum(map(len, result.clusters)) == len(X)
    for k, group in enumerate(result.clusters):
        assert_array_equal(group, X[index[:, k]])
        if stationary and len(group):
            assert_allclose(result.centers[k], group.mean(axis=0), rtol=2e-13, atol=2e-14)
    residuals = X - result.centers[result.labels]
    inertia = np.sum(residuals**2)
    assert_allclose(result.inertia, inertia, rtol=2e-13, atol=0)
    assert result.inertia >= 0
    assert result.inertia_history.shape == (result.n_iter + 1,)
    assert result.inertia == result.inertia_history[-1]
    # Marge d'arrondi proportionnelle à chaque valeur, pas de seuil absolu 1.
    history = result.inertia_history
    assert np.all(np.diff(history) <= 2e-13 * np.maximum(history[:-1], history[1:]))
    assert result.n_iter >= 1
    if stationary:
        assert result.converged


def test_manual_solution_and_initial_objective():
    X = [[0], [2], [10], [12]]
    result = kmeans(X, 2, init=[[0], [10]])
    assert_partition(X, result, 2)
    assert_array_equal(result.centers, [[1], [11]])
    assert_array_equal(result.labels, [0, 0, 1, 1])
    assert_allclose(result.inertia_history, [8, 4], rtol=1e-14)
    assert result.n_iter == 1


def test_multiple_iterations_and_limit_semantics():
    X = np.array([[0], [2], [3], [10]])
    init = [[0], [2]]
    first = kmeans(X, 2, init=init, max_iter=1)
    assert_partition(X, first, 2, stationary=False)
    assert not first.converged
    assert first.n_iter == 1
    assert_allclose(first.centers, [[0], [5]])
    assert_array_equal(first.labels, [0, 0, 1, 1])
    assert not np.allclose(first.centers[0], first.clusters[0].mean(axis=0))
    result = kmeans(X, 2, init=init)
    assert_partition(X, result, 2)
    assert result.n_iter == 3
    assert_allclose(result.centers, [[5 / 3], [10]])
    assert_array_equal(result.labels, [0, 0, 0, 1])
    # Une convergence atteinte à la dernière itération reste signalée True.
    at_limit = kmeans(X, 2, init=init, max_iter=3)
    assert at_limit.converged
    assert_array_equal(at_limit.centers, result.centers)


@pytest.mark.parametrize("dimension", [1, 2, 3, 10])
@pytest.mark.parametrize("K", [1, 2, 5])
def test_random_data_invariants(dimension, K):
    X = np.random.default_rng(2026).normal(size=(47, dimension))
    result = kmeans(X, K, rng=42)
    assert_partition(X, result, K)
    # Un point fixe ne bouge plus lorsqu'on le réutilise comme initialisation.
    restarted = kmeans(X, K, init=result.centers)
    assert restarted.converged and restarted.n_iter == 1
    assert_array_equal(restarted.centers, result.centers)
    assert_array_equal(restarted.labels, result.labels)


@pytest.mark.parametrize("dimension", [1, 3, 10])
def test_one_cluster_is_global_mean(dimension):
    X = np.random.default_rng(32).normal(size=(31, dimension))
    result = kmeans(X, 1, rng=0)
    assert_partition(X, result, 1)
    assert_allclose(result.centers, X.mean(axis=0, keepdims=True), atol=1e-15)
    assert result.n_iter == 1


def test_single_observation():
    result = kmeans([[2, 3, 4]], 1, rng=0)
    assert_partition([[2, 3, 4]], result, 1)
    assert result.inertia == 0


def test_k_equals_n_on_distinct_points():
    X = np.array([[0, 1], [2, 3], [8, 9], [-3, 4]])
    result = kmeans(X, len(X), rng=3)
    assert_partition(X, result, len(X))
    assert result.inertia == 0
    assert_array_equal(np.bincount(result.labels), np.ones(len(X), dtype=int))


@pytest.mark.parametrize("K", [1, 2, 5])
def test_identical_points_ties_and_empty_clusters(K):
    X = np.tile([3.0, -2.0], (5, 1))
    result = kmeans(X, K, rng=0)
    assert_partition(X, result, K)
    assert_array_equal(result.labels, np.zeros(5, dtype=int))
    assert result.inertia == 0
    assert result.n_iter == 1


def test_final_exact_tie_goes_to_lowest_index():
    X = [[-2], [0], [1], [1]]
    result = kmeans(X, 2, init=[[-1], [1]])
    assert_partition(X, result, 2)
    assert_array_equal(result.centers, [[-1], [1]])
    assert_array_equal(result.labels, [0, 0, 1, 1])
    assert_allclose(result.inertia, 2)


def test_empty_cluster_keeps_previous_prototype():
    X = [[0], [2], [10], [12]]
    result = kmeans(X, 3, init=[[0], [100], [10]])
    assert_partition(X, result, 3)
    assert_array_equal(result.centers, [[1], [100], [11]])
    assert result.clusters[1].shape == (0, 1)


def test_repeated_initial_prototypes_and_fewer_distinct_points_than_k():
    X = [[0, 0], [0, 0], [4, 4], [4, 4]]
    result = kmeans(X, 3, init=[[0, 0]] * 3)
    assert_partition(X, result, 3)
    assert len(np.unique(result.labels)) == 2


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.int16, np.float32, np.float64])
def test_dtypes_input_preservation_and_output_ownership(dtype):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=dtype)
    init = X[[0, 3]].copy()
    before_X, before_init = X.copy(), init.copy()
    result = kmeans(X, 2, init=init)
    assert_partition(X, result, 2)
    assert_array_equal(X, before_X)
    assert_array_equal(init, before_init)
    for output in (result.centers, result.labels, *result.clusters):
        assert not np.shares_memory(output, X)
        assert not np.shares_memory(output, init)
    result.centers[:] = 99
    result.clusters[0][:] = 77
    assert_array_equal(X, before_X)
    assert_array_equal(init, before_init)


def test_noncontiguous_and_readonly_inputs():
    X = np.arange(120.0).reshape(20, 6)[:, ::2]
    init = X[::7]
    X.setflags(write=False)
    init.setflags(write=False)
    result = kmeans(X, 3, init=init)
    assert_partition(X, result, 3)


def test_seed_and_initialization_protocol():
    X = np.random.default_rng(1).normal(size=(19, 3))
    random = kmeans(X, np.int64(4), rng=np.int64(42), max_iter=np.int64(30))
    replay = kmeans(X, 4, rng=42)
    chosen = np.random.default_rng(42).choice(len(X), size=4, replace=False)
    explicit = kmeans(X, 4, init=X[chosen])
    for result in (replay, explicit):
        assert_array_equal(result.centers, random.centers)
        assert_array_equal(result.labels, random.labels)
        assert_array_equal(result.inertia_history, random.inertia_history)


def test_generator_advances_only_for_random_initialization():
    X = np.arange(50.0).reshape(25, 2)
    rng, replay = np.random.default_rng(42), np.random.default_rng(42)
    first = kmeans(X, 3, rng=rng)
    chosen = replay.choice(len(X), size=3, replace=False)
    assert rng.bit_generator.state == replay.bit_generator.state
    assert_array_equal(first.centers, kmeans(X, 3, init=X[chosen]).centers)
    before = deepcopy(rng.bit_generator.state)
    kmeans(X, 3, init=X[chosen], rng=rng)
    assert rng.bit_generator.state == before
    kmeans(X, 3, rng=rng)
    assert rng.bit_generator.state != before


@pytest.mark.parametrize("rng", [None, 42, np.random.default_rng(42)])
def test_global_random_state_is_unchanged(rng):
    before = np.random.get_state()
    result = kmeans([[0], [1], [8], [9]], 2, rng=rng)
    assert_partition([[0], [1], [8], [9]], result, 2)
    assert_equal(np.random.get_state(), before)


@pytest.mark.parametrize("bad", [
    1, [1, 2], [], [[], []], np.zeros((2, 2, 2)), [[0, 1], [2]],
    [[np.nan]], [[np.inf]], [[-np.inf]], [[1j]], [["1"]], [[None]],
])
def test_invalid_data_does_not_advance_rng(bad):
    rng = np.random.default_rng(42)
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError):
        kmeans(bad, 1, rng=rng)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("bad", [0, -1, 5, True, np.bool_(False), 2.0, "2", None, [2], 2**100])
def test_invalid_k(bad):
    with pytest.raises(ValueError, match="K"):
        kmeans([[0], [1], [2], [3]], bad, rng=0)


@pytest.mark.parametrize("bad", [0, -1, True, np.bool_(True), 2.0, "2", None, [2]])
def test_invalid_max_iter_does_not_advance_rng(bad):
    rng = np.random.default_rng(42)
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="max_iter"):
        kmeans([[0], [1]], 1, rng=rng, max_iter=bad)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("bad", [-1, 1.5, True, np.bool_(False), "42", [42], np.random.RandomState(0)])
def test_invalid_rng(bad):
    with pytest.raises(ValueError, match="rng"):
        kmeans([[0], [1]], 1, rng=bad)


@pytest.mark.parametrize("bad", [
    [], [0, 1], [[0]], [[0, 1, 2]], [[0, 1], [2, 3]],
    [[[0, 1]]], [[np.nan, 0]], [[np.inf, 0]], [[1j, 0]], [["0", "1"]],
])
def test_invalid_initialization(bad):
    rng = np.random.default_rng(42)
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError):
        kmeans([[0, 0], [1, 1]], 1, init=bad, rng=rng)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("seed", range(8))
def test_matches_sklearn_with_shared_initialization(seed):
    # Données non séparées : contrôle des itérations, pas uniquement des blobs faciles.
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(71, 4))
    init = X[rng.choice(len(X), 4, replace=False)].copy()
    result = kmeans(X, 4, init=init)
    reference = KMeans(n_clusters=4, init=init, n_init=1, algorithm="lloyd",
                       tol=0, max_iter=300).fit(X)
    assert_partition(X, result, 4)
    assert_array_equal(result.labels, reference.labels_)
    assert_allclose(result.centers, reference.cluster_centers_, rtol=2e-12, atol=2e-13)
    assert_allclose(result.inertia, reference.inertia_, rtol=2e-12)


def test_integration_with_u2_in_three_dimensions():
    centers = [[-8, 0, 1], [8, 2, -1], [0, 10, 3]]
    X, truth = generate_gaussian_clouds(centers, [17, 23, 19], std=0.1, rng=2026)
    result = kmeans(X, 3, init=centers)
    assert_partition(X, result, 3)
    assert_array_equal(result.labels, truth)


def test_objective_decreases_for_each_half_step_and_history_is_observed():
    X = np.random.default_rng(28).normal(size=(61, 3))
    init = X[:4]
    initial_distances = cdist(X, init)
    previous_labels = initial_distances.argmin(axis=1)
    previous_inertia = np.sum(initial_distances.min(axis=1)**2)
    full = kmeans(X, 4, init=init)
    assert_partition(X, full, 4)
    # Chaque préfixe indépendant donne accès aux vrais prototypes intermédiaires.
    for limit in range(1, full.n_iter + 1):
        prefix = kmeans(X, 4, init=init, max_iter=limit)
        distances = cdist(X, prefix.centers)
        recentered_inertia = np.sum(distances[np.arange(len(X)), previous_labels]**2)
        assert recentered_inertia <= previous_inertia + 2e-13 * previous_inertia
        assert prefix.inertia <= recentered_inertia + 2e-13 * recentered_inertia
        assert_allclose(prefix.inertia, full.inertia_history[limit], rtol=2e-13)
        previous_labels, previous_inertia = prefix.labels, prefix.inertia


def test_global_optimum_for_one_tiny_problem_by_exhaustive_partition_search():
    X = np.array([[0.0], [1.0], [7.0], [9.0]])
    objectives = []
    for assignment in product(range(2), repeat=len(X)):
        labels = np.asarray(assignment)
        if len(np.unique(labels)) != 2:
            continue
        centers = np.array([X[labels == k].mean(axis=0) for k in range(2)])
        objectives.append(np.sum((X - centers[labels])**2))
    result = kmeans(X, 2, init=[[0], [9]])
    assert_partition(X, result, 2)
    assert_allclose(result.inertia, min(objectives), rtol=1e-14)


@pytest.mark.parametrize("factor", [1e-140, 1e-5, 7, 1e140])
def test_scaling_equivariance_and_squared_inertia(factor):
    X = np.array([[-2.0, 1], [0, 1], [8, 3], [10, 3]])
    baseline = kmeans(X, 2, init=X[[0, 2]])
    result = kmeans(X * factor, 2, init=X[[0, 2]] * factor)
    assert_array_equal(result.labels, baseline.labels)
    assert_allclose(result.centers / factor, baseline.centers, rtol=2e-14, atol=1e-15)
    assert_allclose(result.inertia / factor**2, baseline.inertia, rtol=2e-13)


def test_translation_and_coordinate_permutation():
    X = np.random.default_rng(44).normal(size=(30, 3))
    result = kmeans(X, 3, init=X[:3])
    translated = kmeans(X + [10, -4, 2], 3, init=X[:3] + [10, -4, 2])
    permuted = kmeans(X[:, ::-1], 3, init=X[:3, ::-1])
    assert_array_equal(translated.labels, result.labels)
    assert_array_equal(permuted.labels, result.labels)
    assert_allclose(translated.centers - [10, -4, 2], result.centers, atol=1e-14)
    assert_allclose(permuted.centers[:, ::-1], result.centers, atol=1e-14)
    assert_allclose(translated.inertia, result.inertia, rtol=2e-13)


def test_large_common_offset_with_representable_dispersion():
    X = 1e12 + np.array([[0, 0], [2, 0], [10, 4], [12, 4]])
    result = kmeans(X, 2, init=X[[0, 2]])
    assert_partition(X, result, 2)
    assert_allclose(result.centers - 1e12, [[1, 0], [11, 4]], atol=3e-4, rtol=0)


def test_mean_does_not_overflow_on_large_constant_coordinates():
    X = np.tile([1.6e308, -1.6e308, 0], (10, 1))
    result = kmeans(X, 1, rng=0)
    assert_array_equal(result.centers, X[:1])
    assert result.converged
    assert result.inertia == 0


def test_recentring_preserves_a_representable_mean_at_large_offset():
    base = 1e12
    step = np.spacing(base)
    offsets = np.array([-1, -1, -1, -2, -2, -1, 0, 0])
    X = base + offsets[:, None] * step
    result = kmeans(X, 1, init=[[base - step]])
    assert result.converged
    assert_array_equal(result.centers, [[base - step]])
    assert_allclose(result.inertia_history, [4 * step**2, 4 * step**2], rtol=2e-14)


@pytest.mark.parametrize("values", [[-1e150, 1, 1e150], [1e150, -1e150, 1]])
def test_mean_retains_small_term_despite_cancellation(values):
    result = kmeans(np.array(values)[:, None], 1, init=[[0]])
    assert result.converged
    assert_array_equal(result.centers, [[1 / 3]])


@pytest.mark.parametrize("X, init, message", [
    ([[1e308], [-1e308]], [[1e308]], "distances"),
    ([[1e200], [-1e200]], [[0]], "inertie"),
    ([[1e-200], [-1e-200]], [[0]], "inertie"),
])
def test_unrepresentable_numeric_operations_are_reported(X, init, message):
    with pytest.raises(ValueError, match=message):
        kmeans(X, 1, init=init)


def test_tiny_individual_squares_can_have_representable_total():
    X = np.tile([[-1e-163], [1e-163]], (5000, 1))
    result = kmeans(X, 1, init=[[0]])
    assert result.converged
    assert result.inertia > 0
    assert abs(result.inertia - 1e-322) <= np.nextafter(0.0, 1.0)
