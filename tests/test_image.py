"""Ordre spatial, types, arrondi, groupes vides et contrat de Lloyd."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from lloyd_kmeans.image import image_to_pixels, reconstruct_image, segment_image
from lloyd_kmeans.kmeans import kmeans


def test_flatten_and_inverse_preserve_pixels_and_independent_storage():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    for view in (image, image[:, ::-1], image.transpose(1, 0, 2)):
        pixels = image_to_pixels(view)
        expected = np.array([view[i, j] for i in range(view.shape[0])
                             for j in range(view.shape[1])], dtype=float)
        assert_array_equal(pixels, expected)
        assert pixels.dtype == np.float64 and not np.shares_memory(pixels, view)
        restored = reconstruct_image(np.arange(len(pixels)), pixels, view.shape)
        assert_array_equal(restored, view)
        assert not np.shares_memory(restored, view)
        pixels[0] = 255
        assert_array_equal(image, np.arange(18, dtype=np.uint8).reshape(2, 3, 3))


@pytest.mark.parametrize("image", [np.zeros((2, 2), dtype=np.uint8),
    np.zeros((2, 2, 4), dtype=np.uint8), np.zeros((0, 2, 3), dtype=np.uint8),
    np.zeros((2, 0, 3), dtype=np.uint8), np.zeros((2, 2, 3)),
    np.zeros((2, 2, 3), dtype=np.uint16), np.zeros((2, 2, 3), dtype=bool),
    np.full((1, 1, 3), np.nan), np.full((1, 1, 3), "red")])
def test_reject_ambiguous_image_representation(image):
    with pytest.raises(ValueError):
        image_to_pixels(image)


def test_reconstruction_rounds_half_to_even_and_keeps_assigned_labels():
    centers = np.array([[0.5, 1.5, 254.5], [2.5, 3.5, 255]])
    labels = np.array([1, 0, 1, 0])
    image = reconstruct_image(labels, centers, (2, 2, 3))
    assert_array_equal(image, [[[2, 4, 255], [0, 2, 254]],
                               [[2, 4, 255], [0, 2, 254]]])
    assert image.dtype == np.uint8
    assert_array_equal(centers, [[0.5, 1.5, 254.5], [2.5, 3.5, 255]])


@pytest.mark.parametrize("labels,centers,shape", [
    ([0], [[0, 0, 0]], (1, 1)), ([0], [[0, 0, 0]], (True, 1, 3)),
    ([0], [[0, 0, 0]], (0, 1, 3)), ([0], [[0, 0, 0]], (1.0, 1, 3)),
    ([0], [[0, 0, 0]], (1, 1, 4)), ([0.0], [[0, 0, 0]], (1, 1, 3)),
    ([True], [[0, 0, 0]], (1, 1, 3)), ([-1], [[0, 0, 0]], (1, 1, 3)),
    ([1], [[0, 0, 0]], (1, 1, 3)), ([[0]], [[0, 0, 0]], (1, 1, 3)),
    ([0, 0], [[0, 0, 0]], (1, 1, 3)), ([0], [], (1, 1, 3)),
    ([0], [[0, 0]], (1, 1, 3)), ([0], [[-0.1, 0, 0]], (1, 1, 3)),
    ([0], [[256, 0, 0]], (1, 1, 3)), ([0], [[np.nan, 0, 0]], (1, 1, 3)),
    ([0], [[np.inf, 0, 0]], (1, 1, 3)), ([0], [[1j, 0, 0]], (1, 1, 3)),
])
def test_reconstruction_rejects_invalid_shape_labels_or_centers(labels, centers, shape):
    with pytest.raises(ValueError):
        reconstruct_image(labels, centers, shape)


def test_segmentation_reuses_u3_exactly_and_seed_or_explicit_init_replays():
    image = np.array([[[0, 10, 20], [10, 20, 30], [255, 240, 230]],
                      [[250, 235, 225], [0, 10, 20], [10, 20, 30]]], dtype=np.uint8)
    original = image.copy()
    X = image_to_pixels(image)
    segmented, result = segment_image(image, 2, rng=2026)
    direct = kmeans(X, 2, rng=2026)
    indices = np.random.default_rng(2026).choice(6, size=2, replace=False)
    replay_image, replay = segment_image(image, 2, init=X[indices])
    for name in ("labels", "centers", "inertia_history"):
        assert_array_equal(getattr(result, name), getattr(direct, name))
        assert_array_equal(getattr(result, name), getattr(replay, name))
    assert_array_equal(segmented, replay_image)
    assert_array_equal(image, original)
    assert result.converged and result.inertia == direct.inertia
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            assert_array_equal(segmented[i, j], np.rint(result.centers[result.labels[i*3+j]]))


def test_duplicate_colors_and_empty_clusters_are_not_fabricated():
    image = np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8)
    segmented, result = segment_image(image, 4, rng=0)
    assert result.converged and result.inertia == 0
    assert_array_equal(np.bincount(result.labels, minlength=4), [4, 0, 0, 0])
    assert_array_equal(segmented, image)
    assert len(np.unique(segmented.reshape(-1, 3), axis=0)) == 1


@pytest.mark.parametrize("outside", [-1, 256, np.nan, np.inf])
def test_initial_colors_must_be_rgb_even_for_an_empty_cluster(outside):
    image = np.full((1, 2, 3), [10, 20, 30], dtype=np.uint8)
    with pytest.raises(ValueError, match="RGB"):
        segment_image(image, 2, init=[[10, 20, 30], [outside, 0, 0]])


def test_rounding_can_merge_occupied_clusters_into_one_output_color():
    labels = np.array([0, 1])
    centers = np.array([[10.1, 20, 30], [10.4, 20, 30]])
    image = reconstruct_image(labels, centers, (1, 2, 3))
    assert np.count_nonzero(np.bincount(labels)) == 2
    assert len(np.unique(image.reshape(-1, 3), axis=0)) == 1


def test_uint8_reconstruction_error_and_huygens_identity():
    image = np.array([[[0, 255, 0], [1, 254, 2], [1, 254, 3]]], dtype=np.uint8)
    segmented, result = segment_image(image, 1, rng=0)
    X = image_to_pixels(image)
    direct = np.sum((X - result.centers[result.labels]) ** 2)
    integer_sse = np.sum((X - segmented.reshape(-1, 3).astype(float)) ** 2)
    rounding_sse = len(X) * np.sum((result.centers - np.rint(result.centers)) ** 2)
    assert_allclose(direct, result.inertia)
    assert_allclose(integer_sse, direct + rounding_sse)
    assert 0 < integer_sse - direct <= 0.75 * len(X)


def test_nonconvergence_exposed_without_claiming_stationarity():
    image = np.array([[[v, v, v] for v in (0, 1, 2, 3, 10) ]], dtype=np.uint8)
    segmented, result = segment_image(image, 2, init=[[0, 0, 0], [1, 1, 1]], max_iter=1)
    assert result.n_iter == 1 and not result.converged
    assert_array_equal(segmented.reshape(-1, 3), np.rint(result.centers)[result.labels])
