"""Two small deterministic demos; no external datasets or file writes."""
import json
import numpy as np
from lloyd_kmeans.generation import generate_gaussian_clouds
from lloyd_kmeans.kmeans import kmeans
from lloyd_kmeans.image import segment_image


def main():
    centers = np.array([[-4., 0.], [0., 4.], [4., 0.]])
    X, _ = generate_gaussian_clouds(centers, 60, std=0.35, rng=2026)
    result = kmeans(X, 3, init=centers)
    assert result.converged and result.labels_stable
    assert sorted(np.bincount(result.labels).tolist()) == [60, 60, 60]
    np.testing.assert_allclose(result.centers,
                               [X[result.labels == k].mean(axis=0) for k in range(3)])
    assert np.all(np.diff(result.inertia_history) <= 1e-10)

    palette = np.array([[30, 50, 180], [220, 170, 40]], dtype=np.uint8)
    image = palette[np.indices((24, 32)).sum(axis=0) % 2]
    quantized, rgb = segment_image(image, 2, init=palette.astype(float))
    np.testing.assert_array_equal(quantized, image)
    assert rgb.converged and rgb.inertia == 0.
    print(json.dumps({"seed": 2026, "points": len(X), "clusters": 3,
                      "inertia": result.inertia, "iterations": result.n_iter,
                      "rgb_shape": list(quantized.shape), "rgb_colors": 2,
                      "rgb_inertia": rgb.inertia}, indent=2))


if __name__ == "__main__":
    main()
