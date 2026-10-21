# Lloyd K-means with NumPy

Lloyd clustering for finite real vectors, with Gaussian sample generation and
RGB color quantization. Runtime dependencies are limited to NumPy and the Python
standard library. SciPy and scikit-learn provide independent test comparisons.

## Features

- Euclidean distances with float64 numerical validation.
- Explicit initial centers or reproducible random initialization.
- Label stability or maximum center displacement as the stopping criterion.
- Final labels, centers, clusters, inertia and convergence diagnostics.
- Gaussian point clouds and uint8 RGB image quantization.

## Installation

Python 3.12 is required. From a local checkout:

```sh
python -m pip install .
```

For development and testing:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
```

These commands require the build backend and declared dependencies. For source
execution with dependencies already available:

```sh
PYTHONPATH="$PWD/src" python examples/synthetic_demo.py
PYTHONPATH="$PWD/src" python -m pytest -q -p no:cacheprovider
```

## API

```python
from lloyd_kmeans.generation import generate_gaussian_clouds
from lloyd_kmeans.kmeans import kmeans

X, source_labels = generate_gaussian_clouds([[-3, 0], [3, 0]], 100,
                                           std=0.4, rng=2026)
result = kmeans(X, 2, init=[[-3, 0], [3, 0]])
print(result.inertia, result.n_iter, result.stop_reason)
```

`source_labels` identify generating groups; `result.labels` identify estimated
clusters. Cluster numbers have no semantic identity. `KMeansResult` also exposes
`centers`, `clusters`, `inertia_history`, `converged`, `labels_stable` and
`center_shift_history`. RGB helpers `image_to_pixels`, `reconstruct_image` and
`segment_image` are available in `lloyd_kmeans.image`.

## Example and tests

The synthetic example generates 180 observations in three Gaussian groups and
quantizes a generated 24 × 32 RGB checkerboard. It checks invariants, prints JSON
metrics and writes no files. Seed 2026 and inputs are explicit.

The five test modules cover distances, Gaussian generation, clustering, stopping
criteria and RGB conversion with synthetic fixtures. Test dependencies are
NumPy, SciPy, scikit-learn and pytest. `requirements-tested.txt` records a tested
version set; it is not a transitive dependency lock.

## Numerical and practical limits

Lloyd is a local optimizer sensitive to initialization. K is chosen by the user;
features are not standardized automatically. Assignment storage scales as O(NK).
Extreme float64 inputs may lose precision or raise a validation error. Random
seeds alone do not ensure identical results across dependency versions.

RGB quantization uses Euclidean color distance without spatial information,
perceptual correction or gamma correction. Rounding can merge output colors.
See the [algorithm contract](docs/algorithm.md) for stopping semantics, empty
clusters, numerical behavior and complexity.
