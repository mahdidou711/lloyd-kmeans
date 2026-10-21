# Algorithm contract

```python
kmeans(X, K, *, init=None, rng=None, max_iter=300,
       stop_criterion="labels", tol=0.0)
```

Uses finite float64 matrices. Initialization selects
K distinct observations without replacement unless explicit centers are supplied.
Assignments minimize Euclidean distance, with the lowest index winning computed
ties. Nonempty clusters are recentered to their means; empty ones keep their
previous center. Each iteration reassigns points after recentering. The returned
inertia is the sum of squared distances to the returned prototypes.

The default `labels` criterion requires unchanged assignments. The alternative
`center_shift` compares the largest Euclidean center displacement to `tol`,
an absolute threshold in input units. Inspect `converged`, `labels_stable` and
`stop_reason`; a satisfied shift threshold does not imply stationary labels.
At the iteration limit labels still describe nearest returned centers, but
centers may not be means of those final groups. `inertia_history` includes
initial inertia. Shift history exists only for the shift criterion.

This is a local optimizer: initialization matters, K is supplied by the user,
features are not standardized and no global optimum is promised. Assignment
storage scales as O(NK); large images can be costly. Extreme float64 values can
lose precision or raise a validation error on overflow. There is no streaming,
GPU execution, missing-value imputation or automatic choice of K.

RGB helpers accept nonempty uint8 H×W×3 images and use all pixels in C order,
without spatial coordinates, perceptual distance or gamma correction.
Centers are rounded with NumPy's nearest-even rule before reconstruction.
Rounding may merge colors; the image contains at most K colors. RGB inertia
uses floating centers, not the rounded reconstruction.
