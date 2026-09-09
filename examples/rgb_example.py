
import numpy as np

from christina_washington_cv_benchmarking import (
    benchmark_image_classification
)

rng = np.random.default_rng(42)

n = 300

red = np.zeros((n, 32, 32, 3), dtype=np.uint8)
green = np.zeros((n, 32, 32, 3), dtype=np.uint8)
blue = np.zeros((n, 32, 32, 3), dtype=np.uint8)

red[..., 0] = rng.integers(
    150, 256, size=(n, 32, 32)
)

green[..., 1] = rng.integers(
    150, 256, size=(n, 32, 32)
)

blue[..., 2] = rng.integers(
    150, 256, size=(n, 32, 32)
)

images = np.concatenate([
    red,
    green,
    blue
])

labels = np.array(
    ["red"] * n +
    ["green"] * n +
    ["blue"] * n
)

results = benchmark_image_classification(
    dataset=images,
    dataset_type="array",
    target_labels=labels,
    color_mode="rgb"
)

print(results["summary"])
print("Best model:", results["best_model"])
