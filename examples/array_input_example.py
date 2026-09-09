
import numpy as np
from christina_washington_cv_benchmarking import benchmark_image_classification

rng = np.random.default_rng(42)

X = rng.integers(
    0, 256,
    size=(30, 32, 32, 3),
    dtype=np.uint8
)

y = np.array(
    ["class_a"] * 15 +
    ["class_b"] * 15
)

results = benchmark_image_classification(
    dataset=X,
    dataset_type="array",
    target_labels=y,
    color_mode="rgb"
)

print(results["summary"])
print("Best model:", results["best_model"])
