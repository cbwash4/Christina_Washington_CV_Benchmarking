
from christina_washington_cv_benchmarking import benchmark_image_classification

# JSON records:
# [
#   {"image_path": "images/a.png", "label": "class_a"},
#   {"image_path": "images/b.png", "label": "class_b"}
# ]

results = benchmark_image_classification(
    dataset="manifest.json",
    dataset_type="json",
    target_labels="label",
    color_mode="rgb"
)

print(results["summary"])
print("Best model:", results["best_model"])
