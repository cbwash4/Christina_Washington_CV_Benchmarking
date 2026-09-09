
from christina_washington_cv_benchmarking import benchmark_image_classification

results = benchmark_image_classification(
    dataset="dataset_folder",
    dataset_type="folder",
    target_labels=["class_a", "class_b"],
    color_mode="rgb"
)

print(results["summary"])
print("Best model:", results["best_model"])
