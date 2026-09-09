
from christina_washington_cv_benchmarking import benchmark_image_classification

# CSV columns:
# image_path,label

results = benchmark_image_classification(
    dataset="manifest.csv",
    dataset_type="csv",
    target_labels="label",
    color_mode="rgb"
)

print(results["summary"])
print("Best model:", results["best_model"])
