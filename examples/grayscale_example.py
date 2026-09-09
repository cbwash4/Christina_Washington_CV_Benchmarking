
import numpy as np
import tensorflow as tf

from christina_washington_cv_benchmarking import (
    benchmark_image_classification
)

(X, y), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

selected_classes = [0, 1, 2]

names = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover"
}

rng = np.random.default_rng(42)

images = []
labels = []

for class_id in selected_classes:
    indices = np.where(y == class_id)[0]

    chosen = rng.choice(
        indices,
        size=300,
        replace=False
    )

    images.append(X[chosen])

    labels.extend(
        [names[class_id]] * 300
    )

images = np.concatenate(images)
labels = np.array(labels)

results = benchmark_image_classification(
    dataset=images,
    dataset_type="array",
    target_labels=labels,
    color_mode="grayscale"
)

print(results["summary"])
print("Best model:", results["best_model"])
