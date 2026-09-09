# Christina Washington CV Benchmarking

Reusable Python package for benchmarking image classification models.

## Installation

pip install Christina_Washington_CV_Benchmarking

## Import

from christina_washington_cv_benchmarking import benchmark_image_classification

## Main Function

benchmark_image_classification(dataset, dataset_type, target_labels, color_mode)

## Supported Inputs

- folder
- csv
- json
- array

Color modes: grayscale and rgb.

## Preprocessing

- Images standardized to 64 x 64 pixels
- Grayscale shape: (N, 64, 64, 1)
- RGB shape: (N, 64, 64, 3)
- Pixel values normalized to [0, 1]
- Aspect ratio preserved with padding when practical

## Train-Test Split

- 80% training / 20% testing
- Stratified
- Random seed 42
- Same split used for all six models

## Models

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine
5. Fully Connected Neural Network
6. Simple CNN

## Evaluation Metrics

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1
- Training Time
- Average Inference Time
- Confusion Matrix
- Classification Report

Macro F1 is the primary ranking metric.

## Output

Each run generates benchmark_results containing summary CSV,
metrics JSON, run configuration, plots, confusion matrices,
and classification reports.

## Demonstrations

Grayscale demonstration: Fashion-MNIST reproducible subset,
3 classes, 300 images per class, 900 images total.

RGB demonstration: reproducible synthetic RGB dataset with
red-dominant, green-dominant, and blue-dominant classes,
300 images per class, 900 images total.

## Testing

Run: pytest tests/

The test suite covers data loading, preprocessing, stratification,
models, benchmark execution, output generation, and metadata.

## Reproducibility

Random seed: 42

No pretrained networks, AutoML, or primary-benchmark augmentation.

## License

MIT License