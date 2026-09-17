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

## Image Classification Models

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


---

# Assignment 3: Deep Architecture Benchmarking

Assignment 3 extends the original six-model image classification
benchmark with ten additional deep-learning architectures.

The extension evaluates predictive performance together with
computational efficiency while preserving a fixed dataset split for
direct model comparison.

## Dataset

The deep-learning benchmark uses a reproducible subset of Fashion-MNIST
containing three classes:

- T-shirt/top
- Trouser
- Pullover

A total of 900 images are used, with 300 images per class.

The fixed stratified split contains:

- Training: 576 images
- Validation: 144 images
- Testing: 180 images

Each class contributes:

- 192 training images
- 48 validation images
- 60 testing images

Random seed: 42.

The same saved split indices are reused across the deep architectures.

## Deep Architectures

The following ten architectures were evaluated:

1. ResNet18
2. ResNet50
3. DenseNet121
4. MobileNetV3-Small
5. EfficientNet-B0
6. AlexNet
7. VGG16
8. GoogLeNet
9. ConvNeXt-Tiny
10. YOLO11n Classification

The Torchvision architectures use pretrained weights with their final
classification layers adapted for the three-class task. YOLO11n
Classification is evaluated through the Ultralytics classification
pipeline.

## Training

The deep architectures were trained for 20 epochs using the fixed
training and validation partitions.

The best validation checkpoint was retained for final evaluation on
the untouched test partition.

YOLO11n Classification uses the same image membership in its
train, validation, and test folders so that its evaluation remains
consistent with the fixed split.

## Deep-Model Evaluation Metrics

The deep benchmark records:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1
- Best Validation Accuracy
- Best Validation Epoch
- Training Time
- Average Inference Time
- Frames Per Second (FPS)
- Parameter Count
- Checkpoint Size
- Peak GPU Memory
- Confusion Matrix
- Per-Class Metrics

Macro F1 is used as the primary predictive-performance comparison
metric.

## Deep Benchmark Results

The strongest test results were approximately:

| Model | Accuracy | Macro F1 | Infer (ms/img) |
| --- | ---: | ---: | ---: |
| YOLO11n Classification | 0.9833 | 0.9833 | 1.882 |
| MobileNetV3-Small | 0.9833 | 0.9833 | 2.541 |
| VGG16 | 0.9833 | 0.9832 | 6.238 |
| DenseNet121 | 0.9778 | 0.9778 | 4.570 |
| ResNet18 | 0.9778 | 0.9778 | 2.588 |
| EfficientNet-B0 | 0.9778 | 0.9778 | 4.501 |
| ConvNeXt-Tiny | 0.9778 | 0.9778 | 17.531 |
| ResNet50 | 0.9778 | 0.9778 | 5.197 |
| AlexNet | 0.9667 | 0.9667 | 2.286 |
| GoogLeNet | 0.9611 | 0.9612 | 2.936 |

Full-precision results are available in:

`results/deep_model_master_comparison.csv`

## Original Benchmark vs. Deep Architectures

Assignment 3 also compares the ten additional deep architectures with
the six models from the original benchmarking library.

The combined comparison contains 16 models.

Results are stored in:

`results/final_16_model_comparison.csv`

The results show that increased model size does not necessarily produce
higher classification performance on this benchmark.

For example, VGG16 achieved approximately the same test accuracy as
MobileNetV3-Small and YOLO11n Classification while requiring
substantially more parameters and storage.

Similarly, ResNet50 did not improve test accuracy over ResNet18 despite
its greater model complexity.

## Confusion Matrices

Confusion-matrix data for the ten deep architectures are stored in:

`confusion_matrices/`

Rendered confusion-matrix figures are stored in:

`confusion_matrix_plots/`

Across the strongest models, Trouser was strongly separated from the
other two classes. Most remaining errors occurred between T-shirt/top
and Pullover.

## Training Curves

Training and validation histories are stored in:

`results/`

Accuracy and loss curves are stored in:

`plots/`

These figures show model convergence behavior across the 20-epoch
training runs.

## Model Comparison Visualizations

Cross-model comparison figures are stored in:

`comparison_plots/`

They include comparisons of:

- Macro F1
- Inference time
- Parameter count
- Checkpoint size
- Accuracy versus parameter count
- Macro F1 across all 16 evaluated models

## Grad-CAM

Grad-CAM visualizations were generated for three representative
architectures:

- ResNet18
- MobileNetV3-Small
- VGG16

The visualizations use the same test image to illustrate differences
in spatial activation patterns across architectures.

The Grad-CAM implementation is located at:

`src/christina_washington_cv_benchmarking/gradcam.py`

Example visualizations are stored in:

`plots/`

Grad-CAM is treated as a qualitative visualization of model activation
rather than a quantitative measure of explanation quality.

## Key Finding

The Assignment 3 experiments demonstrate that predictive performance
should be considered together with computational efficiency.

Several lightweight architectures achieved performance comparable to
or better than substantially larger networks while requiring fewer
parameters and smaller checkpoints.

The benchmark therefore evaluates model selection as a tradeoff among
classification performance, inference speed, model size, and
computational requirements.


---

# Assignment #3 — Deep Architecture Benchmarking

Assignment #3 extends the original six-model image-classification
benchmark with ten deep learning architectures and provides a
standardized comparison across all 16 models.

## Dataset

Fashion-MNIST subset containing three classes:

- T-shirt/top
- Trouser
- Pullover

Dataset split:

- Training: 576 images
- Validation: 144 images
- Testing: 180 images

Each test class contains 60 images.

## Original Benchmark Models

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine
5. Fully Connected Neural Network
6. Simple CNN

## Deep Architectures

1. ResNet18
2. ResNet50
3. DenseNet121
4. MobileNetV3-Small
5. EfficientNet-B0
6. AlexNet
7. VGG16
8. GoogLeNet
9. ConvNeXt-Tiny
10. YOLO11n Classification

## Evaluation Metrics

The benchmark evaluates:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1
- Per-class Precision
- Per-class Recall
- Per-class F1
- Training Time
- Inference Latency
- Inference Throughput (FPS)
- Parameter Count
- Trainable Parameter Count
- Checkpoint Size
- Peak GPU Memory
- MACs
- Approximate FLOPs

Macro F1 is used as the primary overall classification metric.

Computational complexity is reported using MACs and approximate
FLOPs, with the convention that one MAC is approximately two FLOPs.

## Standardized Inference Benchmark

Deep-model inference latency and throughput were measured using
at least 1,000 inference images per architecture.

## Explainability

Grad-CAM visualizations were generated for:

- ResNet18
- MobileNetV3-Small
- VGG16

These visualizations provide qualitative evidence of image regions
contributing to model predictions.

## Combined Benchmark

The final comparison contains:

- 6 original benchmark models
- 10 deep architectures
- 16 models total

Primary combined results:

`combined_ml_cnn_benchmark_results.csv`

Additional detailed results are stored in:

`results/`

## Per-Class Analysis

Per-class precision, recall, F1, and support are provided for all
16 models across:

- T-shirt/top
- Trouser
- Pullover

Combined per-class results:

`results/combined_16_model_per_class_metrics.csv`

## Visualizations

Combined comparison plots include:

- Accuracy comparison
- Macro F1 comparison
- Training-time comparison
- Inference-throughput comparison
- Parameter comparison
- Checkpoint-size comparison
- Accuracy versus parameter count
- Accuracy versus inference latency
- Per-class F1 comparison

Plots are stored in:

`combined_comparison_plots/`

Individual deep-model confusion matrices and training curves are
also included in the repository.

## Reproducibility

Random seed: 42

The same held-out test set of 180 images was used for standardized
classification evaluation.

Large datasets, model checkpoints, downloaded pretrained weights,
and temporary YOLO training artifacts are excluded from version
control.

