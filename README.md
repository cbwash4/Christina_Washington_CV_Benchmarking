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
- Trainable Parameter Count
- Checkpoint Size
- Peak GPU Memory
- MACs
- Approximate FLOPs
- Confusion Matrix
- Per-Class Precision, Recall, F1, and Support

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

Complete standardized deep-model results are available in:

`results/deep_model_complete_results.csv`

Computational complexity is reported using MACs and approximate FLOPs,
using the convention that one MAC is approximately two FLOPs.

## Original Benchmark vs. Deep Architectures

Assignment 3 also compares the ten additional deep architectures with
the six models from the original benchmarking library.

The combined comparison contains 16 models.

The final standardized combined benchmark is stored in:

`combined_ml_cnn_benchmark_results.csv`

A second copy is retained in:

`results/combined_ml_cnn_benchmark_results.csv`

The results show that increased model size does not necessarily produce
higher classification performance on this benchmark.

For example, VGG16 achieved approximately the same test accuracy as
MobileNetV3-Small and YOLO11n Classification while requiring
substantially more parameters and storage.

Similarly, ResNet50 did not improve test accuracy over ResNet18 despite
its greater model complexity.

## Standardized Inference Benchmark

Deep-model inference latency and throughput were measured using at
least 1,000 inference images per architecture to provide a more stable
runtime comparison than a single small test-set pass.

The standardized deep-model results also include total and trainable
parameter counts, checkpoint size, peak GPU memory, MACs, and
approximate FLOPs.

## Per-Class Analysis

Per-class precision, recall, F1, and support were standardized across
all 16 models for the three Fashion-MNIST classes.

The combined per-class results contain 48 rows: three classes for each
of the 16 models.

Results are stored in:

`results/combined_16_model_per_class_metrics.csv`

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

Additional final 16-model comparison figures are stored in:

`combined_comparison_plots/`

These include:

- Accuracy comparison
- Macro F1 comparison
- Training-time comparison
- Inference-throughput comparison
- Parameter comparison
- Checkpoint-size comparison
- Accuracy versus parameter count
- Accuracy versus inference latency
- Per-class F1 comparison

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


## Architecture Evolution

The benchmark spans several generations of image-classification architectures,
allowing historical and modern design strategies to be compared on the same
classification task.

- **AlexNet** represents an early deep CNN architecture that helped establish
  the practical use of deep convolutional networks for image recognition.
- **VGG16** increased network depth through repeated small convolutional layers,
  providing a simpler and more uniform design than earlier architectures.
- **GoogLeNet** introduced the Inception approach, using multiple convolutional
  operations at different spatial scales within the network.
- **ResNet18 and ResNet50** introduced residual connections, allowing much
  deeper networks to be trained more effectively. The two ResNet variants also
  provide a direct comparison of network depth within one architectural family.
- **DenseNet121** uses dense connectivity between layers, allowing later layers
  to receive feature information from earlier layers.
- **MobileNetV3-Small** emphasizes computational efficiency through lightweight
  network design, making it relevant to resource-constrained deployment.
- **EfficientNet-B0** uses a compound scaling strategy to balance network depth,
  width, and input resolution while controlling computational cost.
- **ConvNeXt-Tiny** represents a more modern CNN design that revisits
  convolutional architectures using contemporary deep-learning design choices.
- **YOLO11n Classification** provides a modern lightweight classification
  architecture through the Ultralytics classification pipeline.

This progression illustrates that architectural development is not simply a
matter of increasing network depth or parameter count. The benchmark results
show that later lightweight architectures can achieve performance comparable
to substantially larger networks while reducing computational and deployment
costs.

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
