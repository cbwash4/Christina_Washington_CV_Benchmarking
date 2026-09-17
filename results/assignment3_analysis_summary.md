# Assignment 3 — Benchmark Analysis Summary

## Experimental Setup

- Dataset: Fashion-MNIST
- Classes: T-shirt/top, Trouser, Pullover
- Total images: 900
- Training: 576 images
- Validation: 144 images
- Testing: 180 images
- Fixed stratified split used across models
- Deep architectures trained for 20 epochs
- Primary comparison metric: Macro F1
- Inference time used as an efficiency metric

## Original Benchmark Models

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. SVM
5. Fully Connected Neural Network
6. Simple CNN

## Additional Deep Architectures

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

## Major Results

YOLO11n Classification and MobileNetV3-Small both achieved approximately
98.33% test accuracy and 0.9833 Macro F1.

VGG16 also achieved approximately 98.33% accuracy, but required substantially
more parameters, storage, GPU memory, training time, and inference time.

YOLO11n provided strong predictive performance with approximately 1.53 million
parameters, a 3.04 MB checkpoint, and approximately 1.88 ms/image measured
inference time.

MobileNetV3-Small provided similarly strong performance with approximately
1.52 million parameters and a 5.93 MB checkpoint.

ResNet18, ResNet50, DenseNet121, EfficientNet-B0, and ConvNeXt-Tiny all achieved
approximately 97.78% test accuracy.

The Simple CNN and fully connected Neural Network from the original benchmark
both achieved approximately 97.22% accuracy.

AlexNet and GoogLeNet did not outperform the strongest lightweight modern
architectures on this benchmark.

## Efficiency Findings

Model complexity did not correspond directly to higher classification
performance.

VGG16 contained approximately 134.27 million parameters but achieved
approximately the same test accuracy as the much smaller MobileNetV3-Small
and YOLO11n models.

ResNet50 did not improve test accuracy over ResNet18 despite having
substantially more parameters and higher computational requirements.

ConvNeXt-Tiny achieved strong classification performance but had the slowest
measured inference among the evaluated deep architectures.

The results demonstrate the importance of evaluating computational efficiency
alongside predictive accuracy.

## Confusion Matrix Findings

Across the strongest models, Trouser was particularly well separated from the
other two classes.

Most remaining classification errors occurred between T-shirt/top and Pullover,
which are visually more similar garment categories.

## Grad-CAM Findings

Grad-CAM was evaluated using representative high-performing architectures:
ResNet18, MobileNetV3-Small, and VGG16.

The models produced different spatial activation patterns for the same correctly
classified T-shirt/top test image.

ResNet18 and MobileNetV3-Small showed activation associated with garment
regions, while VGG16 produced a more diffuse activation pattern.

Grad-CAM should be interpreted as qualitative evidence of model attention rather
than as a quantitative explanation of model correctness.

## Overall Conclusion

The benchmark demonstrates that larger and more computationally expensive
architectures do not necessarily provide better classification performance on
this dataset.

Several modern lightweight architectures matched or exceeded substantially
larger networks while requiring fewer parameters and smaller checkpoints.

Therefore, model selection should consider predictive performance together with
inference speed, storage requirements, parameter count, and computational cost.
