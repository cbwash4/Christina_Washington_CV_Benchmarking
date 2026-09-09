
from christina_washington_cv_benchmarking.classical_models import (
    get_classical_models
)

from christina_washington_cv_benchmarking.neural_models import (
    build_simple_cnn
)


def test_five_classical_models_exist():
    models = get_classical_models()

    assert len(models) == 5

    assert "Logistic Regression" in models
    assert "Decision Tree" in models
    assert "Random Forest" in models
    assert "SVM" in models
    assert "Neural Network" in models


def test_cnn_output_classes():
    model = build_simple_cnn(
        input_shape=(64, 64, 3),
        number_of_classes=3
    )

    assert model.output_shape[-1] == 3


def test_cnn_grayscale_input():
    model = build_simple_cnn(
        input_shape=(64, 64, 1),
        number_of_classes=2
    )

    assert model.input_shape == (
        None, 64, 64, 1
    )
