
from pathlib import Path
import matplotlib.pyplot as plt


def save_training_plots(history, model_name):

    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    # LOSS CURVE
    plt.figure(figsize=(7, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        plot_dir / f"{model_name}_loss_curve.png",
        dpi=300
    )

    plt.close()

    # ACCURACY CURVE
    plt.figure(figsize=(7, 5))

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        plot_dir / f"{model_name}_accuracy_curve.png",
        dpi=300
    )

    plt.close()
