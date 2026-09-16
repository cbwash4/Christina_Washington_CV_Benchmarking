from pathlib import Path

import time
import copy
import torch
import torch.nn as nn


def train_model(
    model,
    train_loader,
    val_loader,
    model_name,
    epochs=20,
    learning_rate=0.001,
    device="cuda"
):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "epoch_time": [],
        "learning_rate": []
    }

    best_val_accuracy = -1.0
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / f"best_{model_name}.pt"

    total_start = time.time()

    for epoch in range(epochs):

        # -------------------------
        # TRAINING
        # -------------------------
        epoch_start = time.time()

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            # GoogLeNet can return auxiliary outputs
            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()

        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                if hasattr(outputs, "logits"):
                    outputs = outputs.logits

                loss = criterion(outputs, labels)

                val_running_loss += (
                    loss.item() * inputs.size(0)
                )

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_accuracy = val_correct / val_total

        epoch_time = time.time() - epoch_start

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["epoch_time"].append(epoch_time)
        history["learning_rate"].append(current_lr)

        # Save best validation model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                best_weights,
                checkpoint_path
            )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    total_training_time = time.time() - total_start

    # Restore best checkpoint
    model.load_state_dict(best_weights)

    return {
        "model": model,
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "total_training_time": total_training_time,
        "checkpoint_path": str(checkpoint_path)
    }
