
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping


RANDOM_SEED = 42
MAX_EPOCHS = 20
BATCH_SIZE = 32
PATIENCE = 3


def build_simple_cnn(input_shape, number_of_classes):
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    model = Sequential([
        Input(shape=input_shape),

        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu"
        ),

        MaxPooling2D(
            pool_size=(2, 2)
        ),

        Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu"
        ),

        MaxPooling2D(
            pool_size=(2, 2)
        ),

        Flatten(),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.30),

        Dense(
            number_of_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_simple_cnn(X_train, y_train, number_of_classes):
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    model = build_simple_cnn(
        input_shape=X_train.shape[1:],
        number_of_classes=number_of_classes
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=0
    )

    return model, history


def predict_simple_cnn(model, X_test):
    probabilities = model.predict(
        X_test,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    return predictions
