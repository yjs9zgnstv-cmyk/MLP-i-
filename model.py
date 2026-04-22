"""
model.py — MLP модель на Keras для распознавания цифр MNIST
"""

import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODEL_PATH = "mnist_mlp.h5"

class MLPModel:
    def __init__(self):
        self._model = None

    # ── Load saved model ───────────────────────────────────────────────────
    def load(self) -> bool:
        if not os.path.exists(MODEL_PATH):
            return False
        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(MODEL_PATH)
            return True
        except Exception:
            return False

    def is_ready(self) -> bool:
        return self._model is not None

    # ── Build architecture ─────────────────────────────────────────────────
    def _build(self):
        import tensorflow as tf
        model = tf.keras.Sequential([
            # Input: 28×28 = 784 pixels
            tf.keras.layers.Input(shape=(784,)),

            # Hidden layer 1 — 256 neurons, ReLU + Batch Norm + Dropout
            tf.keras.layers.Dense(256, activation="relu",
                                  kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            # Hidden layer 2 — 128 neurons
            tf.keras.layers.Dense(128, activation="relu",
                                  kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            # Hidden layer 3 — 64 neurons
            tf.keras.layers.Dense(64, activation="relu",
                                  kernel_initializer="he_normal"),
            tf.keras.layers.Dropout(0.1),

            # Output: 10 classes (digits 0–9), Softmax
            tf.keras.layers.Dense(10, activation="softmax"),
        ], name="MLP_MNIST")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ── Train ──────────────────────────────────────────────────────────────
    def train(self, epochs: int = 15, callback=None) -> float:
        import tensorflow as tf

        # Load MNIST
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Preprocess: flatten + normalize
        x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
        x_test  = x_test.reshape(-1, 784).astype("float32") / 255.0

        self._model = self._build()

        # Callbacks
        cbs = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=0),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5,
                restore_best_weights=True, verbose=0),
        ]
        if callback:
            class LogCB(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    callback(epoch, logs or {})
            cbs.append(LogCB())

        self._model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=256,
            validation_split=0.1,
            callbacks=cbs,
            verbose=0,
        )

        # Evaluate
        _, acc = self._model.evaluate(x_test, y_test, verbose=0)

        # Save
        self._model.save(MODEL_PATH)
        return acc

    # ── Predict ────────────────────────────────────────────────────────────
    def predict(self, img_array: np.ndarray) -> np.ndarray:
        """
        img_array: 28×28 float32, values in [0, 1]
        Returns: softmax probabilities shape (10,)
        """
        flat = img_array.reshape(1, 784)
        probs = self._model.predict(flat, verbose=0)[0]
        return probs

    # ── Summary ────────────────────────────────────────────────────────────
    def summary(self) -> str:
        if self._model is None:
            return "Модель не загружена"
        lines = []
        self._model.summary(print_fn=lines.append)
        return "\n".join(lines)
