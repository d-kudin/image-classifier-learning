# src/evaluate.py

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory, get_file
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.utils import list_classes

IMG_SIZE = (224, 224)
BATCH = 32
IMAGENET_EFFB0_NOTOP = "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5"

def build_model(num_classes: int):
    """
    Same architecture as training, but without data augmentation:
    - EfficientNetB0 backbone without auto-weights
    - Manually load ImageNet weights with skip_mismatch=True
    - Head: GAP -> Dropout -> Dense
    """
    base = EfficientNetB0(include_top=False, weights=None, input_shape=IMG_SIZE + (3,))
    weights_path = get_file("efficientnetb0_notop.h5", IMAGENET_EFFB0_NOTOP, cache_subdir="models")
    try:
        base.load_weights(weights_path, skip_mismatch=True)
    except TypeError:
        base.load_weights(weights_path, by_name=True, skip_mismatch=True)

    inputs = layers.Input(shape=IMG_SIZE + (3,), name="input_rgb")
    x = preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model

def main():
    classes = list_classes("data/train")
    model = build_model(len(classes))

    # Load trained weights (full model weights)
    model.load_weights("models/efficientnet_weights.weights.h5")

    ds = image_dataset_from_directory(
        "data/test",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="categorical",
        class_names=classes,
        color_mode="rgb",
        shuffle=False
    ).prefetch(tf.data.AUTOTUNE)

    y_true, y_pred = [], []
    for x, y in ds:
        probs = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(np.argmax(y.numpy(), axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    Path("results").mkdir(exist_ok=True)
    with open("results/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"Accuracy: {acc:.4f}\n")
    print(report)

    # Confusion matrix figure
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45, ha="right")
    plt.yticks(ticks=range(len(classes)), labels=classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=160)
    plt.close()

if __name__ == "__main__":
    main()