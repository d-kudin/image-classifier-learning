# src/train.py

import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.utils import image_dataset_from_directory, get_file
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from src.utils import list_classes, ensure_dirs

BATCH = 32
IMG_SIZE = (224, 224)
STAGE1_EPOCHS = 8      # head-only
STAGE2_EPOCHS = 15     # partial fine-tune (tail)
STAGE3_EPOCHS = 5      # full fine-tune (optional)
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-4
LR_STAGE3 = 5e-5
IMAGENET_EFFB0_NOTOP = "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5"

def compute_class_weight_from_dirs(train_root: str, classes: list[str]) -> dict[int, float]:
    counts = []
    for cls in classes:
        counts.append(len(list(Path(train_root, cls).glob("*"))))
    total = sum(counts)
    # sklearn-like balanced: n_samples / (n_classes * n_samples_c)
    weights = [total / (len(classes) * c) if c > 0 else 0.0 for c in counts]
    # normalize to mean 1.0 (optional)
    mean_w = sum(weights) / len(weights)
    weights = [w / mean_w for w in weights]
    return {i: w for i, w in enumerate(weights)}

def make_datasets(classes: list[str]):
    """Create train/val datasets with RGB images and prefetching."""
    common = dict(
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="categorical",
        class_names=classes,
        color_mode="rgb",
    )
    train_ds = image_dataset_from_directory("data/train", shuffle=True, **common)
    val_ds   = image_dataset_from_directory("data/val",   shuffle=False, **common)
    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), val_ds.prefetch(autotune)

def attach_sample_weights(ds, class_weight: dict[int, float]):
    """Return (x, y, sample_weight) so class balancing works with one-hot labels."""
    
    weight_vec = tf.constant([class_weight[i] for i in range(len(class_weight))], dtype=tf.float32)

    def _map(x, y):
        
        idx = tf.argmax(y, axis=1)                  
        w = tf.gather(weight_vec, idx)              
        return x, y, w

    return ds.map(_map)

def build_model(num_classes: int, trainable_base: bool = False):
    """
    Build EfficientNetB0 backbone WITHOUT auto-loading weights, then
    manually load ImageNet weights with skip_mismatch=True to avoid
    shape issues on the stem conv under some TF/Keras versions.
    """
    # Lightweight on-the-fly augmentation
    aug = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    # 1) Build backbone with explicit RGB input and no weights
    base = EfficientNetB0(include_top=False, weights=None, input_shape=IMG_SIZE + (3,))
    base.trainable = trainable_base

    # 2) Manually load ImageNet weights (skip mismatches on the first conv if needed)
    weights_path = get_file("efficientnetb0_notop.h5", IMAGENET_EFFB0_NOTOP, cache_subdir="models")
    try:
        base.load_weights(weights_path, skip_mismatch=True)
    except TypeError:
        # Older signatures: fall back to by_name + skip_mismatch
        base.load_weights(weights_path, by_name=True, skip_mismatch=True)

    # 3) Classification head
    inputs = layers.Input(shape=IMG_SIZE + (3,), name="input_rgb")
    x = aug(inputs)
    x = preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base

def compile_and_fit(model, train_ds, val_ds, lr, epochs, ckpt_path):
    """Compile with label smoothing, train with EarlyStopping + ReduceLROnPlateau."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    cbs = [
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs)
    return hist.history

def main():
    ensure_dirs()
    classes = list_classes("data/train")
    print("Detected classes:", classes)
    class_weight = compute_class_weight_from_dirs("data/train", classes)
    print("Class weights:", class_weight)

    train_ds, val_ds = make_datasets(classes)
    train_ds = attach_sample_weights(train_ds, class_weight)

    model, base = build_model(num_classes=len(classes), trainable_base=False)

    # --- Stage 1: train head only (backbone frozen) ---
    base.trainable = False
    h1 = compile_and_fit(
        model, train_ds, val_ds, lr=LR_STAGE1, epochs=STAGE1_EPOCHS,
        ckpt_path="models/efficientnet_flower_model.keras"
    )

    # --- Stage 2: unfreeze tail (e.g., last ~50 layers) ---
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False
    h2 = compile_and_fit(
        model, train_ds, val_ds, lr=LR_STAGE2, epochs=STAGE2_EPOCHS,
        ckpt_path="models/efficientnet_flower_model.keras"
    )

    # --- Stage 3 (optional): unfreeze all, very small LR ---
    for layer in base.layers:
        layer.trainable = True
    h3 = compile_and_fit(
        model, train_ds, val_ds, lr=LR_STAGE3, epochs=STAGE3_EPOCHS,
        ckpt_path="models/efficientnet_flower_model.keras"
    )

    # Save training histories
    Path("results").mkdir(exist_ok=True)
    with open("results/training_history_stage1.json", "w") as f: json.dump(h1, f)
    with open("results/training_history_stage2.json", "w") as f: json.dump(h2, f)
    with open("results/training_history_stage3.json", "w") as f: json.dump(h3, f)

    # Save weights only (matches repo layout)
    model.save_weights("models/efficientnet_weights.weights.h5")

    # Quick validation eval
    loss, acc = model.evaluate(val_ds, verbose=0)
    print({"val_loss": float(loss), "val_acc": float(acc)})

if __name__ == "__main__":
    main()