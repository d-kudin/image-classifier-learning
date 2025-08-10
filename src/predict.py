# src/predict.py

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array, get_file
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

from src.utils import list_classes

IMG_SIZE = (224, 224)
IMAGENET_EFFB0_NOTOP = "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5"
WEIGHTS_PATH = "models/efficientnet_weights.weights.h5"

def build_model(num_classes: int):
    """
    Same backbone as training:
    - EfficientNetB0 without auto-weights
    - Manually load ImageNet no-top weights with skip_mismatch=True
    - Head: GAP -> Dropout -> Dense
    """
    base = EfficientNetB0(include_top=False, weights=None, input_shape=IMG_SIZE + (3,))
    weights_path = get_file("efficientnetb0_notop.h5", IMAGENET_EFFB0_NOTOP, cache_subdir="models")
    try:
        base.load_weights(weights_path, skip_mismatch=True)
    except TypeError:
        base.load_weights(weights_path, by_name=True, skip_mismatch=True)

    inp = layers.Input(shape=IMG_SIZE + (3,), name="input_rgb")
    x = preprocess_input(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out)

def load_image_tensor(path: Path):
    img = load_img(path, target_size=IMG_SIZE)
    arr = img_to_array(img)[None, ...]  # (1, H, W, 3)
    return img, arr

def viz_prediction(img_pil, classes, probs, save_path: Path | None, top_k: int = 5, title_prefix: str = "Prediction"):
    """Show image and top-K probabilities; optionally save the figure."""
    probs = np.asarray(probs).astype(float)
    top_idx = np.argsort(probs)[::-1][:top_k]
    top_labels = [classes[i] for i in top_idx]
    top_scores = probs[top_idx]

    pred_idx = int(top_idx[0])
    pred_label = classes[pred_idx]
    pred_score = float(top_scores[0])

    plt.figure(figsize=(11, 5))

    # Left: image with predicted label
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img_pil)
    ax1.axis("off")
    ax1.set_title(f"{title_prefix}: {pred_label}\n({pred_score:.2%})")

    # Right: horizontal bar chart of top-K
    ax2 = plt.subplot(1, 2, 2)
    y_pos = np.arange(len(top_labels))
    ax2.barh(y_pos, top_scores)
    ax2.set_yticks(y_pos, labels=top_labels)
    ax2.set_xlabel("Probability")
    ax2.invert_yaxis()  # highest on top
    for i, v in enumerate(top_scores):
        ax2.text(v + 0.01, i, f"{v:.2%}", va="center")
    ax2.set_xlim(0, 1)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160)
        print(f"Saved visualization to: {save_path}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Predict flower class for a single image.")
    # accept paths with spaces: gather remaining args and join
    parser.add_argument("image", nargs="+", help="Path to image (jpg/jpeg/png)")
    parser.add_argument("--topk", type=int, default=5, help="How many top classes to show")
    parser.add_argument("--save", action="store_true", help="Save visualization to results/")
    args = parser.parse_args()

    img_path = Path(" ".join(args.image))
    if not img_path.exists():
        print(f"File not found: {img_path}")
        sys.exit(1)

    classes = list_classes("data/train")
    if not classes:
        print("No classes found in data/train. Make sure your dataset is prepared.")
        sys.exit(1)

    model = build_model(len(classes))
    if not Path(WEIGHTS_PATH).exists():
        print(f"Trained weights not found: {WEIGHTS_PATH}")
        print("Train the model first: python -m src.train")
        sys.exit(1)
    model.load_weights(WEIGHTS_PATH)

    # Load and predict
    img_pil, arr = load_image_tensor(img_path)
    probs = model.predict(arr, verbose=0)[0]

    # Print full probability table
    print("\nProbabilities (sorted):")
    for i in np.argsort(probs)[::-1]:
        print(f"  {classes[i]}: {probs[i]:.4f}")

    # Visualize
    save_path = None
    if args.save:
        ts = time.strftime("%Y%m%d-%H%M%S")
        safe_stem = img_path.stem.replace(" ", "_")
        save_path = Path("results") / f"prediction_{safe_stem}_{ts}.png"

    viz_prediction(img_pil, classes, probs, save_path=save_path, top_k=max(1, args.topk))

if __name__ == "__main__":
    main()



'''# src/predict.py

import sys
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array, get_file
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from src.utils import list_classes

IMG_SIZE = (224, 224)
IMAGENET_EFFB0_NOTOP = "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5"

def build_model(num_classes: int):
    """
    Build the same architecture as in training:
    - EfficientNetB0 backbone built WITHOUT auto-weights
    - Manually load ImageNet weights with skip_mismatch=True
    - Simple classification head (GAP -> Dropout -> Dense)
    """
    # Backbone (no auto-weights), explicit RGB input
    base = EfficientNetB0(include_top=False, weights=None, input_shape=IMG_SIZE + (3,))

    # Load ImageNet weights (skip stem conv mismatch if it appears)
    weights_path = get_file("efficientnetb0_notop.h5", IMAGENET_EFFB0_NOTOP, cache_subdir="models")
    try:
        base.load_weights(weights_path, skip_mismatch=True)
    except TypeError:
        base.load_weights(weights_path, by_name=True, skip_mismatch=True)

    # Head
    inputs = layers.Input(shape=IMG_SIZE + (3,), name="input_rgb")
    x = preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict path/to/image.jpg")
        sys.exit(1)

    classes = list_classes("data/train")
    model = build_model(len(classes))

    # Load trained weights (full model weights)
    model.load_weights("models/efficientnet_weights.weights.h5")

    img = load_img(sys.argv[1], target_size=IMG_SIZE)
    arr = img_to_array(img)[None, ...]  # shape (1, H, W, 3)
    preds = model.predict(arr, verbose=0)[0]

    top = int(np.argmax(preds))
    print("Prediction:", classes[top])
    print("Probabilities:")
    for cls, p in zip(classes, preds):
        print(f"  {cls}: {p:.4f}")

if __name__ == "__main__":
    main()'''