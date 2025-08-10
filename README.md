# Machine Learning — EfficientNetB0 Image Classifier (Transfer Learning)

A clean, reproducible **local (VS Code / CLI)** implementation of an image classifier using **EfficientNetB0** and transfer learning.  
Works with **any** image dataset organized by class (not limited to flowers).

---

## 1) Overview

- **Backbone:** EfficientNetB0 (ImageNet-pretrained, loaded manually for robustness).
- **Head:** GlobalAveragePooling2D → Dropout(0.5) → Dense(*num_classes*, softmax).
- **Preprocessing & Augmentation:** In-model Keras augmentation (`RandomFlip`, `RandomRotation(0.1)`, `RandomZoom(0.1)`, `RandomContrast(0.1)`) and EfficientNet **`preprocess_input`**.
- **Training schedule:** 3 stages (head-only → partial unfreeze → full fine-tune) with **ReduceLROnPlateau** and **EarlyStopping**.
- **Class imbalance:** per-class **sample weights** derived from file counts in `data/train`.
- **Artifacts:** best checkpoint (`.keras`), final weights (`.weights.h5`), JSON training histories, evaluation report + confusion matrix, optional prediction visualizations.

---

## 2) Requirements

- Python 3.10+
- Install dependencies:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## 3) Quick Start

**Train (3 stages):**
```bash
python -m src.train
```

**Evaluate (classification report + confusion matrix):**
```bash
python -m src.evaluate
```

**Predict a single image (shows image + top-K bar chart; supports spaces in paths):**
```bash
python -m src.predict "path/to/your image.jpg"
# save the visualization
python -m src.predict "path/to/your image.jpg" --save
# show more classes in the bar chart
python -m src.predict "path/to/your image.jpg" --topk 10 --save
```

**Outputs & artifacts**
- Best model: `models/efficientnet_flower_model.keras`
- Inference weights: `models/efficientnet_weights.weights.h5`
- Training histories: `results/training_history_stage{1,2,3}.json`
- Eval report: `results/classification_report.txt`
- Confusion matrix: `results/confusion_matrix.png`
- Prediction figures (optional): `results/prediction_*.png`

---

## 4) Model Configuration & Architecture

**Input & preprocessing**
- Input size: **224×224×3**
- EfficientNet **`preprocess_input`**
- On-the-fly augmentation: `RandomFlip`, `RandomRotation(0.1)`, `RandomZoom(0.1)`, `RandomContrast(0.1)`

**Backbone**
- EfficientNetB0 **without top**, built with `weights=None`
- ImageNet **no-top** weights loaded manually via `skip_mismatch=True`  
  *(this avoids rare stem-conv shape mismatches on some TF/Keras versions while keeping ImageNet initialization)*

**Head**
- `GlobalAveragePooling2D` → `Dropout(0.5)` → `Dense(num_classes, activation="softmax")`

**Training — 3 stages**
- **Stage 1 (head-only):** `learning_rate = 1e-3`, `epochs = 8`, backbone frozen
- **Stage 2 (partial unfreeze):** `learning_rate = 1e-4`, `epochs = 15`, unfreeze the **last ~50 layers** of the backbone
- **Stage 3 (full fine-tune, optional):** `learning_rate = 5e-5`, `epochs = 5`, unfreeze **all** backbone layers

**Optimization & metrics**
- Optimizer: **Adam**
- Loss: **CategoricalCrossentropy(label_smoothing=0.05)**
- Metrics: **accuracy**

**Data pipeline**
- `batch_size = 32`
- `tf.data.AUTOTUNE` prefetching

**Class imbalance**
- Class weights computed from file counts in `data/train/*`
- Injected as **`sample_weight`** per batch (works with one-hot labels)

**Callbacks**
- `EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)`
- `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)`
- `ModelCheckpoint(..., save_best_only=True)`

---

## 5) Using Other Datasets

This template is **not restricted to flowers**. It works with any multi-class image dataset as long as each class has its own folder under `data/train`, `data/val`, and (optionally) `data/test`. Class names are taken from folder names in `data/train` and must match across splits.

### Recommended data split (effective practice)
- **70/20/10** (train/val/test) is a solid default for most small-to-medium projects.  
  For larger datasets, **80/10/10** can be beneficial.
- Ensure the split is **stratified per class** so each class appears in all splits.
- Aim for **at least ~50–100 images per class** (more is better). If some classes are scarce, consider stronger augmentation and keep **validation** representative (do **not** oversample val).
- Keep the **test set untouched** until final evaluation.

---

## 6) Data Sources & Credits

- Example dataset used during development: **“Tree Dataset of Urban Street Classification — Flower”** on Kaggle (five well-represented flower classes were selected). The dataset is licensed under the GNU Lesser General Public License (LGPL), which permits use for academic and research purposes.
### link: https://www.kaggle.com/datasets/erickendric/tree-dataset-of-urban-street-classification-flower 
- EfficientNetB0: architecture background and official references by the original authors.
