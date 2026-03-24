from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "LD2_dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_CSV = DATASET_DIR / "labels.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "task1"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
TABLES_DIR = ARTIFACTS_DIR / "tables"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_JSON = ARTIFACTS_DIR / "results.json"
CACHE_FILE = CACHE_DIR / "variant5_images_cache.npz"

SEED = 42
IMAGE_SHAPE = (28, 28, 1)
NUM_CLASSES = 5
MAIN_EPOCHS = 15
CV_EPOCHS = 10
BATCH_SIZE = 256

RAW_TO_GROUPED = {
    2: 0,
    3: 0,
    4: 0,
    0: 1,
    6: 1,
    1: 2,
    5: 3,
    7: 3,
    8: 4,
}

CLASS_NAMES = {
    0: "Outerwear",
    1: "Shirts",
    2: "Pants",
    3: "Low-top shoes",
    4: "Accessories",
}

ARCHITECTURE_NOTES = {
    "variant_2": "3x3 convolution, stride 1, valid padding, MaxPooling after every convolutional block.",
    "variant_7": "3x3 convolution, stride 1, valid padding, AveragePooling then MaxPooling, GlobalAveragePooling and Dropout(0.3).",
    "variant_8": "3x3 convolution, stride 1, valid padding, AveragePooling, BatchNormalization blocks and a Dense(128) classifier head.",
    "custom_model": "3x3 convolution, stride 1, same padding, BatchNormalization, GlobalAveragePooling and Dropout regularization.",
    "defense_model": "A lightweight editable CNN reserved for quick architecture demos and defense-time experiments.",
}


def ensure_directories() -> None:
    for path in [ARTIFACTS_DIR, PLOTS_DIR, TABLES_DIR, MODELS_DIR, REPORTS_DIR, CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def configure_runtime(seed: int = SEED) -> None:
    sns.set_theme(style="whitegrid")
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    try:
        cpu_count = os.cpu_count() or 4
        tf.config.threading.set_intra_op_parallelism_threads(max(1, cpu_count // 2))
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass


def relative_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    return value


def load_variant_dataframe() -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV, dtype={"Image": str, "class_label": int})
    df["filename"] = df["Image"].str.zfill(5)
    df = df[df["class_label"].isin(RAW_TO_GROUPED)].copy()
    df["target"] = df["class_label"].map(RAW_TO_GROUPED).astype(int)
    df["target_name"] = df["target"].map(CLASS_NAMES)
    return df.sort_values("filename").reset_index(drop=True)


def create_split_dataframe(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df["target"],
        random_state=SEED,
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.17647058823529413,
        stratify=train_val_df["target"],
        random_state=SEED,
    )
    split_frames = {
        "train": train_df.sort_values("filename").reset_index(drop=True),
        "validation": val_df.sort_values("filename").reset_index(drop=True),
        "test": test_df.sort_values("filename").reset_index(drop=True),
    }
    manifest_parts: list[pd.DataFrame] = []
    for split_name, split_df in split_frames.items():
        part = split_df.copy()
        part["split"] = split_name
        manifest_parts.append(part)
    manifest_df = pd.concat(manifest_parts, ignore_index=True)
    manifest_df.to_csv(TABLES_DIR / "split_manifest.csv", index=False)
    return split_frames


def save_class_distribution_tables(df: pd.DataFrame, split_frames: dict[str, pd.DataFrame]) -> None:
    overall_counts = (
        df.groupby(["target", "target_name"])
        .size()
        .reset_index(name="count")
        .sort_values("target")
        .reset_index(drop=True)
    )
    overall_counts.to_csv(TABLES_DIR / "overall_class_counts.csv", index=False)

    records: list[dict[str, Any]] = []
    for split_name, split_df in split_frames.items():
        split_counts = split_df.groupby(["target", "target_name"]).size().reset_index(name="count")
        total = split_counts["count"].sum()
        for row in split_counts.itertuples(index=False):
            records.append(
                {
                    "split": split_name,
                    "target": int(row.target),
                    "target_name": row.target_name,
                    "count": int(row.count),
                    "share": float(row.count / total),
                }
            )
    distribution_df = pd.DataFrame(records).sort_values(["split", "target"]).reset_index(drop=True)
    distribution_df.to_csv(TABLES_DIR / "split_class_counts.csv", index=False)


def plot_class_distribution(split_frames: dict[str, pd.DataFrame]) -> Path:
    records: list[dict[str, Any]] = []
    for split_name, split_df in split_frames.items():
        counts = split_df["target"].value_counts().sort_index()
        for class_id, count in counts.items():
            records.append(
                {
                    "split": split_name,
                    "class_name": CLASS_NAMES[int(class_id)],
                    "count": int(count),
                }
            )
    plot_df = pd.DataFrame(records)
    figure_path = PLOTS_DIR / "class_distribution_by_split.png"
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=plot_df, x="class_name", y="count", hue="split", ax=ax)
    ax.set_title("Variantas 5: klasiu pasiskirstymas pagal skaidymus")
    ax.set_xlabel("Klase")
    ax.set_ylabel("Paveikslu skaicius")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def load_or_create_image_cache(df: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
    filenames = df["filename"].astype("U5").to_numpy()
    if CACHE_FILE.exists():
        cached = np.load(CACHE_FILE, allow_pickle=True)
        cached_names = cached["filenames"].astype(str)
        if len(cached_names) == len(filenames) and np.array_equal(cached_names, filenames):
            if cached["filenames"].dtype == object:
                np.savez(CACHE_FILE, images=cached["images"], filenames=filenames)
            return cached["images"], pd.Index(cached_names)

    images = np.empty((len(df), *IMAGE_SHAPE), dtype=np.uint8)
    for index, filename in enumerate(filenames):
        with Image.open(IMAGES_DIR / f"{filename}.png") as image:
            image_array = np.asarray(image, dtype=np.uint8)
            if image_array.ndim == 2:
                images[index, :, :, 0] = image_array
            else:
                images[index] = image_array[:, :, :1]
    np.savez(CACHE_FILE, images=images, filenames=filenames)
    return images, pd.Index(filenames)


def build_split_arrays(
    split_frames: dict[str, pd.DataFrame],
    images: np.ndarray,
    filename_index: pd.Index,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name, split_df in split_frames.items():
        indices = filename_index.get_indexer(split_df["filename"])
        if np.any(indices < 0):
            raise ValueError(f"Could not align all filenames for split: {split_name}")
        x = images[indices].astype("float32") / 255.0
        y = split_df["target"].to_numpy(dtype=np.int64)
        arrays[split_name] = (x, y)
    return arrays


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}


def capture_model_summary(model: keras.Model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def build_variant_2() -> keras.Model:
    inputs = keras.Input(shape=IMAGE_SHAPE, name="input_layer")
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="valid")(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="valid")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="valid")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="variant_2")


def build_variant_7() -> keras.Model:
    inputs = keras.Input(shape=IMAGE_SHAPE, name="input_layer")
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="valid")(inputs)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="valid")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="valid")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="variant_7")


def build_variant_8() -> keras.Model:
    inputs = keras.Input(shape=IMAGE_SHAPE, name="input_layer")
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="valid")(inputs)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="valid")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="variant_8")


def build_custom_model() -> keras.Model:
    inputs = keras.Input(shape=IMAGE_SHAPE, name="input_layer")
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="custom_model")


def build_defense_model() -> keras.Model:
    inputs = keras.Input(shape=IMAGE_SHAPE, name="input_layer")
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(96, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="defense_model")


def get_model_registry() -> dict[str, tuple[str, Callable[[], keras.Model]]]:
    return {
        "variant_2": ("Variantas 2", build_variant_2),
        "variant_7": ("Variantas 7", build_variant_7),
        "variant_8": ("Variantas 8", build_variant_8),
        "custom_model": ("Mano architektura", build_custom_model),
        "defense_model": ("Defense model", build_defense_model),
    }


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def plot_training_history(model_key: str, display_name: str, history_df: pd.DataFrame, best_epoch: int) -> Path:
    figure_path = PLOTS_DIR / f"{model_key}_history.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history_df["epoch"], history_df["loss"], label="Train loss", linewidth=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation loss", linewidth=2)
    axes[0].axvline(best_epoch, color="crimson", linestyle="--", label=f"Best epoch = {best_epoch}")
    axes[0].set_title(f"{display_name}: loss")
    axes[0].set_xlabel("Epocha")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["accuracy"], label="Train accuracy", linewidth=2)
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="Validation accuracy", linewidth=2)
    axes[1].axvline(best_epoch, color="crimson", linestyle="--", label=f"Best epoch = {best_epoch}")
    axes[1].set_title(f"{display_name}: accuracy")
    axes[1].set_xlabel("Epocha")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def plot_confusion_matrix(model_key: str, display_name: str, matrix: np.ndarray) -> Path:
    figure_path = PLOTS_DIR / f"{model_key}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        yticklabels=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        ax=ax,
    )
    ax.set_title(f"{display_name}: testavimo confusion matrix")
    ax.set_xlabel("Prognozuota klase")
    ax.set_ylabel("Tikroji klase")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def save_classification_report(model_key: str, report: dict[str, Any]) -> Path:
    report_path = REPORTS_DIR / f"{model_key}_classification_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(report), handle, indent=2, ensure_ascii=False)
    return report_path


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, float], np.ndarray, dict[str, Any], dict[str, float]]:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    class_accuracy = matrix.diagonal() / matrix.sum(axis=1)
    per_class_accuracy = {
        CLASS_NAMES[class_id]: float(class_accuracy[class_id]) for class_id in range(NUM_CLASSES)
    }
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "test_weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    return metrics, matrix, report, per_class_accuracy


def train_and_evaluate_model(
    model_key: str,
    display_name: str,
    builder: Callable[[], keras.Model],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_weight: dict[int, float],
    epochs: int,
    learning_rate: float = 1e-3,
    batch_size: int = BATCH_SIZE,
    save_model: bool = True,
) -> dict[str, Any]:
    print(f"\n=== Training {display_name} ===", flush=True)
    keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = compile_model(builder(), learning_rate=learning_rate)

    summary_text = capture_model_summary(model)
    summary_path = REPORTS_DIR / f"{model_key}_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5),
    ]

    train_start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        verbose=2,
        callbacks=callbacks,
    )
    training_seconds = time.perf_counter() - train_start

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_path = TABLES_DIR / f"{model_key}_history.csv"
    history_df.to_csv(history_path, index=False)

    best_epoch = int(history_df["val_loss"].idxmin() + 1)
    best_val_loss = float(history_df["val_loss"].min())
    best_val_accuracy = float(history_df.loc[history_df["val_loss"].idxmin(), "val_accuracy"])

    y_pred = model.predict(x_test, batch_size=512, verbose=0).argmax(axis=1)
    metrics, matrix, report, per_class_accuracy = evaluate_predictions(y_test, y_pred)
    history_plot = plot_training_history(model_key, display_name, history_df, best_epoch)
    confusion_plot = plot_confusion_matrix(model_key, display_name, matrix)
    report_path = save_classification_report(model_key, report)

    matrix_path = TABLES_DIR / f"{model_key}_confusion_matrix.csv"
    pd.DataFrame(
        matrix,
        index=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        columns=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
    ).to_csv(matrix_path)

    model_path: Path | None = None
    if save_model:
        model_path = MODELS_DIR / f"{model_key}.keras"
        model.save(model_path)

    result = {
        "model_key": model_key,
        "display_name": display_name,
        "params": int(model.count_params()),
        "epochs_ran": int(len(history_df)),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "training_seconds": float(training_seconds),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "train_samples": int(len(x_train)),
        "history_csv": relative_path(history_path),
        "history_plot": relative_path(history_plot),
        "confusion_matrix_csv": relative_path(matrix_path),
        "confusion_matrix_plot": relative_path(confusion_plot),
        "classification_report_json": relative_path(report_path),
        "model_summary_txt": relative_path(summary_path),
        "model_path": relative_path(model_path) if model_path is not None else None,
        "per_class_accuracy": per_class_accuracy,
        **metrics,
    }

    del model
    gc.collect()
    keras.backend.clear_session()

    return result


def plot_example_images(df: pd.DataFrame, images: np.ndarray, filename_index: pd.Index) -> Path:
    sample_rows = (
        df.groupby("target", sort=True)
        .head(5)
        .sort_values(["target", "filename"])
        .reset_index(drop=True)
    )
    indices = filename_index.get_indexer(sample_rows["filename"])
    figure_path = PLOTS_DIR / "example_images.png"
    fig, axes = plt.subplots(NUM_CLASSES, 5, figsize=(8, 8))
    for ax, idx, row in zip(axes.flat, indices, sample_rows.itertuples(index=False)):
        ax.imshow(images[idx, :, :, 0], cmap="gray")
        ax.set_title(f"{row.filename}\n{row.target_name}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Po 5 pavyzdzius is kiekvienos sugrupuotos klases", y=1.02)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def save_model_comparison(results: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    comparison_columns = [
        "display_name",
        "params",
        "epochs_ran",
        "best_epoch",
        "best_val_loss",
        "best_val_accuracy",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "training_seconds",
    ]
    comparison_df = pd.DataFrame(results)[comparison_columns].sort_values(
        "test_balanced_accuracy", ascending=False
    )
    comparison_path = TABLES_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    per_class_df = pd.DataFrame(
        [
            {"display_name": result["display_name"], **result["per_class_accuracy"]}
            for result in results
        ]
    )
    per_class_path = TABLES_DIR / "per_class_accuracy.csv"
    per_class_df.to_csv(per_class_path, index=False)

    score_plot_path = PLOTS_DIR / "model_comparison_balanced_accuracy.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=comparison_df,
        x="display_name",
        y="test_balanced_accuracy",
        palette="crest",
        ax=ax,
    )
    ax.set_title("Modeliu palyginimas pagal balanced accuracy testavimo aibeje")
    ax.set_xlabel("Modelis")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(f"{height:.3f}", (patch.get_x() + patch.get_width() / 2, height), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(score_plot_path, dpi=160)
    plt.close(fig)

    heatmap_path = PLOTS_DIR / "per_class_accuracy_heatmap.png"
    heatmap_df = per_class_df.set_index("display_name")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
    ax.set_title("Per-klasinis tikslumas testavimo aibeje")
    ax.set_xlabel("Klase")
    ax.set_ylabel("Modelis")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=160)
    plt.close(fig)

    return comparison_path, per_class_path, score_plot_path


def run_sample_size_study(
    train_df: pd.DataFrame,
    images: np.ndarray,
    filename_index: pd.Index,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    sorted_train_df = train_df.sort_values("filename").reset_index(drop=True)
    max_size = len(sorted_train_df)
    candidate_sizes = [5000, 10000, 20000, 30000]
    candidate_sizes = [size for size in candidate_sizes if size < max_size]
    if not candidate_sizes:
        candidate_sizes.append(max_size)

    cv_records: list[dict[str, Any]] = []
    for sample_size in candidate_sizes:
        subset_df = sorted_train_df.iloc[:sample_size].copy()
        subset_indices = filename_index.get_indexer(subset_df["filename"])
        x_subset = images[subset_indices].astype("float32") / 255.0
        y_subset = subset_df["target"].to_numpy(dtype=np.int64)
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

        print(f"\n=== Sample size study: first {sample_size} training images ===", flush=True)
        for fold_idx, (train_index, valid_index) in enumerate(splitter.split(x_subset, y_subset), start=1):
            keras.backend.clear_session()
            tf.keras.utils.set_random_seed(SEED + fold_idx)
            model = compile_model(build_custom_model(), learning_rate=5e-4)
            fold_class_weight = compute_class_weights(y_subset[train_index])
            callbacks = [
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
            ]
            history = model.fit(
                x_subset[train_index],
                y_subset[train_index],
                validation_data=(x_subset[valid_index], y_subset[valid_index]),
                epochs=10,
                batch_size=128,
                class_weight=fold_class_weight,
                verbose=0,
                callbacks=callbacks,
            )
            best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
            y_pred = model.predict(x_subset[valid_index], batch_size=512, verbose=0).argmax(axis=1)
            cv_records.append(
                {
                    "sample_size": int(sample_size),
                    "fold": int(fold_idx),
                    "accuracy": float(accuracy_score(y_subset[valid_index], y_pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_subset[valid_index], y_pred)),
                    "macro_f1": float(f1_score(y_subset[valid_index], y_pred, average="macro")),
                    "best_epoch": best_epoch,
                }
            )
            del model
            gc.collect()
            keras.backend.clear_session()

    cv_df = pd.DataFrame(cv_records)
    cv_path = TABLES_DIR / "sample_size_cv_results.csv"
    cv_df.to_csv(cv_path, index=False)

    summary_df = (
        cv_df.groupby("sample_size")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_best_epoch=("best_epoch", "mean"),
        )
        .reset_index()
    )
    summary_path = TABLES_DIR / "sample_size_cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    figure_path = PLOTS_DIR / "sample_size_cv_curve.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        summary_df["sample_size"],
        summary_df["mean_balanced_accuracy"],
        yerr=summary_df["std_balanced_accuracy"].fillna(0),
        marker="o",
        linewidth=2,
        capsize=4,
    )
    ax.set_title("Custom model: 3-fold CV balanced accuracy pagal imties dydi")
    ax.set_xlabel("Naudotu treniravimo paveikslu skaicius (pirmi N pagal pavadinima)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    best_mean_balanced = float(summary_df["mean_balanced_accuracy"].max())
    threshold_ratio = 0.98
    threshold_value = best_mean_balanced * threshold_ratio
    acceptable_size = int(
        summary_df.loc[summary_df["mean_balanced_accuracy"] >= threshold_value, "sample_size"].min()
    )

    reduced_df = sorted_train_df.iloc[:acceptable_size].copy()
    reduced_indices = filename_index.get_indexer(reduced_df["filename"])
    x_reduced = images[reduced_indices].astype("float32") / 255.0
    y_reduced = reduced_df["target"].to_numpy(dtype=np.int64)
    reduced_class_weight = compute_class_weights(y_reduced)
    reduced_result = train_and_evaluate_model(
        model_key="custom_model_reduced_sample",
        display_name=f"Custom model (first {acceptable_size} images)",
        builder=build_custom_model,
        x_train=x_reduced,
        y_train=y_reduced,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        class_weight=reduced_class_weight,
        epochs=MAIN_EPOCHS,
    )

    return {
        "candidate_sizes": candidate_sizes,
        "criterion": "Accepted the smallest N whose mean 3-fold balanced accuracy reached at least 98% of the best tested mean.",
        "threshold_ratio": threshold_ratio,
        "threshold_value": threshold_value,
        "cv_results_csv": relative_path(cv_path),
        "cv_summary_csv": relative_path(summary_path),
        "cv_plot": relative_path(figure_path),
        "acceptable_size": acceptable_size,
        "reduced_model_result": reduced_result,
    }


def determine_best_presented_model(results: list[dict[str, Any]]) -> dict[str, Any]:
    presented = [result for result in results if result["model_key"] in {"variant_2", "variant_7", "variant_8"}]
    return max(presented, key=lambda item: item["test_balanced_accuracy"])


def build_results_payload(
    variant_df: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    main_results: list[dict[str, Any]],
    sample_size_result: dict[str, Any],
    example_plot_path: Path,
    distribution_plot_path: Path,
    comparison_path: Path,
    per_class_path: Path,
    comparison_plot_path: Path,
) -> dict[str, Any]:
    split_counts = {
        split_name: {
            CLASS_NAMES[int(class_id)]: int(count)
            for class_id, count in split_df["target"].value_counts().sort_index().items()
        }
        for split_name, split_df in split_frames.items()
    }

    best_presented = determine_best_presented_model(main_results)
    custom_full_result = next(result for result in main_results if result["model_key"] == "custom_model")
    reduced_result = sample_size_result["reduced_model_result"]

    payload = {
        "config": {
            "seed": SEED,
            "image_shape": list(IMAGE_SHAPE),
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "raw_to_grouped": RAW_TO_GROUPED,
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "main_epochs": MAIN_EPOCHS,
            "cv_epochs": CV_EPOCHS,
            "batch_size": BATCH_SIZE,
            "tensorflow_version": tf.__version__,
            "python_version": sys.version.split()[0],
            "architecture_notes": ARCHITECTURE_NOTES,
        },
        "dataset": {
            "filtered_samples": int(len(variant_df)),
            "split_counts": split_counts,
            "split_manifest_csv": relative_path(TABLES_DIR / "split_manifest.csv"),
            "overall_class_counts_csv": relative_path(TABLES_DIR / "overall_class_counts.csv"),
            "split_class_counts_csv": relative_path(TABLES_DIR / "split_class_counts.csv"),
            "example_images_plot": relative_path(example_plot_path),
            "class_distribution_plot": relative_path(distribution_plot_path),
        },
        "main_models": main_results,
        "artifacts": {
            "model_comparison_csv": relative_path(comparison_path),
            "per_class_accuracy_csv": relative_path(per_class_path),
            "model_comparison_plot": relative_path(comparison_plot_path),
            "per_class_accuracy_heatmap": relative_path(PLOTS_DIR / "per_class_accuracy_heatmap.png"),
        },
        "sample_size_study": sample_size_result,
        "conclusions": {
            "best_presented_model": best_presented["display_name"],
            "best_presented_balanced_accuracy": best_presented["test_balanced_accuracy"],
            "custom_model_balanced_accuracy": custom_full_result["test_balanced_accuracy"],
            "reduced_custom_balanced_accuracy": reduced_result["test_balanced_accuracy"],
            "acceptable_sample_size": sample_size_result["acceptable_size"],
        },
    }
    return payload


def build_quick_train_subset(
    train_df: pd.DataFrame,
    images: np.ndarray,
    filename_index: pd.Index,
    train_limit: int | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    sorted_train_df = train_df.sort_values("filename").reset_index(drop=True)
    if train_limit is None or train_limit <= 0 or train_limit >= len(sorted_train_df):
        subset_df = sorted_train_df
    else:
        subset_df = sorted_train_df.iloc[:train_limit].copy()

    subset_indices = filename_index.get_indexer(subset_df["filename"])
    if np.any(subset_indices < 0):
        raise ValueError("Could not align all filenames for the quick-run subset.")

    x_subset = images[subset_indices].astype("float32") / 255.0
    y_subset = subset_df["target"].to_numpy(dtype=np.int64)
    return subset_df, x_subset, y_subset


def sanitize_run_name(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "quick_run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1 CNN experiments")
    parser.add_argument("--mode", choices=["full", "quick"], default="full")
    parser.add_argument(
        "--model",
        default="defense_model",
        help="Model key for quick mode. Available keys: variant_2, variant_7, variant_8, custom_model, defense_model.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of epochs.")
    parser.add_argument(
        "--train-limit",
        type=int,
        default=10000,
        help="Use the first N training images by filename in quick mode. Use 0 for the full train split.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size used in quick mode.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate used in quick mode.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional suffix for quick-run artifact names.")
    parser.add_argument("--list-models", action="store_true", help="Print available model keys and exit.")
    return parser.parse_args()


def run_quick_mode(args: argparse.Namespace) -> None:
    registry = get_model_registry()
    if args.model not in registry:
        raise ValueError(f"Unknown model key '{args.model}'. Use --list-models to inspect the options.")

    ensure_directories()
    configure_runtime()

    variant_df = load_variant_dataframe()
    split_frames = create_split_dataframe(variant_df)
    images, filename_index = load_or_create_image_cache(variant_df)
    arrays = build_split_arrays(split_frames, images, filename_index)

    subset_df, x_train, y_train = build_quick_train_subset(
        train_df=split_frames["train"],
        images=images,
        filename_index=filename_index,
        train_limit=args.train_limit,
    )
    x_val, y_val = arrays["validation"]
    x_test, y_test = arrays["test"]
    class_weight = compute_class_weights(y_train)

    display_name, builder = registry[args.model]
    run_suffix = sanitize_run_name(args.run_name) if args.run_name else f"{args.model}_{len(subset_df)}"
    artifact_key = f"quick_{run_suffix}"
    epochs = args.epochs or 6

    print(
        f"\nQuick mode: training '{display_name}' with {len(subset_df)} training images for up to {epochs} epochs.",
        flush=True,
    )
    result = train_and_evaluate_model(
        model_key=artifact_key,
        display_name=f"{display_name} [quick]",
        builder=builder,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        class_weight=class_weight,
        epochs=epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        save_model=False,
    )

    summary_path = REPORTS_DIR / f"{artifact_key}_result.json"
    quick_payload = {
        "mode": "quick",
        "model_key": args.model,
        "artifact_key": artifact_key,
        "train_limit": int(len(subset_df)),
        "epochs_requested": int(epochs),
        "result": result,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(quick_payload), handle, indent=2, ensure_ascii=False)

    print("\nQuick run summary:", flush=True)
    print(json.dumps(to_serializable(quick_payload), indent=2, ensure_ascii=False), flush=True)
    print(f"\nQuick-run result saved to: {summary_path}", flush=True)


def main() -> None:
    args = parse_args()
    registry = get_model_registry()

    if args.list_models:
        for model_key, (display_name, _) in registry.items():
            print(f"{model_key}: {display_name}")
        return

    if args.mode == "quick":
        run_quick_mode(args)
        return

    ensure_directories()
    configure_runtime()

    variant_df = load_variant_dataframe()
    split_frames = create_split_dataframe(variant_df)
    save_class_distribution_tables(variant_df, split_frames)
    distribution_plot_path = plot_class_distribution(split_frames)

    images, filename_index = load_or_create_image_cache(variant_df)
    example_plot_path = plot_example_images(variant_df, images, filename_index)
    arrays = build_split_arrays(split_frames, images, filename_index)

    x_train, y_train = arrays["train"]
    x_val, y_val = arrays["validation"]
    x_test, y_test = arrays["test"]
    class_weight = compute_class_weights(y_train)

    builders = [
        (model_key, display_name, builder)
        for model_key, (display_name, builder) in get_model_registry().items()
        if model_key != "defense_model"
    ]

    main_results: list[dict[str, Any]] = []
    for model_key, display_name, builder in builders:
        result = train_and_evaluate_model(
            model_key=model_key,
            display_name=display_name,
            builder=builder,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            class_weight=class_weight,
            epochs=MAIN_EPOCHS,
        )
        main_results.append(result)

    comparison_path, per_class_path, comparison_plot_path = save_model_comparison(main_results)
    sample_size_result = run_sample_size_study(
        train_df=split_frames["train"],
        images=images,
        filename_index=filename_index,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )

    results_payload = build_results_payload(
        variant_df=variant_df,
        split_frames=split_frames,
        main_results=main_results,
        sample_size_result=sample_size_result,
        example_plot_path=example_plot_path,
        distribution_plot_path=distribution_plot_path,
        comparison_path=comparison_path,
        per_class_path=per_class_path,
        comparison_plot_path=comparison_plot_path,
    )
    with RESULTS_JSON.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(results_payload), handle, indent=2, ensure_ascii=False)

    print(f"\nAll task 1 artifacts saved to: {ARTIFACTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
