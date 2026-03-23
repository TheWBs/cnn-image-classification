from __future__ import annotations

import gc
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


PROJECT_ROOT = Path(__file__).resolve().parent
TASK2_DATASET_DIR = PROJECT_ROOT / "task2_dataset"
CURATED_DIR = TASK2_DATASET_DIR / "curated_96"
MANIFEST_CSV = TASK2_DATASET_DIR / "manifest.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "task2"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
TABLES_DIR = ARTIFACTS_DIR / "tables"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
RESULTS_JSON = ARTIFACTS_DIR / "results.json"

CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_FILE = CACHE_DIR / "task2_curated_images_96.npz"

SEED = 42
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 12
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
PER_CLASS_PER_SOURCE = 40

CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

SOURCE_CONFIG = {
    "tf_flowers": {
        "dataset_name": "tf_flowers",
        "split": "train",
        "label_mapping": {
            "daisy": "daisy",
            "dandelion": "dandelion",
            "roses": "rose",
            "sunflowers": "sunflower",
        },
        "description": "TensorFlow flower photos dataset",
        "source_url": "https://www.tensorflow.org/datasets/catalog/tf_flowers",
    },
    "oxford_flowers102": {
        "dataset_name": "oxford_flowers102",
        "split": "train+validation+test",
        "label_mapping": {
            "oxeye daisy": "daisy",
            "common dandelion": "dandelion",
            "rose": "rose",
            "sunflower": "sunflower",
        },
        "description": "Oxford 102 Flowers dataset",
        "source_url": "https://www.tensorflow.org/datasets/catalog/oxford_flowers102",
    },
}

EXPERIMENTS = [
    {"name": "scratch_no_aug", "use_pretrained": False, "use_augmentation": False, "label": "Be transfer, be augmentacijos"},
    {"name": "scratch_aug", "use_pretrained": False, "use_augmentation": True, "label": "Be transfer, su augmentacija"},
    {"name": "pretrained_no_aug", "use_pretrained": True, "use_augmentation": False, "label": "Transfer learning, be augmentacijos"},
    {"name": "pretrained_aug", "use_pretrained": True, "use_augmentation": True, "label": "Transfer learning, su augmentacija"},
]

SUBSET_FRACTIONS = [0.25, 0.50, 1.00]


def ensure_directories() -> None:
    for path in [TASK2_DATASET_DIR, CURATED_DIR, ARTIFACTS_DIR, PLOTS_DIR, TABLES_DIR, MODELS_DIR, REPORTS_DIR, CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def configure_runtime(seed: int = SEED) -> None:
    sns.set_theme(style="whitegrid")
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
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
    logging.getLogger("tensorflow_datasets").setLevel(logging.ERROR)
    tfds.disable_progress_bar()


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


def sanitize_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def resize_rgb_image(image_array: np.ndarray) -> Image.Image:
    image = Image.fromarray(image_array)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return ImageOps.fit(image, IMAGE_SIZE, method=Image.Resampling.BILINEAR)


def prepare_curated_dataset() -> pd.DataFrame:
    expected_total = len(CLASS_NAMES) * len(SOURCE_CONFIG) * PER_CLASS_PER_SOURCE
    if MANIFEST_CSV.exists():
        manifest = pd.read_csv(MANIFEST_CSV)
        if len(manifest) == expected_total and manifest["filepath"].apply(lambda value: (PROJECT_ROOT / value).exists()).all():
            return manifest

    records: list[dict[str, Any]] = []
    for source_name, source_cfg in SOURCE_CONFIG.items():
        dataset, info = tfds.load(
            source_cfg["dataset_name"],
            split=source_cfg["split"],
            with_info=True,
            as_supervised=True,
            shuffle_files=False,
        )
        label_mapping = source_cfg["label_mapping"]
        collected_counts = defaultdict(int)
        needed_classes = set(label_mapping.values())

        for image_array, label in tfds.as_numpy(dataset):
            label_name = info.features["label"].int2str(int(label))
            if label_name not in label_mapping:
                continue
            class_name = label_mapping[label_name]
            if collected_counts[class_name] >= PER_CLASS_PER_SOURCE:
                if all(collected_counts[target] >= PER_CLASS_PER_SOURCE for target in needed_classes):
                    break
                continue

            output_dir = CURATED_DIR / source_name / class_name
            output_dir.mkdir(parents=True, exist_ok=True)
            file_stem = f"{source_name}_{sanitize_name(class_name)}_{collected_counts[class_name]:03d}"
            output_path = output_dir / f"{file_stem}.jpg"
            resized = resize_rgb_image(image_array)
            resized.save(output_path, format="JPEG", quality=90)

            records.append(
                {
                    "source": source_name,
                    "source_description": source_cfg["description"],
                    "source_url": source_cfg["source_url"],
                    "original_label": label_name,
                    "class_name": class_name,
                    "class_id": CLASS_TO_INDEX[class_name],
                    "filename": output_path.name,
                    "filepath": relative_path(output_path),
                }
            )
            collected_counts[class_name] += 1

        if any(collected_counts[class_name] < PER_CLASS_PER_SOURCE for class_name in CLASS_NAMES):
            raise ValueError(f"Not enough images collected for source {source_name}: {dict(collected_counts)}")

    manifest = pd.DataFrame(records).sort_values(["source", "class_name", "filename"]).reset_index(drop=True)
    manifest["source_class"] = manifest["source"] + "|" + manifest["class_name"]

    train_val_df, test_df = train_test_split(
        manifest,
        test_size=TEST_FRACTION,
        stratify=manifest["source_class"],
        random_state=SEED,
    )
    validation_relative = VALIDATION_FRACTION / (TRAIN_FRACTION + VALIDATION_FRACTION)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=validation_relative,
        stratify=train_val_df["source_class"],
        random_state=SEED,
    )

    split_parts: list[pd.DataFrame] = []
    for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        part = split_df.copy()
        part["split"] = split_name
        split_parts.append(part)

    manifest = pd.concat(split_parts, ignore_index=True).sort_values(["split", "source", "class_name", "filename"]).reset_index(drop=True)
    manifest.to_csv(MANIFEST_CSV, index=False)
    return manifest


def save_dataset_tables(manifest: pd.DataFrame) -> None:
    source_counts = (
        manifest.groupby(["source", "class_name", "split"])
        .size()
        .reset_index(name="count")
        .sort_values(["source", "class_name", "split"])
        .reset_index(drop=True)
    )
    source_counts.to_csv(TABLES_DIR / "dataset_counts_by_source_class_split.csv", index=False)

    summary_counts = (
        manifest.groupby(["class_name", "split"])
        .size()
        .reset_index(name="count")
        .sort_values(["class_name", "split"])
        .reset_index(drop=True)
    )
    summary_counts.to_csv(TABLES_DIR / "dataset_counts_by_class_split.csv", index=False)

    source_info = (
        manifest[["source", "source_description", "source_url"]]
        .drop_duplicates()
        .sort_values("source")
        .reset_index(drop=True)
    )
    source_info.to_csv(TABLES_DIR / "dataset_sources.csv", index=False)


def load_or_create_image_cache(manifest: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
    expected_files = manifest["filepath"].astype(str).to_numpy()
    if CACHE_FILE.exists():
        cached = np.load(CACHE_FILE, allow_pickle=True)
        cached_files = cached["filepaths"].astype(str)
        if len(cached_files) == len(expected_files) and np.array_equal(cached_files, expected_files):
            return cached["images"], pd.Index(cached_files)

    images = np.empty((len(manifest), IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
    for row_index, row in enumerate(manifest.itertuples(index=False)):
        with Image.open(PROJECT_ROOT / row.filepath) as image:
            images[row_index] = np.asarray(image.convert("RGB"), dtype=np.uint8)
    np.savez(CACHE_FILE, images=images, filepaths=expected_files)
    return images, pd.Index(expected_files)


def build_split_arrays(
    manifest: pd.DataFrame,
    images: np.ndarray,
    filepath_index: pd.Index,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    manifest = manifest.copy()
    manifest["array_index"] = filepath_index.get_indexer(manifest["filepath"])
    if (manifest["array_index"] < 0).any():
        raise ValueError("Could not align manifest rows with image cache.")

    for split_name in ["train", "validation", "test"]:
        split_df = manifest[manifest["split"] == split_name].copy().reset_index(drop=True)
        x = images[split_df["array_index"].to_numpy()].astype("float32")
        y = split_df["class_id"].to_numpy(dtype=np.int64)
        arrays[split_name] = (x, y)
    return arrays, manifest


def create_nested_subsets(train_manifest: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    subset_df = train_manifest.copy().reset_index(drop=True)
    subset_df["subset_order"] = -1
    rng = np.random.default_rng(SEED)

    for _, group in subset_df.groupby("source_class", sort=True):
        shuffled_indices = group.index.to_numpy().copy()
        rng.shuffle(shuffled_indices)
        subset_df.loc[shuffled_indices, "subset_order"] = np.arange(len(shuffled_indices))

    subset_records: list[dict[str, Any]] = []
    subset_indices: dict[str, np.ndarray] = {}
    for fraction in SUBSET_FRACTIONS:
        chosen_indices: list[int] = []
        for source_class, group in subset_df.groupby("source_class", sort=True):
            sample_count = max(1, int(round(len(group) * fraction)))
            sample_count = min(sample_count, len(group))
            selected = group.nsmallest(sample_count, "subset_order").index.to_list()
            chosen_indices.extend(selected)
            class_name = group["class_name"].iloc[0]
            source_name = group["source"].iloc[0]
            subset_records.append(
                {
                    "fraction": fraction,
                    "subset_name": f"{int(fraction * 100):02d}pct",
                    "source_class": source_class,
                    "source": source_name,
                    "class_name": class_name,
                    "count": sample_count,
                }
            )
        subset_name = f"{int(fraction * 100):02d}pct"
        subset_indices[subset_name] = np.array(sorted(chosen_indices), dtype=np.int64)

    subset_table = pd.DataFrame(subset_records).sort_values(["fraction", "source", "class_name"]).reset_index(drop=True)
    subset_table.to_csv(TABLES_DIR / "train_subset_counts.csv", index=False)
    return subset_indices, subset_df


def create_example_plot(manifest: pd.DataFrame) -> Path:
    figure_path = PLOTS_DIR / "dataset_examples.png"
    sample_df = (
        manifest.groupby(["source", "class_name"], sort=True)
        .head(1)
        .sort_values(["class_name", "source"])
        .reset_index(drop=True)
    )
    fig, axes = plt.subplots(len(CLASS_NAMES), len(SOURCE_CONFIG), figsize=(7, 10))
    for ax, row in zip(axes.flat, sample_df.itertuples(index=False)):
        with Image.open(PROJECT_ROOT / row.filepath) as image:
            ax.imshow(image)
        ax.set_title(f"{row.class_name}\n{row.source}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Po viena pavyzdi is kiekvienos klases ir kiekvieno saltinio", y=1.01)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def create_dataset_distribution_plot(manifest: pd.DataFrame) -> Path:
    figure_path = PLOTS_DIR / "dataset_distribution.png"
    plot_df = (
        manifest.groupby(["split", "class_name", "source"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "class_name", "source"])
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, split_name in zip(axes, ["train", "validation", "test"]):
        split_plot_df = plot_df[plot_df["split"] == split_name]
        sns.barplot(data=split_plot_df, x="class_name", y="count", hue="source", ax=ax)
        ax.set_title(split_name.capitalize())
        ax.set_xlabel("Klase")
        ax.set_ylabel("Paveikslu skaicius")
        ax.tick_params(axis="x", rotation=15)
        if split_name != "train":
            ax.get_legend().remove()
    axes[0].legend(title="Saltinis")
    fig.suptitle("Custom duomenu rinkinio balansavimas pagal skaidymus", y=1.02)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def build_data_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )


def build_model(use_pretrained: bool, use_augmentation: bool) -> keras.Model:
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="image")
    x = layers.Lambda(lambda value: tf.cast(value, tf.float32), name="cast_float32")(inputs)
    if use_augmentation:
        x = build_data_augmentation()(x)
    x = layers.Lambda(preprocess_input, name="mobilenet_preprocess")(x)

    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet" if use_pretrained else None,
        alpha=0.35,
    )
    base_model.trainable = True
    x = base_model(x, training=not use_pretrained)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"mobilenetv2_{'pretrained' if use_pretrained else 'scratch'}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def save_history_plot(experiment_id: str, label: str, history_df: pd.DataFrame, best_epoch: int) -> Path:
    figure_path = PLOTS_DIR / f"{experiment_id}_history.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["loss"], label="Train loss", linewidth=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation loss", linewidth=2)
    axes[0].axvline(best_epoch, color="crimson", linestyle="--", label=f"Best epoch = {best_epoch}")
    axes[0].set_title(f"{label}: loss")
    axes[0].set_xlabel("Epocha")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["accuracy"], label="Train accuracy", linewidth=2)
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="Validation accuracy", linewidth=2)
    axes[1].axvline(best_epoch, color="crimson", linestyle="--", label=f"Best epoch = {best_epoch}")
    axes[1].set_title(f"{label}: accuracy")
    axes[1].set_xlabel("Epocha")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def save_confusion_matrix_plot(experiment_id: str, label: str, matrix: np.ndarray) -> Path:
    figure_path = PLOTS_DIR / f"{experiment_id}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title(f"{label}: testavimo confusion matrix")
    ax.set_xlabel("Prognozuota klase")
    ax.set_ylabel("Tikroji klase")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, float], np.ndarray, dict[str, Any]]:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "test_weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    return metrics, matrix, report


def capture_model_summary(model: keras.Model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def run_single_experiment(
    experiment_cfg: dict[str, Any],
    subset_name: str,
    subset_fraction: float,
    subset_indices: np.ndarray,
    train_manifest: pd.DataFrame,
    x_train_full: np.ndarray,
    y_train_full: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    experiment_id = f"{experiment_cfg['name']}_{subset_name}"
    label = f"{experiment_cfg['label']} ({subset_name})"
    print(f"\n=== Running {label} ===", flush=True)

    keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    x_train = x_train_full[subset_indices]
    y_train = y_train_full[subset_indices]
    subset_manifest = train_manifest.iloc[subset_indices].copy()
    subset_count = int(len(subset_manifest))

    model = build_model(
        use_pretrained=bool(experiment_cfg["use_pretrained"]),
        use_augmentation=bool(experiment_cfg["use_augmentation"]),
    )

    summary_text = capture_model_summary(model)
    summary_path = REPORTS_DIR / f"{experiment_id}_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    start_time = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=callbacks,
    )
    training_seconds = time.perf_counter() - start_time

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_path = TABLES_DIR / f"{experiment_id}_history.csv"
    history_df.to_csv(history_path, index=False)

    best_epoch_idx = int(history_df["val_loss"].idxmin())
    best_epoch = best_epoch_idx + 1
    best_val_accuracy = float(history_df.loc[best_epoch_idx, "val_accuracy"])
    best_val_loss = float(history_df.loc[best_epoch_idx, "val_loss"])

    y_pred = model.predict(x_test, batch_size=64, verbose=0).argmax(axis=1)
    metrics, matrix, report = evaluate_predictions(y_test, y_pred)

    report_path = REPORTS_DIR / f"{experiment_id}_classification_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(report), handle, indent=2, ensure_ascii=False)

    confusion_csv = TABLES_DIR / f"{experiment_id}_confusion_matrix.csv"
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(confusion_csv)

    history_plot = save_history_plot(experiment_id, label, history_df, best_epoch)
    confusion_plot = save_confusion_matrix_plot(experiment_id, label, matrix)

    result = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_cfg["name"],
        "label": experiment_cfg["label"],
        "subset_name": subset_name,
        "subset_fraction": subset_fraction,
        "train_size": subset_count,
        "use_pretrained": bool(experiment_cfg["use_pretrained"]),
        "use_augmentation": bool(experiment_cfg["use_augmentation"]),
        "params": int(model.count_params()),
        "epochs_ran": int(len(history_df)),
        "best_epoch": int(best_epoch),
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "training_seconds": float(training_seconds),
        "history_csv": relative_path(history_path),
        "history_plot": relative_path(history_plot),
        "confusion_matrix_csv": relative_path(confusion_csv),
        "confusion_matrix_plot": relative_path(confusion_plot),
        "classification_report_json": relative_path(report_path),
        "model_summary_txt": relative_path(summary_path),
        **metrics,
    }

    del model
    gc.collect()
    keras.backend.clear_session()
    return result


def create_results_summary(experiment_results: list[dict[str, Any]]) -> tuple[Path, Path]:
    results_df = pd.DataFrame(experiment_results).sort_values(
        ["use_pretrained", "use_augmentation", "subset_fraction"]
    )
    summary_path = TABLES_DIR / "experiment_summary.csv"
    results_df.to_csv(summary_path, index=False)

    plot_path = PLOTS_DIR / "accuracy_vs_train_size.png"
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.lineplot(
        data=results_df,
        x="train_size",
        y="test_accuracy",
        hue="label",
        style="use_pretrained",
        markers=True,
        dashes=False,
        linewidth=2.5,
        ax=ax,
    )
    ax.set_title("Test accuracy priklausomybe nuo treniravimo duomenu kiekio")
    ax.set_xlabel("Treniravimo paveikslu skaicius")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    bar_path = PLOTS_DIR / "final_size_comparison.png"
    final_df = results_df[results_df["subset_fraction"] == max(SUBSET_FRACTIONS)].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=final_df, x="label", y="test_accuracy", palette="crest", ax=ax)
    ax.set_title("Pilnos treniravimo imties rezultatai")
    ax.set_xlabel("Eksperimentas")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(f"{height:.3f}", (patch.get_x() + patch.get_width() / 2, height), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(bar_path, dpi=160)
    plt.close(fig)

    return summary_path, plot_path


def select_best_experiment(experiment_results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(experiment_results, key=lambda item: (item["best_val_accuracy"], item["test_accuracy"]))


def save_best_model(
    best_result: dict[str, Any],
    experiment_lookup: dict[str, dict[str, Any]],
    subset_lookup: dict[str, np.ndarray],
    train_manifest: pd.DataFrame,
    x_train_full: np.ndarray,
    y_train_full: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> str:
    experiment_cfg = experiment_lookup[best_result["experiment_name"]]
    subset_name = best_result["subset_name"]
    indices = subset_lookup[subset_name]
    x_train = x_train_full[indices]
    y_train = y_train_full[indices]

    keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = build_model(
        use_pretrained=bool(experiment_cfg["use_pretrained"]),
        use_augmentation=bool(experiment_cfg["use_augmentation"]),
    )
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=callbacks,
    )
    model_path = MODELS_DIR / f"{best_result['experiment_id']}.keras"
    model.save(model_path)
    del model
    gc.collect()
    keras.backend.clear_session()
    return relative_path(model_path)


def build_results_payload(
    manifest: pd.DataFrame,
    subset_counts: pd.DataFrame,
    example_plot_path: Path,
    distribution_plot_path: Path,
    summary_csv_path: Path,
    accuracy_plot_path: Path,
    experiment_results: list[dict[str, Any]],
    best_result: dict[str, Any],
    best_model_path: str,
) -> dict[str, Any]:
    dataset_counts = (
        manifest.groupby(["source", "class_name", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    payload = {
        "config": {
            "seed": SEED,
            "image_size": list(IMAGE_SIZE),
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "per_class_per_source": PER_CLASS_PER_SOURCE,
            "class_names": CLASS_NAMES,
            "subset_fractions": SUBSET_FRACTIONS,
            "model_architecture": "MobileNetV2 alpha=0.35, input 96x96, include_top=False",
            "tensorflow_version": tf.__version__,
            "python_version": sys.version.split()[0],
        },
        "dataset": {
            "total_samples": int(len(manifest)),
            "train_samples": int((manifest["split"] == "train").sum()),
            "validation_samples": int((manifest["split"] == "validation").sum()),
            "test_samples": int((manifest["split"] == "test").sum()),
            "sources": list(SOURCE_CONFIG.keys()),
            "source_table_csv": relative_path(TABLES_DIR / "dataset_sources.csv"),
            "counts_table_csv": relative_path(TABLES_DIR / "dataset_counts_by_source_class_split.csv"),
            "class_split_table_csv": relative_path(TABLES_DIR / "dataset_counts_by_class_split.csv"),
            "manifest_csv": relative_path(MANIFEST_CSV),
            "subset_counts_csv": relative_path(TABLES_DIR / "train_subset_counts.csv"),
            "example_plot": relative_path(example_plot_path),
            "distribution_plot": relative_path(distribution_plot_path),
            "counts_preview": to_serializable(dataset_counts.to_dict(orient="records")),
        },
        "experiments": experiment_results,
        "artifacts": {
            "experiment_summary_csv": relative_path(summary_csv_path),
            "accuracy_vs_train_size_plot": relative_path(accuracy_plot_path),
            "final_size_comparison_plot": relative_path(PLOTS_DIR / "final_size_comparison.png"),
        },
        "best_experiment": {
            **best_result,
            "model_path": best_model_path,
        },
    }
    return payload


def main() -> None:
    ensure_directories()
    configure_runtime()

    manifest = prepare_curated_dataset()
    save_dataset_tables(manifest)
    example_plot_path = create_example_plot(manifest)
    distribution_plot_path = create_dataset_distribution_plot(manifest)

    images, filepath_index = load_or_create_image_cache(manifest)
    arrays, manifest_with_index = build_split_arrays(manifest, images, filepath_index)

    train_manifest = manifest_with_index[manifest_with_index["split"] == "train"].copy().reset_index(drop=True)
    subset_lookup, subset_counts = create_nested_subsets(train_manifest)

    x_train_full, y_train_full = arrays["train"]
    x_val, y_val = arrays["validation"]
    x_test, y_test = arrays["test"]

    experiment_results: list[dict[str, Any]] = []
    experiment_lookup = {config["name"]: config for config in EXPERIMENTS}
    for experiment_cfg in EXPERIMENTS:
        for fraction in SUBSET_FRACTIONS:
            subset_name = f"{int(fraction * 100):02d}pct"
            result = run_single_experiment(
                experiment_cfg=experiment_cfg,
                subset_name=subset_name,
                subset_fraction=fraction,
                subset_indices=subset_lookup[subset_name],
                train_manifest=train_manifest,
                x_train_full=x_train_full,
                y_train_full=y_train_full,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
            )
            experiment_results.append(result)

    summary_csv_path, accuracy_plot_path = create_results_summary(experiment_results)
    best_result = select_best_experiment(experiment_results)
    best_model_path = save_best_model(
        best_result=best_result,
        experiment_lookup=experiment_lookup,
        subset_lookup=subset_lookup,
        train_manifest=train_manifest,
        x_train_full=x_train_full,
        y_train_full=y_train_full,
        x_val=x_val,
        y_val=y_val,
    )

    payload = build_results_payload(
        manifest=manifest,
        subset_counts=subset_counts,
        example_plot_path=example_plot_path,
        distribution_plot_path=distribution_plot_path,
        summary_csv_path=summary_csv_path,
        accuracy_plot_path=accuracy_plot_path,
        experiment_results=experiment_results,
        best_result=best_result,
        best_model_path=best_model_path,
    )
    with RESULTS_JSON.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, ensure_ascii=False)

    print(f"\nTask 2 artifacts saved to: {ARTIFACTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
