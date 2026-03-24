from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_ROOT / "Task1_defense.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.12"}

    cells = [
        md(
            """# Task 1 Defense Notebook

Papildomas notebookas CNN architekturiai testuoti ir rezultatams perziureti vienoje vietoje.
"""
        ),
        code(
            """import json
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display
from tensorflow import keras
from tensorflow.keras import layers

from task1_pipeline import (
    CLASS_NAMES,
    IMAGE_SHAPE,
    NUM_CLASSES,
    REPORTS_DIR,
    TABLES_DIR,
    build_quick_train_subset,
    build_split_arrays,
    capture_model_summary,
    compute_class_weights,
    configure_runtime,
    create_split_dataframe,
    ensure_directories,
    load_or_create_image_cache,
    load_variant_dataframe,
    sanitize_run_name,
    train_and_evaluate_model,
)

PROJECT_ROOT = Path.cwd()
RESULTS_JSON = PROJECT_ROOT / "artifacts" / "task1" / "results.json"

pd.options.display.float_format = lambda value: f"{value:.4f}"
"""
        ),
        md(
            """## 1. Konfiguracija"""
        ),
        code(
            """RUN_NAME = "gynyba_arch_v1"
TRAIN_LIMIT = 0
EPOCHS = 3
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
SAVE_MODEL = False
"""
        ),
        md(
            """## 2. Nuoroda i ankstesnius pilnus rezultatus

Si lentele padeda greitai prisiminti, kaip atrode pilnai apmokytu modeliu rezultatai.
"""
        ),
        code(
            """if RESULTS_JSON.exists():
    full_results_df = pd.read_csv(TABLES_DIR / "model_comparison.csv").sort_values(
        "test_balanced_accuracy", ascending=False
    )
    display(Markdown("### Pilno eksperimento rezultatai"))
    display(full_results_df[[
        "display_name",
        "epochs_ran",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
    ]])
else:
    display(Markdown("`artifacts/task1/results.json` nerastas. Palyginimo lentele nebus parodyta."))
"""
        ),
        md(
            """## 3. Kandidatine architektura"""
        ),
        code(
            """def build_candidate_model() -> keras.Model:
    inputs = keras.Input(shape=IMAGE_SHAPE, name="input_layer")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="valid")(inputs)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="valid")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="candidate_model")
"""
        ),
        code(
            """preview_model = build_candidate_model()
print(capture_model_summary(preview_model))
keras.backend.clear_session()
"""
        ),
        md(
            """## 4. Duomenu ikelimas

Notebookas naudoja ta pati cache mechanizma kaip pagrindinis pipeline, todel pakartotiniai paleidimai buna greitesni.
"""
        ),
        code(
            """ensure_directories()
configure_runtime()

variant_df = load_variant_dataframe()
split_frames = create_split_dataframe(variant_df)
images, filename_index = load_or_create_image_cache(variant_df)
arrays = build_split_arrays(split_frames, images, filename_index)

subset_df, x_train, y_train = build_quick_train_subset(
    train_df=split_frames["train"],
    images=images,
    filename_index=filename_index,
    train_limit=TRAIN_LIMIT,
)
x_val, y_val = arrays["validation"]
x_test, y_test = arrays["test"]
class_weight = compute_class_weights(y_train)

split_summary = pd.DataFrame(
    [
        {"split": "train_used", "samples": len(x_train)},
        {"split": "validation", "samples": len(x_val)},
        {"split": "test", "samples": len(x_test)},
    ]
)
display(Markdown("### Naudojamu duomenu dydziai"))
display(split_summary)

class_summary = (
    subset_df.groupby(["target", "target_name"])
    .size()
    .reset_index(name="count")
    .sort_values("target")
    .reset_index(drop=True)
)
display(Markdown("### Train dalies klasiu pasiskirstymas"))
display(class_summary)
"""
        ),
        md(
            """## 5. Architekturos apmokymas
"""
        ),
        code(
            """run_suffix = sanitize_run_name(RUN_NAME)
artifact_key = f"notebook_{run_suffix}"

result = train_and_evaluate_model(
    model_key=artifact_key,
    display_name=f"Notebook candidate ({RUN_NAME})",
    builder=build_candidate_model,
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    x_test=x_test,
    y_test=y_test,
    class_weight=class_weight,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    save_model=SAVE_MODEL,
)

summary_path = REPORTS_DIR / f"{artifact_key}_result.json"
with summary_path.open("w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, ensure_ascii=False)

display(Markdown(f"Rezultato santrauka issaugota i `{summary_path}`"))
"""
        ),
        md(
            """## 6. Rezultatai"""
        ),
        code(
            """metrics_df = pd.DataFrame(
    [
        {
            "display_name": result["display_name"],
            "train_samples": result["train_samples"],
            "epochs_ran": result["epochs_ran"],
            "best_epoch": result["best_epoch"],
            "best_val_accuracy": result["best_val_accuracy"],
            "test_accuracy": result["test_accuracy"],
            "test_balanced_accuracy": result["test_balanced_accuracy"],
            "test_macro_f1": result["test_macro_f1"],
            "training_seconds": result["training_seconds"],
        }
    ]
)
display(Markdown("### Naujos architekturos rezultatas"))
display(metrics_df)

if RESULTS_JSON.exists():
    baseline_df = pd.read_csv(TABLES_DIR / "model_comparison.csv")[[
        "display_name",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
    ]]
    candidate_df = pd.DataFrame(
        [
            {
                "display_name": f"{result['display_name']} (quick run)",
                "test_accuracy": result["test_accuracy"],
                "test_balanced_accuracy": result["test_balanced_accuracy"],
                "test_macro_f1": result["test_macro_f1"],
            }
        ]
    )
    comparison_df = pd.concat([baseline_df, candidate_df], ignore_index=True).sort_values(
        "test_balanced_accuracy", ascending=False
    )
    display(Markdown("### Palyginimas su anksciau apmokytais modeliais"))
    display(comparison_df)
"""
        ),
        code(
            """display(Markdown("### Mokymo istorija"))
display(Image(filename=str(PROJECT_ROOT / Path(result["history_plot"]))))

display(Markdown("### Confusion matrix"))
display(Image(filename=str(PROJECT_ROOT / Path(result["confusion_matrix_plot"]))))
"""
        ),
        code(
            """report = json.loads((PROJECT_ROOT / Path(result["classification_report_json"])).read_text(encoding="utf-8"))
report_rows = list(CLASS_NAMES.values()) + ["accuracy", "macro avg", "weighted avg"]
report_df = pd.DataFrame(report).T.loc[report_rows]

display(Markdown("### Classification report"))
display(report_df)
"""
        ),
    ]

    notebook["cells"] = cells
    nbf.write(notebook, NOTEBOOK_PATH)


if __name__ == "__main__":
    build_notebook()
