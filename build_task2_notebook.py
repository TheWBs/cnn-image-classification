from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_ROOT / "Task2.ipynb"


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

    notebook["cells"] = [
        md(
            """# GDL 1 uzduotis, 2 dalis

Tema: **individualiai suformuoto keliu saltiniu geliu nuotrauku rinkinio klasifikavimas**

Siame darbe realizuotas pilnas sprendimas:

- parengtas **savas** subalansuotas vaizdu rinkinys is **dvieju skirtingu saltiniu**;
- pasirinkta ir pagrysta `MobileNetV2` architektura;
- palyginti 4 eksperimentu rezimai:
  - be `transfer learning`, be augmentacijos;
  - be `transfer learning`, su augmentacija;
  - su `transfer learning`, be augmentacijos;
  - su `transfer learning`, su augmentacija;
- ivertinta **tikslumo priklausomybe nuo duomenu kiekio**;
- pateiktos galutines isvados lietuviu kalba.
"""
        ),
        code(
            """import json
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

PROJECT_ROOT = Path.cwd()
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "task2"
TABLES_DIR = ARTIFACTS_DIR / "tables"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
RESULTS = json.loads((ARTIFACTS_DIR / "results.json").read_text(encoding="utf-8"))

pd.options.display.float_format = lambda value: f"{value:.4f}"
"""
        ),
        md(
            """## Uzdavinio pasirinkimas ir pagrindimas

Pasirinkta 4 klasiu geliu nuotrauku klasifikavimo problema:

- `daisy`
- `dandelion`
- `rose`
- `sunflower`

Sis pasirinkimas tiko uzduociai del keliu priezasciu:

1. Kiekviena klase turi vizualiai atpazistamus bruozus, todel galima suprantamai palyginti modeliu elgsena.
2. Klases buvo surinktos is **dviejų atskiru saltiniu**, todel tenkinamas reikalavimas nenaudoti tik vieno is anksto parengto rinkinio.
3. `MobileNetV2` su `ImageNet` svoriais yra klasikinis, gerai pagrindziamas pasirinkimas `transfer learning` analizei.

Naudoti du saltiniai:

- `tf_flowers`
- `oxford_flowers102`

Abi kolekcijos buvo individualiai apdorotos: atrinktos tik persidengiancios klases, suvienodinti pavadinimai, subalansuotas vienodas paveikslu skaicius kiekvienai `klasei x saltiniui`.
"""
        ),
        code(
            """sources_df = pd.read_csv(TABLES_DIR / "dataset_sources.csv")
counts_df = pd.read_csv(TABLES_DIR / "dataset_counts_by_source_class_split.csv")
subset_counts_df = pd.read_csv(TABLES_DIR / "train_subset_counts.csv")

display(Markdown("### Duomenu saltiniai"))
display(sources_df)

display(Markdown("### Klasiu ir saltiniu pasiskirstymas po skaidymo"))
display(counts_df)

subset_total_df = subset_counts_df.groupby(["fraction", "subset_name"], as_index=False)["count"].sum()
display(Markdown("### Naudotos treniravimo imtys"))
display(subset_total_df)

display(Markdown("### Pavyzdiniai paveikslai"))
display(Image(filename=str(PROJECT_ROOT / RESULTS["dataset"]["example_plot"])))

display(Markdown("### Duomenu rinkinio balansavimas"))
display(Image(filename=str(PROJECT_ROOT / RESULTS["dataset"]["distribution_plot"])))
"""
        ),
        md(
            """## Modelis ir eksperimentu schema

Pasirinkta architektura:

- `MobileNetV2`
- `alpha = 0.35`
- ivestis: `96x96x3`
- `include_top=False`
- `GlobalAveragePooling + Dropout(0.2) + Dense(softmax)`

Pagrindimas:

- architektura lengva ir efektyvi, todel tinka ribotiems resursams;
- turi oficialius `ImageNet` svorius `transfer learning` tyrimui;
- gerai veikia mazesniuose individualiai suformuotuose rinkiniuose.

Eksperimentu schema:

- `transfer learning`: su / be is anksto apmokytu `ImageNet` svoriu;
- `data augmentation`: su / be atsitiktiniu transformaciju;
- treniravimo imties dydis: `25 %`, `50 %`, `100 %` nuo fiksuotos `train` aibes.

Taigi is viso atlikta **12 pilnu eksperimento paleidimu** toje pacioje testavimo aibeje.
"""
        ),
        code(
            """summary_df = pd.read_csv(TABLES_DIR / "experiment_summary.csv")
display(Markdown("### Visu eksperimentu suvestine"))
display(
    summary_df[
        [
            "label",
            "subset_name",
            "train_size",
            "epochs_ran",
            "best_epoch",
            "best_val_accuracy",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
            "training_seconds",
        ]
    ].sort_values(["train_size", "label"])
)

display(Markdown("### Test accuracy priklausomybe nuo treniravimo duomenu kiekio"))
display(Image(filename=str(PROJECT_ROOT / RESULTS["artifacts"]["accuracy_vs_train_size_plot"])))

display(Markdown("### Pilnos imties palyginimas"))
display(Image(filename=str(PROJECT_ROOT / RESULTS["artifacts"]["final_size_comparison_plot"])))
"""
        ),
        code(
            """comparison_df = summary_df.copy()
comparison_df["transfer_learning"] = comparison_df["use_pretrained"].map({True: "Taip", False: "Ne"})
comparison_df["augmentation"] = comparison_df["use_augmentation"].map({True: "Taip", False: "Ne"})

transfer_pivot = comparison_df.pivot_table(
    index="train_size",
    columns=["transfer_learning", "augmentation"],
    values="test_accuracy",
)
display(Markdown("### Tikslumo lentele pagal duomenu kieki, transfer learning ir augmentacija"))
display(transfer_pivot)

transfer_effect_rows = []
for subset_name, group in comparison_df.groupby("subset_name"):
    for augmentation_value in [False, True]:
        without_transfer = group[
            (group["use_pretrained"] == False) & (group["use_augmentation"] == augmentation_value)
        ]["test_accuracy"].iloc[0]
        with_transfer = group[
            (group["use_pretrained"] == True) & (group["use_augmentation"] == augmentation_value)
        ]["test_accuracy"].iloc[0]
        transfer_effect_rows.append(
            {
                "subset_name": subset_name,
                "augmentation": "Taip" if augmentation_value else "Ne",
                "accuracy_be_transfer": without_transfer,
                "accuracy_su_transfer": with_transfer,
                "pagerinimas": with_transfer - without_transfer,
            }
        )

transfer_effect_df = pd.DataFrame(transfer_effect_rows).sort_values(["subset_name", "augmentation"])
display(Markdown("### Transfer learning poveikis"))
display(transfer_effect_df)

augmentation_effect_rows = []
for subset_name, group in comparison_df.groupby("subset_name"):
    for pretrained_value in [False, True]:
        without_aug = group[
            (group["use_pretrained"] == pretrained_value) & (group["use_augmentation"] == False)
        ]["test_accuracy"].iloc[0]
        with_aug = group[
            (group["use_pretrained"] == pretrained_value) & (group["use_augmentation"] == True)
        ]["test_accuracy"].iloc[0]
        augmentation_effect_rows.append(
            {
                "subset_name": subset_name,
                "transfer_learning": "Taip" if pretrained_value else "Ne",
                "accuracy_be_aug": without_aug,
                "accuracy_su_aug": with_aug,
                "pagerinimas": with_aug - without_aug,
            }
        )

augmentation_effect_df = pd.DataFrame(augmentation_effect_rows).sort_values(["subset_name", "transfer_learning"])
display(Markdown("### Data augmentation poveikis"))
display(augmentation_effect_df)
"""
        ),
        code(
            """best = RESULTS["best_experiment"]
best_df = pd.DataFrame(
    [
        {
            "Geriausias eksperimentas": best["label"],
            "Naudota imtis": best["subset_name"],
            "Train size": best["train_size"],
            "Best epoch": best["best_epoch"],
            "Best validation accuracy": best["best_val_accuracy"],
            "Test accuracy": best["test_accuracy"],
            "Balanced accuracy": best["test_balanced_accuracy"],
            "Macro F1": best["test_macro_f1"],
        }
    ]
)
display(Markdown("## Geriausias modelis"))
display(best_df)
display(Image(filename=str(PROJECT_ROOT / best["history_plot"])))
display(Image(filename=str(PROJECT_ROOT / best["confusion_matrix_plot"])))
"""
        ),
        code(
            """summary_df = summary_df.sort_values(["subset_fraction", "use_pretrained", "use_augmentation"])
best = RESULTS["best_experiment"]

full_size_df = summary_df[summary_df["subset_fraction"] == summary_df["subset_fraction"].max()].copy()
best_full = full_size_df.sort_values("test_accuracy", ascending=False).iloc[0]
worst_full = full_size_df.sort_values("test_accuracy", ascending=True).iloc[0]
best_test_overall = summary_df.sort_values("test_accuracy", ascending=False).iloc[0]

transfer_gain_full = (
    full_size_df[(full_size_df["use_pretrained"] == True) & (full_size_df["use_augmentation"] == False)]["test_accuracy"].iloc[0]
    - full_size_df[(full_size_df["use_pretrained"] == False) & (full_size_df["use_augmentation"] == False)]["test_accuracy"].iloc[0]
)
augmentation_gain_pretrained_full = (
    full_size_df[(full_size_df["use_pretrained"] == True) & (full_size_df["use_augmentation"] == True)]["test_accuracy"].iloc[0]
    - full_size_df[(full_size_df["use_pretrained"] == True) & (full_size_df["use_augmentation"] == False)]["test_accuracy"].iloc[0]
)
augmentation_gain_scratch_full = (
    full_size_df[(full_size_df["use_pretrained"] == False) & (full_size_df["use_augmentation"] == True)]["test_accuracy"].iloc[0]
    - full_size_df[(full_size_df["use_pretrained"] == False) & (full_size_df["use_augmentation"] == False)]["test_accuracy"].iloc[0]
)

conclusion_md = f'''
## Galutines isvados

1. **Duomenu rinkinys buvo parengtas individualiai is dvieju saltiniu.** Galutiniam modeliui naudotos tik 4 persidengiancios klases (`daisy`, `dandelion`, `rose`, `sunflower`), o kiekvienai `klasei x saltiniui` paliktas vienodas paveikslu skaicius. Tai leido isvengti dirbtinio klasiu disbalanso.

2. **Pagal validation accuracy atrinktas geriausias modelis buvo:** **{best["label"]}**, naudojant **{best["subset_name"]}** treniravimo imti. Jo `test accuracy` buvo **{best["test_accuracy"]:.4f}**, o `balanced accuracy` buvo **{best["test_balanced_accuracy"]:.4f}**.

3. **Didziausias test accuracy tarp visu eksperimentu** buvo gautas konfiguracijoje **{best_test_overall["label"]}** su **{best_test_overall["subset_name"]}** imtimi: **{best_test_overall["test_accuracy"]:.4f}**.

4. **Transfer learning poveikis buvo aiskiai teigiamas.** Pilnos imties atveju vien `ImageNet` svoriu naudojimas be augmentacijos pakele `test accuracy` per **{transfer_gain_full:.4f}** punkto, lyginant su tuo paciu modeliu be is anksto apmokytu svoriu.

5. **Data augmentation poveikis priklause nuo pradinio modelio.** Pilnos imties atveju su transfer learning augmentacija pakeite `accuracy` per **{augmentation_gain_pretrained_full:.4f}**, o be transfer learning pakeitimas buvo **{augmentation_gain_scratch_full:.4f}**. Tai rodo, kad augmentacija ne visada duoda vienodai dideli laimejima visiems rezimams.

6. **Duomenu kiekis buvo svarbus visais atvejais.** Didejant treniravimo imciai nuo 25 % iki 100 %, modeliu tikslumas nuosekliai gerejo. Tai gerai matyti is `accuracy vs train size` kreives.

7. **Praktiskai naudingiausia konfiguracija** yra ta, kuri sujungia `transfer learning` ir saikinga augmentacija arba bent jau `transfer learning`, nes mazu individualiai suformuotu rinkiniu atveju is anksto apmokyti svoriai suteikia stipria pradzia.

8. **Blogiausiai pasirode:** **{worst_full["label"]}** su pilna imtimi, kurio `test accuracy` buvo **{worst_full["test_accuracy"]:.4f}**. Tuo tarpu geriausias pilnos imties variantas buvo **{best_full["label"]}** su **{best_full["test_accuracy"]:.4f}** `accuracy`.
'''

display(Markdown(conclusion_md))
"""
        ),
    ]

    with NOTEBOOK_PATH.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)


if __name__ == "__main__":
    build_notebook()
