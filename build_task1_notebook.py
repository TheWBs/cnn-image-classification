from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_ROOT / "Task1.ipynb"


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
            """# GDL 1 uzduotis, 1 dalis

Variantas: **5**  
Pateiktos architekturos: **Variantas 2**, **Variantas 7**, **Variantas 8**

Notebooke pateikiamas:

- duomenu rinkinio paruosimas;
- klasiu disbalanso ivertinimas;
- triju nurodytu CNN architekturu apmokymas ir palyginimas;
- mano pasiulyta architektura;
- mazesnes duomenu imties tyrimas su `Stratified K-Fold`;
- galutines isvados lietuviu kalba.
"""
        ),
        code(
            """import json
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

PROJECT_ROOT = Path.cwd()
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "task1"
TABLES_DIR = ARTIFACTS_DIR / "tables"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
RESULTS = json.loads((ARTIFACTS_DIR / "results.json").read_text(encoding="utf-8"))

pd.options.display.float_format = lambda value: f"{value:.4f}"
"""
        ),
        md(
            """## Uzdavinio nustatymas

Is pradinio duomenu rinkinio paliktos tik tavo variantui priklausancios klases:

| Nauja klase | Pavadinimas | Originalios klases |
| --- | --- | --- |
| 0 | Outerwear | 2, 3, 4 |
| 1 | Shirts | 0, 6 |
| 2 | Pants | 1 |
| 3 | Low-top shoes | 5, 7 |
| 4 | Accessories | 8 |

Klasiu is originalaus rinkinio, kuriu lenteleje nera, buvo atsisakyta.

Duomenu skaidymas atliktas **stratified** budu, kad kiekviename skaidyme isliktu panasi klasiu proporcija:

- `train`: 70 %
- `validation`: 15 %
- `test`: 15 %

Kadangi klase `Outerwear` apjungia tris originalias klases, o `Pants` ir `Accessories` tik po viena, vertinimui papildomai naudotas:

- `class_weight` apmokymo metu;
- `balanced accuracy`;
- `macro F1`;
- per-klasinis tikslumas.
"""
        ),
        code(
            """overall_counts = pd.read_csv(TABLES_DIR / "overall_class_counts.csv")
split_counts = pd.read_csv(TABLES_DIR / "split_class_counts.csv")

display(Markdown("### Bendras klasiu pasiskirstymas"))
display(overall_counts)

display(Markdown("### Klasiu pasiskirstymas po train / validation / test skaidymo"))
display(split_counts)
"""
        ),
        code(
            """display(Markdown("### Pavyzdiniai paveikslai"))
display(Image(filename=str(PROJECT_ROOT / RESULTS["dataset"]["example_images_plot"])))

display(Markdown("### Klasiu pasiskirstymo vizualizacija"))
display(Image(filename=str(PROJECT_ROOT / RESULTS["dataset"]["class_distribution_plot"])))
"""
        ),
        md(
            """## Konvoliuciniu operaciju interpretacija

- **Convolution**: 3x3 branduoliai slenka per 28x28 pilko tono vaizda ir isgauna lokalinius bruozus, pavyzdziui, krastus, konturus ir teksturos fragmentus.
- **Stride**: pateiktose architekturose naudotas numatytasis `stride = 1`, todel po kiekvieno `Conv2D` bruozu zemelapis mazeja tik tiek, kiek leidzia branduolio dydis.
- **Padding**: pateiktuose variantuose naudotas `valid` tipo padding, todel vaizdo matmenys po konvoliuciju mazeja (`28 -> 26 -> 24` ir t. t.). Mano architekturoje naudotas `same`, kad mazu 28x28 paveikslu informacija butu prarandama letesniu tempu.
- **Pooling**: `MaxPooling2D` issaugo ryskiausius signalus, o `AveragePooling2D` labiau glotnina reprezentacija. Tai svarbu, nes uzduotyje lyginami skirtingi pooling tipai.
"""
        ),
        md(
            """## Modeliu konfiguracija

Pateiktuose paveiksluose ne visos hiperparametru reiksmes buvo uzrasytos, todel priimtos sios darbinei realizacijai reikalingos prielaidos:

- visoms `Conv2D` ir pasleptoms `Dense` sluoksniu aktyvacijoms naudota `ReLU`;
- isejimo sluoksnyje naudota `Softmax`, nes yra 5 klases;
- `Variantas 7` `Dropout` sluoksniui pasirinkta `0.3`;
- optimizatorius: `Adam(learning_rate=1e-3)`;
- geriausia epocha parenkama pagal maziausia `validation loss`, taikant `EarlyStopping`.
"""
        ),
        code(
            """comparison_df = pd.read_csv(TABLES_DIR / "model_comparison.csv").sort_values(
    "test_balanced_accuracy", ascending=False
)
per_class_df = pd.read_csv(TABLES_DIR / "per_class_accuracy.csv")

display(Markdown("### Bendras architekturu palyginimas"))
display(comparison_df)

display(Markdown("### Balanced accuracy palyginimas"))
display(Image(filename=str(PROJECT_ROOT / RESULTS["artifacts"]["model_comparison_plot"])))

display(Markdown("### Per-klasinio tikslumo silumos zemelapis"))
display(Image(filename=str(PROJECT_ROOT / RESULTS["artifacts"]["per_class_accuracy_heatmap"])))
"""
        ),
        code(
            """for model in RESULTS["main_models"]:
    metrics_df = pd.DataFrame([
        {
            "display_name": model["display_name"],
            "params": model["params"],
            "epochs_ran": model["epochs_ran"],
            "best_epoch": model["best_epoch"],
            "best_val_loss": model["best_val_loss"],
            "best_val_accuracy": model["best_val_accuracy"],
            "test_accuracy": model["test_accuracy"],
            "test_balanced_accuracy": model["test_balanced_accuracy"],
            "test_macro_f1": model["test_macro_f1"],
            "test_weighted_f1": model["test_weighted_f1"],
        }
    ])
    display(Markdown(f"## {model['display_name']}"))
    display(metrics_df)
    display(Image(filename=str(PROJECT_ROOT / model["history_plot"])))
    display(Image(filename=str(PROJECT_ROOT / model["confusion_matrix_plot"])))
"""
        ),
        md(
            """## Mano architektura ir mazesnes imties tyrimas

Papildomai pasiulyta mano CNN architektura:

- du `Conv-BN-ReLU` blokai su `same` padding ir `MaxPooling`;
- antras analogiskas blokas su daugiau filtru;
- `Conv(128) + GlobalAveragePooling`;
- `Dense(128) + Dropout(0.3)`;
- `Softmax` klasifikatorius.

Ji pasirinkta tam, kad:

- mazi 28x28 vaizdai neprarastu informacijos per greitai;
- `BatchNormalization` stabilizuotu mokyma;
- `GlobalAveragePooling` sumazintu parametru skaiciu ir overfitting rizika;
- `Dropout` apsaugotu nuo per didelio prisitaikymo prie treniravimo aibes.

Mazesnes imties analizei naudoti **pirmi N paveikslu pagal failo pavadinima abeceles tvarka** is `train` aibes ir `Stratified K-Fold (k=3)`.
"""
        ),
        code(
            """sample_size_summary = pd.read_csv(TABLES_DIR / "sample_size_cv_summary.csv")
display(Markdown("### 3-fold CV suvestine pagal imties dydi"))
display(sample_size_summary)
display(Image(filename=str(PROJECT_ROOT / RESULTS["sample_size_study"]["cv_plot"])))

reduced = RESULTS["sample_size_study"]["reduced_model_result"]
full_custom = next(model for model in RESULTS["main_models"] if model["model_key"] == "custom_model")
full_train_size = sum(RESULTS["dataset"]["split_counts"]["train"].values())
comparison = pd.DataFrame(
    [
        {
            "variantas": "Pilna mano architektura",
            "train_size": full_train_size,
            "test_accuracy": full_custom["test_accuracy"],
            "test_balanced_accuracy": full_custom["test_balanced_accuracy"],
            "test_macro_f1": full_custom["test_macro_f1"],
        },
        {
            "variantas": f"Mano architektura, pirmi {RESULTS['sample_size_study']['acceptable_size']}",
            "train_size": RESULTS["sample_size_study"]["acceptable_size"],
            "test_accuracy": reduced["test_accuracy"],
            "test_balanced_accuracy": reduced["test_balanced_accuracy"],
            "test_macro_f1": reduced["test_macro_f1"],
        },
    ]
)
display(Markdown("### Pilnos ir sumazintos imties palyginimas"))
display(comparison)
display(Image(filename=str(PROJECT_ROOT / reduced["history_plot"])))
display(Image(filename=str(PROJECT_ROOT / reduced["confusion_matrix_plot"])))
"""
        ),
        code(
            """best_presented = RESULTS["conclusions"]["best_presented_model"]
best_presented_score = RESULTS["conclusions"]["best_presented_balanced_accuracy"]
custom_score = RESULTS["conclusions"]["custom_model_balanced_accuracy"]
reduced_score = RESULTS["conclusions"]["reduced_custom_balanced_accuracy"]
acceptable_size = RESULTS["conclusions"]["acceptable_sample_size"]
train_size_full = sum(RESULTS["dataset"]["split_counts"]["train"].values())
if custom_score > best_presented_score:
    custom_statement = "Mano architektura buvo geresne uz visas pateiktas."
else:
    custom_statement = "Mano architektura buvo konkurencinga pateiktoms architekturoms, bet nevirsijo geriausio pateikto varianto."

final_md = f'''
## Galutines isvados

1. **Duomenu paruosimas.** Pagal Varianta 5 is pradinio rinkinio paliktos 5 sugrupuotos klases, o skaidymas atliktas stratified principu i `train`, `validation` ir `test` aibes. Tai leido isvengti pernelyg stipriu proporciju svyravimu tarp skaidymu.

2. **Klasiu disbalansas yra svarbus.** `Outerwear` klase sudaryta is triju originaliu klasiu, todel jos pavyzdziu daugiausia. Vien bendro `accuracy` nepakanka, todel modeliai lyginti ir pagal `balanced accuracy`, `macro F1` bei per-klasini tiksluma.

3. **Is pateiktu architekturu geriausia pasirode {best_presented}.** Jos `balanced accuracy` testavimo aibeje sieke **{best_presented_score:.4f}**. Tai rodo, kad sio uzdavinio sprendimui svarbu ne tik sluoksniu skaicius, bet ir tai, kaip greitai mazina vaizdo matmenis pooling sluoksniai bei kiek parametru turi klasifikatoriaus galva.

4. **{custom_statement}** Pilnos imties atveju ji pasieke **{custom_score:.4f}** `balanced accuracy` naudodama **{train_size_full}** treniravimo paveikslu. Privalumas atsirado del `same padding`, `BatchNormalization`, saikingo `Dropout` ir `GlobalAveragePooling`, kurie padejo geriau islaikyti informacija mazuose 28x28 vaizduose.

5. **Priimtina mazesne treniravimo imtis.** Pagal 3-fold CV kriteriju maziausia priimtina imtimi pasirinkta **pirmi {acceptable_size} paveikslu pagal pavadinima**. Tokia imtis dar islaike auksta kokybe, o sumazintos imties mano modelis testavimo aibeje pasieke **{reduced_score:.4f}** `balanced accuracy`.

6. **Praktine rekomendacija.** Jei svarbiausia yra kokybe, verta naudoti mano architektura su pilna imtimi. Jei svarbi mokymo trukme ir pakanka labai artimo rezultato, galima naudoti pirmus {acceptable_size} treniravimo paveikslus pagal pavadinima ir ta pacia architektura.
'''

display(Markdown(final_md))
"""
        ),
    ]

    notebook["cells"] = cells
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)


if __name__ == "__main__":
    build_notebook()
