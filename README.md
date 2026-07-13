# Blast Structural Damage Predictor

🔗 **Live app:** (https://drdoblastpredictionproject-sc5mgpbadsgvgxp8zs7k6w.streamlit.app/)

A machine learning project predicting structural damage severity from blast
loading parameters...


# Blast Structural Damage Predictor

A machine learning project predicting structural damage severity from blast
loading parameters — a companion project to **HazardScope**, extending it
from raw overpressure calculation into a learned damage-classification
pipeline.

Given a charge weight, explosive type, charge shape, standoff distance, and
structure type, this project predicts the likely damage category (none /
minor / moderate / severe / collapse) and visualizes the scenario on a
classic Pressure-Impulse (P-I) diagram.

**v2 update:** added explosive type (TNT / RDX / C4 / ANFO, via TNT mass-
equivalence factors) and charge shape (spherical / hemispherical /
cylindrical, via effective-yield multipliers) as new inputs. Model macro-F1
improved from 0.914 (v1) to 0.936 (v2), suggesting these dimensions carry
real predictive signal.

## Why this project exists

There's no large public dataset of labeled real-world blast damage, so this
project builds one synthetically from established blast-engineering physics
(Kingery-Bulmash overpressure/impulse relationships + Pressure-Impulse
damage curves), then trains and compares several ML models on top of it.
Full reasoning and honest limitations are documented in the app's
Methodology tab and in the notebooks.

## Project structure

```
blast-damage-predictor/
├── data/
│   ├── generate_blast_damage_dataset.py   # synthetic data generator
│   └── blast_damage_dataset.csv           # 6000-sample generated dataset
├── notebooks/
│   ├── 01_eda.ipynb                       # exploratory data analysis
│   └── 02_model_training.ipynb            # RF / XGBoost / MLP comparison
├── model/
│   └── train_and_save_model.py            # trains + saves production model
├── app/
│   ├── app.py                             # Streamlit web app
│   └── artifacts/                         # saved model + preprocessing objects
├── requirements.txt
└── README.md
```

## Quickstart

```bash
git clone <your-repo-url>
cd blast-damage-predictor
pip install -r requirements.txt

# Run the app (uses the already-trained model in app/artifacts/)
cd app
streamlit run app.py
```

To regenerate everything from scratch:

```bash
cd data && python3 generate_blast_damage_dataset.py && cd ..
cd model && python3 train_and_save_model.py && cd ..
```

## Pipeline summary

1. **Data generation** — synthetic (charge weight, explosive type, charge
   shape, standoff distance) combinations sampled and converted to a TNT-
   equivalent effective weight, then to scaled distance via cube-root
   scaling, then to peak overpressure/impulse via a Kingery-Bulmash style
   empirical fit. Damage labels assigned via nested P-I curves per
   structure type.
2. **EDA** — class balance, feature distributions, and a physics sanity
   check confirming damage severity tracks scaled distance as expected.
   (Notebooks in `notebooks/` were built against the v1 dataset schema —
   re-run them against the v2 CSV if you want EDA/model-comparison plots
   reflecting the new explosive type / charge shape columns.)
3. **Model training** — Random Forest, XGBoost, and an MLP compared on a
   stratified 80/20 split; XGBoost selected for production.
4. **Tuning experiment (v1)** — SMOTE oversampling + Optuna hyperparameter
   search was tried on the v1 feature set and evaluated honestly against
   the untuned baseline — the simpler baseline actually won on held-out
   data, a useful negative result documented in the app's history.
5. **Deployment** — Streamlit app with a live predictor and an interactive
   P-I diagram, plus a Methodology tab explaining the full pipeline
   including the v2 explosive-type/charge-shape additions.

## Model performance (production model, v2)

| Metric | Score |
|---|---|
| Test Accuracy | 96.9% |
| Test Macro-F1 | 0.936 |

Per-class performance is strongest on `none` and `severe` (best represented
classes) and improved across the board versus v1 after adding explosive
type and charge shape as features.

## Honest limitations

- The entire dataset is synthetic; P-I curve constants are representative
  approximations, not exact published reference values.
- The overpressure/impulse fit is a simplified polynomial approximation of
  Kingery-Bulmash, not the full tabulated UFC 3-340-02 curves.
- This is a portfolio/research demonstration of an ML pipeline applied to a
  physically-grounded problem — not a validated engineering tool for
  real-world safety decisions.

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo
3. Set the app file path to `app/app.py`
4. Deploy — you'll get a public URL, same idea as HazardScope on Netlify

## Related project

**HazardScope** — the DRDO CFEES internship project this builds on, which
computes blast overpressure directly from empirical formulas.
