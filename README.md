# Blast Structural Damage Predictor

A machine learning project predicting structural damage severity from blast
loading parameters — a companion project to **HazardScope**, extending it
from raw overpressure calculation into a learned damage-classification
pipeline.

Given a charge weight, standoff distance, and structure type, this project
predicts the likely damage category (none / minor / moderate / severe /
collapse) and visualizes the scenario on a classic Pressure-Impulse (P-I)
diagram.

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
│   ├── train_and_save_model.py            # trains + saves production model
│   └── tune_and_finalize_model.py         # SMOTE + Optuna tuning experiment
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
# optional: cd model && python3 tune_and_finalize_model.py && cd ..
```

## Pipeline summary

1. **Data generation** — synthetic (charge weight, standoff distance) pairs
   sampled log-uniformly, converted to scaled distance via cube-root scaling,
   then to peak overpressure/impulse via a Kingery-Bulmash style empirical
   fit. Damage labels assigned via nested P-I curves per structure type.
2. **EDA** — class balance, feature distributions, and a physics sanity
   check confirming damage severity tracks scaled distance as expected.
3. **Model training** — Random Forest, XGBoost, and an MLP compared on a
   stratified 80/20 split; XGBoost selected for production.
4. **Tuning experiment** — SMOTE oversampling + Optuna hyperparameter search,
   evaluated honestly against the untuned baseline (see Methodology tab in
   the app for the result — the baseline actually won on held-out data).
5. **Deployment** — Streamlit app with a live predictor and an interactive
   P-I diagram, plus a Methodology tab explaining the full pipeline.

## Model performance (production model)

| Metric | Score |
|---|---|
| Test Accuracy | 96% |
| Test Macro-F1 | 0.914 |

Per-class performance is strongest on `none` and `severe` (best represented
classes) and weaker on `minor`/`moderate`/`collapse`, which is expected
given the class imbalance discussed in the EDA notebook.

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
