"""
Final model tuning: SMOTE (minority oversampling) + Optuna hyperparameter
search for XGBoost, compared against the plain class-weighted baseline from
train_and_save_model.py. Whichever wins on held-out macro-F1 gets saved as
the production artifact.
"""

import json
import numpy as np
import pandas as pd
import joblib
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
damage_order = ["none", "minor", "moderate", "severe", "collapse"]

df = pd.read_csv("../data/blast_damage_dataset.csv")
df_model = df.copy()
for col in ["charge_weight_kg", "standoff_distance_m", "peak_overpressure_kpa", "impulse_kpa_ms"]:
    df_model[f"log_{col}"] = np.log1p(df_model[col])

structure_types = sorted(df_model["structure_type"].unique().tolist())
df_model = pd.get_dummies(df_model, columns=["structure_type"], prefix="struct")

feature_cols = [
    "log_charge_weight_kg", "log_standoff_distance_m", "scaled_distance_Z",
    "log_peak_overpressure_kpa", "log_impulse_kpa_ms", "quality_factor",
] + [f"struct_{s}" for s in structure_types]

X = df_model[feature_cols]
label_encoder = LabelEncoder()
label_encoder.fit(damage_order)
y = label_encoder.transform(df_model["damage_category"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# Baseline (class-weighted, no SMOTE) — for reference
# ---------------------------------------------------------------------------
class_counts = pd.Series(y_train).value_counts()
class_weights = {cls: len(y_train) / (len(class_counts) * count)
                  for cls, count in class_counts.items()}
sample_weights_baseline = np.array([class_weights[label] for label in y_train])

baseline_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    objective="multi:softmax", num_class=len(damage_order),
    eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1,
)
baseline_model.fit(X_train, y_train, sample_weight=sample_weights_baseline)
baseline_preds = baseline_model.predict(X_test)
baseline_f1 = f1_score(y_test, baseline_preds, average="macro")
print(f"[Baseline] accuracy={accuracy_score(y_test, baseline_preds):.4f}  macro-F1={baseline_f1:.4f}")

# ---------------------------------------------------------------------------
# SMOTE-resampled training set
# ---------------------------------------------------------------------------
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Resampled class distribution:", pd.Series(y_train_res).value_counts().to_dict())

# ---------------------------------------------------------------------------
# Optuna hyperparameter search on the SMOTE-resampled data
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "objective": "multi:softmax",
        "num_class": len(damage_order),
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    clf = xgb.XGBClassifier(**params)
    scores = cross_val_score(clf, X_train_res, y_train_res, cv=cv, scoring="f1_macro", n_jobs=1)
    return scores.mean()


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=15, show_progress_bar=False)

print("\nBest CV macro-F1:", study.best_value)
print("Best params:", study.best_params)

best_params = study.best_params
best_params.update({
    "objective": "multi:softmax",
    "num_class": len(damage_order),
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
})

tuned_model = xgb.XGBClassifier(**best_params)
tuned_model.fit(X_train_res, y_train_res)
tuned_preds = tuned_model.predict(X_test)
tuned_f1 = f1_score(y_test, tuned_preds, average="macro")
tuned_acc = accuracy_score(y_test, tuned_preds)

print(f"\n[Tuned + SMOTE] accuracy={tuned_acc:.4f}  macro-F1={tuned_f1:.4f}")
print(classification_report(y_test, tuned_preds, target_names=damage_order))

# ---------------------------------------------------------------------------
# Pick the winner and save as the production model
# ---------------------------------------------------------------------------
if tuned_f1 > baseline_f1:
    print(f"\n>>> Tuned+SMOTE model wins ({tuned_f1:.4f} > {baseline_f1:.4f}). Saving as production model.")
    final_model = tuned_model
    final_note = {
        "strategy": "SMOTE + Optuna-tuned XGBoost",
        "test_accuracy": tuned_acc,
        "test_macro_f1": tuned_f1,
        "best_params": study.best_params,
    }
else:
    print(f"\n>>> Baseline class-weighted model wins ({baseline_f1:.4f} >= {tuned_f1:.4f}). Keeping it as production model.")
    final_model = baseline_model
    final_note = {
        "strategy": "class-weighted XGBoost (no SMOTE)",
        "test_accuracy": accuracy_score(y_test, baseline_preds),
        "test_macro_f1": baseline_f1,
        "best_params": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
    }

scaler = StandardScaler()
scaler.fit(X)

joblib.dump(final_model, "../app/artifacts/xgb_model.joblib")
joblib.dump(scaler, "../app/artifacts/scaler.joblib")
joblib.dump(label_encoder, "../app/artifacts/label_encoder.joblib")

with open("../app/artifacts/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)
with open("../app/artifacts/structure_types.json", "w") as f:
    json.dump(structure_types, f)
with open("../app/artifacts/damage_order.json", "w") as f:
    json.dump(damage_order, f)
with open("../app/artifacts/model_metadata.json", "w") as f:
    json.dump(final_note, f, indent=2)

print("\nSaved final production model + metadata to ../app/artifacts/")
