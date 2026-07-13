"""
Train the final model and save all artifacts needed for the Streamlit app:
  - trained XGBoost model
  - StandardScaler (kept for consistency even though tree models don't need it,
    in case you swap in the MLP later)
  - LabelEncoder for damage categories
  - list of feature columns in the exact order the model expects
  - the raw P-I curve constants (so the app can draw the diagram)
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
import xgboost as xgb

RANDOM_STATE = 42
damage_order = ["none", "minor", "moderate", "severe", "collapse"]

STRUCTURE_PI_CURVES = {
    "glazing_window": {
        "minor":    (2,  3,  8),
        "moderate": (4,  6,  20),
        "severe":   (7,  10, 45),
        "collapse": (12, 18, 90),
    },
    "masonry_wall": {
        "minor":    (10, 15, 120),
        "moderate": (18, 25, 260),
        "severe":   (30, 40, 550),
        "collapse": (48, 60, 1000),
    },
    "rc_wall": {
        "minor":    (25, 35,  400),
        "moderate": (45, 55,  900),
        "severe":   (70, 85,  1700),
        "collapse": (100, 120, 3000),
    },
    "steel_frame": {
        "minor":    (20, 20, 300),
        "moderate": (35, 40, 700),
        "severe":   (55, 65, 1400),
        "collapse": (85, 100, 2600),
    },
}

import os
os.makedirs("../app/artifacts", exist_ok=True)

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

class_counts = pd.Series(y_train).value_counts()
class_weights = {cls: len(y_train) / (len(class_counts) * count)
                 for cls, count in class_counts.items()}
sample_weights = np.array([class_weights[label] for label in y_train])

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=len(damage_order),
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X_train, y_train, sample_weight=sample_weights)

preds = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, preds))
print("Test macro-F1:", f1_score(y_test, preds, average="macro"))
print(classification_report(y_test, preds, target_names=damage_order))

# Fit a scaler on the full feature set too (kept for parity / future MLP swap-in)
scaler = StandardScaler()
scaler.fit(X)

# --- Save all artifacts ---
joblib.dump(model, "../app/artifacts/xgb_model.joblib")
joblib.dump(scaler, "../app/artifacts/scaler.joblib")
joblib.dump(label_encoder, "../app/artifacts/label_encoder.joblib")

with open("../app/artifacts/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

with open("../app/artifacts/structure_types.json", "w") as f:
    json.dump(structure_types, f)

with open("../app/artifacts/pi_curves.json", "w") as f:
    json.dump(STRUCTURE_PI_CURVES, f)

with open("../app/artifacts/damage_order.json", "w") as f:
    json.dump(damage_order, f)

print("\nAll artifacts saved to ../app/artifacts/")
