"""
Blast Structural Damage Predictor — Streamlit App (v2)
=========================================================
Adds explosive type (TNT/RDX/C4/ANFO) and charge shape (spherical/
hemispherical/cylindrical) on top of the v1 predictor.

Run locally:
    streamlit run app.py
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Blast Damage Predictor", page_icon=None, layout="wide")

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "xgb_model.joblib"))
    label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "structure_types.json")) as f:
        structure_types = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "explosive_types.json")) as f:
        explosive_types = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "charge_shapes.json")) as f:
        charge_shapes = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "pi_curves.json")) as f:
        pi_curves = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "explosive_tnt_equivalence.json")) as f:
        explosive_tnt_equivalence = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "charge_shape_factor.json")) as f:
        charge_shape_factor = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "damage_order.json")) as f:
        damage_order = json.load(f)
    try:
        with open(os.path.join(ARTIFACTS_DIR, "model_metadata.json")) as f:
            model_metadata = json.load(f)
    except FileNotFoundError:
        model_metadata = {"strategy": "class-weighted XGBoost", "test_accuracy": None, "test_macro_f1": None}
    return (model, label_encoder, feature_cols, structure_types, explosive_types,
            charge_shapes, pi_curves, explosive_tnt_equivalence, charge_shape_factor,
            damage_order, model_metadata)


(model, label_encoder, feature_cols, structure_types, explosive_types, charge_shapes,
 pi_curves, explosive_tnt_equivalence, charge_shape_factor, damage_order,
 model_metadata) = load_artifacts()

DAMAGE_COLORS = {
    "none": "#2ecc71", "minor": "#f1c40f", "moderate": "#e67e22",
    "severe": "#e74c3c", "collapse": "#7d3c98",
}

STRUCTURE_LABELS = {
    "glazing_window": "Glazing / Window", "masonry_wall": "Masonry Wall",
    "rc_wall": "Reinforced Concrete Wall", "steel_frame": "Steel Frame",
}

SHAPE_LABELS = {
    "spherical": "Spherical (free-air burst)",
    "hemispherical": "Hemispherical (surface burst)",
    "cylindrical": "Cylindrical",
}


def scaled_distance(R_m, W_eff_kg):
    return R_m / (W_eff_kg ** (1 / 3))


def peak_overpressure_kpa(Z):
    Z = np.clip(Z, 0.2, 40)
    logZ = np.log(Z)
    log_P = 7.2 - 1.85 * logZ - 0.10 * logZ ** 2 + 0.06 * logZ ** 3
    return np.exp(log_P)


def impulse_kpa_ms(Z, W_eff_kg):
    Z = np.clip(Z, 0.2, 40)
    logZ = np.log(Z)
    log_Ibar = 2.65 - 0.95 * logZ + 0.02 * logZ ** 2
    I_bar = np.exp(log_Ibar)
    return I_bar * (W_eff_kg ** (1 / 3))


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------

st.sidebar.title("Scenario Inputs")

charge_weight = st.sidebar.slider(
    "Actual charge weight (kg)", min_value=0.5, max_value=2000.0, value=50.0, step=0.5,
)
explosive_type = st.sidebar.selectbox(
    "Explosive type", options=explosive_types,
    help="Converts actual mass into a TNT-equivalent mass using standard "
         "mass-based equivalence factors before computing blast effects.",
)
charge_shape = st.sidebar.selectbox(
    "Charge shape / burst type", options=charge_shapes,
    format_func=lambda s: SHAPE_LABELS.get(s, s),
    help="Hemispherical (surface) bursts roughly double effective yield due "
         "to ground reflection, a standard simplification in blast engineering.",
)
standoff_distance = st.sidebar.slider(
    "Standoff distance (m)", min_value=1.0, max_value=200.0, value=20.0, step=0.5,
)
structure_type = st.sidebar.selectbox(
    "Structure type", options=structure_types,
    format_func=lambda s: STRUCTURE_LABELS.get(s, s),
)
quality_factor = st.sidebar.slider(
    "Construction quality factor", min_value=0.7, max_value=1.3, value=1.0, step=0.01,
    help="1.0 = typical construction. <1.0 = weaker than typical. "
         ">1.0 = stronger than typical.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Synthetic model combining Kingery-Bulmash style overpressure/impulse "
    "estimates, TNT-equivalence conversion, and Pressure-Impulse damage "
    "curves. For research and portfolio purposes — not a substitute for "
    "formal structural or explosives analysis."
)

st.title("Blast Structural Damage Predictor")
st.markdown(
    "Companion tool to **HazardScope** — predicts likely structural damage "
    "category from blast parameters, explosive type, and charge geometry "
    "using a trained XGBoost classifier."
)


def render_predictor_tab():
    tnt_factor = explosive_tnt_equivalence[explosive_type]
    shape_factor = charge_shape_factor[charge_shape]
    W_eff = charge_weight * tnt_factor * shape_factor

    Z = scaled_distance(standoff_distance, W_eff)
    P = peak_overpressure_kpa(Z) * quality_factor ** -0.5
    I = impulse_kpa_ms(Z, W_eff) * quality_factor ** -0.5

    row = {
        "log_charge_weight_kg": np.log1p(charge_weight),
        "log_standoff_distance_m": np.log1p(standoff_distance),
        "scaled_distance_Z": Z,
        "log_peak_overpressure_kpa": np.log1p(P),
        "log_impulse_kpa_ms": np.log1p(I),
        "quality_factor": quality_factor,
    }
    for s in structure_types:
        row[f"struct_{s}"] = 1 if s == structure_type else 0
    for e in explosive_types:
        row[f"explosive_{e}"] = 1 if e == explosive_type else 0
    for s in charge_shapes:
        row[f"shape_{s}"] = 1 if s == charge_shape else 0

    X_input = pd.DataFrame([row])[feature_cols]

    pred_encoded = model.predict(X_input)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    pred_proba = model.predict_proba(X_input)[0]

    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.subheader("Predicted Damage")
        color = DAMAGE_COLORS.get(pred_label, "#333333")
        st.markdown(
            f"<div style='padding:24px; border-radius:12px; background-color:{color}20; "
            f"border:2px solid {color}; text-align:center;'>"
            f"<span style='font-size:14px; color:#666;'>DAMAGE CATEGORY</span><br>"
            f"<span style='font-size:36px; font-weight:700; color:{color};'>{pred_label.upper()}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Scenario Physics")
        m1, m2 = st.columns(2)
        m1.metric("Actual Charge Weight", f"{charge_weight:.1f} kg")
        m2.metric("TNT-Equivalent Weight", f"{W_eff:.1f} kg")
        m3, m4, m5 = st.columns(3)
        m3.metric("Scaled Distance Z", f"{Z:.2f} m/kg¹ᐟ³")
        m4.metric("Peak Overpressure", f"{P:.1f} kPa")
        m5.metric("Impulse", f"{I:.1f} kPa·ms")

        st.markdown("#### Prediction Confidence")
        proba_df = pd.DataFrame({
            "Damage Category": [label_encoder.classes_[i] for i in range(len(pred_proba))],
            "Probability": pred_proba,
        })
        proba_df["Damage Category"] = pd.Categorical(
            proba_df["Damage Category"], categories=damage_order, ordered=True
        )
        proba_df = proba_df.sort_values("Damage Category")

        fig_bar = go.Figure(go.Bar(
            x=proba_df["Probability"], y=proba_df["Damage Category"], orientation="h",
            marker_color=[DAMAGE_COLORS.get(d, "#333") for d in proba_df["Damage Category"]],
            text=[f"{p:.0%}" for p in proba_df["Probability"]], textposition="outside",
        ))
        fig_bar.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(range=[0, 1], title="Probability"), yaxis=dict(title=""),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("Pressure-Impulse Diagram")
        st.caption(
            f"{SHAPE_LABELS.get(charge_shape, charge_shape)} {explosive_type} charge vs. "
            f"{STRUCTURE_LABELS.get(structure_type, structure_type)} damage thresholds"
        )

        curves = pi_curves[structure_type]
        fig = go.Figure()

        I_range = np.logspace(-0.5, 3.5, 300)
        for level in ["minor", "moderate", "severe", "collapse"]:
            P0, I0, A = curves[level]
            with np.errstate(divide="ignore", invalid="ignore"):
                P_curve = P0 + A / np.maximum(I_range - I0, 1e-6)
            mask = (I_range > I0) & (P_curve > P0) & (P_curve < 1000)
            fig.add_trace(go.Scatter(
                x=I_range[mask], y=P_curve[mask], mode="lines", name=f"{level} threshold",
                line=dict(color=DAMAGE_COLORS[level], width=2, dash="dot"),
            ))

        fig.add_trace(go.Scatter(
            x=[I], y=[P], mode="markers", name="Your scenario",
            marker=dict(size=16, color=DAMAGE_COLORS.get(pred_label, "black"),
                        line=dict(width=2, color="black"), symbol="star"),
        ))

        fig.update_layout(
            xaxis=dict(type="log", title="Impulse (kPa·ms)"),
            yaxis=dict(type="log", title="Peak Overpressure (kPa)"),
            height=480, margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    acc = model_metadata.get("test_accuracy")
    f1 = model_metadata.get("test_macro_f1")
    acc_str = f"{acc:.1%}" if acc is not None else "N/A"
    f1_str = f"{f1:.3f}" if f1 is not None else "N/A"
    with st.expander("About this model"):
        st.markdown(
            f"""
            - Trained on a **synthetic dataset** (8000 samples) generated from
              Kingery-Bulmash style empirical overpressure/impulse formulas,
              TNT-equivalence conversion for 4 explosive types, 3 charge shapes,
              and Pressure-Impulse (P-I) damage curves for 4 structure types.
            - Model: **XGBoost classifier** ({model_metadata.get('strategy', 'class-weighted XGBoost')}),
              {acc_str} test accuracy, {f1_str} macro-F1.
            - This is a portfolio/research project, not a validated tool for real
              safety-critical decisions. For real blast-resistant design, refer to
              UFC 3-340-02 and consult a qualified structural engineer.
            """
        )


def render_methodology_tab():
    equiv_lines = "\n".join(
        f"  - **{e}**: {v}× TNT-equivalent (by mass)" for e, v in explosive_tnt_equivalence.items()
    )
    shape_lines = "\n".join(
        f"  - **{SHAPE_LABELS.get(s, s)}**: {v}× effective yield multiplier"
        for s, v in charge_shape_factor.items()
    )
    st.markdown(
        f"""
        ### How this project was built

        This tool follows a physics-informed synthetic data pipeline, then
        layers machine learning on top of it — a natural extension of
        **HazardScope**, which computes blast overpressure directly from
        empirical formulas without any learned model.

        #### 1. Synthetic data generation
        - **Physics inputs**: actual charge weight (0.5–2000 kg) and standoff
          distance (1–200 m) are sampled log-uniformly.
        - **Explosive type**: actual mass is converted to a TNT-equivalent
          mass using representative mass-based equivalence factors:
{equiv_lines}
        - **Charge shape / burst type**: an additional multiplier reflects how
          burst geometry changes effective blast coupling:
{shape_lines}
        - **Scaled distance**: `Z = R / W_eff^(1/3)` (Hopkinson–Cranz cube-root
          scaling), where `W_eff` is the fully adjusted TNT-equivalent weight.
        - **Overpressure & impulse**: estimated from `Z` using simplified
          Kingery-Bulmash style empirical fits.
        - **Damage labeling**: nested Pressure-Impulse (P-I) curves per
          structure type, `(P - P0)(I - I0) = A`, assign the highest damage
          level a scenario falls beyond.
        - **Noise**: construction-quality factor and measurement noise are
          added so the dataset isn't a perfect deterministic lookup table.

        #### 2. Model training
        An XGBoost classifier trained on 11 features (physics + one-hot
        encoded structure/explosive/shape categories) reaches **96.9% test
        accuracy and 0.936 macro-F1** — an improvement over the earlier
        version without explosive type and charge shape (0.914 macro-F1),
        suggesting these additional dimensions carry real predictive signal
        rather than just adding noise.

        #### 3. Deployment
        The final model, scaler, and label encoder are serialized with
        `joblib` and loaded into this Streamlit app, which recomputes the
        blast physics live from user inputs and calls the model for a
        prediction — no retraining needed at request time.

        ---

        **Honest limitations:**
        - The entire dataset is synthetic — no real blast-test measurements
          were used to validate the P-I curve constants, TNT-equivalence
          factors, or shape multipliers, which are representative
          approximations rather than exact reference-table values.
        - The Kingery-Bulmash fit used is a simplified polynomial
          approximation, not the full tabulated TM5-855-1 / UFC 3-340-02
          curves.
        - The cylindrical charge shape multiplier is a simplified
          representative value; real cylindrical charges have orientation-
          dependent (side-on vs end-on) effects not modeled here.
        - This should be read as a demonstration of an ML pipeline applied
          to a physically-grounded problem, not a validated engineering
          tool.
        """
    )


tab_predict, tab_methodology = st.tabs(["Predictor", "Methodology"])

with tab_predict:
    render_predictor_tab()

with tab_methodology:
    render_methodology_tab()
