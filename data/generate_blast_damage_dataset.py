"""
Synthetic Blast Damage Dataset Generator
==========================================
Generates labeled training data for the "Structural Damage Prediction from
Blast Loading" ML project.

Pipeline:
  1. Sample charge weight (W) and standoff distance (R)
  2. Compute scaled distance Z = R / W^(1/3)
  3. Compute peak reflected overpressure (P) and impulse (I) using
     Kingery-Bulmash style empirical curve fits
  4. For each structure type, compare (P, I) against nested P-I damage
     curves to assign a damage category
  5. Inject realistic noise/variability (material quality, construction
     variance) so the dataset isn't a perfect lookup table
  6. Save everything to a CSV

Notes:
  - The Kingery-Bulmash fit used here is a simplified polynomial
    approximation (log-log space) commonly used for quick engineering
    estimates. It is NOT the full US Army TM5-855-1 / UFC 3-340-02 table,
    but tracks the same qualitative trend and is good enough for a
    synthetic ML dataset. Swap in your HazardScope formulas here if you
    want numeric consistency with that project.
  - P-I curve constants (P0, I0, A) per structure type/damage level are
    representative values assembled from published blast-engineering
    literature trends (UFC 3-340-02, Krauthammer, Smith & Hetherington),
    not exact table lookups. Treat them as reasonable starting points to
    refine later if you get access to real reference tables.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Kingery-Bulmash style overpressure & impulse (simplified empirical fit)
# ---------------------------------------------------------------------------

def scaled_distance(R_m, W_kg):
    """Hopkinson-Cranz scaled distance Z = R / W^(1/3)  [m/kg^(1/3)]"""
    return R_m / (W_kg ** (1 / 3))


def peak_overpressure_kpa(Z):
    """
    Simplified empirical fit for peak incident overpressure (kPa) vs
    scaled distance Z (m/kg^1/3). Approximates the Kingery-Bulmash curve
    shape: steep falloff at small Z, asymptotic at large Z.
    Valid roughly for Z in [0.2, 40].
    """
    Z = np.clip(Z, 0.2, 40)
    # log-log polynomial fit (coefficients chosen to match KB curve trend)
    logZ = np.log(Z)
    log_P = 7.2 - 1.85 * logZ - 0.10 * logZ ** 2 + 0.06 * logZ ** 3
    P = np.exp(log_P)  # kPa
    return P


def impulse_kpa_ms(Z, W_kg):
    """
    Simplified empirical fit for scaled positive impulse, then converted
    to actual impulse (kPa*ms) using cube-root scaling: I = I_bar * W^(1/3)
    """
    Z = np.clip(Z, 0.2, 40)
    logZ = np.log(Z)
    log_Ibar = 2.65 - 0.95 * logZ + 0.02 * logZ ** 2
    I_bar = np.exp(log_Ibar)  # scaled impulse
    I = I_bar * (W_kg ** (1 / 3))
    return I


# ---------------------------------------------------------------------------
# 2. Structure types & nested P-I damage curves
#    (P - P0)(I - I0) = A   -->  point is at/beyond this damage level if
#    (P - P0)(I - I0) >= A  for P > P0 and I > I0
# ---------------------------------------------------------------------------

# Each entry: damage_level -> (P0 [kPa], I0 [kPa*ms], A)
# Curves are nested: collapse threshold is hardest to reach, minor is easiest.
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

DAMAGE_ORDER = ["none", "minor", "moderate", "severe", "collapse"]


def classify_damage(P, I, structure_type, quality_factor=1.0):
    """
    Determine the damage category for given (P, I) and structure type.
    quality_factor scales the P0/I0/A thresholds slightly to simulate
    real-world variability in construction quality / material strength.
    quality_factor > 1.0  => structure is tougher (needs more P,I to damage)
    quality_factor < 1.0  => structure is weaker
    """
    curves = STRUCTURE_PI_CURVES[structure_type]
    reached = "none"
    for level in ["minor", "moderate", "severe", "collapse"]:
        P0, I0, A = curves[level]
        P0 *= quality_factor
        I0 *= quality_factor
        A *= quality_factor ** 2
        if P > P0 and I > I0 and (P - P0) * (I - I0) >= A:
            reached = level
        else:
            break  # nested curves: if this level isn't met, higher ones aren't either
    return reached


# ---------------------------------------------------------------------------
# 3. Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(n_samples=6000, seed=42):
    rng = np.random.default_rng(seed)
    structure_types = list(STRUCTURE_PI_CURVES.keys())

    rows = []
    for _ in range(n_samples):
        # Sample charge weight (kg TNT eq.) - log-uniform is realistic
        # (covers small IEDs to large vehicle-borne charges)
        W = np.exp(rng.uniform(np.log(0.5), np.log(2000)))  # 0.5 kg to 2000 kg

        # Sample standoff distance (m) - log-uniform
        R = np.exp(rng.uniform(np.log(1), np.log(200)))  # 1 m to 200 m

        Z = scaled_distance(R, W)
        P = peak_overpressure_kpa(Z)
        I = impulse_kpa_ms(Z, W)

        structure_type = rng.choice(structure_types)

        # Construction quality variability (real-world noise)
        quality_factor = rng.normal(loc=1.0, scale=0.12)
        quality_factor = np.clip(quality_factor, 0.7, 1.3)

        # Measurement/environmental noise on P and I themselves
        P_noisy = P * rng.normal(loc=1.0, scale=0.08)
        I_noisy = I * rng.normal(loc=1.0, scale=0.08)
        P_noisy = max(P_noisy, 0.01)
        I_noisy = max(I_noisy, 0.01)

        damage = classify_damage(P_noisy, I_noisy, structure_type, quality_factor)

        rows.append({
            "charge_weight_kg": round(W, 3),
            "standoff_distance_m": round(R, 3),
            "scaled_distance_Z": round(Z, 4),
            "structure_type": structure_type,
            "quality_factor": round(quality_factor, 3),
            "peak_overpressure_kpa": round(P_noisy, 3),
            "impulse_kpa_ms": round(I_noisy, 3),
            "damage_category": damage,
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_dataset(n_samples=6000, seed=42)

    print("Dataset shape:", df.shape)
    print("\nDamage category distribution:")
    print(df["damage_category"].value_counts())
    print("\nDamage category distribution by structure type:")
    print(pd.crosstab(df["structure_type"], df["damage_category"]))

    out_path = "/mnt/user-data/outputs/blast_damage_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved dataset to: {out_path}")
