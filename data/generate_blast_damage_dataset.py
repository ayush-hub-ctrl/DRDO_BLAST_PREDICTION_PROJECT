"""
Synthetic Blast Damage Dataset Generator (v2)
================================================
Adds two new dimensions on top of the original charge weight / standoff
distance / structure type model:

  - explosive_type: converts actual charge mass into an effective TNT
    equivalent mass using standard mass-based TNT equivalence factors
  - charge_shape: spherical (free-air burst), hemispherical (surface
    burst), or cylindrical, each applying a multiplier to effective yield
    to approximate how burst geometry changes blast coupling

Pipeline:
  1. Sample actual charge weight (W_actual), standoff distance (R),
     explosive type, and charge shape
  2. Convert to effective TNT-equivalent weight:
         W_eff = W_actual * TNT_equivalence_factor * shape_factor
  3. Compute scaled distance Z = R / W_eff^(1/3)
  4. Compute peak overpressure (P) and impulse (I) from Z using
     Kingery-Bulmash style empirical curve fits
  5. Label damage using nested Pressure-Impulse (P-I) curves per
     structure type
  6. Add noise/variability, save to CSV

Notes on the TNT equivalence and shape factors:
  These are widely-cited representative approximations used in blast
  engineering literature and safety-distance tables. They vary somewhat
  by source and by which effect (pressure vs impulse) is being matched.
  Treat them as reasonable starting points for a synthetic/portfolio
  dataset, not as authoritative reference values for real safety
  assessments — for that, consult UFC 3-340-02 or an equivalent
  validated reference and a qualified professional.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Explosive type -> TNT mass-equivalence factor
#    (representative values commonly cited in blast-engineering references;
#    treat as approximate, not authoritative)
# ---------------------------------------------------------------------------

EXPLOSIVE_TNT_EQUIVALENCE = {
    "TNT":  1.00,
    "RDX":  1.60,
    "C4":   1.37,
    "ANFO": 0.82,
}

# ---------------------------------------------------------------------------
# 2. Charge shape -> effective yield multiplier
#    Free-air (spherical) burst is the baseline (1.0). A surface
#    (hemispherical) burst roughly doubles effective yield due to ground
#    reflection, a standard simplification in blast engineering. A
#    cylindrical charge is treated as an intermediate case reflecting
#    partial confinement / asymmetric coupling.
# ---------------------------------------------------------------------------

CHARGE_SHAPE_FACTOR = {
    "spherical":     1.00,
    "hemispherical": 2.00,
    "cylindrical":   1.30,
}

# ---------------------------------------------------------------------------
# 3. Kingery-Bulmash style overpressure & impulse (simplified empirical fit)
#    — unchanged from v1, just now applied to W_eff instead of raw W
# ---------------------------------------------------------------------------

def scaled_distance(R_m, W_eff_kg):
    """Hopkinson-Cranz scaled distance Z = R / W^(1/3)  [m/kg^(1/3)]"""
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
# 4. Structure types & nested P-I damage curves — unchanged from v1
# ---------------------------------------------------------------------------

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
            break
    return reached


# ---------------------------------------------------------------------------
# 5. Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(n_samples=8000, seed=42):
    rng = np.random.default_rng(seed)
    structure_types = list(STRUCTURE_PI_CURVES.keys())
    explosive_types = list(EXPLOSIVE_TNT_EQUIVALENCE.keys())
    charge_shapes = list(CHARGE_SHAPE_FACTOR.keys())

    rows = []
    for _ in range(n_samples):
        W_actual = np.exp(rng.uniform(np.log(0.5), np.log(2000)))  # kg, actual mass
        R = np.exp(rng.uniform(np.log(1), np.log(200)))            # m

        explosive_type = rng.choice(explosive_types)
        charge_shape = rng.choice(charge_shapes)

        tnt_factor = EXPLOSIVE_TNT_EQUIVALENCE[explosive_type]
        shape_factor = CHARGE_SHAPE_FACTOR[charge_shape]
        W_eff = W_actual * tnt_factor * shape_factor

        Z = scaled_distance(R, W_eff)
        P = peak_overpressure_kpa(Z)
        I = impulse_kpa_ms(Z, W_eff)

        structure_type = rng.choice(structure_types)

        quality_factor = rng.normal(loc=1.0, scale=0.12)
        quality_factor = np.clip(quality_factor, 0.7, 1.3)

        P_noisy = max(P * rng.normal(loc=1.0, scale=0.08), 0.01)
        I_noisy = max(I * rng.normal(loc=1.0, scale=0.08), 0.01)

        damage = classify_damage(P_noisy, I_noisy, structure_type, quality_factor)

        rows.append({
            "charge_weight_kg": round(W_actual, 3),
            "explosive_type": explosive_type,
            "charge_shape": charge_shape,
            "tnt_equivalent_weight_kg": round(W_eff, 3),
            "standoff_distance_m": round(R, 3),
            "scaled_distance_Z": round(Z, 4),
            "structure_type": structure_type,
            "quality_factor": round(quality_factor, 3),
            "peak_overpressure_kpa": round(P_noisy, 3),
            "impulse_kpa_ms": round(I_noisy, 3),
            "damage_category": damage,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset(n_samples=8000, seed=42)

    print("Dataset shape:", df.shape)
    print("\nDamage category distribution:")
    print(df["damage_category"].value_counts())
    print("\nExplosive type distribution:")
    print(df["explosive_type"].value_counts())
    print("\nCharge shape distribution:")
    print(df["charge_shape"].value_counts())
    print("\nDamage by explosive type:")
    print(pd.crosstab(df["explosive_type"], df["damage_category"]))
    print("\nDamage by charge shape:")
    print(pd.crosstab(df["charge_shape"], df["damage_category"]))

    out_path = "blast_damage_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved dataset to: {out_path}")
