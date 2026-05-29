"""
Módulo de clases para el modelo Ensemble_top3 v21 · Tesis Sepsis UPCH.
Estas clases viven a nivel de módulo (no en __main__) para que joblib pueda
serializar/deserializar sin errores de unpickling.

Pipeline 6 pasos (sección 4.3.1 de la tesis):
  1. MICE (IterativeImputer)
  2. StandardScaler
  3. Feature engineering clínico (8 derivadas, función add_clinical_features)
  4. Predicción individual XGB+LGB+ExtraTrees (cada uno cascade-calibrado)
  5. Promedio igual (ensemble) — clase EnsembleTop3
  6. Clasificación con umbral Youden 0.135
"""
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# CASCADE CALIBRATION (Platt + Isotonic)
# ═══════════════════════════════════════════════════════════════
class CascadeCal:
    """Calibración en cascada: sigmoid (Platt) seguido de isotónica."""
    def __init__(self, sig, iso):
        self.sig = sig
        self.iso = iso
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = sig.n_features_in_

    def predict_proba(self, X):
        p1 = np.clip(self.iso.transform(self.sig.predict_proba(X)[:, 1]), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ═══════════════════════════════════════════════════════════════
# ENSEMBLE_TOP3 · Promedio igual de 3 base learners calibrados
# ═══════════════════════════════════════════════════════════════
class EnsembleTop3:
    """Ensemble por promedio igual de 3 base learners cascade-calibrados.

    members: dict {nombre: CascadeCal}, esperado {'XGB', 'LGB', 'ExtraTrees'}.
    """
    def __init__(self, members: dict):
        self.members = members
        self.names = list(members.keys())
        first = next(iter(members.values()))
        self.n_features_in_ = first.n_features_in_
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        probs = np.column_stack([m.predict_proba(X)[:, 1] for m in self.members.values()])
        p1 = probs.mean(axis=1)
        return np.column_stack([1 - p1, p1])

    def predict_proba_per_member(self, X):
        """Probabilidades de cada base learner por separado (para análisis)."""
        return {name: m.predict_proba(X)[:, 1] for name, m in self.members.items()}

    def predict(self, X, threshold=0.135):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING · 8 variables derivadas (paso 3)
# ═══════════════════════════════════════════════════════════════
def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 8 features clínicas derivadas (fe_*) al dataframe.

    Idéntica a la función `add_clinical_features` del notebook v21 (celda 9).
    """
    d = df.copy()
    eps = 1e-6
    cols = set(d.columns)
    if {"heart_rate", "sbp"} <= cols:
        d["fe_shock_index"] = d["heart_rate"] / (d["sbp"].clip(lower=1) + eps)
    if {"bun", "creatinine"} <= cols:
        d["fe_bun_creat"] = d["bun"] / (d["creatinine"].clip(lower=0.1) + eps)
    if {"age", "sofa_score"} <= cols:
        d["fe_age_sofa"] = d["age"] * d["sofa_score"].fillna(0)
    if {"hemoglobin", "lactate"} <= cols:
        d["fe_hgb_lact"] = d["hemoglobin"] / (d["lactate"].clip(lower=0.1) + eps)
    if {"inr", "total_bilirubin"} <= cols:
        d["fe_liver"] = d["inr"].fillna(1) * d["total_bilirubin"].fillna(0)
    if {"bun", "albumin"} <= cols:
        d["fe_bun_alb"] = d["bun"] / (d["albumin"].clip(lower=0.1) + eps)
    if {"aniongap", "albumin"} <= cols:
        d["fe_ag_alb"] = d["aniongap"] + 2.5 * (4.0 - d["albumin"].fillna(4.0))
    if {"platelets", "inr"} <= cols:
        d["fe_coag"] = d["platelets"].fillna(150) / (d["inr"].clip(lower=0.5) + eps)
    return d


# ═══════════════════════════════════════════════════════════════
# CONSTANTES · Lista canónica de features
# ═══════════════════════════════════════════════════════════════
# Orden CANÓNICO de features (igual al meta_top3_final.json del notebook v21).
# CRÍTICO: cambiar este orden cambia el modelo (random feature subsampling).
FEATS_STRUCT = [
    "age", "icu_los_days", "sofa_score", "albumin",
    "hemoglobin", "aptt", "total_bilirubin", "wbc",
    "platelets", "hematocrit", "inr", "creatinine",
    "bun", "sodium", "potassium", "chloride",
    "bicarbonate", "glucose", "calcium", "aniongap",
    "alt", "ast", "alp", "bilirubin_direct",
    "pt", "lactate", "ph", "heart_rate",
    "mean_bp", "sbp", "resp_rate", "spo2",
]

# Mapeo a 5 grupos clínicos (para UI del formulario)
GROUPS_5 = {
    "demograficas":  ["age", "icu_los_days"],
    "signos_vitales": ["heart_rate", "sbp", "mean_bp", "resp_rate", "spo2"],
    "lab_general":   ["hemoglobin", "hematocrit", "platelets", "wbc",
                      "albumin", "total_bilirubin", "bilirubin_direct",
                      "alt", "ast", "alp", "creatinine", "bun",
                      "sodium", "potassium", "chloride", "bicarbonate",
                      "calcium", "aniongap", "glucose", "lactate", "ph"],
    "coagulacion":   ["aptt", "pt", "inr"],
    "scores":        ["sofa_score"],
}

FE_COLS = [
    "fe_shock_index", "fe_bun_creat", "fe_age_sofa", "fe_hgb_lact",
    "fe_liver", "fe_bun_alb", "fe_ag_alb", "fe_coag",
]

ALL_COLS = FEATS_STRUCT + FE_COLS  # 32 + 8 = 40

# Justificación clínica de cada derivada (para tooltips en UI)
FE_DESCRIPTIONS = {
    "fe_shock_index":  ("Shock Index", "HR/SBP — predictor temprano de shock séptico"),
    "fe_bun_creat":    ("Ratio BUN/Cr", "Diferencia azoemia prerrenal vs intrínseca"),
    "fe_age_sofa":     ("Edad × SOFA", "Interacción edad-disfunción orgánica"),
    "fe_hgb_lact":     ("Hb/Lactato", "Capacidad O₂ vs anaerobiosis tisular"),
    "fe_liver":        ("INR × Bili", "Score hepatosintético compuesto"),
    "fe_bun_alb":      ("BUN/Albúmina", "Catabolismo + falla renal (NEJM 2019)"),
    "fe_ag_alb":       ("AG corregido", "Anion gap corregido por albúmina (Figge 1998)"),
    "fe_coag":         ("Plaq/INR", "Score compuesto de coagulopatía"),
}
