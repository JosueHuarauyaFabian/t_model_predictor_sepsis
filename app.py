"""
app_sepsis.py — Predictor de Mortalidad UCI · Sepsis/SAD · UPCH
UX/UI v6: modelo real Ensemble_top3, pipeline 6 pasos, 5 grupos clínicos
Ejecutar: streamlit run app_sepsis.py

Pipeline 4.3.1: MICE → StandardScaler → FE 8 derivadas → XGB+LGB+ET calibrados → Ensemble → umbral 0.135
"""
import warnings, json, math, sys
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Importar clases del modelo (necesario para que joblib unpickle correctamente)
sys.path.insert(0, str(Path(__file__).parent))
from src.sepsis_model import (
    CascadeCal, EnsembleTop3, add_clinical_features,
    FEATS_STRUCT, FE_COLS, ALL_COLS, FE_DESCRIPTIONS, GROUPS_5,
)

# ═══════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════
UPCH_PRIMARY   = "#003A70"
UPCH_SECONDARY = "#0066A1"
UPCH_GOLD      = "#C8A044"
SURFACE        = "#FFFFFF"
SURFACE_TINT   = "#F8FAFC"
BG             = "#F1F5F9"
BORDER         = "#E2E8F0"
BORDER_STRONG  = "#CBD5E1"
TEXT_PRIMARY   = "#0F172A"
TEXT_SECONDARY = "#475569"
TEXT_MUTED     = "#94A3B8"
# Paleta de riesgo · tonos pastel medicinales (sage/amber/rose)
RISK_LOW       = "#7FA98C"   # sage green
RISK_MOD       = "#C9985A"   # muted amber
RISK_HIGH      = "#C57878"   # dusty rose
RISK_CRIT      = "#8B5050"   # muted burgundy
# Backgrounds (light tints)
RISK_LOW_BG    = "#ECF2EE"
RISK_MOD_BG    = "#F4ECDB"
RISK_HIGH_BG   = "#F2E2E2"
# Borders (intermediate tints)
RISK_LOW_BORDER  = "#BCD4C5"
RISK_MOD_BORDER  = "#DCC4A0"
RISK_HIGH_BORDER = "#DDB4B4"
# Text para backgrounds claros
RISK_LOW_TEXT  = "#4F7C66"
RISK_MOD_TEXT  = "#8C6235"
RISK_HIGH_TEXT = "#8B5050"

LOGO_FACULTAD  = "https://ciencias.cayetano.edu.pe/wp-content/uploads/sites/28/2023/05/ciencias-e-ingenieria-oficial.png"

MODEL_META = {
    "version":     "v21", "fecha_es": "19 May 2026", "fecha_en": "May 19, 2026",
    "cohorte":     "MIMIC-IV v3.1", "n_train": 12_564, "n_test": 1_885,
    "prevalencia": 0.143,
    "auc":         0.900, "auc_ic_lo": 0.883, "auc_ic_hi": 0.918,
    "sens":        0.859, "esp": 0.780, "brier": 0.076, "hl_p": 0.619,
    "umbral":      0.135,
}

# ═══════════════════════════════════════════════════════════════
# i18n · DICCIONARIO COMPLETO ES / EN
# ═══════════════════════════════════════════════════════════════
TR = {
    "es": {
        # Page title
        "page_title":  "Predictor de Mortalidad a 28 días · Sepsis / SAD",
        "page_sub_1":  "Facultad de Ciencias e Ingeniería",
        "page_sub_2":  "Modelo Ensemble_top3",
        "page_sub_n":  "pacientes",

        # Tabs
        "tab1": "🩺  1. Formulario para probar el modelo",
        "tab2": "📈  2. Rendimiento del modelo",
        "tab3": "🏆  3. Comparativa entre modelos",
        "tab4": "🧬  4. Cohorte del estudio",
        "tab5": "🧠  5. Cómo funciona el Modelo A + Notas",

        # Sidebar
        "sb_auc":         "AUC validada",
        "sb_lang_lbl":    "Idioma de la interfaz",
        "sb_lang_help":   "Cambia ES/EN para todo el contenido y la nota clínica.",
        "sb_model":       "Modelo",
        "sb_cohort":      "Cohorte",
        "sb_prev":        "Prevalencia",
        "sb_sens":        "Sensibilidad",
        "sb_esp":         "Especificidad",
        "sb_hlp":         "HL p",
        "sb_threshold":   "Umbral Youden",
        "sb_updated":     "Actualización",
        "sb_translator_warn": "Traductor offline. Cambia a EN o instala `deep-translator`.",
        "sb_ref_lbl":     "Zhang et al. (2024)",
        "sb_doi":         "DOI: 10.1038/s41598-024-69332-4",

        # Form
        "patient_data":     "Datos del paciente",
        "patient_data_help":"Las **5 variables centrales** son obligatorias. La nota clínica es opcional y activa el **Modelo A + Notas**.",
        "input_icu":        "Estancia en UCI (días) · X1",
        "input_hb":         "Hemoglobina (g/dL) · X2",
        "input_alb":        "Albúmina (g/dL) · X3",
        "input_aptt":       "aPTT (segundos) · X4",
        "input_bili":       "Bilirrubina total (mg/dL) · X5",
        "additional_expander":"➕ Variables clínicas adicionales (opcionales)",
        "input_age":        "Edad (años)",
        "input_sofa":       "SOFA score",
        "input_lact":       "Lactato (mmol/L)",
        "input_creat":      "Creatinina (mg/dL)",
        "note_section":     "📝 Nota clínica (Modelo A)",
        "note_label_active":"Escribe en español · se traduce automáticamente al inglés",
        "note_placeholder": "Ejemplo: Paciente varón de 72 años con shock séptico de origen pulmonar, hipotensión refractaria a noradrenalina, oliguria progresiva y lactato 6.2. Intubación orotraqueal.",
        "note_translating": "Traduciendo...",
        "translation_label":"🇬🇧 Traducción al inglés (entrada del modelo)",
        "translation_failed":"La traducción falló. Verifica conexión o cambia a modo EN.",
        "btn_calculate":    "Calcular predicción  ▸",

        # Result
        "result_section":   "Resultado de la predicción",
        "model_a_name":     "Modelo A · Sin notas",
        "model_b_name":     "Modelo A + Notas",
        "model_b_empty":    "Sin nota clínica · escribe una nota en el formulario para activar el enriquecimiento textual",
        "verdict_vive":     "SOBREVIVE",
        "verdict_muere":    "FALLECE",
        "risk_prefix":      "Riesgo",
        "risk_low":         "BAJO",
        "risk_mod":         "MODERADO",
        "risk_high":        "ALTO",
        "risk_crit":        "CRÍTICO",
        "death_risk_label":   "Riesgo de muerte a 28 días",
        "survival_prob_label":"Probabilidad de supervivencia",
        "card_a_foot":      "5 variables Zhang + extras · Sin información textual · Umbral",
        "card_b_signals":   "señales detectadas",
        "card_b_no_signals":"Sin nota clínica activa",
        "card_b_foot_sep":  "· Umbral",
        "delta_label":      "Aporte de la nota clínica · Δ (B − A)",
        "delta_higher":     "La nota sugiere mayor severidad clínica",
        "delta_lower":      "La nota sugiere mejoría clínica",
        "delta_neutral":    "La nota no modifica significativamente la predicción",
        "alerts_section":   "⚠ Alertas clínicas detectadas",
        "empty_text":       "Ingresa los datos del paciente",
        "empty_sub":        "El resultado aparecerá aquí al presionar **Calcular predicción**",

        # Tab 2
        "tab2_metric_auc":  "AUC",
        "tab2_metric_sens": "Sensibilidad",
        "tab2_metric_esp":  "Especificidad",
        "tab2_metric_hlp":  "HL p",
        "tab2_calib_ok":    "Calibración adecuada",
        "tab2_calib_sub":   "Subestándar",
        "tab2_caption_roc": "Curva ROC empírica (test n=1,885)",
        "tab2_caption_cal": "Curva de calibración empírica (reliability diagram)",
        "tab2_missing":     "Ejecuta `python scripts/generar_figuras_4_3_4_4.py` para generar.",

        # Tab 3
        "tab3_caption":     "Tabla comparativa de los **9 modelos evaluados** sobre el conjunto de prueba independiente (n = 1,885). El modelo final **Ensemble_top3** fue seleccionado por calibración óptima (HL p > 0.5), no por máxima AUC.",
        "tab3_missing":     "Tabla no disponible. Ejecuta el notebook `sepsis_v21_final.ipynb`.",

        # Tab 4
        "tab4_metric_n":    "Pacientes en cohorte",
        "tab4_metric_prev": "Prevalencia muerte 28d",
        "tab4_metric_dead": "Fallecidos",
        "tab4_comparison":  "Comparación con Zhang et al. (2024)",
        "tab4_paper_lbl":   "Paper:",
        "tab4_ours_lbl":    "Nuestra cohorte:",
        "tab4_mortality":   "mortalidad",
        "tab4_pacientes":   "pacientes",
        "tab4_flow_title":  "Selección de cohorte · Sepsis/SAD",
        "tab4_step_1":      "Dataset inicial MIMIC-IV",
        "tab4_step_2":      "Diagnóstico sepsis (ICD-10)",
        "tab4_step_3":      "Sepsis-3: SOFA ≥ 2",
        "tab4_step_4":      "Edad ≥ 18 años",
        "tab4_step_5":      "Estancia UCI > 24h",
        "tab4_step_6":      "Labs re-extraídos 72h",
        "tab4_step_7":      "✓ Cohorte final",

        # Tab 5
        "tab5_intro":       "El **Modelo A + Notas** combina la predicción del modelo estructurado con un análisis textual de la nota clínica:",
        "tab5_step_1":      "**Traducción automática** ES → EN mediante `deep-translator` (motor: Google Translate).",
        "tab5_step_2":      "**Detección de señales clínicas** sobre el texto en inglés:",
        "tab5_severe":      "🔴 **Severas**",
        "tab5_severe_lbl":  "términos",
        "tab5_severe_ex":   "septic shock, vasopressor, mechanical ventilation, ARDS, multi-organ failure, etc.",
        "tab5_moderate":    "🟡 **Moderadas**",
        "tab5_moderate_ex": "hypotension, oliguria, encephalopathy, etc.",
        "tab5_protective":  "🟢 **Protectoras**",
        "tab5_protective_ex":"stable, improving, extubated, responding to treatment, etc.",
        "tab5_step_3":      "**Ajuste logit** del modelo base según las señales detectadas (suma de pesos × 4).",
        "tab5_note":        "> En la versión completa de la tesis, este ajuste sintáctico es reemplazado por **BioBERT embeddings** que capturan contexto semántico, no solo presencia de keywords.",

        # Footer
        "footer":           "**Herramienta de apoyo clínico** — No reemplaza el juicio médico · Modelo Ensemble_top3 v21 sobre MIMIC-IV v3.1 · Zhang et al. (2024) · **UPCH** · Facultad de Ciencias e Ingeniería · 2026",

        # Alerts (clinical)
        "alert_alb":        "Albúmina < 2.5 g/dL",
        "alert_bili":       "Bilirrubina > 2.0 mg/dL",
        "alert_hb":         "Hemoglobina < 8.0 g/dL",
        "alert_sofa":       "SOFA ≥ 8",
        "alert_lact":       "Lactato > 4.0 mmol/L",
        "alert_creat":      "Creatinina > 3.0 mg/dL",
        "alert_aptt":       "aPTT > 100 seg",
    },

    "en": {
        # Page title
        "page_title":  "28-Day Mortality Predictor · Sepsis / SAD",
        "page_sub_1":  "School of Sciences and Engineering",
        "page_sub_2":  "Ensemble_top3 model",
        "page_sub_n":  "patients",

        # Tabs
        "tab1": "🩺  1. Form to test the model",
        "tab2": "📈  2. Model performance",
        "tab3": "🏆  3. Model comparison",
        "tab4": "🧬  4. Study cohort",
        "tab5": "🧠  5. How Model A + Notes works",

        # Sidebar
        "sb_auc":         "Validated AUC",
        "sb_lang_lbl":    "Interface language",
        "sb_lang_help":   "Switch ES/EN for all content and clinical note.",
        "sb_model":       "Model",
        "sb_cohort":      "Cohort",
        "sb_prev":        "Prevalence",
        "sb_sens":        "Sensitivity",
        "sb_esp":         "Specificity",
        "sb_hlp":         "HL p",
        "sb_threshold":   "Youden threshold",
        "sb_updated":     "Last updated",
        "sb_translator_warn": "Translator offline. Switch to EN or install `deep-translator`.",
        "sb_ref_lbl":     "Zhang et al. (2024)",
        "sb_doi":         "DOI: 10.1038/s41598-024-69332-4",

        # Form
        "patient_data":     "Patient data",
        "patient_data_help":"The **5 core variables** are mandatory. The clinical note is optional and activates **Model A + Notes**.",
        "input_icu":        "ICU length of stay (days) · X1",
        "input_hb":         "Hemoglobin (g/dL) · X2",
        "input_alb":        "Albumin (g/dL) · X3",
        "input_aptt":       "aPTT (seconds) · X4",
        "input_bili":       "Total bilirubin (mg/dL) · X5",
        "additional_expander":"➕ Additional clinical variables (optional)",
        "input_age":        "Age (years)",
        "input_sofa":       "SOFA score",
        "input_lact":       "Lactate (mmol/L)",
        "input_creat":      "Creatinine (mg/dL)",
        "note_section":     "📝 Clinical note (Model A)",
        "note_label_active":"Write clinical note in English · direct input to model",
        "note_placeholder": "Example: 72-year-old male with septic shock of pulmonary origin, hypotension refractory to norepinephrine, progressive oliguria and lactate 6.2. Orotracheal intubation.",
        "note_translating": "Translating...",
        "translation_label":"🇬🇧 English translation (model input)",
        "translation_failed":"Translation failed. Check connection or switch to EN mode.",
        "btn_calculate":    "Calculate prediction  ▸",

        # Result
        "result_section":   "Prediction result",
        "model_a_name":     "Model A · No notes",
        "model_b_name":     "Model A + Notes",
        "model_b_empty":    "No clinical note · write a note in the form to activate textual enrichment",
        "verdict_vive":     "SURVIVES",
        "verdict_muere":    "DIES",
        "risk_prefix":      "Risk",
        "risk_low":         "LOW",
        "risk_mod":         "MODERATE",
        "risk_high":        "HIGH",
        "risk_crit":        "CRITICAL",
        "death_risk_label":   "28-day death risk",
        "survival_prob_label":"Survival probability",
        "card_a_foot":      "5 Zhang variables + extras · No textual information · Threshold",
        "card_b_signals":   "signals detected",
        "card_b_no_signals":"No clinical note active",
        "card_b_foot_sep":  "· Threshold",
        "delta_label":      "Clinical note contribution · Δ (B − A)",
        "delta_higher":     "The note suggests higher clinical severity",
        "delta_lower":      "The note suggests clinical improvement",
        "delta_neutral":    "The note does not significantly modify the prediction",
        "alerts_section":   "⚠ Clinical alerts detected",
        "empty_text":       "Enter the patient data",
        "empty_sub":        "Result will appear here when you press **Calculate prediction**",

        # Tab 2
        "tab2_metric_auc":  "AUC",
        "tab2_metric_sens": "Sensitivity",
        "tab2_metric_esp":  "Specificity",
        "tab2_metric_hlp":  "HL p",
        "tab2_calib_ok":    "Adequate calibration",
        "tab2_calib_sub":   "Substandard",
        "tab2_caption_roc": "Empirical ROC curve (test n=1,885)",
        "tab2_caption_cal": "Empirical calibration curve (reliability diagram)",
        "tab2_missing":     "Run `python scripts/generar_figuras_4_3_4_4.py` to generate.",

        # Tab 3
        "tab3_caption":     "Comparison table of the **9 models evaluated** on the independent test set (n = 1,885). The final **Ensemble_top3** model was selected for optimal calibration (HL p > 0.5), not for maximum AUC.",
        "tab3_missing":     "Table not available. Run the `sepsis_v21_final.ipynb` notebook.",

        # Tab 4
        "tab4_metric_n":    "Patients in cohort",
        "tab4_metric_prev": "28-day mortality prevalence",
        "tab4_metric_dead": "Deceased",
        "tab4_comparison":  "Comparison with Zhang et al. (2024)",
        "tab4_paper_lbl":   "Paper:",
        "tab4_ours_lbl":    "Our cohort:",
        "tab4_mortality":   "mortality",
        "tab4_pacientes":   "patients",
        "tab4_flow_title":  "Cohort selection · Sepsis/SAD",
        "tab4_step_1":      "Initial MIMIC-IV dataset",
        "tab4_step_2":      "Sepsis diagnosis (ICD-10)",
        "tab4_step_3":      "Sepsis-3: SOFA ≥ 2",
        "tab4_step_4":      "Age ≥ 18 years",
        "tab4_step_5":      "ICU stay > 24h",
        "tab4_step_6":      "Re-extracted labs 72h",
        "tab4_step_7":      "✓ Final cohort",

        # Tab 5
        "tab5_intro":       "**Model A + Notes** combines the structured model prediction with a textual analysis of the clinical note:",
        "tab5_step_1":      "**Automatic translation** ES → EN via `deep-translator` (engine: Google Translate).",
        "tab5_step_2":      "**Clinical signal detection** on the English text:",
        "tab5_severe":      "🔴 **Severe**",
        "tab5_severe_lbl":  "terms",
        "tab5_severe_ex":   "septic shock, vasopressor, mechanical ventilation, ARDS, multi-organ failure, etc.",
        "tab5_moderate":    "🟡 **Moderate**",
        "tab5_moderate_ex": "hypotension, oliguria, encephalopathy, etc.",
        "tab5_protective":  "🟢 **Protective**",
        "tab5_protective_ex":"stable, improving, extubated, responding to treatment, etc.",
        "tab5_step_3":      "**Logit adjustment** of the base model according to detected signals (sum of weights × 4).",
        "tab5_note":        "> In the full thesis version, this syntactic adjustment is replaced by **BioBERT embeddings** that capture semantic context, not just keyword presence.",

        # Footer
        "footer":           "**Clinical decision support tool** — Does not replace medical judgment · Ensemble_top3 model v21 on MIMIC-IV v3.1 · Zhang et al. (2024) · **UPCH** · School of Sciences and Engineering · 2026",

        # Alerts
        "alert_alb":        "Albumin < 2.5 g/dL",
        "alert_bili":       "Bilirubin > 2.0 mg/dL",
        "alert_hb":         "Hemoglobin < 8.0 g/dL",
        "alert_sofa":       "SOFA ≥ 8",
        "alert_lact":       "Lactate > 4.0 mmol/L",
        "alert_creat":      "Creatinine > 3.0 mg/dL",
        "alert_aptt":       "aPTT > 100 sec",
    },
}

st.set_page_config(
    page_title="Predictor Sepsis · UPCH",
    page_icon=LOGO_FACULTAD,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, input, textarea, button {{
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
    color: {TEXT_PRIMARY};
}}
.main {{ background: {BG}; }}

.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 1500px;
}}

[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {{ display: none !important; }}
section[data-testid="stSidebar"] {{
    background: {SURFACE};
    min-width: 260px !important;
    max-width: 260px !important;
    border-right: 2px solid {UPCH_GOLD};
}}
section[data-testid="stSidebar"] > div {{ padding-top: 1.5rem; padding-left: 1rem; padding-right: 1rem; }}
section[data-testid="stSidebar"] img {{ max-width: 200px !important; height: auto !important; margin: 0 auto !important; display: block !important; }}

.page-title {{ padding: 4px 0 12px; margin-bottom: 18px; border-bottom: 1px solid {BORDER}; }}
.page-title h1 {{ font-size: 22px; font-weight: 700; color: {TEXT_PRIMARY}; margin: 0; letter-spacing: -0.4px; line-height: 1.2; }}
.page-title .page-sub {{ font-size: 12px; color: {TEXT_SECONDARY}; margin-top: 5px; font-weight: 500; }}
.page-title .page-sub .acc {{ color: {UPCH_PRIMARY}; font-weight: 700; }}
.page-title .page-sub .auc {{ background: {UPCH_PRIMARY}; color: white; padding: 2px 10px; border-radius: 999px; font-family: 'JetBrains Mono', monospace; font-size: 11px; margin-left: 6px; }}
.page-title .page-sub .auc .gold {{ color: {UPCH_GOLD}; font-weight: 700; }}

.sec-label {{
    font-size: 11px; font-weight: 700; color: {TEXT_SECONDARY};
    text-transform: uppercase; letter-spacing: 1px;
    margin: 4px 0 12px; padding-bottom: 6px;
    border-bottom: 1px solid {BORDER};
}}

.stNumberInput label, .stTextArea label, .stRadio label {{
    font-size: 13px !important; font-weight: 600 !important;
    color: {TEXT_PRIMARY} !important;
}}
.stNumberInput input, .stTextArea textarea {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important; font-weight: 500 !important;
    border-radius: 8px !important; border: 1px solid {BORDER_STRONG} !important;
}}
.stNumberInput input:focus, .stTextArea textarea:focus {{
    border-color: {UPCH_SECONDARY} !important;
    box-shadow: 0 0 0 3px rgba(0,102,161,0.12) !important;
}}

.translate-box {{
    background: {SURFACE_TINT}; border: 1px dashed {BORDER_STRONG};
    border-radius: 10px; padding: 12px 14px; margin: 8px 0 4px;
    font-size: 13px; color: {TEXT_PRIMARY}; line-height: 1.5;
    font-style: italic;
}}
.translate-label {{
    font-size: 10px; font-weight: 700; color: {UPCH_SECONDARY};
    text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px;
    font-style: normal;
}}

/* Primary button */
.stButton > button[kind="primary"],
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] div,
.stButton > button[kind="primary"] span {{ color: #FFFFFF !important; fill: #FFFFFF !important; }}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {UPCH_PRIMARY}, {UPCH_SECONDARY}) !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 14px !important;
    padding: 14px 24px !important; letter-spacing: 0.3px;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(0,58,112,0.25);
}}
.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 10px 24px -6px rgba(0,58,112,0.4) !important;
    transform: translateY(-1px);
}}
.stButton > button[kind="primary"]:focus {{ outline: none !important; }}
.stButton > button {{ border-radius: 10px !important; }}

/* Result cards */
.result-card {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 16px; padding: 20px 22px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
    text-align: center;
    min-height: 280px;
    display: flex; flex-direction: column;
}}
.result-card.modelo-a {{ border-top: 4px solid {UPCH_SECONDARY}; }}
.result-card.modelo-b {{ border-top: 4px solid {UPCH_GOLD}; }}
.result-card-tag {{
    display: inline-block; font-size: 10px; font-weight: 700;
    padding: 3px 10px; border-radius: 999px;
    letter-spacing: 0.5px; text-transform: uppercase;
    margin-bottom: 10px; align-self: center;
}}
.tag-a {{ background: rgba(0,102,161,0.08); color: {UPCH_SECONDARY}; }}
.tag-b {{ background: rgba(200,160,68,0.14); color: #8B6914; }}
.result-card-icon {{ font-size: 48px; line-height: 1; margin: 8px 0 4px; }}
.result-card-prob {{
    font-size: 44px; font-weight: 800; line-height: 1;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -1.8px; margin: 4px 0 2px;
}}
.result-card-prob .unit {{ font-size: 20px; font-weight: 600; opacity: 0.7; }}
.result-card-verdict {{
    font-size: 22px; font-weight: 800; letter-spacing: 1.5px;
    margin: 6px 0 8px; line-height: 1;
}}
.verdict-vive  {{ color: {RISK_LOW}; }}
.verdict-muere {{ color: {RISK_HIGH}; }}
.result-card-tier {{
    display: inline-block; font-size: 11px; font-weight: 700;
    padding: 4px 12px; border-radius: 999px; letter-spacing: 0.3px;
    text-transform: uppercase; margin: 0 auto;
}}
.result-card-tier.tier-low  {{ background: #ECF2EE; color: {RISK_LOW}; }}
.result-card-tier.tier-mod  {{ background: #F4ECDB; color: #8C6235; }}
.result-card-tier.tier-high {{ background: #F2E2E2; color: {RISK_HIGH}; }}
.result-card-tier.tier-crit {{ background: {RISK_CRIT}; color: white; }}
.result-card-foot {{
    font-size: 11px; color: {TEXT_MUTED}; margin-top: auto;
    padding-top: 10px; line-height: 1.4;
}}

.delta-card {{
    background: {SURFACE_TINT}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 14px 18px; margin-top: 8px;
    text-align: center;
}}
.delta-label {{ font-size: 11px; font-weight: 600; color: {TEXT_SECONDARY}; text-transform: uppercase; letter-spacing: 0.6px; }}
.delta-value {{
    font-size: 22px; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2px; letter-spacing: -1px;
}}
.delta-up   {{ color: {RISK_HIGH}; }}
.delta-down {{ color: {RISK_LOW}; }}
.delta-zero {{ color: {TEXT_MUTED}; }}

.kw-pills {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 10px; justify-content: center; }}
.kw-pill {{
    font-size: 10px; font-weight: 600;
    padding: 3px 9px; border-radius: 999px;
    background: {SURFACE_TINT}; border: 1px solid {BORDER};
    color: {TEXT_SECONDARY};
}}
.kw-pill.severe  {{ background: #F2E2E2; color: {RISK_HIGH}; border-color: #DDB4B4; }}
.kw-pill.protect {{ background: #ECF2EE; color: {RISK_LOW};  border-color: #BCD4C5; }}

.empty {{
    text-align: center; padding: 60px 24px;
    background: {SURFACE}; border: 2px dashed {BORDER};
    border-radius: 16px; color: {TEXT_MUTED};
}}
.empty .e-icon {{ font-size: 48px; opacity: 0.5; }}
.empty .e-text {{ font-size: 14px; font-weight: 600; color: {TEXT_SECONDARY}; margin-top: 14px; }}
.empty .e-sub {{ font-size: 12px; color: {TEXT_MUTED}; margin-top: 6px; }}

.sb-meta-card {{
    background: linear-gradient(135deg, {UPCH_PRIMARY}, {UPCH_SECONDARY});
    color: white; padding: 14px 16px; border-radius: 12px;
    margin: 14px 0;
}}
.sb-meta-card .lbl {{ font-size: 10px; opacity: 0.85; text-transform: uppercase; letter-spacing: 0.6px; }}
.sb-meta-card .val {{ font-size: 22px; font-weight: 800; font-family: 'JetBrains Mono', monospace; line-height: 1.1; margin-top: 4px; }}
.sb-meta-card .ic  {{ font-size: 11px; opacity: 0.85; margin-top: 4px; }}
.sb-meta-card .acc {{ color: {UPCH_GOLD}; }}
.sb-row {{ display: flex; justify-content: space-between; font-size: 12px; padding: 5px 0; border-bottom: 1px solid {BORDER}; }}
.sb-row:last-child {{ border-bottom: none; }}
.sb-row .lbl {{ color: {TEXT_SECONDARY}; font-weight: 500; }}
.sb-row .val {{ color: {TEXT_PRIMARY}; font-weight: 700; font-family: 'JetBrains Mono', monospace; }}

/* Sidebar language radio: 2 pills horizontales, una línea */
section[data-testid="stSidebar"] .stRadio > div {{
    flex-direction: row !important; gap: 6px !important;
    background: {SURFACE_TINT}; padding: 4px; border-radius: 10px;
    border: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .stRadio > div label {{
    flex: 1; text-align: center;
    padding: 8px 4px !important; border-radius: 8px;
    margin: 0 !important;
    font-size: 13px !important; font-weight: 700 !important;
    letter-spacing: 0.5px;
    cursor: pointer; transition: all 0.15s ease;
    white-space: nowrap !important;
    overflow: hidden;
    line-height: 1.2 !important;
    color: {TEXT_SECONDARY} !important;
}}
section[data-testid="stSidebar"] .stRadio > div label > div {{
    display: inline-block !important;
    white-space: nowrap !important;
}}
section[data-testid="stSidebar"] .stRadio > div label:has(input:checked) {{
    background: {UPCH_PRIMARY}; color: white !important;
}}
section[data-testid="stSidebar"] .stRadio > div label:has(input:checked) p {{
    color: white !important;
}}
section[data-testid="stSidebar"] .stRadio input {{ display: none; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px; background: transparent;
    border-bottom: 2px solid {BORDER};
    padding-bottom: 2px; margin-bottom: 16px;
}}
.stTabs [data-baseweb="tab"] {{
    background: {SURFACE_TINT}; border-radius: 10px 10px 0 0;
    padding: 10px 16px !important; font-weight: 600 !important;
    font-size: 13px !important; color: {TEXT_SECONDARY};
    border: 1px solid {BORDER}; border-bottom: none;
    margin-right: 2px;
    transition: all 0.15s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{ background: {SURFACE}; color: {UPCH_PRIMARY}; }}
.stTabs [aria-selected="true"] {{
    color: {UPCH_PRIMARY} !important;
    background: {SURFACE} !important;
    border-top: 3px solid {UPCH_GOLD} !important;
    font-weight: 700 !important;
}}

/* Hide Streamlit chrome (deploy bar, toolbar, header, sidebar header) */
#MainMenu, footer {{ visibility: hidden; }}
.stDeployButton,
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"],
[data-testid="stSidebarHeader"] {{ display: none !important; }}
header[data-testid="stHeader"] {{
    display: none !important;
    height: 0 !important;
}}
.stApp > header {{ display: none !important; }}

.app-footer {{
    text-align: center; color: {TEXT_MUTED}; font-size: 11px;
    padding: 18px 0 6px; margin-top: 22px;
    border-top: 1px solid {BORDER};
}}
.app-footer b {{ color: {UPCH_PRIMARY}; }}

/* ── Cohort flowchart · prolijo, numerado, paper-style ──── */
.cf-grid {{
    display: grid;
    grid-template-columns: 440px 60px 360px;
    justify-content: center;
    align-items: start;
    margin: 18px auto 0;
    max-width: 1100px;
    row-gap: 0;
}}

/* Step-unit · contiene cf-step + cf-vline (línea + cabezal estirable) */
.cf-step-unit {{
    grid-column: 1;
    align-self: stretch;
    display: flex;
    flex-direction: column;
}}

/* Cajas principales (dentro de cf-step-unit) */
.cf-step {{
    background: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    border-radius: 8px;
    padding: 14px 18px;
    display: flex; align-items: center; gap: 14px;
    box-shadow: 0 1px 2px rgba(15,23,42,0.05);
    flex-shrink: 0;
}}

/* Línea vertical + cabezal · STRETCH a la altura del row (driven by cf-side) */
.cf-vline {{
    flex: 1;
    min-height: 56px;
    position: relative;
    margin: 4px 0;
}}
.cf-vline::before {{
    content: '';
    position: absolute;
    left: 50%; top: 0; bottom: 10px;
    width: 1.5px;
    background: #94A3B8;
    transform: translateX(-50%);
}}
.cf-vline::after {{
    content: '';
    position: absolute;
    left: 50%; bottom: 0;
    width: 0; height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 9px solid #94A3B8;
    transform: translateX(-50%);
}}

/* Caja de cohorte final · alineada con cf-step (col 1) */
.cf-final {{
    grid-column: 1;
    align-self: start;
    background: #EFF4F8;
    border: 1.5px solid {UPCH_PRIMARY};
    border-radius: 8px;
    padding: 14px 18px;
    display: flex; align-items: center; gap: 14px;
    box-shadow: 0 2px 4px rgba(0,58,112,0.10);
}}

/* Badge numerada */
.cf-num {{
    width: 28px; height: 28px; border-radius: 50%;
    background: {UPCH_PRIMARY}; color: white;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 13px;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
}}
.cf-num-sage {{ background: #7FA98C !important; }}
.cf-num-rose {{ background: #C57878 !important; }}

.cf-body {{ flex: 1; min-width: 0; }}
.cf-title {{ font-size: 13px; font-weight: 600; color: {TEXT_PRIMARY}; line-height: 1.4; }}
.cf-sub {{ font-size: 11px; color: {TEXT_MUTED}; margin-top: 2px; font-style: italic; }}
.cf-n {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700; font-size: 14px;
    color: {UPCH_PRIMARY}; margin-top: 5px;
}}

/* Flecha horizontal (columna 2) · alineada con centro de cf-step */
.cf-arrow-h {{
    grid-column: 2;
    align-self: start;
    margin-top: 32px;
    display: flex; align-items: center; justify-content: center;
}}

/* Cuadro lateral de exclusión (columna 3) · natural height, no stretch */
.cf-side {{
    grid-column: 3;
    background: #FAF6EE;
    border: 1px solid #E5D9C0;
    border-radius: 8px;
    padding: 12px 16px;
    align-self: start;
    display: flex; flex-direction: column;
}}
.cf-side-title {{
    font-size: 11px; font-weight: 700;
    color: #7A5F2E;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}
.cf-side-list {{
    margin: 0; padding-left: 16px;
    font-size: 11px; color: #5C4A2D;
    line-height: 1.65;
}}
.cf-side-note {{
    font-size: 10px; color: #7A5F2E;
    margin-top: 8px; font-style: italic;
}}

/* Branching arrow (fork) entre cohorte final y SAD+/SAD- */
.cf-branch {{
    grid-column: 1 / -1;
    display: flex; justify-content: flex-start;
    margin: 8px auto 10px;
    width: 100%;
}}

/* Fila con dos cohortes SAD+/SAD- · alineadas con grid (col 1 y col 3) */
.cf-cohorts-row {{
    grid-column: 1 / -1;
    display: flex; gap: 40px;
    justify-content: flex-start;
    margin: 4px 0 0;
    width: 100%;
}}
.cf-cohort-block {{
    width: 380px; min-width: 380px;
    display: flex; flex-direction: column;
}}
.cf-cohort-block:first-child {{ margin-left: 30px; }}
.cf-sadp, .cf-sadn {{
    border-radius: 8px;
    padding: 12px 16px;
    display: flex; align-items: center; gap: 12px;
    font-size: 13px; font-weight: 600;
}}
.cf-sadp {{
    background: #E8F0EA;
    border: 1px solid #9EBEAB;
    color: #3D5D49;
}}
.cf-sadn {{
    background: #F2DCDC;
    border: 1px solid #C49A9A;
    color: #6D4040;
}}
.cf-sadp .cf-n {{ color: #3D5D49; }}
.cf-sadn .cf-n {{ color: #6D4040; }}

/* Fork dentro de cada bloque (de SAD+/SAD- a outcomes) */
.cf-fork {{
    display: flex; justify-content: center;
    padding: 2px 0;
}}

/* Fila de outcomes (supervivientes / fallecidos) */
.cf-outcomes-row {{
    display: flex; gap: 10px;
    margin-top: 4px;
}}
.cf-out {{
    flex: 1;
    background: {SURFACE};
    border-radius: 8px;
    padding: 10px 12px;
    display: flex; align-items: center; gap: 10px;
}}
.cf-out-surv {{
    border: 1px solid #9EBEAB;
    color: #4F7C66;
}}
.cf-out-dead {{
    border: 1px solid #C49A9A;
    color: #8B5050;
}}
.cf-out-num {{
    width: 24px; height: 24px; border-radius: 50%;
    color: white;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    flex-shrink: 0;
}}
.cf-out-surv .cf-out-num {{ background: #7FA98C; }}
.cf-out-dead .cf-out-num {{ background: #C57878; }}
.cf-out-info {{ display: flex; flex-direction: column; line-height: 1.2; }}
.cf-out-label {{ font-size: 11px; font-weight: 500; }}
.cf-out-n {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700; font-size: 12px;
    color: {TEXT_PRIMARY};
    margin-top: 2px;
}}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TRADUCTOR Y MODELOS
# ═══════════════════════════════════════════════════════════════
ROOT      = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"

@st.cache_resource
def get_translator():
    try:
        from deep_translator import GoogleTranslator
        # No hacer traducción de prueba; solo crear el objeto
        return GoogleTranslator(source="es", target="en")
    except Exception:
        return None

TRANSLATOR = get_translator()

@st.cache_data(show_spinner=False)
def translate_es_en(text: str) -> str:
    if not text or not text.strip(): return ""
    if TRANSLATOR is None: return ""
    try:
        return TRANSLATOR.translate(text[:4500])
    except Exception:
        return ""

@st.cache_resource
def load_models():
    """Carga el Ensemble_top3 real + preprocesadores desde artifacts/app_model_v21/."""
    models = {"modo": "demo", "modelos_cargados": []}
    app_model_dir = ARTIFACTS / "app_model_v21"
    if app_model_dir.exists():
        try:
            models["ensemble"] = joblib.load(app_model_dir / "ensemble_top3.joblib")
            models["mice"]     = joblib.load(app_model_dir / "mice.joblib")
            models["scaler"]   = joblib.load(app_model_dir / "scaler.joblib")
            meta = json.loads((app_model_dir / "meta.json").read_text())
            models["meta"]       = meta
            models["features"]   = meta.get("features_all", ALL_COLS)
            models["threshold"]  = meta.get("threshold_youden", MODEL_META["umbral"])
            models["modo"]       = "real"
            models["modelos_cargados"].append("Ensemble_top3 v21")
        except Exception as e:
            models["error"] = str(e)
    return models

MODELS    = load_models()
DEMO      = MODELS["modo"] == "demo"
FINAL_THR = MODELS.get("threshold", MODEL_META["umbral"])
FEATURES  = MODELS.get("features", ALL_COLS)


# ═══════════════════════════════════════════════════════════════
# LÓGICA DE PREDICCIÓN
# ═══════════════════════════════════════════════════════════════
def _simulate_fallback(vals):
    """Fallback heurístico si el modelo real no está disponible (modo DEMO)."""
    risk = 0.08
    if vals.get("albumin", 3)         <= 2.5: risk += 0.18
    if vals.get("total_bilirubin", 1) >= 2.0: risk += 0.12
    if vals.get("hemoglobin", 10)     <= 8.0: risk += 0.10
    if vals.get("sofa_score", 5)      >= 8:   risk += 0.22
    if vals.get("lactate", 2)         >= 4.0: risk += 0.20
    if vals.get("creatinine", 1.2)    >= 3.0: risk += 0.10
    risk += max(0, (vals.get("age", 65) - 65) * 0.004)
    return min(0.97, max(0.02, risk))


def predict_ensemble(vals: dict) -> dict:
    """Pipeline completo 4.3.1 (6 pasos) sobre los valores del formulario.

    Returns:
        dict con keys:
          'prob'        : probabilidad ensemble final
          'per_member'  : dict {XGB, LGB, ExtraTrees} con probabilidad de cada base learner
          'threshold'   : umbral Youden aplicado
          'verdict'     : 1 si prob >= threshold, 0 si no
          'derived'     : dict con las 8 features derivadas calculadas
          'mode'        : 'real' o 'demo'
    """
    if DEMO:
        prob = _simulate_fallback(vals)
        return {
            "prob": prob, "per_member": {}, "threshold": FINAL_THR,
            "verdict": int(prob >= FINAL_THR), "derived": {}, "mode": "demo",
        }

    # Paso 3 PRIMERO: feature engineering desde inputs estructurales del formulario
    row_struct = pd.DataFrame([{f: vals.get(f, np.nan) for f in FEATS_STRUCT}])
    row_fe = add_clinical_features(row_struct)

    # Asegurar que todas las features estén presentes en el orden canónico
    X_row = row_fe[FEATURES].astype(float).values  # shape (1, 40)

    # Paso 1: MICE imputation
    X_imp = MODELS["mice"].transform(X_row)

    # Paso 2: StandardScaler
    X_scl = MODELS["scaler"].transform(X_imp)

    # Pasos 4 y 5: predicción individual + ensemble por promedio igual
    ensemble = MODELS["ensemble"]
    per_member = {name: float(p[0]) for name, p in ensemble.predict_proba_per_member(X_scl).items()}
    prob = float(ensemble.predict_proba(X_scl)[0, 1])

    # Paso 6: clasificación con umbral Youden
    verdict = int(prob >= FINAL_THR)

    # Extraer derivadas para mostrar al usuario
    derived = {col: float(row_fe[col].iloc[0]) for col in FE_COLS if col in row_fe.columns}

    return {
        "prob": prob,
        "per_member": per_member,
        "threshold": FINAL_THR,
        "verdict": verdict,
        "derived": derived,
        "mode": "real",
    }

NOTE_KEYWORDS = {
    "severe": {
        "septic shock": 0.10, "vasopressor": 0.08, "norepinephrine": 0.07,
        "intubat": 0.07, "mechanical vent": 0.08, "ards": 0.09,
        "multi-organ": 0.10, "multiorgan": 0.10, "anuri": 0.07,
        "dialysis": 0.06, "cardiac arrest": 0.12, "coma": 0.08,
        "gcs 3": 0.10, "gcs 4": 0.08, "lactic acidosis": 0.06,
        "refractory": 0.05, "disseminated": 0.07, "dic": 0.08,
        "pneumonia": 0.04, "vasopressin": 0.06,
    },
    "moderate": {
        "hypotension": 0.04, "tachycardia": 0.02, "fever": 0.02,
        "altered conscious": 0.04, "oliguria": 0.03, "encephalopath": 0.04,
        "respiratory fail": 0.05, "hypoxemia": 0.04, "thrombocyt": 0.03,
    },
    "protective": {
        "stable": -0.04, "improving": -0.05, "extubat": -0.06,
        "responding to treat": -0.05, "weaning": -0.03, "afebrile": -0.02,
        "alert and oriented": -0.04, "tolerating": -0.03,
        "hemodynamically compensated": -0.05,
    },
}

def detect_keywords(english_note: str):
    if not english_note: return 0.0, []
    note = english_note.lower()
    delta = 0.0; detected = []
    for severity, kws in NOTE_KEYWORDS.items():
        for kw, val in kws.items():
            if kw in note:
                delta += val
                detected.append((kw, severity))
    return delta, detected

def predict_with_note(vals, english_note: str):
    """Modelo A + Notas: predicción ensemble base + ajuste textual por nota."""
    base = predict_ensemble(vals)["prob"]
    delta, _ = detect_keywords(english_note)
    if delta == 0:
        return base, 0.0
    logit = math.log(base / (1 - base + 1e-9))
    logit += delta * 4.0
    adjusted = 1.0 / (1.0 + math.exp(-logit))
    return float(max(0.02, min(0.97, adjusted))), delta

# Alias para compatibilidad con código existente
def predict_b2(vals):
    return predict_ensemble(vals)["prob"]


# ═══════════════════════════════════════════════════════════════
# SIDEBAR · LANGUAGE FIRST
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo institucional · controlado por HTML para alineación precisa con título
    st.markdown(f"""
    <div style="text-align: center; padding: 0 0 14px;">
      <img src="{LOGO_FACULTAD}" alt="UPCH" style="max-width: 200px; width: 100%; height: auto; display: block; margin: 0 auto;">
    </div>
    """, unsafe_allow_html=True)

    # Idioma — primero, para que LANG esté disponible para el resto.
    # El widget gestiona su propio estado con key="lang_sel"; NO se usa
    # index= ligado a session_state (eso provocaba el bug de desfase).
    LANG_OPTS = ["ES", "EN"]

    st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:11px; font-weight:700; color:{TEXT_SECONDARY}; '
        f'text-transform:uppercase; letter-spacing:0.8px; margin: 0 0 6px;">'
        f'{"Idioma · Language"}</div>',
        unsafe_allow_html=True
    )
    lang_choice = st.selectbox(
        "lang_selector", LANG_OPTS,
        key="lang_sel",
        label_visibility="collapsed",
        format_func=lambda x: "🇵🇪 ES · Español" if x == "ES"
        else "🇬🇧 EN · English",
    )
    LANG = "es" if lang_choice == "ES" else "en"

def t(key: str) -> str:
    """Devuelve la traducción del key según LANG."""
    return TR.get(LANG, TR["es"]).get(key, key)

# Ahora seguir construyendo el sidebar con LANG activo
with st.sidebar:
    st.markdown(f"""
    <div class="sb-meta-card">
      <div class="lbl">{t("sb_auc")}</div>
      <div class="val">{MODEL_META["auc"]:.3f}</div>
      <div class="ic">IC95% [{MODEL_META["auc_ic_lo"]:.3f}, {MODEL_META["auc_ic_hi"]:.3f}]</div>
    </div>
    """, unsafe_allow_html=True)

    if TRANSLATOR is None and LANG == "es":
        st.warning(t("sb_translator_warn"))

    fecha = MODEL_META["fecha_es"] if LANG == "es" else MODEL_META["fecha_en"]
    st.markdown(f"""
    <div style="padding: 6px 0;">
      <div class="sb-row"><span class="lbl">{t("sb_model")}</span><span class="val">Ensemble_top3</span></div>
      <div class="sb-row"><span class="lbl">{t("sb_cohort")}</span><span class="val">{MODEL_META["n_train"]:,}</span></div>
      <div class="sb-row"><span class="lbl">{t("sb_prev")}</span><span class="val">{MODEL_META["prevalencia"]*100:.1f}%</span></div>
      <div class="sb-row"><span class="lbl">{t("sb_sens")}</span><span class="val">{MODEL_META["sens"]*100:.1f}%</span></div>
      <div class="sb-row"><span class="lbl">{t("sb_esp")}</span><span class="val">{MODEL_META["esp"]*100:.1f}%</span></div>
      <div class="sb-row"><span class="lbl">{t("sb_hlp")}</span><span class="val">{MODEL_META["hl_p"]:.3f}</span></div>
      <div class="sb-row"><span class="lbl">{t("sb_threshold")}</span><span class="val">{MODEL_META["umbral"]:.3f}</span></div>
      <div class="sb-row"><span class="lbl">{t("sb_updated")}</span><span class="val">{fecha}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("UPCH 2026")

    # ── Autores de la tesis ──────────────────────────────
    st.markdown(f"""
    <div style="margin-top: 14px; padding: 12px 14px;
                background: {SURFACE_TINT}; border: 1px solid {BORDER};
                border-radius: 10px;">
      <div style="font-size: 10px; color: {TEXT_SECONDARY}; font-weight: 700;
                  text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 8px;">
        Autores · Authors
      </div>
      <div style="font-size: 11px; color: {TEXT_PRIMARY}; line-height: 1.55;">
        <div style="margin-bottom: 4px;">▸ Huarauya Fabian, Josué Eduardo</div>
        <div>▸ Oviedo Chahua, Gilmar Rony</div>
      </div>
      <div style="font-size: 10px; color: {TEXT_MUTED}; margin-top: 8px; font-style: italic;">
        Tesis · Facultad de Ciencias e Ingeniería
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HELPERS dependientes de LANG
# ═══════════════════════════════════════════════════════════════
def risk_tier(p):
    if p < 0.10:  return (t("risk_low"),  "low",  RISK_LOW)
    if p < 0.30:  return (t("risk_mod"),  "mod",  RISK_MOD)
    if p < 0.60:  return (t("risk_high"), "high", RISK_HIGH)
    return            (t("risk_crit"), "crit", RISK_CRIT)

ALERTAS_RULES = [
    ("albumin","<=",2.5,"red","alert_alb"),
    ("total_bilirubin",">=",2.0,"yellow","alert_bili"),
    ("hemoglobin","<=",8.0,"yellow","alert_hb"),
    ("sofa_score",">=",8,"red","alert_sofa"),
    ("lactate",">=",4.0,"red","alert_lact"),
    ("creatinine",">=",3.0,"yellow","alert_creat"),
    ("aptt",">=",100,"yellow","alert_aptt"),
]

def get_alertas(vals):
    out = []
    for var, op, thr, css, key in ALERTAS_RULES:
        v = vals.get(var, float("nan"))
        try:
            if op == "<=" and float(v) <= thr: out.append((css, t(key)))
            if op == ">=" and float(v) >= thr: out.append((css, t(key)))
        except: pass
    return out

def result_card_html(name, tag_class, prob, threshold, has_note=False, detected=None):
    """Card de resultado.

    `prob` es la probabilidad de muerte a 28 días (lo que sale del modelo).
    El número grande mostrado es coherente con el verdict:
      - Si prob < umbral → predicción VIVE → muestra P(supervivencia) grande
      - Si prob >= umbral → predicción MUERE → muestra P(muerte) grande
    """
    tier_name, tier_cls, tier_color = risk_tier(prob)
    vive = prob < threshold

    if vive:
        big_pct  = (1 - prob) * 100      # probabilidad de supervivencia
        sub_pct  = prob * 100            # riesgo de muerte
        sub_lbl  = t("death_risk_label")
        emoji    = "😊"
        verdict_word = t("verdict_vive")
        verdict_cls  = "verdict-vive"
    else:
        big_pct  = prob * 100            # probabilidad de muerte
        sub_pct  = (1 - prob) * 100      # probabilidad de supervivencia
        sub_lbl  = t("survival_prob_label")
        emoji    = "💀"
        verdict_word = t("verdict_muere")
        verdict_cls  = "verdict-muere"

    kw_html = ""
    if has_note and detected:
        pills = []
        for kw, sev in detected[:8]:
            cls = "severe" if sev == "severe" else "protect" if sev == "protective" else ""
            pills.append(f'<span class="kw-pill {cls}">{kw}</span>')
        kw_html = f'<div class="kw-pills">{"".join(pills)}</div>'

    if tag_class == "a":
        foot = f"{t('card_a_foot')} {threshold:.3f}"
    else:
        n_sig = len(detected) if detected else 0
        info = f"{n_sig} {t('card_b_signals')}" if has_note else t('card_b_no_signals')
        foot = f"{info} {t('card_b_foot_sep')} {threshold:.3f}"

    # Subtítulo con la probabilidad complementaria (informativo)
    sub_html = f"""
    <div style="font-size: 11px; color: {TEXT_SECONDARY}; margin-top: 6px;
                font-family: 'Inter', sans-serif;">
      {sub_lbl}: <b style="font-family:'JetBrains Mono', monospace; color: {TEXT_PRIMARY};">{sub_pct:.1f}%</b>
    </div>
    """

    return f"""
    <div class="result-card modelo-{tag_class}">
      <span class="result-card-tag tag-{tag_class}">{name}</span>
      <div class="result-card-icon">{emoji}</div>
      <div class="result-card-prob" style="color:{tier_color};">{big_pct:.1f}<span class="unit">%</span></div>
      <div class="result-card-verdict {verdict_cls}">{verdict_word}</div>
      <span class="result-card-tier tier-{tier_cls}">{t("risk_prefix")} {tier_name}</span>
      {sub_html}
      {kw_html}
      <div class="result-card-foot">{foot}</div>
    </div>
    """

def result_card_empty(name, tag_class, msg):
    return f"""
    <div class="result-card modelo-{tag_class}" style="opacity: 0.55;">
      <span class="result-card-tag tag-{tag_class}">{name}</span>
      <div style="margin-top: 40px;">
        <div style="font-size:42px; opacity:0.4;">◯</div>
        <div style="font-size:13px; color:{TEXT_SECONDARY}; font-weight:600; margin-top:12px;">{msg}</div>
      </div>
    </div>
    """


# ═══════════════════════════════════════════════════════════════
# TÍTULO DE PÁGINA
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="page-title">
  <h1>{t("page_title")}</h1>
  <div class="page-sub">
    <span class="acc">UPCH</span> · {t("page_sub_1")} ·
    {t("page_sub_2")} ({MODEL_META["version"]}) · n = {MODEL_META["n_train"]:,} {t("page_sub_n")}
    <span class="auc">AUC <span class="gold">{MODEL_META["auc"]:.3f}</span></span>
  </div>
</div>

<div style="background: #F4ECDB; border: 1px solid #DCC4A0; border-radius: 8px;
            padding: 10px 14px; margin-bottom: 16px;
            font-size: 12px; color: #6B5535; line-height: 1.5;">
  <b style="color: #5C4A2D; letter-spacing: 0.3px;">⚠ AVISO ACADÉMICO ·</b>
  Herramienta de investigación · Fase 1 / Estadio 1 según marco OMS de IA en salud ·
  <b>No usar como apoyo único para decisión clínica autónoma</b>. Validación externa pendiente.
  El juicio clínico del médico tratante prevalece siempre.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    t("tab1"), t("tab2"), t("tab3"), t("tab4"), t("tab5"),
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 · FORMULARIO + RESULTADOS
# ═══════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1, 1.05], gap="medium")

    with col_form:
        st.markdown(f'<div class="sec-label">{t("patient_data")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:13px; color:{TEXT_SECONDARY}; margin-bottom: 14px; line-height: 1.5;">'
                    f'Las <b>5 variables centrales</b> del modelo de Zhang et al. (2024) son obligatorias. '
                    f'Las demás son <b>opcionales</b>: lo que no se ingrese es imputado automáticamente por MICE.</div>',
                    unsafe_allow_html=True)

        # ═══ 5 VARIABLES CENTRALES (siempre visibles, obligatorias) ═══
        icu    = st.number_input(t("input_icu"),  0.0, 90.0, 5.0, 0.5)
        hb     = st.number_input(t("input_hb"),   3.0, 20.0, 10.0, 0.1)
        alb    = st.number_input(t("input_alb"),  0.5, 6.0, 3.0, 0.05)
        aptt_v = st.number_input(t("input_aptt"), 10.0, 200.0, 35.0, 1.0)
        bili   = st.number_input(t("input_bili"), 0.0, 40.0, 1.5, 0.1)

        # ═══ VARIABLES ADICIONALES (expander único, sub-grupos por checkbox) ═══
        # _va: helper de traducción local (ES/EN) para esta sección.
        _va = (lambda es, en: en if LANG == "en" else es)
        with st.expander(
                _va("➕ Variables adicionales (opcional · MICE imputa lo no ingresado)",
                    "➕ Additional variables (optional · MICE imputes what is not entered)"),
                expanded=False):
            st.caption(_va(
                "Activa cada sub-grupo para ingresar datos. Las variables no activadas "
                "son imputadas automáticamente por el MICE entrenado sobre la cohorte.",
                "Enable each sub-group to enter data. Variables not enabled are imputed "
                "automatically by the MICE imputer trained on the cohort."))

            st.markdown(f"""
            <div style="margin: 8px 0 6px 0; padding: 10px 14px; background: {SURFACE_TINT};
                        border-left: 3px solid {UPCH_GOLD}; border-radius: 0 8px 8px 0;
                        font-size: 11.5px; color: {TEXT_SECONDARY}; line-height: 1.6;">
              <b style="color: {UPCH_PRIMARY};">{_va(
                  '¿Por qué se pueden dejar variables en blanco?',
                  'Why can variables be left blank?')}</b><br>
              {_va(
                  'El imputador <b>MICE</b> (Multivariate Imputation by Chained Equations) '
                  'fue entrenado sobre los 12&nbsp;564 pacientes de la cohorte. Estima cada '
                  'valor faltante a partir de las variables que sí se ingresan, aprovechando '
                  'las correlaciones clínicas entre ellas. Por eso el modelo entrega una '
                  'predicción válida aun con datos incompletos —situación habitual en la '
                  'práctica—; cuantas más variables reales se ingresen, más precisa es la '
                  'estimación y menor la incertidumbre del resultado.',
                  'The <b>MICE</b> imputer (Multivariate Imputation by Chained Equations) '
                  'was trained on the 12&nbsp;564 patients of the cohort. It estimates each '
                  'missing value from the variables that are entered, exploiting the '
                  'clinical correlations among them. That is why the model yields a valid '
                  'prediction even with incomplete data —a common situation in practice—; '
                  'the more real variables entered, the more precise the estimate and the '
                  'lower the uncertainty of the result.')}
            </div>
            """, unsafe_allow_html=True)

            # ─── 1. Demográficas y severidad
            if st.checkbox(_va("**Demográficas y severidad** · edad, SOFA score",
                               "**Demographics & severity** · age, SOFA score"), key="g_demo"):
                cd1, cd2 = st.columns(2)
                with cd1: age  = st.number_input(_va("Edad (años)", "Age (years)"), 18, 110, 65, 1, key="i_age")
                with cd2: sofa = st.number_input("SOFA score (0–24)", 0, 24, 5, 1, key="i_sofa")
            else:
                age = sofa = np.nan

            # ─── 2. Signos vitales
            if st.checkbox(_va("**Signos vitales** · HR, SBP, Mean BP, RR, SpO₂",
                               "**Vital signs** · HR, SBP, Mean BP, RR, SpO₂"), key="g_vital"):
                cv1, cv2, cv3 = st.columns(3)
                with cv1:
                    hr  = st.number_input(_va("HR (lpm)", "HR (bpm)"), 30.0, 220.0, 95.0, 1.0, key="i_hr")
                    rr  = st.number_input(_va("Resp. rate (rpm)", "Resp. rate (/min)"), 5.0, 60.0, 20.0, 1.0, key="i_rr")
                with cv2:
                    sbp  = st.number_input("SBP (mmHg)", 40.0, 250.0, 110.0, 1.0, key="i_sbp")
                    spo2 = st.number_input("SpO₂ (%)", 50.0, 100.0, 95.0, 0.5, key="i_spo2")
                with cv3:
                    mbp = st.number_input("Mean BP (mmHg)", 30.0, 180.0, 75.0, 1.0, key="i_mbp")
            else:
                hr = sbp = mbp = rr = spo2 = np.nan

            # ─── 3. Hematología
            if st.checkbox(_va("**Hematología** · hematocrito, plaquetas, WBC",
                               "**Hematology** · hematocrit, platelets, WBC"), key="g_hema"):
                ch1, ch2, ch3 = st.columns(3)
                with ch1: hct   = st.number_input(_va("Hematocrito (%)", "Hematocrit (%)"), 10.0, 60.0, 30.0, 0.5, key="i_hct")
                with ch2: plt_v = st.number_input(_va("Plaquetas (×10³/μL)", "Platelets (×10³/μL)"), 5.0, 1000.0, 150.0, 5.0, key="i_plt")
                with ch3: wbc   = st.number_input("WBC (×10⁹/L)", 0.5, 80.0, 11.0, 0.1, key="i_wbc")
            else:
                hct = plt_v = wbc = np.nan

            # ─── 4. Coagulación (aPTT ya está como variable central)
            if st.checkbox(_va("**Coagulación** · PT, INR",
                               "**Coagulation** · PT, INR"), key="g_coag"):
                cc1, cc2 = st.columns(2)
                with cc1: pt  = st.number_input(_va("PT (seg)", "PT (sec)"), 8.0, 100.0, 14.0, 0.5, key="i_pt")
                with cc2: inr = st.number_input("INR", 0.5, 15.0, 1.3, 0.1, key="i_inr")
            else:
                pt = inr = np.nan

            # ─── 5. Bioquímica hepática
            if st.checkbox(_va("**Bioquímica hepática** · bili directa, ALT, AST, ALP",
                               "**Liver chemistry** · direct bili, ALT, AST, ALP"), key="g_hep"):
                chh1, chh2, chh3, chh4 = st.columns(4)
                with chh1: bili_d = st.number_input(_va("Bili D (mg/dL)", "Direct bili (mg/dL)"), 0.0, 30.0, 0.4, 0.1, key="i_bilid")
                with chh2: alt    = st.number_input("ALT (U/L)", 5.0, 2000.0, 30.0, 1.0, key="i_alt")
                with chh3: ast    = st.number_input("AST (U/L)", 5.0, 2000.0, 40.0, 1.0, key="i_ast")
                with chh4: alp    = st.number_input("ALP (U/L)", 20.0, 1000.0, 100.0, 5.0, key="i_alp")
            else:
                bili_d = alt = ast = alp = np.nan

            # ─── 6. Función renal
            if st.checkbox(_va("**Función renal** · creatinina, BUN",
                               "**Renal function** · creatinine, BUN"), key="g_renal"):
                cr1, cr2 = st.columns(2)
                with cr1: creat = st.number_input(_va("Creatinina (mg/dL)", "Creatinine (mg/dL)"), 0.1, 20.0, 1.2, 0.05, key="i_creat")
                with cr2: bun   = st.number_input("BUN (mg/dL)", 2.0, 200.0, 25.0, 1.0, key="i_bun")
            else:
                creat = bun = np.nan

            # ─── 7. Electrolitos
            if st.checkbox(_va("**Electrolitos** · Na, K, Cl, HCO₃, Ca, anion gap",
                               "**Electrolytes** · Na, K, Cl, HCO₃, Ca, anion gap"), key="g_elec"):
                ce1, ce2, ce3 = st.columns(3)
                with ce1:
                    na = st.number_input(_va("Sodio (mEq/L)", "Sodium (mEq/L)"), 110.0, 170.0, 138.0, 1.0, key="i_na")
                    k  = st.number_input(_va("Potasio (mEq/L)", "Potassium (mEq/L)"), 2.0, 8.0, 4.0, 0.1, key="i_k")
                with ce2:
                    cl  = st.number_input(_va("Cloruro (mEq/L)", "Chloride (mEq/L)"), 80.0, 130.0, 104.0, 1.0, key="i_cl")
                    hco = st.number_input(_va("Bicarbonato (mEq/L)", "Bicarbonate (mEq/L)"), 5.0, 50.0, 23.0, 0.5, key="i_hco")
                with ce3:
                    ca_v = st.number_input(_va("Calcio (mg/dL)", "Calcium (mg/dL)"), 5.0, 14.0, 8.5, 0.1, key="i_ca")
                    ag   = st.number_input("Anion gap (mEq/L)", 0.0, 50.0, 14.0, 0.5, key="i_ag")
            else:
                na = k = cl = hco = ca_v = ag = np.nan

            # ─── 8. Metabolismo
            if st.checkbox(_va("**Metabolismo** · glucosa, lactato, pH",
                               "**Metabolism** · glucose, lactate, pH"), key="g_metab"):
                cm1, cm2, cm3 = st.columns(3)
                with cm1: glu  = st.number_input(_va("Glucosa (mg/dL)", "Glucose (mg/dL)"), 30.0, 800.0, 130.0, 5.0, key="i_glu")
                with cm2: lact = st.number_input(_va("Lactato (mmol/L)", "Lactate (mmol/L)"), 0.1, 25.0, 2.0, 0.1, key="i_lact")
                with cm3: ph   = st.number_input("pH", 6.8, 7.7, 7.35, 0.01, key="i_ph")
            else:
                glu = lact = ph = np.nan

            # Info sobre las 8 derivadas (calculadas, no ingresadas)
            st.markdown(f"""
            <div style="margin-top: 12px; padding: 10px 14px; background: {SURFACE_TINT};
                        border-left: 3px solid {UPCH_GOLD}; border-radius: 0 8px 8px 0;
                        font-size: 11px; color: {TEXT_SECONDARY}; line-height: 1.55;">
              <b style="color: {UPCH_PRIMARY};">{_va('📊 8 variables derivadas (automáticas)', '📊 8 derived variables (automatic)')}</b> ·
              fe_shock_index (HR/SBP) · fe_bun_creat · fe_age_sofa · fe_hgb_lact (Hb/Lact) ·
              fe_liver (INR×Bili) · fe_bun_alb · fe_ag_alb ({_va('AG corregido', 'corrected AG')}) · fe_coag (Plaq/INR).
              {_va('Se calculan a partir de las variables anteriores; lo que falta lo imputa MICE.', 'Computed from the variables above; whatever is missing is imputed by MICE.')}
            </div>
            """, unsafe_allow_html=True)

        # ═══ Indicador "X/32 ingresadas · Y imputadas por MICE" ═══
        _struct_check = {
            "icu": icu, "hb": hb, "alb": alb, "aptt": aptt_v, "bili": bili,  # 5 centrales
            "age": age, "sofa": sofa,
            "hr": hr, "sbp": sbp, "mbp": mbp, "rr": rr, "spo2": spo2,
            "hct": hct, "plt": plt_v, "wbc": wbc,
            "pt": pt, "inr": inr,
            "bili_d": bili_d, "alt": alt, "ast": ast, "alp": alp,
            "creat": creat, "bun": bun,
            "na": na, "k": k, "cl": cl, "hco": hco, "ca_v": ca_v, "ag": ag,
            "glu": glu, "lact": lact, "ph": ph,
        }
        _n_entered = sum(1 for v in _struct_check.values()
                         if not (isinstance(v, float) and np.isnan(v)))
        _n_total = len(_struct_check)  # 32
        _n_imputed = _n_total - _n_entered
        _pct = _n_entered / _n_total
        _ind_color = RISK_LOW if _pct >= 0.6 else "#8C6235" if _pct >= 0.3 else UPCH_SECONDARY

        st.markdown(f"""
        <div style="background: {SURFACE_TINT}; border: 1px solid {BORDER}; border-radius: 8px;
                    padding: 10px 14px; margin: 12px 0; font-size: 12px;
                    display: flex; justify-content: space-between; align-items: center;">
          <span>
            <span style="color: {TEXT_SECONDARY};">{_va("Variables ingresadas:", "Variables entered:")}</span>
            <span style="font-family:'JetBrains Mono', monospace; font-weight:700;
                         color: {_ind_color}; margin-left: 6px;">{_n_entered}/{_n_total}</span>
          </span>
          <span style="color: {TEXT_MUTED};">
            {_n_imputed} {_va("serán imputadas por MICE", "will be imputed by MICE")}
          </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sec-label">{t("note_section")}</div>', unsafe_allow_html=True)

        # Nota clínica — directa en el idioma activo
        nota_en = ""
        nota_input = st.text_area(
            t("note_label_active"),
            height=110,
            placeholder=t("note_placeholder"),
            key=f"nota_input_{LANG}",
        )

        if LANG == "es":
            if nota_input and nota_input.strip():
                with st.spinner(t("note_translating")):
                    nota_en = translate_es_en(nota_input)
                if nota_en:
                    st.markdown(f"""
                    <div class="translate-box">
                      <div class="translate-label">{t("translation_label")}</div>
                      {nota_en}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(t("translation_failed"))
        else:
            nota_en = nota_input

        st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)
        btn = st.button(t("btn_calculate"), type="primary", width="stretch")

    with col_result:
        st.markdown(f'<div class="sec-label">{t("result_section")}</div>', unsafe_allow_html=True)

        # Mapeo completo de 32 variables estructurales (ALL_COLS names del modelo)
        vals = {
            # G1 demográficas y severidad
            "age": age, "icu_los_days": icu, "sofa_score": sofa,
            # G2 signos vitales
            "heart_rate": hr, "sbp": sbp, "mean_bp": mbp,
            "resp_rate": rr, "spo2": spo2,
            # G3 hematología + bioquímica
            "hemoglobin": hb, "hematocrit": hct, "platelets": plt_v, "wbc": wbc,
            "albumin": alb, "total_bilirubin": bili, "bilirubin_direct": bili_d,
            "alt": alt, "ast": ast, "alp": alp,
            "creatinine": creat, "bun": bun,
            "sodium": na, "potassium": k, "chloride": cl, "bicarbonate": hco,
            "calcium": ca_v, "aniongap": ag, "glucose": glu,
            "lactate": lact, "ph": ph,
            # G4 coagulación
            "aptt": aptt_v, "pt": pt, "inr": inr,
        }

        if btn:
            ensemble_result = predict_ensemble(vals)
            st.session_state["last"] = {
                "vals":     vals.copy(),
                "nota_en":  nota_en,
                "result":   ensemble_result,
                "p_A":      ensemble_result["prob"],
                "p_B":      predict_with_note(vals, nota_en),
                "alertas":  get_alertas(vals),
            }

        if "last" in st.session_state:
            r = st.session_state["last"]
            p_A = r["p_A"]
            p_B, _ = r["p_B"]
            delta_pp = (p_B - p_A) * 100
            has_note = bool(r["nota_en"])
            _, detected = detect_keywords(r["nota_en"]) if has_note else (0.0, [])

            cA, cB = st.columns(2, gap="small")
            with cA:
                st.markdown(result_card_html(
                    t("model_a_name"), "a", p_A, FINAL_THR, has_note=False, detected=None
                ), unsafe_allow_html=True)
            with cB:
                if has_note:
                    st.markdown(result_card_html(
                        t("model_b_name"), "b", p_B, FINAL_THR, has_note=True, detected=detected
                    ), unsafe_allow_html=True)
                else:
                    st.markdown(result_card_empty(
                        t("model_b_name"), "b", t("model_b_empty")
                    ), unsafe_allow_html=True)

            if has_note:
                sign = "+" if delta_pp > 0 else ""
                cls_d = "delta-up" if delta_pp > 0.5 else "delta-down" if delta_pp < -0.5 else "delta-zero"
                arrow = "▲" if delta_pp > 0.5 else "▼" if delta_pp < -0.5 else "≈"
                interp = (t("delta_higher") if delta_pp > 1
                          else t("delta_lower") if delta_pp < -1
                          else t("delta_neutral"))
                st.markdown(f"""
                <div class="delta-card">
                  <div class="delta-label">{t("delta_label")}</div>
                  <div class="delta-value {cls_d}">{arrow} {sign}{delta_pp:.1f} pp</div>
                  <div style="font-size: 11px; color: {TEXT_MUTED}; margin-top: 4px;">{interp}</div>
                </div>
                """, unsafe_allow_html=True)

            if r["alertas"]:
                st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="sec-label">{t("alerts_section")}</div>', unsafe_allow_html=True)
                for css, msg in r["alertas"][:5]:
                    color = RISK_HIGH if css == "red" else "#8C6235"
                    bg    = "#F2E2E2" if css == "red" else "#F4ECDB"
                    st.markdown(f"""
                    <div style="background: {bg}; border-left: 4px solid {color}; padding: 8px 14px;
                                border-radius: 0 8px 8px 0; margin: 4px 0; font-size: 13px; color: {color};">
                      ▸ {msg}
                    </div>
                    """, unsafe_allow_html=True)

            # ── Desglose por miembro del ensemble (XGB, LGB, ExtraTrees) ──
            res = r["result"]
            if res.get("mode") == "real" and res.get("per_member"):
                st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
                st.markdown('<div class="sec-label">🧩 Desglose del ensemble · probabilidad por base learner</div>',
                            unsafe_allow_html=True)
                pm = res["per_member"]
                cols_pm = st.columns(len(pm))
                for col_pm, (name, p) in zip(cols_pm, pm.items()):
                    tier_color = RISK_HIGH if p >= FINAL_THR else RISK_LOW
                    col_pm.markdown(f"""
                    <div style="background: white; border: 1px solid {BORDER}; border-radius: 8px;
                                padding: 10px 14px; text-align: center;">
                      <div style="font-size: 10px; color: {TEXT_SECONDARY}; font-weight: 700;
                                  text-transform: uppercase; letter-spacing: 0.5px;">{name}</div>
                      <div style="font-family: 'JetBrains Mono', monospace; font-size: 20px;
                                  font-weight: 700; color: {tier_color}; margin-top: 4px;">
                        {p*100:.1f}<span style="font-size:13px; opacity:0.7;">%</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown(f"""
                <div style="font-size: 11px; color: {TEXT_MUTED}; margin-top: 8px; text-align: center;">
                  Ensemble final = promedio igual de los 3 base learners cascade-calibrados (Platt + Isotonic).
                </div>
                """, unsafe_allow_html=True)

            # ── Descarga JSON del resultado ──
            st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)
            export = {
                "timestamp":   pd.Timestamp.now().isoformat(),
                "model":       MODELS.get("meta", {}).get("model_name", "Ensemble_top3"),
                "version":     MODEL_META["version"],
                "inputs":      {k: float(v) for k, v in r["vals"].items()},
                "derived_features": res.get("derived", {}),
                "predictions": {
                    "modelo_A_sin_nota": round(p_A, 4),
                    "modelo_B_con_nota": round(p_B, 4) if r["nota_en"] else None,
                    "delta_pp":          round(delta_pp, 2) if r["nota_en"] else None,
                    "per_member":        {k: round(v, 4) for k, v in res.get("per_member", {}).items()},
                    "threshold_youden":  FINAL_THR,
                    "verdict_A":         "fallece" if p_A >= FINAL_THR else "sobrevive",
                },
                "clinical_note_en":  r["nota_en"],
                "alertas":           [msg for _, msg in r["alertas"]],
                "model_metrics_validation": {
                    "auc":   MODEL_META["auc"],
                    "ic95":  [MODEL_META["auc_ic_lo"], MODEL_META["auc_ic_hi"]],
                    "sens":  MODEL_META["sens"],
                    "esp":   MODEL_META["esp"],
                    "hl_p":  MODEL_META["hl_p"],
                    "brier": MODEL_META["brier"],
                },
                "disclaimer": "Herramienta de investigación. No usar como apoyo único para decisión clínica autónoma.",
            }
            st.download_button(
                "⬇  Descargar resultado (JSON)",
                data=json.dumps(export, indent=2, ensure_ascii=False),
                file_name=f"prediccion_sepsis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        else:
            st.markdown(f"""
            <div class="empty">
              <div class="e-icon">◀</div>
              <div class="e-text">{t("empty_text")}</div>
              <div class="e-sub">{t("empty_sub")}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 · RENDIMIENTO (gráficas al 80%)
# ═══════════════════════════════════════════════════════════════
with tab2:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t("tab2_metric_auc"), f"{MODEL_META['auc']:.3f}",
              f"IC95 [{MODEL_META['auc_ic_lo']:.3f}, {MODEL_META['auc_ic_hi']:.3f}]")
    k2.metric(t("tab2_metric_sens"), f"{MODEL_META['sens']*100:.1f}%")
    k3.metric(t("tab2_metric_esp"),  f"{MODEL_META['esp']*100:.1f}%")
    k4.metric(t("tab2_metric_hlp"),  f"{MODEL_META['hl_p']:.3f}",
              t("tab2_calib_ok") if MODEL_META["hl_p"] > 0.05 else t("tab2_calib_sub"))

    roc_path = ARTIFACTS / "run_v21" / "roc_ensemble_top3_v21.png"
    cal_path = ARTIFACTS / "run_v21" / "calibration_ensemble_top3_v21.png"

    # Gráficas al 80% — usando columnas anidadas [1, 8, 1]
    c1, c2 = st.columns(2)
    with c1:
        sub = st.columns([1, 8, 1])
        with sub[1]:
            st.caption(t("tab2_caption_roc"))
            if roc_path.exists(): st.image(str(roc_path), width="stretch")
            else: st.info(t("tab2_missing"))
    with c2:
        sub = st.columns([1, 8, 1])
        with sub[1]:
            st.caption(t("tab2_caption_cal"))
            if cal_path.exists(): st.image(str(cal_path), width="stretch")
            else: st.info(t("tab2_missing"))


# ═══════════════════════════════════════════════════════════════
# TAB 3 · COMPARATIVA
# ═══════════════════════════════════════════════════════════════
with tab3:
    tab_path = ARTIFACTS / "run_v21" / "tabla_modelos_v21.csv"
    if tab_path.exists():
        df_tab = pd.read_csv(tab_path)
        st.markdown(f"<div style='font-size:13px; color:{TEXT_SECONDARY}; margin-bottom: 10px;'>"
                    f"{t('tab3_caption')}</div>", unsafe_allow_html=True)
        st.dataframe(df_tab, width="stretch", height=380, hide_index=True)
    else:
        st.info(t("tab3_missing"))


# ═══════════════════════════════════════════════════════════════
# TAB 4 · COHORTE · Flowchart estilo paper (HTML, no matplotlib)
# ═══════════════════════════════════════════════════════════════
with tab4:
    # KPIs arriba en una fila
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t("tab4_metric_n"),    f"{MODEL_META['n_train']:,}")
    k2.metric(t("tab4_metric_prev"), f"{MODEL_META['prevalencia']*100:.1f}%")
    k3.metric(t("tab4_metric_dead"), f"{int(MODEL_META['n_train']*MODEL_META['prevalencia']):,}")
    k4.metric("SAD+ / SAD−", "5,213 / 7,351", "Sub-cohortes")

    # Textos del flowchart
    if LANG == "es":
        cf = {
            "top_title":   "Pacientes diagnosticados con sepsis, sepsis grave o shock séptico",
            "top_sub":     "MIMIC-IV v3.1",
            "top_n":       "n = 25,570",
            "excl1_title": "Se excluyen los siguientes pacientes:",
            "excl1_items": [
                "Edad &lt; 18 años (pre-filtrado MIMIC)",
                "Estancia en UCI ≤ 24h (n = 2,563)",
                "Sin registro CAM-ICU (n = 6,317)",
            ],
            "g1_title":    "Casos restantes tras criterios de inclusión (Grupo 1)",
            "g1_n":        "n = 16,690",
            "excl2_title": "Se excluyen los siguientes pacientes:",
            "excl2_items": [
                "Enf. cerebral primaria (n = 3,410)",
                "Trastorno mental (n = 806)",
                "Abuso alcohol/drogas (n = 194)",
                "Epilepsia (n = 682)",
                "Encefalopatía metab./hepática (n = 759)",
                "Datos clínicos faltantes (n = 1,575)",
                "Readmisiones / duplicados (SQL)",
            ],
            "excl2_note":  "* Categorías no mutuamente excluyentes (suma 7,426 &gt; eliminados 4,126)",
            "final_title": "Cohorte final tras la eliminación",
            "final_sub":   "5,213 SAD+ y 7,351 SAD−",
            "final_n":     "n = 12,564 pacientes",
            "sadp":        "5,213 SAD+ (CAM-ICU +)",
            "sadn":        "7,351 SAD− (CAM-ICU −)",
            "surv":        "Supervivientes",
            "dead":        "Fallecidos",
        }
    else:
        cf = {
            "top_title":   "Patients diagnosed with sepsis, severe sepsis or septic shock",
            "top_sub":     "MIMIC-IV v3.1",
            "top_n":       "n = 25,570",
            "excl1_title": "The following patients are excluded:",
            "excl1_items": [
                "Age &lt; 18 years (MIMIC pre-filter)",
                "ICU stay ≤ 24h (n = 2,563)",
                "No CAM-ICU record (n = 6,317)",
            ],
            "g1_title":    "Remaining cases after inclusion criteria (Group 1)",
            "g1_n":        "n = 16,690",
            "excl2_title": "The following patients are excluded:",
            "excl2_items": [
                "Primary brain disease (n = 3,410)",
                "Mental disorder (n = 806)",
                "Alcohol / drug abuse (n = 194)",
                "Epilepsy (n = 682)",
                "Metabolic / hepatic encephalopathy (n = 759)",
                "Missing clinical data (n = 1,575)",
                "Readmissions / duplicates (SQL)",
            ],
            "excl2_note":  "* Non mutually exclusive categories (sum 7,426 &gt; removed 4,126)",
            "final_title": "Final cohort after elimination",
            "final_sub":   "5,213 SAD+ and 7,351 SAD−",
            "final_n":     "n = 12,564 patients",
            "sadp":        "5,213 SAD+ (CAM-ICU +)",
            "sadn":        "7,351 SAD− (CAM-ICU −)",
            "surv":        "Survivors",
            "dead":        "Deceased",
        }

    excl1_html = "".join(f"<li>{it}</li>" for it in cf["excl1_items"])
    excl2_html = "".join(f"<li>{it}</li>" for it in cf["excl2_items"])

    # SVG arrows como strings de UNA línea (evita que markdown lo lea como code block)
    # V_ARROW · más larga (54px de línea + cabezal) para más respiración entre pasos
    # BRANCH   · línea vertical izquierda continua (sin junciones que dejen artefactos)
    H_ARROW = '<svg width="48" height="14" viewBox="0 0 48 14" xmlns="http://www.w3.org/2000/svg"><line x1="2" y1="7" x2="40" y2="7" stroke="#94A3B8" stroke-width="1.5"/><polygon points="40,3 48,7 40,11" fill="#94A3B8"/></svg>'
    V_ARROW = '<svg width="14" height="54" viewBox="0 0 14 54" xmlns="http://www.w3.org/2000/svg"><line x1="7" y1="2" x2="7" y2="46" stroke="#94A3B8" stroke-width="1.5"/><polygon points="3,46 7,54 11,46" fill="#94A3B8"/></svg>'
    BRANCH = '<svg width="860" height="72" viewBox="0 0 860 72" xmlns="http://www.w3.org/2000/svg"><line x1="220" y1="0" x2="220" y2="62" stroke="#94A3B8" stroke-width="1.5"/><line x1="220" y1="28" x2="640" y2="28" stroke="#94A3B8" stroke-width="1.5"/><line x1="640" y1="28" x2="640" y2="62" stroke="#94A3B8" stroke-width="1.5"/><polygon points="216,62 220,70 224,62" fill="#94A3B8"/><polygon points="636,62 640,70 644,62" fill="#94A3B8"/></svg>'
    FORK = '<svg width="320" height="34" viewBox="0 0 320 34" xmlns="http://www.w3.org/2000/svg"><line x1="160" y1="0" x2="160" y2="10" stroke="#94A3B8" stroke-width="1.5"/><line x1="78" y1="10" x2="242" y2="10" stroke="#94A3B8" stroke-width="1.5"/><line x1="78" y1="10" x2="78" y2="26" stroke="#94A3B8" stroke-width="1.5"/><polygon points="74,26 78,34 82,26" fill="#94A3B8"/><line x1="242" y1="10" x2="242" y2="26" stroke="#94A3B8" stroke-width="1.5"/><polygon points="238,26 242,34 246,26" fill="#94A3B8"/></svg>'

    # HTML del flowchart · SIN indentación · cf-step-unit envuelve step + vline estirable
    flowchart_html = (
f'<div class="cf-grid">'
f'<div class="cf-step-unit">'
f'<div class="cf-step"><div class="cf-num">1</div><div class="cf-body">'
f'<div class="cf-title">{cf["top_title"]}</div>'
f'<div class="cf-sub">{cf["top_sub"]}</div>'
f'<div class="cf-n">{cf["top_n"]}</div></div></div>'
f'<div class="cf-vline"></div></div>'
f'<div class="cf-arrow-h">{H_ARROW}</div>'
f'<div class="cf-side"><div class="cf-side-title">{cf["excl1_title"]}</div>'
f'<ul class="cf-side-list">{excl1_html}</ul></div>'
f'<div class="cf-step-unit">'
f'<div class="cf-step"><div class="cf-num">2</div><div class="cf-body">'
f'<div class="cf-title">{cf["g1_title"]}</div>'
f'<div class="cf-n">{cf["g1_n"]}</div></div></div>'
f'<div class="cf-vline"></div></div>'
f'<div class="cf-arrow-h">{H_ARROW}</div>'
f'<div class="cf-side"><div class="cf-side-title">{cf["excl2_title"]}</div>'
f'<ul class="cf-side-list">{excl2_html}</ul>'
f'<div class="cf-side-note">{cf["excl2_note"]}</div></div>'
f'<div class="cf-final"><div class="cf-num">3</div><div class="cf-body">'
f'<div class="cf-title">{cf["final_title"]}</div>'
f'<div class="cf-sub">{cf["final_sub"]}</div>'
f'<div class="cf-n">{cf["final_n"]}</div></div></div>'
f'<div class="cf-branch">{BRANCH}</div>'
f'<div class="cf-cohorts-row">'
f'<div class="cf-cohort-block">'
f'<div class="cf-sadp"><div class="cf-num cf-num-sage">4</div><div class="cf-body">'
f'<div>{cf["sadp"]}</div><div class="cf-n">n = 5,213</div></div></div>'
f'<div class="cf-fork">{FORK}</div>'
f'<div class="cf-outcomes-row">'
f'<div class="cf-out cf-out-surv"><div class="cf-out-num">4a</div>'
f'<div class="cf-out-info"><span class="cf-out-label">{cf["surv"]}</span>'
f'<span class="cf-out-n">n = 4,013 (77.0%)</span></div></div>'
f'<div class="cf-out cf-out-dead"><div class="cf-out-num">4b</div>'
f'<div class="cf-out-info"><span class="cf-out-label">{cf["dead"]}</span>'
f'<span class="cf-out-n">n = 1,200 (23.0%)</span></div></div>'
f'</div></div>'
f'<div class="cf-cohort-block">'
f'<div class="cf-sadn"><div class="cf-num cf-num-rose">5</div><div class="cf-body">'
f'<div>{cf["sadn"]}</div><div class="cf-n">n = 7,351</div></div></div>'
f'<div class="cf-fork">{FORK}</div>'
f'<div class="cf-outcomes-row">'
f'<div class="cf-out cf-out-surv"><div class="cf-out-num">5a</div>'
f'<div class="cf-out-info"><span class="cf-out-label">{cf["surv"]}</span>'
f'<span class="cf-out-n">n = 6,758 (91.9%)</span></div></div>'
f'<div class="cf-out cf-out-dead"><div class="cf-out-num">5b</div>'
f'<div class="cf-out-info"><span class="cf-out-label">{cf["dead"]}</span>'
f'<span class="cf-out-n">n = 593 (8.1%)</span></div></div>'
f'</div></div>'
f'</div>'
f'</div>'
    )
    st.markdown(flowchart_html, unsafe_allow_html=True)

    # Comparación con paper Zhang abajo
    st.markdown(f"""
    <div style="background:{SURFACE_TINT}; border:1px solid {BORDER}; border-radius:12px;
                padding: 14px 18px; margin-top: 20px; font-size:13px; line-height:1.65; max-width: 1180px; margin-left:auto; margin-right:auto;">
      <b style="color:{UPCH_PRIMARY};">{t("tab4_comparison")}</b><br>
      <span style="color:{TEXT_SECONDARY};">{t("tab4_paper_lbl")}</span> 598 {t("tab4_pacientes")}, 13.7% {t("tab4_mortality")} ·
      <span style="color:{TEXT_SECONDARY};">{t("tab4_ours_lbl")}</span> <b>{MODEL_META["n_train"]:,}</b> {t("tab4_pacientes")}, <b>{MODEL_META["prevalencia"]*100:.1f}%</b> {t("tab4_mortality")}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 5 · CÓMO FUNCIONA EL MODELO A + NOTAS
# ═══════════════════════════════════════════════════════════════
with tab5:
    n_sev  = len(NOTE_KEYWORDS["severe"])
    n_mod  = len(NOTE_KEYWORDS["moderate"])
    n_pro  = len(NOTE_KEYWORDS["protective"])
    st.markdown(f"""
{t("tab5_intro")}

1. {t("tab5_step_1")}
2. {t("tab5_step_2")}
   - {t("tab5_severe")} ({n_sev} {t("tab5_severe_lbl")}): {t("tab5_severe_ex")}
   - {t("tab5_moderate")} ({n_mod} {t("tab5_severe_lbl")}): {t("tab5_moderate_ex")}
   - {t("tab5_protective")} ({n_pro} {t("tab5_severe_lbl")}): {t("tab5_protective_ex")}
3. {t("tab5_step_3")}

{t("tab5_note")}
""")


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown(f'<div class="app-footer">{t("footer")}</div>', unsafe_allow_html=True)
