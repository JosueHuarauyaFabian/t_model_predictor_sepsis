import os
import logging
from pathlib import Path
from typing import Tuple, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from autogluon.tabular import TabularPredictor
from langdetect import detect, LangDetectException
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import deepl


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_multimodal_clinicalbert_best"


TAB_FEATURES = {
    "hemoglobin": "Hemoglobina (g/dL)",
    "albumin": "Albúmina (g/dL)",
    "aptt": "aPTT (segundos)",
    "bilirubin": "Bilirrubina total (mg/dL)",
    "icu_los_days": "Estancia UCI (días)",
}

# Rangos de validación según el entrenamiento
FEATURE_RANGES = {
    "hemoglobin": (3.0, 25.0),
    "albumin": (1.0, 8.0),
    "aptt": (10.0, 200.0),
    "bilirubin": (0.1, 50.0),
    "icu_los_days": (0.1, 365.0),
}


@st.cache_resource(show_spinner=False)
def load_models() -> Tuple[TabularPredictor, AutoTokenizer, AutoModel, torch.device]:
    """Carga el modelo AutoGluon y ClinicalBERT con progress bar optimizado."""
    logger.info("Iniciando carga de modelos...")

    # Progress bar placeholder
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Paso 1: Cargar AutoGluon (60% del tiempo)
        status_text.text("🔄 Cargando AutoGluon predictor...")
        progress_bar.progress(10)
        predictor = TabularPredictor.load(
            MODEL_PATH,
            require_py_version_match=False  # Permitir carga cross-platform
        )
        logger.info(f"✓ AutoGluon cargado: {len(predictor.model_names())} modelos")
        progress_bar.progress(60)

        # Paso 2: Cargar ClinicalBERT tokenizer (10% del tiempo)
        status_text.text("🔄 Cargando ClinicalBERT tokenizer...")
        progress_bar.progress(65)
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        logger.info("✓ Tokenizer cargado")
        progress_bar.progress(75)

        # Paso 3: Cargar ClinicalBERT model (20% del tiempo)
        status_text.text("🔄 Cargando ClinicalBERT model...")
        progress_bar.progress(80)
        bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        progress_bar.progress(90)

        # Paso 4: Mover a dispositivo (10% del tiempo)
        status_text.text("🔄 Optimizando para dispositivo...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model = bert_model.to(device)
        bert_model.eval()
        logger.info(f"✓ ClinicalBERT cargado en dispositivo: {device}")
        progress_bar.progress(100)

        # Limpiar UI
        status_text.text("✅ Modelos cargados exitosamente!")
        import time
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        return predictor, tokenizer, bert_model, device

    except Exception as e:
        logger.error(f"Error cargando modelos: {e}")
        status_text.text(f"❌ Error: {str(e)}")
        raise


@st.cache_resource(show_spinner=False)
def get_hf_translator():
    return pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-es-en",
        tokenizer="Helsinki-NLP/opus-mt-es-en",
    )


def translate_es_en(text: str) -> str:
    """Traduce texto de español a inglés usando DeepL (prioritario) o Helsinki-NLP (fallback)."""
    deepl_api_key = os.getenv("DEEPL_API_KEY", "").strip()
    if not deepl_api_key:
        try:
            deepl_api_key = st.secrets.get("DEEPL_API_KEY", "").strip()
        except Exception:
            deepl_api_key = ""

    # Intentar DeepL primero (corregido: Translator en vez de DeepLClient)
    if deepl_api_key:
        try:
            translator = deepl.Translator(deepl_api_key)
            result = translator.translate_text(text, source_lang="ES", target_lang="EN-US")
            logger.info("✓ Traducción con DeepL exitosa")
            return result.text
        except Exception as e:
            logger.warning(f"DeepL falló ({e}), usando Helsinki-NLP como fallback")

    # Fallback a Helsinki-NLP
    try:
        hf_translator = get_hf_translator()
        translated = hf_translator(text, max_length=512)[0]["translation_text"]
        logger.info("✓ Traducción con Helsinki-NLP exitosa")
        return translated
    except Exception as e:
        logger.error(f"Error en traducción: {e}")
        # Si ambos fallan, devolver el texto original
        return text


def detect_and_maybe_translate(text: str) -> Tuple[str, str]:
    cleaned = text.strip()
    if not cleaned:
        return "", "unknown"

    try:
        lang = detect(cleaned)
    except LangDetectException:
        lang = "unknown"

    if lang.startswith("es"):
        translated = translate_es_en(cleaned)
        return translated, "es"

    return cleaned, lang


def compute_clinicalbert_embedding(
    text_en: str, tokenizer: AutoTokenizer, bert_model: AutoModel, device: torch.device
) -> np.ndarray:
    """Genera embedding de 768 dimensiones usando ClinicalBERT."""
    if not text_en.strip():
        return np.zeros(768, dtype="float32")

    try:
        encoded = tokenizer(
            text_en,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = bert_model(**encoded)

        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return cls_embedding.astype("float32")

    except Exception as e:
        logger.error(f"Error generando embedding: {e}")
        # Devolver vector de ceros como fallback
        return np.zeros(768, dtype="float32")


def validate_feature_ranges(tab_values: dict) -> Tuple[bool, Optional[str]]:
    """Valida que los valores tabulares estén dentro de los rangos de entrenamiento."""
    for feature, value in tab_values.items():
        if feature in FEATURE_RANGES:
            min_val, max_val = FEATURE_RANGES[feature]
            if not (min_val <= value <= max_val):
                return False, (
                    f"⚠️ {TAB_FEATURES[feature]}: {value:.2f} está fuera del rango válido "
                    f"[{min_val}, {max_val}]. Los resultados pueden ser poco confiables."
                )
    return True, None


def build_feature_row(
    predictor: TabularPredictor,
    tab_values: dict,
    embedding: np.ndarray,
) -> pd.DataFrame:
    """Construye el DataFrame con features tabulares + embeddings para predicción."""
    feature_names: List[str] = list(predictor.feature_metadata.get_features())

    emb_cols = [c for c in feature_names if c.startswith("emb_")]
    emb_cols_sorted = sorted(
        emb_cols, key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 0
    )

    row = {}
    for col in feature_names:
        if col in TAB_FEATURES:
            row[col] = float(tab_values[col])
        elif col in emb_cols_sorted:
            idx = int(col.split("_")[1])
            if 0 <= idx < len(embedding):
                row[col] = float(embedding[idx])
            else:
                row[col] = 0.0
        else:
            row[col] = np.nan

    return pd.DataFrame([row])


def main() -> None:
    st.set_page_config(
        page_title="Predicción de mortalidad a 28 días",
        layout="wide",
    )

    st.title("Predicción de mortalidad a 28 días")
    st.caption(
        "Herramienta de apoyo a la investigación basada en datos clínicos tempranos y texto de la nota médica."
    )

    # Cargar modelos (con progress bar interno)
    predictor, tokenizer, bert_model, device = load_models()

    # Mostrar información del modelo en sidebar
    with st.sidebar:
        st.subheader("ℹ️ Información del Modelo")
        st.write(f"**Modelos totales:** {len(predictor.model_names())}")
        st.write(f"**Modelo usado:** {predictor.model_best}")
        st.write(f"**Dispositivo:** {device}")
        st.write(f"**Features tabulares:** {len(TAB_FEATURES)}")
        st.write(f"**Embeddings:** 768 (ClinicalBERT)")

        # Leer versión del modelo si existe
        version_file = MODEL_PATH / "version.txt"
        if version_file.exists():
            version = version_file.read_text().strip()
            st.write(f"**Versión AutoGluon:** {version}")

        st.divider()
        st.caption("⚠️ Solo para investigación. No usar para decisiones clínicas.")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Datos clínicos de las primeras 24 horas")

        cols = st.columns(2)

        hemoglobin = cols[0].number_input(
            TAB_FEATURES["hemoglobin"],
            min_value=FEATURE_RANGES["hemoglobin"][0],
            max_value=FEATURE_RANGES["hemoglobin"][1],
            value=10.0,
            step=0.1,
            help=f"Rango válido: {FEATURE_RANGES['hemoglobin'][0]}-{FEATURE_RANGES['hemoglobin'][1]}"
        )

        albumin = cols[1].number_input(
            TAB_FEATURES["albumin"],
            min_value=FEATURE_RANGES["albumin"][0],
            max_value=FEATURE_RANGES["albumin"][1],
            value=3.0,
            step=0.1,
            help=f"Rango válido: {FEATURE_RANGES['albumin'][0]}-{FEATURE_RANGES['albumin'][1]}"
        )

        aptt = cols[0].number_input(
            TAB_FEATURES["aptt"],
            min_value=FEATURE_RANGES["aptt"][0],
            max_value=FEATURE_RANGES["aptt"][1],
            value=30.0,
            step=0.5,
            help=f"Rango válido: {FEATURE_RANGES['aptt'][0]}-{FEATURE_RANGES['aptt'][1]}"
        )

        bilirubin = cols[1].number_input(
            TAB_FEATURES["bilirubin"],
            min_value=FEATURE_RANGES["bilirubin"][0],
            max_value=FEATURE_RANGES["bilirubin"][1],
            value=1.2,
            step=0.1,
            help=f"Rango válido: {FEATURE_RANGES['bilirubin'][0]}-{FEATURE_RANGES['bilirubin'][1]}"
        )

        icu_los_days = st.number_input(
            TAB_FEATURES["icu_los_days"],
            min_value=FEATURE_RANGES["icu_los_days"][0],
            max_value=FEATURE_RANGES["icu_los_days"][1],
            value=3.0,
            step=0.5,
            help=f"Rango válido: {FEATURE_RANGES['icu_los_days'][0]}-{FEATURE_RANGES['icu_los_days'][1]}"
        )

        st.subheader("Nota clínica")
        note_text = st.text_area(
            "Pegue aquí la nota médica (español o inglés)",
            height=220,
        )

    with col_right:
        st.subheader("Resultado")
        placeholder_metric = st.empty()
        placeholder_info = st.empty()

    if st.button("Calcular riesgo", type="primary"):
        if not note_text.strip():
            st.error("Por favor, introduzca la nota clínica.")
            return

        try:
            # Validar rangos de features tabulares
            tab_values = {
                "hemoglobin": hemoglobin,
                "albumin": albumin,
                "aptt": aptt,
                "bilirubin": bilirubin,
                "icu_los_days": icu_los_days,
            }

            is_valid, warning_msg = validate_feature_ranges(tab_values)
            if not is_valid:
                st.warning(warning_msg)

            # Detectar idioma y traducir
            with st.spinner("Detectando idioma y traduciendo (si es necesario)..."):
                text_en, lang = detect_and_maybe_translate(note_text)
                logger.info(f"Idioma detectado: {lang}")

            # Generar embedding
            with st.spinner("Generando embedding con ClinicalBERT..."):
                embedding = compute_clinicalbert_embedding(text_en, tokenizer, bert_model, device)
                logger.info(f"Embedding generado: {embedding.shape}")

            # Preparar features y predecir
            with st.spinner("Ejecutando modelo AutoGluon..."):
                features_df = build_feature_row(predictor, tab_values, embedding)
                logger.info(f"Features preparadas: {features_df.shape}")

                # Predecir con el mejor modelo (dinámico)
                proba_raw = predictor.predict_proba(features_df, model=predictor.model_best)

                # Simplificar parsing: AutoGluon siempre devuelve DataFrame con columnas [0, 1]
                if isinstance(proba_raw, pd.DataFrame):
                    proba = float(proba_raw.iloc[0, 1])  # Columna 1 = probabilidad de clase positiva
                else:
                    # Fallback por si cambia el comportamiento
                    proba = float(proba_raw.iloc[0] if isinstance(proba_raw, pd.Series) else proba_raw[0])

                logger.info(f"Predicción completada: {proba:.4f}")

            # Mostrar resultados
            with col_right:
                placeholder_metric.metric(
                    label="Probabilidad estimada de mortalidad a 28 días",
                    value=f"{proba * 100:.1f} %",
                    delta=None,
                )

                # Interpretación del riesgo
                if proba < 0.3:
                    risk_level = "🟢 Bajo"
                elif proba < 0.6:
                    risk_level = "🟡 Moderado"
                else:
                    risk_level = "🔴 Alto"

                placeholder_info.write(f"**Nivel de riesgo:** {risk_level}")
                placeholder_info.caption(
                    f"Idioma detectado: {lang.upper() if lang != 'unknown' else 'desconocido'}  \n"
                    f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
                    "⚠️ **Solo para investigación.** No usar para decisiones clínicas."
                )

        except Exception as e:
            logger.error(f"Error durante la predicción: {e}", exc_info=True)
            st.error(
                f"❌ Error durante la predicción: {str(e)}  \n"
                "Por favor, verifica los datos ingresados e intenta nuevamente."
            )


if __name__ == "__main__":
    main()
