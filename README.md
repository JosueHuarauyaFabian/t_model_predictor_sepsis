# 🏥 Predictor Multimodal de Mortalidad a 28 días en Sepsis

Sistema de predicción de mortalidad basado en **datos clínicos tabulares** y **notas médicas** usando **AutoGluon** y **ClinicalBERT**.

> ⚠️ **IMPORTANTE**: Esta herramienta es exclusivamente para **investigación**. No debe usarse para tomar decisiones clínicas.

---

## 📊 Arquitectura del Modelo

### Modelos Base
- **110 modelos entrenados** con preset `best_quality`
- Algoritmos: CatBoost, LightGBM, XGBoost, RandomForest, ExtraTrees, NeuralNetFastAI, NeuralNetTorch
- Estrategia: 8-fold bagging + 3 niveles de stacking
- **Modelo final**: WeightedEnsemble_L3

### Features
**Tabulares (5):**
- Hemoglobina (g/dL): 3.0-25.0
- Albúmina (g/dL): 1.0-8.0
- aPTT (segundos): 10.0-200.0
- Bilirrubina total (mg/dL): 0.1-50.0
- Estancia UCI (días): 0.1-365.0

**Texto:**
- 768 embeddings de ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`)
- Soporte para español (traducción automática con DeepL/Helsinki-NLP)

---

## 🚀 Instalación y Uso

### Opción 1: Docker (Recomendado)

1. **Clonar el repositorio**
```bash
git clone <tu-repo>
cd t_model_predictor_sepsis
```

2. **Configurar variables de entorno**
```bash
cp .env.example .env
# Editar .env y agregar tu DEEPL_API_KEY (opcional)
```

3. **Construir y ejecutar con Docker Compose**
```bash
docker-compose up -d
```

4. **Acceder a la aplicación**
```
http://localhost:8501
```

### Opción 2: Docker Manual

```bash
# Construir imagen
docker build -t sepsis-multimodal-app .

# Ejecutar contenedor
docker run --rm -p 8501:8501 \
  --env-file .env \
  --memory=8g \
  --cpus=4 \
  sepsis-multimodal-app
```

### Opción 3: Local (Python)

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app.py
```

---

## 🔧 Configuración

### Variables de Entorno

| Variable | Descripción | Obligatorio | Ejemplo |
|----------|-------------|-------------|---------|
| `DEEPL_API_KEY` | API Key de DeepL para traducción | No | `3e3ebc35-...` |
| `LOG_LEVEL` | Nivel de logging | No | `INFO` |

### Recursos Recomendados

| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| RAM | 4 GB | 8 GB |
| CPU | 2 cores | 4 cores |
| GPU | No requerida | CUDA (acelera embeddings) |

---

## 📝 Uso de la Aplicación

1. **Ingresar datos clínicos** de las primeras 24 horas:
   - Hemoglobina, Albúmina, aPTT, Bilirrubina, Estancia UCI

2. **Pegar nota clínica**:
   - Español o inglés
   - Se traduce automáticamente si es necesario

3. **Calcular riesgo**:
   - Obtener probabilidad de mortalidad a 28 días
   - Ver nivel de riesgo: 🟢 Bajo / 🟡 Moderado / 🔴 Alto

---

## 🔍 Monitoreo y Logs

### Ver logs del contenedor
```bash
docker logs -f sepsis-predictor
```

### Ver logs de la aplicación
```bash
docker exec sepsis-predictor tail -f /app/logs/app.log
```

### Healthcheck
```bash
curl http://localhost:8501/_stcore/health
```

---

## 🧪 Validación del Modelo

### Conjunto de Validación
- 10% del dataset original (guardado en `validation_set_10pct.parquet`)
- División estratificada por `mortality_28d`

### Métricas
- Métrica principal: **ROC-AUC**
- Ver leaderboard completo: `leaderboard_completo_todas_columnas.csv`

---

## 📂 Estructura del Proyecto

```
.
├── app.py                              # Aplicación Streamlit
├── requirements.txt                     # Dependencias Python
├── Dockerfile                           # Imagen Docker optimizada
├── docker-compose.yml                   # Orquestación Docker
├── .env.example                         # Template de variables de entorno
├── .gitignore                           # Archivos ignorados por Git
├── README.md                            # Este archivo
└── modelo_multimodal_clinicalbert_best/ # Modelo AutoGluon
    ├── predictor.pkl
    ├── learner.pkl
    ├── metadata.json
    ├── models/                          # 110 modelos base
    │   ├── CatBoost_BAG_L1/
    │   ├── LightGBM_BAG_L1/
    │   ├── WeightedEnsemble_L3/         # ⭐ Mejor modelo
    │   └── ...
    └── utils/
```

---

## 🔐 Seguridad

### Mejoras Implementadas
- ✅ Usuario no-root en Docker
- ✅ XSRF Protection habilitado
- ✅ Secrets en variables de entorno (no hardcoded)
- ✅ .gitignore para archivos sensibles
- ✅ Validación de rangos de entrada
- ✅ Manejo robusto de errores

### Notas de Seguridad
- No exponer la aplicación directamente a internet sin autenticación
- Usar HTTPS en producción (reverse proxy como nginx)
- Rotar API keys regularmente

---

## 🐛 Troubleshooting

### Problema: DeepL API falla
**Solución**: La aplicación usará Helsinki-NLP automáticamente. Verifica:
```bash
# Comprobar que la API key es válida
echo $DEEPL_API_KEY
```

### Problema: Out of Memory
**Solución**: Incrementar límite de memoria:
```bash
docker run --memory=16g ...
```

### Problema: Predicción lenta
**Causas posibles**:
1. Sin GPU → Embeddings de BERT son más lentos
2. Primera predicción → Caché de modelos
3. CPU limitado → Incrementar `--cpus`

---

## 📈 Mejoras Implementadas

### v1.0.0 (2025-01-25)
- ✅ Corregido error de `deepl.DeepLClient` → `deepl.Translator`
- ✅ BERT movido a GPU solo una vez (reducción de latencia)
- ✅ Rangos de validación alineados con entrenamiento
- ✅ Parsing de probabilidades simplificado
- ✅ Manejo robusto de errores con try-catch
- ✅ Logging estructurado
- ✅ Dockerfile optimizado con healthcheck
- ✅ Docker Compose con límites de recursos
- ✅ Sidebar con información del modelo
- ✅ Interpretación de nivel de riesgo (Bajo/Moderado/Alto)
- ✅ Timestamp en resultados

---

## 📄 Licencia

Este proyecto es para uso de investigación académica.

---

## 👥 Contacto

Para preguntas o problemas, abrir un issue en el repositorio.

---

## 🙏 Agradecimientos

- **MIMIC-IV**: Dataset de UCI
- **ClinicalBERT**: Modelo de embeddings médicos
- **AutoGluon**: Framework de AutoML
- **DeepL**: API de traducción
