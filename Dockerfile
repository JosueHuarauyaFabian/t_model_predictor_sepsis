FROM python:3.11-slim

# Metadata
LABEL maintainer="Sepsis Predictor Team"
LABEL description="Modelo multimodal para predicción de mortalidad a 28 días en sepsis"
LABEL version="1.0.0"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar y instalar dependencias Python
COPY requirements.txt .
RUN pip install --upgrade pip==25.3 && \
    pip install -r requirements.txt

# Copiar código y modelo
COPY app.py ./app.py
COPY modelo_multimodal_clinicalbert_best ./modelo_multimodal_clinicalbert_best

# Crear directorio para logs
RUN mkdir -p /app/logs

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Exponer puerto
EXPOSE 8501

# Usuario no-root (mejor práctica de seguridad)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Comando de inicio con configuración optimizada
# Nota: CORS debe estar habilitado si XSRF Protection está activo
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableXsrfProtection=true", \
     "--browser.gatherUsageStats=false"]
