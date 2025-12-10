#!/bin/bash

# Script de inicio rápido para el predictor de sepsis

set -e

echo "=============================================="
echo "🏥 Predictor Multimodal de Mortalidad"
echo "=============================================="
echo ""

# Verificar si existe .env
if [ ! -f .env ]; then
    echo "⚠️  No se encontró archivo .env"
    echo "📝 Copiando .env.example a .env..."
    cp .env.example .env
    echo "✅ Archivo .env creado. Edítalo con tu DEEPL_API_KEY si lo tienes."
    echo ""
fi

# Verificar si Docker está corriendo
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker no está corriendo. Por favor inicia Docker Desktop."
    exit 1
fi

echo "🐳 Docker está corriendo"
echo ""

# Opción de usuario
echo "Selecciona una opción:"
echo "1) Docker Compose (recomendado)"
echo "2) Docker manual"
echo "3) Detener contenedor"
echo "4) Ver logs"
echo ""
read -p "Opción [1-4]: " option

case $option in
    1)
        echo ""
        echo "🚀 Iniciando con Docker Compose..."
        docker-compose up -d
        echo ""
        echo "✅ Aplicación iniciada!"
        echo "🌐 Accede en: http://localhost:8501"
        echo ""
        echo "Para ver logs:"
        echo "   docker-compose logs -f"
        echo ""
        echo "Para detener:"
        echo "   docker-compose down"
        ;;
    2)
        echo ""
        echo "🔨 Construyendo imagen..."
        docker build -t sepsis-multimodal-app .
        echo ""
        echo "🚀 Ejecutando contenedor..."
        docker run -d --rm \
            --name sepsis-predictor \
            -p 8501:8501 \
            --env-file .env \
            --memory=8g \
            --cpus=4 \
            sepsis-multimodal-app
        echo ""
        echo "✅ Aplicación iniciada!"
        echo "🌐 Accede en: http://localhost:8501"
        echo ""
        echo "Para ver logs:"
        echo "   docker logs -f sepsis-predictor"
        echo ""
        echo "Para detener:"
        echo "   docker stop sepsis-predictor"
        ;;
    3)
        echo ""
        echo "🛑 Deteniendo contenedor..."
        docker-compose down 2>/dev/null || docker stop sepsis-predictor 2>/dev/null || true
        echo "✅ Contenedor detenido"
        ;;
    4)
        echo ""
        echo "📋 Mostrando logs (Ctrl+C para salir)..."
        docker-compose logs -f 2>/dev/null || docker logs -f sepsis-predictor 2>/dev/null || echo "❌ No hay contenedor corriendo"
        ;;
    *)
        echo "❌ Opción inválida"
        exit 1
        ;;
esac
