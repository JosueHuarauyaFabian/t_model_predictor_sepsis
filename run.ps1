# Script de inicio rápido para Windows PowerShell

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "🏥 Predictor Multimodal de Mortalidad" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# Verificar si existe .env
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  No se encontró archivo .env" -ForegroundColor Yellow
    Write-Host "📝 Copiando .env.example a .env..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "✅ Archivo .env creado. Edítalo con tu DEEPL_API_KEY si lo tienes." -ForegroundColor Green
    Write-Host ""
}

# Verificar si Docker está corriendo
try {
    docker info | Out-Null
    Write-Host "🐳 Docker está corriendo" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker no está corriendo. Por favor inicia Docker Desktop." -ForegroundColor Red
    exit 1
}

Write-Host ""

# Menú de opciones
Write-Host "Selecciona una opción:" -ForegroundColor Yellow
Write-Host "1) Docker Compose (recomendado)"
Write-Host "2) Docker manual"
Write-Host "3) Detener contenedor"
Write-Host "4) Ver logs"
Write-Host ""

$option = Read-Host "Opción [1-4]"

switch ($option) {
    "1" {
        Write-Host ""
        Write-Host "🚀 Iniciando con Docker Compose..." -ForegroundColor Cyan
        docker-compose up -d
        Write-Host ""
        Write-Host "✅ Aplicación iniciada!" -ForegroundColor Green
        Write-Host "🌐 Accede en: http://localhost:8501" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Para ver logs:" -ForegroundColor Yellow
        Write-Host "   docker-compose logs -f" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Para detener:" -ForegroundColor Yellow
        Write-Host "   docker-compose down" -ForegroundColor Gray
    }
    "2" {
        Write-Host ""
        Write-Host "🔨 Construyendo imagen..." -ForegroundColor Cyan
        docker build -t sepsis-multimodal-app .
        Write-Host ""
        Write-Host "🚀 Ejecutando contenedor..." -ForegroundColor Cyan
        docker run -d --rm `
            --name sepsis-predictor `
            -p 8501:8501 `
            --env-file .env `
            --memory=8g `
            --cpus=4 `
            sepsis-multimodal-app
        Write-Host ""
        Write-Host "✅ Aplicación iniciada!" -ForegroundColor Green
        Write-Host "🌐 Accede en: http://localhost:8501" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Para ver logs:" -ForegroundColor Yellow
        Write-Host "   docker logs -f sepsis-predictor" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Para detener:" -ForegroundColor Yellow
        Write-Host "   docker stop sepsis-predictor" -ForegroundColor Gray
    }
    "3" {
        Write-Host ""
        Write-Host "🛑 Deteniendo contenedor..." -ForegroundColor Yellow
        try {
            docker-compose down 2>$null
        } catch {
            docker stop sepsis-predictor 2>$null
        }
        Write-Host "✅ Contenedor detenido" -ForegroundColor Green
    }
    "4" {
        Write-Host ""
        Write-Host "📋 Mostrando logs (Ctrl+C para salir)..." -ForegroundColor Cyan
        try {
            docker-compose logs -f
        } catch {
            try {
                docker logs -f sepsis-predictor
            } catch {
                Write-Host "❌ No hay contenedor corriendo" -ForegroundColor Red
            }
        }
    }
    default {
        Write-Host "❌ Opción inválida" -ForegroundColor Red
        exit 1
    }
}
