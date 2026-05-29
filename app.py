"""
app.py — Punto de entrada para Streamlit Cloud
Ejecuta el contenido de app_sepsis.py de forma segura
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "app_sepsis.py"), run_name="__main__")
