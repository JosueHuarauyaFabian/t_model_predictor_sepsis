# 📋 Datos de Ejemplo para Pruebas

## 🧪 Caso 1: Paciente de Riesgo BAJO (🟢)

### Datos Clínicos (24h):
- **Hemoglobina:** 12.5 g/dL
- **Albúmina:** 3.8 g/dL
- **aPTT:** 28.5 segundos
- **Bilirrubina:** 0.8 mg/dL
- **Estancia UCI:** 2.0 días

### Nota Clínica (Español):
```
Paciente masculino de 45 años ingresado por sepsis de foco urinario. Se inició tratamiento antibiótico empírico con piperacilina-tazobactam. Hemodinamicamente estable sin requerimiento de vasopresores. Glasgow 15/15. Diuresis adecuada. Gasometría con pH 7.38, lactato 1.2 mmol/L. Mejoría clínica progresiva. Tolera vía oral. Plan: continuar antibiótico y vigilancia en UCI.
```

**Resultado Esperado:** ~15-25% mortalidad (BAJO)

---

## ⚠️ Caso 2: Paciente de Riesgo MODERADO (🟡)

### Datos Clínicos (24h):
- **Hemoglobina:** 8.2 g/dL
- **Albúmina:** 2.3 g/dL
- **aPTT:** 45.0 segundos
- **Bilirrubina:** 2.5 mg/dL
- **Estancia UCI:** 5.0 días

### Nota Clínica (Español):
```
Paciente femenino de 68 años con shock séptico secundario a neumonía bilateral. Requiere noradrenalina 0.15 mcg/kg/min para mantener PAM >65 mmHg. Ventilación mecánica invasiva con PEEP 8, FiO2 50%. Oliguria con diuresis <0.5 ml/kg/h. Lactato elevado 3.8 mmol/L. Coagulopatía leve. Leucocitosis 18,000. Procalcitonina 8.5 ng/ml. Sedación con midazolam. Plan: escalar antibióticos a meropenem, reposición volumétrica, soporte hemodinámico.
```

**Resultado Esperado:** ~40-55% mortalidad (MODERADO)

---

## 🔴 Caso 3: Paciente de Riesgo ALTO (🔴)

### Datos Clínicos (24h):
- **Hemoglobina:** 6.8 g/dL
- **Albúmina:** 1.8 g/dL
- **aPTT:** 78.0 segundos
- **Bilirrubina:** 4.2 mg/dL
- **Estancia UCI:** 10.0 días

### Nota Clínica (Español):
```
Paciente masculino de 72 años con shock séptico refractario de origen abdominal. Falla multiorgánica establecida. Requiere noradrenalina 0.45 mcg/kg/min más vasopresina 0.04 UI/min. Ventilación mecánica con parámetros altos: PEEP 14, FiO2 100%, PaO2/FiO2 85. Anuria con criterios de hemodiálisis urgente. Acidosis metabólica severa pH 7.15, lactato 8.2 mmol/L. Coagulación intravascular diseminada, plaquetas 45,000. Ictericia progresiva, bilirrubina total en ascenso. Hipotermia 35.2°C. Pronóstico reservado. Reunión familiar programada para discutir limitación de esfuerzo terapéutico.
```

**Resultado Esperado:** ~70-85% mortalidad (ALTO)

---

## 🌍 Caso 4: Prueba de Traducción (Inglés)

### Datos Clínicos (24h):
- **Hemoglobina:** 10.5 g/dL
- **Albúmina:** 3.2 g/dL
- **aPTT:** 35.0 segundos
- **Bilirrubina:** 1.5 mg/dL
- **Estancia UCI:** 3.5 días

### Nota Clínica (English):
```
65-year-old female patient admitted with septic shock secondary to intra-abdominal infection. Post-operative day 3 after emergency exploratory laparotomy. Currently on norepinephrine 0.08 mcg/kg/min. Mechanical ventilation with PEEP 6, FiO2 40%. Adequate urine output. Lactate improving from 4.5 to 2.1 mmol/L. Cultures pending. Broad-spectrum antibiotics continued. Hemodynamically improving. Plan: wean vasopressors, extubation trial tomorrow.
```

**Resultado Esperado:** ~30-40% mortalidad (MODERADO-BAJO)

---

## 📊 Tabla Resumen

| Caso | Edad | Diagnóstico | Hemoglobina | Albúmina | aPTT | Bilirrubina | UCI días | Riesgo |
|------|------|-------------|-------------|----------|------|-------------|----------|--------|
| 1 | 45 | Sepsis urinaria | 12.5 | 3.8 | 28.5 | 0.8 | 2.0 | 🟢 BAJO |
| 2 | 68 | Shock séptico | 8.2 | 2.3 | 45.0 | 2.5 | 5.0 | 🟡 MODERADO |
| 3 | 72 | Falla multiorgánica | 6.8 | 1.8 | 78.0 | 4.2 | 10.0 | 🔴 ALTO |
| 4 | 65 | Sepsis abdominal | 10.5 | 3.2 | 35.0 | 1.5 | 3.5 | 🟡 MODERADO |

---

## 🎯 Qué Observar en cada Prueba:

### ✅ Funcionalidades:
1. **Detección de idioma** (ES/EN)
2. **Traducción con DeepL** (debe decir "✓ Traducción con DeepL exitosa" en logs)
3. **Generación de embedding** (768 dimensiones)
4. **Predicción con WeightedEnsemble_L3**
5. **Nivel de riesgo visual** (🟢🟡🔴)
6. **Timestamp** de la predicción
7. **Información en sidebar** (110 modelos, dispositivo, etc.)

### 📊 En el Sidebar verás:
- **Modelos base:** 110
- **Mejor modelo:** WeightedEnsemble_L3
- **Dispositivo:** cpu o cuda
- **Features tabulares:** 5
- **Embeddings:** 768 (ClinicalBERT)

---

## 🧪 Pruebas Avanzadas:

### Prueba de Validación (valores fuera de rango):
- **Hemoglobina:** 2.0 g/dL (< 3.0, debería advertir)
- **aPTT:** 250.0 segundos (> 200.0, debería advertir)

### Prueba de Error Handling:
- Dejar la nota clínica vacía → debe mostrar error
- Nota muy corta: "Paciente enfermo" → debería procesar sin fallar

---

## 📝 Notas:
- Los resultados son probabilidades del modelo, no diagnósticos
- La traducción ES→EN ocurre automáticamente si es necesario
- Si DeepL falla, usa Helsinki-NLP automáticamente
- Los logs muestran cada paso del proceso
