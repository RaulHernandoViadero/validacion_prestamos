# Estado del TFM — Verificación Documental

## Repositorio
- GitHub: https://github.com/RaulHernandoViadero/validacion_prestamos
- Rama activa: `dev`
- Python: Anaconda `C:\Users\raulh\anaconda3\python.exe`
- Ejecutar tests: `C:\Users\raulh\anaconda3\python.exe -m pytest tests/ -v`

## Progreso por fases

### ✅ FASE 1 — Generación de datos sintéticos (COMPLETA)
- `fake_data_factory.py` ✅ — NIF, MRZ ICAO, préstamos, 24 tests pasan
- `dni_generator.py` ✅ — DNI 1518×957 px, 9 ROIs, layout escalado con _B=1.5
- `loan_form_generator.py` ✅ — Formulario 1588×2246 px, 14 ROIs
- `augmentation.py` ✅ — 11 transformaciones
- `annotation_writer.py` ✅ — ROIs → YOLO .txt + YAML dataset
- `generate_dataset.py` ✅ — Script maestro CLI (probado con 5 y 400 expedientes)
- Dataset de 400 expedientes generándose en background cuando se cerró

### ✅ FASE 2 — Modelos (COMPLETA — pendiente entrenar)
- `yolo_trainer.py` ✅ — entrena yolo_dni.pt y yolo_loan.pt
- `authenticity_classifier.py` ✅ — ResNet-18 fine-tuned, 8.4M params entrenables
- `model_utils.py` ✅ — utilidades compartidas

### 🔶 FASE 3 — Pipeline de inferencia (PARCIALMENTE COMPLETA)
- `yolo_inference.py` ✅ — wrapper YOLOv8 con ROIDetectado + ResultadoDeteccion
- `easyocr_engine.py` ✅ — singleton EasyOCR, extrae texto de ROIs
- `text_postprocessor.py` ✅ — normaliza NIF, fechas, importes, nombres (testeado OK)
- `cross_validator.py` ✅ — 9 reglas R01-R09 implementadas y commiteadas
- `verdict_engine.py` ✅ — motor de veredicto APTO/INCONSISTENTE con confianza
- `business_rules.py` ❌ — PENDIENTE (siguiente fichero a escribir)
- `document_pipeline.py` ❌ — PENDIENTE (orquestador end-to-end)

### ⏳ FASE 4 — API REST FastAPI (PENDIENTE)
### ⏳ FASE 5 — Frontend Streamlit (PENDIENTE)
### ⏳ FASE 6 — Integración, tests, Docker (PENDIENTE)

## Último commit en GitHub
`feat(fase3): inferencia YOLO + OCR + postprocesado + validación cruzada + veredicto`
- cross_validator.py y verdict_engine.py están escritos pero NO commiteados todavía
- Hay que hacer git add + commit al empezar mañana

## Próximos pasos (en orden)
1. `git add src/validation/cross_validator.py src/validation/verdict_engine.py && git commit && git push`
2. Escribir `src/validation/business_rules.py`
3. Escribir `src/pipeline/document_pipeline.py` (orquestador end-to-end)
4. Empezar Fase 4: `api/schemas/models.py` → routers → `api/main.py`

## Decisiones tomadas
- DNI: _SCALE=3 → 1518×957 px, _B=1.5 para escalar todo el layout
- Formulario: _SCALE=2 → 1588×2246 px
- ResNet-18: solo layer4 + fc entrenables (fine-tuning parcial)
- Veredicto: 70% peso reglas + 30% peso confianza OCR
- Tolerancia similitud nombres: 85% (Levenshtein normalizado)
- Dataset: 400 expedientes, 15% inconsistentes, splits 70/17.5/12.5
