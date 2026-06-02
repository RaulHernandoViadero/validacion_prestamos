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
- `generate_dataset.py` ✅ — Script maestro CLI

### ✅ FASE 2 — Modelos (COMPLETA — pendiente entrenar)
- `yolo_trainer.py` ✅ — entrena yolo_dni.pt y yolo_loan.pt
- `authenticity_classifier.py` ✅ — ResNet-18 fine-tuned, 8.4M params entrenables
- `model_utils.py` ✅ — utilidades compartidas

### ✅ FASE 3 — Pipeline de inferencia (COMPLETA)
- `yolo_inference.py` ✅ — wrapper YOLOv8
- `easyocr_engine.py` ✅ — singleton EasyOCR
- `text_postprocessor.py` ✅ — normaliza NIF, fechas, importes, nombres
- `cross_validator.py` ✅ — 9 reglas R01-R09
- `verdict_engine.py` ✅ — motor de veredicto APTO/INCONSISTENTE
- `business_rules.py` ✅ — MRZ check digits, cuota francesa, perfil riesgo
- `document_pipeline.py` ✅ — orquestador end-to-end 6 etapas

### ✅ FASE 4 — API REST FastAPI (COMPLETA)
- `api/schemas/models.py` ✅ — Pydantic v2 schemas (request/response/error)
- `api/routers/verification.py` ✅ — POST /api/v1/verify
- `api/routers/history.py` ✅ — GET/DELETE /api/v1/history
- `api/routers/metrics.py` ✅ — GET /api/v1/metrics
- `api/main.py` ✅ — app FastAPI con lifespan, CORS, logging, /health
- `api/Dockerfile` ✅ — multi-stage python:3.11-slim

### ✅ FASE 5 — Frontend Streamlit (COMPLETA)
- `app/streamlit_app.py` ✅ — app principal con sidebar y estado API
- `app/pages/verificacion.py` ✅ — subida DNI+formulario, veredicto, 5 tabs
- `app/pages/historial.py` ✅ — lista paginada con filtros
- `app/pages/metricas.py` ✅ — KPIs, barras de progreso por regla
- `app/Dockerfile` ✅ — imagen slim solo streamlit+httpx+Pillow

### 🔶 FASE 6 — Integración, tests, Docker (PARCIAL)
- `docker-compose.yml` ✅ — api+frontend con healthchecks y red interna
- `requirements.txt` ✅ — dependencias completas con versiones fijadas
- `tests/test_api.py` ✅ — 17 tests integración API (17/17 passing)
- `src/reporting/pdf_report_generator.py` ❌ — PENDIENTE
- `tests/test_pipeline.py` ❌ — PENDIENTE (tests del pipeline completo)
- `notebooks/` ❌ — PENDIENTE (análisis exploratorio y demo)
- `README.md` ❌ — PENDIENTE

## Último commit en GitHub
`feat(fase6): docker-compose + tests integración API (17/17 passing)` (rama dev)

## Próximos pasos (en orden)
1. Escribir `src/reporting/pdf_report_generator.py` — genera PDF del expediente
2. Escribir `tests/test_pipeline.py` — tests del pipeline sin modelos
3. Escribir `notebooks/01_exploracion_datos.ipynb` — análisis del dataset
4. Escribir `README.md` — documentación del proyecto

## Decisiones tomadas
- DNI: _SCALE=3 → 1518×957 px, _B=1.5 para escalar todo el layout
- Formulario: _SCALE=2 → 1588×2246 px
- ResNet-18: solo layer4 + fc entrenables (fine-tuning parcial)
- Veredicto: 70% peso reglas + 30% peso confianza OCR
- Tolerancia similitud nombres: 85% (Levenshtein normalizado)
- Dataset: 400 expedientes, 15% inconsistentes, splits 70/17.5/12.5
- API: FastAPI + registro directo en historial/métricas desde verification.py
- Tests: TestClient de FastAPI con pipeline sin modelos (YOLO/OCR deshabilitados)
