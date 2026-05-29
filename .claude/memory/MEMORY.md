# Estado del TFM — Verificación Documental

## Repositorio
- GitHub: https://github.com/RaulHernandoViadero/validacion_prestamos
- Rama activa: `dev`
- Python: Anaconda `C:\Users\raulh\anaconda3\python.exe`
- Ejecutar tests: `C:\Users\raulh\anaconda3\python.exe -m pytest tests/ -v`

## Progreso por fases

### ✅ FASE 1 — Generación de datos sintéticos (COMPLETA)
| Fichero | Estado |
|---|---|
| `src/config.py` | ✅ Hiperparámetros centralizados |
| `src/data_generation/fake_data_factory.py` | ✅ NIF, MRZ ICAO, préstamos, 24 tests pasan |
| `src/data_generation/dni_generator.py` | ✅ DNI 1518×957 px, 9 ROIs, layout escalado con _B |
| `src/data_generation/loan_form_generator.py` | ✅ Formulario 1588×2246 px, 14 ROIs |
| `src/data_generation/augmentation.py` | ✅ 11 transformaciones: rotación, ruido, perspectiva, oclusión… |
| `src/data_generation/annotation_writer.py` | ✅ ROIs → YOLO .txt, genera YAML de dataset |
| `generate_dataset.py` | ✅ Script maestro CLI — 0 errores en prueba de 5 expedientes |
| `tests/test_fake_data_factory.py` | ✅ 24/24 tests pasan |

### ⏳ FASE 2 — Entrenamiento de modelos (PENDIENTE)
Siguiente fichero: `src/models/yolo_trainer.py`
- Entrenar YOLOv8n sobre dataset DNI (9 clases) y formulario (14 clases)
- Modelo base: `yolov8n.pt`, 100 epochs, batch 16, imgsz 640
- Después: `src/models/authenticity_classifier.py` (ResNet-18 fine-tuned)

### ⏳ FASE 3 — Pipeline de inferencia (PENDIENTE)
`yolo_inference.py` → `easyocr_engine.py` → `text_postprocessor.py` → `cross_validator.py` → `verdict_engine.py` → `document_pipeline.py`

### ⏳ FASE 4 — API REST FastAPI (PENDIENTE)
### ⏳ FASE 5 — Frontend Streamlit (PENDIENTE)
### ⏳ FASE 6 — Integración, tests, Docker (PENDIENTE)

## Decisiones tomadas
- Datos sintéticos 100% con Faker es_ES + Pillow (sin datos reales)
- DNI: _SCALE=3 → 1518×957 px (legible para OCR y humanos)
- Formulario: _SCALE=2 → 1588×2246 px
- Commits por fichero + push automático a `dev`
- Sin downscale — se guarda a resolución completa de generación
- Dataset: 400 expedientes, 15% inconsistentes, splits 70/17.5/12.5

## Notas importantes
- Las imágenes de preview están en `data/raw/` pero NO se suben al repo (.gitignore)
- Los pesos entrenados tampoco se suben (`weights/`)
- El dataset completo (400 exp) hay que generarlo con: `C:\Users\raulh\anaconda3\python.exe generate_dataset.py --size 400 --seed 42`
- Tarda ~8-10 minutos

## Siguiente sesión — continuar aquí
1. Verificar que el dataset completo se generó OK (`data/splits/` tiene imágenes)
2. Empezar `src/models/yolo_trainer.py`
