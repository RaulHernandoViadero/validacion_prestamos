# Verificación Documental de Identidad y Solicitudes de Préstamo mediante Visión Artificial y Deep Learning

**Trabajo de Fin de Máster — Máster en Inteligencia Artificial**  
Autor: Raúl Hernando Viadero

---

## Descripción

Sistema end-to-end de verificación documental que combina **detección de objetos (YOLOv8n)**, **OCR (EasyOCR)** y **clasificación de autenticidad (ResNet-18)** para validar automáticamente expedientes de préstamo hipotecario formados por:

- DNI español — 9 campos detectados, MRZ ICAO 9303 TD1
- Formulario de préstamo — 14 campos, amortización francesa

El sistema emite un veredicto final (**APTO PARA TRÁMITE** / **EXPEDIENTE INCONSISTENTE**) con un nivel de confianza calculado a partir de 9 reglas de negocio cruzadas (R01–R09) y las confianzas OCR.

---

## Arquitectura

```
Imagen DNI + Imagen Formulario
        │
        ▼
┌─────────────────────────────────────┐
│  YOLOv8n  — Detección de ROIs       │  mAP50 = 99.5%
│  9 clases DNI / 14 clases formulario│
└──────────────────┬──────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────┐    ┌──────────────────┐
│  EasyOCR     │    │  ResNet-18       │
│  es + en     │    │  Autenticidad    │
│  Texto → str │    │  LEGITIMO /      │
└──────┬───────┘    │  MANIPULADO      │
       │            └────────┬─────────┘
       ▼                     │
┌──────────────────────────────────────┐
│  TextPostprocessor                   │
│  NIF · Fechas · Importes · Nombres   │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  CrossValidator — Reglas R01–R09     │
│  R01 Nombre    R02 Apellidos         │
│  R03 NIF       R04 Fecha nacimiento  │
│  R05 Caducidad R06 Mayoria de edad   │
│  R07 Ratio deuda <= 35%              │
│  R08 Importe valido                  │
│  R09 Autenticidad documentos         │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  VerdictEngine                       │
│  0.70 x reglas_ok + 0.30 x conf_OCR  │
│  APTO PARA TRAMITE / INCONSISTENTE   │
└──────────────────────────────────────┘
```

---

## Resultados de los modelos

| Modelo | mAP50 | mAP50-95 | Precision | Recall |
|--------|-------|----------|-----------|--------|
| YOLOv8n DNI | **99.5%** | 98.9% | 99.9% | 100% |
| YOLOv8n Formulario | **99.5%** | 95.5% | 99.7% | 100% |

Dataset: **400 expedientes sinteticos** · splits 70 / 17.5 / 12.5 · aumentacion x3 en train

---

## Estructura del proyecto

```
validacion_prestamos/
├── src/
│   ├── config.py                      # Hiperparametros y rutas centralizados
│   ├── data_generation/
│   │   ├── fake_data_factory.py       # NIF, MRZ ICAO, datos sinteticos (Faker)
│   │   ├── dni_generator.py           # Imagenes DNI 1518x957 px
│   │   ├── loan_form_generator.py     # Formularios 1588x2246 px
│   │   ├── augmentation.py            # 11 transformaciones
│   │   └── annotation_writer.py       # Anotaciones YOLO .txt + YAML
│   ├── models/
│   │   ├── yolo_trainer.py            # Entrenamiento YOLOv8n
│   │   ├── yolo_inference.py          # Inferencia YOLO + ROIDetectado
│   │   ├── authenticity_classifier.py # ResNet-18 fine-tuning
│   │   └── model_utils.py             # Utilidades compartidas
│   ├── ocr/
│   │   ├── easyocr_engine.py          # Singleton EasyOCR (es+en)
│   │   └── text_postprocessor.py      # Normalizacion NIF/fechas/importes
│   ├── validation/
│   │   ├── business_rules.py          # MRZ check digits, cuota francesa
│   │   ├── cross_validator.py         # Reglas R01-R09
│   │   └── verdict_engine.py          # Veredicto final con confianza
│   ├── pipeline/
│   │   └── document_pipeline.py       # Orquestador end-to-end
│   └── reporting/
│       └── pdf_report_generator.py    # Informes PDF con reportlab
├── api/
│   ├── main.py                        # FastAPI app (lifespan, CORS, /health)
│   ├── schemas/models.py              # Pydantic v2 request/response
│   ├── routers/
│   │   ├── verification.py            # POST /api/v1/verify
│   │   ├── history.py                 # GET/DELETE /api/v1/history
│   │   └── metrics.py                 # GET /api/v1/metrics
│   └── Dockerfile                     # Multi-stage python:3.11-slim
├── app/
│   ├── streamlit_app.py               # Frontend principal
│   ├── _pages/
│   │   ├── verificacion.py            # Pagina subida + veredicto
│   │   ├── historial.py               # Pagina historial paginado
│   │   └── metricas.py                # Pagina KPIs del sistema
│   └── Dockerfile
├── tests/
│   ├── test_fake_data_factory.py      # 24 tests datos sinteticos
│   ├── test_pipeline.py               # 36 tests pipeline + reglas
│   └── test_api.py                    # 17 tests integracion FastAPI
├── generate_dataset.py                # CLI generacion dataset
├── docker-compose.yml                 # API (8000) + Frontend (8501)
├── requirements.txt
└── weights/                           # Pesos entrenados (.pt / .pth)
```

---

## Inicio rapido

### Con Docker (recomendado)

Requisitos: Docker Desktop con WSL2

```bash
git clone https://github.com/RaulHernandoViadero/validacion_prestamos.git
cd validacion_prestamos
docker compose up --build
```

- Frontend: http://localhost:8501
- API + Swagger: http://localhost:8000/docs

### Sin Docker (Anaconda Python)

```bash
pip install -r requirements.txt

# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Frontend
streamlit run app/streamlit_app.py
```

---

## Generacion del dataset sintetico

```bash
python generate_dataset.py --size 400 --seed 42 --aug-variants 3 --aug-intensity media
```

Todos los datos son 100% sinteticos (Faker `es_ES`) — cumple el RGPD.  
Los NIFs se validan con el algoritmo oficial espanol: `letra = "TRWAGMYFPDXBNJZSQVHLCKE"[numero % 23]`  
La MRZ sigue el estandar ICAO 9303 TD1 (30 chars x 2 lineas, digitos de control ICAO).

---

## Entrenamiento de modelos

```bash
# YOLO — deteccion de ROIs (requiere dataset generado)
python -c "from src.models.yolo_trainer import YOLOTrainer; YOLOTrainer().entrenar('dni')"
python -c "from src.models.yolo_trainer import YOLOTrainer; YOLOTrainer().entrenar('prestamo')"

# ResNet-18 — clasificador de autenticidad
python -c "
from src.models.authenticity_classifier import AuthenticityClassifier
from pathlib import Path
base = Path('data/splits/clf_autenticidad')
AuthenticityClassifier().entrenar(train_dir=base/'train', val_dir=base/'val')
"
```

Los pesos se guardan automaticamente en `weights/`.

---

## Tests

```bash
pytest tests/ -v
# 77 tests · 0 fallos · ~7 segundos
```

| Suite | Tests | Cobertura |
|-------|-------|-----------|
| `test_fake_data_factory.py` | 24 | NIF, MRZ, fechas, prestamos |
| `test_pipeline.py` | 36 | Pipeline, reglas R01-R09, veredicto |
| `test_api.py` | 17 | Endpoints REST, historial, metricas |

---

## API REST

| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| `POST` | `/api/v1/verify/` | Verificar expediente (DNI + formulario) |
| `GET` | `/api/v1/history/` | Historial paginado con filtros |
| `GET` | `/api/v1/history/{id}` | Detalle de un expediente |
| `DELETE` | `/api/v1/history/{id}` | Eliminar expediente |
| `GET` | `/api/v1/metrics/` | KPIs del sistema |
| `GET` | `/health` | Estado y modelos cargados |

Documentacion interactiva: http://localhost:8000/docs

---

## Decisiones de diseno

| Decision | Valor | Motivo |
|----------|-------|--------|
| Resolucion DNI | 1518x957 px | Legibilidad OCR en campos pequeños |
| Resolucion formulario | 1588x2246 px | A4 a 96 dpi x escala 2 |
| ResNet-18 capas entrenables | `layer4` + `fc` | Fine-tuning parcial, evita overfitting |
| Peso veredicto | 70% reglas + 30% OCR | Prioridad a consistencia documental |
| Tolerancia Levenshtein | 85% | Robustez ante errores OCR de 1 caracter |
| Aumentacion train | x3 variantes | Rotacion, perspectiva, ruido, oclusion |
| `fliplr=0, flipud=0` en YOLO | desactivado | Texto y MRZ no deben invertirse |

---

## Tecnologias

Python 3.11 · YOLOv8n (ultralytics) · EasyOCR · PyTorch / ResNet-18 · FastAPI · Streamlit · Pydantic v2 · Faker es_ES · Pillow · reportlab · Docker / WSL2
