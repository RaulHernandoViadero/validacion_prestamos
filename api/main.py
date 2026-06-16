"""
main.py — Aplicación FastAPI del sistema de verificación documental.

Punto de entrada de la API REST. Configura:

  - Lifespan: carga del pipeline al arrancar, liberación al parar
  - Middleware: CORS, logging de requests, límite de tamaño
  - Routers: /verify, /history, /metrics
  - Documentación: Swagger UI en /docs, ReDoc en /redoc
  - Manejo de errores: excepciones globales con respuesta JSON uniforme

Arranque local (desarrollo):
    uvicorn api.main:app --reload --port 8000

Producción (Docker):
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import history, metrics, verification
from api.routers.verification import set_pipeline
from api.schemas.models import ErrorResponse, HealthResponse
from src.pipeline.document_pipeline import DocumentPipeline, PipelineConfig

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Versión ───────────────────────────────────────────────────────────────────

_VERSION = "1.0.0"


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Gestiona el ciclo de vida de la aplicación:
      - Arranque: instancia el pipeline (carga lazy de YOLO, OCR, ResNet-18)
      - Parada: libera recursos
    """
    logger.info("=== Arrancando API de Verificación Documental v%s ===", _VERSION)

    # Crear pipeline con configuración de producción
    config = PipelineConfig(
        usar_yolo=True,
        usar_clasificador=True,
        anotar_imagen=True,
        tolerancia_nombres=0.85,
        umbral_confianza_ocr=0.50,
    )
    pipeline = DocumentPipeline(config=config)
    set_pipeline(pipeline)
    logger.info("Pipeline de verificación instanciado (carga de modelos lazy)")

    # Guardar referencia en el estado de la app para acceso desde tests
    app.state.pipeline = pipeline

    yield  # ← La app atiende requests entre yield y el bloque de parada

    logger.info("=== Deteniendo API de Verificación Documental ===")
    # Liberar recursos GPU si procede
    if pipeline._clasificador is not None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# ── Aplicación ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="API de Verificación Documental",
    description="""
## Sistema de Verificación Documental de Identidad y Solicitudes de Préstamo

API REST del TFM **"Verificación Documental de Identidad y Solicitudes de Préstamo
mediante Visión Artificial y Deep Learning"** (Máster en Inteligencia Artificial).

### Flujo de verificación

```
Imagen DNI + Imagen Formulario
    │
    ├─ YOLOv8n ──────── Detección de ROIs (9 clases DNI, 14 clases formulario)
    ├─ EasyOCR ──────── Extracción de texto (español + inglés)
    ├─ ResNet-18 ─────── Clasificación de autenticidad (LEGÍTIMO / MANIPULADO)
    ├─ CrossValidator ── Validación cruzada R01–R09
    └─ VerdictEngine ─── Veredicto final con nivel de confianza
```

### Veredictos posibles

- ✅ **APTO PARA TRÁMITE** — todas las reglas superadas
- ❌ **EXPEDIENTE INCONSISTENTE** — al menos una regla ha fallado

### Autenticación

Esta versión del TFM no requiere autenticación. En producción se añadiría
un middleware JWT o API-key.
    """,
    version=_VERSION,
    contact={
        "name": "Raúl Hernando Viadero",
        "email": "raulhervia@gmail.com",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ── Middleware ────────────────────────────────────────────────────────────────

# CORS — permite peticiones desde el frontend Streamlit y desarrollo local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # En producción: restringir al dominio del frontend
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Registra cada request con su tiempo de respuesta."""
    t_ini = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t_ini) * 1000
    logger.info(
        "%s %s → %d (%.0f ms)",
        request.method, request.url.path, response.status_code, ms,
    )
    return response


# ── Manejo global de excepciones ──────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Captura cualquier excepción no manejada y devuelve JSON uniforme."""
    logger.exception("Excepción no manejada en %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            codigo="ERROR_INTERNO",
            mensaje="Error interno del servidor. Revise los logs para más detalles.",
            detalle=str(exc) if app.debug else None,
        ).model_dump(),
    )


# ── Endpoints de sistema ──────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Sistema"],
    summary="Estado del sistema",
    description="Comprueba que la API está activa y qué modelos están cargados.",
)
async def health() -> HealthResponse:
    """Endpoint de salud — usado por Docker/Kubernetes para liveness probes."""
    pipeline = getattr(app.state, "pipeline", None)
    return HealthResponse(
        status="ok",
        version=_VERSION,
        yolo_cargado=pipeline._yolo_dni is not None if pipeline else False,
        ocr_cargado=pipeline._ocr_engine is not None if pipeline else False,
        cls_cargado=pipeline._clasificador is not None if pipeline else False,
    )


@app.get(
    "/",
    include_in_schema=False,
)
async def root():
    """Redirige a la documentación Swagger."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


# ── Routers ───────────────────────────────────────────────────────────────────

# Prefijo común de versión: /api/v1
_PREFIX = "/api/v1"

# POST /api/v1/verify — verificación de expedientes
app.include_router(verification.router, prefix=_PREFIX)

# GET /api/v1/history — historial de verificaciones
app.include_router(history.router, prefix=_PREFIX)

# GET /api/v1/metrics — métricas del sistema
app.include_router(metrics.router, prefix=_PREFIX)


# El registro en historial y métricas se realiza directamente en
# api/routers/verification.py tras cada verificación exitosa.


# ── Arranque directo (desarrollo) ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
