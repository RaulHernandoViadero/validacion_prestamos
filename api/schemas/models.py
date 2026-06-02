"""
models.py — Esquemas Pydantic para la API REST de verificación documental.

Define los modelos de entrada y salida de todos los endpoints:

  POST /verify      → VerificacionRequest  → VerificacionResponse
  GET  /history     → (query params)       → HistorialResponse
  GET  /metrics     → (query params)       → MetricasResponse

Los modelos usan Pydantic v2 con validadores estrictos y ejemplos
integrados en el schema OpenAPI para facilitar la documentación.

Uso típico:
    from api.schemas.models import VerificacionResponse
    response = VerificacionResponse(**resultado.to_dict())
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Schemas compartidos ───────────────────────────────────────────────────────

class CampoOCRSchema(BaseModel):
    """Campo de texto extraído y normalizado por OCR."""
    clase:         str   = Field(description="Clase YOLO del campo (ej. 'numero_dni')")
    texto_norm:    str   = Field(description="Texto normalizado")
    confianza_ocr: float = Field(ge=0.0, le=1.0, description="Confianza del motor OCR")
    valido:        bool  = Field(description="El campo supera la validación semántica")
    mensaje_error: str   = Field(default="", description="Detalle del error si no es válido")

    model_config = {"from_attributes": True}


class ResultadoReglaSchema(BaseModel):
    """Resultado de una regla de validación cruzada R01–R09."""
    codigo:    str  = Field(description="Código de la regla (R01–R09)")
    nombre:    str  = Field(description="Descripción de la regla")
    pasada:    bool = Field(description="True si la regla se cumple")
    detalle:   str  = Field(description="Valores comparados y resultado")
    severidad: str  = Field(default="ERROR", description="ERROR | WARNING")


class ValidacionSchema(BaseModel):
    """Resultado completo de la validación cruzada."""
    es_consistente:  bool                      = Field(description="True si todas las reglas pasan")
    n_reglas_ok:     int                       = Field(ge=0, le=9)
    n_reglas_fallo:  int                       = Field(ge=0, le=9)
    reglas_fallidas: list[str]                 = Field(description="Códigos de las reglas fallidas")
    reglas:          list[ResultadoReglaSchema] = Field(description="Detalle de cada regla")


class AutenticidadSchema(BaseModel):
    """Resultado del clasificador ResNet-18 de autenticidad."""
    etiqueta:  str   = Field(description="LEGÍTIMO | MANIPULADO")
    confianza: float = Field(ge=0.0, le=1.0)
    prob_legitimo:   float = Field(ge=0.0, le=1.0, default=0.0)
    prob_manipulado: float = Field(ge=0.0, le=1.0, default=0.0)


class PerfilRiesgoSchema(BaseModel):
    """Perfil de riesgo crediticio del solicitante."""
    nivel:       str   = Field(description="BAJO | MEDIO | ALTO")
    ratio_deuda: float = Field(ge=0.0, description="Cuota / ingresos netos")
    detalle:     str   = Field(description="Descripción del perfil")


class VeredictoSchema(BaseModel):
    """Veredicto final del sistema de verificación."""
    expediente_id:      str   = Field(description="Identificador del expediente")
    resultado:          str   = Field(description="APTO PARA TRÁMITE | EXPEDIENTE INCONSISTENTE")
    es_apto:            bool  = Field(description="True si el expediente es apto")
    confianza:          float = Field(ge=0.0, le=1.0, description="Confianza global del sistema")
    reglas_fallidas:    list[str] = Field(default_factory=list)
    resumen_fallos:     list[str] = Field(default_factory=list)
    confianza_ocr_media: float   = Field(ge=0.0, le=1.0, default=0.0)
    timestamp:          str


# ── Endpoint POST /verify ─────────────────────────────────────────────────────

class VerificacionRequest(BaseModel):
    """
    Solicitud de verificación de un expediente de préstamo.

    Ambas imágenes deben enviarse como ficheros multipart/form-data
    (gestionado por FastAPI directamente), por lo que este schema
    se usa para validar los campos de texto opcionales.
    """
    expediente_id: Optional[str] = Field(
        default=None,
        description="ID del expediente (se genera automáticamente si no se proporciona)",
        examples=["EXP-2024-0042"],
    )

    @field_validator("expediente_id")
    @classmethod
    def validar_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) > 64:
            raise ValueError("El expediente_id no puede superar 64 caracteres")
        return v


class VerificacionResponse(BaseModel):
    """
    Respuesta completa de la verificación de un expediente.

    Incluye el veredicto final, el detalle de cada regla,
    los campos extraídos por OCR y los metadatos de rendimiento.
    """
    # Veredicto
    expediente_id:     str
    resultado:         str   = Field(description="APTO PARA TRÁMITE | EXPEDIENTE INCONSISTENTE")
    es_apto:           bool
    confianza:         float = Field(ge=0.0, le=1.0)

    # Validación y autenticidad
    validacion:        Optional[ValidacionSchema]    = None
    autenticidad_dni:  Optional[AutenticidadSchema]  = None
    autenticidad_form: Optional[AutenticidadSchema]  = None

    # Campos OCR
    campos_dni:        dict[str, CampoOCRSchema]     = Field(default_factory=dict)
    campos_form:       dict[str, CampoOCRSchema]     = Field(default_factory=dict)

    # Reglas de negocio complementarias
    perfil_riesgo:        Optional[PerfilRiesgoSchema] = None
    mrz_checkdigits_ok:   bool                         = True
    cuota_coherente:      bool                         = True

    # Métricas de rendimiento
    tiempo_total_ms:   float                           = 0.0
    tiempos_etapas:    dict[str, float]                = Field(default_factory=dict)

    # Advertencias no fatales
    advertencias:      list[str]                       = Field(default_factory=list)

    # Timestamp
    timestamp:         str = Field(default_factory=lambda: datetime.now().isoformat())

    model_config = {
        "json_schema_extra": {
            "example": {
                "expediente_id": "EXP-2024-0042",
                "resultado": "APTO PARA TRÁMITE",
                "es_apto": True,
                "confianza": 0.912,
                "validacion": {
                    "es_consistente": True,
                    "n_reglas_ok": 9,
                    "n_reglas_fallo": 0,
                    "reglas_fallidas": [],
                    "reglas": [],
                },
                "tiempo_total_ms": 1243.5,
                "advertencias": [],
                "timestamp": "2024-11-15T10:23:45.123456",
            }
        }
    }


# ── Endpoint GET /history ─────────────────────────────────────────────────────

class ExpedienteResumen(BaseModel):
    """Resumen de un expediente en el historial."""
    expediente_id:  str
    resultado:      str
    es_apto:        bool
    confianza:      float
    timestamp:      str
    reglas_fallidas: list[str] = Field(default_factory=list)


class HistorialResponse(BaseModel):
    """Respuesta del endpoint GET /history."""
    total:       int              = Field(description="Total de expedientes en el historial")
    pagina:      int              = Field(ge=1, description="Página actual")
    por_pagina:  int              = Field(ge=1, le=100)
    expedientes: list[ExpedienteResumen]

    model_config = {
        "json_schema_extra": {
            "example": {
                "total": 42,
                "pagina": 1,
                "por_pagina": 20,
                "expedientes": [
                    {
                        "expediente_id": "EXP-2024-0042",
                        "resultado": "APTO PARA TRÁMITE",
                        "es_apto": True,
                        "confianza": 0.912,
                        "timestamp": "2024-11-15T10:23:45",
                        "reglas_fallidas": [],
                    }
                ],
            }
        }
    }


# ── Endpoint GET /metrics ─────────────────────────────────────────────────────

class MetricasResponse(BaseModel):
    """
    Métricas agregadas del sistema desde su arranque.

    Útil para monitorización y para la demo del TFM.
    """
    # Contadores
    total_expedientes:    int   = 0
    expedientes_aptos:    int   = 0
    expedientes_rechazados: int = 0
    tasa_aprobacion:      float = Field(ge=0.0, le=1.0, default=0.0)

    # Confianza media
    confianza_media:      float = Field(ge=0.0, le=1.0, default=0.0)
    confianza_ocr_media:  float = Field(ge=0.0, le=1.0, default=0.0)

    # Reglas — cuántas veces ha fallado cada una
    fallos_por_regla:     dict[str, int] = Field(
        default_factory=dict,
        description="Número de veces que ha fallado cada regla R01–R09",
    )

    # Rendimiento
    tiempo_medio_ms:      float = 0.0
    tiempo_p95_ms:        float = 0.0

    # Ventana temporal
    desde:                Optional[str] = None   # ISO 8601
    hasta:                Optional[str] = None   # ISO 8601

    model_config = {
        "json_schema_extra": {
            "example": {
                "total_expedientes": 150,
                "expedientes_aptos": 127,
                "expedientes_rechazados": 23,
                "tasa_aprobacion": 0.847,
                "confianza_media": 0.883,
                "confianza_ocr_media": 0.791,
                "fallos_por_regla": {"R07": 12, "R05": 5, "R09": 6},
                "tiempo_medio_ms": 1320.4,
                "tiempo_p95_ms": 2100.0,
            }
        }
    }


# ── Respuestas de error ───────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Respuesta de error estándar de la API."""
    codigo:  str = Field(description="Código de error interno")
    mensaje: str = Field(description="Descripción legible del error")
    detalle: Optional[Any] = Field(default=None, description="Información adicional")

    model_config = {
        "json_schema_extra": {
            "example": {
                "codigo": "FORMATO_IMAGEN_INVALIDO",
                "mensaje": "La imagen del DNI no pudo ser leída. Asegúrese de subir un PNG o JPEG válido.",
                "detalle": None,
            }
        }
    }


class HealthResponse(BaseModel):
    """Respuesta del endpoint GET /health."""
    status:        str  = "ok"
    version:       str
    yolo_cargado:  bool = False
    ocr_cargado:   bool = False
    cls_cargado:   bool = False
    timestamp:     str  = Field(default_factory=lambda: datetime.now().isoformat())
