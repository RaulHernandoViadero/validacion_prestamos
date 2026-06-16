"""
verification.py — Router FastAPI para el endpoint de verificación documental.

Expone el endpoint principal del sistema:

  POST /verify
    Recibe las imágenes del DNI y del formulario de préstamo como
    ficheros multipart/form-data, ejecuta el pipeline completo y
    devuelve el veredicto con todos los artefactos.

El pipeline se instancia una única vez en el arranque de la aplicación
(lifespan) y se reutiliza en cada request (lazy loading de modelos).

Uso:
    from api.routers.verification import router
    app.include_router(router, prefix="/api/v1")
"""

import io
import logging
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from api.schemas.models import ErrorResponse, VerificacionResponse
from src.pipeline.document_pipeline import DocumentPipeline, PipelineConfig

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/verify",
    tags=["Verificación"],
)

# Pipeline singleton — se inyecta desde api/main.py (lifespan)
_pipeline: Optional[DocumentPipeline] = None


def set_pipeline(pipeline: DocumentPipeline) -> None:
    """Inyecta el pipeline desde el lifespan de la app."""
    global _pipeline
    _pipeline = pipeline


def get_pipeline() -> DocumentPipeline:
    """
    Devuelve el pipeline activo o crea uno nuevo si aún no existe.

    En producción el pipeline se inyecta en el arranque. Este fallback
    permite usar el router en tests sin arrancar el lifespan completo.
    """
    global _pipeline
    if _pipeline is None:
        logger.info("Pipeline no inyectado — creando instancia por defecto")
        _pipeline = DocumentPipeline()
    return _pipeline


# ── Constantes ────────────────────────────────────────────────────────────────

_FORMATOS_VALIDOS   = {"image/jpeg", "image/png", "image/webp", "image/tiff"}
_TAMANO_MAX_BYTES   = 20 * 1024 * 1024   # 20 MB


# ── Utilidades ────────────────────────────────────────────────────────────────

def _leer_imagen(fichero: UploadFile, nombre_campo: str) -> Image.Image:
    """
    Lee un fichero subido como imagen PIL y valida su formato.

    Args:
        fichero: Fichero subido por el cliente (UploadFile de FastAPI).
        nombre_campo: Nombre del campo para mensajes de error claros.

    Returns:
        Imagen PIL en modo RGB.

    Raises:
        HTTPException 422 si el formato no es válido o el fichero es muy grande.
    """
    # Validar tipo MIME
    if fichero.content_type and fichero.content_type not in _FORMATOS_VALIDOS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ErrorResponse(
                codigo="FORMATO_IMAGEN_INVALIDO",
                mensaje=(
                    f"El campo '{nombre_campo}' tiene formato no soportado: "
                    f"'{fichero.content_type}'. Use JPEG, PNG, WebP o TIFF."
                ),
            ).model_dump(),
        )

    try:
        contenido = fichero.file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ErrorResponse(
                codigo="ERROR_LECTURA_IMAGEN",
                mensaje=f"No se pudo leer el fichero '{nombre_campo}': {exc}",
            ).model_dump(),
        ) from exc

    # Validar tamaño
    if len(contenido) > _TAMANO_MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=ErrorResponse(
                codigo="IMAGEN_DEMASIADO_GRANDE",
                mensaje=(
                    f"La imagen '{nombre_campo}' supera el límite de "
                    f"{_TAMANO_MAX_BYTES // (1024*1024)} MB."
                ),
            ).model_dump(),
        )

    # Decodificar imagen
    try:
        imagen = Image.open(io.BytesIO(contenido)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ErrorResponse(
                codigo="IMAGEN_NO_RECONOCIDA",
                mensaje=(
                    f"'{nombre_campo}' no es una imagen válida. "
                    "Asegúrese de subir un archivo de imagen no corrupto."
                ),
            ).model_dump(),
        ) from exc

    return imagen


def _construir_response(resultado) -> VerificacionResponse:
    """
    Convierte un ExpedienteResult en un VerificacionResponse Pydantic.

    Transforma los dataclasses del pipeline en los schemas de la API,
    garantizando que el JSON de salida sea siempre válido.
    """
    from api.schemas.models import (
        ValidacionSchema, ResultadoReglaSchema,
        AutenticidadSchema, PerfilRiesgoSchema, CampoOCRSchema,
    )

    # Validación
    validacion_schema = None
    if resultado.resultado_validacion:
        rv = resultado.resultado_validacion
        validacion_schema = ValidacionSchema(
            es_consistente=rv.es_consistente,
            n_reglas_ok=rv.n_reglas_ok,
            n_reglas_fallo=rv.n_reglas_fallo,
            reglas_fallidas=rv.reglas_fallidas,
            reglas=[
                ResultadoReglaSchema(
                    codigo=r.codigo,
                    nombre=r.nombre,
                    pasada=r.pasada,
                    detalle=r.detalle,
                    severidad=r.severidad,
                )
                for r in rv.reglas
            ],
        )

    # Autenticidad
    def _aut(d: Optional[dict]) -> Optional[AutenticidadSchema]:
        if not d:
            return None
        return AutenticidadSchema(
            etiqueta=d.get("etiqueta", "DESCONOCIDO"),
            confianza=d.get("confianza", 0.0),
            prob_legitimo=d.get("prob_legitimo", 0.0),
            prob_manipulado=d.get("prob_manipulado", 0.0),
        )

    # Perfil de riesgo
    perfil_schema = None
    if resultado.perfil_riesgo:
        pr = resultado.perfil_riesgo
        perfil_schema = PerfilRiesgoSchema(
            nivel=pr.nivel,
            ratio_deuda=pr.ratio_deuda,
            detalle=pr.detalle,
        )

    # Campos OCR
    def _campos(ocr) -> dict:
        if not ocr:
            return {}
        return {
            k: CampoOCRSchema(
                clase=v.clase,
                texto_norm=v.texto_norm,
                confianza_ocr=v.confianza_ocr,
                valido=v.valido,
                mensaje_error=v.mensaje_error,
            )
            for k, v in ocr.campos.items()
        }

    veredicto = resultado.veredicto
    return VerificacionResponse(
        expediente_id=veredicto.expediente_id if veredicto else resultado.expediente_id,
        resultado=veredicto.resultado if veredicto else "ERROR",
        es_apto=veredicto.es_apto if veredicto else False,
        confianza=veredicto.confianza if veredicto else 0.0,
        validacion=validacion_schema,
        autenticidad_dni=_aut(resultado.autenticidad_dni),
        autenticidad_form=_aut(resultado.autenticidad_form),
        campos_dni=_campos(resultado.ocr_dni),
        campos_form=_campos(resultado.ocr_form),
        perfil_riesgo=perfil_schema,
        mrz_checkdigits_ok=resultado.mrz_checkdigits_ok,
        cuota_coherente=resultado.cuota_coherente,
        tiempo_total_ms=resultado.tiempo_total_ms,
        tiempos_etapas=resultado.tiempos_etapas,
        advertencias=resultado.advertencias,
        timestamp=veredicto.timestamp if veredicto else resultado.timestamp,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=VerificacionResponse,
    status_code=status.HTTP_200_OK,
    summary="Verificar expediente de préstamo",
    description="""
Procesa un expediente completo de préstamo hipotecario:

1. **Detección** — YOLOv8 localiza los campos en las imágenes
2. **OCR** — EasyOCR extrae el texto de cada campo
3. **Normalización** — corrección de errores OCR, validación de NIF, fechas, importes
4. **Autenticidad** — ResNet-18 clasifica cada documento como LEGÍTIMO o MANIPULADO
5. **Validación cruzada** — 9 reglas de negocio (R01–R09) comparan DNI y formulario
6. **Veredicto** — APTO PARA TRÁMITE o EXPEDIENTE INCONSISTENTE con nivel de confianza

**Formatos aceptados:** JPEG, PNG, WebP, TIFF (máx. 20 MB por imagen)
    """,
    responses={
        200: {"description": "Verificación completada con éxito"},
        413: {"model": ErrorResponse, "description": "Imagen demasiado grande (>20 MB)"},
        422: {"model": ErrorResponse, "description": "Formato de imagen no válido"},
        500: {"model": ErrorResponse, "description": "Error interno del sistema"},
    },
)
async def verificar_expediente(
    imagen_dni: Annotated[
        UploadFile,
        File(description="Imagen del DNI del solicitante (JPEG/PNG, máx 20 MB)"),
    ],
    imagen_formulario: Annotated[
        UploadFile,
        File(description="Imagen del formulario de préstamo cumplimentado (JPEG/PNG, máx 20 MB)"),
    ],
    expediente_id: Annotated[
        Optional[str],
        Form(description="ID del expediente (se genera automáticamente si no se proporciona)"),
    ] = None,
) -> VerificacionResponse:
    """
    Verifica un expediente de préstamo hipotecario con visión artificial.

    Recibe el DNI y el formulario como imágenes y devuelve el veredicto
    completo con el detalle de cada regla de validación.
    """
    logger.info(
        "POST /verify — expediente=%s dni=%s form=%s",
        expediente_id or "(auto)",
        imagen_dni.filename,
        imagen_formulario.filename,
    )

    # Leer y validar las imágenes
    img_dni  = _leer_imagen(imagen_dni,         "imagen_dni")
    img_form = _leer_imagen(imagen_formulario,  "imagen_formulario")

    # Ejecutar el pipeline
    try:
        pipeline  = get_pipeline()
        resultado = pipeline.procesar(
            imagen_dni=img_dni,
            imagen_form=img_form,
            expediente_id=expediente_id,
        )
    except Exception as exc:
        logger.exception("Error inesperado en el pipeline: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                codigo="ERROR_PIPELINE",
                mensaje="Error interno al procesar el expediente. Inténtelo de nuevo.",
                detalle=str(exc),
            ).model_dump(),
        ) from exc

    # Construir la respuesta
    response = _construir_response(resultado)

    # Registrar en historial y métricas (importación local para evitar
    # importaciones circulares en el arranque del módulo)
    try:
        from api.routers.history import registrar_expediente
        from api.routers.metrics import registrar_resultado
        registrar_expediente(response)
        registrar_resultado(response)
    except Exception as exc:
        logger.warning("No se pudo registrar en historial/métricas: %s", exc)

    return response


@router.get(
    "/health",
    response_model=None,
    summary="Estado del servicio de verificación",
    tags=["Sistema"],
    include_in_schema=False,
)
async def health_verificacion() -> dict:
    """Endpoint de salud específico del router de verificación."""
    p = get_pipeline()
    return {
        "router": "verification",
        "pipeline_activo": p is not None,
        "yolo_cargado":    p._yolo_dni is not None if p else False,
        "ocr_cargado":     p._ocr_engine is not None if p else False,
        "cls_cargado":     p._clasificador is not None if p else False,
    }
