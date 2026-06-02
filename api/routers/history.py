"""
history.py — Router FastAPI para el historial de expedientes verificados.

Expone los endpoints:

  GET /history          → lista paginada de expedientes
  GET /history/{id}     → detalle completo de un expediente
  DELETE /history/{id}  → elimina un expediente del historial

El historial se almacena en memoria durante la sesión de la aplicación.
En una versión de producción se reemplazaría por una base de datos
(SQLite, PostgreSQL…), pero para el TFM el almacenamiento en memoria
es suficiente para demostrar la funcionalidad.

Uso:
    from api.routers.history import router
    app.include_router(router, prefix="/api/v1")
"""

import logging
from collections import deque
from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Query, status

from api.schemas.models import (
    ErrorResponse,
    ExpedienteResumen,
    HistorialResponse,
    VerificacionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/history",
    tags=["Historial"],
)

# ── Almacén en memoria ────────────────────────────────────────────────────────

# Cola FIFO con un límite máximo de expedientes en memoria
_MAX_EXPEDIENTES = 1000
_historial: deque[VerificacionResponse] = deque(maxlen=_MAX_EXPEDIENTES)


def registrar_expediente(response: VerificacionResponse) -> None:
    """
    Añade un expediente verificado al historial.

    Llamado desde el router de verificación tras cada POST /verify exitoso.

    Args:
        response: Resultado completo de la verificación.
    """
    _historial.appendleft(response)   # más reciente primero
    logger.debug(
        "Expediente %s registrado en historial (%d total)",
        response.expediente_id, len(_historial),
    )


def limpiar_historial() -> None:
    """Vacía el historial (útil para tests)."""
    _historial.clear()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=HistorialResponse,
    summary="Listar expedientes verificados",
    description="""
Devuelve la lista paginada de los expedientes procesados en esta sesión,
ordenados del más reciente al más antiguo.

Se puede filtrar por resultado (apto / rechazado) y por rango de fechas.
    """,
)
async def listar_historial(
    pagina: Annotated[int, Query(ge=1, description="Número de página")] = 1,
    por_pagina: Annotated[int, Query(ge=1, le=100, description="Resultados por página")] = 20,
    solo_aptos: Annotated[
        Optional[bool],
        Query(description="Si se especifica, filtra por resultado (True=aptos, False=rechazados)"),
    ] = None,
    desde: Annotated[
        Optional[str],
        Query(description="Filtrar expedientes desde esta fecha (ISO 8601: YYYY-MM-DDTHH:MM:SS)"),
    ] = None,
    hasta: Annotated[
        Optional[str],
        Query(description="Filtrar expedientes hasta esta fecha (ISO 8601)"),
    ] = None,
) -> HistorialResponse:
    """Lista paginada del historial de verificaciones."""
    items: list[VerificacionResponse] = list(_historial)

    # Filtro por resultado
    if solo_aptos is not None:
        items = [e for e in items if e.es_apto == solo_aptos]

    # Filtro por rango de fechas
    if desde:
        try:
            dt_desde = datetime.fromisoformat(desde)
            items = [e for e in items if datetime.fromisoformat(e.timestamp) >= dt_desde]
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ErrorResponse(
                    codigo="FECHA_INVALIDA",
                    mensaje=f"Formato de fecha inválido en 'desde': '{desde}'. Use ISO 8601.",
                ).model_dump(),
            )

    if hasta:
        try:
            dt_hasta = datetime.fromisoformat(hasta)
            items = [e for e in items if datetime.fromisoformat(e.timestamp) <= dt_hasta]
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ErrorResponse(
                    codigo="FECHA_INVALIDA",
                    mensaje=f"Formato de fecha inválido en 'hasta': '{hasta}'. Use ISO 8601.",
                ).model_dump(),
            )

    # Paginación
    total   = len(items)
    inicio  = (pagina - 1) * por_pagina
    fin     = inicio + por_pagina
    pagina_items = items[inicio:fin]

    # Convertir a resúmenes
    resumenes = [
        ExpedienteResumen(
            expediente_id=e.expediente_id,
            resultado=e.resultado,
            es_apto=e.es_apto,
            confianza=e.confianza,
            timestamp=e.timestamp,
            reglas_fallidas=(
                e.validacion.reglas_fallidas if e.validacion else []
            ),
        )
        for e in pagina_items
    ]

    return HistorialResponse(
        total=total,
        pagina=pagina,
        por_pagina=por_pagina,
        expedientes=resumenes,
    )


@router.get(
    "/{expediente_id}",
    response_model=VerificacionResponse,
    summary="Obtener expediente por ID",
    responses={
        404: {"model": ErrorResponse, "description": "Expediente no encontrado"},
    },
)
async def obtener_expediente(expediente_id: str) -> VerificacionResponse:
    """Devuelve el detalle completo de un expediente por su ID."""
    for expediente in _historial:
        if expediente.expediente_id == expediente_id:
            return expediente

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=ErrorResponse(
            codigo="EXPEDIENTE_NO_ENCONTRADO",
            mensaje=f"No se encontró el expediente '{expediente_id}' en el historial.",
        ).model_dump(),
    )


@router.delete(
    "/{expediente_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Eliminar expediente del historial",
    description="Elimina un expediente del historial en memoria. No afecta a bases de datos externas.",
    responses={
        404: {"model": ErrorResponse, "description": "Expediente no encontrado"},
    },
)
async def eliminar_expediente(expediente_id: str) -> None:
    """Elimina un expediente del historial en memoria."""
    global _historial
    antes = len(_historial)
    nueva = deque(
        (e for e in _historial if e.expediente_id != expediente_id),
        maxlen=_MAX_EXPEDIENTES,
    )

    if len(nueva) == antes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                codigo="EXPEDIENTE_NO_ENCONTRADO",
                mensaje=f"No se encontró el expediente '{expediente_id}'.",
            ).model_dump(),
        )

    _historial = nueva
    logger.info("Expediente %s eliminado del historial", expediente_id)
