"""
metrics.py — Router FastAPI para las métricas del sistema.

Expone:

  GET /metrics          → KPIs agregados desde el arranque
  GET /metrics/reset    → reinicia los contadores (solo en desarrollo)

Las métricas se calculan sobre el historial en memoria. Para el TFM
esto es suficiente; en producción se integraría con Prometheus/Grafana.

Uso:
    from api.routers.metrics import router, registrar_resultado
    app.include_router(router, prefix="/api/v1")
    registrar_resultado(resultado)   # llamar tras cada verificación
"""

import logging
import math
from collections import defaultdict
from typing import Annotated, Optional

from fastapi import APIRouter, Query

from api.schemas.models import MetricasResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/metrics",
    tags=["Métricas"],
)


# ── Acumuladores en memoria ───────────────────────────────────────────────────

class _Acumuladores:
    """Acumula métricas de todas las verificaciones de la sesión."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total            = 0
        self.aptos            = 0
        self.rechazados       = 0
        self.suma_confianza   = 0.0
        self.suma_ocr         = 0.0
        self.fallos_regla: dict[str, int] = defaultdict(int)
        self.tiempos_ms: list[float]      = []

    def registrar(self, response) -> None:
        """
        Registra una respuesta de verificación en los acumuladores.

        Args:
            response: VerificacionResponse del endpoint POST /verify.
        """
        self.total += 1
        if response.es_apto:
            self.aptos += 1
        else:
            self.rechazados += 1

        self.suma_confianza += response.confianza
        if response.validacion:
            # Confianza OCR media del veredicto
            ocr_conf = getattr(response, "confianza_ocr_media", 0.0)
            self.suma_ocr += ocr_conf
            # Contar fallos por regla
            for codigo in (response.validacion.reglas_fallidas or []):
                self.fallos_regla[codigo] += 1

        if response.tiempo_total_ms:
            self.tiempos_ms.append(response.tiempo_total_ms)

    def calcular(
        self,
        desde: Optional[str] = None,
        hasta: Optional[str] = None,
    ) -> MetricasResponse:
        """Calcula el MetricasResponse con los acumuladores actuales."""
        tasa = self.aptos / self.total if self.total else 0.0
        conf_media = self.suma_confianza / self.total if self.total else 0.0
        ocr_media  = self.suma_ocr / self.total if self.total else 0.0

        # Percentil 95 de los tiempos
        tiempos_ord = sorted(self.tiempos_ms)
        t_medio = sum(tiempos_ord) / len(tiempos_ord) if tiempos_ord else 0.0
        if tiempos_ord:
            idx_p95 = min(int(math.ceil(0.95 * len(tiempos_ord))) - 1, len(tiempos_ord) - 1)
            t_p95   = tiempos_ord[idx_p95]
        else:
            t_p95 = 0.0

        return MetricasResponse(
            total_expedientes=self.total,
            expedientes_aptos=self.aptos,
            expedientes_rechazados=self.rechazados,
            tasa_aprobacion=round(tasa, 4),
            confianza_media=round(conf_media, 4),
            confianza_ocr_media=round(ocr_media, 4),
            fallos_por_regla=dict(self.fallos_regla),
            tiempo_medio_ms=round(t_medio, 1),
            tiempo_p95_ms=round(t_p95, 1),
            desde=desde,
            hasta=hasta,
        )


_acumuladores = _Acumuladores()


def registrar_resultado(response) -> None:
    """
    Registra una respuesta en los acumuladores de métricas.

    Debe llamarse desde el router de verificación tras cada POST /verify.

    Args:
        response: VerificacionResponse ya construida.
    """
    _acumuladores.registrar(response)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=MetricasResponse,
    summary="Métricas del sistema",
    description="""
Devuelve los KPIs agregados del sistema desde el arranque:

- Tasa de aprobación de expedientes
- Confianza media del sistema y del OCR
- Número de fallos por regla de validación (R01–R09)
- Tiempos de respuesta: media y percentil 95

Las métricas se reinician con cada reinicio del servidor.
    """,
)
async def obtener_metricas(
    desde: Annotated[
        Optional[str],
        Query(description="Filtrar desde esta fecha (ISO 8601) — informativo, no filtra acumuladores"),
    ] = None,
    hasta: Annotated[
        Optional[str],
        Query(description="Filtrar hasta esta fecha (ISO 8601) — informativo, no filtra acumuladores"),
    ] = None,
) -> MetricasResponse:
    """Devuelve las métricas acumuladas del sistema."""
    return _acumuladores.calcular(desde=desde, hasta=hasta)


@router.post(
    "/reset",
    status_code=200,
    summary="Reiniciar métricas",
    description="Reinicia todos los contadores a cero. Solo disponible en entornos de desarrollo.",
    include_in_schema=False,   # ocultado en producción
)
async def reiniciar_metricas() -> dict:
    """Reinicia los acumuladores de métricas."""
    _acumuladores.reset()
    logger.info("Métricas reiniciadas")
    return {"mensaje": "Métricas reiniciadas correctamente"}
