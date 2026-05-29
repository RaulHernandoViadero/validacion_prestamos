"""
verdict_engine.py — Motor de decisión final del sistema de verificación.

Toma el ResultadoValidacion producido por CrossValidator y emite el
veredicto final del expediente:

  ✅  APTO PARA TRÁMITE       → todas las reglas pasadas
  ❌  EXPEDIENTE INCONSISTENTE → al menos una regla ha fallado

El veredicto incluye un nivel de confianza calculado a partir del
número de reglas pasadas y las confianzas OCR de los campos clave.

Uso típico:
    from src.validation.verdict_engine import VerdictEngine
    engine = VerdictEngine()
    veredicto = engine.emitir(resultado_validacion, campos_dni, campos_form)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.validation.cross_validator import ResultadoValidacion
from src.ocr.text_postprocessor import CampoNormalizado

logger = logging.getLogger(__name__)

VEREDICTO_APTO         = "APTO PARA TRÁMITE"
VEREDICTO_INCONSISTENTE = "EXPEDIENTE INCONSISTENTE"


@dataclass
class Veredicto:
    """Veredicto final emitido por el motor de decisión."""
    expediente_id:    str
    resultado:        str             # APTO PARA TRÁMITE | EXPEDIENTE INCONSISTENTE
    confianza:        float           # [0, 1] — confianza global del sistema
    reglas_fallidas:  list[str]       = field(default_factory=list)
    resumen_fallos:   list[str]       = field(default_factory=list)
    confianza_ocr_media: float        = 0.0
    timestamp:        str             = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def es_apto(self) -> bool:
        return self.resultado == VEREDICTO_APTO

    def to_dict(self) -> dict:
        return {
            "expediente_id":      self.expediente_id,
            "resultado":          self.resultado,
            "es_apto":            self.es_apto,
            "confianza":          round(self.confianza, 4),
            "reglas_fallidas":    self.reglas_fallidas,
            "resumen_fallos":     self.resumen_fallos,
            "confianza_ocr_media": round(self.confianza_ocr_media, 4),
            "timestamp":          self.timestamp,
        }


class VerdictEngine:
    """
    Motor de decisión final del sistema de verificación documental.

    Combina los resultados de la validación cruzada con las confianzas
    del OCR para producir un veredicto final con nivel de confianza global.

    Args:
        umbral_confianza_ocr: Confianza OCR mínima en campos clave para
                              considerar el veredicto fiable.
    """

    # Campos clave cuya confianza OCR pondera más en el veredicto
    _CAMPOS_CLAVE = [
        "numero_dni", "sol_nif",
        "nombre", "sol_nombre",
        "apellidos", "sol_apellidos",
        "fecha_nacimiento", "sol_fecha_nacimiento",
    ]

    def __init__(self, umbral_confianza_ocr: float = 0.5) -> None:
        self._umbral_ocr = umbral_confianza_ocr

    def emitir(
        self,
        expediente_id: str,
        resultado_val: ResultadoValidacion,
        campos_dni:  dict[str, CampoNormalizado],
        campos_form: dict[str, CampoNormalizado],
    ) -> Veredicto:
        """
        Emite el veredicto final para un expediente.

        Args:
            expediente_id: Identificador del expediente.
            resultado_val: Resultado de CrossValidator.validar().
            campos_dni:    Campos normalizados del DNI.
            campos_form:   Campos normalizados del formulario.

        Returns:
            Veredicto con resultado, confianza y detalle de fallos.
        """
        # Resultado de la validación
        es_apto = resultado_val.es_consistente

        # Confianza OCR media sobre los campos clave
        todos_campos = {**campos_dni, **campos_form}
        confs_ocr = [
            todos_campos[k].confianza_ocr
            for k in self._CAMPOS_CLAVE
            if k in todos_campos and todos_campos[k].confianza_ocr > 0
        ]
        conf_ocr_media = sum(confs_ocr) / len(confs_ocr) if confs_ocr else 0.5

        # Confianza global: combinación de reglas OK + confianza OCR
        frac_reglas_ok = resultado_val.n_reglas_ok / max(len(resultado_val.reglas), 1)
        # Peso: 70% reglas, 30% OCR
        confianza_global = 0.70 * frac_reglas_ok + 0.30 * conf_ocr_media

        # Si el OCR tiene confianza muy baja → bajar confianza del veredicto
        if conf_ocr_media < self._umbral_ocr:
            confianza_global *= 0.7
            logger.warning(
                "Confianza OCR baja (%.2f) — veredicto menos fiable", conf_ocr_media
            )

        # Resumen de fallos legible
        resumen_fallos = [
            f"{r.codigo}: {r.detalle}"
            for r in resultado_val.reglas
            if not r.pasada
        ]

        veredicto = Veredicto(
            expediente_id=expediente_id,
            resultado=VEREDICTO_APTO if es_apto else VEREDICTO_INCONSISTENTE,
            confianza=round(confianza_global, 4),
            reglas_fallidas=resultado_val.reglas_fallidas,
            resumen_fallos=resumen_fallos,
            confianza_ocr_media=round(conf_ocr_media, 4),
        )

        logger.info(
            "Veredicto %s — %s (confianza=%.2f)",
            expediente_id, veredicto.resultado, veredicto.confianza,
        )
        if resumen_fallos:
            for fallo in resumen_fallos:
                logger.info("  ✗ %s", fallo)

        return veredicto
