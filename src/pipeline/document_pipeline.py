"""
document_pipeline.py — Orquestador end-to-end del sistema de verificación.

Conecta todas las capas del sistema en un único punto de entrada:

  Imagen DNI + Imagen Formulario
        │
        ▼
  [YOLO] Detección de ROIs          (yolo_inference.py)
        │
        ▼
  [OCR]  Extracción de texto        (easyocr_engine.py)
        │
        ▼
  [POST] Normalización de campos    (text_postprocessor.py)
        │
        ▼
  [CLS]  Clasificación autenticidad (authenticity_classifier.py)
        │
        ▼
  [VAL]  Validación cruzada R01–R09 (cross_validator.py)
        │
        ▼
  [BIZ]  Reglas de negocio extras   (business_rules.py)
        │
        ▼
  [VERD] Veredicto final            (verdict_engine.py)

El resultado es un ExpedienteResult con todos los artefactos del proceso
(ROIs, textos, campos normalizados, reglas, veredicto) listo para ser
serializado y devuelto por la API o persistido en base de datos.

Uso típico:
    from src.pipeline.document_pipeline import DocumentPipeline, PipelineConfig
    pipeline = DocumentPipeline()
    resultado = pipeline.procesar(
        imagen_dni=dni_pil,
        imagen_form=form_pil,
        expediente_id="EXP-0042",
    )
    print(resultado.veredicto.resultado)
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image

from src.validation.cross_validator import CrossValidator, ResultadoValidacion
from src.validation.verdict_engine import VerdictEngine, Veredicto
from src.validation.business_rules import (
    verificar_mrz_checkdigit,
    verificar_cuota_francesa,
    evaluar_perfil_riesgo,
    verificar_coherencia_fechas,
    PerfilRiesgo,
)
from src.ocr.text_postprocessor import CampoNormalizado, TextPostprocessor
from src.ocr.easyocr_engine import EasyOCREngine, TextoExtraido

logger = logging.getLogger(__name__)


# ── Configuración del pipeline ────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    Configuración del pipeline de verificación.

    Permite activar o desactivar etapas según el entorno
    (ej. deshabilitar YOLO en tests unitarios).
    """
    usar_yolo:              bool  = True    # detección de ROIs con YOLOv8
    usar_clasificador:      bool  = True    # clasificación de autenticidad
    anotar_imagen:          bool  = True    # dibuja bboxes en la imagen
    tolerancia_nombres:     float = 0.85   # umbral Levenshtein para R01/R02
    umbral_confianza_ocr:   float = 0.50   # confianza OCR mínima para el veredicto
    guardar_recortes:       bool  = False   # guarda los ROIs como ficheros PNG
    directorio_recortes:    Optional[Path] = None


# ── Resultado del pipeline ────────────────────────────────────────────────────

@dataclass
class ResultadoOCR:
    """Textos extraídos y normalizados de un documento."""
    tipo:              str                           # "dni" | "formulario"
    textos_raw:        list[TextoExtraido]           = field(default_factory=list)
    campos:            dict[str, CampoNormalizado]   = field(default_factory=dict)
    confianza_media:   float                         = 0.0


@dataclass
class ExpedienteResult:
    """
    Resultado completo del procesamiento de un expediente.

    Contiene todos los artefactos intermedios para trazabilidad,
    debugging y generación del informe PDF.
    """
    # Identificación
    expediente_id:     str
    timestamp:         str = field(default_factory=lambda: datetime.now().isoformat())

    # Imágenes anotadas (con bboxes dibujadas por YOLO)
    imagen_dni_anotada:  Optional[Image.Image] = None
    imagen_form_anotada: Optional[Image.Image] = None

    # OCR y normalización
    ocr_dni:   Optional[ResultadoOCR] = None
    ocr_form:  Optional[ResultadoOCR] = None

    # Clasificador de autenticidad
    autenticidad_dni:   Optional[dict] = None
    autenticidad_form:  Optional[dict] = None

    # Validación cruzada
    resultado_validacion: Optional[ResultadoValidacion] = None

    # Reglas de negocio complementarias
    perfil_riesgo:        Optional[PerfilRiesgo]       = None
    mrz_checkdigits_ok:   bool                         = True
    cuota_coherente:      bool                         = True

    # Veredicto final
    veredicto:            Optional[Veredicto]          = None

    # Métricas de rendimiento
    tiempo_total_ms:      float = 0.0
    tiempos_etapas:       dict  = field(default_factory=dict)

    # Errores no fatales recogidos durante el procesamiento
    advertencias:         list[str] = field(default_factory=list)

    @property
    def es_apto(self) -> bool:
        return self.veredicto is not None and self.veredicto.es_apto

    def to_dict(self) -> dict:
        """Serializa el resultado a un diccionario JSON-serializable."""
        return {
            "expediente_id":   self.expediente_id,
            "timestamp":       self.timestamp,
            "es_apto":         self.es_apto,
            "veredicto":       self.veredicto.to_dict()        if self.veredicto        else None,
            "validacion":      self.resultado_validacion.to_dict() if self.resultado_validacion else None,
            "autenticidad_dni":  self.autenticidad_dni,
            "autenticidad_form": self.autenticidad_form,
            "campos_dni":      {k: v.to_dict() for k, v in self.ocr_dni.campos.items()}  if self.ocr_dni  else {},
            "campos_form":     {k: v.to_dict() for k, v in self.ocr_form.campos.items()} if self.ocr_form else {},
            "perfil_riesgo": {
                "nivel":       self.perfil_riesgo.nivel,
                "ratio_deuda": self.perfil_riesgo.ratio_deuda,
                "detalle":     self.perfil_riesgo.detalle,
            } if self.perfil_riesgo else None,
            "mrz_checkdigits_ok": self.mrz_checkdigits_ok,
            "cuota_coherente":    self.cuota_coherente,
            "tiempo_total_ms":    round(self.tiempo_total_ms, 1),
            "tiempos_etapas":     {k: round(v, 1) for k, v in self.tiempos_etapas.items()},
            "advertencias":       self.advertencias,
        }


# ── Pipeline ──────────────────────────────────────────────────────────────────

class DocumentPipeline:
    """
    Orquestador end-to-end del sistema de verificación documental.

    Encadena las etapas de detección (YOLO), extracción (OCR),
    normalización, clasificación de autenticidad, validación cruzada,
    reglas de negocio y veredicto.

    Todas las etapas son tolerantes a fallos: si una etapa falla, se
    registra una advertencia y el pipeline continúa con los datos
    disponibles para ofrecer el mejor veredicto posible.

    Args:
        config: Configuración del pipeline (ver PipelineConfig).
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self._cfg        = config or PipelineConfig()
        self._validator  = CrossValidator(tolerancia_nombres=self._cfg.tolerancia_nombres)
        self._verdict    = VerdictEngine(umbral_confianza_ocr=self._cfg.umbral_confianza_ocr)
        self._postproc   = TextPostprocessor()
        # Carga lazy — se instancian solo cuando son necesarios
        self._ocr_engine:      Optional[EasyOCREngine] = None
        self._yolo_dni:        Optional[object]        = None
        self._yolo_form:       Optional[object]        = None
        self._clasificador:    Optional[object]        = None

    # ── Punto de entrada principal ────────────────────────────────────────────

    def procesar(
        self,
        imagen_dni:    Image.Image,
        imagen_form:   Image.Image,
        expediente_id: Optional[str] = None,
    ) -> ExpedienteResult:
        """
        Procesa un expediente completo (DNI + formulario de préstamo).

        Args:
            imagen_dni:    Imagen PIL del DNI del solicitante.
            imagen_form:   Imagen PIL del formulario de préstamo cumplimentado.
            expediente_id: Identificador único del expediente. Si es None
                           se genera uno aleatorio con UUID4.

        Returns:
            ExpedienteResult con el veredicto y todos los artefactos.
        """
        exp_id = expediente_id or f"EXP-{uuid.uuid4().hex[:8].upper()}"
        t_ini  = time.perf_counter()
        result = ExpedienteResult(expediente_id=exp_id)

        logger.info("=== Iniciando pipeline para expediente %s ===", exp_id)

        try:
            # 1. Detección de ROIs con YOLO
            t0 = time.perf_counter()
            det_dni, det_form = self._etapa_deteccion(imagen_dni, imagen_form, result)
            result.tiempos_etapas["deteccion_ms"] = (time.perf_counter() - t0) * 1000

            # 2. Extracción y normalización OCR
            t0 = time.perf_counter()
            result.ocr_dni  = self._etapa_ocr(det_dni,  "dni",        result)
            result.ocr_form = self._etapa_ocr(det_form, "formulario", result)
            result.tiempos_etapas["ocr_ms"] = (time.perf_counter() - t0) * 1000

            # Imágenes anotadas
            result.imagen_dni_anotada  = getattr(det_dni,  "imagen_anotada", imagen_dni)
            result.imagen_form_anotada = getattr(det_form, "imagen_anotada", imagen_form)

            # 3. Clasificación de autenticidad
            t0 = time.perf_counter()
            self._etapa_clasificacion(imagen_dni, imagen_form, result)
            result.tiempos_etapas["clasificacion_ms"] = (time.perf_counter() - t0) * 1000

            # 4. Validación cruzada R01–R09
            t0 = time.perf_counter()
            result.resultado_validacion = self._etapa_validacion(result)
            result.tiempos_etapas["validacion_ms"] = (time.perf_counter() - t0) * 1000

            # 5. Reglas de negocio complementarias
            t0 = time.perf_counter()
            self._etapa_reglas_negocio(result)
            result.tiempos_etapas["reglas_negocio_ms"] = (time.perf_counter() - t0) * 1000

            # 6. Veredicto final
            t0 = time.perf_counter()
            result.veredicto = self._etapa_veredicto(result)
            result.tiempos_etapas["veredicto_ms"] = (time.perf_counter() - t0) * 1000

        except Exception as exc:
            logger.exception("Error no esperado en el pipeline: %s", exc)
            result.advertencias.append(f"Error en pipeline: {exc}")
            # Emitir veredicto de emergencia
            result.veredicto = self._veredicto_emergencia(exp_id, str(exc))

        result.tiempo_total_ms = (time.perf_counter() - t_ini) * 1000
        logger.info(
            "=== Pipeline finalizado: %s en %.0f ms ===",
            result.veredicto.resultado if result.veredicto else "SIN VEREDICTO",
            result.tiempo_total_ms,
        )
        return result

    # ── Etapas privadas ───────────────────────────────────────────────────────

    def _etapa_deteccion(
        self,
        imagen_dni:  Image.Image,
        imagen_form: Image.Image,
        result:      ExpedienteResult,
    ) -> tuple:
        """
        Etapa 1: Detección de ROIs con YOLOv8.

        Si YOLO no está habilitado o los pesos no existen, devuelve
        objetos vacíos que permiten continuar el pipeline sin ROIs.
        """
        if not self._cfg.usar_yolo:
            logger.info("YOLO deshabilitado — omitiendo detección de ROIs")
            return _ResultadoVacio(imagen_dni), _ResultadoVacio(imagen_form)

        from src.models.yolo_inference import YOLOInference

        if self._yolo_dni is None:
            try:
                self._yolo_dni  = YOLOInference(tipo="dni")
                self._yolo_form = YOLOInference(tipo="prestamo")
            except Exception as exc:
                logger.warning("No se pudo cargar YOLO: %s", exc)
                result.advertencias.append(f"YOLO no disponible: {exc}")
                return _ResultadoVacio(imagen_dni), _ResultadoVacio(imagen_form)

        try:
            det_dni  = self._yolo_dni.detectar(imagen_dni,   anotar=self._cfg.anotar_imagen)
            det_form = self._yolo_form.detectar(imagen_form, anotar=self._cfg.anotar_imagen)
            logger.info(
                "Detección: %d ROIs en DNI, %d ROIs en formulario",
                len(det_dni.rois), len(det_form.rois),
            )
            return det_dni, det_form
        except Exception as exc:
            logger.warning("Error en detección YOLO: %s", exc)
            result.advertencias.append(f"Error detección YOLO: {exc}")
            return _ResultadoVacio(imagen_dni), _ResultadoVacio(imagen_form)

    def _etapa_ocr(
        self,
        deteccion,
        tipo:   str,
        result: ExpedienteResult,
    ) -> ResultadoOCR:
        """
        Etapa 2: Extracción OCR de los ROIs detectados y normalización.

        Si YOLO no detectó ROIs, aplica OCR sobre la imagen completa
        como fallback de emergencia (útil en tests y demos).
        """
        if self._ocr_engine is None:
            try:
                self._ocr_engine = EasyOCREngine.instancia()
            except Exception as exc:
                logger.warning("No se pudo cargar EasyOCR: %s", exc)
                result.advertencias.append(f"OCR no disponible: {exc}")
                return ResultadoOCR(tipo=tipo)

        try:
            if hasattr(deteccion, "rois") and deteccion.rois:
                textos_raw = self._ocr_engine.extraer_texto_rois(deteccion)
            else:
                # Fallback: OCR sobre imagen completa (sin ROIs)
                texto, conf = self._ocr_engine.leer_imagen(deteccion.imagen_original)
                from src.ocr.easyocr_engine import TextoExtraido
                textos_raw = [TextoExtraido(
                    clase="texto_completo",
                    texto_raw=texto,
                    confianza=conf,
                    bbox_roi=(0, 0, 0, 0),
                )]
                result.advertencias.append(f"OCR fallback (sin ROIs) para {tipo}")

            campos = self._postproc.procesar_lote(textos_raw)

            # Confianza media sobre todos los campos
            confs = [c.confianza_ocr for c in campos.values() if c.confianza_ocr > 0]
            conf_media = sum(confs) / len(confs) if confs else 0.0

            logger.info(
                "OCR %s: %d campos extraídos (conf media=%.2f)",
                tipo, len(campos), conf_media,
            )
            return ResultadoOCR(
                tipo=tipo,
                textos_raw=textos_raw,
                campos=campos,
                confianza_media=conf_media,
            )

        except Exception as exc:
            logger.warning("Error en OCR (%s): %s", tipo, exc)
            result.advertencias.append(f"Error OCR {tipo}: {exc}")
            return ResultadoOCR(tipo=tipo)

    def _etapa_clasificacion(
        self,
        imagen_dni:  Image.Image,
        imagen_form: Image.Image,
        result:      ExpedienteResult,
    ) -> None:
        """
        Etapa 3: Clasificación de autenticidad con ResNet-18.

        Clasifica cada documento como LEGÍTIMO o MANIPULADO.
        Si el clasificador no está disponible, se omite (R09 se marca
        como WARNING en lugar de ERROR).
        """
        if not self._cfg.usar_clasificador:
            logger.info("Clasificador deshabilitado — omitiendo R09")
            return

        if self._clasificador is None:
            try:
                from src.models.authenticity_classifier import AuthenticityClassifier
                self._clasificador = AuthenticityClassifier()
            except Exception as exc:
                logger.warning("No se pudo cargar el clasificador: %s", exc)
                result.advertencias.append(f"Clasificador no disponible: {exc}")
                return

        try:
            result.autenticidad_dni  = self._clasificador.predecir(imagen_dni)
            result.autenticidad_form = self._clasificador.predecir(imagen_form)
            logger.info(
                "Autenticidad: DNI=%s(%.2f) Form=%s(%.2f)",
                result.autenticidad_dni.get("etiqueta",  "?"),
                result.autenticidad_dni.get("confianza",  0),
                result.autenticidad_form.get("etiqueta", "?"),
                result.autenticidad_form.get("confianza", 0),
            )
        except Exception as exc:
            logger.warning("Error en clasificación: %s", exc)
            result.advertencias.append(f"Error clasificador: {exc}")

    def _etapa_validacion(self, result: ExpedienteResult) -> ResultadoValidacion:
        """Etapa 4: Validación cruzada R01–R09."""
        campos_dni  = result.ocr_dni.campos  if result.ocr_dni  else {}
        campos_form = result.ocr_form.campos if result.ocr_form else {}

        return self._validator.validar(
            campos_dni=campos_dni,
            campos_form=campos_form,
            autenticidad_dni=result.autenticidad_dni,
            autenticidad_form=result.autenticidad_form,
        )

    def _etapa_reglas_negocio(self, result: ExpedienteResult) -> None:
        """
        Etapa 5: Reglas de negocio complementarias (no bloquean el veredicto).

        Calcula:
          - Perfil de riesgo crediticio (BAJO / MEDIO / ALTO)
          - Verificación de dígitos de control MRZ
          - Coherencia de la cuota con la fórmula francesa
          - Coherencia temporal de las fechas del DNI

        Los resultados se almacenan como metadatos del expediente
        y enriquecen el informe PDF pero no alteran el veredicto
        (eso corresponde a R01–R09).
        """
        campos_dni  = result.ocr_dni.campos  if result.ocr_dni  else {}
        campos_form = result.ocr_form.campos if result.ocr_form else {}

        def get(campos: dict, clave: str) -> str:
            c = campos.get(clave)
            return (c.texto_norm or c.texto_raw) if c else ""

        # ── Perfil de riesgo ──────────────────────────────────────────────────
        try:
            ingresos_str = get(campos_form, "sol_ingresos_netos")
            cuota_str    = get(campos_form, "prestamo_cuota")
            importe_str  = get(campos_form, "prestamo_importe")
            plazo_str    = get(campos_form, "prestamo_plazo")
            f_nac        = get(campos_dni,  "fecha_nacimiento")

            ingresos = float(ingresos_str) if ingresos_str else 0.0
            cuota    = float(cuota_str)    if cuota_str    else 0.0
            importe  = float(importe_str)  if importe_str  else 0.0
            plazo    = int(plazo_str)      if plazo_str    else 0

            if ingresos > 0 and cuota > 0:
                result.perfil_riesgo = evaluar_perfil_riesgo(
                    ingresos_netos=ingresos,
                    cuota_mensual=cuota,
                    importe=importe,
                    plazo_meses=plazo,
                    fecha_nacimiento=f_nac,
                )
                logger.info("Perfil riesgo: %s", result.perfil_riesgo.nivel)
        except Exception as exc:
            logger.debug("No se pudo calcular perfil riesgo: %s", exc)
            result.advertencias.append(f"Perfil riesgo no calculado: {exc}")

        # ── Verificación MRZ ──────────────────────────────────────────────────
        try:
            mrz1 = get(campos_dni, "mrz_line")   # línea 1 (o ambas concatenadas)
            if mrz1 and len(mrz1) >= 30:
                l1 = mrz1[:30]
                l2 = mrz1[30:60] if len(mrz1) >= 60 else ""
                if l2:
                    checks = verificar_mrz_checkdigit(l1, l2)
                    result.mrz_checkdigits_ok = all(c.valido for c in checks)
                    if not result.mrz_checkdigits_ok:
                        fallos = [c.detalle for c in checks if not c.valido]
                        result.advertencias.append(
                            f"MRZ dígitos de control incorrectos: {fallos}"
                        )
        except Exception as exc:
            logger.debug("Error verificando MRZ: %s", exc)

        # ── Coherencia cuota francesa ─────────────────────────────────────────
        try:
            from src.config import LOAN_TAE_DEFAULT
            cuota_str   = get(campos_form, "prestamo_cuota")
            importe_str = get(campos_form, "prestamo_importe")
            plazo_str   = get(campos_form, "prestamo_plazo")

            if cuota_str and importe_str and plazo_str:
                r_cuota = verificar_cuota_francesa(
                    cuota_declarada=float(cuota_str),
                    importe=float(importe_str),
                    tae=LOAN_TAE_DEFAULT,
                    plazo_meses=int(plazo_str),
                )
                result.cuota_coherente = r_cuota.valido
                if not r_cuota.valido:
                    result.advertencias.append(
                        f"Cuota no coincide con fórmula francesa: {r_cuota.detalle}"
                    )
        except (ImportError, AttributeError):
            # LOAN_TAE_DEFAULT puede no estar en config si no se ha definido
            pass
        except Exception as exc:
            logger.debug("Error verificando cuota francesa: %s", exc)

    def _etapa_veredicto(self, result: ExpedienteResult) -> Veredicto:
        """Etapa 6: Emisión del veredicto final."""
        campos_dni  = result.ocr_dni.campos  if result.ocr_dni  else {}
        campos_form = result.ocr_form.campos if result.ocr_form else {}

        return self._verdict.emitir(
            expediente_id=result.expediente_id,
            resultado_val=result.resultado_validacion,
            campos_dni=campos_dni,
            campos_form=campos_form,
        )

    @staticmethod
    def _veredicto_emergencia(expediente_id: str, motivo: str) -> Veredicto:
        """Genera un veredicto de emergencia cuando el pipeline falla."""
        from src.validation.verdict_engine import VEREDICTO_INCONSISTENTE
        return Veredicto(
            expediente_id=expediente_id,
            resultado=VEREDICTO_INCONSISTENTE,
            confianza=0.0,
            reglas_fallidas=["PIPELINE_ERROR"],
            resumen_fallos=[f"Error interno del pipeline: {motivo}"],
        )


# ── Objeto vacío para cuando YOLO no está disponible ─────────────────────────

class _ResultadoVacio:
    """
    Sustituye a ResultadoDeteccion cuando YOLO no está disponible.
    Permite que la etapa OCR use la imagen completa como fallback.
    """
    def __init__(self, imagen: Image.Image) -> None:
        self.rois           = []
        self.imagen_anotada = imagen
        self.imagen_original = imagen
        self.tipo           = "desconocido"
