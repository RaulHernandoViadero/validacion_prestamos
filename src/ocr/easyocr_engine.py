"""
easyocr_engine.py — Motor OCR basado en EasyOCR para extracción de texto.

Recibe los ROIs detectados por YOLO y extrae el texto de cada uno usando
EasyOCR. El motor se instancia una única vez (singleton) para evitar
recargar el modelo en cada request.

Idiomas configurados: español + inglés (necesario para MRZ).

Uso típico:
    from src.ocr.easyocr_engine import EasyOCREngine
    ocr = EasyOCREngine()
    resultados = ocr.extraer_texto_rois(resultado_deteccion)
    for clase, texto, conf in resultados:
        print(clase, "->" ,texto, f"({conf:.2%})")
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from src.config import OCR_LANGUAGES, OCR_GPU
from src.models.yolo_inference import ResultadoDeteccion, ROIDetectado

logger = logging.getLogger(__name__)


@dataclass
class TextoExtraido:
    """Texto extraído de un ROI con su confianza."""
    clase:     str
    texto_raw: str          # texto tal como lo devuelve EasyOCR
    confianza: float        # confianza media del OCR [0, 1]
    bbox_roi:  tuple        # bounding box del ROI en la imagen original

    def to_dict(self) -> dict:
        return {
            "clase":     self.clase,
            "texto_raw": self.texto_raw,
            "confianza": round(self.confianza, 4),
        }


class EasyOCREngine:
    """
    Motor OCR singleton basado en EasyOCR.

    Se inicializa una única vez al primer uso (lazy init) para evitar
    recargar el modelo en cada request, ya que la carga tarda ~3 segundos.

    Args:
        idiomas: Lista de códigos de idioma para EasyOCR.
        gpu: Si True, usa GPU para la inferencia OCR.
    """

    _instancia: Optional["EasyOCREngine"] = None

    def __init__(
        self,
        idiomas: list[str] = OCR_LANGUAGES,
        gpu: bool = OCR_GPU,
    ) -> None:
        self._idiomas  = idiomas
        self._gpu      = gpu
        self._reader   = None   # carga lazy

    @classmethod
    def instancia(cls) -> "EasyOCREngine":
        """Devuelve la instancia singleton del motor OCR."""
        if cls._instancia is None:
            cls._instancia = cls()
        return cls._instancia

    def _cargar_reader(self) -> None:
        """Carga el reader de EasyOCR (solo la primera vez)."""
        import easyocr
        logger.info("Cargando EasyOCR (idiomas=%s, gpu=%s)…", self._idiomas, self._gpu)
        self._reader = easyocr.Reader(self._idiomas, gpu=self._gpu, verbose=False)
        logger.info("EasyOCR listo")

    # ── API pública ───────────────────────────────────────────────────────────

    def leer_imagen(self, imagen: Image.Image) -> tuple[str, float]:
        """
        Extrae el texto completo de una imagen PIL.

        Args:
            imagen: Imagen PIL (modo RGB), idealmente el recorte de un ROI.

        Returns:
            Tupla (texto_concatenado, confianza_media).
        """
        if self._reader is None:
            self._cargar_reader()

        import numpy as np
        arr = np.array(imagen.convert("RGB"))

        try:
            resultados = self._reader.readtext(arr, detail=1, paragraph=False)
        except Exception as exc:
            logger.warning("Error OCR: %s", exc)
            return "", 0.0

        if not resultados:
            return "", 0.0

        textos     = [r[1] for r in resultados]
        confianzas = [r[2] for r in resultados]
        texto_final    = " ".join(textos).strip()
        confianza_media = sum(confianzas) / len(confianzas)

        logger.debug("OCR extraído: '%s' (conf=%.2f)", texto_final, confianza_media)
        return texto_final, confianza_media

    def extraer_texto_rois(
        self,
        resultado_deteccion: ResultadoDeteccion,
        clases_excluir: Optional[list[str]] = None,
    ) -> list[TextoExtraido]:
        """
        Extrae el texto de cada ROI detectado por YOLO.

        Args:
            resultado_deteccion: Resultado de YOLOInference.detectar().
            clases_excluir: Lista de clases a omitir (ej. ["foto", "firma"]).

        Returns:
            Lista de TextoExtraido ordenada por clase.
        """
        excluir = set(clases_excluir or ["foto", "firma"])
        textos: list[TextoExtraido] = []

        for roi in resultado_deteccion.rois:
            if roi.clase in excluir:
                continue

            if roi.recorte.width < 5 or roi.recorte.height < 5:
                logger.debug("ROI '%s' demasiado pequeño — omitido", roi.clase)
                continue

            texto, confianza = self.leer_imagen(roi.recorte)

            textos.append(TextoExtraido(
                clase=roi.clase,
                texto_raw=texto,
                confianza=confianza,
                bbox_roi=roi.bbox,
            ))

        logger.debug("Texto extraído de %d ROIs", len(textos))
        return textos

    def leer_zona_mrz(self, imagen_completa: Image.Image, bbox_mrz: tuple) -> tuple[str, str]:
        """
        Lee específicamente la zona MRZ y separa las dos líneas.

        Args:
            imagen_completa: Imagen completa del documento.
            bbox_mrz: Bounding box de la zona MRZ (x1, y1, x2, y2).

        Returns:
            Tupla (linea_1_mrz, linea_2_mrz).
        """
        x1, y1, x2, y2 = bbox_mrz
        recorte_mrz = imagen_completa.crop((x1, y1, x2, y2))
        texto, _ = self.leer_imagen(recorte_mrz)

        # Separar en dos líneas (la MRZ tiene exactamente 2 líneas de 30 chars)
        partes = texto.split()
        # Buscar patrones MRZ: cadenas de mayúsculas con '<'
        lineas_mrz = [p for p in partes if len(p) >= 15 and (p.isupper() or '<' in p)]

        if len(lineas_mrz) >= 2:
            return lineas_mrz[0][:30], lineas_mrz[1][:30]
        elif len(lineas_mrz) == 1:
            return lineas_mrz[0][:30], ""
        else:
            return texto[:30], ""
