"""
yolo_inference.py — Inferencia YOLOv8 y extracción de ROIs.

Carga el modelo entrenado y detecta los campos (ROIs) en una imagen
de documento. Devuelve:
  - Bounding boxes con clase, confianza y recorte de imagen.
  - Imagen anotada con los bounding boxes dibujados.

Uso típico:
    from src.models.yolo_inference import YOLOInference
    detector = YOLOInference(tipo="dni")
    resultado = detector.detectar(imagen_pil)
    for roi in resultado.rois:
        print(roi.clase, roi.confianza, roi.recorte.size)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from src.config import (
    WEIGHTS_DIR,
    YOLO_IMGSZ,
    DNI_CLASSES,
    LOAN_CLASSES,
)

logger = logging.getLogger(__name__)

# Colores por clase (para visualización)
_COLORES_DNI = [
    (220, 50,  50),   # nombre         — rojo
    (50,  120, 220),  # apellidos       — azul
    (50,  180, 50),   # numero_dni      — verde
    (200, 140, 30),   # fecha_nacimiento — naranja
    (140, 50,  200),  # fecha_caducidad — morado
    (30,  180, 180),  # nacionalidad    — cian
    (180, 90,  30),   # foto            — marrón
    (90,  90,  200),  # firma           — lila
    (50,  50,  50),   # mrz_line        — gris oscuro
]

_COLORES_LOAN = [
    (220, 50,  50),  (50, 120, 220),  (50,  180,  50),  (200, 140,  30),
    (140, 50, 200),  (30, 180, 180),  (180,  90,  30),  ( 90,  90, 200),
    (50,  50,  50),  (180, 30,  30),  (30,  30,  180),  (30,  130,  30),
    (130, 30, 130),  (30, 130, 130),
]


@dataclass
class ROIDetectado:
    """Resultado de detección de un único campo (ROI)."""
    clase:      str
    clase_id:   int
    confianza:  float
    bbox:       tuple[int, int, int, int]   # (x1, y1, x2, y2) en píxeles
    recorte:    Image.Image                  # imagen recortada del ROI

    def to_dict(self) -> dict:
        return {
            "clase":     self.clase,
            "clase_id":  self.clase_id,
            "confianza": round(self.confianza, 4),
            "bbox":      list(self.bbox),
        }


@dataclass
class ResultadoDeteccion:
    """Resultado completo de la detección sobre una imagen."""
    tipo:           str                      # "dni" o "prestamo"
    rois:           list[ROIDetectado] = field(default_factory=list)
    imagen_anotada: Optional[Image.Image] = None

    def to_dict(self) -> dict:
        return {
            "tipo": self.tipo,
            "n_rois": len(self.rois),
            "rois": [r.to_dict() for r in self.rois],
        }

    def roi_por_clase(self, clase: str) -> Optional[ROIDetectado]:
        """Devuelve el ROI de mayor confianza para la clase indicada."""
        candidatos = [r for r in self.rois if r.clase == clase]
        return max(candidatos, key=lambda r: r.confianza) if candidatos else None

    def clases_detectadas(self) -> list[str]:
        return [r.clase for r in self.rois]


class YOLOInference:
    """
    Wrapper de inferencia sobre un modelo YOLOv8 entrenado.

    Gestiona la carga lazy del modelo (solo se carga en el primer uso)
    y expone una API simple para detectar ROIs en imágenes PIL.

    Args:
        tipo: "dni" o "prestamo".
        weights_dir: Directorio de pesos entrenados.
        confianza_min: Umbral mínimo de confianza para aceptar una detección.
        device: Dispositivo de inferencia.
    """

    def __init__(
        self,
        tipo: str = "dni",
        weights_dir: Path = WEIGHTS_DIR,
        confianza_min: float = 0.25,
        device: str = "cpu",
    ) -> None:
        assert tipo in ("dni", "prestamo"), f"tipo debe ser 'dni' o 'prestamo', got: {tipo}"
        self._tipo          = tipo
        self._weights_dir   = Path(weights_dir)
        self._confianza_min = confianza_min
        self._device        = device
        self._clases        = DNI_CLASSES if tipo == "dni" else LOAN_CLASSES
        self._colores       = _COLORES_DNI if tipo == "dni" else _COLORES_LOAN
        self._modelo        = None   # carga lazy

        nombre_pesos = "yolo_dni.pt" if tipo == "dni" else "yolo_loan.pt"
        self._ruta_pesos = self._weights_dir / nombre_pesos

        logger.info("YOLOInference inicializado — tipo=%s, pesos=%s", tipo, self._ruta_pesos)

    def _cargar_modelo(self) -> None:
        """Carga el modelo YOLO desde disco (solo la primera vez)."""
        from ultralytics import YOLO
        if not self._ruta_pesos.exists():
            raise FileNotFoundError(
                f"Pesos YOLO no encontrados: {self._ruta_pesos}\n"
                "Entrena el modelo con: python -m src.models.yolo_trainer --tipo "
                + self._tipo
            )
        self._modelo = YOLO(str(self._ruta_pesos))
        logger.info("Modelo YOLO cargado: %s", self._ruta_pesos)

    def detectar(
        self,
        imagen: Image.Image,
        anotar: bool = True,
    ) -> ResultadoDeteccion:
        """
        Detecta todos los ROIs en la imagen de documento.

        Args:
            imagen: Imagen PIL de entrada (modo RGB).
            anotar: Si True, genera imagen_anotada con bounding boxes.

        Returns:
            ResultadoDeteccion con la lista de ROIs y la imagen anotada.
        """
        if self._modelo is None:
            self._cargar_modelo()

        img_rgb = imagen.convert("RGB")
        resultado = ResultadoDeteccion(tipo=self._tipo)

        # Inferencia YOLO
        predicciones = self._modelo.predict(
            source    = img_rgb,
            imgsz     = YOLO_IMGSZ,
            conf      = self._confianza_min,
            device    = self._device,
            verbose   = False,
            save      = False,
        )

        w, h = img_rgb.size

        for pred in predicciones:
            if pred.boxes is None:
                continue
            for box in pred.boxes:
                clase_id  = int(box.cls[0])
                confianza = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                # Clampear coordenadas
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                if clase_id >= len(self._clases):
                    continue

                clase  = self._clases[clase_id]
                recorte = img_rgb.crop((x1, y1, x2, y2))

                resultado.rois.append(ROIDetectado(
                    clase=clase,
                    clase_id=clase_id,
                    confianza=confianza,
                    bbox=(x1, y1, x2, y2),
                    recorte=recorte,
                ))

        # Ordenar por confianza descendente
        resultado.rois.sort(key=lambda r: r.confianza, reverse=True)

        logger.debug(
            "Detección %s: %d ROIs encontrados",
            self._tipo, len(resultado.rois),
        )

        if anotar:
            resultado.imagen_anotada = self._anotar_imagen(img_rgb, resultado.rois)

        return resultado

    def _anotar_imagen(
        self,
        imagen: Image.Image,
        rois: list[ROIDetectado],
    ) -> Image.Image:
        """
        Dibuja los bounding boxes y etiquetas sobre la imagen.

        Returns:
            Copia de la imagen con las anotaciones.
        """
        img_ann = imagen.copy()
        draw    = ImageDraw.Draw(img_ann)

        try:
            fuente = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 14)
        except Exception:
            fuente = ImageFont.load_default()

        for roi in rois:
            color = self._colores[roi.clase_id % len(self._colores)]
            x1, y1, x2, y2 = roi.bbox

            # Bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Etiqueta con fondo opaco
            etiqueta = f"{roi.clase} {roi.confianza:.2f}"
            try:
                bb = draw.textbbox((x1, y1 - 18), etiqueta, font=fuente)
                draw.rectangle(bb, fill=color)
                draw.text((x1, y1 - 18), etiqueta, fill=(255, 255, 255), font=fuente)
            except Exception:
                draw.text((x1, max(0, y1 - 14)), etiqueta, fill=color, font=fuente)

        return img_ann
