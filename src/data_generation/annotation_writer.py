"""
annotation_writer.py — Escritura de anotaciones en formato YOLO.

Convierte los diccionarios de ROIs (coordenadas absolutas en píxeles) al
formato de anotación de Ultralytics YOLO:

    <class_id> <x_center> <y_center> <width> <height>

Donde todas las coordenadas están normalizadas en [0, 1] respecto al
tamaño de la imagen.

También genera los ficheros YAML de configuración del dataset necesarios
para lanzar el entrenamiento con `ultralytics`.

Uso típico:
    from src.data_generation.annotation_writer import AnnotationWriter
    writer = AnnotationWriter()
    writer.guardar_anotacion(rois, img_w, img_h, ruta_txt, tipo="dni")
    writer.generar_yaml_dataset()
"""

import json
import logging
from pathlib import Path
from typing import Literal

from src.config import (
    DNI_CLASSES,
    LOAN_CLASSES,
    ANNOTATIONS_DNIS_DIR,
    ANNOTATIONS_LOANS_DIR,
    SPLITS_DIR,
    DATA_DIR,
)

logger = logging.getLogger(__name__)

TipoDocumento = Literal["dni", "prestamo"]


class AnnotationWriter:
    """
    Escribe anotaciones YOLO (.txt) y ficheros de configuración del dataset.

    Formato YOLO por línea:
        <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>

    Args:
        annotations_dnis_dir: Directorio donde se guardan las anotaciones de DNI.
        annotations_loans_dir: Directorio donde se guardan las anotaciones de formulario.
    """

    def __init__(
        self,
        annotations_dnis_dir: Path = ANNOTATIONS_DNIS_DIR,
        annotations_loans_dir: Path = ANNOTATIONS_LOANS_DIR,
    ) -> None:
        self._dirs = {
            "dni":     Path(annotations_dnis_dir),
            "prestamo": Path(annotations_loans_dir),
        }
        for d in self._dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    # ── API pública ───────────────────────────────────────────────────────────

    def guardar_anotacion(
        self,
        rois: dict[str, tuple[int, int, int, int]],
        img_w: int,
        img_h: int,
        ruta_txt: Path,
        tipo: TipoDocumento = "dni",
    ) -> None:
        """
        Convierte el diccionario de ROIs a formato YOLO y escribe el fichero .txt.

        Solo se escriben los ROIs cuyo nombre está en la lista de clases del
        tipo de documento correspondiente. Los ROIs no reconocidos se ignoran
        con un aviso de log.

        Args:
            rois: Diccionario {nombre_clase: (x1, y1, x2, y2)} en píxeles.
            img_w: Ancho de la imagen en píxeles.
            img_h: Alto de la imagen en píxeles.
            ruta_txt: Ruta de destino del fichero .txt.
            tipo: "dni" o "prestamo".
        """
        clases = DNI_CLASSES if tipo == "dni" else LOAN_CLASSES
        lineas: list[str] = []

        for nombre_roi, (x1, y1, x2, y2) in rois.items():
            if nombre_roi not in clases:
                logger.debug("ROI '%s' no está en las clases de '%s' — omitido", nombre_roi, tipo)
                continue

            class_id = clases.index(nombre_roi)

            # Validar coordenadas
            x1c = max(0, min(x1, img_w - 1))
            y1c = max(0, min(y1, img_h - 1))
            x2c = max(0, min(x2, img_w))
            y2c = max(0, min(y2, img_h))

            if x2c <= x1c or y2c <= y1c:
                logger.warning("ROI '%s' tiene dimensiones inválidas (%d,%d,%d,%d) — omitido",
                               nombre_roi, x1, y1, x2, y2)
                continue

            # Coordenadas normalizadas
            x_center = ((x1c + x2c) / 2) / img_w
            y_center = ((y1c + y2c) / 2) / img_h
            width    = (x2c - x1c) / img_w
            height   = (y2c - y1c) / img_h

            # Asegurar que están en [0, 1]
            x_center = min(1.0, max(0.0, x_center))
            y_center = min(1.0, max(0.0, y_center))
            width    = min(1.0, max(0.0, width))
            height   = min(1.0, max(0.0, height))

            lineas.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        ruta_txt = Path(ruta_txt)
        ruta_txt.parent.mkdir(parents=True, exist_ok=True)
        ruta_txt.write_text("\n".join(lineas), encoding="utf-8")

        logger.debug("Anotación guardada: %s (%d ROIs)", ruta_txt.name, len(lineas))

    def cargar_anotacion(
        self,
        ruta_txt: Path,
        img_w: int,
        img_h: int,
        tipo: TipoDocumento = "dni",
    ) -> dict[str, tuple[int, int, int, int]]:
        """
        Lee un fichero de anotación YOLO y devuelve el diccionario de ROIs
        en coordenadas absolutas de píxeles.

        Args:
            ruta_txt: Ruta del fichero .txt de anotación YOLO.
            img_w: Ancho de la imagen en píxeles.
            img_h: Alto de la imagen en píxeles.
            tipo: "dni" o "prestamo".

        Returns:
            Diccionario {nombre_clase: (x1, y1, x2, y2)}.
        """
        clases = DNI_CLASSES if tipo == "dni" else LOAN_CLASSES
        rois: dict[str, tuple[int, int, int, int]] = {}

        ruta_txt = Path(ruta_txt)
        if not ruta_txt.exists():
            logger.warning("Fichero de anotación no encontrado: %s", ruta_txt)
            return rois

        for linea in ruta_txt.read_text(encoding="utf-8").strip().splitlines():
            partes = linea.strip().split()
            if len(partes) != 5:
                continue
            class_id = int(partes[0])
            xc, yc, w, h = map(float, partes[1:])

            if class_id >= len(clases):
                continue

            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)

            rois[clases[class_id]] = (x1, y1, x2, y2)

        return rois

    def generar_yaml_dataset(
        self,
        splits_dir: Path = SPLITS_DIR,
        data_dir: Path = DATA_DIR,
    ) -> tuple[Path, Path]:
        """
        Genera los ficheros YAML de configuración del dataset para YOLOv8.

        Crea dos ficheros:
          - dataset_dni.yaml: configuración para el modelo de DNI.
          - dataset_prestamo.yaml: configuración para el modelo de formulario.

        Args:
            splits_dir: Directorio raíz de los splits (contiene train/val/test).
            data_dir: Directorio raíz de datos del proyecto.

        Returns:
            Tupla (ruta_yaml_dni, ruta_yaml_prestamo).
        """
        splits_dir = Path(splits_dir)

        yaml_dni = self._generar_yaml(
            nombre="dataset_dni",
            splits_dir=splits_dir,
            tipo="dni",
            clases=DNI_CLASSES,
            data_dir=data_dir,
        )
        yaml_prestamo = self._generar_yaml(
            nombre="dataset_prestamo",
            splits_dir=splits_dir,
            tipo="prestamo",
            clases=LOAN_CLASSES,
            data_dir=data_dir,
        )
        return yaml_dni, yaml_prestamo

    def _generar_yaml(
        self,
        nombre: str,
        splits_dir: Path,
        tipo: TipoDocumento,
        clases: list[str],
        data_dir: Path,
    ) -> Path:
        """Genera un único fichero YAML de dataset YOLO."""
        # Rutas relativas al directorio raíz del proyecto
        train_path = splits_dir / tipo / "train" / "images"
        val_path   = splits_dir / tipo / "val"   / "images"
        test_path  = splits_dir / tipo / "test"  / "images"

        # Construir contenido YAML manualmente para evitar dependencia de PyYAML
        nc = len(clases)
        nombres_yaml = "\n".join(f"  - {c}" for c in clases)

        contenido = f"""# Dataset YOLO — {tipo.upper()} — TFM Verificación Documental
# Generado automáticamente por annotation_writer.py

path: {data_dir.resolve()}   # directorio raíz del dataset

train: {train_path}
val:   {val_path}
test:  {test_path}

# Número de clases
nc: {nc}

# Nombres de las clases (en orden de class_id)
names:
{nombres_yaml}
"""
        ruta = splits_dir / f"{nombre}.yaml"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(contenido, encoding="utf-8")
        logger.info("YAML dataset generado: %s", ruta)
        return ruta

    @staticmethod
    def yolo_a_absoluto(
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        img_w: int,
        img_h: int,
    ) -> tuple[int, int, int, int]:
        """
        Convierte coordenadas YOLO normalizadas a coordenadas absolutas (x1,y1,x2,y2).

        Args:
            x_center, y_center, width, height: Coordenadas YOLO normalizadas.
            img_w, img_h: Dimensiones de la imagen.

        Returns:
            Tupla (x1, y1, x2, y2) en píxeles.
        """
        x1 = int((x_center - width  / 2) * img_w)
        y1 = int((y_center - height / 2) * img_h)
        x2 = int((x_center + width  / 2) * img_w)
        y2 = int((y_center + height / 2) * img_h)
        return x1, y1, x2, y2

    @staticmethod
    def absoluto_a_yolo(
        x1: int, y1: int, x2: int, y2: int,
        img_w: int, img_h: int,
    ) -> tuple[float, float, float, float]:
        """
        Convierte coordenadas absolutas (x1,y1,x2,y2) al formato YOLO normalizado.

        Returns:
            Tupla (x_center, y_center, width, height) normalizadas.
        """
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width    = (x2 - x1) / img_w
        height   = (y2 - y1) / img_h
        return x_center, y_center, width, height
