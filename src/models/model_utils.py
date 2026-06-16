"""
model_utils.py — Utilidades compartidas para los modelos de IA del TFM.

Funciones de apoyo para:
  - Preparar el dataset del clasificador de autenticidad a partir de
    los expedientes generados (synthetic_records.json).
  - Calcular y mostrar métricas de forma consistente.
  - Verificar que los pesos existen antes de lanzar inferencia.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Literal

from src.config import (
    SYNTHETIC_RECORDS_PATH,
    SPLITS_DIR,
    WEIGHTS_DIR,
)

logger = logging.getLogger(__name__)

SplitName = Literal["train", "val", "test"]


def preparar_dataset_clasificador(
    records_path: Path = SYNTHETIC_RECORDS_PATH,
    splits_dir: Path = SPLITS_DIR,
    tipo: Literal["dni", "prestamo"] = "dni",
) -> dict[SplitName, Path]:
    """
    Organiza las imágenes del dataset en la estructura requerida por el
    clasificador de autenticidad:

        splits/{tipo}_clf/{split}/legitimo/    ← expedientes consistentes
        splits/{tipo}_clf/{split}/manipulado/  ← expedientes inconsistentes

    Lee el ground truth de synthetic_records.json para saber qué imagen
    es legítima y cuál es manipulada.

    Args:
        records_path: Ruta al fichero de ground truth JSON.
        splits_dir: Directorio raíz de los splits.
        tipo: "dni" o "prestamo".

    Returns:
        Diccionario {split: ruta_directorio} con las rutas de cada split.
    """
    if not records_path.exists():
        raise FileNotFoundError(
            f"Ground truth no encontrado: {records_path}\n"
            "Ejecuta primero: python generate_dataset.py"
        )

    with records_path.open(encoding="utf-8") as f:
        datos = json.load(f)

    expedientes = datos["expedientes"]

    # Crear estructura de directorios
    rutas: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        for clase in ("legitimo", "manipulado"):
            d = splits_dir / f"{tipo}_clf" / split / clase
            d.mkdir(parents=True, exist_ok=True)
        rutas[split] = splits_dir / f"{tipo}_clf" / split

    copiados = {"train": 0, "val": 0, "test": 0}

    for exp in expedientes:
        split   = exp.get("split", "train")
        consiste = exp["es_consistente"]
        clase   = "legitimo" if consiste else "manipulado"

        # Obtener ruta de la imagen según el tipo
        campo_ruta = "ruta_dni" if tipo == "dni" else "ruta_prestamo"
        ruta_src = Path(exp.get(campo_ruta, ""))

        if not ruta_src.exists():
            logger.debug("Imagen no encontrada: %s", ruta_src)
            continue

        dest = rutas[split] / clase / ruta_src.name
        if not dest.exists():
            shutil.copy2(ruta_src, dest)
            copiados[split] += 1

    logger.info(
        "Dataset clasificador preparado (%s): train=%d, val=%d, test=%d",
        tipo, copiados["train"], copiados["val"], copiados["test"],
    )
    return rutas


def verificar_pesos(
    nombres: list[str],
    weights_dir: Path = WEIGHTS_DIR,
) -> dict[str, bool]:
    """
    Comprueba qué ficheros de pesos existen en el directorio de pesos.

    Args:
        nombres: Lista de nombres de fichero a verificar.
        weights_dir: Directorio de pesos.

    Returns:
        Diccionario {nombre: existe}.
    """
    resultado = {}
    for nombre in nombres:
        ruta = Path(weights_dir) / nombre
        existe = ruta.exists()
        resultado[nombre] = existe
        if not existe:
            logger.warning("Pesos no encontrados: %s", ruta)
    return resultado


def resumen_metricas(metricas: dict, titulo: str = "Métricas") -> str:
    """
    Formatea un diccionario de métricas en una cadena legible para logs.

    Args:
        metricas: Diccionario con claves de métricas y valores float.
        titulo: Título del bloque.

    Returns:
        Cadena formateada lista para imprimir.
    """
    lineas = [f"{'─'*40}", f"  {titulo}", f"{'─'*40}"]
    for k, v in metricas.items():
        if isinstance(v, float):
            lineas.append(f"  {k:<20s}: {v:.4f}")
        else:
            lineas.append(f"  {k:<20s}: {v}")
    lineas.append(f"{'─'*40}")
    return "\n".join(lineas)
