"""
generate_dataset.py — Script maestro de generación del dataset sintético.

Orquesta todas las fases de la generación:
  1. Genera N expedientes (DNI + formulario) con FakeDataFactory.
  2. Renderiza las imágenes PNG con DNIGenerator y LoanFormGenerator.
  3. Aplica data augmentation con DocumentAugmentor.
  4. Escribe las anotaciones YOLO con AnnotationWriter.
  5. Divide en splits train/val/test.
  6. Guarda el ground truth en synthetic_records.json.
  7. Genera los ficheros YAML del dataset.

Uso:
    python generate_dataset.py [--size 400] [--seed 42] [--no-aug]

Ejemplo:
    python generate_dataset.py --size 100 --seed 0
"""

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Asegurar que el directorio raíz está en el path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DATASET_SIZE,
    RANDOM_SEED,
    SPLIT_RATIOS,
    INCONSISTENCY_RATE,
    DNI_WIDTH_PX,
    DNI_HEIGHT_PX,
    LOAN_WIDTH_PX,
    LOAN_HEIGHT_PX,
    DATA_DIR,
    RAW_DNIS_DIR,
    RAW_LOANS_DIR,
    ANNOTATIONS_DNIS_DIR,
    ANNOTATIONS_LOANS_DIR,
    SPLITS_DIR,
    SYNTHETIC_RECORDS_PATH,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
)
from src.data_generation.fake_data_factory import FakeDataFactory
from src.data_generation.dni_generator import DNIGenerator
from src.data_generation.loan_form_generator import LoanFormGenerator
from src.data_generation.augmentation import DocumentAugmentor, Intensidad
from src.data_generation.annotation_writer import AnnotationWriter

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "generation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("generate_dataset")


# ─── Utilidades ──────────────────────────────────────────────────────────────

def _crear_estructura_splits() -> None:
    """Crea los subdirectorios de train/val/test para imágenes y etiquetas."""
    for tipo in ["dni", "prestamo"]:
        for split in ["train", "val", "test"]:
            (SPLITS_DIR / tipo / split / "images").mkdir(parents=True, exist_ok=True)
            (SPLITS_DIR / tipo / split / "labels").mkdir(parents=True, exist_ok=True)
    logger.info("Estructura de splits creada en %s", SPLITS_DIR)


def _asignar_split(idx: int, n_train: int, n_val: int) -> str:
    """Asigna el split a un expediente según su índice."""
    if idx < n_train:
        return "train"
    elif idx < n_train + n_val:
        return "val"
    return "test"


def _copiar_a_split(
    ruta_img: Path,
    ruta_ann: Path,
    split: str,
    tipo: str,
) -> None:
    """Copia imagen y anotación al directorio del split correspondiente."""
    dest_img = SPLITS_DIR / tipo / split / "images" / ruta_img.name
    dest_lbl = SPLITS_DIR / tipo / split / "labels" / ruta_ann.name
    shutil.copy2(ruta_img, dest_img)
    shutil.copy2(ruta_ann, dest_lbl)


# ─── Generación principal ─────────────────────────────────────────────────────

def generar_dataset(
    size: int = DATASET_SIZE,
    seed: int = RANDOM_SEED,
    aplicar_aug: bool = True,
    n_aug_por_imagen: int = 3,
    intensidad_aug: str = "media",
) -> None:
    """
    Genera el dataset sintético completo.

    Args:
        size: Número total de expedientes a generar.
        seed: Semilla aleatoria para reproducibilidad.
        aplicar_aug: Si True, aplica data augmentation.
        n_aug_por_imagen: Número de variantes aumentadas por imagen base.
        intensidad_aug: Intensidad del augmentation ("suave", "media", "fuerte").
    """
    inicio = datetime.now()
    logger.info("=" * 60)
    logger.info("Inicio de generación del dataset")
    logger.info("  Tamaño: %d expedientes", size)
    logger.info("  Semilla: %d", seed)
    logger.info("  Augmentation: %s (x%d)", intensidad_aug if aplicar_aug else "NO", n_aug_por_imagen)
    logger.info("=" * 60)

    # ── 1. Inicializar componentes ──────────────────────────────────────────
    factory = FakeDataFactory(seed=seed, inconsistency_rate=INCONSISTENCY_RATE)
    dni_gen = DNIGenerator(output_dir=RAW_DNIS_DIR, seed=seed)
    loan_gen = LoanFormGenerator(output_dir=RAW_LOANS_DIR, seed=seed)
    augmentor = DocumentAugmentor(seed=seed) if aplicar_aug else None
    writer = AnnotationWriter()

    # ── 2. Crear estructura de directorios ──────────────────────────────────
    _crear_estructura_splits()

    # ── 3. Calcular tamaños de splits ───────────────────────────────────────
    n_train = round(size * SPLIT_RATIOS["train"])
    n_val   = round(size * SPLIT_RATIOS["val"])
    n_test  = size - n_train - n_val

    logger.info("Splits: train=%d, val=%d, test=%d", n_train, n_val, n_test)

    # ── 4. Generar expedientes ──────────────────────────────────────────────
    expedientes = factory.generar_lote(size)

    # Crear índice aleatorio para asignar splits de forma mezclada
    rng_split = random.Random(seed + 1)
    indices = list(range(size))
    rng_split.shuffle(indices)
    split_asignado = {}
    for i, exp_idx in enumerate(indices):
        split_asignado[exp_idx] = _asignar_split(i, n_train, n_val)

    # ── 5. Bucle principal de generación ────────────────────────────────────
    records = []
    n_ok_dni = n_ok_loan = n_err = 0

    for i, exp in enumerate(expedientes):
        split = split_asignado[i]
        exp_id = exp.expediente_id
        logger.info("[%d/%d] %s — split=%s — consistente=%s",
                    i + 1, size, exp_id, split, exp.es_consistente)

        try:
            # ── 5a. Generar imagen DNI ──────────────────────────────────────
            ruta_dni, rois_dni = dni_gen.generar(exp.dni, exp_id)
            ruta_ann_dni = ANNOTATIONS_DNIS_DIR / f"{exp_id}_dni.txt"
            writer.guardar_anotacion(rois_dni, DNI_WIDTH_PX, DNI_HEIGHT_PX,
                                     ruta_ann_dni, tipo="dni")
            _copiar_a_split(ruta_dni, ruta_ann_dni, split, "dni")
            n_ok_dni += 1

            # ── 5b. Augmentation DNI ────────────────────────────────────────
            if aplicar_aug and augmentor:
                from PIL import Image
                img_dni = Image.open(ruta_dni)
                variantes = augmentor.aplicar_lote(
                    img_dni, rois_dni,
                    n_variantes=n_aug_por_imagen,
                    intensidad=intensidad_aug,
                )
                for j, (img_aug, rois_aug) in enumerate(variantes):
                    aug_id = f"{exp_id}_aug{j:02d}"
                    ruta_aug = RAW_DNIS_DIR / f"{aug_id}_dni.png"
                    img_aug.save(str(ruta_aug), "PNG")
                    ruta_ann_aug = ANNOTATIONS_DNIS_DIR / f"{aug_id}_dni.txt"
                    writer.guardar_anotacion(rois_aug, DNI_WIDTH_PX, DNI_HEIGHT_PX,
                                             ruta_ann_aug, tipo="dni")
                    # Las variantes aumentadas van solo a train
                    _copiar_a_split(ruta_aug, ruta_ann_aug, "train", "dni")

            # ── 5c. Generar imagen formulario ───────────────────────────────
            ruta_loan, rois_loan = loan_gen.generar(exp.formulario, exp_id)
            ruta_ann_loan = ANNOTATIONS_LOANS_DIR / f"{exp_id}_prestamo.txt"
            writer.guardar_anotacion(rois_loan, LOAN_WIDTH_PX, LOAN_HEIGHT_PX,
                                     ruta_ann_loan, tipo="prestamo")
            _copiar_a_split(ruta_loan, ruta_ann_loan, split, "prestamo")
            n_ok_loan += 1

            # ── 5d. Augmentation formulario ─────────────────────────────────
            if aplicar_aug and augmentor:
                from PIL import Image
                img_loan = Image.open(ruta_loan)
                variantes_loan = augmentor.aplicar_lote(
                    img_loan, rois_loan,
                    n_variantes=n_aug_por_imagen,
                    intensidad=intensidad_aug,
                )
                for j, (img_aug, rois_aug) in enumerate(variantes_loan):
                    aug_id = f"{exp_id}_aug{j:02d}"
                    ruta_aug = RAW_LOANS_DIR / f"{aug_id}_prestamo.png"
                    img_aug.save(str(ruta_aug), "PNG")
                    ruta_ann_aug = ANNOTATIONS_LOANS_DIR / f"{aug_id}_prestamo.txt"
                    writer.guardar_anotacion(rois_aug, LOAN_WIDTH_PX, LOAN_HEIGHT_PX,
                                             ruta_ann_aug, tipo="prestamo")
                    _copiar_a_split(ruta_aug, ruta_ann_aug, "train", "prestamo")

            # ── 5e. Registrar en ground truth ───────────────────────────────
            records.append({
                **exp.to_dict(),
                "split": split,
                "ruta_dni": str(ruta_dni),
                "ruta_prestamo": str(ruta_loan),
            })

        except Exception as exc:
            logger.error("Error procesando %s: %s", exp_id, exc, exc_info=True)
            n_err += 1
            continue

    # ── 6. Guardar ground truth ─────────────────────────────────────────────
    SYNTHETIC_RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SYNTHETIC_RECORDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "generado": inicio.isoformat(),
                    "seed": seed,
                    "total_expedientes": size,
                    "n_consistentes": sum(1 for r in records if r["es_consistente"]),
                    "n_inconsistentes": sum(1 for r in records if not r["es_consistente"]),
                    "splits": {"train": n_train, "val": n_val, "test": n_test},
                    "augmentation": {
                        "activa": aplicar_aug,
                        "variantes": n_aug_por_imagen,
                        "intensidad": intensidad_aug,
                    },
                },
                "expedientes": records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Ground truth guardado: %s", SYNTHETIC_RECORDS_PATH)

    # ── 7. Generar YAMLs del dataset ────────────────────────────────────────
    yaml_dni, yaml_loan = writer.generar_yaml_dataset()
    logger.info("YAML DNI: %s", yaml_dni)
    logger.info("YAML Formulario: %s", yaml_loan)

    # ── 8. Resumen final ────────────────────────────────────────────────────
    duracion = (datetime.now() - inicio).total_seconds()
    logger.info("=" * 60)
    logger.info("GENERACIÓN COMPLETADA en %.1f s", duracion)
    logger.info("  DNIs generados:       %d", n_ok_dni)
    logger.info("  Formularios generados:%d", n_ok_loan)
    logger.info("  Errores:              %d", n_err)
    if aplicar_aug:
        logger.info("  Variantes aumentadas: %d por imagen (%d total por tipo)",
                    n_aug_por_imagen, n_ok_dni * n_aug_por_imagen)
    logger.info("=" * 60)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera el dataset sintético completo para el TFM."
    )
    parser.add_argument(
        "--size", type=int, default=DATASET_SIZE,
        help=f"Número de expedientes a generar (default: {DATASET_SIZE})",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Semilla aleatoria (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--no-aug", action="store_true",
        help="Desactiva el data augmentation",
    )
    parser.add_argument(
        "--aug-variants", type=int, default=3,
        help="Número de variantes aumentadas por imagen (default: 3)",
    )
    parser.add_argument(
        "--aug-intensity", choices=["suave", "media", "fuerte"], default="media",
        help="Intensidad del augmentation (default: media)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    generar_dataset(
        size=args.size,
        seed=args.seed,
        aplicar_aug=not args.no_aug,
        n_aug_por_imagen=args.aug_variants,
        intensidad_aug=args.aug_intensity,
    )
