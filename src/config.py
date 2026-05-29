"""
config.py — Configuración centralizada del proyecto TFM.

Todos los hiperparámetros, rutas y constantes del sistema se definen aquí.
Los módulos importan desde este fichero en lugar de usar valores hardcoded.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Cargar variables de entorno ──────────────────────────────────────────────
load_dotenv()

# ─── Rutas base ───────────────────────────────────────────────────────────────

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
"""Directorio raíz del proyecto (validacion_prestamos/)."""

DATA_DIR: Path = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
"""Directorio raíz de datos."""

WEIGHTS_DIR: Path = Path(os.getenv("WEIGHTS_DIR", ROOT_DIR / "weights"))
"""Directorio donde se almacenan los pesos entrenados."""

# ─── Subdirectorios de datos ──────────────────────────────────────────────────

RAW_DNIS_DIR: Path = DATA_DIR / "raw" / "dnis"
RAW_LOANS_DIR: Path = DATA_DIR / "raw" / "prestamos"
ANNOTATIONS_DNIS_DIR: Path = DATA_DIR / "annotations" / "dnis"
ANNOTATIONS_LOANS_DIR: Path = DATA_DIR / "annotations" / "prestamos"
AUGMENTED_DIR: Path = DATA_DIR / "augmented"
SPLITS_DIR: Path = DATA_DIR / "splits"
SYNTHETIC_RECORDS_PATH: Path = DATA_DIR / "synthetic_records.json"

# ─── Dataset ──────────────────────────────────────────────────────────────────

DATASET_SIZE: int = int(os.getenv("DATASET_SIZE", 400))
"""Número total de expedientes (DNI + formulario) a generar."""

INCONSISTENCY_RATE: float = float(os.getenv("INCONSISTENCY_RATE", 0.15))
"""Fracción de expedientes con inconsistencias deliberadas (ground truth negativo)."""

RANDOM_SEED: int = 42
"""Semilla para reproducibilidad en todos los módulos."""

# ─── Splits del dataset ───────────────────────────────────────────────────────

SPLIT_RATIOS: dict[str, float] = {
    "train": 0.70,
    "val":   0.175,
    "test":  0.125,
}
"""Proporciones de train / val / test (deben sumar 1.0)."""

# ─── Dimensiones de imagen ───────────────────────────────────────────────────

# DNI español — formato ID-1 (85.6 mm × 53.98 mm) a 450 DPI (3× base de 150)
# Resolución alta para legibilidad humana y precisión OCR
DNI_WIDTH_PX: int = 1518
DNI_HEIGHT_PX: int = 957

# Formulario A4 a 192 DPI (2× base de 96)
LOAN_WIDTH_PX: int = 1588
LOAN_HEIGHT_PX: int = 2246

# Tamaño de entrada para YOLOv8
YOLO_IMGSZ: int = 640

# Tamaño de entrada para el clasificador de autenticidad (ResNet-18)
CLASSIFIER_IMGSZ: int = 224

# ─── Clases YOLO — DNI (9 clases) ────────────────────────────────────────────

DNI_CLASSES: list[str] = [
    "nombre",           # 0
    "apellidos",        # 1
    "numero_dni",       # 2
    "fecha_nacimiento", # 3
    "fecha_caducidad",  # 4
    "nacionalidad",     # 5
    "foto",             # 6
    "firma",            # 7
    "mrz_line",         # 8
]

# ─── Clases YOLO — Formulario de préstamo (14 clases) ────────────────────────

LOAN_CLASSES: list[str] = [
    "sol_nombre",            # 0
    "sol_apellidos",         # 1
    "sol_nif",               # 2
    "sol_fecha_nacimiento",  # 3
    "sol_domicilio",         # 4
    "sol_telefono",          # 5
    "sol_email",             # 6
    "sol_situacion_laboral", # 7
    "sol_empresa",           # 8
    "sol_ingresos_netos",    # 9
    "prestamo_importe",      # 10
    "prestamo_plazo",        # 11
    "prestamo_finalidad",    # 12
    "prestamo_cuota",        # 13
]

# ─── Entrenamiento YOLO ───────────────────────────────────────────────────────

YOLO_BASE_MODEL: str = "yolov8n.pt"
"""Modelo base YOLOv8 nano (equilibrio velocidad/precisión para TFM)."""

YOLO_EPOCHS: int = 100
YOLO_BATCH: int = 16
YOLO_LR0: float = 0.01
YOLO_PATIENCE: int = 20
"""Early stopping: detiene el entrenamiento si no mejora en N epochs."""

# ─── Entrenamiento clasificador de autenticidad ───────────────────────────────

CLASSIFIER_EPOCHS: int = 30
CLASSIFIER_BATCH: int = 32
CLASSIFIER_LR: float = 1e-4
CLASSIFIER_DROPOUT: float = 0.3

# ─── Data Augmentation ───────────────────────────────────────────────────────

AUG_ROTATION_RANGE: tuple[float, float] = (-15.0, 15.0)
"""Rango de rotación en grados."""

AUG_BRIGHTNESS_RANGE: tuple[float, float] = (0.7, 1.3)
"""Factor de brillo (1.0 = sin cambio)."""

AUG_CONTRAST_RANGE: tuple[float, float] = (0.7, 1.3)
"""Factor de contraste."""

AUG_OCCLUSION_PROB: float = 0.10
"""Probabilidad de aplicar oclusión parcial."""

AUG_BLUR_KERNEL_RANGE: tuple[int, int] = (1, 3)
"""Rango del kernel de desenfoque (valores impares)."""

# ─── OCR ─────────────────────────────────────────────────────────────────────

OCR_LANGUAGES: list[str] = ["es", "en"]
"""Idiomas para EasyOCR."""

OCR_GPU: bool = False
"""Usar GPU para EasyOCR. Poner True si hay CUDA disponible."""

# ─── Reglas de negocio ────────────────────────────────────────────────────────

MIN_AGE_YEARS: int = 18
"""Edad mínima del solicitante en años."""

MAX_DEBT_RATIO: float = 0.35
"""Ratio máximo cuota/ingresos (regla de oro bancaria del 35%)."""

MIN_LOAN_AMOUNT: float = 1_000.0
"""Importe mínimo del préstamo en euros."""

MAX_LOAN_AMOUNT: float = 100_000.0
"""Importe máximo del préstamo en euros."""

MIN_LOAN_TERM_MONTHS: int = 6
MAX_LOAN_TERM_MONTHS: int = 120

# ─── API ─────────────────────────────────────────────────────────────────────

API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", 8000))
API_PREFIX: str = "/api/v1"

# ─── Logging ─────────────────────────────────────────────────────────────────

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
