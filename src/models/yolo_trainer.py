"""
yolo_trainer.py — Entrenamiento del modelo YOLOv8 para detección de ROIs.

Entrena dos modelos independientes:
  - Modelo DNI:       detecta los 9 campos del documento nacional de identidad.
  - Modelo Formulario: detecta los 14 campos del formulario de préstamo.

Ambos modelos parten de YOLOv8n (nano) pre-entrenado en COCO y se
fine-tunean sobre el dataset sintético generado en la Fase 1.

Métricas objetivo (TFM):
  - mAP@0.5        > 0.85
  - mAP@0.5:0.95   > 0.65

Uso típico:
    from src.models.yolo_trainer import YOLOTrainer
    trainer = YOLOTrainer()
    results_dni  = trainer.entrenar(tipo="dni")
    results_loan = trainer.entrenar(tipo="prestamo")

O desde CLI:
    python -m src.models.yolo_trainer --tipo dni
    python -m src.models.yolo_trainer --tipo prestamo
    python -m src.models.yolo_trainer --tipo ambos
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Literal

from src.config import (
    WEIGHTS_DIR,
    SPLITS_DIR,
    YOLO_BASE_MODEL,
    YOLO_EPOCHS,
    YOLO_BATCH,
    YOLO_IMGSZ,
    YOLO_LR0,
    YOLO_PATIENCE,
    DNI_CLASSES,
    LOAN_CLASSES,
    RANDOM_SEED,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
)

logger = logging.getLogger(__name__)

TipoModelo = Literal["dni", "prestamo", "ambos"]

# Nombres de los ficheros de pesos guardados
_PESOS_DNI     = "yolo_dni.pt"
_PESOS_LOAN    = "yolo_loan.pt"


class YOLOTrainer:
    """
    Entrena y evalúa modelos YOLOv8 para detección de ROIs en documentos.

    Cada entrenamiento genera:
      - Pesos del mejor modelo en `weights/yolo_{tipo}.pt`
      - Métricas en `runs/detect/{nombre_experimento}/`
      - Fichero de resultados JSON en `weights/`

    Args:
        weights_dir: Directorio donde se guardan los pesos entrenados.
        splits_dir: Directorio raíz de los splits del dataset.
        epochs: Número de epochs de entrenamiento.
        batch: Tamaño del batch.
        imgsz: Tamaño de imagen de entrada para YOLO.
        lr0: Learning rate inicial.
        patience: Epochs sin mejora antes de early stopping.
        device: Dispositivo de entrenamiento ('cpu', '0', 'cuda:0').
    """

    def __init__(
        self,
        weights_dir: Path = WEIGHTS_DIR,
        splits_dir: Path = SPLITS_DIR,
        epochs: int = YOLO_EPOCHS,
        batch: int = YOLO_BATCH,
        imgsz: int = YOLO_IMGSZ,
        lr0: float = YOLO_LR0,
        patience: int = YOLO_PATIENCE,
        device: str = "cpu",
    ) -> None:
        self._weights_dir = Path(weights_dir)
        self._weights_dir.mkdir(parents=True, exist_ok=True)
        self._splits_dir  = Path(splits_dir)
        self._epochs   = epochs
        self._batch    = batch
        self._imgsz    = imgsz
        self._lr0      = lr0
        self._patience = patience
        self._device   = device

        logger.info(
            "YOLOTrainer — epochs=%d, batch=%d, imgsz=%d, device=%s",
            epochs, batch, imgsz, device,
        )

    # ── API pública ───────────────────────────────────────────────────────────

    def entrenar(self, tipo: TipoModelo = "ambos") -> dict:
        """
        Lanza el entrenamiento del modelo indicado.

        Args:
            tipo: "dni", "prestamo" o "ambos".

        Returns:
            Diccionario con las métricas finales de cada modelo entrenado.
        """
        from ultralytics import YOLO  # import tardío para no bloquear el módulo

        resultados = {}

        if tipo in ("dni", "ambos"):
            resultados["dni"] = self._entrenar_modelo(
                YOLO,
                tipo="dni",
                yaml_path=self._splits_dir / "dataset_dni.yaml",
                nombre_exp=f"yolo_dni_{datetime.now().strftime('%Y%m%d_%H%M')}",
                pesos_salida=self._weights_dir / _PESOS_DNI,
                n_clases=len(DNI_CLASSES),
            )

        if tipo in ("prestamo", "ambos"):
            resultados["prestamo"] = self._entrenar_modelo(
                YOLO,
                tipo="prestamo",
                yaml_path=self._splits_dir / "dataset_prestamo.yaml",
                nombre_exp=f"yolo_loan_{datetime.now().strftime('%Y%m%d_%H%M')}",
                pesos_salida=self._weights_dir / _PESOS_LOAN,
                n_clases=len(LOAN_CLASSES),
            )

        return resultados

    def evaluar(self, tipo: TipoModelo = "ambos") -> dict:
        """
        Evalúa el modelo ya entrenado sobre el split de test.

        Args:
            tipo: "dni", "prestamo" o "ambos".

        Returns:
            Diccionario con métricas mAP@0.5 y mAP@0.5:0.95 por modelo.
        """
        from ultralytics import YOLO

        resultados = {}

        if tipo in ("dni", "ambos"):
            pesos = self._weights_dir / _PESOS_DNI
            if not pesos.exists():
                logger.error("Pesos DNI no encontrados: %s", pesos)
            else:
                resultados["dni"] = self._evaluar_modelo(
                    YOLO, pesos,
                    yaml_path=self._splits_dir / "dataset_dni.yaml",
                    split="test",
                )

        if tipo in ("prestamo", "ambos"):
            pesos = self._weights_dir / _PESOS_LOAN
            if not pesos.exists():
                logger.error("Pesos formulario no encontrados: %s", pesos)
            else:
                resultados["prestamo"] = self._evaluar_modelo(
                    YOLO, pesos,
                    yaml_path=self._splits_dir / "dataset_prestamo.yaml",
                    split="test",
                )

        return resultados

    # ── Lógica interna ────────────────────────────────────────────────────────

    def _entrenar_modelo(
        self,
        YOLO,
        tipo: str,
        yaml_path: Path,
        nombre_exp: str,
        pesos_salida: Path,
        n_clases: int,
    ) -> dict:
        """
        Entrena un único modelo YOLOv8 y guarda los mejores pesos.

        Returns:
            Diccionario con métricas finales del entrenamiento.
        """
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"YAML del dataset no encontrado: {yaml_path}\n"
                f"Ejecuta primero: python generate_dataset.py"
            )

        logger.info("=" * 55)
        logger.info("Entrenando modelo YOLO — tipo=%s", tipo.upper())
        logger.info("  Dataset:  %s", yaml_path)
        logger.info("  Clases:   %d", n_clases)
        logger.info("  Epochs:   %d | Batch: %d | Imgsz: %d", self._epochs, self._batch, self._imgsz)
        logger.info("  Device:   %s", self._device)
        logger.info("=" * 55)

        modelo = YOLO(YOLO_BASE_MODEL)

        results = modelo.train(
            data       = str(yaml_path),
            epochs     = self._epochs,
            batch      = self._batch,
            imgsz      = self._imgsz,
            lr0        = self._lr0,
            patience   = self._patience,
            device     = self._device,
            name       = nombre_exp,
            seed       = RANDOM_SEED,
            # Augmentation interna de YOLO (complementa la nuestra)
            degrees    = 5.0,        # rotación leve adicional
            translate  = 0.1,
            scale      = 0.3,
            fliplr     = 0.0,        # no reflejar horizontalmente (rompería texto)
            flipud     = 0.0,
            mosaic     = 0.5,
            mixup      = 0.0,
            # Optimizador
            optimizer  = "AdamW",
            weight_decay = 0.0005,
            warmup_epochs = 3,
            # Guardado
            save       = True,
            save_period = 10,
            # Verbosidad
            verbose    = True,
            exist_ok   = True,
        )

        # Copiar mejor modelo a weights/
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            import shutil
            shutil.copy2(best_pt, pesos_salida)
            logger.info("Mejores pesos guardados en: %s", pesos_salida)
        else:
            logger.warning("No se encontró best.pt en %s", results.save_dir)

        # Extraer métricas finales
        metricas = self._extraer_metricas(results)
        self._loguear_metricas(tipo, metricas)
        self._guardar_metricas_json(tipo, metricas, nombre_exp)

        return metricas

    def _evaluar_modelo(
        self,
        YOLO,
        pesos: Path,
        yaml_path: Path,
        split: str = "test",
    ) -> dict:
        """Evalúa un modelo sobre un split y devuelve las métricas."""
        modelo = YOLO(str(pesos))
        metrics = modelo.val(
            data   = str(yaml_path),
            split  = split,
            imgsz  = self._imgsz,
            device = self._device,
            verbose = True,
        )
        return self._extraer_metricas(metrics)

    def _extraer_metricas(self, results) -> dict:
        """
        Extrae las métricas más relevantes del objeto results de ultralytics.

        Returns:
            Diccionario con mAP50, mAP50_95, precision, recall.
        """
        try:
            box = results.box
            return {
                "mAP50":     float(box.map50),
                "mAP50_95":  float(box.map),
                "precision": float(box.mp),
                "recall":    float(box.mr),
            }
        except AttributeError:
            # Fallback para versiones antiguas de ultralytics
            try:
                return {
                    "mAP50":     float(results.results_dict.get("metrics/mAP50(B)", 0)),
                    "mAP50_95":  float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
                    "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
                    "recall":    float(results.results_dict.get("metrics/recall(B)", 0)),
                }
            except Exception:
                logger.warning("No se pudieron extraer métricas del resultado")
                return {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0}

    def _loguear_metricas(self, tipo: str, metricas: dict) -> None:
        """Muestra las métricas en el log con comparación vs objetivos."""
        obj_map50    = 0.85
        obj_map5095  = 0.65
        ok50   = "✓" if metricas["mAP50"]    >= obj_map50   else "✗"
        ok5095 = "✓" if metricas["mAP50_95"] >= obj_map5095 else "✗"

        logger.info("─" * 55)
        logger.info("Resultados YOLO — %s", tipo.upper())
        logger.info("  mAP@0.5:       %.4f  %s (objetivo >%.2f)",
                    metricas["mAP50"],    ok50,   obj_map50)
        logger.info("  mAP@0.5:0.95:  %.4f  %s (objetivo >%.2f)",
                    metricas["mAP50_95"], ok5095, obj_map5095)
        logger.info("  Precision:     %.4f", metricas["precision"])
        logger.info("  Recall:        %.4f", metricas["recall"])
        logger.info("─" * 55)

    def _guardar_metricas_json(
        self, tipo: str, metricas: dict, nombre_exp: str
    ) -> None:
        """Guarda las métricas en un fichero JSON junto a los pesos."""
        import json
        ruta = self._weights_dir / f"metrics_{tipo}.json"
        datos = {
            "experimento": nombre_exp,
            "fecha": datetime.now().isoformat(),
            "tipo": tipo,
            "metricas": metricas,
            "objetivos": {"mAP50": 0.85, "mAP50_95": 0.65},
            "cumple_objetivos": {
                "mAP50":    metricas["mAP50"]    >= 0.85,
                "mAP50_95": metricas["mAP50_95"] >= 0.65,
            },
        }
        ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Métricas guardadas: %s", ruta)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena los modelos YOLOv8 para detección de ROIs."
    )
    parser.add_argument(
        "--tipo", choices=["dni", "prestamo", "ambos"], default="ambos",
        help="Modelo a entrenar (default: ambos)",
    )
    parser.add_argument(
        "--epochs", type=int, default=YOLO_EPOCHS,
        help=f"Número de epochs (default: {YOLO_EPOCHS})",
    )
    parser.add_argument(
        "--batch", type=int, default=YOLO_BATCH,
        help=f"Tamaño de batch (default: {YOLO_BATCH})",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Dispositivo: 'cpu', '0' para GPU (default: cpu)",
    )
    parser.add_argument(
        "--solo-eval", action="store_true",
        help="Solo evalúa modelos ya entrenados (no entrena)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    args = _parse_args()
    trainer = YOLOTrainer(
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
    )

    if args.solo_eval:
        resultados = trainer.evaluar(tipo=args.tipo)
    else:
        resultados = trainer.entrenar(tipo=args.tipo)

    print("\nResultados finales:")
    for tipo, metricas in resultados.items():
        print(f"  {tipo}: mAP@0.5={metricas['mAP50']:.4f}, "
              f"mAP@0.5:0.95={metricas['mAP50_95']:.4f}")
