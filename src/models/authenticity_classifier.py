"""
authenticity_classifier.py — Clasificador CNN de autenticidad documental.

Implementa un clasificador binario basado en ResNet-18 fine-tuned que
distingue entre documentos legítimos (generados con la plantilla correcta)
y documentos manipulados/inconsistentes (plantilla alterada, campos fuera
de posición, fuentes incorrectas).

Arquitectura:
  - Backbone: ResNet-18 pre-entrenado en ImageNet (torchvision)
  - Capa final reemplazada: Linear(512 → 2) + Dropout(0.3)
  - Input: imagen completa del documento redimensionada a 224×224
  - Output: probabilidad [P(legítimo), P(manipulado)]

Métricas objetivo (TFM):
  - F1-score  > 0.88
  - AUC-ROC   > 0.90

Uso típico:
    from src.models.authenticity_classifier import AuthenticityClassifier
    clf = AuthenticityClassifier()
    clf.entrenar(train_dir, val_dir)
    resultado = clf.predecir(imagen_pil)  # {"etiqueta": "LEGÍTIMO", "confianza": 0.97}
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

from src.config import (
    WEIGHTS_DIR,
    SPLITS_DIR,
    CLASSIFIER_IMGSZ,
    CLASSIFIER_EPOCHS,
    CLASSIFIER_BATCH,
    CLASSIFIER_LR,
    CLASSIFIER_DROPOUT,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)

# Fichero de pesos del clasificador
_PESOS_CLF = "authenticity_classifier.pth"

# Etiquetas
_LABEL_LEGITIMO    = "LEGÍTIMO"
_LABEL_MANIPULADO  = "MANIPULADO"
_CLASES = [_LABEL_LEGITIMO, _LABEL_MANIPULADO]  # índice 0 = legítimo, 1 = manipulado

# Normalización ImageNet (requerida por ResNet pre-entrenado)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── Dataset personalizado ────────────────────────────────────────────────────

class DocumentDataset(Dataset):
    """
    Dataset PyTorch para el clasificador de autenticidad.

    Espera la siguiente estructura de directorios:
        directorio/
          legitimo/      ← imágenes de documentos consistentes
          manipulado/    ← imágenes de documentos inconsistentes

    Args:
        directorio: Ruta al directorio con las dos subcarpetas.
        transform: Transformaciones de torchvision a aplicar.
    """

    def __init__(self, directorio: Path, transform=None) -> None:
        self._directorio = Path(directorio)
        self._transform  = transform
        self._muestras: list[tuple[Path, int]] = []

        for label_idx, label_name in enumerate(_CLASES):
            subdir = self._directorio / label_name.lower().replace("é", "e").replace("í", "i")
            # Intentar también con nombre sin acentos
            if not subdir.exists():
                subdir = self._directorio / ("legitimo" if label_idx == 0 else "manipulado")
            if subdir.exists():
                for ext in ("*.png", "*.jpg", "*.jpeg"):
                    for img_path in sorted(subdir.glob(ext)):
                        self._muestras.append((img_path, label_idx))
            else:
                logger.warning("Subdirectorio no encontrado: %s", subdir)

        logger.info(
            "DocumentDataset cargado: %d imágenes en %s",
            len(self._muestras), directorio,
        )

    def __len__(self) -> int:
        return len(self._muestras)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        ruta, etiqueta = self._muestras[idx]
        imagen = Image.open(ruta).convert("RGB")
        if self._transform:
            imagen = self._transform(imagen)
        return imagen, etiqueta


# ─── Modelo ───────────────────────────────────────────────────────────────────

def _construir_modelo(dropout: float = CLASSIFIER_DROPOUT) -> nn.Module:
    """
    Construye ResNet-18 con la capa final reemplazada para clasificación binaria.

    Args:
        dropout: Probabilidad de dropout antes de la capa final.

    Returns:
        Modelo PyTorch listo para fine-tuning.
    """
    modelo = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Congelar todas las capas excepto layer4 y fc (fine-tuning parcial)
    for nombre, parametro in modelo.named_parameters():
        if not (nombre.startswith("layer4") or nombre.startswith("fc")):
            parametro.requires_grad = False

    # Reemplazar la capa final
    num_features = modelo.fc.in_features  # 512 en ResNet-18
    modelo.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(num_features, 2),
    )

    return modelo


# ─── Transformaciones ─────────────────────────────────────────────────────────

def _transforms_train() -> transforms.Compose:
    """Transformaciones con augmentation para el split de entrenamiento."""
    return transforms.Compose([
        transforms.Resize((CLASSIFIER_IMGSZ + 32, CLASSIFIER_IMGSZ + 32)),
        transforms.RandomCrop(CLASSIFIER_IMGSZ),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _transforms_eval() -> transforms.Compose:
    """Transformaciones sin augmentation para validación y test."""
    return transforms.Compose([
        transforms.Resize((CLASSIFIER_IMGSZ, CLASSIFIER_IMGSZ)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ─── Clase principal ──────────────────────────────────────────────────────────

class AuthenticityClassifier:
    """
    Clasificador de autenticidad de documentos basado en ResNet-18.

    Distingue entre documentos con estructura visual correcta (legítimos)
    y documentos con alteraciones deliberadas (manipulados).

    Args:
        weights_dir: Directorio de pesos.
        device: 'cpu' o 'cuda' / 'mps'.
        epochs: Epochs de entrenamiento.
        batch_size: Tamaño de batch.
        lr: Learning rate.
    """

    def __init__(
        self,
        weights_dir: Path = WEIGHTS_DIR,
        device: Optional[str] = None,
        epochs: int = CLASSIFIER_EPOCHS,
        batch_size: int = CLASSIFIER_BATCH,
        lr: float = CLASSIFIER_LR,
    ) -> None:
        self._weights_dir = Path(weights_dir)
        self._weights_dir.mkdir(parents=True, exist_ok=True)
        self._epochs     = epochs
        self._batch_size = batch_size
        self._lr         = lr

        # Selección automática de device
        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        self._modelo: Optional[nn.Module] = None
        self._transform_eval = _transforms_eval()

        logger.info("AuthenticityClassifier — device=%s, epochs=%d, batch=%d",
                    self._device, epochs, batch_size)

    # ── API pública ───────────────────────────────────────────────────────────

    def entrenar(
        self,
        train_dir: Path,
        val_dir: Path,
        guardar: bool = True,
    ) -> dict:
        """
        Entrena el clasificador con los datos indicados.

        Args:
            train_dir: Directorio de entrenamiento (con subcarpetas legitimo/manipulado).
            val_dir: Directorio de validación.
            guardar: Si True, guarda los mejores pesos al finalizar.

        Returns:
            Diccionario con métricas finales (f1, auc_roc, accuracy).
        """
        torch.manual_seed(RANDOM_SEED)

        # Datasets y loaders
        ds_train = DocumentDataset(Path(train_dir), transform=_transforms_train())
        ds_val   = DocumentDataset(Path(val_dir),   transform=_transforms_eval())

        if len(ds_train) == 0:
            raise ValueError(
                f"Dataset de entrenamiento vacío: {train_dir}\n"
                "Asegúrate de haber ejecutado generate_dataset.py primero."
            )

        dl_train = DataLoader(ds_train, batch_size=self._batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
        dl_val   = DataLoader(ds_val,   batch_size=self._batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

        # Modelo, optimizador y función de pérdida
        modelo = _construir_modelo().to(self._device)
        optimizador = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, modelo.parameters()),
            lr=self._lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizador, T_max=self._epochs, eta_min=1e-6
        )
        criterio = nn.CrossEntropyLoss()

        mejor_f1 = 0.0
        mejor_estado = None
        historial: list[dict] = []

        logger.info("Inicio entrenamiento — %d train, %d val", len(ds_train), len(ds_val))

        for epoch in range(1, self._epochs + 1):
            # ── Entrenamiento ─────────────────────────────────────────────────
            modelo.train()
            loss_train = 0.0
            for imgs, etiquetas in dl_train:
                imgs      = imgs.to(self._device)
                etiquetas = etiquetas.to(self._device)
                optimizador.zero_grad()
                salidas = modelo(imgs)
                loss = criterio(salidas, etiquetas)
                loss.backward()
                optimizador.step()
                loss_train += loss.item()

            scheduler.step()
            loss_train /= max(len(dl_train), 1)

            # ── Validación ────────────────────────────────────────────────────
            metricas_val = self._evaluar_loader(modelo, dl_val)
            f1_val       = metricas_val["f1"]

            historial.append({
                "epoch": epoch,
                "loss_train": round(loss_train, 4),
                **metricas_val,
            })

            logger.info(
                "Epoch %3d/%d | loss=%.4f | acc=%.4f | f1=%.4f | auc=%.4f | lr=%.2e",
                epoch, self._epochs,
                loss_train,
                metricas_val["accuracy"],
                f1_val,
                metricas_val["auc_roc"],
                optimizador.param_groups[0]["lr"],
            )

            # Guardar mejor modelo (criterio: F1)
            if f1_val > mejor_f1:
                mejor_f1     = f1_val
                mejor_estado = {k: v.cpu().clone() for k, v in modelo.state_dict().items()}
                logger.info("  ↑ Nuevo mejor modelo (F1=%.4f)", mejor_f1)

        # Cargar mejor estado
        if mejor_estado:
            modelo.load_state_dict(mejor_estado)

        self._modelo = modelo

        # Guardar pesos
        if guardar and mejor_estado:
            self._guardar_pesos(mejor_estado, historial)

        return historial[-1] if historial else {}

    def predecir(self, imagen: Image.Image) -> dict:
        """
        Clasifica una imagen de documento.

        Args:
            imagen: Imagen PIL (modo RGB).

        Returns:
            Diccionario con:
              - "etiqueta":   "LEGÍTIMO" o "MANIPULADO"
              - "confianza":  probabilidad de la clase predicha [0, 1]
              - "prob_legitimo":  probabilidad clase 0
              - "prob_manipulado": probabilidad clase 1
        """
        if self._modelo is None:
            self.cargar_pesos()

        self._modelo.eval()
        tensor = self._transform_eval(imagen).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._modelo(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()

        idx_pred   = int(torch.argmax(torch.tensor(probs)))
        etiqueta   = _CLASES[idx_pred]
        confianza  = probs[idx_pred]

        return {
            "etiqueta":        etiqueta,
            "confianza":       round(confianza, 4),
            "prob_legitimo":   round(probs[0], 4),
            "prob_manipulado": round(probs[1], 4),
        }

    def predecir_lote(self, imagenes: list[Image.Image]) -> list[dict]:
        """Clasifica una lista de imágenes en un solo pase forward."""
        if self._modelo is None:
            self.cargar_pesos()

        self._modelo.eval()
        tensores = torch.stack([self._transform_eval(img) for img in imagenes])
        tensores = tensores.to(self._device)

        with torch.no_grad():
            logits = self._modelo(tensores)
            probs  = torch.softmax(logits, dim=1).cpu().tolist()

        resultados = []
        for p in probs:
            idx = int(p[0] < p[1])
            resultados.append({
                "etiqueta":        _CLASES[idx],
                "confianza":       round(p[idx], 4),
                "prob_legitimo":   round(p[0], 4),
                "prob_manipulado": round(p[1], 4),
            })
        return resultados

    def cargar_pesos(self, ruta: Optional[Path] = None) -> None:
        """
        Carga los pesos entrenados desde disco.

        Args:
            ruta: Ruta al fichero .pth. Si None, usa la ruta por defecto.
        """
        ruta = Path(ruta) if ruta else self._weights_dir / _PESOS_CLF
        if not ruta.exists():
            raise FileNotFoundError(
                f"Pesos del clasificador no encontrados: {ruta}\n"
                "Entrena el modelo primero con AuthenticityClassifier.entrenar()"
            )
        modelo = _construir_modelo().to(self._device)
        estado = torch.load(str(ruta), map_location=self._device)
        modelo.load_state_dict(estado)
        modelo.eval()
        self._modelo = modelo
        logger.info("Pesos cargados desde: %s", ruta)

    # ── Utilidades internas ───────────────────────────────────────────────────

    def _evaluar_loader(self, modelo: nn.Module, loader: DataLoader) -> dict:
        """
        Evalúa el modelo sobre un DataLoader y calcula métricas.

        Returns:
            Diccionario con accuracy, f1, auc_roc, precision, recall.
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score,
            precision_score, recall_score,
        )

        modelo.eval()
        all_labels: list[int] = []
        all_preds:  list[int] = []
        all_probs:  list[float] = []

        with torch.no_grad():
            for imgs, etiquetas in loader:
                imgs = imgs.to(self._device)
                logits = modelo(imgs)
                probs  = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                preds  = logits.argmax(dim=1).cpu().tolist()
                all_labels.extend(etiquetas.tolist())
                all_preds.extend(preds)
                all_probs.extend(probs)

        if len(set(all_labels)) < 2:
            # Solo una clase presente → AUC no definida
            return {
                "accuracy": accuracy_score(all_labels, all_preds),
                "f1":       f1_score(all_labels, all_preds, zero_division=0),
                "auc_roc":  0.5,
                "precision": precision_score(all_labels, all_preds, zero_division=0),
                "recall":    recall_score(all_labels, all_preds, zero_division=0),
            }

        return {
            "accuracy":  round(accuracy_score(all_labels, all_preds), 4),
            "f1":        round(f1_score(all_labels, all_preds, zero_division=0), 4),
            "auc_roc":   round(roc_auc_score(all_labels, all_probs), 4),
            "precision": round(precision_score(all_labels, all_preds, zero_division=0), 4),
            "recall":    round(recall_score(all_labels, all_preds, zero_division=0), 4),
        }

    def _guardar_pesos(self, estado: dict, historial: list[dict]) -> None:
        """Guarda el mejor estado del modelo y el historial de entrenamiento."""
        ruta_pesos = self._weights_dir / _PESOS_CLF
        torch.save(estado, str(ruta_pesos))
        logger.info("Pesos del clasificador guardados: %s", ruta_pesos)

        # Guardar historial de entrenamiento
        ruta_hist = self._weights_dir / "classifier_history.json"
        mejor = max(historial, key=lambda x: x["f1"])
        datos = {
            "fecha":    datetime.now().isoformat(),
            "epochs":   self._epochs,
            "device":   str(self._device),
            "mejor_epoch": mejor["epoch"],
            "metricas_finales": mejor,
            "objetivos": {"f1": 0.88, "auc_roc": 0.90},
            "cumple_objetivos": {
                "f1":      mejor["f1"]      >= 0.88,
                "auc_roc": mejor["auc_roc"] >= 0.90,
            },
            "historial": historial,
        }
        ruta_hist.write_text(
            json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Historial guardado: %s", ruta_hist)


# Variable global para evitar typo en el nombre
_PESOS_CLF = _PESOS_CLF
