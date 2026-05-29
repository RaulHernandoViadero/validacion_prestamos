"""
augmentation.py — Data augmentation para documentos sintéticos.

Aplica transformaciones realistas sobre imágenes de DNI y formularios para
simular las condiciones reales de captura con cámara de móvil o escáner:

  1. Rotación leve (±5° a ±15°): documentos no puestos perfectamente rectos.
  2. Ruido gaussiano y sal/pimienta: sensor de cámara, digitalización antigua.
  3. Variación de brillo y contraste: distintas condiciones de iluminación.
  4. Desenfoque gaussiano leve: fotos con móvil, baja calidad de cámara.
  5. Transformación de perspectiva: captura en ángulo (no frontal perfecta).
  6. Oclusión parcial aleatoria: dedos, bolígrafos, bordes del documento.
  7. Compresión JPEG simulada: pérdida de calidad al compartir por WhatsApp.

Todas las transformaciones actualizan también las anotaciones YOLO
(bounding boxes) para mantener la coherencia.

Uso típico:
    from src.data_generation.augmentation import DocumentAugmentor
    aug = DocumentAugmentor(seed=42)
    img_aug, rois_aug = aug.aplicar(imagen_pil, rois, intensidad="media")
"""

import math
import random
import logging
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.config import (
    RANDOM_SEED,
    AUG_ROTATION_RANGE,
    AUG_BRIGHTNESS_RANGE,
    AUG_CONTRAST_RANGE,
    AUG_OCCLUSION_PROB,
    AUG_BLUR_KERNEL_RANGE,
)

logger = logging.getLogger(__name__)

# Tipo para las ROIs: nombre → (x1, y1, x2, y2) en píxeles
RoisDict = dict[str, tuple[int, int, int, int]]


class Intensidad(str, Enum):
    """Nivel de intensidad del data augmentation."""
    SUAVE  = "suave"    # Cambios muy leves (±5°, ruido mínimo)
    MEDIA  = "media"    # Cambios moderados (especificaciones del TFM)
    FUERTE = "fuerte"   # Cambios agresivos (para aumentar variabilidad)


class DocumentAugmentor:
    """
    Aplica data augmentation realista sobre imágenes de documentos.

    Diseñado para mantener la legibilidad del texto (OCR) mientras
    introduce suficiente variabilidad para evitar el sobreajuste del
    modelo YOLO.

    Args:
        seed: Semilla aleatoria para reproducibilidad.
        occlusion_prob: Probabilidad de aplicar oclusión parcial [0, 1].
    """

    def __init__(
        self,
        seed: int = RANDOM_SEED,
        occlusion_prob: float = AUG_OCCLUSION_PROB,
    ) -> None:
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._occlusion_prob = occlusion_prob
        logger.info("DocumentAugmentor inicializado — seed=%d", seed)

    # ── API pública ───────────────────────────────────────────────────────────

    def aplicar(
        self,
        imagen: Image.Image,
        rois: RoisDict,
        intensidad: Intensidad | str = Intensidad.MEDIA,
        transformaciones: Optional[list[str]] = None,
    ) -> tuple[Image.Image, RoisDict]:
        """
        Aplica el pipeline completo de data augmentation.

        El orden de las transformaciones importa: primero las geométricas
        (que alteran las coordenadas de ROIs) y luego las fotométricas
        (que solo modifican píxeles).

        Args:
            imagen: Imagen PIL de entrada (modo RGB).
            rois: Diccionario {nombre_clase: (x1,y1,x2,y2)}.
            intensidad: Nivel de intensidad de las transformaciones.
            transformaciones: Lista de transformaciones a aplicar. Si None,
                              se aplican todas con probabilidad proporcional
                              a la intensidad.

        Returns:
            Tupla (imagen_aumentada, rois_actualizadas).
        """
        if isinstance(intensidad, str):
            intensidad = Intensidad(intensidad)

        img = imagen.copy()
        r = {k: list(v) for k, v in rois.items()}  # mutable

        pipeline = transformaciones or self._pipeline_por_intensidad(intensidad)

        for nombre in pipeline:
            metodo = getattr(self, f"_aug_{nombre}", None)
            if metodo is None:
                logger.warning("Transformación desconocida: %s", nombre)
                continue
            try:
                img, r = metodo(img, r, intensidad)
            except Exception as exc:
                logger.warning("Error en transformación '%s': %s", nombre, exc)

        rois_out = {k: tuple(v) for k, v in r.items()}
        return img, rois_out

    def aplicar_lote(
        self,
        imagen: Image.Image,
        rois: RoisDict,
        n_variantes: int = 5,
        intensidad: Intensidad | str = Intensidad.MEDIA,
    ) -> list[tuple[Image.Image, RoisDict]]:
        """
        Genera N variantes aumentadas de una misma imagen.

        Args:
            imagen: Imagen original.
            rois: ROIs originales.
            n_variantes: Número de variantes a generar.
            intensidad: Nivel de intensidad.

        Returns:
            Lista de N tuplas (imagen_aumentada, rois_actualizadas).
        """
        variantes = []
        for i in range(n_variantes):
            # Usar semilla diferente para cada variante
            self._rng = random.Random(self._rng.randint(0, 2**32))
            self._np_rng = np.random.default_rng(self._rng.randint(0, 2**32))
            img_aug, rois_aug = self.aplicar(imagen, rois, intensidad)
            variantes.append((img_aug, rois_aug))
        return variantes

    # ── Pipeline ─────────────────────────────────────────────────────────────

    def _pipeline_por_intensidad(self, intensidad: Intensidad) -> list[str]:
        """
        Devuelve la lista de transformaciones a aplicar según la intensidad.
        Cada transformación tiene su propia probabilidad interna de activación.
        """
        base = [
            "brillo",
            "contraste",
            "ruido_gaussiano",
        ]
        media = base + [
            "rotacion",
            "desenfoque",
            "ruido_sal_pimienta",
            "compresion_jpeg",
        ]
        fuerte = media + [
            "perspectiva",
            "oclusion",
            "saturacion",
            "distorsion_barrel",
        ]
        return {
            Intensidad.SUAVE:  base,
            Intensidad.MEDIA:  media,
            Intensidad.FUERTE: fuerte,
        }[intensidad]

    # ── Transformaciones geométricas ──────────────────────────────────────────

    def _aug_rotacion(
        self,
        img: Image.Image,
        rois: dict,
        intensidad: Intensidad,
    ) -> tuple[Image.Image, dict]:
        """
        Rotación aleatoria dentro del rango configurado.
        Fondo relleno con el color medio del borde de la imagen.
        """
        rangos = {
            Intensidad.SUAVE:  (-5.0,  5.0),
            Intensidad.MEDIA:  (-12.0, 12.0),
            Intensidad.FUERTE: (-18.0, 18.0),
        }
        lo, hi = rangos[intensidad]

        # Probabilidad de aplicar: 70% en media, 90% en fuerte
        prob = {Intensidad.SUAVE: 0.5, Intensidad.MEDIA: 0.7, Intensidad.FUERTE: 0.9}
        if self._rng.random() > prob[intensidad]:
            return img, rois

        angulo = self._rng.uniform(lo, hi)
        w, h = img.size

        # Color de relleno: blanco (documentos suelen tener fondo blanco)
        fill_color = (245, 243, 238)
        img_rot = img.rotate(
            angulo,
            resample=Image.BICUBIC,
            expand=False,
            fillcolor=fill_color,
        )

        # Transformar coordenadas de ROIs
        cx, cy = w / 2, h / 2
        rad = math.radians(-angulo)  # PIL rota en sentido horario para ángulos positivos
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        def rotar_punto(px: float, py: float) -> tuple[float, float]:
            px -= cx
            py -= cy
            rx = px * cos_a - py * sin_a + cx
            ry = px * sin_a + py * cos_a + cy
            return rx, ry

        rois_rot = {}
        for nombre, coords in rois.items():
            x1, y1, x2, y2 = coords
            # Rotar las 4 esquinas del bbox
            esquinas = [
                rotar_punto(x1, y1),
                rotar_punto(x2, y1),
                rotar_punto(x2, y2),
                rotar_punto(x1, y2),
            ]
            xs = [p[0] for p in esquinas]
            ys = [p[1] for p in esquinas]
            rois_rot[nombre] = [
                max(0, int(min(xs))),
                max(0, int(min(ys))),
                min(w, int(max(xs))),
                min(h, int(max(ys))),
            ]

        return img_rot, rois_rot

    def _aug_perspectiva(
        self,
        img: Image.Image,
        rois: dict,
        intensidad: Intensidad,
    ) -> tuple[Image.Image, dict]:
        """
        Transformación de perspectiva para simular captura en ángulo.
        Aplica una homografía que desplaza las esquinas aleatoriamente.
        """
        # Solo se aplica en fuerte con 60% de probabilidad
        probs = {Intensidad.SUAVE: 0.0, Intensidad.MEDIA: 0.25, Intensidad.FUERTE: 0.6}
        if self._rng.random() > probs[intensidad]:
            return img, rois

        w, h = img.size
        margen = {
            Intensidad.MEDIA:  int(min(w, h) * 0.04),
            Intensidad.FUERTE: int(min(w, h) * 0.08),
        }.get(intensidad, 0)

        def jitter() -> int:
            return self._rng.randint(-margen, margen)

        # Esquinas originales: TL, TR, BL, BR
        src = [(0, 0), (w, 0), (0, h), (w, h)]
        dst = [
            (0 + jitter(), 0 + jitter()),
            (w + jitter(), 0 + jitter()),
            (0 + jitter(), h + jitter()),
            (w + jitter(), h + jitter()),
        ]

        # Calcular coeficientes de la transformación perspectiva de PIL
        coeffs = self._calcular_coeffs_perspectiva(src, dst)
        img_persp = img.transform(
            (w, h),
            Image.PERSPECTIVE,
            coeffs,
            Image.BICUBIC,
        )

        # Aplicar la misma transformación a los centros de los ROIs (aproximado)
        rois_persp = {}
        for nombre, coords in rois.items():
            x1, y1, x2, y2 = coords
            cx_r = (x1 + x2) / 2
            cy_r = (y1 + y2) / 2
            ancho_r = x2 - x1
            alto_r = y2 - y1
            # Transformar el centro
            cx_t, cy_t = self._aplicar_perspectiva_punto(cx_r, cy_r, coeffs)
            rois_persp[nombre] = [
                max(0, int(cx_t - ancho_r / 2)),
                max(0, int(cy_t - alto_r / 2)),
                min(w, int(cx_t + ancho_r / 2)),
                min(h, int(cy_t + alto_r / 2)),
            ]

        return img_persp, rois_persp

    def _calcular_coeffs_perspectiva(
        self,
        src: list[tuple],
        dst: list[tuple],
    ) -> list[float]:
        """Calcula los 8 coeficientes de la transformación perspectiva de PIL."""
        # Resolver el sistema lineal Ax = b
        matrix = []
        for (x, y), (X, Y) in zip(dst, src):
            matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
            matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
        A = np.array(matrix, dtype=np.float64)
        b = np.array([coord for pt in src for coord in pt], dtype=np.float64)
        try:
            res = np.linalg.lstsq(A, b, rcond=None)[0]
            return list(res)
        except np.linalg.LinAlgError:
            return [1, 0, 0, 0, 1, 0, 0, 0]

    def _aplicar_perspectiva_punto(
        self,
        x: float, y: float,
        coeffs: list[float],
    ) -> tuple[float, float]:
        """Aplica la transformación perspectiva a un punto."""
        a, b, c, d, e, f, g, h = coeffs
        denom = g * x + h * y + 1
        if abs(denom) < 1e-10:
            return x, y
        return (a * x + b * y + c) / denom, (d * x + e * y + f) / denom

    def _aug_oclusion(
        self,
        img: Image.Image,
        rois: dict,
        intensidad: Intensidad,
    ) -> tuple[Image.Image, dict]:
        """
        Oclusión parcial aleatoria: rectangulos que simulan dedos, bolígrafos
        o bordes del documento tapados.
        """
        if self._rng.random() > self._occlusion_prob:
            return img, rois

        img_copy = img.copy()
        from PIL import ImageDraw as PIDraw
        draw = PIDraw.Draw(img_copy)
        w, h = img.size

        n_oclusiones = self._rng.randint(1, 3)
        for _ in range(n_oclusiones):
            tipo = self._rng.choice(["dedo", "borde", "objeto"])

            if tipo == "dedo":
                # Rectángulo estrecho y largo (dedo)
                ancho_d = self._rng.randint(int(w * 0.04), int(w * 0.08))
                alto_d  = self._rng.randint(int(h * 0.2),  int(h * 0.5))
                x_d = self._rng.randint(0, w - ancho_d)
                y_d = self._rng.randint(0, h - alto_d)
                color_dedo = (
                    self._rng.randint(180, 230),
                    self._rng.randint(140, 190),
                    self._rng.randint(120, 170),
                )
                draw.rectangle([x_d, y_d, x_d + ancho_d, y_d + alto_d],
                               fill=color_dedo)

            elif tipo == "borde":
                # Franja en uno de los 4 bordes (borde cortado)
                lado = self._rng.choice(["top", "bottom", "left", "right"])
                grosor = self._rng.randint(int(min(w, h) * 0.03),
                                           int(min(w, h) * 0.08))
                color_borde = (240, 238, 232)
                if lado == "top":
                    draw.rectangle([0, 0, w, grosor], fill=color_borde)
                elif lado == "bottom":
                    draw.rectangle([0, h - grosor, w, h], fill=color_borde)
                elif lado == "left":
                    draw.rectangle([0, 0, grosor, h], fill=color_borde)
                else:
                    draw.rectangle([w - grosor, 0, w, h], fill=color_borde)

            else:
                # Objeto pequeño (bolígrafo, tarjeta encima)
                ancho_o = self._rng.randint(int(w * 0.05), int(w * 0.15))
                alto_o  = self._rng.randint(int(h * 0.02), int(h * 0.06))
                x_o = self._rng.randint(0, w - ancho_o)
                y_o = self._rng.randint(0, h - alto_o)
                color_obj = (
                    self._rng.randint(20, 80),
                    self._rng.randint(20, 80),
                    self._rng.randint(20, 80),
                )
                draw.rectangle([x_o, y_o, x_o + ancho_o, y_o + alto_o],
                               fill=color_obj)

        return img_copy, rois  # ROIs no cambian con oclusión

    def _aug_distorsion_barrel(
        self,
        img: Image.Image,
        rois: dict,
        intensidad: Intensidad,
    ) -> tuple[Image.Image, dict]:
        """
        Distorsión de barril/cojín leve: simula la óptica de cámaras de móvil
        de gama baja. Implementada mediante remapeo numpy.
        """
        if self._rng.random() > 0.4:
            return img, rois

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        k = self._rng.uniform(-0.08, 0.08)  # coeficiente de distorsión

        cx, cy = w / 2, h / 2
        y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)

        # Normalizar a [-1, 1]
        xn = (x_grid - cx) / cx
        yn = (y_grid - cy) / cy
        r2 = xn ** 2 + yn ** 2

        # Aplicar distorsión radial
        xnd = xn * (1 + k * r2)
        ynd = yn * (1 + k * r2)

        # Desnormalizar
        x_src = xnd * cx + cx
        y_src = ynd * cy + cy

        # Remapear (interpolación bilineal manual con numpy)
        x_src = np.clip(x_src, 0, w - 1)
        y_src = np.clip(y_src, 0, h - 1)

        x0 = np.floor(x_src).astype(int)
        y0 = np.floor(y_src).astype(int)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)

        dx = (x_src - x0)[..., np.newaxis]
        dy = (y_src - y0)[..., np.newaxis]

        resultado = (
            arr[y0, x0] * (1 - dx) * (1 - dy) +
            arr[y0, x1] * dx * (1 - dy) +
            arr[y1, x0] * (1 - dx) * dy +
            arr[y1, x1] * dx * dy
        ).astype(np.uint8)

        return Image.fromarray(resultado), rois

    # ── Transformaciones fotométricas ─────────────────────────────────────────

    def _aug_brillo(
        self,
        img: Image.Image,
        rois: dict,
        intensidad: Intensidad,
    ) -> tuple[Image.Image, dict]:
        """Ajuste de brillo para simular distintas condiciones de iluminación."""
        rangos = {
            Intensidad.SUAVE:  (0.85, 1.15),
            Intensidad.MEDIA:  AUG_BRIGHTNESS_RANGE,
            Intensidad.FUERTE: (0.55, 1.50),
        }
        lo, hi = rangos[intensidad]
        factor = self._rng.uniform(lo, hi)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor), rois

    def _aug_contraste(
        self,
        img: Image.Image,
        rois: dict,
        intensidad: Intensidad,
    ) -> tuple[Image.Image, dict]:
        """Ajuste de contraste."""
        rangos = {
            Intensidad.SUAVE:  (0.88, 1.12),
            Intensidad.MEDIA:  AUG_CONTRAST_RANGE,
            Intensidad.FUERTE: (0.55, 1.60),
        }
        lo, hi = rangos[intensidad]
        factor = self._rng.uniform(lo, hi)
        return ImageEnhance.Contrast(img).enhance(factor), rois

    def _aug_saturacion(
        self,
        img: Image.Image,
        rois: dict,
        intensidad: Intensidad,
    ) -> tuple[Image.Image, dict]:
        """Ajuste de saturación (escáneres pueden desaturar colores)."""
        factor = self._rng.uniform(0.5, 1.3)
        return ImageEnhance.Color(img).enhance(factor), rois

    def _aug_ruido_gaussiano(
        self,
        img: Image.Image,
        rois: dict,
        intensidad: Intensidad,
    ) -> tuple[Image.Image, dict]:
        """Añade ruido gaussiano para simular el sensor de la cámara."""
        sigmas = {
            Intensidad.SUAVE:  (1.0, 5.0),
            Intensidad.MEDIA:  (2.0, 12.0),
            Intensidad.FUERTE: (5.0, 25.0),
        }
        lo, hi = sigmas[intensidad]
        sigma = self._rng.uniform(lo, hi)

        arr = np.array(img, dtype=np.float32)
        ruido = self._np_rng.normal(0, sigma, arr.shape).astype(np.float32)
        arr_ruidosa = np.clip(arr + ruido, 0, 255).astype(np.uint8)
        return Image.fromarray(arr_ruidosa), rois

    def _aug_ruido_sal_pimienta(
        self,
        img: Image.Image,
        rois: dict,
        intensidad: Intensidad,
    ) -> tuple[Image.Image, dict]:
        """Añade ruido sal y pimienta para simular digitalización antigua."""
        densidades = {
            Intensidad.SUAVE:  0.002,
            Intensidad.MEDIA:  0.005,
            Intensidad.FUERTE: 0.015,
        }
        densidad = densidades[intensidad]

        arr = np.array(img)
        n_pixeles = arr.shape[0] * arr.shape[1]
        n_sal = int(n_pixeles * densidad / 2)
        n_pim = int(n_pixeles * densidad / 2)

        # Sal (blanco)
        coords_sal = [
            self._np_rng.integers(0, arr.shape[0], n_sal),
            self._np_rng.integers(0, arr.shape[1], n_sal),
        ]
        arr[coords_sal[0], coords_sal[1]] = [255, 255, 255]

        # Pimienta (negro)
        coords_pim = [
            self._np_rng.integers(0, arr.shape[0], n_pim),
            self._np_rng.integers(0, arr.shape[1], n_pim),
        ]
        arr[coords_pim[0], coords_pim[1]] = [0, 0, 0]

        return Image.fromarray(arr), rois

    def _aug_desenfoque(
        self,
        img: Image.Image,
        rois: dict,
        intensidad: Intensidad,
    ) -> tuple[Image.Image, dict]:
        """Desenfoque gaussiano leve para simular fotos movidas o desenfocadas."""
        radios = {
            Intensidad.SUAVE:  (0.3, 0.8),
            Intensidad.MEDIA:  (0.5, 1.5),
            Intensidad.FUERTE: (0.8, 3.0),
        }
        lo, hi = radios[intensidad]

        # Solo aplicar el 60% de las veces
        if self._rng.random() > 0.6:
            return img, rois

        radio = self._rng.uniform(lo, hi)
        return img.filter(ImageFilter.GaussianBlur(radius=radio)), rois

    def _aug_compresion_jpeg(
        self,
        img: Image.Image,
        rois: dict,
        intensidad: Intensidad,
    ) -> tuple[Image.Image, dict]:
        """
        Simula la pérdida de calidad por compresión JPEG (artefactos de bloque).
        Típico al compartir fotos por WhatsApp o email.
        """
        calidades = {
            Intensidad.SUAVE:  (85, 95),
            Intensidad.MEDIA:  (65, 85),
            Intensidad.FUERTE: (40, 70),
        }
        lo, hi = calidades[intensidad]

        # Solo aplicar el 50% de las veces
        if self._rng.random() > 0.5:
            return img, rois

        calidad = self._rng.randint(lo, hi)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=calidad)
        buffer.seek(0)
        img_jpeg = Image.open(buffer).convert("RGB")
        return img_jpeg, rois
