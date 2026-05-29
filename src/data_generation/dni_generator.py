"""
dni_generator.py — Generador de imágenes sintéticas del DNI español.

Produce imágenes PNG del anverso del DNI español (3.ª generación, vigente desde 2015)
a partir de los datos de un objeto DatosDNI. Las imágenes son 100% sintéticas y
están diseñadas para:
  1. Entrenar el detector YOLOv8 (posiciones de ROIs bien definidas).
  2. Servir como entrada al pipeline OCR (texto legible y bien posicionado).
  3. Parecer documentos reales ante el clasificador de autenticidad.

Diseño visual basado en el DNI español ID-1 (85,6 × 53,98 mm):
  - Cabecera burdeos con escudo de España simplificado y estrellas EU.
  - Franja izquierda con estrella EU y banda de color.
  - Área de foto con silueta procedural.
  - Área de firma con trazos Bézier aleatorios.
  - Todos los campos de texto con tipografía correcta.
  - Fondo con patrón de seguridad guilloche.
  - Zona MRZ con fuente monoespaciada.

Uso típico:
    from src.data_generation.dni_generator import DNIGenerator
    gen = DNIGenerator(output_dir=Path("data/raw/dnis"), seed=42)
    ruta_imagen, rois = gen.generar(datos_dni, expediente_id="EXP00001")
"""

import math
import random
import logging
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from src.config import (
    DNI_WIDTH_PX,
    DNI_HEIGHT_PX,
    RAW_DNIS_DIR,
    RANDOM_SEED,
    DNI_CLASSES,
)
from src.data_generation.fake_data_factory import DatosDNI

logger = logging.getLogger(__name__)

# ─── Paleta de colores del DNI español ───────────────────────────────────────

_COL_HEADER_BG      = (110, 20, 20)       # Burdeos oscuro (cabecera)
_COL_HEADER_ACCENT  = (160, 30, 30)       # Burdeos claro (detalle)
_COL_BODY_BG        = (252, 250, 244)     # Crema muy claro (fondo cuerpo)
_COL_MRZ_BG         = (228, 228, 228)     # Gris claro (zona MRZ)
_COL_EU_BLUE        = (0,   51,  153)     # Azul EU
_COL_EU_GOLD        = (255, 204,  0)      # Amarillo EU
_COL_PHOTO_BG       = (210, 220, 230)     # Azul grisáceo (fondo foto)
_COL_PHOTO_SKIN     = (220, 185, 155)     # Tono piel silueta
_COL_PHOTO_HAIR     = (80,  55,  35)      # Color pelo silueta
_COL_PHOTO_SHIRT    = (70,  90, 130)      # Color ropa silueta
_COL_SIG_BG         = (248, 248, 248)     # Blanco roto (fondo firma)
_COL_SIG_INK        = (15,  15,  80)      # Azul tinta (firma)
_COL_LABEL          = (100, 100, 100)     # Gris etiquetas
_COL_VALUE          = (5,   5,   5)       # Negro valores
_COL_VALUE_DNI      = (20,  20, 100)      # Azul oscuro para el número de DNI
_COL_BORDER         = (180, 180, 180)     # Gris claro bordes
_COL_GUILLOCHE      = (235, 230, 222)     # Patrón guilloche (barely visible)
_COL_GOLD           = (185, 140, 40)      # Dorado para el escudo
_COL_WHITE          = (255, 255, 255)
_COL_BLACK          = (0,   0,   0)

# ─── Layout (1518 × 957 px, _SCALE = 3 respecto al ID-1 a 150 DPI) ───────────

_SCALE = 3          # Factor de resolución respecto al ID-1 físico a 150 DPI.
                    # Imagen final: 1518×957 px — alta calidad para OCR y visualización.

_W = DNI_WIDTH_PX    # 1518 px
_H = DNI_HEIGHT_PX   # 957 px

# Todas las constantes de posición escalan con _SCALE
# (diseño base creado para _SCALE=2 → multiplicar por _SCALE/2 = 1.5)
_B = _SCALE / 2      # factor de escala respecto al diseño base

_HDR_H    = int(76   * _B)          # 114 px — cabecera burdeos
_MRZ_H    = int(106  * _B)          # 159 px — zona MRZ
_MRZ_Y    = _H - _MRZ_H             # 798 px — inicio zona MRZ
_LEFT_W   = int(310  * _B)          # 465 px — panel izquierdo (foto + firma)
_MARGIN   = int(16   * _B)          # 24 px — margen general
_EU_W     = int(24   * _B)          # 36 px — franja EU

# Foto
_PHOTO_X  = _EU_W + _MARGIN
_PHOTO_Y  = _HDR_H + _MARGIN
_PHOTO_W  = _LEFT_W - _EU_W - 2 * _MARGIN
_PHOTO_H  = int(352  * _B)          # 528 px — altura de la foto

# Firma
_SIG_X    = _PHOTO_X
_SIG_Y    = _PHOTO_Y + _PHOTO_H + _MARGIN
_SIG_W    = _PHOTO_W
_SIG_H    = int(70   * _B)          # 105 px — altura de la firma

# Panel de texto (derecha)
_TEXT_X   = _LEFT_W + _MARGIN
_TEXT_Y   = _HDR_H + _MARGIN
_TEXT_W   = _W - _LEFT_W - 2 * _MARGIN


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    """Carga una fuente TrueType; devuelve la fuente por defecto si no existe."""
    try:
        return ImageFont.truetype(str(path), size)
    except (IOError, OSError):
        return ImageFont.load_default()


# ─── Rutas de fuentes del sistema (Windows) ───────────────────────────────────

_FONTS_DIR = Path("C:/Windows/Fonts")
_F_REGULAR = _FONTS_DIR / "arial.ttf"
_F_BOLD    = _FONTS_DIR / "arialbd.ttf"
_F_NARROW  = _FONTS_DIR / "arialn.ttf"   # Arial Narrow
_F_MONO    = _FONTS_DIR / "cour.ttf"     # Courier New (MRZ)
_F_ITALIC  = _FONTS_DIR / "ariali.ttf"


class DNIGenerator:
    """
    Genera imágenes sintéticas del anverso del DNI español (3.ª generación).

    Cada llamada a `generar()` produce:
      - Un fichero PNG guardado en `output_dir`.
      - Un diccionario con las coordenadas de cada ROI (para anotaciones YOLO).

    Args:
        output_dir: Directorio donde se guardan las imágenes.
        seed: Semilla para la aleatoriedad (variaciones menores de foto/firma).
    """

    def __init__(
        self,
        output_dir: Path = RAW_DNIS_DIR,
        seed: int = RANDOM_SEED,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._rng = random.Random(seed)
        self._precargar_fuentes()
        logger.info("DNIGenerator inicializado — output_dir=%s", self._output_dir)

    def _precargar_fuentes(self) -> None:
        """Pre-carga todas las fuentes a los tamaños necesarios."""
        s = _SCALE
        self._fnt_label       = _font(_F_REGULAR, 13 * s)
        self._fnt_value       = _font(_F_BOLD,    15 * s)
        self._fnt_value_sm    = _font(_F_BOLD,    13 * s)
        self._fnt_dni_num     = _font(_F_BOLD,    28 * s)
        self._fnt_header      = _font(_F_BOLD,    11 * s)
        self._fnt_header_sm   = _font(_F_REGULAR,  9 * s)
        self._fnt_mrz         = _font(_F_MONO,    17 * s)
        self._fnt_mrz_label   = _font(_F_REGULAR,  8 * s)
        self._fnt_titulo      = _font(_F_BOLD,     9 * s)
        self._fnt_footer      = _font(_F_REGULAR,  8 * s)

    # ── API pública ───────────────────────────────────────────────────────────

    def generar(
        self,
        datos: DatosDNI,
        expediente_id: str,
        guardar: bool = True,
    ) -> tuple[Path, dict[str, tuple[int, int, int, int]]]:
        """
        Genera la imagen del DNI y devuelve la ruta y los ROIs.

        Args:
            datos: Objeto DatosDNI con todos los campos del documento.
            expediente_id: ID del expediente (usado para el nombre del fichero).
            guardar: Si True, guarda la imagen en disco.

        Returns:
            Tupla (ruta_imagen, rois) donde rois es un diccionario
            {nombre_clase: (x1, y1, x2, y2)} en coordenadas de la imagen guardada
            (1518 × 957 con _SCALE=3).
        """
        # Canvas de alta resolución
        img = Image.new("RGB", (_W, _H), _COL_BODY_BG)
        draw = ImageDraw.Draw(img)

        rois: dict[str, tuple[int, int, int, int]] = {}

        # Capas en orden de pintado
        self._dibujar_fondo_guilloche(draw)
        self._dibujar_cabecera(draw, datos)
        self._dibujar_franja_eu(draw)
        self._dibujar_foto(draw, datos.sexo, rois)
        self._dibujar_firma(draw, datos.nombre, datos.apellidos, rois)
        self._dibujar_campos_texto(draw, datos, rois)
        self._dibujar_zona_mrz(draw, datos, rois)
        self._dibujar_bordes_exteriores(draw)

        # Guardar a la resolución de generación completa (sin downscale)
        ruta = self._output_dir / f"{expediente_id}_dni.png"
        if guardar:
            img.save(str(ruta), "PNG", dpi=(150 * _SCALE, 150 * _SCALE))
            logger.debug("DNI guardado: %s (%dx%d)", ruta, _W, _H)

        return ruta, rois

    # ── Capas de dibujo ───────────────────────────────────────────────────────

    def _dibujar_fondo_guilloche(self, draw: ImageDraw.ImageDraw) -> None:
        """
        Dibuja un patrón de fondo guilloche: líneas sinusoidales superpuestas
        en colores muy suaves, simulando el papel de seguridad del DNI.
        """
        # Líneas horizontales sinusoidales
        for y_base in range(0, _H, 6):
            puntos = []
            for x in range(0, _W, 2):
                y_offset = int(3 * math.sin(x / 30 + y_base / 20))
                puntos.append((x, y_base + y_offset))
            if len(puntos) > 1:
                draw.line(puntos, fill=_COL_GUILLOCHE, width=1)

        # Segunda familia de líneas en diagonal
        col2 = (230, 225, 215)
        for x_base in range(-_H, _W, 18):
            draw.line(
                [(x_base, 0), (x_base + _H, _H)],
                fill=col2, width=1,
            )

    def _dibujar_cabecera(self, draw: ImageDraw.ImageDraw, datos: DatosDNI) -> None:
        """
        Dibuja la cabecera burdeos con:
          - Escudo de España simplificado (izquierda).
          - Texto "REINO DE ESPAÑA" / "DOCUMENTO NACIONAL DE IDENTIDAD".
          - Símbolo UE y número de DNI grande (derecha).
        """
        # Fondo burdeos
        draw.rectangle([0, 0, _W, _HDR_H], fill=_COL_HEADER_BG)

        # Línea de separación dorada
        draw.rectangle([0, _HDR_H - 4, _W, _HDR_H], fill=_COL_EU_GOLD)

        # ── Escudo simplificado ────────────────────────────────────────────
        ex, ey = int(18*_B), int(6*_B)
        self._dibujar_escudo(draw, ex, ey, ancho=int(58*_B), alto=int(62*_B))

        # ── Texto central ──────────────────────────────────────────────────
        cx = int(340 * _B)
        draw.text((cx, int(10*_B)), "REINO DE ESPAÑA", font=self._fnt_header,
                  fill=_COL_WHITE, anchor="mt")
        draw.text((cx, int(34*_B)), "DOCUMENTO NACIONAL DE IDENTIDAD",
                  font=self._fnt_header_sm, fill=(230, 210, 150), anchor="mt")

        # Número de DNI prominente en la cabecera (parte derecha)
        nx = _W - _MARGIN - int(10*_B)
        draw.text((nx, int(12*_B)), datos.numero_dni,
                  font=self._fnt_dni_num, fill=_COL_EU_GOLD, anchor="rt")

        # Etiqueta "DNI/NIF" pequeña encima del número
        draw.text((nx, int(4*_B)), "DNI / NIF",
                  font=self._fnt_titulo, fill=(210, 180, 100), anchor="rt")

        # Estrellas UE (esquina superior izquierda, zona estrecha)
        self._dibujar_circulo_estrellas(
            draw, int(82*_B), _HDR_H // 2, radio=int(22*_B), n=12
        )

    def _dibujar_escudo(
        self, draw: ImageDraw.ImageDraw,
        ox: int, oy: int,
        ancho: int = 58, alto: int = 60,
    ) -> None:
        """
        Dibuja el escudo de España simplificado (silueta heráldica con los
        cuarteles principales y la corona real) usando formas geométricas.
        """
        g = _COL_GOLD
        r = (180, 30, 30)    # rojo
        am = (230, 180, 0)   # amarillo

        # Silueta del escudo (forma de escudo heráldico)
        escudo_pts = [
            (ox,          oy),
            (ox + ancho,  oy),
            (ox + ancho,  oy + int(alto * 0.65)),
            (ox + ancho // 2, oy + alto),
            (ox,          oy + int(alto * 0.65)),
        ]
        draw.polygon(escudo_pts, fill=am, outline=g)

        # Cuarteles (4 cuadrantes simplificados)
        cx = ox + ancho // 2
        cy = oy + int(alto * 0.4)
        # Cuartel superior izquierdo (Castilla - rojo con castillo)
        draw.rectangle([ox + 2, oy + 2, cx - 1, cy], fill=r)
        # Cuartel superior derecho (León - blanco con león)
        draw.rectangle([cx + 1, oy + 2, ox + ancho - 2, cy], fill=_COL_WHITE)
        # Cuartel inferior izquierdo (Aragón - barras)
        for i in range(4):
            col_barra = r if i % 2 == 0 else am
            draw.rectangle([
                ox + 2,
                cy + i * (int(alto * 0.25) // 4),
                cx - 1,
                cy + (i + 1) * (int(alto * 0.25) // 4),
            ], fill=col_barra)
        # Cuartel inferior derecho (Navarra - cadenas en rojo)
        draw.rectangle([cx + 1, cy, ox + ancho - 2, oy + int(alto * 0.65)],
                       fill=r)

        # Castillo (cuartel CL) — simplificado
        torre_x = ox + 5
        draw.rectangle([torre_x, oy + 5, torre_x + 10, cy - 2], fill=am)
        draw.rectangle([torre_x - 2, oy + 5, torre_x + 12, oy + 10], fill=am)

        # León (cuartel LE) — simplificado con círculo
        draw.ellipse([cx + 4, oy + 5, cx + 18, cy - 2], fill=(220, 30, 30))

        # Granada (parte inferior central)
        draw.ellipse([cx - 8, oy + int(alto * 0.55),
                      cx + 8, oy + int(alto * 0.70)], fill=(200, 30, 50))

        # Contorno del escudo
        draw.polygon(escudo_pts, outline=g, fill=None)

        # Corona real (sobre el escudo)
        corona_y = oy - 14
        draw.arc([ox + 12, corona_y, ox + ancho - 12, oy + 2],
                 start=180, end=0, fill=g, width=4)
        # Puntas de la corona
        for px_c in [ox + 18, ox + ancho // 2, ox + ancho - 18]:
            draw.line([(px_c, corona_y), (px_c, corona_y - 8)], fill=g, width=3)
            draw.ellipse([px_c - 4, corona_y - 12,
                          px_c + 4, corona_y - 4], fill=g)

        # Columnas de Hércules (a los lados del escudo)
        for col_x in [ox - 8, ox + ancho + 2]:
            draw.rectangle([col_x, oy + 10, col_x + 6, oy + alto - 10],
                            fill=g, outline=(150, 110, 20))
            draw.rectangle([col_x - 2, oy + 8, col_x + 8, oy + 18], fill=g)
            draw.rectangle([col_x - 2, oy + alto - 18,
                             col_x + 8, oy + alto - 8], fill=g)

    def _dibujar_circulo_estrellas(
        self,
        draw: ImageDraw.ImageDraw,
        cx: int, cy: int,
        radio: int = 22,
        n: int = 12,
    ) -> None:
        """Dibuja el círculo de N estrellas de la bandera europea."""
        for i in range(n):
            angulo = 2 * math.pi * i / n - math.pi / 2
            sx = cx + int(radio * math.cos(angulo))
            sy = cy + int(radio * math.sin(angulo))
            self._dibujar_estrella(draw, sx, sy, r_ext=5, r_int=2, fill=_COL_EU_GOLD)

    def _dibujar_estrella(
        self,
        draw: ImageDraw.ImageDraw,
        cx: int, cy: int,
        r_ext: int = 5,
        r_int: int = 2,
        n: int = 5,
        fill: tuple = _COL_EU_GOLD,
    ) -> None:
        """Dibuja una estrella de N puntas centrada en (cx, cy)."""
        puntos = []
        for i in range(n * 2):
            angulo = math.pi * i / n - math.pi / 2
            r = r_ext if i % 2 == 0 else r_int
            puntos.append((
                cx + int(r * math.cos(angulo)),
                cy + int(r * math.sin(angulo)),
            ))
        draw.polygon(puntos, fill=fill)

    def _dibujar_franja_eu(self, draw: ImageDraw.ImageDraw) -> None:
        """
        Dibuja la franja lateral izquierda con colores EU (azul + estrellas doradas)
        característica del DNI español.
        """
        # Franja azul EU
        draw.rectangle([0, _HDR_H, _EU_W, _MRZ_Y], fill=_COL_EU_BLUE)

        # Estrella EU central en la franja
        self._dibujar_estrella(
            draw, _EU_W // 2, (_HDR_H + _MRZ_Y) // 2,
            r_ext=9, r_int=4, fill=_COL_EU_GOLD,
        )

        # Pequeñas estrellas decorativas
        paso = int(60 * _B)
        for y_s in [_HDR_H + paso, _HDR_H + paso*2, _MRZ_Y - paso*2, _MRZ_Y - paso]:
            self._dibujar_estrella(
                draw, _EU_W // 2, y_s,
                r_ext=int(4*_B), r_int=int(2*_B), fill=_COL_EU_GOLD,
            )

        # "ESP" vertical en la franja
        letras_esp = ["E", "S", "P"]
        for i, letra in enumerate(letras_esp):
            draw.text(
                (int(3*_B), _HDR_H + int(100*_B) + i * int(20*_B)),
                letra, font=self._fnt_titulo, fill=_COL_WHITE,
            )

    def _dibujar_foto(
        self,
        draw: ImageDraw.ImageDraw,
        sexo: str,
        rois: dict,
    ) -> None:
        """
        Dibuja el área de la foto con una silueta humana procedural.
        Registra las coordenadas del ROI 'foto'.
        """
        x1, y1 = _PHOTO_X, _PHOTO_Y
        x2, y2 = _PHOTO_X + _PHOTO_W, _PHOTO_Y + _PHOTO_H

        # Fondo del área de foto
        draw.rectangle([x1, y1, x2, y2], fill=_COL_PHOTO_BG, outline=_COL_BORDER, width=2)

        cx = (x1 + x2) // 2
        # Cabeza
        cabeza_r = int(46 * _B)
        cabeza_cx = cx
        cabeza_cy = y1 + int(95 * _B)
        draw.ellipse([
            cabeza_cx - cabeza_r, cabeza_cy - cabeza_r,
            cabeza_cx + cabeza_r, cabeza_cy + cabeza_r,
        ], fill=_COL_PHOTO_SKIN)

        # Pelo
        pelo_r = cabeza_r + int(6 * _B)
        if sexo == "F":
            draw.ellipse([
                cabeza_cx - pelo_r, cabeza_cy - pelo_r,
                cabeza_cx + pelo_r, cabeza_cy + int(20*_B),
            ], fill=_COL_PHOTO_HAIR)
            draw.ellipse([
                cabeza_cx - cabeza_r, cabeza_cy - cabeza_r,
                cabeza_cx + cabeza_r, cabeza_cy + cabeza_r,
            ], fill=_COL_PHOTO_SKIN)
        else:
            draw.arc([
                cabeza_cx - pelo_r, cabeza_cy - pelo_r,
                cabeza_cx + pelo_r, cabeza_cy - cabeza_r + int(10*_B),
            ], start=180, end=0, fill=_COL_PHOTO_HAIR, width=int(18*_B))

        # Rasgos faciales — Ojos
        ojo_sep = int(16 * _B)
        ojo_rx, ojo_ry = int(6*_B), int(6*_B)
        for ojo_x in [cabeza_cx - ojo_sep, cabeza_cx + ojo_sep]:
            draw.ellipse([ojo_x - ojo_rx, cabeza_cy - ojo_ry,
                          ojo_x + ojo_rx, cabeza_cy + ojo_ry], fill=_COL_WHITE)
            iris = int(3*_B)
            draw.ellipse([ojo_x - iris, cabeza_cy - iris,
                          ojo_x + iris, cabeza_cy + iris], fill=(40, 30, 20))
        # Nariz
        draw.polygon([
            (cabeza_cx, cabeza_cy + int(5*_B)),
            (cabeza_cx - int(6*_B), cabeza_cy + int(22*_B)),
            (cabeza_cx + int(6*_B), cabeza_cy + int(22*_B)),
        ], fill=(190, 155, 125))
        # Boca
        draw.arc([cabeza_cx - int(16*_B), cabeza_cy + int(20*_B),
                  cabeza_cx + int(16*_B), cabeza_cy + int(38*_B)],
                 start=0, end=180, fill=(160, 80, 80), width=int(3*_B))
        # Cejas
        for ceja_x in [cabeza_cx - int(22*_B), cabeza_cx + int(8*_B)]:
            draw.line([
                (ceja_x, cabeza_cy - int(18*_B)),
                (ceja_x + int(14*_B), cabeza_cy - int(22*_B)),
            ], fill=_COL_PHOTO_HAIR, width=int(4*_B))

        # Cuello
        cuello_w   = int(22 * _B)
        cuello_top = cabeza_cy + cabeza_r - int(10*_B)
        cuello_bot = cuello_top + int(45*_B)
        draw.rectangle([
            cabeza_cx - cuello_w // 2, cuello_top,
            cabeza_cx + cuello_w // 2, cuello_bot,
        ], fill=_COL_PHOTO_SKIN)

        # Hombros y torso
        hombro_y    = cuello_bot - int(10*_B)
        torso_ancho = int(150*_B) if sexo == "M" else int(130*_B)
        draw.polygon([
            (cabeza_cx - torso_ancho // 2, y2),
            (cabeza_cx + torso_ancho // 2, y2),
            (cabeza_cx + torso_ancho // 2 - int(10*_B), hombro_y),
            (cabeza_cx - torso_ancho // 2 + int(10*_B), hombro_y),
        ], fill=_COL_PHOTO_SHIRT)

        # Etiqueta de foto
        draw.text(
            (x1 + _PHOTO_W // 2, y2 + int(6*_B)),
            "FOTOGRAFÍA", font=self._fnt_titulo, fill=_COL_LABEL, anchor="mt",
        )

        rois["foto"] = (x1, y1, x2, y2)

    def _dibujar_firma(
        self,
        draw: ImageDraw.ImageDraw,
        nombre: str,
        apellidos: str,
        rois: dict,
    ) -> None:
        """
        Dibuja el área de firma con trazos Bézier procedurales que simulan
        una firma manuscrita. Registra el ROI 'firma'.
        """
        x1, y1 = _SIG_X, _SIG_Y
        x2, y2 = _SIG_X + _SIG_W, _SIG_Y + _SIG_H

        draw.rectangle([x1, y1, x2, y2], fill=_COL_SIG_BG, outline=_COL_BORDER, width=1)

        # Línea base de la firma
        draw.line([
            (x1 + 8, y2 - 12),
            (x2 - 8, y2 - 12),
        ], fill=_COL_BORDER, width=1)

        # Trazos de la firma (curvas Bézier cúbicas aproximadas como polilíneas)
        cx_sig = (x1 + x2) // 2
        cy_sig = (y1 + y2) // 2

        self._trazar_firma(draw, x1 + 10, y1 + 8, x2 - 10, y2 - 16, nombre, apellidos)

        # Etiqueta
        draw.text(
            (x1 + _SIG_W // 2, y2 + 6),
            "FIRMA DEL TITULAR",
            font=self._fnt_titulo,
            fill=_COL_LABEL,
            anchor="mt",
        )

        rois["firma"] = (x1, y1, x2, y2)

    def _trazar_firma(
        self,
        draw: ImageDraw.ImageDraw,
        x1: int, y1: int,
        x2: int, y2: int,
        nombre: str,
        apellidos: str,
    ) -> None:
        """
        Genera trazos que simulan una firma manuscrita usando
        curvas sinusoidales compuestas con parámetros derivados del nombre.
        """
        # Semilla derivada del nombre para que la misma persona siempre tenga la misma firma
        seed_val = sum(ord(c) for c in nombre + apellidos)
        rng = random.Random(seed_val)

        w = x2 - x1
        h = y2 - y1
        cy = (y1 + y2) // 2
        n_trazos = rng.randint(3, 5)

        for trazo in range(n_trazos):
            puntos = []
            # Amplitud y frecuencia variables por trazo
            amp   = rng.uniform(h * 0.15, h * 0.45)
            freq  = rng.uniform(1.5, 4.0)
            fase  = rng.uniform(0, 2 * math.pi)
            x_ini = x1 + trazo * w // (n_trazos + 1)
            x_fin = x_ini + rng.randint(w // 6, w // 3)
            x_fin = min(x_fin, x2 - 4)

            y_off = rng.uniform(-h * 0.2, h * 0.2)

            for x in range(x_ini, x_fin, 2):
                t = (x - x_ini) / max(1, (x_fin - x_ini))
                # Curva sinusoidal con decaimiento
                y_val = cy + y_off + amp * math.sin(freq * t * math.pi + fase) * (1 - t * 0.4)
                puntos.append((x, int(y_val)))

            if len(puntos) > 3:
                draw.line(puntos, fill=_COL_SIG_INK, width=rng.randint(2, 3))

        # Rúbrica final (trazo largo horizontal)
        rubrica_y = y2 - 18
        puntos_rubrica = []
        for x in range(x1 + 5, x2 - 5, 2):
            t = (x - x1) / w
            y_r = rubrica_y + int(4 * math.sin(t * 5 * math.pi)) - int(t * 8)
            puntos_rubrica.append((x, y_r))
        if puntos_rubrica:
            draw.line(puntos_rubrica, fill=_COL_SIG_INK, width=2)

    def _dibujar_campos_texto(
        self,
        draw: ImageDraw.ImageDraw,
        datos: DatosDNI,
        rois: dict,
    ) -> None:
        """
        Dibuja todos los campos de texto del DNI con sus etiquetas y valores.
        Registra los ROIs de cada campo de texto.
        """
        x = _TEXT_X
        max_x = _W - _MARGIN

        # Espaciados escalados
        LBL_H  = int(20 * _B)   # altura de la etiqueta
        VAL_H  = int(28 * _B)   # altura del valor
        SEP_H  = int(12 * _B)   # separación tras la línea divisoria
        ROW_H  = LBL_H + VAL_H + SEP_H   # altura total de una fila

        def campo(
            label: str,
            valor: str,
            y: int,
            roi_key: Optional[str] = None,
            fnt_val: Optional[ImageFont.FreeTypeFont] = None,
            col_val: tuple = _COL_VALUE,
        ) -> int:
            """Dibuja un campo (etiqueta + valor) y devuelve la Y del siguiente."""
            fnt_v = fnt_val or self._fnt_value
            draw.text((x, y), label, font=self._fnt_label, fill=_COL_LABEL)
            y_val = y + LBL_H
            draw.text((x, y_val), valor, font=fnt_v, fill=col_val)
            y_sep = y_val + VAL_H
            draw.line([(x, y_sep), (max_x - 4, y_sep)], fill=_COL_BORDER, width=1)
            if roi_key:
                try:
                    bb = draw.textbbox((x, y_val), valor, font=fnt_v)
                    rois[roi_key] = (bb[0]-2, bb[1]-2, min(bb[2]+2, max_x), bb[3]+2)
                except Exception:
                    rois[roi_key] = (x, y_val, max_x, y_sep)
            return y_sep + SEP_H

        # ── APELLIDOS ─────────────────────────────────────────────────────────
        y = _TEXT_Y
        y = campo("APELLIDOS", datos.apellidos, y, roi_key="apellidos")

        # ── NOMBRE ────────────────────────────────────────────────────────────
        y = campo("NOMBRE", datos.nombre, y, roi_key="nombre")

        # ── SEXO | NACIONALIDAD | FECHA NACIMIENTO ────────────────────────────
        ancho_total = _W - x - _MARGIN
        col_sexo = int(ancho_total * 0.12)
        col_nac  = int(ancho_total * 0.38)

        # SEXO
        draw.text((x, y), "SEXO", font=self._fnt_label, fill=_COL_LABEL)
        draw.text((x, y + LBL_H), datos.sexo, font=self._fnt_value_sm, fill=_COL_VALUE)

        # NACIONALIDAD
        x_nac = x + col_sexo + int(10 * _B)
        draw.text((x_nac, y), "NACIONALIDAD", font=self._fnt_label, fill=_COL_LABEL)
        draw.text((x_nac, y + LBL_H), datos.nacionalidad,
                  font=self._fnt_value_sm, fill=_COL_VALUE)
        try:
            bb = draw.textbbox((x_nac, y + LBL_H), datos.nacionalidad, font=self._fnt_value_sm)
            rois["nacionalidad"] = (bb[0]-2, bb[1]-2, bb[2]+2, bb[3]+2)
        except Exception:
            rois["nacionalidad"] = (x_nac, y + LBL_H, x_nac + col_nac, y + ROW_H)

        # FECHA NACIMIENTO
        x_fnac = x_nac + col_nac
        draw.text((x_fnac, y), "FECHA DE NACIMIENTO", font=self._fnt_label, fill=_COL_LABEL)
        draw.text((x_fnac, y + LBL_H), datos.fecha_nacimiento,
                  font=self._fnt_value_sm, fill=_COL_VALUE)
        try:
            bb = draw.textbbox((x_fnac, y + LBL_H), datos.fecha_nacimiento,
                               font=self._fnt_value_sm)
            rois["fecha_nacimiento"] = (bb[0]-2, bb[1]-2, bb[2]+2, bb[3]+2)
        except Exception:
            rois["fecha_nacimiento"] = (x_fnac, y + LBL_H, _W - _MARGIN, y + ROW_H)

        y += ROW_H
        draw.line([(x, y), (max_x - 4, y)], fill=_COL_BORDER, width=1)
        y += SEP_H

        # ── LUGAR DE NACIMIENTO ───────────────────────────────────────────────
        draw.text((x, y), "LUGAR DE NACIMIENTO", font=self._fnt_label, fill=_COL_LABEL)
        draw.text((x, y + LBL_H), datos.lugar_nacimiento,
                  font=self._fnt_value_sm, fill=_COL_VALUE)
        y += ROW_H
        draw.line([(x, y), (max_x - 4, y)], fill=_COL_BORDER, width=1)
        y += SEP_H

        # ── FECHA DE CADUCIDAD | NÚMERO SOPORTE ──────────────────────────────
        draw.text((x, y), "VALIDEZ", font=self._fnt_label, fill=_COL_LABEL)
        y_vcad = y + LBL_H
        draw.text((x, y_vcad), datos.fecha_caducidad, font=self._fnt_value, fill=_COL_VALUE)
        try:
            bb = draw.textbbox((x, y_vcad), datos.fecha_caducidad, font=self._fnt_value)
            rois["fecha_caducidad"] = (bb[0]-2, bb[1]-2, bb[2]+2, bb[3]+2)
        except Exception:
            rois["fecha_caducidad"] = (x, y_vcad, x + int(200*_B), y_vcad + VAL_H)

        x_ns = x + int(300 * _B)
        draw.text((x_ns, y), "NÚMERO SOPORTE", font=self._fnt_label, fill=_COL_LABEL)
        draw.text((x_ns, y + LBL_H), datos.numero_soporte,
                  font=self._fnt_value_sm, fill=_COL_LABEL)

        y += ROW_H
        draw.line([(x, y), (max_x - 4, y)], fill=_COL_BORDER, width=1)
        y += SEP_H

        # ── NÚMERO DNI PROMINENTE ─────────────────────────────────────────────
        box_h = int(70 * _B)
        draw.rectangle([x - 4, y, max_x, y + box_h],
                       fill=(245, 243, 235), outline=_COL_BORDER, width=2)
        draw.text((x + 6, y + 4),
                  "Número de documento / Document number",
                  font=self._fnt_titulo, fill=_COL_LABEL)
        y_dni_val = y + int(18 * _B)
        draw.text((x + 6, y_dni_val), datos.numero_dni,
                  font=self._fnt_dni_num, fill=_COL_VALUE_DNI)
        try:
            bb = draw.textbbox((x + 6, y_dni_val), datos.numero_dni, font=self._fnt_dni_num)
            rois["numero_dni"] = (bb[0]-2, bb[1]-2, bb[2]+2, bb[3]+2)
        except Exception:
            rois["numero_dni"] = (x + 6, y_dni_val, x + int(280*_B), y_dni_val + int(40*_B))

    def _dibujar_zona_mrz(
        self,
        draw: ImageDraw.ImageDraw,
        datos: DatosDNI,
        rois: dict,
    ) -> None:
        """
        Dibuja la zona MRZ (Machine Readable Zone) en la parte inferior
        con fuente monoespaciada. Registra el ROI 'mrz_line'.
        """
        # Fondo MRZ
        draw.rectangle([0, _MRZ_Y, _W, _H], fill=_COL_MRZ_BG)

        # Línea separadora
        draw.line([(0, _MRZ_Y), (_W, _MRZ_Y)], fill=_COL_BORDER, width=2)

        # Etiqueta de zona
        draw.text(
            (_W // 2, _MRZ_Y + int(6*_B)),
            "<<  ZONA DE LECTURA AUTOMÁTICA  /  MACHINE READABLE ZONE  >>",
            font=self._fnt_mrz_label,
            fill=_COL_LABEL,
            anchor="mt",
        )

        # Líneas MRZ
        mrz_y1 = _MRZ_Y + int(26*_B)
        mrz_y2 = mrz_y1 + int(44*_B)

        # Fondo blanco de las líneas MRZ
        draw.rectangle([_MARGIN, mrz_y1 - int(4*_B),
                        _W - _MARGIN, mrz_y2 + int(44*_B)],
                       fill=_COL_WHITE, outline=_COL_BORDER, width=2)

        # Texto MRZ línea 1
        draw.text((_MARGIN + int(6*_B), mrz_y1),
                  datos.mrz_linea_1, font=self._fnt_mrz, fill=_COL_BLACK)

        # Texto MRZ línea 2
        draw.text((_MARGIN + int(6*_B), mrz_y2),
                  datos.mrz_linea_2, font=self._fnt_mrz, fill=_COL_BLACK)

        try:
            off = int(6*_B)
            bb1 = draw.textbbox((_MARGIN + off, mrz_y1), datos.mrz_linea_1, font=self._fnt_mrz)
            bb2 = draw.textbbox((_MARGIN + off, mrz_y2), datos.mrz_linea_2, font=self._fnt_mrz)
            rois["mrz_line"] = (
                min(bb1[0], bb2[0]) - int(4*_B),
                bb1[1] - int(4*_B),
                max(bb1[2], bb2[2]) + int(4*_B),
                bb2[3] + int(4*_B),
            )
        except Exception:
            rois["mrz_line"] = (_MARGIN, mrz_y1, _W - _MARGIN, mrz_y2 + int(28*_B))

    def _dibujar_bordes_exteriores(self, draw: ImageDraw.ImageDraw) -> None:
        """Dibuja el borde exterior de la tarjeta con esquinas redondeadas simuladas."""
        # Borde principal
        draw.rectangle([0, 0, _W - 1, _H - 1],
                       outline=(120, 100, 80), width=4)
        # Borde interior dorado sutil
        draw.rectangle([4, 4, _W - 5, _H - 5],
                       outline=(200, 170, 80), width=1)
