"""
loan_form_generator.py — Generador de imágenes sintéticas del formulario de préstamo.

Produce imágenes PNG de un formulario bancario de solicitud de préstamo personal
(formato A4, diseño corporativo español) a partir de un objeto DatosFormulario.

El formulario está estructurado en tres secciones:
  1. Cabecera corporativa con logotipo del banco sintético.
  2. Datos del solicitante (8 campos personales y laborales).
  3. Datos del préstamo solicitado (5 campos financieros + cuadro resumen).
  4. Pie de página con información legal y fecha.

Cada campo generado tiene una posición fija que permite al detector YOLOv8
aprender a localizarlos de manera consistente.

Uso típico:
    from src.data_generation.loan_form_generator import LoanFormGenerator
    gen = LoanFormGenerator(output_dir=Path("data/raw/prestamos"), seed=42)
    ruta, rois = gen.generar(datos_formulario, expediente_id="EXP00001")
"""

import math
import random
import logging
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from src.config import (
    LOAN_WIDTH_PX,
    LOAN_HEIGHT_PX,
    RAW_LOANS_DIR,
    RANDOM_SEED,
)
from src.data_generation.fake_data_factory import DatosFormulario

logger = logging.getLogger(__name__)

# ─── Paleta corporativa del banco sintético ───────────────────────────────────

_COL_BANK_PRIMARY   = (0,  60, 120)      # Azul corporativo (cabecera)
_COL_BANK_SECONDARY = (0,  90, 170)      # Azul medio (subencabezados)
_COL_BANK_ACCENT    = (230, 165,  0)     # Amarillo dorado (acentos)
_COL_BODY_BG        = (255, 255, 255)    # Blanco (fondo del formulario)
_COL_SECTION_BG     = (240, 245, 252)    # Azul muy claro (fondo secciones)
_COL_FIELD_BG       = (252, 252, 254)    # Casi blanco (fondo casillas)
_COL_FIELD_BORDER   = (180, 195, 215)    # Azul grisáceo (bordes casillas)
_COL_LABEL          = (60,  80, 110)     # Azul oscuro (etiquetas)
_COL_VALUE          = (10,  10,  10)     # Negro (valores rellenados)
_COL_HEADER_TEXT    = (255, 255, 255)    # Blanco (texto cabecera)
_COL_SECTION_TITLE  = (0,  50, 110)      # Azul oscuro (títulos sección)
_COL_LEGAL          = (130, 130, 130)    # Gris claro (texto legal)
_COL_HIGHLIGHT      = (255, 248, 220)    # Amarillo pálido (cuadro resumen)
_COL_HIGHLIGHT_BDR  = (200, 160,  20)    # Borde cuadro resumen
_COL_LINE           = (200, 210, 225)    # Azul pálido (líneas separadoras)
_COL_WHITE          = (255, 255, 255)    # Blanco puro

# ─── Escala y dimensiones ─────────────────────────────────────────────────────

_SCALE = 2                               # Sobremuestreo × 2 para mejor OCR
_W     = LOAN_WIDTH_PX  * _SCALE        # 1588 px
_H     = LOAN_HEIGHT_PX * _SCALE        # 2246 px
_M     = 60                             # Margen lateral (px escalados)

# ─── Fuentes del sistema (Windows) ────────────────────────────────────────────

_FONTS_DIR = Path("C:/Windows/Fonts")
_F_REGULAR = _FONTS_DIR / "arial.ttf"
_F_BOLD    = _FONTS_DIR / "arialbd.ttf"
_F_ITALIC  = _FONTS_DIR / "ariali.ttf"
_F_NARROW  = _FONTS_DIR / "arialn.ttf"
_F_MONO    = _FONTS_DIR / "cour.ttf"


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    """Carga fuente TrueType; fallback a la fuente por defecto si no existe."""
    try:
        return ImageFont.truetype(str(path), size)
    except (IOError, OSError):
        return ImageFont.load_default()


class LoanFormGenerator:
    """
    Genera imágenes sintéticas del formulario de solicitud de préstamo personal.

    El formulario imita el diseño de los formularios bancarios españoles reales,
    con campos identificables por posición constante en todas las imágenes
    generadas (necesario para entrenar el detector YOLOv8).

    Args:
        output_dir: Directorio donde se guardan las imágenes.
        seed: Semilla para la aleatoriedad menor (variaciones de relleno manual).
    """

    def __init__(
        self,
        output_dir: Path = RAW_LOANS_DIR,
        seed: int = RANDOM_SEED,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._rng = random.Random(seed)
        self._precargar_fuentes()
        logger.info("LoanFormGenerator inicializado — output_dir=%s", self._output_dir)

    def _precargar_fuentes(self) -> None:
        s = _SCALE
        self._fnt_bank_name   = _font(_F_BOLD,    22 * s)
        self._fnt_bank_slogan = _font(_F_ITALIC,  10 * s)
        self._fnt_form_title  = _font(_F_BOLD,    18 * s)
        self._fnt_form_ref    = _font(_F_REGULAR,  9 * s)
        self._fnt_section     = _font(_F_BOLD,    12 * s)
        self._fnt_label       = _font(_F_REGULAR, 10 * s)
        self._fnt_label_sm    = _font(_F_REGULAR,  9 * s)
        self._fnt_value       = _font(_F_BOLD,    12 * s)
        self._fnt_value_sm    = _font(_F_BOLD,    10 * s)
        self._fnt_legal       = _font(_F_REGULAR,  8 * s)
        self._fnt_footer      = _font(_F_REGULAR,  9 * s)
        self._fnt_highlight   = _font(_F_BOLD,    13 * s)
        self._fnt_checkbox    = _font(_F_BOLD,    12 * s)

    # ── API pública ───────────────────────────────────────────────────────────

    def generar(
        self,
        datos: DatosFormulario,
        expediente_id: str,
        guardar: bool = True,
    ) -> tuple[Path, dict[str, tuple[int, int, int, int]]]:
        """
        Genera la imagen del formulario y devuelve la ruta y los ROIs.

        Args:
            datos: Objeto DatosFormulario con todos los campos.
            expediente_id: ID del expediente (nombre del fichero).
            guardar: Si True, guarda la imagen en disco.

        Returns:
            Tupla (ruta_imagen, rois) donde rois es un dict
            {nombre_clase: (x1, y1, x2, y2)} en coordenadas finales (794×1123).
        """
        img  = Image.new("RGB", (_W, _H), _COL_BODY_BG)
        draw = ImageDraw.Draw(img)
        rois: dict[str, tuple[int, int, int, int]] = {}

        self._dibujar_fondo_sutil(draw)
        self._dibujar_cabecera(draw, expediente_id)
        y = self._dibujar_titulo_formulario(draw)
        y = self._dibujar_seccion_solicitante(draw, datos, y, rois)
        y = self._dibujar_seccion_laboral(draw, datos, y, rois)
        y = self._dibujar_seccion_prestamo(draw, datos, y, rois)
        y = self._dibujar_cuadro_resumen(draw, datos, y, rois)
        self._dibujar_declaracion_firma(draw, y)
        self._dibujar_pie(draw)

        # Escalar a resolución final
        img_final = img.resize((LOAN_WIDTH_PX, LOAN_HEIGHT_PX), Image.LANCZOS)

        rois_scaled = {
            k: (int(x1/_SCALE), int(y1/_SCALE), int(x2/_SCALE), int(y2/_SCALE))
            for k, (x1, y1, x2, y2) in rois.items()
        }

        ruta = self._output_dir / f"{expediente_id}_prestamo.png"
        if guardar:
            img_final.save(str(ruta), "PNG", dpi=(96, 96))
            logger.debug("Formulario guardado: %s", ruta)

        return ruta, rois_scaled

    # ── Capas de dibujo ───────────────────────────────────────────────────────

    def _dibujar_fondo_sutil(self, draw: ImageDraw.ImageDraw) -> None:
        """Fondo blanco con micropatrón de puntos muy tenue (papel corporativo)."""
        for y in range(0, _H, 20):
            for x in range(0, _W, 20):
                draw.ellipse([x, y, x + 1, y + 1], fill=(240, 242, 246))

    def _dibujar_cabecera(self, draw: ImageDraw.ImageDraw, exp_id: str) -> None:
        """
        Dibuja la cabecera corporativa del banco:
        - Barra azul con logotipo y nombre del banco (izquierda).
        - Código de expediente y fecha (derecha).
        - Banda dorada decorativa.
        """
        cab_h = 140

        # Fondo cabecera
        draw.rectangle([0, 0, _W, cab_h], fill=_COL_BANK_PRIMARY)

        # Banda dorada inferior de la cabecera
        draw.rectangle([0, cab_h - 8, _W, cab_h], fill=_COL_BANK_ACCENT)

        # ── Logotipo simplificado (rombo + texto) ─────────────────────────────
        lx, ly = _M, 20
        # Rombo
        rombo = [
            (lx + 30, ly),
            (lx + 60, ly + 30),
            (lx + 30, ly + 60),
            (lx,      ly + 30),
        ]
        draw.polygon(rombo, fill=_COL_BANK_ACCENT)
        # Rombo interior más pequeño
        rombo_int = [
            (lx + 30, ly + 10),
            (lx + 50, ly + 30),
            (lx + 30, ly + 50),
            (lx + 10, ly + 30),
        ]
        draw.polygon(rombo_int, fill=_COL_BANK_PRIMARY)

        # Nombre del banco
        draw.text(
            (lx + 74, ly + 8),
            "BANCO DIGITAL ESPAÑOL",
            font=self._fnt_bank_name,
            fill=_COL_HEADER_TEXT,
        )
        draw.text(
            (lx + 74, ly + 46),
            "Tu banco de confianza · www.bdespanol.es",
            font=self._fnt_bank_slogan,
            fill=(190, 215, 245),
        )

        # ── Información de expediente (derecha) ───────────────────────────────
        from datetime import date
        hoy = date.today().strftime("%d/%m/%Y")

        draw.text(
            (_W - _M, ly + 6),
            f"Ref. Expediente: {exp_id}",
            font=self._fnt_form_ref,
            fill=(210, 225, 245),
            anchor="rt",
        )
        draw.text(
            (_W - _M, ly + 24),
            f"Fecha: {hoy}",
            font=self._fnt_form_ref,
            fill=(210, 225, 245),
            anchor="rt",
        )
        draw.text(
            (_W - _M, ly + 42),
            "Oficina: Gestión Digital",
            font=self._fnt_form_ref,
            fill=(180, 205, 235),
            anchor="rt",
        )

        # Número de página
        draw.text(
            (_W - _M, cab_h - 28),
            "Página 1 de 1  |  FORMULARIO F-PRE-001 v4.2",
            font=self._fnt_form_ref,
            fill=(170, 195, 225),
            anchor="rt",
        )

    def _dibujar_titulo_formulario(self, draw: ImageDraw.ImageDraw) -> int:
        """Dibuja el título principal del formulario. Devuelve la Y siguiente."""
        y = 160
        draw.text(
            (_W // 2, y),
            "SOLICITUD DE PRÉSTAMO PERSONAL",
            font=self._fnt_form_title,
            fill=_COL_SECTION_TITLE,
            anchor="mt",
        )
        draw.line([_M, y + 38, _W - _M, y + 38], fill=_COL_BANK_ACCENT, width=3)
        draw.text(
            (_W // 2, y + 48),
            "Cumplimente todos los campos. Los campos marcados con (*) son obligatorios.",
            font=self._fnt_label_sm,
            fill=_COL_LEGAL,
            anchor="mt",
        )
        return y + 74

    # ── Secciones del formulario ──────────────────────────────────────────────

    def _dibujar_titulo_seccion(
        self,
        draw: ImageDraw.ImageDraw,
        titulo: str,
        y: int,
        icono: str = "■",
    ) -> int:
        """Dibuja el encabezado de una sección. Devuelve la Y siguiente."""
        # Fondo del título de sección
        draw.rectangle([_M, y, _W - _M, y + 44], fill=_COL_BANK_SECONDARY)
        draw.text(
            (_M + 16, y + 12),
            f"{icono}  {titulo}",
            font=self._fnt_section,
            fill=_COL_HEADER_TEXT,
        )
        return y + 52

    def _campo(
        self,
        draw: ImageDraw.ImageDraw,
        label: str,
        valor: str,
        x: int, y: int,
        ancho: int,
        alto: int = 64,
        roi_key: Optional[str] = None,
        rois: Optional[dict] = None,
        obligatorio: bool = True,
    ) -> None:
        """
        Dibuja una casilla de formulario con etiqueta y valor.

        Args:
            label: Etiqueta del campo.
            valor: Valor rellenado.
            x, y: Esquina superior izquierda.
            ancho: Anchura de la casilla.
            alto: Altura de la casilla.
            roi_key: Clave YOLO para registrar en rois.
            rois: Diccionario de ROIs donde registrar.
            obligatorio: Si True, añade (*) a la etiqueta.
        """
        # Contorno de la casilla
        draw.rectangle(
            [x, y, x + ancho, y + alto],
            fill=_COL_FIELD_BG,
            outline=_COL_FIELD_BORDER,
            width=2,
        )

        # Etiqueta interior (esquina superior)
        label_txt = f"{label} {'(*)' if obligatorio else ''}"
        draw.text(
            (x + 8, y + 6),
            label_txt,
            font=self._fnt_label_sm,
            fill=_COL_LABEL,
        )

        # Línea sutil bajo la etiqueta
        draw.line([x + 6, y + 24, x + ancho - 6, y + 24],
                  fill=_COL_LINE, width=1)

        # Valor del campo
        y_val = y + 30
        # Truncar valor si es demasiado largo para el ancho
        valor_display = valor
        draw.text(
            (x + 8, y_val),
            valor_display,
            font=self._fnt_value_sm,
            fill=_COL_VALUE,
        )

        # Registrar ROI del área del valor
        if roi_key and rois is not None:
            try:
                bb = draw.textbbox((x + 8, y_val), valor_display, font=self._fnt_value_sm)
                # El ROI cubre toda la zona del valor (no solo el texto)
                rois[roi_key] = (
                    x + 4,
                    y_val - 2,
                    x + ancho - 4,
                    y + alto - 4,
                )
            except Exception:
                rois[roi_key] = (x + 4, y_val, x + ancho - 4, y + alto - 4)

    def _dibujar_seccion_solicitante(
        self,
        draw: ImageDraw.ImageDraw,
        datos: DatosFormulario,
        y: int,
        rois: dict,
    ) -> int:
        """
        Sección 1: Datos personales del solicitante.
        Campos: nombre, apellidos, NIF, fecha_nacimiento, domicilio, municipio,
                teléfono, email.
        """
        y = self._dibujar_titulo_seccion(draw, "1. DATOS PERSONALES DEL SOLICITANTE", y, "👤")

        pad = 8           # padding entre casillas
        col2 = (_W - 2 * _M - pad) // 2    # ancho de 2 columnas
        col3 = (_W - 2 * _M - 2 * pad) // 3  # ancho de 3 columnas
        full = _W - 2 * _M                  # ancho completo

        # Fila 1: APELLIDOS (full)
        self._campo(draw, "Apellidos", datos.sol_apellidos,
                    _M, y, full, roi_key="sol_apellidos", rois=rois)
        y += 68 + pad

        # Fila 2: NOMBRE (2/3) | FECHA NAC (1/3)
        ancho_nombre = int(full * 0.62)
        ancho_fn = full - ancho_nombre - pad
        self._campo(draw, "Nombre / First name", datos.sol_nombre,
                    _M, y, ancho_nombre, roi_key="sol_nombre", rois=rois)
        self._campo(draw, "Fecha de nacimiento", datos.sol_fecha_nacimiento,
                    _M + ancho_nombre + pad, y, ancho_fn,
                    roi_key="sol_fecha_nacimiento", rois=rois)
        y += 68 + pad

        # Fila 3: NIF/DNI (1/3) | TELÉFONO (1/3) | EMAIL (1/3)
        self._campo(draw, "NIF / DNI (*)", datos.sol_nif,
                    _M, y, col3, roi_key="sol_nif", rois=rois)
        self._campo(draw, "Teléfono de contacto", datos.sol_telefono,
                    _M + col3 + pad, y, col3, roi_key="sol_telefono", rois=rois)
        self._campo(draw, "Correo electrónico", datos.sol_email,
                    _M + 2 * (col3 + pad), y, col3, roi_key="sol_email", rois=rois)
        y += 68 + pad

        # Fila 4: DOMICILIO (full)
        self._campo(draw, "Domicilio (Calle, número, piso)", datos.sol_domicilio,
                    _M, y, full, roi_key="sol_domicilio", rois=rois)
        y += 68 + pad

        # Fila 5: MUNICIPIO (1/2) | PROVINCIA (1/4) | CP (1/4)
        ancho_mun = int(full * 0.48)
        ancho_prov = int(full * 0.28)
        ancho_cp = full - ancho_mun - ancho_prov - 2 * pad
        self._campo(draw, "Municipio", datos.sol_municipio,
                    _M, y, ancho_mun, obligatorio=False)
        self._campo(draw, "Provincia", datos.sol_provincia,
                    _M + ancho_mun + pad, y, ancho_prov, obligatorio=False)
        self._campo(draw, "Código postal", datos.sol_codigo_postal,
                    _M + ancho_mun + ancho_prov + 2 * pad, y, ancho_cp, obligatorio=False)
        y += 68 + pad + 10

        return y

    def _dibujar_seccion_laboral(
        self,
        draw: ImageDraw.ImageDraw,
        datos: DatosFormulario,
        y: int,
        rois: dict,
    ) -> int:
        """
        Sección 2: Situación laboral e ingresos.
        Campos: situacion_laboral, empresa, ingresos_netos.
        """
        y = self._dibujar_titulo_seccion(draw, "2. SITUACIÓN LABORAL E INGRESOS", y, "💼")

        pad  = 8
        full = _W - 2 * _M
        col2 = (full - pad) // 2

        # Fila 1: SITUACIÓN LABORAL (con checkboxes visuales)
        y = self._dibujar_checkboxes_situacion(draw, datos.sol_situacion_laboral, y, rois)
        y += pad

        # Fila 2: EMPRESA (2/3) | INGRESOS (1/3)
        ancho_empresa = int(full * 0.62)
        ancho_ing = full - ancho_empresa - pad
        self._campo(draw, "Empresa / Empleador actual", datos.sol_empresa,
                    _M, y, ancho_empresa, roi_key="sol_empresa", rois=rois)
        self._campo(
            draw, "Ingresos netos mensuales (€)",
            f"{datos.sol_ingresos_netos:,.2f} €",
            _M + ancho_empresa + pad, y, ancho_ing,
            roi_key="sol_ingresos_netos", rois=rois,
        )
        y += 68 + pad + 10

        return y

    def _dibujar_checkboxes_situacion(
        self,
        draw: ImageDraw.ImageDraw,
        situacion_actual: str,
        y: int,
        rois: dict,
    ) -> int:
        """
        Dibuja los checkboxes de situación laboral con la opción correcta marcada.
        """
        opciones = [
            ("Empleado indefinido",   "indefinido"),
            ("Empleado temporal",     "temporal"),
            ("Autónomo",              "autónomo"),
            ("Funcionario",           "funcionario"),
            ("Pensionista",           "pensionista"),
        ]

        # Fondo contenedor
        draw.rectangle(
            [_M, y, _W - _M, y + 56],
            fill=_COL_FIELD_BG,
            outline=_COL_FIELD_BORDER,
            width=2,
        )
        draw.text((_M + 8, y + 6), "Situación laboral (*)",
                  font=self._fnt_label_sm, fill=_COL_LABEL)
        draw.line([_M + 6, y + 24, _W - _M - 6, y + 24], fill=_COL_LINE, width=1)

        item_w = (_W - 2 * _M) // len(opciones)
        roi_y1 = y
        roi_x1 = _M
        roi_x2 = _W - _M
        roi_y2 = y + 56

        for i, (label, key) in enumerate(opciones):
            ix = _M + i * item_w + 10
            iy = y + 30
            marcado = key in situacion_actual.lower()

            # Checkbox cuadrado
            draw.rectangle([ix, iy, ix + 18, iy + 18],
                           fill=_COL_WHITE if not marcado else _COL_BANK_PRIMARY,
                           outline=_COL_BANK_SECONDARY, width=2)
            if marcado:
                # Marca de verificación (✓)
                draw.line([(ix + 3, iy + 9), (ix + 8, iy + 14)],
                          fill=_COL_HEADER_TEXT, width=3)
                draw.line([(ix + 8, iy + 14), (ix + 16, iy + 4)],
                          fill=_COL_HEADER_TEXT, width=3)
            draw.text((ix + 22, iy + 1), label,
                      font=self._fnt_label_sm, fill=_COL_VALUE)

        # ROI: zona completa de situación laboral
        rois["sol_situacion_laboral"] = (roi_x1, roi_y1, roi_x2, roi_y2)

        return y + 56

    def _dibujar_seccion_prestamo(
        self,
        draw: ImageDraw.ImageDraw,
        datos: DatosFormulario,
        y: int,
        rois: dict,
    ) -> int:
        """
        Sección 3: Datos del préstamo solicitado.
        Campos: importe, plazo, finalidad, cuota (calculada), TAE.
        """
        y = self._dibujar_titulo_seccion(draw, "3. DATOS DEL PRÉSTAMO SOLICITADO", y, "📋")

        pad  = 8
        full = _W - 2 * _M
        col3 = (full - 2 * pad) // 3

        # Fila 1: IMPORTE | PLAZO | FINALIDAD
        self._campo(
            draw, "Importe solicitado (€)",
            f"{datos.prestamo_importe:,.2f} €",
            _M, y, col3, roi_key="prestamo_importe", rois=rois,
        )
        self._campo(
            draw, "Plazo de amortización (meses)",
            f"{datos.prestamo_plazo} meses",
            _M + col3 + pad, y, col3, roi_key="prestamo_plazo", rois=rois,
        )
        self._campo(
            draw, "Finalidad del préstamo",
            datos.prestamo_finalidad,
            _M + 2 * (col3 + pad), y, col3,
            roi_key="prestamo_finalidad", rois=rois,
        )
        y += 68 + pad

        # Fila 2: CUOTA MENSUAL (1/2) | TAE (1/4) | NOTA CUOTA (1/4)
        ancho_cuota = int(full * 0.40)
        ancho_tae   = int(full * 0.22)
        ancho_nota  = full - ancho_cuota - ancho_tae - 2 * pad

        self._campo(
            draw, "Cuota mensual estimada (€)",
            f"{datos.prestamo_cuota:,.2f} €/mes",
            _M, y, ancho_cuota, roi_key="prestamo_cuota", rois=rois,
        )
        self._campo(
            draw, "T.A.E. aplicable (%)",
            f"{datos.tae:.2f} %",
            _M + ancho_cuota + pad, y, ancho_tae, obligatorio=False,
        )
        # Nota sobre la cuota
        draw.rectangle(
            [_M + ancho_cuota + ancho_tae + 2 * pad, y,
             _M + full, y + 64],
            fill=(255, 245, 220),
            outline=_COL_BANK_ACCENT,
            width=1,
        )
        draw.text(
            (_M + ancho_cuota + ancho_tae + 2 * pad + 8, y + 8),
            "⚠ Cuota calculada mediante\namortización francesa.\nSujeta a aprobación.",
            font=self._fnt_label_sm,
            fill=(130, 90, 0),
        )
        y += 68 + pad + 10

        return y

    def _dibujar_cuadro_resumen(
        self,
        draw: ImageDraw.ImageDraw,
        datos: DatosFormulario,
        y: int,
        rois: dict,
    ) -> int:
        """
        Dibuja el cuadro de resumen financiero con los datos clave del préstamo
        (destacado visualmente para facilitar la verificación).
        """
        pad  = 16
        full = _W - 2 * _M
        box_h = 220

        # Fondo destacado
        draw.rectangle(
            [_M, y, _W - _M, y + box_h],
            fill=_COL_HIGHLIGHT,
            outline=_COL_HIGHLIGHT_BDR,
            width=3,
        )

        # Título del cuadro
        draw.text(
            (_M + pad, y + 10),
            "RESUMEN DE LA OPERACIÓN",
            font=self._fnt_section,
            fill=_COL_SECTION_TITLE,
        )
        draw.line(
            [_M + pad, y + 38, _W - _M - pad, y + 38],
            fill=_COL_HIGHLIGHT_BDR, width=2,
        )

        # Datos en dos columnas
        col_a = _M + pad
        col_b = _W // 2 + pad

        datos_izq = [
            ("Titular:",                 f"{datos.sol_nombre} {datos.sol_apellidos}"),
            ("NIF:",                     datos.sol_nif),
            ("Importe del préstamo:",    f"{datos.prestamo_importe:,.2f} €"),
        ]
        datos_der = [
            ("Plazo:",                   f"{datos.prestamo_plazo} meses"),
            ("T.A.E.:",                  f"{datos.tae:.2f} %"),
            ("Cuota mensual:",           f"{datos.prestamo_cuota:,.2f} €"),
        ]

        for i, (lbl, val) in enumerate(datos_izq):
            ry = y + 52 + i * 44
            draw.text((col_a, ry), lbl, font=self._fnt_label_sm, fill=_COL_LABEL)
            draw.text((col_a, ry + 16), val, font=self._fnt_highlight, fill=_COL_VALUE)

        for i, (lbl, val) in enumerate(datos_der):
            ry = y + 52 + i * 44
            draw.text((col_b, ry), lbl, font=self._fnt_label_sm, fill=_COL_LABEL)
            draw.text((col_b, ry + 16), val, font=self._fnt_highlight, fill=_COL_VALUE)

        # Ratio de endeudamiento
        ratio_str = f"Ratio cuota/ingresos: {datos.ratio_endeudamiento:.1f} % (máx. 35 %)"
        col_ratio = (60, 150, 60) if datos.ratio_endeudamiento <= 35 else (200, 50, 50)
        draw.text(
            (_W // 2, y + box_h - 30),
            ratio_str,
            font=self._fnt_value_sm,
            fill=col_ratio,
            anchor="mt",
        )

        return y + box_h + 20

    def _dibujar_declaracion_firma(
        self,
        draw: ImageDraw.ImageDraw,
        y: int,
    ) -> None:
        """Dibuja la sección de declaración jurada y zonas de firma."""
        full = _W - 2 * _M
        pad  = 16

        # Texto de declaración
        declaracion = (
            "El/La solicitante declara que los datos consignados en el presente formulario son verídicos y completos, "
            "autorizando a BANCO DIGITAL ESPAÑOL S.A. a realizar las consultas necesarias para la evaluación de la "
            "presente solicitud, de conformidad con la Ley Orgánica 3/2018, de 5 de diciembre, de Protección de "
            "Datos Personales y garantía de los derechos digitales (LOPDGDD)."
        )
        draw.rectangle(
            [_M, y, _W - _M, y + 100],
            fill=(248, 248, 252),
            outline=_COL_FIELD_BORDER,
            width=1,
        )

        # Renderizar el texto de declaración en múltiples líneas
        max_chars_line = 120
        words = declaracion.split()
        lines, line = [], ""
        for word in words:
            if len(line) + len(word) + 1 <= max_chars_line:
                line = f"{line} {word}".strip()
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)

        for i, ln in enumerate(lines[:4]):
            draw.text((_M + pad, y + 10 + i * 20), ln,
                      font=self._fnt_legal, fill=_COL_LEGAL)

        y += 110

        # Zonas de firma
        ancho_firma = int(full * 0.42)
        gap_firma   = full - 2 * ancho_firma

        for i, (label, x_f) in enumerate([
            ("Firma del solicitante / Applicant's signature", _M),
            ("Firma del representante del banco / Bank officer", _M + ancho_firma + gap_firma),
        ]):
            draw.rectangle(
                [x_f, y, x_f + ancho_firma, y + 120],
                fill=_COL_BODY_BG,
                outline=_COL_FIELD_BORDER,
                width=2,
            )
            draw.text(
                (x_f + 8, y + 6),
                label,
                font=self._fnt_legal,
                fill=_COL_LABEL,
            )
            # Línea de firma
            draw.line(
                [x_f + 20, y + 95, x_f + ancho_firma - 20, y + 95],
                fill=_COL_FIELD_BORDER,
                width=1,
            )
            draw.text(
                (x_f + 8, y + 100),
                "Fecha: ___/___/______",
                font=self._fnt_legal,
                fill=_COL_LABEL,
            )

    def _dibujar_pie(self, draw: ImageDraw.ImageDraw) -> None:
        """Dibuja el pie de página con información legal y de contacto del banco."""
        pie_h = 60
        pie_y = _H - pie_h

        draw.rectangle([0, pie_y, _W, _H], fill=_COL_BANK_PRIMARY)
        draw.rectangle([0, pie_y, _W, pie_y + 3], fill=_COL_BANK_ACCENT)

        textos = [
            "BANCO DIGITAL ESPAÑOL S.A. · CIF A-12345678 · Inscrita en el Registro Mercantil de Madrid",
            "Supervisada por el Banco de España · IBAN ES12 3456 7890 1234 5678 9012 · BIC BDESESBXXX",
            "Dirección: C/ Gran Vía, 1, 28013 Madrid · Tel. 900 123 456 · info@bdespanol.es",
        ]
        for i, txt in enumerate(textos):
            draw.text(
                (_W // 2, pie_y + 8 + i * 16),
                txt,
                font=self._fnt_legal,
                fill=(190, 210, 235),
                anchor="mt",
            )
