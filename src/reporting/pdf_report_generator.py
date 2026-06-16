"""
pdf_report_generator.py — Generador de informes PDF para expedientes verificados.

Produce un informe profesional en PDF con:
  - Cabecera corporativa (Banco Digital Español)
  - Veredicto final con semáforo visual
  - Detalle de cada regla de validación R01–R09
  - Campos extraídos por OCR (DNI y formulario)
  - Perfil de riesgo crediticio
  - Pie de página con timestamp y número de expediente

La generación usa únicamente la librería estándar `reportlab`
(incluida en requirements.txt). No requiere LibreOffice ni Word.

Uso típico:
    from src.reporting.pdf_report_generator import PDFReportGenerator
    gen = PDFReportGenerator()
    ruta = gen.generar(resultado, ruta_salida=Path("informes/EXP-001.pdf"))
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Colores corporativos ──────────────────────────────────────────────────────
_AZUL_BANCO    = (26,  58, 107)     # #1a3a6b
_AZUL_CLARO    = (46, 109, 164)     # #2e6da4
_VERDE_APTO    = (46, 125,  50)     # #2e7d32
_ROJO_RECHAZO  = (198,  40,  40)    # #c62828
_NARANJA_WARN  = (245, 124,   0)    # #f57c00
_GRIS_CLARO    = (245, 245, 245)    # #f5f5f5
_GRIS_MEDIO    = (224, 224, 224)    # #e0e0e0
_NEGRO         = (33,  33,  33)     # #212121
_BLANCO        = (255, 255, 255)

# Convertir tuplas RGB (0-255) a (0.0-1.0) para reportlab
def _rgb(r, g, b):
    from reportlab.lib.colors import Color
    return Color(r/255, g/255, b/255)


class PDFReportGenerator:
    """
    Genera informes PDF de expedientes de verificación documental.

    Args:
        logo_path: Ruta opcional a un fichero PNG/JPG con el logo del banco.
                   Si no se proporciona, se usa un placeholder de texto.
    """

    def __init__(self, logo_path: Optional[Path] = None) -> None:
        self._logo = logo_path

    def generar(
        self,
        resultado: dict,
        ruta_salida: Optional[Path] = None,
    ) -> Path:
        """
        Genera el informe PDF de un expediente.

        Args:
            resultado: Diccionario con el resultado de la verificación
                       (salida de ExpedienteResult.to_dict() o VerificacionResponse).
            ruta_salida: Ruta donde guardar el PDF. Si es None, se genera
                         en el directorio actual con el ID del expediente.

        Returns:
            Ruta al fichero PDF generado.
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                HRFlowable, KeepTogether,
            )
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        except ImportError as exc:
            raise ImportError(
                "reportlab no está instalado. Ejecuta: pip install reportlab"
            ) from exc

        exp_id = resultado.get("expediente_id", "DESCONOCIDO")

        if ruta_salida is None:
            ruta_salida = Path(f"informe_{exp_id}.pdf")
        ruta_salida = Path(ruta_salida)
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)

        # Configurar documento A4
        doc = SimpleDocTemplate(
            str(ruta_salida),
            pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm,   bottomMargin=2*cm,
        )

        # Estilos
        estilos = self._crear_estilos()

        # Construir el contenido página a página
        historia = []
        historia += self._seccion_cabecera(resultado, estilos)
        historia += self._seccion_veredicto(resultado, estilos)
        historia.append(HRFlowable(width="100%", thickness=1, color=_rgb(*_GRIS_MEDIO)))
        historia.append(Spacer(1, 0.3*cm))
        historia += self._seccion_reglas(resultado, estilos)
        historia.append(Spacer(1, 0.3*cm))
        historia += self._seccion_campos_ocr(resultado, estilos)
        historia.append(Spacer(1, 0.3*cm))
        historia += self._seccion_perfil_riesgo(resultado, estilos)
        historia += self._seccion_autenticidad(resultado, estilos)
        historia += self._seccion_pie(resultado, estilos)

        # Generar PDF
        doc.build(historia)
        logger.info("Informe PDF generado: %s", ruta_salida)
        return ruta_salida

    # ── Estilos ───────────────────────────────────────────────────────────────

    def _crear_estilos(self) -> dict:
        """Crea el diccionario de estilos ParagraphStyle del informe."""
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

        return {
            "titulo_banco": ParagraphStyle(
                "titulo_banco",
                fontName="Helvetica-Bold",
                fontSize=16,
                textColor=_rgb(*_AZUL_BANCO),
                spaceAfter=2,
                alignment=TA_CENTER,
            ),
            "subtitulo_banco": ParagraphStyle(
                "subtitulo_banco",
                fontName="Helvetica",
                fontSize=9,
                textColor=_rgb(*_AZUL_CLARO),
                spaceAfter=8,
                alignment=TA_CENTER,
            ),
            "titulo_seccion": ParagraphStyle(
                "titulo_seccion",
                fontName="Helvetica-Bold",
                fontSize=11,
                textColor=_rgb(*_AZUL_BANCO),
                spaceBefore=10,
                spaceAfter=4,
            ),
            "veredicto_apto": ParagraphStyle(
                "veredicto_apto",
                fontName="Helvetica-Bold",
                fontSize=18,
                textColor=_rgb(*_VERDE_APTO),
                alignment=TA_CENTER,
                spaceBefore=6,
                spaceAfter=4,
            ),
            "veredicto_rechazado": ParagraphStyle(
                "veredicto_rechazado",
                fontName="Helvetica-Bold",
                fontSize=18,
                textColor=_rgb(*_ROJO_RECHAZO),
                alignment=TA_CENTER,
                spaceBefore=6,
                spaceAfter=4,
            ),
            "confianza": ParagraphStyle(
                "confianza",
                fontName="Helvetica",
                fontSize=10,
                textColor=_rgb(*_NEGRO),
                alignment=TA_CENTER,
                spaceAfter=6,
            ),
            "normal": ParagraphStyle(
                "normal",
                fontName="Helvetica",
                fontSize=9,
                textColor=_rgb(*_NEGRO),
                spaceAfter=2,
            ),
            "negrita": ParagraphStyle(
                "negrita",
                fontName="Helvetica-Bold",
                fontSize=9,
                textColor=_rgb(*_NEGRO),
                spaceAfter=2,
            ),
            "pie": ParagraphStyle(
                "pie",
                fontName="Helvetica",
                fontSize=7,
                textColor=_rgb(150, 150, 150),
                alignment=TA_CENTER,
            ),
        }

    # ── Secciones ─────────────────────────────────────────────────────────────

    def _seccion_cabecera(self, r: dict, e: dict) -> list:
        """Cabecera corporativa con nombre del banco y datos del expediente."""
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import cm

        elementos = []

        # Nombre del banco
        elementos.append(Paragraph("BANCO DIGITAL ESPAÑOL", e["titulo_banco"]))
        elementos.append(Paragraph(
            "Departamento de Análisis de Riesgo · Verificación Documental",
            e["subtitulo_banco"],
        ))

        # Datos del expediente en tabla de 2 columnas
        ts = r.get("timestamp", datetime.now().isoformat())[:19].replace("T", " ")
        tabla_datos = [
            ["Expediente:", r.get("expediente_id", "—"),
             "Fecha:", ts],
        ]
        t = Table(tabla_datos, colWidths=[3*cm, 7*cm, 2.5*cm, 4.5*cm])
        t.setStyle(TableStyle([
            ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
            ("FONTNAME",  (0,0), (0,-1),  "Helvetica-Bold"),
            ("FONTNAME",  (2,0), (2,-1),  "Helvetica-Bold"),
            ("FONTSIZE",  (0,0), (-1,-1), 9),
            ("TEXTCOLOR", (0,0), (-1,-1), _rgb(*_NEGRO)),
            ("BACKGROUND",(0,0), (-1,-1), _rgb(*_GRIS_CLARO)),
            ("BOX",       (0,0), (-1,-1), 0.5, _rgb(*_GRIS_MEDIO)),
            ("PADDING",   (0,0), (-1,-1), 5),
        ]))
        elementos.append(Spacer(1, 0.3*cm))
        elementos.append(t)
        elementos.append(Spacer(1, 0.4*cm))
        return elementos

    def _seccion_veredicto(self, r: dict, e: dict) -> list:
        """Bloque principal del veredicto con semáforo visual."""
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import cm

        es_apto   = r.get("es_apto", False)
        resultado = r.get("resultado", "DESCONOCIDO")
        confianza = r.get("confianza", 0.0)

        icono    = "✓  " if es_apto else "✗  "
        estilo   = e["veredicto_apto"] if es_apto else e["veredicto_rechazado"]
        color_bg = _rgb(*_VERDE_APTO) if es_apto else _rgb(*_ROJO_RECHAZO)

        # Tabla de una celda con fondo de color
        tabla = Table([[Paragraph(icono + resultado, estilo)]], colWidths=[17*cm])
        tabla.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,-1), _rgb(*(
                (230, 244, 234) if es_apto else (252, 232, 230)
            ))),
            ("BOX",         (0,0), (-1,-1), 2, color_bg),
            ("ALIGN",       (0,0), (-1,-1), "CENTER"),
            ("PADDING",     (0,0), (-1,-1), 10),
            ("ROUNDEDCORNERS", [4]),
        ]))

        elementos = [tabla]
        elementos.append(Paragraph(
            f"Confianza del sistema: <b>{confianza:.1%}</b>",
            e["confianza"],
        ))

        # Reglas fallidas (si las hay)
        reglas_fail = (r.get("validacion") or {}).get("reglas_fallidas", [])
        if reglas_fail:
            elementos.append(Paragraph(
                f"Reglas fallidas: <font color='#c62828'><b>{', '.join(reglas_fail)}</b></font>",
                e["confianza"],
            ))

        return elementos

    def _seccion_reglas(self, r: dict, e: dict) -> list:
        """Tabla con el resultado de cada regla R01–R09."""
        from reportlab.platypus import Paragraph, Table, TableStyle, KeepTogether
        from reportlab.lib.units import cm

        validacion = r.get("validacion") or {}
        reglas     = validacion.get("reglas", [])

        if not reglas:
            return []

        elementos = [Paragraph("Resultado de las Reglas de Validación", e["titulo_seccion"])]

        # Cabecera de la tabla
        filas = [["Código", "Regla", "Resultado", "Detalle"]]

        for regla in reglas:
            pasada = regla.get("pasada", False)
            icono  = "✓" if pasada else "✗"
            color_texto = "#2e7d32" if pasada else "#c62828"
            filas.append([
                regla.get("codigo", ""),
                Paragraph(regla.get("nombre", ""), e["normal"]),
                Paragraph(
                    f'<font color="{color_texto}"><b>{icono}</b></font>',
                    e["normal"],
                ),
                Paragraph(regla.get("detalle", ""), e["normal"]),
            ])

        t = Table(filas, colWidths=[1.5*cm, 4*cm, 1.8*cm, 9.7*cm])
        t.setStyle(TableStyle([
            # Cabecera
            ("BACKGROUND",  (0,0), (-1,0),  _rgb(*_AZUL_BANCO)),
            ("TEXTCOLOR",   (0,0), (-1,0),  _rgb(*_BLANCO)),
            ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,0),  9),
            # Cuerpo
            ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE",    (0,1), (-1,-1), 8),
            ("TEXTCOLOR",   (0,1), (-1,-1), _rgb(*_NEGRO)),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [_rgb(*_BLANCO), _rgb(*_GRIS_CLARO)]),
            # Bordes
            ("GRID",        (0,0), (-1,-1), 0.5, _rgb(*_GRIS_MEDIO)),
            ("ALIGN",       (0,0), (0,-1),  "CENTER"),
            ("ALIGN",       (2,0), (2,-1),  "CENTER"),
            ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
            ("PADDING",     (0,0), (-1,-1), 4),
        ]))
        elementos.append(t)
        return [KeepTogether(elementos)]

    def _seccion_campos_ocr(self, r: dict, e: dict) -> list:
        """Tabla con los campos extraídos por OCR de DNI y formulario."""
        from reportlab.platypus import Paragraph, Table, TableStyle, KeepTogether
        from reportlab.lib.units import cm

        campos_dni  = r.get("campos_dni",  {})
        campos_form = r.get("campos_form", {})

        if not campos_dni and not campos_form:
            return []

        elementos = [Paragraph("Campos Extraídos por OCR", e["titulo_seccion"])]

        def _filas_campos(campos: dict, prefijo: str) -> list:
            filas = []
            for clase, datos in campos.items():
                valido = datos.get("valido", False)
                icono  = "✓" if valido else "✗"
                color  = "#2e7d32" if valido else "#c62828"
                filas.append([
                    Paragraph(clase, e["normal"]),
                    Paragraph(datos.get("texto_norm", "—"), e["normal"]),
                    Paragraph(f'{datos.get("confianza_ocr", 0):.0%}', e["normal"]),
                    Paragraph(f'<font color="{color}">{icono}</font>', e["normal"]),
                ])
            return filas

        cabecera = [["Campo", "Valor extraído", "Conf. OCR", "Válido"]]
        anchos   = [4.5*cm, 8*cm, 2.5*cm, 2*cm]

        for titulo, campos in [("DNI", campos_dni), ("Formulario de Préstamo", campos_form)]:
            filas_campo = _filas_campos(campos, titulo)
            if not filas_campo:
                continue
            filas_tabla = cabecera + filas_campo
            t = Table(filas_tabla, colWidths=anchos)
            t.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (-1,0),  _rgb(*_AZUL_CLARO)),
                ("TEXTCOLOR",   (0,0), (-1,0),  _rgb(*_BLANCO)),
                ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
                ("FONTSIZE",    (0,0), (-1,-1), 8),
                ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
                ("TEXTCOLOR",   (0,1), (-1,-1), _rgb(*_NEGRO)),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [_rgb(*_BLANCO), _rgb(*_GRIS_CLARO)]),
                ("GRID",        (0,0), (-1,-1), 0.5, _rgb(*_GRIS_MEDIO)),
                ("ALIGN",       (2,0), (3,-1),  "CENTER"),
                ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
                ("PADDING",     (0,0), (-1,-1), 4),
            ]))
            elementos.append(Paragraph(titulo, e["negrita"]))
            elementos.append(t)

        return [KeepTogether(elementos)]

    def _seccion_perfil_riesgo(self, r: dict, e: dict) -> list:
        """Bloque del perfil de riesgo crediticio."""
        from reportlab.platypus import Paragraph, Table, TableStyle, Spacer
        from reportlab.lib.units import cm

        perfil = r.get("perfil_riesgo")
        if not perfil:
            return []

        nivel  = perfil.get("nivel", "?")
        ratio  = perfil.get("ratio_deuda", 0)
        detalle = perfil.get("detalle", "")

        color_nivel = {
            "BAJO":  _rgb(*_VERDE_APTO),
            "MEDIO": _rgb(*_NARANJA_WARN),
            "ALTO":  _rgb(*_ROJO_RECHAZO),
        }.get(nivel, _rgb(*_NEGRO))

        filas = [
            ["Nivel de Riesgo:", nivel, "Ratio de Endeudamiento:", f"{ratio:.1%}"],
            ["Detalle:", detalle, "", ""],
        ]
        t = Table(filas, colWidths=[3.5*cm, 4*cm, 4.5*cm, 5*cm])
        t.setStyle(TableStyle([
            ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
            ("FONTNAME",    (0,0), (0,-1),  "Helvetica-Bold"),
            ("FONTNAME",    (2,0), (2,0),   "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("TEXTCOLOR",   (0,0), (-1,-1), _rgb(*_NEGRO)),
            ("TEXTCOLOR",   (1,0), (1,0),   color_nivel),
            ("FONTNAME",    (1,0), (1,0),   "Helvetica-Bold"),
            ("BACKGROUND",  (0,0), (-1,-1), _rgb(*_GRIS_CLARO)),
            ("BOX",         (0,0), (-1,-1), 0.5, _rgb(*_GRIS_MEDIO)),
            ("GRID",        (0,0), (-1,-1), 0.5, _rgb(*_GRIS_MEDIO)),
            ("SPAN",        (1,1), (3,1)),
            ("PADDING",     (0,0), (-1,-1), 5),
        ]))

        return [
            Paragraph("Perfil de Riesgo Crediticio", e["titulo_seccion"]),
            t,
            Spacer(1, 0.2*cm),
        ]

    def _seccion_autenticidad(self, r: dict, e: dict) -> list:
        """Resultado del clasificador de autenticidad para cada documento."""
        from reportlab.platypus import Paragraph, Table, TableStyle, Spacer
        from reportlab.lib.units import cm

        aut_dni  = r.get("autenticidad_dni")
        aut_form = r.get("autenticidad_form")

        if not aut_dni and not aut_form:
            return []

        elementos = [Paragraph("Clasificación de Autenticidad", e["titulo_seccion"])]

        filas = [["Documento", "Clasificación", "Confianza"]]
        for doc, datos in [("DNI", aut_dni), ("Formulario", aut_form)]:
            if datos:
                etq   = datos.get("etiqueta", "?")
                conf  = datos.get("confianza", 0)
                color = "#2e7d32" if etq == "LEGÍTIMO" else "#c62828"
                filas.append([
                    doc,
                    Paragraph(f'<font color="{color}"><b>{etq}</b></font>', e["normal"]),
                    f"{conf:.1%}",
                ])

        t = Table(filas, colWidths=[4*cm, 7*cm, 6*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0),  _rgb(*_AZUL_BANCO)),
            ("TEXTCOLOR",   (0,0), (-1,0),  _rgb(*_BLANCO)),
            ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
            ("TEXTCOLOR",   (0,0), (0,-1),  _rgb(*_NEGRO)),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [_rgb(*_BLANCO), _rgb(*_GRIS_CLARO)]),
            ("GRID",        (0,0), (-1,-1), 0.5, _rgb(*_GRIS_MEDIO)),
            ("ALIGN",       (2,0), (2,-1),  "CENTER"),
            ("PADDING",     (0,0), (-1,-1), 5),
        ]))
        elementos.append(t)
        return [Spacer(1, 0.2*cm)] + elementos

    def _seccion_pie(self, r: dict, e: dict) -> list:
        """Pie de página con disclaimer legal y timestamp."""
        from reportlab.platypus import Paragraph, Spacer, HRFlowable
        from reportlab.lib.units import cm

        ts = r.get("timestamp", datetime.now().isoformat())[:19].replace("T", " ")
        return [
            Spacer(1, 0.5*cm),
            HRFlowable(width="100%", thickness=0.5, color=_rgb(*_GRIS_MEDIO)),
            Spacer(1, 0.2*cm),
            Paragraph(
                "Este informe ha sido generado automáticamente por el Sistema de "
                "Verificación Documental del Banco Digital Español. Los resultados "
                "tienen carácter orientativo y deben ser revisados por un analista "
                "de riesgo antes de tomar decisiones definitivas.",
                e["pie"],
            ),
            Paragraph(
                f"Generado: {ts} · "
                f"Expediente: {r.get('expediente_id', '—')} · "
                f"TFM — Máster en Inteligencia Artificial",
                e["pie"],
            ),
        ]
