"""
roi_visualizer.py — Componente Streamlit para visualizar ROIs sobre documentos.

Dibuja los bounding boxes detectados por YOLO sobre la imagen del documento,
con código de color por clase y etiquetas con nivel de confianza.
"""

from PIL import Image, ImageDraw, ImageFont
import streamlit as st

_COLORES = {
    # DNI
    "nombre":           "#2196F3",
    "apellidos":        "#3F51B5",
    "numero_dni":       "#E91E63",
    "fecha_nacimiento": "#009688",
    "fecha_caducidad":  "#FF5722",
    "nacionalidad":     "#795548",
    "foto":             "#9E9E9E",
    "firma":            "#607D8B",
    "mrz_line":         "#000000",
    # Formulario
    "sol_nombre":           "#2196F3",
    "sol_apellidos":        "#3F51B5",
    "sol_nif":              "#E91E63",
    "sol_fecha_nacimiento": "#009688",
    "sol_domicilio":        "#8BC34A",
    "sol_telefono":         "#FF9800",
    "sol_email":            "#00BCD4",
    "sol_situacion_laboral":"#9C27B0",
    "sol_empresa":          "#673AB7",
    "sol_ingresos_netos":   "#4CAF50",
    "prestamo_importe":     "#F44336",
    "prestamo_plazo":       "#FF5722",
    "prestamo_finalidad":   "#795548",
    "prestamo_cuota":       "#E91E63",
}


def dibujar_rois(imagen: Image.Image, campos: dict, titulo: str = "Documento") -> Image.Image:
    """
    Dibuja los bounding boxes de los campos OCR sobre la imagen.

    Args:
        imagen:  Imagen PIL del documento.
        campos:  Diccionario {clase: CampoOCRSchema} con bbox_roi si está disponible.
        titulo:  Título para el componente.

    Returns:
        Imagen PIL con los bboxes dibujados.
    """
    img_draw = imagen.copy().convert("RGB")
    draw = ImageDraw.Draw(img_draw)

    for clase, campo in campos.items():
        color = _COLORES.get(clase, "#FF0000")
        # Intentar dibujar si hay info de bbox en el campo
        if hasattr(campo, "bbox_roi") and campo.bbox_roi:
            x1, y1, x2, y2 = campo.bbox_roi
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f"{clase} ({campo.confianza_ocr:.0%})"
            draw.text((x1 + 2, max(0, y1 - 16)), label, fill=color)

    return img_draw


def mostrar_roi_visualizer(imagen: Image.Image, campos: dict, titulo: str) -> None:
    """Renderiza el visualizador de ROIs en Streamlit."""
    st.subheader(f"🔍 ROIs detectados — {titulo}")

    if not campos:
        st.image(imagen, caption=titulo, use_container_width=True)
        st.info("No hay campos OCR para visualizar.")
        return

    img_con_rois = dibujar_rois(imagen, campos, titulo)
    st.image(img_con_rois, caption=f"{titulo} — {len(campos)} campos detectados", use_container_width=True)

    # Leyenda de colores
    with st.expander("Leyenda de clases"):
        cols = st.columns(3)
        for i, (clase, color) in enumerate(_COLORES.items()):
            with cols[i % 3]:
                st.markdown(
                    f'<span style="color:{color}">■</span> <small>{clase}</small>',
                    unsafe_allow_html=True,
                )
