"""
confidence_chart.py — Componente Streamlit para gráficos de confianza OCR.

Visualiza la confianza del OCR por campo y el desglose del veredicto final.
"""

import streamlit as st


def mostrar_confianza_campos(campos: dict, titulo: str = "Confianza OCR por campo") -> None:
    """
    Muestra un gráfico de barras horizontales con la confianza OCR por campo.

    Args:
        campos: Diccionario {clase: CampoOCRSchema}
        titulo: Título del gráfico
    """
    if not campos:
        return

    st.subheader(titulo)

    datos = sorted(
        [(k, v.confianza_ocr, v.valido) for k, v in campos.items()],
        key=lambda x: x[1], reverse=True
    )

    for clase, conf, valido in datos:
        col1, col2, col3 = st.columns([3, 5, 1])
        with col1:
            st.caption(clase)
        with col2:
            color = "normal" if conf >= 0.7 else ("off" if conf >= 0.4 else "inverse")
            st.progress(conf)
        with col3:
            st.caption(f"{'✅' if valido else '⚠'} {conf:.0%}")


def mostrar_desglose_veredicto(confianza: float, n_reglas_ok: int, conf_ocr: float) -> None:
    """
    Visualiza el desglose de la confianza global del sistema.

    Fórmula: 0.70 × (reglas_ok/9) + 0.30 × conf_ocr_media
    """
    st.subheader("📊 Desglose de confianza del sistema")

    peso_reglas = 0.70 * (n_reglas_ok / 9)
    peso_ocr    = 0.30 * conf_ocr

    col1, col2, col3 = st.columns(3)
    col1.metric("Reglas (70%)",     f"{peso_reglas:.1%}", f"{n_reglas_ok}/9 OK")
    col2.metric("OCR (30%)",        f"{peso_ocr:.1%}",    f"conf media {conf_ocr:.0%}")
    col3.metric("Confianza global", f"{confianza:.1%}")

    st.progress(confianza, text=f"Confianza global del sistema: {confianza:.1%}")


def mostrar_grafico_reglas(reglas: list) -> None:
    """Muestra el estado de cada regla R01-R09 en formato visual."""
    if not reglas:
        return

    st.subheader("📐 Estado de reglas R01–R09")
    cols = st.columns(3)
    for i, regla in enumerate(reglas):
        with cols[i % 3]:
            icono = "✅" if regla.get("pasada") else "❌"
            color = "green" if regla.get("pasada") else "red"
            st.markdown(
                f'<div style="padding:6px;border-radius:6px;border:1px solid {"#4caf50" if regla.get("pasada") else "#f44336"};margin:3px">'
                f'<b>{icono} {regla.get("codigo","")}</b><br>'
                f'<small>{regla.get("nombre","")}</small>'
                f'</div>',
                unsafe_allow_html=True
            )
