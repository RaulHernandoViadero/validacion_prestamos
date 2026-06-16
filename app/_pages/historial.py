"""
historial.py — Página de historial de expedientes verificados.

Muestra la lista paginada de expedientes procesados en la sesión
con filtros y la posibilidad de ver el detalle de cada uno.
"""

from typing import Optional

import httpx
import streamlit as st


def mostrar_pagina_historial(api_url: str) -> None:
    """Renderiza la página de historial de expedientes."""

    st.markdown("""
    <div class="main-header">
        <h1>📋 Historial de Expedientes</h1>
        <p>Expedientes procesados en esta sesión del sistema</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Filtros ───────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filtro_resultado = st.selectbox(
            "Resultado",
            options=["Todos", "Solo aptos", "Solo rechazados"],
        )
    with col_f2:
        por_pagina = st.selectbox("Resultados por página", options=[10, 20, 50], index=1)
    with col_f3:
        pagina = st.number_input("Página", min_value=1, value=1, step=1)

    solo_aptos: Optional[bool] = None
    if filtro_resultado == "Solo aptos":
        solo_aptos = True
    elif filtro_resultado == "Solo rechazados":
        solo_aptos = False

    # ── Llamada a la API ──────────────────────────────────────────────────────
    params: dict = {"pagina": pagina, "por_pagina": por_pagina}
    if solo_aptos is not None:
        params["solo_aptos"] = str(solo_aptos).lower()

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{api_url}/api/v1/history/", params=params)
        if resp.status_code != 200:
            st.error(f"❌ Error de la API: {resp.status_code}")
            return
        data = resp.json()
    except Exception as exc:
        st.error(f"❌ No se pudo conectar con la API: {exc}")
        return

    # ── Tabla de expedientes ──────────────────────────────────────────────────
    total      = data.get("total", 0)
    expedientes = data.get("expedientes", [])

    st.metric("Total de expedientes", total)

    if not expedientes:
        st.info("No hay expedientes en el historial. Verifica algún expediente primero.")
        return

    st.divider()

    for exp in expedientes:
        es_apto = exp.get("es_apto", False)
        icono   = "✅" if es_apto else "❌"
        conf    = exp.get("confianza", 0)
        ts      = exp.get("timestamp", "?")[:19].replace("T", " ")

        col1, col2, col3, col4 = st.columns([3, 3, 2, 2])
        with col1:
            st.markdown(f"**{exp.get('expediente_id', '?')}**")
        with col2:
            st.markdown(f"{icono} {exp.get('resultado', '?')}")
        with col3:
            st.markdown(f"Confianza: **{conf:.1%}**")
        with col4:
            st.caption(ts)

        reglas_fail = exp.get("reglas_fallidas", [])
        if reglas_fail:
            st.caption(f"  Reglas fallidas: {', '.join(reglas_fail)}")

        st.divider()

    # Paginación info
    paginas_total = (total + por_pagina - 1) // por_pagina
    st.caption(f"Página {pagina} de {paginas_total} | {total} expedientes en total")
