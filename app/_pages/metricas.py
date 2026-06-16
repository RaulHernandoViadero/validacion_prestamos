"""
metricas.py — Página de métricas y KPIs del sistema.

Muestra en tiempo real los indicadores de rendimiento del sistema:
  - Tasa de aprobación de expedientes
  - Confianza media del sistema y del OCR
  - Fallos por regla de validación
  - Tiempos de respuesta (media y percentil 95)
"""

import httpx
import streamlit as st


def mostrar_pagina_metricas(api_url: str) -> None:
    """Renderiza la página de métricas del sistema."""

    st.markdown("""
    <div class="main-header">
        <h1>📊 Métricas del Sistema</h1>
        <p>Indicadores de rendimiento en tiempo real desde el arranque del servidor</p>
    </div>
    """, unsafe_allow_html=True)

    # Botón de refresco
    col_ref, _ = st.columns([1, 4])
    with col_ref:
        refrescar = st.button("🔄 Actualizar métricas", use_container_width=True)

    # Obtener métricas de la API
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{api_url}/api/v1/metrics/")
        if resp.status_code != 200:
            st.error(f"❌ Error obteniendo métricas: {resp.status_code}")
            return
        m = resp.json()
    except Exception as exc:
        st.error(f"❌ No se pudo conectar con la API: {exc}")
        return

    total      = m.get("total_expedientes",    0)
    aptos      = m.get("expedientes_aptos",    0)
    rechazados = m.get("expedientes_rechazados", 0)
    tasa       = m.get("tasa_aprobacion",      0.0)
    conf_media = m.get("confianza_media",       0.0)
    ocr_media  = m.get("confianza_ocr_media",   0.0)
    t_medio    = m.get("tiempo_medio_ms",       0.0)
    t_p95      = m.get("tiempo_p95_ms",         0.0)
    fallos     = m.get("fallos_por_regla",       {})

    st.divider()

    # ── KPIs principales ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Expedientes",    total)
    col2.metric("Expedientes Aptos",    aptos,     delta=f"{tasa:.1%} tasa")
    col3.metric("Rechazados",           rechazados)
    col4.metric("Tasa de Aprobación",   f"{tasa:.1%}")

    st.divider()

    # ── Confianza ─────────────────────────────────────────────────────────────
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("🎯 Confianza del Sistema")
        st.metric("Confianza media global",  f"{conf_media:.1%}")
        st.progress(conf_media)
        st.metric("Confianza OCR media",     f"{ocr_media:.1%}")
        st.progress(ocr_media)

    # ── Rendimiento ───────────────────────────────────────────────────────────
    with col6:
        st.subheader("⏱ Tiempos de Respuesta")
        st.metric("Tiempo medio",         f"{t_medio:.0f} ms")
        st.metric("Percentil 95 (P95)",   f"{t_p95:.0f} ms",
                  help="El 95% de las requests se resuelven en menos de este tiempo")

    st.divider()

    # ── Fallos por regla ──────────────────────────────────────────────────────
    st.subheader("📐 Fallos por Regla de Validación")

    _NOMBRES_REGLAS = {
        "R01": "Coincidencia nombre",
        "R02": "Coincidencia apellidos",
        "R03": "NIF coincidente y válido",
        "R04": "Coincidencia fecha nacimiento",
        "R05": "Documento no caducado",
        "R06": "Titular mayor de edad",
        "R07": "Ratio endeudamiento ≤35%",
        "R08": "Importe dentro de límites",
        "R09": "Autenticidad documentos",
    }

    if fallos:
        # Ordenar por número de fallos descendente
        fallos_ord = sorted(fallos.items(), key=lambda x: x[1], reverse=True)
        for codigo, n_fallos in fallos_ord:
            nombre = _NOMBRES_REGLAS.get(codigo, codigo)
            pct    = n_fallos / total if total else 0
            col_r, col_n, col_v, col_b = st.columns([1, 3, 1, 4])
            col_r.markdown(f"**{codigo}**")
            col_n.markdown(nombre)
            col_v.markdown(f"**{n_fallos}**")
            col_b.progress(pct, text=f"{pct:.1%} de expedientes")
    else:
        if total == 0:
            st.info("No se han procesado expedientes todavía.")
        else:
            st.success("✅ Ninguna regla ha fallado en los expedientes procesados.")

    st.divider()
    st.caption("Las métricas se reinician con cada reinicio del servidor.")
