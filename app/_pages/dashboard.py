"""
dashboard.py — Página de dashboard con métricas avanzadas del sistema.

Muestra KPIs, distribución de veredictos, heatmap de errores OCR
por campo y evolución temporal de las verificaciones.
"""

import httpx
import streamlit as st
from collections import Counter


def mostrar_pagina_dashboard(api_url: str) -> None:
    st.markdown("""
    <div class="main-header">
        <h1>📊 Dashboard de Análisis</h1>
        <p>Métricas avanzadas y análisis del rendimiento del sistema</p>
    </div>
    """, unsafe_allow_html=True)

    # Obtener datos
    try:
        with httpx.Client(timeout=10.0) as client:
            r_m = client.get(f"{api_url}/api/v1/metrics/")
            r_h = client.get(f"{api_url}/api/v1/history/", params={"por_pagina": 100})
        m = r_m.json()
        h = r_h.json()
    except Exception as exc:
        st.error(f"❌ No se pudo conectar con la API: {exc}")
        return

    expedientes = h.get("expedientes", [])
    total = m.get("total_expedientes", 0)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total expedientes",   total)
    c2.metric("Aptos",               m.get("expedientes_aptos", 0))
    c3.metric("Rechazados",          m.get("expedientes_rechazados", 0))
    c4.metric("Tasa aprobación",     f"{m.get('tasa_aprobacion',0):.1%}")
    c5.metric("Confianza media",     f"{m.get('confianza_media',0):.1%}")

    st.divider()

    if not expedientes:
        st.info("Procesa expedientes para ver el dashboard. Ve a 🔍 Verificar Expediente.")
        return

    col_izq, col_der = st.columns(2)

    # ── Distribución de veredictos ────────────────────────────────────────────
    with col_izq:
        st.subheader("🥧 Distribución de veredictos")
        aptos    = sum(1 for e in expedientes if e.get("es_apto"))
        rechaz   = len(expedientes) - aptos
        st.markdown(f"""
        | Veredicto | Nº | % |
        |-----------|----|----|
        | ✅ APTO | {aptos} | {aptos/len(expedientes):.1%} |
        | ❌ INCONSISTENTE | {rechaz} | {rechaz/len(expedientes):.1%} |
        """)
        # Barra visual
        st.progress(aptos / len(expedientes) if expedientes else 0,
                    text=f"Tasa de aprobación: {aptos/len(expedientes):.1%}")

    # ── Confianzas ────────────────────────────────────────────────────────────
    with col_der:
        st.subheader("🎯 Métricas de confianza")
        st.metric("Confianza media global", f"{m.get('confianza_media',0):.1%}")
        st.progress(m.get("confianza_media", 0))
        st.metric("Confianza OCR media",    f"{m.get('confianza_ocr_media',0):.1%}")
        st.progress(m.get("confianza_ocr_media", 0))

    st.divider()

    # ── Fallos por regla ──────────────────────────────────────────────────────
    st.subheader("📐 Heatmap de fallos por regla de validación")
    fallos = m.get("fallos_por_regla", {})
    _NOMBRES = {
        "R01":"Coincidencia nombre","R02":"Coincidencia apellidos",
        "R03":"NIF válido","R04":"Fecha nacimiento","R05":"No caducado",
        "R06":"Mayor de edad","R07":"Ratio deuda ≤35%",
        "R08":"Importe válido","R09":"Autenticidad",
    }
    todas_reglas = [f"R0{i}" for i in range(1,10)]
    if fallos or total > 0:
        for codigo in todas_reglas:
            n = fallos.get(codigo, 0)
            pct = n / total if total else 0
            r_nombre = _NOMBRES.get(codigo, codigo)
            col_c, col_n, col_b, col_p = st.columns([1, 3, 4, 1])
            col_c.markdown(f"**{codigo}**")
            col_n.caption(r_nombre)
            col_b.progress(pct)
            col_p.caption(f"{n} ({pct:.0%})")
    else:
        st.success("✅ Ninguna regla ha fallado todavía.")

    st.divider()

    # ── Tiempos de respuesta ──────────────────────────────────────────────────
    st.subheader("⏱ Rendimiento del sistema")
    col_t1, col_t2 = st.columns(2)
    col_t1.metric("Tiempo medio",   f"{m.get('tiempo_medio_ms',0):.0f} ms")
    col_t2.metric("Percentil P95",  f"{m.get('tiempo_p95_ms',0):.0f} ms",
                  help="El 95% de las verificaciones tardan menos de este tiempo")

    # ── Historial reciente ────────────────────────────────────────────────────
    if expedientes:
        st.divider()
        st.subheader("🕐 Últimas verificaciones")
        for exp in expedientes[:5]:
            icono = "✅" if exp.get("es_apto") else "❌"
            ts = exp.get("timestamp","?")[:19].replace("T"," ")
            st.markdown(
                f"{icono} **{exp.get('expediente_id','?')}** — "
                f"{exp.get('resultado','?')} — "
                f"confianza {exp.get('confianza',0):.1%} — "
                f"_{ts}_"
            )
