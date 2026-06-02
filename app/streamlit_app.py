"""
streamlit_app.py — Aplicación principal del frontend Streamlit.

Punto de entrada del interfaz gráfico del sistema de verificación
documental. Organiza la aplicación en tres páginas:

  1. Verificar Expediente — sube DNI + formulario y muestra el veredicto
  2. Historial            — lista los expedientes procesados en la sesión
  3. Métricas del Sistema — KPIs y estadísticas en tiempo real

Arranque:
    streamlit run app/streamlit_app.py

Variables de entorno:
    API_URL  — URL base de la API FastAPI (default: http://localhost:8000)
"""

import os

import streamlit as st

# ── Configuración de la página ────────────────────────────────────────────────

st.set_page_config(
    page_title="Verificación Documental — TFM",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Sistema de Verificación Documental**\n\n"
            "TFM: Verificación Documental de Identidad y Solicitudes de Préstamo "
            "mediante Visión Artificial y Deep Learning.\n\n"
            "Autor: Raúl Hernando Viadero"
        ),
    },
)

# ── CSS personalizado ─────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Cabecera de la app */
.main-header {
    background: linear-gradient(135deg, #1a3a6b 0%, #2e6da4 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}
.main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
.main-header p  { color: #c8d8f0; margin: 0.3rem 0 0; font-size: 0.95rem; }

/* Tarjeta de veredicto */
.veredicto-apto {
    background: #e6f4ea;
    border-left: 6px solid #2e7d32;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
}
.veredicto-rechazado {
    background: #fce8e6;
    border-left: 6px solid #c62828;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
}

/* Badge de regla */
.regla-ok   { color: #2e7d32; font-weight: 600; }
.regla-fail { color: #c62828; font-weight: 600; }

/* Separador sutil */
.separador { border-top: 1px solid #e0e0e0; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ── URL de la API ─────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/bank-building.png",
        width=72,
    )
    st.title("Banco Digital Español")
    st.caption("Sistema de Verificación Documental")
    st.divider()

    pagina = st.radio(
        "Navegación",
        options=["🔍 Verificar Expediente", "📋 Historial", "📊 Métricas"],
        index=0,
    )
    st.divider()

    # Estado de la API
    import httpx
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{API_URL}/health")
        if r.status_code == 200:
            h = r.json()
            st.success("✅ API conectada")
            st.caption(f"v{h.get('version', '?')}")
            if h.get("yolo_cargado"):
                st.caption("🟢 YOLO cargado")
            else:
                st.caption("🟡 YOLO (carga lazy)")
        else:
            st.error("❌ API no responde")
    except Exception:
        st.error("❌ API no disponible")
        st.caption(f"Comprueba que la API está en {API_URL}")

    st.divider()
    st.caption("TFM — Máster en IA")
    st.caption("Raúl Hernando Viadero")


# ── Páginas ───────────────────────────────────────────────────────────────────

if pagina == "🔍 Verificar Expediente":
    from app.pages.verificacion import mostrar_pagina_verificacion
    mostrar_pagina_verificacion(API_URL)

elif pagina == "📋 Historial":
    from app.pages.historial import mostrar_pagina_historial
    mostrar_pagina_historial(API_URL)

elif pagina == "📊 Métricas":
    from app.pages.metricas import mostrar_pagina_metricas
    mostrar_pagina_metricas(API_URL)
