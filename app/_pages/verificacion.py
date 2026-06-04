"""
verificacion.py — Página principal de verificación de expedientes.

Permite al usuario:
  1. Subir la imagen del DNI y del formulario de préstamo
  2. Lanzar la verificación llamando al endpoint POST /api/v1/verify
  3. Ver el veredicto con semáforo visual (APTO / INCONSISTENTE)
  4. Explorar el detalle de cada regla R01–R09
  5. Ver los campos extraídos por OCR
  6. Ver el perfil de riesgo crediticio
"""

import io
from typing import Optional

import httpx
import streamlit as st
from PIL import Image


def mostrar_pagina_verificacion(api_url: str) -> None:
    """Renderiza la página de verificación de expedientes."""

    # Cabecera
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Verificación de Expediente</h1>
        <p>Suba el DNI y el formulario de préstamo para verificar la coherencia del expediente</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Formulario de subida ──────────────────────────────────────────────────
    col_dni, col_form = st.columns(2, gap="large")

    with col_dni:
        st.subheader("📄 DNI del Solicitante")
        archivo_dni = st.file_uploader(
            "Sube la imagen del DNI",
            type=["jpg", "jpeg", "png", "webp"],
            key="upload_dni",
            help="Imagen del documento nacional de identidad (anverso). Max 20 MB.",
        )
        if archivo_dni:
            img = Image.open(archivo_dni)
            st.image(img, caption="DNI cargado", use_container_width=True)

    with col_form:
        st.subheader("📋 Formulario de Préstamo")
        archivo_form = st.file_uploader(
            "Sube la imagen del formulario",
            type=["jpg", "jpeg", "png", "webp"],
            key="upload_form",
            help="Formulario de solicitud de préstamo cumplimentado. Max 20 MB.",
        )
        if archivo_form:
            img_form = Image.open(archivo_form)
            st.image(img_form, caption="Formulario cargado", use_container_width=True)

    # ID de expediente opcional
    exp_id = st.text_input(
        "ID del Expediente (opcional)",
        placeholder="EXP-2024-0042 — se genera automáticamente si se deja vacío",
        max_chars=64,
    )

    st.divider()

    # ── Botón de verificación ─────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        verificar = st.button(
            "▶ Verificar Expediente",
            type="primary",
            disabled=(archivo_dni is None or archivo_form is None),
            use_container_width=True,
        )
    with col_info:
        if archivo_dni is None or archivo_form is None:
            st.info("Sube ambos documentos para habilitar la verificación.")

    # ── Proceso y resultado ───────────────────────────────────────────────────
    if verificar and archivo_dni and archivo_form:
        with st.spinner("Procesando expediente… (YOLO + OCR + validación)"):
            resultado = _llamar_api(
                api_url=api_url,
                archivo_dni=archivo_dni,
                archivo_form=archivo_form,
                expediente_id=exp_id or None,
            )

        if resultado is None:
            return   # error ya mostrado dentro de _llamar_api

        _mostrar_resultado(resultado)


def _llamar_api(
    api_url: str,
    archivo_dni,
    archivo_form,
    expediente_id: Optional[str],
) -> Optional[dict]:
    """
    Llama al endpoint POST /api/v1/verify con los ficheros subidos.

    Returns:
        Diccionario JSON de la respuesta o None si hubo error.
    """
    # Rebobinar ficheros (Streamlit los puede haber leído antes)
    archivo_dni.seek(0)
    archivo_form.seek(0)

    files = {
        "imagen_dni": (archivo_dni.name, archivo_dni.read(), archivo_dni.type or "image/png"),
        "imagen_formulario": (archivo_form.name, archivo_form.read(), archivo_form.type or "image/png"),
    }
    data = {}
    if expediente_id:
        data["expediente_id"] = expediente_id

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{api_url}/api/v1/verify/",
                files=files,
                data=data,
            )
        if resp.status_code == 200:
            return resp.json()
        else:
            error = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            st.error(
                f"❌ Error de la API ({resp.status_code}): "
                + error.get("detail", {}).get("mensaje", resp.text[:200])
            )
            return None
    except httpx.ConnectError:
        st.error(f"❌ No se puede conectar con la API en {api_url}. ¿Está arrancada?")
        return None
    except httpx.TimeoutException:
        st.error("❌ La API tardó demasiado en responder. Inténtelo de nuevo.")
        return None
    except Exception as exc:
        st.error(f"❌ Error inesperado: {exc}")
        return None


def _mostrar_resultado(r: dict) -> None:
    """Renderiza el resultado completo de la verificación."""

    es_apto    = r.get("es_apto", False)
    confianza  = r.get("confianza", 0.0)
    resultado  = r.get("resultado", "DESCONOCIDO")
    exp_id     = r.get("expediente_id", "—")

    st.divider()

    # ── Veredicto principal ───────────────────────────────────────────────────
    if es_apto:
        st.markdown(f"""
        <div class="veredicto-apto">
            <h2>✅ {resultado}</h2>
            <p style="margin:0">Expediente <strong>{exp_id}</strong> —
               Confianza del sistema: <strong>{confianza:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        reglas_fail = r.get("validacion", {}).get("reglas_fallidas", [])
        st.markdown(f"""
        <div class="veredicto-rechazado">
            <h2>❌ {resultado}</h2>
            <p style="margin:0">Expediente <strong>{exp_id}</strong> —
               Confianza del sistema: <strong>{confianza:.1%}</strong><br>
               Reglas fallidas: <strong>{', '.join(reglas_fail) or 'ninguna'}</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # Tiempo de proceso + botón PDF
    t_ms = r.get("tiempo_total_ms", 0)
    col_t, col_pdf = st.columns([3, 1])
    col_t.caption(f"⏱ Procesado en {t_ms:.0f} ms")
    with col_pdf:
        try:
            import tempfile
            from pathlib import Path
            from src.reporting.pdf_report_generator import PDFReportGenerator
            gen = PDFReportGenerator()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                ruta_pdf = Path(tmp.name)
            gen.generar(r, ruta_salida=ruta_pdf)
            with open(ruta_pdf, "rb") as f_pdf:
                pdf_bytes = f_pdf.read()
            st.download_button(
                label="📄 Exportar PDF",
                data=pdf_bytes,
                file_name=f"informe_{exp_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            ruta_pdf.unlink(missing_ok=True)
        except Exception:
            pass   # PDF no disponible en Docker sin montaje src/

    # Advertencias
    for adv in r.get("advertencias", []):
        st.warning(f"⚠ {adv}")

    st.divider()

    # ── Tabs de detalle ───────────────────────────────────────────────────────
    tab_reglas, tab_ocr, tab_riesgo, tab_auth, tab_raw = st.tabs([
        "📐 Reglas R01–R09",
        "🔤 Campos OCR",
        "💰 Perfil de Riesgo",
        "🔒 Autenticidad",
        "🗃 JSON Completo",
    ])

    # Tab 1: Reglas
    with tab_reglas:
        validacion = r.get("validacion") or {}
        reglas = validacion.get("reglas", [])
        if reglas:
            n_ok   = validacion.get("n_reglas_ok",   0)
            n_fail = validacion.get("n_reglas_fallo", 0)
            st.metric("Reglas superadas", f"{n_ok}/9", delta=f"{n_fail} fallos" if n_fail else "Sin fallos")
            for regla in reglas:
                icono   = "✅" if regla["pasada"] else "❌"
                clase_c = "regla-ok" if regla["pasada"] else "regla-fail"
                with st.expander(f'{icono} {regla["codigo"]} — {regla["nombre"]}'):
                    st.markdown(
                        f'<span class="{clase_c}">{regla["detalle"]}</span>',
                        unsafe_allow_html=True,
                    )
                    if regla.get("severidad") == "WARNING":
                        st.caption("⚠ Severidad: WARNING (no bloquea el veredicto)")
        else:
            st.info("No hay datos de validación disponibles.")

    # Tab 2: Campos OCR
    with tab_ocr:
        col_d, col_f = st.columns(2)
        with col_d:
            st.subheader("🪪 DNI")
            campos_dni = r.get("campos_dni", {})
            if campos_dni:
                for campo, datos in campos_dni.items():
                    icono = "✅" if datos.get("valido") else "⚠"
                    st.markdown(f"**{campo}**")
                    st.code(datos.get("texto_norm", "—"))
                    st.caption(f"{icono} Confianza OCR: {datos.get('confianza_ocr', 0):.1%}")
            else:
                st.info("Sin campos DNI extraídos.")

        with col_f:
            st.subheader("📄 Formulario")
            campos_form = r.get("campos_form", {})
            if campos_form:
                for campo, datos in campos_form.items():
                    icono = "✅" if datos.get("valido") else "⚠"
                    st.markdown(f"**{campo}**")
                    st.code(datos.get("texto_norm", "—"))
                    st.caption(f"{icono} Confianza OCR: {datos.get('confianza_ocr', 0):.1%}")
            else:
                st.info("Sin campos del formulario extraídos.")

    # Tab 3: Perfil de riesgo
    with tab_riesgo:
        perfil = r.get("perfil_riesgo")
        if perfil:
            nivel  = perfil.get("nivel", "?")
            ratio  = perfil.get("ratio_deuda", 0)
            color  = {"BAJO": "🟢", "MEDIO": "🟡", "ALTO": "🔴"}.get(nivel, "⚪")
            st.metric("Nivel de Riesgo", f"{color} {nivel}")
            st.metric("Ratio de Endeudamiento", f"{ratio:.1%}", help="Cuota mensual / Ingresos netos")
            st.info(perfil.get("detalle", ""))

            # Gauge visual
            st.progress(
                min(ratio, 1.0),
                text=f"Ratio deuda: {ratio:.1%} (máx legal: 35%)",
            )
        else:
            st.info("Perfil de riesgo no calculado (campos OCR insuficientes).")

        st.markdown("---")
        mrz_ok   = r.get("mrz_checkdigits_ok", True)
        cuota_ok = r.get("cuota_coherente", True)
        st.markdown(f"**Dígitos control MRZ:** {'✅ Correctos' if mrz_ok else '❌ Incorrectos'}")
        st.markdown(f"**Cuota fórmula francesa:** {'✅ Coherente' if cuota_ok else '⚠ No coincide'}")

    # Tab 4: Autenticidad
    with tab_auth:
        col_ad, col_af = st.columns(2)
        for col, titulo, datos in [
            (col_ad, "DNI",       r.get("autenticidad_dni")),
            (col_af, "Formulario", r.get("autenticidad_form")),
        ]:
            with col:
                st.subheader(titulo)
                if datos:
                    etq   = datos.get("etiqueta", "?")
                    conf  = datos.get("confianza", 0)
                    icono = "✅" if etq == "LEGÍTIMO" else "❌"
                    st.metric("Clasificación", f"{icono} {etq}")
                    st.metric("Confianza", f"{conf:.1%}")
                    st.progress(conf)
                else:
                    st.info("Clasificador no disponible o deshabilitado.")

    # Tab 5: JSON raw
    with tab_raw:
        st.json(r)
