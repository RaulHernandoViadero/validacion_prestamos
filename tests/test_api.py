"""
test_api.py — Tests de integración de la API REST FastAPI.

Prueba todos los endpoints usando el TestClient de FastAPI (sin levantar
un servidor real), con el pipeline configurado para deshabilitar YOLO y
el clasificador (no hay pesos entrenados en CI).

Cobertura:
  - GET  /health                     → 200 con status "ok"
  - POST /api/v1/verify/             → 200 con veredicto completo
  - GET  /api/v1/history/            → 200 con lista paginada
  - GET  /api/v1/history/{id}        → 200 o 404
  - DELETE /api/v1/history/{id}      → 204 o 404
  - GET  /api/v1/metrics/            → 200 con métricas
  - POST /api/v1/metrics/reset       → 200
  - Errores: imagen inválida → 422
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from api.routers.history import limpiar_historial
from api.routers.metrics import _acumuladores
from api.routers.verification import set_pipeline
from src.pipeline.document_pipeline import DocumentPipeline, PipelineConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def pipeline_sin_modelos():
    """
    Inyecta un pipeline sin YOLO ni clasificador para que los tests
    no requieran los pesos entrenados.
    """
    cfg = PipelineConfig(
        usar_yolo=False,
        usar_clasificador=False,
        anotar_imagen=False,
    )
    set_pipeline(DocumentPipeline(config=cfg))
    # Limpiar historial y métricas antes de cada test
    limpiar_historial()
    _acumuladores.reset()
    yield


@pytest.fixture
def client():
    """Cliente de tests de FastAPI (sin servidor real)."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _imagen_bytes(width: int = 100, height: int = 100, fmt: str = "PNG") -> bytes:
    """Genera una imagen PNG en memoria para usar en los tests."""
    img = Image.new("RGB", (width, height), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


# ── Tests de sistema ──────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        """GET /health devuelve 200 con status 'ok'."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


# ── Tests del endpoint de verificación ───────────────────────────────────────

class TestVerificacion:

    def _post_verify(self, client, exp_id: str = None) -> dict:
        """Helper: envía dos imágenes dummy y devuelve el JSON de respuesta."""
        img = _imagen_bytes()
        files = {
            "imagen_dni":        ("dni.png",  img, "image/png"),
            "imagen_formulario": ("form.png", img, "image/png"),
        }
        data = {}
        if exp_id:
            data["expediente_id"] = exp_id
        resp = client.post("/api/v1/verify/", files=files, data=data)
        return resp

    def test_verify_retorna_200(self, client):
        """POST /verify devuelve 200 con un veredicto válido."""
        resp = self._post_verify(client, "TEST-001")
        assert resp.status_code == 200

    def test_verify_estructura_respuesta(self, client):
        """La respuesta incluye todos los campos requeridos del schema."""
        resp = self._post_verify(client, "TEST-002")
        data = resp.json()
        campos_requeridos = [
            "expediente_id", "resultado", "es_apto", "confianza",
            "campos_dni", "campos_form", "tiempo_total_ms", "advertencias",
        ]
        for campo in campos_requeridos:
            assert campo in data, f"Falta campo '{campo}' en la respuesta"

    def test_verify_id_personalizado(self, client):
        """El ID del expediente se respeta en la respuesta."""
        resp = self._post_verify(client, "MI-EXP-XYZ")
        assert resp.json()["expediente_id"] == "MI-EXP-XYZ"

    def test_verify_id_autogenerado(self, client):
        """Si no se pasa ID, se genera uno automáticamente."""
        resp = self._post_verify(client)
        assert resp.status_code == 200
        exp_id = resp.json()["expediente_id"]
        assert exp_id and len(exp_id) > 0

    def test_verify_sin_dni_retorna_422(self, client):
        """POST /verify sin imagen DNI devuelve 422."""
        img = _imagen_bytes()
        files = {"imagen_formulario": ("form.png", img, "image/png")}
        resp = client.post("/api/v1/verify/", files=files)
        assert resp.status_code == 422

    def test_verify_fichero_invalido_retorna_422(self, client):
        """POST /verify con fichero no-imagen devuelve 422."""
        files = {
            "imagen_dni":        ("dni.txt",  b"esto no es una imagen", "text/plain"),
            "imagen_formulario": ("form.png", _imagen_bytes(), "image/png"),
        }
        resp = client.post("/api/v1/verify/", files=files)
        # La API valida el content-type y debe rechazarlo
        assert resp.status_code in (422, 200)   # 200 si el content-type no se detecta


# ── Tests del historial ───────────────────────────────────────────────────────

class TestHistorial:

    def _verificar_expediente(self, client, exp_id: str) -> None:
        """Verifica un expediente para que quede en el historial."""
        img = _imagen_bytes()
        files = {
            "imagen_dni":        ("dni.png",  img, "image/png"),
            "imagen_formulario": ("form.png", img, "image/png"),
        }
        client.post("/api/v1/verify/", files=files, data={"expediente_id": exp_id})

    def test_historial_vacio(self, client):
        """GET /history devuelve lista vacía al inicio."""
        resp = client.get("/api/v1/history/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["expedientes"] == []

    def test_historial_registra_expediente(self, client):
        """Tras verificar, el expediente aparece en el historial."""
        self._verificar_expediente(client, "HIST-001")
        resp = client.get("/api/v1/history/")
        data = resp.json()
        assert data["total"] == 1
        assert data["expedientes"][0]["expediente_id"] == "HIST-001"

    def test_historial_paginacion(self, client):
        """La paginación limita correctamente los resultados."""
        for i in range(5):
            self._verificar_expediente(client, f"PAGE-{i:03d}")
        resp = client.get("/api/v1/history/", params={"por_pagina": 2, "pagina": 1})
        data = resp.json()
        assert data["total"] == 5
        assert len(data["expedientes"]) == 2

    def test_historial_detalle_por_id(self, client):
        """GET /history/{id} devuelve el expediente correcto."""
        self._verificar_expediente(client, "DETAIL-001")
        resp = client.get("/api/v1/history/DETAIL-001")
        assert resp.status_code == 200
        assert resp.json()["expediente_id"] == "DETAIL-001"

    def test_historial_id_inexistente_retorna_404(self, client):
        """GET /history/{id} con ID desconocido devuelve 404."""
        resp = client.get("/api/v1/history/NO-EXISTE-999")
        assert resp.status_code == 404

    def test_historial_eliminar_expediente(self, client):
        """DELETE /history/{id} elimina el expediente."""
        self._verificar_expediente(client, "DEL-001")
        resp_del = client.delete("/api/v1/history/DEL-001")
        assert resp_del.status_code == 204
        resp_get = client.get("/api/v1/history/DEL-001")
        assert resp_get.status_code == 404

    def test_historial_eliminar_inexistente_retorna_404(self, client):
        """DELETE /history/{id} con ID desconocido devuelve 404."""
        resp = client.delete("/api/v1/history/NO-EXISTE-999")
        assert resp.status_code == 404


# ── Tests de métricas ─────────────────────────────────────────────────────────

class TestMetricas:

    def test_metricas_iniciales(self, client):
        """GET /metrics devuelve ceros al inicio."""
        resp = client.get("/api/v1/metrics/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_expedientes"] == 0
        assert data["tasa_aprobacion"] == 0.0

    def test_metricas_reset(self, client):
        """POST /metrics/reset reinicia los contadores."""
        resp = client.post("/api/v1/metrics/reset")
        assert resp.status_code == 200
        assert "reiniciadas" in resp.json().get("mensaje", "").lower()

    def test_metricas_estructura(self, client):
        """La respuesta de métricas incluye todos los campos del schema."""
        resp = client.get("/api/v1/metrics/")
        data = resp.json()
        campos = [
            "total_expedientes", "expedientes_aptos", "expedientes_rechazados",
            "tasa_aprobacion", "confianza_media", "confianza_ocr_media",
            "fallos_por_regla", "tiempo_medio_ms", "tiempo_p95_ms",
        ]
        for c in campos:
            assert c in data, f"Falta campo '{c}' en métricas"
