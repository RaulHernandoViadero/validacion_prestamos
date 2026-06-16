# ── Dockerfile del Frontend Streamlit ────────────────────────────────────────
#
# Construcción:
#   docker build -t verificacion-frontend:latest -f app/Dockerfile .
#
# Ejecución (standalone, apunta a API local):
#   docker run -p 8501:8501 -e API_URL=http://localhost:8000 verificacion-frontend:latest
#
# En docker-compose la variable API_URL se inyecta automáticamente.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias mínimas del frontend (sin torch, ultralytics, easyocr)
RUN pip install --upgrade pip && pip install \
    streamlit==1.41.1 \
    httpx==0.28.1 \
    Pillow==11.0.0

# Código del frontend
COPY app/ app/

# Usuario no-root
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
