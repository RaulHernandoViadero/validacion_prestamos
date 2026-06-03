# ── Dockerfile de la API REST de Verificación Documental ─────────────────────
#
# Imagen multi-stage:
#   builder  — instala dependencias Python en un virtualenv limpio
#   runtime  — imagen mínima con solo lo necesario para ejecutar la API
#
# Construcción:
#   docker build -t verificacion-api:latest -f api/Dockerfile .
#
# Ejecución:
#   docker run -p 8000:8000 verificacion-api:latest
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Variables de entorno para evitar ficheros .pyc y buffering en logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Dependencias del sistema necesarias para compilar algunas librerías Python
# Nota: libgl1-mesa-glx fue renombrado a libgl1 en Debian trixie+
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear virtualenv aislado
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Dependencias de sistema en tiempo de ejecución (solo las de runtime)
# Nota: libgl1-mesa-glx fue renombrado a libgl1 en Debian trixie+
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar el virtualenv del builder
COPY --from=builder /opt/venv /opt/venv

# Crear usuario no-root con home directory real (EasyOCR descarga modelos en ~)
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --home /home/appuser appuser && \
    mkdir -p /home/appuser/.EasyOCR && \
    chown -R appuser:appgroup /home/appuser

WORKDIR /app
ENV HOME=/home/appuser

# Copiar solo el código fuente necesario (no tests, no notebooks)
COPY src/       src/
COPY api/       api/
COPY weights/   weights/

# Cambiar propietario de los ficheros al usuario de la app
RUN chown -R appuser:appgroup /app

USER appuser

# Puerto de la API
EXPOSE 8000

# Healthcheck: el endpoint /health devuelve 200 si la API está operativa
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Comando de arranque
# --workers 2: suficiente para el TFM; ajustar según núcleos en producción
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
