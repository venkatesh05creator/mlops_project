# ============================================================================
# Traffic Accident Severity Prediction - FastAPI Serving Container
# ============================================================================

FROM python:3.11-slim as runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    APP_HOME=/app

WORKDIR ${APP_HOME}

# Install essential system dependencies (libgomp1 is required for LightGBM/XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies for serving
COPY requirements-serve.txt ${APP_HOME}/requirements-serve.txt
RUN pip install --no-cache-dir -r requirements-serve.txt

# Create non-privileged user for security
RUN groupadd -r appuser && useradd -r -g appuser -d ${APP_HOME} -s /sbin/nologin appuser

# Copy application code, configuration, and trained models
COPY config.py ${APP_HOME}/config.py
COPY src/ ${APP_HOME}/src/
COPY models/ ${APP_HOME}/models/

# Set file permissions
RUN chown -R appuser:appuser ${APP_HOME}
USER appuser

EXPOSE 8000

# Health check to ensure the serving container is active
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production ASGI server
CMD ["uvicorn", "src.ml.serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
