FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    PORT=8000

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY --chown=appuser:appuser data ./data
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser static ./static
COPY --chown=appuser:appuser web_app.py ./web_app.py

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/api/health/ready', timeout=4)" || exit 1

CMD ["sh", "-c", "exec python -m uvicorn web_app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
