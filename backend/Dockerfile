FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --all-extras --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --all-extras

ENV PATH="/app/.venv/bin:$PATH" DOCUCHAT_DATA_DIR=/data
RUN useradd --create-home app && mkdir -p /data && chown -R app /app /data
USER app
# Bake the ONNX models into the image so containers start without downloading.
RUN docuchat download-models

EXPOSE 8000
VOLUME ["/data"]
HEALTHCHECK --interval=10s --timeout=3s --retries=10 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["docuchat", "serve", "--host", "0.0.0.0", "--port", "8000"]
