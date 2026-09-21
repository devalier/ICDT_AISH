# EUIBA AI Safety Harness — production image.
# Build: docker build -t aish:latest .
FROM python:3.13-slim AS base

# Fail fast, no .pyc, unbuffered logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------- build stage
FROM base AS build
RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential libffi-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt

# --------------------------------------------------------------- runtime stage
# Compilers and headers stay in the build stage; the runtime image has neither.
FROM base AS runtime

RUN apt-get update \
 && apt-get install --no-install-recommends -y tini \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin aish

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY --chown=aish:aish aish/ ./aish/
COPY --chown=aish:aish harness/ ./harness/
COPY --chown=aish:aish packs/ ./packs/
COPY --chown=aish:aish templates/ ./templates/
COPY --chown=aish:aish static/ ./static/
COPY --chown=aish:aish scripts/ ./scripts/
COPY --chown=aish:aish docs/ ./docs/

# Writable state lives in one volume; the rest of the filesystem can be read-only.
RUN mkdir -p /data && chown aish:aish /data
ENV AISH_DATABASE_URL=sqlite:////data/aish.db \
    AISH_ENV=production
VOLUME ["/data"]

USER aish
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"

ENTRYPOINT ["/usr/bin/tini", "--"]
# TLS terminates at the reverse proxy; forwarded headers are trusted from it only.
CMD ["uvicorn", "aish.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*", \
     "--no-server-header", "--workers", "2"]
