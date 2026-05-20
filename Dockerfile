FROM python:3.11-slim

WORKDIR /app

# Install locked dependencies first (cached layer unless requirements change)
COPY requirements.txt requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

# Copy application code
COPY . .

# Create data directories for SQLite
# /data is the Railway persistent volume mount point — Railway mounts
# volumes as root, so we run as root (container is already sandboxed).
RUN mkdir -p /app/data /app/logs /app/reports /data \
    && adduser --disabled-password --gecos "" --uid 1000 bot \
    && chown -R bot:bot /app

# Railway persistent volumes often mount /data as root. Keep root as the
# default for Railway compatibility, but allow hardened runtimes to build with:
#   docker build --build-arg RUN_AS_USER=bot .
ARG RUN_AS_USER=root
USER ${RUN_AS_USER}

# Health check — hits the /api/health endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health')" || exit 1

EXPOSE 8080

# Default to JSON logging in container
ENV LOG_FORMAT=json
ENV PORT=8080
# CRITICAL: Disable Python stdout/stderr buffering so logs appear in real-time
# Without this, Railway/Docker log collectors may miss log output entirely
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "main.py"]
