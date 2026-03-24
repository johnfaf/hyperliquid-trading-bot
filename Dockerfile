FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory for SQLite
RUN mkdir -p /app/data /app/logs /app/reports

# Non-root user for security
RUN useradd -m botuser && chown -R botuser:botuser /app
USER botuser

# Health check — hits the /api/health endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health')" || exit 1

EXPOSE 8080

# Default to JSON logging in container
ENV LOG_FORMAT=json
ENV PORT=8080

CMD ["python", "main.py"]
