FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

# System deps for fonts/locales if needed
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Cloud Run provides PORT; default to 8080
ENV PORT=8080
# ---- App configuration (override in Cloud Run UI if needed) ----
ENV GCP_PROJECT=ops-sensei-ai-assistant
ENV REGION=us-central1
ENV GCS_BUCKET=ops-sensei-docs-assistant
ENV EMBED_MODEL=text-embedding-004
ENV CHAT_MODEL=gemini-1.5-pro
ENV COLL_DOCS=db-assistant-docs
ENV COLL_CHUNKS=db-assistant-chunks
ENV COLL_SESSIONS=db-assistant-sessions
# ---------------------------------------------------------------

EXPOSE 8080
CMD ["sh","-c","python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
