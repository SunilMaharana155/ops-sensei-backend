FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
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
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
