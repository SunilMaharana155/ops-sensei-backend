
import os
import uuid
from typing import List
from google.cloud import storage, firestore
from pypdf2 import PdfReader
import vertexai
from vertexai.language_models import TextEmbeddingModel

# ---- Configuration ----
PROJECT_ID = os.getenv("GCP_PROJECT", "ops-sensei-ai-assistant")
REGION = os.getenv("REGION", "us-central1")
BUCKET = os.getenv("GCS_BUCKET", "ops-sensei-docs-assistant")
PREFIX = os.getenv("ASSISTANT_DOCS_PREFIX", "docs/")  # scan this prefix in the bucket
COLL_DOCS = os.getenv("COLL_DOCS", "db-assistant-docs")
COLL_CHUNKS = os.getenv("COLL_CHUNKS", "db-assistant-chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")

vertexai.init(project=PROJECT_ID, location=REGION)

storage_client = storage.Client()
db = firestore.Client()
embed_model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)

# ---- Helpers ----
def chunk_text(text: str, max_chars=1500, overlap=200) -> List[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+max_chars]
        if chunk.strip():
            chunks.append(chunk)
        i += max_chars - overlap
    return chunks


def embed_chunks(chunks: List[str]):
    out = []
    for i in range(0, len(chunks), 16):
        batch = chunks[i:i+16]
        res = embed_model.get_embeddings(batch)
        out.extend([e.values for e in res])
    return out


def extract_pdf_text(local_path: str) -> str:
    reader = PdfReader(local_path)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "
".join(pages)


def ingest_blob(blob: storage.Blob):
    # Download to /tmp and process
    local_path = f"/tmp/{os.path.basename(blob.name)}"
    os.makedirs("/tmp", exist_ok=True)
    blob.download_to_filename(local_path)

    title = os.path.basename(blob.name)
    text = extract_pdf_text(local_path)

    if not text.strip():
        print(f"[WARN] No text extracted from {blob.name}")
        return None

    chunks = chunk_text(text)
    vectors = embed_chunks(chunks)

    doc_id = str(uuid.uuid4())
    db.collection(COLL_DOCS).document(doc_id).set({
        "title": title,
        "gcs_uri": f"gs://{BUCKET}/{blob.name}",
        "num_chunks": len(chunks),
        "source": "gcs",
    })

    for i, (c, v) in enumerate(zip(chunks, vectors)):
        db.collection(COLL_CHUNKS).add({
            "doc_id": doc_id,
            "chunk_index": i,
            "text": c,
            "embedding": v,
        })

    print(f"[OK] Ingested {blob.name} -> doc_id={doc_id}, chunks={len(chunks)}")
    return {"doc_id": doc_id, "chunks": len(chunks)}


def ingest_bucket(bucket_name: str, prefix: str = "docs/"):
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        print(f"[INFO] No files found in gs://{bucket_name}/{prefix}")
        return []

    results = []
    for b in blobs:
        # Only process PDFs for now
        if not b.name.lower().endswith('.pdf'):
            print(f"[SKIP] Not a PDF: {b.name}")
            continue
        res = ingest_blob(b)
        if res:
            results.append(res)
    return results


if __name__ == "__main__":
    print(f"[START] Ingesting from gs://{BUCKET}/{PREFIX} (project={PROJECT_ID}, region={REGION})")
    out = ingest_bucket(BUCKET, PREFIX)
    print({"ingested": out})
