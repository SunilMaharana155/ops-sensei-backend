
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from google.cloud import firestore
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

# ---- Environment & configuration ----
PROJECT_ID = os.getenv("GCP_PROJECT", "ops-sensei-ai-assistant")
REGION = os.getenv("REGION", "us-central1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-1.5-pro")
COLL_DOCS = os.getenv("COLL_DOCS", "db-assistant-docs")
COLL_CHUNKS = os.getenv("COLL_CHUNKS", "db-assistant-chunks")
COLL_SESSIONS = os.getenv("COLL_SESSIONS", "db-assistant-sessions")

vertexai.init(project=PROJECT_ID, location=REGION)

db = firestore.Client()
llm = GenerativeModel(CHAT_MODEL)
embed_model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)

app = FastAPI(title="OpsSensei Backend API", version="1.0.0")

class ChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: int = 5

@app.get("/healthz")
def healthz():
    return {"status": "ok", "project": PROJECT_ID, "region": REGION}

# ---- Utility functions ----
def embed_text(text: str) -> np.ndarray:
    e = embed_model.get_embeddings([text])[0].values
    return np.array(e, dtype=float)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float((a @ b) / denom)


def retrieve(query_vec: np.ndarray, top_k: int = 5):
    """Naive retrieval by scanning Firestore. Suitable for coursework scale.
    For production, use Vertex AI Matching Engine.
    """
    chunks_ref = db.collection(COLL_CHUNKS).stream()
    scored = []
    for snap in chunks_ref:
        data = snap.to_dict()
        v = np.array(data["embedding"], dtype=float)
        score = cosine_sim(query_vec, v)
        scored.append((score, data))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:top_k]]

# ---- Chat endpoint ----
@app.post("/chat")
def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    q_vec = embed_text(req.message)
    contexts = retrieve(q_vec, req.top_k)

    if not contexts:
        contexts_text = "No relevant context found. If you can, upload or ingest documents first."
    else:
        contexts_text = "

".join(
            f"[{c['doc_id']}#{c['chunk_index']}] {c['text']}" for c in contexts
        )

    system_prompt = """You are OpsSensei, a project & docs assistant.
Answer concisely using ONLY the provided context.
Cite sources by [doc_id#chunk_index].
If uncertain, say you are uncertain and suggest next steps.
"""

    user_prompt = f"""User question: {req.message}
Relevant context:
{contexts_text}

Return markdown with a "Sources" section listing the cited chunk IDs.
"""

    resp = llm.generate_content([system_prompt, user_prompt])
    answer = resp.text

    # Persist messages under session
    sess_ref = db.collection(COLL_SESSIONS).document(req.session_id)
    sess_ref.collection("messages").add({"role": "user", "text": req.message})
    sess_ref.collection("messages").add({"role": "assistant", "text": answer})

    return {"answer": answer}
