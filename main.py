import os
import sys
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger("opsensei")

# --- Config (from env) ---
PROJECT_ID = os.getenv("GCP_PROJECT", "ops-sensei-ai-assistant")
REGION = os.getenv("REGION", "us-central1")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "text-embedding-004")
CHAT_MODEL_NAME  = os.getenv("CHAT_MODEL",  "gemini-1.5-pro")
COLL_DOCS        = os.getenv("COLL_DOCS", "db-assistant-docs")
COLL_CHUNKS      = os.getenv("COLL_CHUNKS","db-assistant-chunks")
COLL_SESSIONS    = os.getenv("COLL_SESSIONS","db-assistant-sessions")

def create_app() -> FastAPI:
    """
    Factory that ALWAYS returns an app so Cloud Run can bind to $PORT.
    All heavy/optional imports are deferred inside request handlers.
    """
    app = FastAPI(title="OpsSensei Backend API", version="1.2.0")

    class ChatRequest(BaseModel):
        session_id: str
        message: str
        top_k: int = 5

    @app.get("/")
    @app.get("/healthz")
    def healthz():
        status = {"status": "ok", "project": PROJECT_ID, "region": REGION}
        # Try Firestore (optional)
        try:
            from google.cloud import firestore
            firestore.Client()  # just to validate import/ADC
            status["firestore"] = "ok"
        except Exception as e:
            status["firestore"] = f"error: {e}"
        # Try Vertex AI (optional)
        try:
            import vertexai
            vertexai.init(project=PROJECT_ID, location=REGION)
            status["vertex_ai"] = "ok"
        except Exception as e:
            status["vertex_ai"] = f"error: {e}"
        return status

    def embed_text(text: str) -> np.ndarray:
        import vertexai
        from vertexai.language_models import TextEmbeddingModel
        vertexai.init(project=PROJECT_ID, location=REGION)
        model = TextEmbeddingModel.from_pretrained(EMBED_MODEL_NAME)
        e = model.get_embeddings([text])[0].values
        return np.array(e, dtype=float)

    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        import numpy as _np
        denom = (_np.linalg.norm(a) * _np.linalg.norm(b)) + 1e-9
        return float((a @ b) / denom)

    def retrieve(query_vec: np.ndarray, top_k: int = 5):
        from google.cloud import firestore
        db = firestore.Client()
        scored = []
        for snap in db.collection(COLL_CHUNKS).stream():
            d = snap.to_dict()
            v = np.array(d.get("embedding", []), dtype=float)
            if v.size == 0:
                continue
            score = cosine_sim(query_vec, v)
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for s, d in scored[:top_k]]

    @app.post("/chat")
    def chat(req: ChatRequest):
        if not req.message.strip():
            raise HTTPException(status_code=400, detail="Empty message")
        try:
            q_vec = embed_text(req.message)
            ctxs = retrieve(q_vec, req.top_k)
            if not ctxs:
                contexts_text = ("No relevant context found. "
                                 "Upload/ingest documents first.")
            else:
                contexts_text = "\n\n".join(
                    f"[{c['doc_id']}#{c['chunk_index']}] {c['text']}" for c in ctxs
                )

            import vertexai
            from vertexai.generative_models import GenerativeModel
            vertexai.init(project=PROJECT_ID, location=REGION)
            llm = GenerativeModel(CHAT_MODEL_NAME)

            system_prompt = (
                "You are OpsSensei, a project & docs assistant.\n"
                "Answer concisely using ONLY the provided context.\n"
                "Cite sources by [doc_id#chunk_index].\n"
                "If uncertain, say you are uncertain and suggest next steps.\n"
            )
            user_prompt = (
                f"User question: {req.message}\n"
                f"Relevant context:\n{contexts_text}\n\n"
                "Return markdown with a \"Sources\" section listing the cited chunk IDs.\n"
            )
            resp = llm.generate_content([system_prompt, user_prompt])
            answer = resp.text

            # Persist messages
            from google.cloud import firestore
            db = firestore.Client()
            sess_ref = db.collection(COLL_SESSIONS).document(req.session_id)
            sess_ref.collection("messages").add({"role": "user", "text": req.message})
            sess_ref.collection("messages").add({"role": "assistant", "text": answer})

            return {"answer": answer}
        except Exception as e:
            log.exception("/chat failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return app

# Uvicorn will import `main:app`
app = create_app()
``