import uuid
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from typing import List, Optional
import sqlalchemy
import os
import pickle
from contextlib import asynccontextmanager

# Import logic from sibling modules
from app.database import init_db, get_db, SessionLocal, DBMessage
from app.encryption import encrypt_text, decrypt_text

# For model caching
MODEL_CACHE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    print("Initializing Data Models...")
    
    # Attempt to load ML models
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'model', 'saved_model.pkl')
    vec_path = os.path.join(base_dir, 'model', 'vectorizer.pkl')
    
    if os.path.exists(model_path) and os.path.exists(vec_path):
        try:
            with open(model_path, 'rb') as f:
                MODEL_CACHE['model'] = pickle.load(f)
            with open(vec_path, 'rb') as f:
                MODEL_CACHE['vectorizer'] = pickle.load(f)
            print("ML Model and Vectorizer loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load models. Exception: {e}")
    else:
        print("Warning: Model files not found. Please run the training script inside /model.")
        
    yield
    # Shutdown (cleanup if necessary)
    MODEL_CACHE.clear()

app = FastAPI(
    title="Emotion-Aware Encryption API",
    description="A system combining NLP-based emotion detection and secure AES encryption.",
    version="1.0.0",
    lifespan=lifespan
)

# ── CORS — allow browser requests from any origin (localhost dev) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve the frontend static folder ──
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# --- Pydantic Models ---
class TextSubmission(BaseModel):
    text: str = Field(..., max_length=2000, description="Plaintext input for analysis and encryption")

class SubmissionResponse(BaseModel):
    emotion: str
    confidence: float
    encrypted_text: str

class DecryptionRequest(BaseModel):
    encrypted_text: str

class DecryptionResponse(BaseModel):
    original_text: str
    emotion: str
    confidence: float

class StatResponse(BaseModel):
    emotion: str
    count: int

# --- Helper Functions ---
def predict_emotion(text: str):
    """
    Given a raw string, returns (top_emotion, confidence_score).
    Provides fallback behavior missing model contexts.
    """
    model = MODEL_CACHE.get('model')
    vectorizer = MODEL_CACHE.get('vectorizer')
    
    if not model or not vectorizer:
        return "unknown (model offline)", 0.0
        
    # Transform text to feature array
    X = vectorizer.transform([text])
    
    # Get probabilities
    probs = model.predict_proba(X)[0]
    
    # Find highest probability index
    top_class_idx = probs.argmax()
    
    emotion = model.classes_[top_class_idx]
    confidence = float(probs[top_class_idx])
    
    return emotion, confidence

# --- API Endpoints ---
@app.post("/submit", response_model=SubmissionResponse)
def submit_text(payload: TextSubmission, db: sqlalchemy.orm.Session = Depends(get_db)):
    """
    1. Analyzes text for emotion.
    2. Encrypts the raw text safely using AES.
    3. Stores metadata + cipher blob in SQLite db.
    """
    try:
        emotion, confidence = predict_emotion(payload.text)
        enc_text = encrypt_text(payload.text)
        
        # Build strict record model 
        new_record = DBMessage(
            encrypted_text=enc_text,
            emotion=emotion,
            confidence=confidence
        )
        
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        
        return SubmissionResponse(
            emotion=emotion,
            confidence=round(confidence, 4),
            encrypted_text=enc_text
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decrypt", response_model=DecryptionResponse)
def decrypt_message(payload: DecryptionRequest, db: sqlalchemy.orm.Session = Depends(get_db)):
    """
    1. Look up the message based on encrypted ciphertext in DB
    2. Reverse encrypt string to standard structure
    """
    # Fetch record matching that ciphertext
    record = db.query(DBMessage).filter(DBMessage.encrypted_text == payload.encrypted_text).first()
    
    if not record:
        raise HTTPException(status_code=404, detail="Encrypted payload not found in system")
        
    try:
        original = decrypt_text(payload.encrypted_text)
        
        return DecryptionResponse(
            original_text=original,
            emotion=record.emotion,
            confidence=record.confidence
        )
    except Exception as e:
         raise HTTPException(status_code=400, detail=f"Decryption failed: {str(e)}")


@app.get("/stats", response_model=List[StatResponse])
def get_stats(db: sqlalchemy.orm.Session = Depends(get_db)):
    """
    Return distribution of emotions analyzed in system securely.
    """
    results = db.query(DBMessage.emotion, sqlalchemy.func.count(DBMessage.id))\
        .group_by(DBMessage.emotion)\
        .all()
        
    return [{"emotion": r[0], "count": r[1]} for r in results]

@app.get("/health")
def health_check():
    """
    Render / uptime-robot health probe. Must return 2xx.
    """
    return {"status": "ok"}


@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    """
    Serves the frontend UI if available; otherwise returns a JSON welcome message.
    Handles HEAD requests so Render's port-scanner sees a 200, not a 405.
    """
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {
        "message": "Welcome to the Emotion-Aware Encryption System API.",
        "docs_url": "/docs",
        "ui_url": "/ui"
    }
