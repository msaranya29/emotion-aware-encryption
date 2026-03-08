import uuid
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import sqlalchemy
import os
import re
import pickle
import json
from contextlib import asynccontextmanager

# Import logic from sibling modules
from app.database import init_db, get_db, SessionLocal, DBMessage
from app.encryption import encrypt_text, decrypt_text

# For model caching
MODEL_CACHE = {}

# ── Emotion metadata: emoji + colour ──
EMOTION_META = {
    "joy":        {"emoji": "😄", "color": "#f59e0b"},
    "sadness":    {"emoji": "😢", "color": "#60a5fa"},
    "anger":      {"emoji": "😠", "color": "#f87171"},
    "anxiety":    {"emoji": "😰", "color": "#a78bfa"},
    "fear":       {"emoji": "😨", "color": "#c084fc"},
    "surprise":   {"emoji": "😲", "color": "#34d399"},
    "disgust":    {"emoji": "🤢", "color": "#4ade80"},
    "excitement": {"emoji": "🤩", "color": "#fb923c"},
    "love":       {"emoji": "❤️",  "color": "#f472b6"},
    "neutral":    {"emoji": "😐", "color": "#94a3b8"},
}

# ── Keyword lexicon for each emotion (word → weight boost) ──
EMOTION_KEYWORDS: Dict[str, Dict[str, float]] = {
    "joy": {
        "happy": 0.35, "happiness": 0.35, "joyful": 0.40, "joy": 0.40,
        "ecstatic": 0.45, "wonderful": 0.30, "great": 0.20, "amazing": 0.30,
        "love": 0.20, "glad": 0.30, "delighted": 0.35, "bliss": 0.40,
        "grateful": 0.25, "beautiful": 0.20, "smile": 0.25, "laugh": 0.25,
        "cheerful": 0.35, "thrilled": 0.38, "overjoyed": 0.45, "wonderful": 0.30,
        "good": 0.15, "positive": 0.20, "content": 0.25, "peaceful": 0.25,
        "lucky": 0.25, "fantastic": 0.35, "brilliant": 0.25,
    },
    "sadness": {
        "sad": 0.40, "sadness": 0.40, "depressed": 0.45, "depression": 0.45,
        "cry": 0.40, "crying": 0.40, "tears": 0.35, "grief": 0.45,
        "grieve": 0.45, "heartbroken": 0.50, "miserable": 0.45,
        "lonely": 0.40, "alone": 0.25, "empty": 0.35, "hopeless": 0.45,
        "disappointed": 0.40, "disappointment": 0.40, "devastated": 0.45,
        "unhappy": 0.35, "hurt": 0.30, "pain": 0.25, "suffering": 0.35,
        "miss": 0.25, "lost": 0.20, "gloomy": 0.35, "dark": 0.20,
        "worthless": 0.45, "failed": 0.30, "failure": 0.30,
    },
    "anger": {
        "angry": 0.40, "anger": 0.40, "furious": 0.50, "fury": 0.50,
        "rage": 0.50, "livid": 0.50, "enraged": 0.50, "mad": 0.35,
        "hate": 0.35, "frustrated": 0.35, "frustration": 0.35,
        "annoyed": 0.30, "irritated": 0.30, "outraged": 0.45,
        "infuriated": 0.45, "seething": 0.45, "boiling": 0.40,
        "bitter": 0.30, "resentful": 0.35, "hostile": 0.35,
        "disgusted": 0.30, "unfair": 0.25, "injustice": 0.35,
    },
    "anxiety": {
        "anxious": 0.45, "anxiety": 0.45, "worried": 0.40, "worry": 0.40,
        "nervous": 0.40, "stress": 0.35, "stressed": 0.40, "tense": 0.35,
        "panic": 0.45, "overwhelmed": 0.40, "overthink": 0.40,
        "overthinking": 0.40, "dread": 0.40, "dreading": 0.40,
        "uncertain": 0.30, "uneasy": 0.35, "restless": 0.30,
        "deadline": 0.25, "pressure": 0.30, "burden": 0.30,
        "apprehensive": 0.40, "frantic": 0.40, "trepidation": 0.40,
    },
    "fear": {
        "fear": 0.45, "afraid": 0.45, "scared": 0.45, "terrified": 0.55,
        "terror": 0.55, "frightened": 0.50, "fright": 0.50, "horror": 0.50,
        "nightmare": 0.40, "phobia": 0.45, "dread": 0.40, "shaking": 0.30,
        "trembling": 0.35, "paralyzed": 0.40, "creep": 0.25,
    },
    "surprise": {
        "surprised": 0.45, "surprise": 0.45, "amazed": 0.40, "amazement": 0.40,
        "astonished": 0.45, "shocked": 0.40, "unbelievable": 0.40,
        "unexpected": 0.35, "wow": 0.40, "stunned": 0.45, "speechless": 0.40,
        "disbelief": 0.40, "jaw": 0.25, "caught off guard": 0.45,
        "out of nowhere": 0.40, "blindsided": 0.45,
    },
    "disgust": {
        "disgusted": 0.45, "disgusting": 0.45, "disgust": 0.45, "gross": 0.40,
        "repulsed": 0.50, "revolting": 0.50, "nauseated": 0.50, "sick": 0.30,
        "repulsive": 0.45, "appalled": 0.40, "vile": 0.45, "yuck": 0.35,
        "offensive": 0.30, "filthy": 0.35,
    },
    "excitement": {
        "excited": 0.45, "excitement": 0.45, "thrilled": 0.40, "thrill": 0.40,
        "pumped": 0.40, "energized": 0.40, "can't wait": 0.40, "cannot wait": 0.40,
        "anticipation": 0.35, "eager": 0.35, "enthusiastic": 0.40,
        "stoked": 0.40, "buzzing": 0.35, "ecstatic": 0.35, "hyped": 0.40,
        "exhilarated": 0.45, "looking forward": 0.35, "countdown": 0.30,
    },
    "love": {
        "love": 0.45, "loved": 0.40, "loving": 0.40, "adore": 0.45,
        "adoring": 0.45, "cherish": 0.45, "affection": 0.40, "devoted": 0.40,
        "devotion": 0.40, "passion": 0.35, "passionate": 0.35, "romance": 0.35,
        "romantic": 0.35, "caring": 0.30, "bond": 0.25, "connection": 0.25,
        "heart": 0.25, "tender": 0.30, "warmth": 0.25, "beloved": 0.45,
    },
    "neutral": {
        "okay": 0.20, "fine": 0.20, "normal": 0.20, "routine": 0.25,
        "regular": 0.20, "nothing": 0.20, "scheduled": 0.20,
        "completed": 0.20, "done": 0.15, "just": 0.10,
    },
}


def keyword_emotion_scores(text: str) -> Dict[str, float]:
    """
    Compute keyword-based emotion scores using a lexicon.
    Returns a dict of {emotion: raw_score} normalized to [0,1].
    Multi-word phrases are matched first; single words second.
    """
    lower = text.lower()
    raw: Dict[str, float] = {e: 0.0 for e in EMOTION_KEYWORDS}

    for emotion, lexicon in EMOTION_KEYWORDS.items():
        score = 0.0
        for phrase, weight in lexicon.items():
            if phrase in lower:
                # Count occurrences
                count = lower.count(phrase)
                score += weight * count
        raw[emotion] = score

    # Softmax-style normalization for output scores
    total = sum(raw.values())
    if total == 0:
        return {e: 0.0 for e in raw}
    return {e: v / total for e, v in raw.items()}


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
    MODEL_CACHE.clear()


app = FastAPI(
    title="mini Emotion Cipher API",
    description=(
        "Emotion-Aware Encryption: NLP emotion intelligence meets AES-256. "
        "Detects multiple emotions from text and encrypts the message while "
        "preserving its emotional signature."
    ),
    version="2.0.0",
    lifespan=lifespan
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve the frontend static folder ──
FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# --- Pydantic Models ---
class TextSubmission(BaseModel):
    text: str = Field(..., max_length=2000,
                      description="Plaintext input for analysis and encryption")

class EmotionScore(BaseModel):
    emotion: str
    confidence: float
    emoji: str
    color: str

class SubmissionResponse(BaseModel):
    primary_emotion: str
    primary_emoji: str
    primary_confidence: float
    all_emotions: List[EmotionScore]
    encrypted_text: str
    emotional_signature: str

class DecryptionRequest(BaseModel):
    encrypted_text: str

class DecryptionResponse(BaseModel):
    original_text: str
    primary_emotion: str
    primary_emoji: str
    primary_confidence: float
    all_emotions: List[EmotionScore]
    emotional_signature: str

class StatResponse(BaseModel):
    emotion: str
    count: int
    emoji: str
    color: str


# --- Core Emotion Engine ---
def predict_emotions(text: str):
    """
    Hybrid emotion detection:
      1. Keyword lexicon scan → reliable multi-emotion signals
      2. ML model (TF-IDF + LogReg) → statistical context
    Blends both with configurable weights (70% keyword, 30% ML).
    Returns (List[EmotionScore], emotional_signature_string).
    """
    # ── Step 1: Keyword scores ──
    kw_scores = keyword_emotion_scores(text)

    # ── Step 2: ML model scores ──
    ml_scores: Dict[str, float] = {e: 0.0 for e in EMOTION_KEYWORDS}
    model = MODEL_CACHE.get('model')
    vectorizer = MODEL_CACHE.get('vectorizer')
    if model and vectorizer:
        X = vectorizer.transform([text])
        probs = model.predict_proba(X)[0]
        for idx, prob in enumerate(probs):
            emo = model.classes_[idx]
            if emo in ml_scores:
                ml_scores[emo] = float(prob)

    # ── Step 3: Blend (70% kw, 30% ml) ──
    KW_WEIGHT = 0.70
    ML_WEIGHT = 0.30
    blended: Dict[str, float] = {}
    for emotion in EMOTION_KEYWORDS:
        blended[emotion] = KW_WEIGHT * kw_scores.get(emotion, 0.0) \
                         + ML_WEIGHT * ml_scores.get(emotion, 0.0)

    # ── Step 4: Re-normalise so top probabilities make sense ──
    total = sum(blended.values())
    if total > 0:
        blended = {e: v / total for e, v in blended.items()}

    # If keyword engine found nothing (no known words), fall back to pure ML
    kw_total = sum(kw_scores.values())
    if kw_total == 0 and model and vectorizer:
        blended = ml_scores.copy()

    # ── Step 5: Pick significant emotions ──
    sorted_scores = sorted(blended.items(), key=lambda x: -x[1])

    # Include emotions where confidence ≥ 8% OR top-2 always included
    THRESHOLD = 0.08
    significant = [(e, c) for e, c in sorted_scores if c >= THRESHOLD]
    if not significant:
        significant = sorted_scores[:1]
    elif len(significant) > 3:
        significant = significant[:3]   # Cap at 3 for readability

    # Build EmotionScore objects
    result = []
    for emotion, conf in significant:
        meta = EMOTION_META.get(emotion, {"emoji": "❓", "color": "#64748b"})
        result.append(EmotionScore(
            emotion=emotion,
            confidence=round(conf, 4),
            emoji=meta["emoji"],
            color=meta["color"]
        ))

    # Emotional signature string: "Joy + Anxiety"
    sig_parts = [s.emotion.capitalize() for s in result]
    emotional_signature = " + ".join(sig_parts)

    return result, emotional_signature


def build_encrypted_payload(enc_text: str, emotional_signature: str) -> str:
    """
    Wraps the AES ciphertext with an emotion-derived hash prefix.
    Format: EMOSIG:<8-char-b64-hash>.<AES_ciphertext>

    This lets AI systems identify the emotional category from the prefix
    WITHOUT decrypting the text — privacy + empathy preserved.
    """
    import hashlib, base64
    sig_hash = hashlib.sha256(emotional_signature.encode()).digest()[:8]
    sig_encoded = base64.urlsafe_b64encode(sig_hash).decode().rstrip('=')
    return f"EMOSIG:{sig_encoded}.{enc_text}"


def parse_encrypted_payload(payload: str):
    """
    Parse the EMOSIG wrapped payload.
    Returns (sig_prefix, aes_ciphertext).
    Falls back to treating the whole payload as raw AES for legacy records.
    """
    if payload.startswith("EMOSIG:") and "." in payload:
        _, rest = payload.split(":", 1)
        sig_encoded, aes_text = rest.split(".", 1)
        return sig_encoded, aes_text
    return None, payload


# --- API Endpoints ---
@app.post("/submit", response_model=SubmissionResponse)
def submit_text(payload: TextSubmission,
                db: sqlalchemy.orm.Session = Depends(get_db)):
    """
    1. Multi-emotion analysis (hybrid keyword + ML).
    2. AES-256 encryption of original text.
    3. EMOSIG wrapping to embed emotional signature.
    4. Store metadata in DB (no plaintext ever stored).
    """
    try:
        emotions, emotional_signature = predict_emotions(payload.text)

        enc_text = encrypt_text(payload.text)
        wrapped = build_encrypted_payload(enc_text, emotional_signature)

        primary = emotions[0]

        emotions_json = json.dumps([
            {"emotion": e.emotion, "confidence": e.confidence,
             "emoji": e.emoji, "color": e.color}
            for e in emotions
        ])

        new_record = DBMessage(
            encrypted_text=wrapped,
            emotion=emotional_signature,
            confidence=primary.confidence,
            emotions_json=emotions_json
        )

        db.add(new_record)
        db.commit()
        db.refresh(new_record)

        return SubmissionResponse(
            primary_emotion=primary.emotion,
            primary_emoji=primary.emoji,
            primary_confidence=round(primary.confidence, 4),
            all_emotions=emotions,
            encrypted_text=wrapped,
            emotional_signature=emotional_signature
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/decrypt", response_model=DecryptionResponse)
def decrypt_message(payload: DecryptionRequest,
                    db: sqlalchemy.orm.Session = Depends(get_db)):
    """
    1. Parse the EMOSIG payload.
    2. Lookup record in DB for emotional metadata.
    3. AES-decrypt the ciphertext to recover original text.
    """
    sig_prefix, aes_ciphertext = parse_encrypted_payload(payload.encrypted_text)

    record = db.query(DBMessage).filter(
        DBMessage.encrypted_text == payload.encrypted_text
    ).first()

    if not record:
        raise HTTPException(
            status_code=404,
            detail="Encrypted payload not found. Only messages encrypted by this system can be decrypted."
        )

    try:
        original = decrypt_text(aes_ciphertext)

        all_emotions = []
        if record.emotions_json:
            try:
                raw = json.loads(record.emotions_json)
                all_emotions = [
                    EmotionScore(
                        emotion=e["emotion"],
                        confidence=e["confidence"],
                        emoji=e.get("emoji", "❓"),
                        color=e.get("color", "#64748b")
                    ) for e in raw
                ]
            except Exception:
                pass

        if not all_emotions:
            # Fallback for legacy records
            meta = EMOTION_META.get(
                record.emotion.lower().split(" + ")[0].strip(),
                {"emoji": "❓", "color": "#64748b"}
            )
            all_emotions = [EmotionScore(
                emotion=record.emotion,
                confidence=record.confidence,
                emoji=meta["emoji"],
                color=meta["color"]
            )]

        primary = all_emotions[0]

        return DecryptionResponse(
            original_text=original,
            primary_emotion=primary.emotion,
            primary_emoji=primary.emoji,
            primary_confidence=primary.confidence,
            all_emotions=all_emotions,
            emotional_signature=record.emotion
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Decryption failed: {str(e)}")


@app.get("/stats", response_model=List[StatResponse])
def get_stats(db: sqlalchemy.orm.Session = Depends(get_db)):
    """Emotion distribution across all stored encrypted messages."""
    results = (
        db.query(DBMessage.emotion, sqlalchemy.func.count(DBMessage.id))
        .group_by(DBMessage.emotion)
        .all()
    )

    output = []
    for emotion_sig, count in results:
        first = emotion_sig.lower().split(" + ")[0].strip()
        meta = EMOTION_META.get(first, {"emoji": "❓", "color": "#64748b"})
        output.append({
            "emotion": emotion_sig,
            "count": count,
            "emoji": meta["emoji"],
            "color": meta["color"]
        })
    return output


@app.get("/health")
def health_check():
    """Health probe for deployment platforms."""
    model_loaded = 'model' in MODEL_CACHE and 'vectorizer' in MODEL_CACHE
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "version": "2.0.0"
    }


@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    """Serve frontend or JSON welcome."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {
        "message": "Welcome to mini Emotion Cipher.",
        "docs_url": "/docs",
        "ui_url": "/ui",
        "version": "2.0.0"
    }
