# mini Emotion Cipher 🔐🧠

**Theme: AI + NLP + Encryption**

A full-stack **Emotion-Aware Encryption System** that detects the emotional signature of any text message and encrypts it with AES-256 — keeping **words private** while **feelings stay readable**.

---

## 🎯 Problem Statement

In the world of *Empathy Encryption*, emotional data must be protected yet still understood by AI systems. The challenge: design an algorithm that encrypts a message such that its **emotional signature** (not the text) can still be recognized, and then decrypt it to retrieve both the emotion and the original text.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Multi-Emotion NLP** | Detects up to 4 simultaneous emotions with confidence scores (e.g. *Joy + Anxiety*)|
| 🔐 **AES-256 Encryption** | Fernet symmetric encryption — text is completely unreadable without the key |
| 💡 **EMOSIG Format** | Encrypted payloads embed a hashed emotional signature prefix, readable by AI without decrypting the text |
| 📊 **Live Analytics** | Dashboard showing emotion distribution across all encrypted messages |
| 🎨 **Premium UI** | Animated, glassmorphism design with real-time emotion confidence bar visualizations |
| ⚡ **FastAPI Backend** | RESTful JSON API with OpenAPI docs at `/docs` |

---

## 🧬 Architecture

```
 User Input (Text)
       │
       ▼
 ┌───────────────┐
 │   FastAPI     │  /submit endpoint
 │   Backend     │
 └──────┬────────┘
        │
   ┌────┴────────┐
   │             │
   ▼             ▼
┌──────────┐  ┌──────────────┐
│ NLP Model│  │ AES-256      │
│ TF-IDF + │  │ Fernet Encr  │
│ LogReg   │  │              │
└──────────┘  └──────────────┘
   │ Emotions      │ Ciphertext
   │               │
   └───────┬───────┘
           │
           ▼
   ┌────────────────┐
   │  EMOSIG Payload │
   │ EMOSIG:abc123.  │  ← emotion hash prefix
   │ gAAAAABk...xZY= │  ← AES ciphertext
   └────────────────┘
           │
           ▼
   ┌────────────────┐
   │  SQLite DB     │
   │  (no plaintext)│
   └────────────────┘
```

### EMOSIG Format — The Innovation

The payload format `EMOSIG:<hash>.<AES_ciphertext>` is the core innovation:

- **`<hash>`** — a SHA-256 hash of the emotional signature (e.g. hash of "Joy + Anxiety"), base64-encoded. This lets AI systems identify the emotional category **without decrypting** the text.
- **`<AES_ciphertext>`** — the AES-256 Fernet encrypted original message. Completely private.

This achieves the balance: **Privacy** (text stays encrypted) + **Empathy** (emotion is preserved and detectable).

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.9+
- Windows / Linux / macOS

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd emotion-aware-encryption
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy the example env file
copy .env.example .env   # Windows
cp .env.example .env     # Linux/Mac

# The .env already has a generated Fernet key:
# ENCRYPTION_KEY=""
# To generate your own:
# python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### 3. Train the NLP Model
```bash
# From the project root:
python -m model.train_model
```
This trains a TF-IDF + Logistic Regression model on 120+ multi-emotion samples and saves `model/saved_model.pkl` and `model/vectorizer.pkl`.

### 4. Run the Server
```bash
uvicorn app.main:app --reload
```

### 5. Open the App


## 📖 API Reference

### POST `/submit` — Analyze & Encrypt

**Request:**
```json
{ "text": "Feeling ecstatic about the AI team, though a bit anxious about deadlines." }
```

**Response:**
```json
{
  "primary_emotion": "joy",
  "primary_emoji": "😄",
  "primary_confidence": 0.62,
  "all_emotions": [
    { "emotion": "joy",     "confidence": 0.62, "emoji": "😄", "color": "#f59e0b" },
    { "emotion": "anxiety", "confidence": 0.28, "emoji": "😰", "color": "#a78bfa" }
  ],
  "encrypted_text": "EMOSIG:Kx7aP2Lm.gAAAAABk...xZY=",
  "emotional_signature": "Joy + Anxiety"
}
```

### POST `/decrypt` — Decrypt & Reveal

**Request:**
```json
{ "encrypted_text": "EMOSIG:Kx7aP2Lm.gAAAAABk...xZY=" }
```

**Response:**
```json
{
  "original_text": "Feeling ecstatic about the AI team, though a bit anxious about deadlines.",
  "primary_emotion": "joy",
  "primary_emoji": "😄",
  "primary_confidence": 0.62,
  "all_emotions": [...],
  "emotional_signature": "Joy + Anxiety"
}
```

### GET `/stats` — Emotion Analytics

Returns emotion distribution across all stored encrypted messages.

### GET `/health` — Health Check

Returns `{ "status": "ok", "model_loaded": true, "version": "2.0.0" }`.

---

## 📊 Sample Input/Output

### Example 1
**Input:** `"Feeling ecstatic about joining the new AI research team, though a bit anxious about the deadlines ahead."`

**Detected Emotion:** `Joy + Anxiety`

**Encrypted:** `EMOSIG:Kx7aP2Lm.gAAAAABkT9x@T!aZkP#13qWv$...`

**Decrypted:** Original message + emotional profile restored ✅

---

### Example 2
**Input:** `"I can't believe I failed that test again. I'm so disappointed and frustrated right now."`

**Detected Emotion:** `Sadness + Anger`

**Encrypted:** `EMOSIG:Rv2bN8Xp.XvT$7Lp#Wk3z@R9...`

---

### Example 3
**Input:** `"Finally got the job offer! I'm thrilled and can't wait to start this new journey."`

**Detected Emotion:** `Joy + Excitement`

**Encrypted:** `EMOSIG:Pm4cQ6Yt.Qn3@BxT!82pLz#Jk...`

---

## 🔒 Security Design

| Aspect | Implementation |
|---|---|
| **Encryption Algorithm** | AES-256 via Fernet (authenticated symmetric encryption) |
| **Key Management** | Key stored in `.env` — never committed to git |
| **Database** | Only ciphertext is stored — plaintext is **never** persisted |
| **Decryption** | Happens in memory on request, never cached |
| **EMOSIG Hash** | SHA-256 — one-way, emotional signature unrecoverable from prefix alone |

---

## 🧠 NLP Model Details

| Aspect | Detail |
|---|---|
| **Algorithm** | TF-IDF Vectorizer + Multinomial Logistic Regression |
| **Features** | Unigrams + Bigrams, 5000 max features, sublinear TF |
| **Emotions Supported** | joy, sadness, anger, anxiety, fear, surprise, disgust, excitement, love, neutral |
| **Multi-emotion** | Returns all emotions above 15% confidence threshold |
| **Dataset** | 120+ curated samples (extended synthetic dataset) |

---

## 📁 Project Structure

```
emotion-aware-encryption/
├── app/
│   ├── main.py          # FastAPI routes, emotion analysis, EMOSIG logic
│   ├── database.py      # SQLAlchemy models (DBMessage)
│   └── encryption.py    # AES-256 Fernet encrypt/decrypt helpers
├── frontend/
│   └── index.html       # Single-file SPA (HTML + CSS + JS)
├── model/
│   ├── train_model.py   # NLP model training script
│   ├── saved_model.pkl  # Trained LogisticRegression (auto-generated)
│   └── vectorizer.pkl   # TF-IDF vectorizer (auto-generated)
├── .env                 # Environment secrets (not committed)
├── .env.example         # Template for environment setup
├── requirements.txt     # Python dependencies
├── Procfile             # Render.com deployment config
└── render.yaml          # Render service definition
```

---

## 🎥 Demo Video

> 📹 **[Watch the live demo on Google Drive](#)**  
> *(Replace `#` with your actual Google Drive video link)*

---

## 🌐 Live Deployment

> 🔗 **[Live App on Render](#)**  
> *(Replace `#` with your Render / hosted URL)*

---

## 📦 Tech Stack

- **Backend:** Python 3.11, FastAPI, Uvicorn
- **Encryption:** `cryptography` (Fernet / AES-256)
- **NLP:** scikit-learn (TF-IDF + Logistic Regression)
- **Database:** SQLite + SQLAlchemy ORM
- **Frontend:** Vanilla HTML/CSS/JavaScript (no frameworks, fully self-contained)
- **Deployment:** Render.com

---


