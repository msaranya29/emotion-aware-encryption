# Emotion-Aware Encryption System

A complete full-stack backend system combining ML-based emotion detection and AES encryption.

## Overview
This system takes plaintext input, runs a TF-IDF Logistic Regression NLP model to determine the core emotion and a confidence score, and then encrypts the text using AES-256 (Fernet). Only the encrypted text and emotion metadata are stored in the database.

## Architecture Diagram

```ascii
     +-------------+
     | User Input  |  (Text)
     +------+------+
            |
            v
     +------+------+
     |   FastAPI   |  /submit 
     +---+------+--+
         |      |
         v      v
+--------+-+  +-+--------+
| ML Model |  | AES Encr |
| (Pickle) |  | (Fernet) |
+--------+-+  +-+--------+
         |      |
         v      v
    +----+------+----+
    | SQLite DB      |
    | (messages tbl) |
    +----------------+
```

## Setup Instructions

1. **Install requirements:**
   `pip install -r requirements.txt`

2. **Set up the .env file:**
   The project requires an `ENCRYPTION_KEY`. Ensure `.env` contains a valid Fernet key. (Already done for you: `ENCRYPTION_KEY="w9jFo_qhesDXhtYYFZcrkvXpClD_-hbkplMuYqUihUU="`)

3. **Train the ML model:**
   `cd model`
   `python train_model.py`
   This will generate `vectorizer.pkl` and `saved_model.pkl`.

4. **Run the server:**
   `uvicorn app.main:app --reload`

## API Usage Examples

### 1. Submit Text
**POST /submit**
```json
{
  "text": "I feel anxious about tomorrow"
}
```
**Response:**
```json
{
  "emotion": "anxiety",
  "confidence": 0.81,
  "encrypted_text": "gAAAAABk...xZY="
}
```

### 2. Decrypt Text
**POST /decrypt**
```json
{
  "encrypted_text": "gAAAAABk...xZY="
}
```
**Response:**
```json
{
  "original_text": "I feel anxious about tomorrow",
  "emotion": "anxiety",
  "confidence": 0.81
}
```

### 3. Get Stats
**GET /stats**
**Response:**
```json
[
  {
    "emotion": "anxiety",
    "count": 1
  }
]
```

## Security Overview
The application uses the `cryptography.fernet` module for AES encryption. The key is securely loaded from environment variables using `python-dotenv`. Plaintext records are never committed to the SQLite DB; only the AES encrypted byte-string is saved to guarantee absolute confidentiality. The endpoint layer decrypts dynamically in memory for valid token inputs.
