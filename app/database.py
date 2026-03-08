from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "emotions_encryption.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# For SQLite, check_same_thread=False is needed for scoped sessions
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class DBMessage(Base):
    """
    Stores only the encrypted text. Plaintext is NEVER stored here.
    Emotional metadata is kept for analytics, multi-emotion support, and
    system functionality (decryption lookup).
    
    Schema:
    - encrypted_text: The AES-256 Fernet ciphertext wrapped with EMOSIG format
    - emotion: The emotional signature string e.g. "Joy + Anxiety"
    - confidence: Primary emotion confidence score (0.0 - 1.0)
    - emotions_json: JSON array of all detected emotions with scores
    - timestamp: UTC datetime of ingestion
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    encrypted_text = Column(String, index=True, unique=True)
    emotion = Column(String, index=True)          # Full signature e.g. "Joy + Anxiety"
    confidence = Column(Float)
    emotions_json = Column(Text, nullable=True)   # JSON: [{emotion, confidence, emoji, color}]
    timestamp = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
