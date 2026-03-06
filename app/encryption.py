import os
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

def get_cipher() -> Fernet:
    """
    Retrieves the AES encryption key from the environment and initializes a Fernet cipher.
    Fernet is an implementation of symmetric authenticated cryptography built on AES.
    """
    key = os.getenv("ENCRYPTION_KEY")
    if not key:
        # Fallback for hackathon purposes if user forgot to set env.
        # In a real production system, this should rigorously fail.
        raise ValueError("Critical Security Error: ENCRYPTION_KEY environment variable is not set. Please set it in your .env file.")
    return Fernet(key.encode('utf-8'))

def encrypt_text(text: str) -> str:
    """
    Encrypts a plaintext string to an AES encrypted token.
    """
    cipher = get_cipher()
    encrypted_bytes = cipher.encrypt(text.encode('utf-8'))
    return encrypted_bytes.decode('utf-8')

def decrypt_text(encrypted_text: str) -> str:
    """
    Decrypts an AES encrypted token back to plaintext.
    """
    cipher = get_cipher()
    decrypted_bytes = cipher.decrypt(encrypted_text.encode('utf-8'))
    return decrypted_bytes.decode('utf-8')
