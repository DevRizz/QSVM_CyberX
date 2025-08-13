from typing import Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

class SecureChannel:
    """
    AES-GCM symmetric encryption using a session key (e.g., from QKD).
    """

    def __init__(self, key: bytes):
        if len(key) not in (16, 24, 32):
            raise ValueError("Key must be 128/192/256 bits")
        self._key = key
        self._aes = AESGCM(self._key)

    def encrypt(self, plaintext: bytes, aad: bytes = b"") -> Tuple[bytes, bytes]:
        nonce = os.urandom(12)
        ct = self._aes.encrypt(nonce, plaintext, aad)
        return nonce, ct

    def decrypt(self, nonce: bytes, ciphertext: bytes, aad: bytes = b"") -> bytes:
        return self._aes.decrypt(nonce, ciphertext, aad)
