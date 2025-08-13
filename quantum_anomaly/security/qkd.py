import secrets
import hashlib
from typing import Tuple, List, Optional
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

"""
BB84 protocol simulation (no quantum hardware):
- Alice picks random bits and bases
- Bob picks random bases, measures
- They publicly compare bases (not bits) to get sifted key
- Estimate error rate (optionally simulate eavesdropping)
- If error below threshold, derive shared key via HKDF
"""

def _random_bits(n: int, seed: Optional[int] = None) -> List[int]:
    rng = secrets.SystemRandom() if seed is None else (secrets.SystemRandom(seed))
    return [rng.randrange(2) for _ in range(n)]

def _random_bases(n: int, seed: Optional[int] = None) -> List[int]:
    # 0 = rectilinear (+), 1 = diagonal (x)
    return _random_bits(n, seed)

def bb84_simulate_key(length: int = 1024, seed: Optional[int] = None, error_rate: float = 0.0) -> Tuple[bytes, float, int]:
    """
    Returns (raw_key_bytes, observed_error_rate, sifted_length)
    """
    alice_bits = _random_bits(length, seed)
    alice_bases = _random_bases(length, seed)
    bob_bases = _random_bases(length, None if seed is None else seed + 1)

    # Bob's measured bits: if bases match, matches; if not, random
    bob_bits = []
    rng = secrets.SystemRandom()
    for i in range(length):
        if alice_bases[i] == bob_bases[i]:
            bit = alice_bits[i]
        else:
            bit = rng.randrange(2)
        # Simulate eavesdropping noise
        if rng.random() < error_rate:
            bit ^= 1
        bob_bits.append(bit)

    # Publicly compare bases and keep only matches
    sifted_a = []
    sifted_b = []
    for a_bit, a_base, b_bit, b_base in zip(alice_bits, alice_bases, bob_bits, bob_bases):
        if a_base == b_base:
            sifted_a.append(a_bit)
            sifted_b.append(b_bit)
    sifted_len = len(sifted_a)

    # Estimate error
    errors = sum(1 for x, y in zip(sifted_a, sifted_b) if x != y)
    observed_error = errors / max(1, sifted_len)

    # Use matching subset to derive a raw shared key (take Alice's bits)
    raw_bits = sifted_a[:sifted_len]
    # Convert to bytes
    byte_str = bytes(int("".join(str(b) for b in raw_bits[i:i+8]).ljust(8, "0"), 2) for i in range(0, len(raw_bits), 8))
    return byte_str, observed_error, sifted_len

def derive_session_key(raw_key: bytes, out_len: int = 32, salt: Optional[bytes] = None, info: bytes = b"QKD-BB84-Session"):
    if not salt:
        # Derive salt from raw key digest
        salt = hashlib.sha256(raw_key).digest()[:16]
    hkdf = HKDF(algorithm=hashes.SHA256(), length=out_len, salt=salt, info=info)
    return hkdf.derive(raw_key)
