"""Minimal salted-PBKDF2 authentication with roles (admin / viewer)."""
import hashlib, hmac, json, os
from .config import USERS_FILE

def _hash(pw, salt): return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 120_000).hex()

def _load():
    if not USERS_FILE.exists():
        USERS_FILE.parent.mkdir(exist_ok=True); _save({}); add_user("admin", "admin123", "admin"); add_user("viewer", "viewer123", "viewer")
    return json.loads(USERS_FILE.read_text())

def _save(u): USERS_FILE.write_text(json.dumps(u, indent=2))

def add_user(name, pw, role="viewer"):
    u = json.loads(USERS_FILE.read_text()) if USERS_FILE.exists() else {}
    salt = os.urandom(16); u[name] = {"salt": salt.hex(), "hash": _hash(pw, salt), "role": role}; _save(u)

def delete_user(name):
    u = _load(); u.pop(name, None); _save(u)

def list_users(): return {k: v["role"] for k, v in _load().items()}

def verify(name, pw):
    rec = _load().get(name)
    if rec and hmac.compare_digest(_hash(pw, bytes.fromhex(rec["salt"])), rec["hash"]): return rec["role"]
    return None
