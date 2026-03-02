"""User management — register, login, get, update.

Stores one JSON file per user in ``data/users/{email_slug}.json``.
Passwords are hashed with SHA-256 + random salt.
"""

import hashlib
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

USERS_DIR = Path(__file__).resolve().parent.parent / "data" / "users"


def _email_slug(email: str) -> str:
    """Convert an email address to a safe filename slug."""
    return re.sub(r"[^a-zA-Z0-9]", "_", email.lower())


def _user_path(email: str) -> Path:
    return USERS_DIR / f"{_email_slug(email)}.json"


def _hash_password(password: str, salt: str | None = None) -> str:
    """Return ``salt$hash`` string."""
    if salt is None:
        salt = secrets.token_hex(16)
    h = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}${h}"


def _verify_password(password: str, stored: str) -> bool:
    salt, _ = stored.split("$", 1)
    return _hash_password(password, salt) == stored


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def register_user(email: str, password: str) -> dict | str:
    """Create a new user. Returns user dict on success, error string on failure."""
    email = email.strip().lower()
    if not email or not password:
        return "Email and password are required."
    if _user_path(email).exists():
        return "An account with this email already exists."

    USERS_DIR.mkdir(parents=True, exist_ok=True)

    user = {
        "email": email,
        "password_hash": _hash_password(password),
        "favorite_teams": [],
        "subscription": "free",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _user_path(email).write_text(json.dumps(user, indent=2))
    return user


def login_user(email: str, password: str) -> dict | None:
    """Validate credentials. Returns user dict or None."""
    email = email.strip().lower()
    path = _user_path(email)
    if not path.exists():
        return None
    user = json.loads(path.read_text())
    if _verify_password(password, user["password_hash"]):
        return user
    return None


def get_user(email: str) -> dict | None:
    """Read user JSON, or None if not found."""
    path = _user_path(email.strip().lower())
    if not path.exists():
        return None
    return json.loads(path.read_text())


def update_user(email: str, updates: dict) -> dict | None:
    """Merge *updates* into the user's JSON and persist. Returns updated user."""
    email = email.strip().lower()
    path = _user_path(email)
    if not path.exists():
        return None
    user = json.loads(path.read_text())
    user.update(updates)
    path.write_text(json.dumps(user, indent=2))
    return user
