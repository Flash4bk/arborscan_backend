"""Verified Google identity policy for ArborScan's existing opaque sessions.

This is not Supabase Auth. Identity claims must remain disabled until a reviewed
unique constraint on public.users.google_sub has been installed. A conditional
row update prevents reassignment of that row, but cannot replace the database's
global uniqueness guarantee. Existing, unambiguous Google identities can log in
without changing any identity or requiring the claims feature flag.
"""

from dataclasses import dataclass
from collections.abc import Mapping
import re
from typing import Callable


class GoogleIdentityError(Exception):
    """A safe public failure that never contains tokens or database details."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


_INVALID = "Google не подтвердил данные аккаунта. Войдите другим способом."
_CONFLICT = (
    "Данные входа Google конфликтуют с существующим профилем. "
    "Войдите прежним способом и обратитесь к разработчику."
)
_UNAVAILABLE = (
    "Подключение Google к новому профилю пока недоступно на сервере. "
    "Используйте существующий вход по почте."
)


@dataclass(frozen=True)
class VerifiedGoogleIdentity:
    subject: str
    email: str
    name: str
    avatar_url: str
    authoritative_email: bool


def verified_google_identity(info: Mapping) -> VerifiedGoogleIdentity:
    """Accept only claims returned by signature/audience/issuer/expiry validation.

Client email/name/photo fields are deliberately not arguments. Google is an
authority for a Gmail address or a verified Workspace address (hd present).
Third-party email alone may not link an existing password-based profile.
"""
    if not isinstance(info, Mapping):
        raise GoogleIdentityError(401, _INVALID)
    subject = info.get("sub")
    email = info.get("email")
    if (not isinstance(subject, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,255}", subject)
            or not isinstance(email, str)
            or info.get("email_verified") is not True):
        raise GoogleIdentityError(401, _INVALID)
    email = email.strip().lower()
    if len(email) > 254 or not re.fullmatch(r"[^@\s(),]+@[^@\s(),]+\.[^@\s(),]+", email):
        raise GoogleIdentityError(401, _INVALID)
    name = info.get("name")
    if not isinstance(name, str) or not name.strip():
        name = email.split("@", 1)[0]
    avatar = info.get("picture")
    if not isinstance(avatar, str) or not avatar.startswith("https://"):
        avatar = ""
    hosted_domain = info.get("hd")
    authoritative = (email.rsplit("@", 1)[1] == "gmail.com"
                     or (isinstance(hosted_domain, str)
                         and bool(re.fullmatch(r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}", hosted_domain))))
    return VerifiedGoogleIdentity(subject, email, name.strip()[:256],
                                  avatar[:2048], authoritative)


def _single_row(db_json: Callable, key: str, value: str):
    rows = db_json("GET", "users", params={key: f"eq.{value}",
                                            "select": "*", "limit": "2"})
    if not isinstance(rows, list):
        raise GoogleIdentityError(503, _UNAVAILABLE)
    if len(rows) > 1:
        raise GoogleIdentityError(409, _CONFLICT)
    if not rows:
        return None
    user = rows[0]
    if not isinstance(user, dict) or not isinstance(user.get("id"), str):
        raise GoogleIdentityError(503, _UNAVAILABLE)
    return user


def _consistent_subject_user(db_json: Callable, identity: VerifiedGoogleIdentity):
    user = _single_row(db_json, "google_sub", identity.subject)
    email_user = _single_row(db_json, "email", identity.email)
    if user and email_user and user["id"] != email_user["id"]:
        raise GoogleIdentityError(409, _CONFLICT)
    return user, email_user


def resolve_google_user(db_json: Callable, create_user: Callable,
                        identity: VerifiedGoogleIdentity, *,
                        allow_identity_claims: bool = False,
                        updated_at: str) -> dict:
    """Resolve a stable subject; guarded linking never changes another subject.

When enabling claims, the operator must first verify subject uniqueness in the
real database. The helper refuses duplicate observations and ambiguous matches,
uses compare-and-set for a link, and re-reads a concurrently established claim.
No role, password, email or owner ID is accepted from the client or patched.
"""
    user, email_user = _consistent_subject_user(db_json, identity)
    if user:
        return user
    if email_user and email_user.get("google_sub") not in (None, ""):
        raise GoogleIdentityError(409, _CONFLICT)
    if email_user and not identity.authoritative_email:
        raise GoogleIdentityError(409, _CONFLICT)
    if not allow_identity_claims:
        raise GoogleIdentityError(503, _UNAVAILABLE)

    if email_user:
        previous_subject = email_user.get("google_sub")
        params = {
            "id": f"eq.{email_user['id']}",
            "email": f"eq.{identity.email}",
            "google_sub": "is.null" if previous_subject is None else "eq.",
        }
        try:
            updated = db_json("PATCH", "users", params=params,
                              json_body={"google_sub": identity.subject,
                                         "provider": "google",
                                         "name": identity.name,
                                         "avatar_url": identity.avatar_url,
                                         "updated_at": updated_at},
                              prefer="return=representation")
        except Exception:
            # The unique-subject constraint may have rejected a concurrent
            # claim. Resolve only the same resulting owner; never retry PATCH.
            updated = []
        if isinstance(updated, list) and len(updated) == 1:
            candidate = updated[0]
            if (isinstance(candidate, dict)
                    and candidate.get("id") == email_user["id"]
                    and candidate.get("google_sub") == identity.subject):
                return candidate
        current, _ = _consistent_subject_user(db_json, identity)
        if current and current["id"] == email_user["id"]:
            return current
        raise GoogleIdentityError(409, _CONFLICT)

    try:
        candidate = create_user(identity)
    except Exception:
        # A simultaneous insert for the same verified subject/email may win.
        # Read only; failures do not fall back to somebody else's email owner.
        current, _ = _consistent_subject_user(db_json, identity)
        if current:
            return current
        raise GoogleIdentityError(503, _UNAVAILABLE)
    current, _ = _consistent_subject_user(db_json, identity)
    if (not isinstance(candidate, dict) or not current
            or candidate.get("id") != current["id"]):
        raise GoogleIdentityError(409, _CONFLICT)
    return current
