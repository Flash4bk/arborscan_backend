from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


# IMPORTANT:
# This table is deliberately empty in v4 alpha.  ArborScan must not silently
# substitute properties of another species.  Profiles are added only after a
# specific species + property + source has been reviewed and documented.
_VALIDATED_MECHANICAL_PROFILES: Dict[str, Dict[str, Any]] = {}


def normalize_scientific_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    normalized = " ".join(str(name).strip().split())
    return normalized or None


def get_mechanical_profile(scientific_name: Optional[str]) -> Dict[str, Any]:
    """Return an exact validated profile or an explicit unavailable result.

    There is intentionally no genus/group/pine fallback.  This function is a
    guardrail against the v3 behaviour where an unsupported species could be
    treated as ``Сосна`` for structural calculations.
    """

    canonical = normalize_scientific_name(scientific_name)
    if not canonical:
        return {
            "available": False,
            "scientific_name": None,
            "source": None,
            "properties": {},
            "reason": "species_not_identified",
        }

    profile = _VALIDATED_MECHANICAL_PROFILES.get(canonical.casefold())
    if profile is None:
        return {
            "available": False,
            "scientific_name": canonical,
            "source": None,
            "properties": {},
            "reason": "mechanical_profile_not_validated",
        }

    result = deepcopy(profile)
    result.setdefault("available", True)
    result.setdefault("scientific_name", canonical)
    result.setdefault("properties", {})
    return result
