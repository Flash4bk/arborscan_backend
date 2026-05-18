"""
ArborScan analytical wind-load module.

This is a practical Python adaptation of the modelling idea from the Maple
sample: the tree is represented as a vertical chain of elements, and the
aerodynamic coefficient beta is distributed across crown elements.

It is NOT a full fall-dynamics solver yet. It is a fast analytical model for:
- beta_i distribution along the crown;
- wind force per element F_i = beta_i * v;
- total wind force;
- center of wind load;
- bending moment at the tree base.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional


@dataclass
class CrownElementLoad:
    index: int
    z_mid_m: float
    beta_i_kg_s: float
    force_n: float
    moment_nm: float
    weight: float


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.replace(",", ".").strip()
            if not value:
                return default
        return float(value)
    except Exception:
        return default


def _species_strength_mpa(species: str) -> float:
    """Very approximate bending strength proxy, used only for a normalized indicator."""
    table = {
        "Сосна": 45.0,
        "Ель": 38.0,
        "Береза": 60.0,
        "Дуб": 80.0,
        "Тополь": 35.0,
    }
    return table.get(species, 45.0)


def _default_crown_start_height(height_m: float) -> float:
    # In the paper sample for tree 4: H≈20 m, crown starts at ≈14 m.
    # For a general mobile estimate we use a softer default: crown is upper 45% of tree.
    return 0.55 * height_m


def _crown_weight_profile(z_mid: float, crown_start: float, height: float, density_factor: float) -> float:
    """Smooth crown mass/intensity profile.

    The Maple sample distributes beta by crown mass fractions. Without direct
    mass measurements we approximate crown density by a bell-like profile inside
    the crown: lower/middle crown gets more load than the very top.
    """
    crown_h = max(0.01, height - crown_start)
    u = _clamp((z_mid - crown_start) / crown_h, 0.0, 1.0)

    # Bell/triangular hybrid: low at crown base and top, max in the upper-middle.
    bell = math.sin(math.pi * u)
    upper_bias = 0.75 + 0.50 * u
    return max(0.0, bell * upper_bias * max(0.1, density_factor))


def compute_analytic_wind_model(
    *,
    species: str,
    height_m: Any,
    crown_width_m: Any,
    trunk_diameter_m: Any,
    beta_kg_s: Any,
    wind_speed_m_s: Any = None,
    wind_gust_m_s: Any = None,
    crown_start_height_m: Any = None,
    crown_density_factor: Any = 1.0,
    crown_shape_factor: Any = 1.0,
    n_elements: int = 20,
) -> Dict[str, Any]:
    """Compute a fast analytical wind-load model.

    Parameters
    ----------
    height_m, crown_width_m, trunk_diameter_m:
        Tree geometry from AR/photo.
    beta_kg_s:
        Total beta coefficient for the tree crown.
    wind_speed_m_s / wind_gust_m_s:
        Prefer gust if available.
    crown_start_height_m:
        Optional manual height where crown begins. If missing, H_crown≈0.45H.
    crown_density_factor:
        Manual/expert correction for crown density. Typical range: 0.6..1.5.
    crown_shape_factor:
        Manual/expert correction for exposed crown shape. Typical range: 0.6..1.4.

    Returns
    -------
    dict with distributed load, total force, load center and base moment.
    """
    h = _as_float(height_m, 0.0) or 0.0
    w = _as_float(crown_width_m, 0.0) or 0.0
    d = _as_float(trunk_diameter_m, 0.0) or 0.0
    beta = _as_float(beta_kg_s, 0.0) or 0.0

    gust = _as_float(wind_gust_m_s, None)
    speed = _as_float(wind_speed_m_s, None)
    v = gust if gust is not None and gust > 0 else speed
    if v is None:
        v = 0.0

    density_factor = _clamp(_as_float(crown_density_factor, 1.0) or 1.0, 0.2, 3.0)
    shape_factor = _clamp(_as_float(crown_shape_factor, 1.0) or 1.0, 0.2, 3.0)

    if h <= 0 or beta <= 0:
        return {
            "available": False,
            "reason": "Недостаточно данных: нужны высота дерева и β.",
        }

    if n_elements < 5:
        n_elements = 5
    if n_elements > 80:
        n_elements = 80

    crown_start = _as_float(crown_start_height_m, None)
    if crown_start is None or crown_start <= 0 or crown_start >= h:
        crown_start = _default_crown_start_height(h)
        crown_start_source = "default_0_55_height"
    else:
        crown_start = _clamp(crown_start, 0.05 * h, 0.95 * h)
        crown_start_source = "manual"

    dz = h / n_elements
    raw = []
    for i in range(n_elements):
        z_mid = (i + 0.5) * dz
        if z_mid >= crown_start:
            weight = _crown_weight_profile(z_mid, crown_start, h, density_factor)
        else:
            weight = 0.0
        raw.append((i + 1, z_mid, weight))

    total_weight = sum(x[2] for x in raw)
    if total_weight <= 0:
        return {
            "available": False,
            "reason": "Не удалось распределить нагрузку по кроне.",
        }

    # Crown shape factor changes effective exposed force.
    # If beta was estimated from area, this is a small manual correction.
    beta_effective = beta * shape_factor
    total_force_n = beta_effective * v

    elements: List[CrownElementLoad] = []
    for idx, z_mid, weight in raw:
        if weight <= 0:
            continue
        share = weight / total_weight
        beta_i = beta_effective * share
        force = beta_i * v
        moment = force * z_mid
        elements.append(
            CrownElementLoad(
                index=idx,
                z_mid_m=z_mid,
                beta_i_kg_s=beta_i,
                force_n=force,
                moment_nm=moment,
                weight=share,
            )
        )

    if not elements:
        return {
            "available": False,
            "reason": "Нет активных элементов кроны.",
        }

    base_moment_nm = sum(e.moment_nm for e in elements)
    center_of_load_m = base_moment_nm / total_force_n if total_force_n > 0 else None

    # Very approximate resistance proxy: M_allow = MOR * section_modulus.
    # Section modulus for circular section: W = pi*d^3/32.
    # MOR in Pa.
    strength_pa = _species_strength_mpa(species) * 1e6
    section_modulus = math.pi * (d ** 3) / 32.0 if d > 0 else None
    resistance_moment_nm = strength_pa * section_modulus if section_modulus else None

    if resistance_moment_nm and resistance_moment_nm > 0:
        moment_ratio = base_moment_nm / resistance_moment_nm
    else:
        moment_ratio = None

    # Practical analytical score. It is intentionally conservative but not decisive alone.
    if moment_ratio is None:
        moment_score = 0.5
    elif moment_ratio < 0.10:
        moment_score = 0.25
    elif moment_ratio < 0.25:
        moment_score = 0.45
    elif moment_ratio < 0.50:
        moment_score = 0.70
    else:
        moment_score = 1.0

    slenderness = h / d if d > 0 else None
    if slenderness is None:
        slenderness_factor = 0.5
    elif slenderness >= 80:
        slenderness_factor = 1.0
    elif slenderness >= 60:
        slenderness_factor = 0.75
    elif slenderness >= 40:
        slenderness_factor = 0.45
    else:
        slenderness_factor = 0.25

    analytical_score = _clamp(0.65 * moment_score + 0.35 * slenderness_factor, 0.0, 1.0)

    return {
        "available": True,
        "model": "fast_chain_crown_load_v1",
        "formula": "F_i = β_i · v; M_base = Σ(F_i · z_i)",
        "inputs": {
            "species": species,
            "height_m": round(h, 3),
            "crown_width_m": round(w, 3) if w else None,
            "trunk_diameter_m": round(d, 3) if d else None,
            "beta_kg_s": round(beta, 3),
            "beta_effective_kg_s": round(beta_effective, 3),
            "wind_speed_used_m_s": round(v, 3),
            "wind_speed_source": "gust" if gust is not None and gust > 0 else "speed",
            "crown_start_height_m": round(crown_start, 3),
            "crown_start_source": crown_start_source,
            "crown_density_factor": round(density_factor, 3),
            "crown_shape_factor": round(shape_factor, 3),
            "n_elements": n_elements,
        },
        "outputs": {
            "total_force_n": round(total_force_n, 2),
            "center_of_load_m": round(center_of_load_m, 2) if center_of_load_m is not None else None,
            "base_moment_nm": round(base_moment_nm, 2),
            "resistance_moment_nm_proxy": round(resistance_moment_nm, 2) if resistance_moment_nm else None,
            "moment_ratio_proxy": round(moment_ratio, 4) if moment_ratio is not None else None,
            "slenderness": round(slenderness, 2) if slenderness is not None else None,
            "analytical_score": round(analytical_score, 3),
        },
        "elements": [
            {
                "index": e.index,
                "z_mid_m": round(e.z_mid_m, 2),
                "weight": round(e.weight, 4),
                "beta_i_kg_s": round(e.beta_i_kg_s, 4),
                "force_n": round(e.force_n, 2),
                "moment_nm": round(e.moment_nm, 2),
            }
            for e in elements
        ],
        "notes": [
            "Модель является быстрой аналитической аппроксимацией Maple-подхода.",
            "β распределяется по элементам кроны пропорционально условной массе/плотности кроны.",
            "Для точной динамики падения требуется отдельная численная модель с уравнениями Лагранжа.",
        ],
    }
