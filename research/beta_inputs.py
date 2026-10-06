"""Versioned, provenance-preserving inputs for the research-only planar chain.

Unknown values/visibility, incomplete calibration and unsupported supports fail
explicitly. Unit conversion happens once; the original document remains intact.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

SCHEMA_VERSION = 2
MODEL = 'planar-hinge-chain-v1'
FRAME = {'coordinates': ['x', 'z'], 'x': 'horizontal', 'z': 'up',
         'angle': 'from_vertical_toward_positive_x', 'origin': 'fixed_world'}
UNITS = {'length': 'm', 'time': 's', 'mass': 'kg', 'angle': 'rad',
         'beta': 'kg/s', 'elastic_modulus': 'Pa', 'inertia': 'kg m^2'}
CONVERSIONS = {
    'm': {'m': 1, 'cm': .01, 'mm': .001},
    's': {'s': 1, 'ms': .001},
    'kg': {'kg': 1, 'g': .001},
    'rad': {'rad': 1, 'deg': math.pi / 180},
    'rad/s': {'rad/s': 1, 'deg/s': math.pi / 180},
    'Pa': {'Pa': 1, 'MPa': 1e6, 'GPa': 1e9},
    'kg m^2': {'kg m^2': 1}, 'kg/m^3': {'kg/m^3': 1},
    'm/s': {'m/s': 1, 'cm/s': .01}, 'm/s^2': {'m/s^2': 1},
    'kg/s': {'kg/s': 1},
}


def numeric(value, name):
    """No nulls, strings, booleans, NaN/Inf, or ragged numeric arrays."""
    def check(v):
        if isinstance(v, (list, tuple)):
            for item in v:
                check(item)
        elif isinstance(v, (bool, str)) or v is None:
            raise ValueError(f'{name}: finite numeric values required; missing data are unsupported')
    check(value)
    try:
        if np.asarray(value).dtype.kind not in 'iuf':
            raise ValueError(f'{name}: numeric arrays required')
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}: rectangular numeric data required') from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f'{name}: finite numeric values required')
    return result


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name}: explicit description/source required')
    return value


def mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name}: JSON object required')
    return value


def quantity(entry, unit, name):
    if not isinstance(entry, dict) or 'standard_uncertainty' not in entry:
        raise ValueError(f'{name}: value, unit, source and standard_uncertainty (or null) required')
    text(entry.get('source'), name + '.source')
    factor = CONVERSIONS[unit].get(entry.get('unit'))
    if factor is None:
        raise ValueError(f'{name}: unsupported unit; expected {list(CONVERSIONS[unit])}')
    values = numeric(entry.get('value'), name) * factor
    uncertainty = entry['standard_uncertainty']
    if uncertainty is not None:
        uncertainty = numeric(uncertainty, name + '.standard_uncertainty') * abs(factor)
        if np.any(uncertainty < 0) or uncertainty.shape not in [(), values.shape]:
            raise ValueError(f'{name}: nonnegative scalar or matching uncertainty required')
        uncertainty = uncertainty.tolist()
    return values, {'value': values.tolist(), 'unit': unit, 'source': entry['source'],
                    'standard_uncertainty': uncertainty, 'input_unit': entry['unit']}


def scalar(value, name, *, positive=False, nonnegative=False):
    a = numeric(value, name)
    if a.shape != () or (positive and a <= 0) or (nonnegative and a < 0):
        raise ValueError(f'{name}: invalid scalar')
    return float(a)


def integer(value, name, lo, hi):
    a = scalar(value, name)
    if not a.is_integer() or not lo <= a <= hi:
        raise ValueError(f'{name}: integer in [{lo}, {hi}] required')
    return int(a)


@dataclass
class DynamicInput:
    chain: object
    theta0: np.ndarray
    omega0: np.ndarray
    times: np.ndarray
    observed: np.ndarray
    points: dict
    config: dict
    normalized: dict


def parse_document(document):
    from .planar_chain import PlanarChain, cylinder_mass, cylinder_inertia, triangular_weights
    if not isinstance(document, dict) or document.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Coupled dynamics requires schema_version=2')
    if document.get('model') != MODEL or not isinstance(document.get('synthetic'), bool):
        raise ValueError('Explicit supported model and synthetic boolean required')
    if document.get('units') != UNITS or document.get('frame') != FRAME:
        raise ValueError('Explicit normalized SI units and x/z fixed-hinge frame required')
    if document.get('support') != 'fixed_hinge_free_tip':
        raise ValueError('Only fixed_hinge_free_tip is supported; no clamped/root/moving base')
    text(document.get('dataset_id'), 'dataset_id')
    p = mapping(document.get('parameters'), 'parameters')
    normalized = {}
    def q(name, unit):
        value, normalized[name] = quantity(p.get(name), unit, name)
        return value
    lengths = q('lengths_m', 'm')
    if lengths.ndim != 1 or not 1 <= len(lengths) <= 64:
        raise ValueError('Require 1..64 bodies in lengths_m')
    n = len(lengths)
    diameters = q('diameters_m', 'm')
    joint_diameters = q('joint_diameters_m', 'm')
    e = scalar(q('elastic_modulus_pa', 'Pa'), 'elastic_modulus_pa', positive=True)
    base = q('base_position_m', 'm')
    g = scalar(q('gravity_m_s2', 'm/s^2'), 'gravity_m_s2', positive=True)
    ground = scalar(q('ground_z_m', 'm'), 'ground_z_m')
    air = q('air_velocity_m_s', 'm/s')
    air_model = document.get('air_model')
    if air_model not in ['still', 'constant_velocity'] or (air_model == 'still' and np.any(air != 0)):
        raise ValueError('Explicit still air or constant_velocity matching air_velocity_m_s required')
    theta, omega = q('initial_angles_rad', 'rad'), q('initial_angular_velocities_rad_s', 'rad/s')
    if theta.shape != (n,) or omega.shape != (n,):
        raise ValueError('One supplied initial angle AND angular velocity per body required')
    if document.get('initial_velocity_mode') != 'supplied':
        raise ValueError('Only supplied initial angular velocities supported; no implicit zero or joint fit')
    body = mapping(document.get('body'), 'body')
    assumption = text(body.get('midpoint_assumption'), 'body.midpoint_assumption')
    mode = body.get('mode')
    if mode == 'uniform_wood_cylinders':
        density = q('density_kg_m3', 'kg/m^3')
        if density.shape not in [(), (n,)]:
            raise ValueError('Density must be scalar or one per body')
        masses = cylinder_mass(density, diameters, lengths)
        inertias = cylinder_inertia(masses, diameters, lengths)
    elif mode in ['effective_midpoint_cylinders', 'measured_midpoint_inertias']:
        masses = q('masses_kg', 'kg')
        if mode == 'effective_midpoint_cylinders':
            text(body.get('effective_body_assumption'), 'body.effective_body_assumption')
            inertias = cylinder_inertia(masses, diameters, lengths)
        else:
            inertias = q('inertias_kg_m2', 'kg m^2')
    else:
        raise ValueError('Unsupported body model; midpoint COM/inertia assumptions must be explicit')
    load = mapping(document.get('load_profile'), 'load_profile')
    text(load.get('source'), 'load_profile.source')
    if load.get('kind') == 'crown_mass_fractions':
        crown, normalized['crown_masses_kg'] = quantity(load.get('crown_masses_kg'), 'kg', 'crown_masses_kg')
        if crown.shape != (n,) or np.any(crown < 0) or crown.sum() <= 0:
            raise ValueError('Explicit nonnegative crown masses with positive sum required')
        if mode == 'uniform_wood_cylinders' or masses.shape != (n,) or np.any(crown > masses):
            raise ValueError('Full body masses/inertias must include crown; wood-only cylinders cannot ignore it')
        weights = crown / crown.sum()
    elif load.get('kind') == 'normalized_triangular_density':
        text(load.get('assumption'), 'load_profile.assumption')
        limits, normalized['triangle_support_m'] = quantity(load.get('support_m'), 'm', 'triangle support')
        if limits.shape != (3,):
            raise ValueError('Triangle support [start, peak, end] required')
        weights = triangular_weights(lengths, (limits[0], limits[2]), peak_m=limits[1])
    else:
        raise ValueError('Load requires crown masses or an explicitly assumed normalized triangular density')
    chain = PlanarChain(lengths_m=lengths, diameters_m=diameters, masses_kg=masses,
                        inertias_kg_m2=inertias, elastic_modulus_pa=e,
                        joint_diameters_m=joint_diameters, weights=weights, base_m=base,
                        gravity_m_s2=g, air_velocity_m_s=air, ground_z_m=ground,
                        support=document['support'], effective_body_assumption=(
                            body['effective_body_assumption'] if mode == 'effective_midpoint_cylinders'
                            else assumption))
    obs = mapping(document.get('observations'), 'observations')
    times, normalized['times_s'] = quantity(obs.get('times_s'), 's', 'times_s')
    observed, normalized['positions_m'] = quantity(obs.get('positions_m'), 'm', 'positions_m')
    if times.ndim != 1 or not 3 <= len(times) <= 2000 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError('3..2000 strictly increasing times starting at initial-state t=0 required')
    if observed.ndim != 3 or observed.shape[0] != len(times) or observed.shape[2] != 2:
        raise ValueError('Dense observed x,z array [time, point, 2] required')
    if times[-1] > 120:
        raise ValueError('First research contract limits duration to 120 s')
    if obs.get('time_origin') != 'initial_state':
        raise ValueError('time_origin must identify the supplied initial_state')
    visibility = obs.get('visibility')
    if visibility is not None:
        visible = np.asarray(visibility)
        if visible.dtype.kind != 'b' or visible.shape != observed.shape[:2] or not np.all(visible):
            raise ValueError('Invisible/missing points unsupported: provide only complete pre-contact tracks')
    points = mapping(obs.get('points'), 'observations.points')
    kind = points.get('kind')
    indices = points.get('indices')
    if kind not in ['com', 'nodes', 'material_points'] or not isinstance(indices, list) or not indices:
        raise ValueError('Explicit point kind com/nodes/material_points and indices required')
    indices = [integer(i, 'point index', 0, n if kind == 'nodes' else n-1) for i in indices]
    if (kind != 'material_points' and len(set(indices)) != len(indices)) or len(indices) != observed.shape[1]:
        raise ValueError('Point indices must be unique for COM/nodes and match observed points')
    points = {'kind': kind, 'indices': indices}
    if kind == 'material_points':
        fractions = numeric(obs['points'].get('fractions'), 'material fractions')
        if fractions.shape != (len(indices),) or np.any((fractions < 0) | (fractions > 1)):
            raise ValueError('One material fraction in [0,1] per selected body required')
        if len(set(zip(indices, fractions))) != len(indices):
            raise ValueError('Material (body, fraction) pairs must be unique')
        points['fractions'] = fractions.tolist()
    calibration = mapping(obs.get('calibration'), 'observations.calibration')
    expected_kind = 'synthetic_exact' if document['synthetic'] else 'calibrated_planar_video'
    if calibration.get('kind') != expected_kind:
        raise ValueError('Calibration kind must match synthetic/experimental provenance')
    for name in ['source', 'metric_mapping', 'time_origin_description', 'plane_assumption', 'uncertainty_note']:
        text(calibration.get(name), 'calibration.' + name)
    if not document['synthetic']:
        assets = calibration.get('assets')
        if not isinstance(assets, list) or not assets:
            raise ValueError('Experimental calibration requires original video/calibration asset references with SHA256')
        for asset in assets:
            asset = mapping(asset, 'calibration.assets[]')
            text(asset.get('reference'), 'asset.reference')
            digest = asset.get('sha256', '')
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('asset.sha256 must be a lowercase SHA256 digest')
    cfg = dict(mapping(document.get('identification'), 'identification'))
    for name in ['max_step_s', 'beta_tolerance_kg_s', 'trajectory_tolerance_m',
                 'beta_step_convergence_kg_s', 'residual_tolerance_m', 'energy_relative_tolerance',
                 'geometry_tolerance_m', 'sensitivity_fraction', 'initial_speed_perturbation_rad_s']:
        cfg[name] = scalar(cfg.get(name), name, positive=True)
    if cfg['sensitivity_fraction'] >= .5:
        raise ValueError('sensitivity_fraction must be less than .5')
    cfg['rtol'] = scalar(cfg.get('rtol'), 'rtol', positive=True)
    cfg['atol'] = scalar(cfg.get('atol'), 'atol', positive=True)
    if not 1e-13 <= cfg['rtol'] <= .001 or not 1e-15 <= cfg['atol'] <= .001:
        raise ValueError('Supported tolerances: rtol [1e-13,.001], atol [1e-15,.001]')
    cfg['grid_size'] = integer(cfg.get('grid_size'), 'grid_size', 5, 101)
    cfg['max_rhs_calls'] = integer(cfg.get('max_rhs_calls'), 'max_rhs_calls', 100, 1_000_000)
    if times[-1] / cfg['max_step_s'] > 500_000:
        raise ValueError('Requested integration exceeds minimum-step budget')
    bounds = numeric(cfg.get('bounds_kg_s'), 'bounds_kg_s')
    if bounds.shape != (2,) or not 0 <= bounds[0] < bounds[1]:
        raise ValueError('Finite beta bounds 0 <= lo < hi in kg/s required')
    cfg['bounds_kg_s'] = bounds.tolist()
    text(cfg.get('bounds_reason'), 'bounds_reason')
    text(cfg.get('tolerance_reason'), 'tolerance_reason')
    if cfg.get('method') not in ['Radau', 'DOP853', 'BDF']:
        raise ValueError('Explicit supported integration method required')
    cfg['max_forward_evaluations'] = integer(cfg.get('max_forward_evaluations', 512), 'max_forward_evaluations', 50, 2500)
    cfg['max_total_rhs_calls'] = integer(cfg.get('max_total_rhs_calls', 10_000_000), 'max_total_rhs_calls', 1000, 100_000_000)
    cfg['max_wall_time_s'] = scalar(cfg.get('max_wall_time_s', 600), 'max_wall_time_s', positive=True)
    if cfg['max_wall_time_s'] > 3600:
        raise ValueError('Workflow wall time limit cannot exceed 3600 s')
    normalized.update(effective_body_masses_kg=masses.tolist(), effective_body_inertias_kg_m2=inertias.tolist(),
                      load_weights=weights.tolist(), body_mode=mode, midpoint_assumption=assumption)
    return DynamicInput(chain, theta, omega, times, observed, points, cfg, normalized)
