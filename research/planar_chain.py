"""Research reconstruction of a planar hinged chain, never an app beta result.

Angles are absolute radians from the upward vertical: e_i=(sin(theta_i),
cos(theta_i)).  A midpoint COM is r_k=base+sum_i a_ki e_i, where a_ki is
l_i before body k, l_k/2 on k, and zero after k.  With
S_ij=sum_k m_k a_ki a_kj the exact Lagrange equations are

 M_ij = S_ij cos(theta_i-theta_j) + I_i delta_ij,
 C_i  = sum_j S_ij sin(theta_i-theta_j) omega_j**2,
 M alpha = Q_gravity + Q_springs + Q_drag - C.

Joint k=2..N has c_k=E*pi*d_joint,k**4/(64*uniform_length); no spring
connects body 1 to ground. Drag acts at COM: F_k=-beta*w_k*(v_k-air).
For moving air, E(t)-E(0)+integral(beta*w*|v-air|**2)-integral(F*air)=0;
energy need not decrease. Ground contact uses the zero-width centreline and
excludes the immobile base node. It is not a crown/finite-radius contact model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numbers
import time

import numpy as np
from scipy.integrate import solve_ivp

MODEL_VERSION = 'planar-hinge-chain-v1'
INTEGRATOR_VERSION = 'scipy-first-order-chain-energy-v1'


class ContactBeforeObservation(ValueError):
    """The requested observation window reaches/passes first ground contact."""


def _number(value, name, *, minimum=0.0, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f'{name} must be a real SI number, not a string/bool')
    value = float(value)
    if not math.isfinite(value) or (value <= minimum if positive else value < minimum):
        raise ValueError(f'{name} has invalid SI value')
    return value


def _array(value, name, shape=None):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in 'fiu':
            raise ValueError()
        if not isinstance(value, np.ndarray) and any(
                isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real)
                for x in np.asarray(value, dtype=object).flat):
            raise ValueError()
        result = raw.astype(float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must contain real numbers, not missing/string/bool values') from exc
    if not np.all(np.isfinite(result)) or (shape is not None and result.shape != shape):
        raise ValueError(f'{name} has nonfinite values or incorrect dimensions')
    result.setflags(write=False)
    return result


def cylinder_mass(density_kg_m3, diameter_m, length_m):
    """Physical mass for a diameter (not radius); includes the required /4."""
    density, diameter, length = (_array(v, name) for v, name in (
        (density_kg_m3, 'density'), (diameter_m, 'diameter'), (length_m, 'length')))
    if any(np.any(x <= 0) for x in (density, diameter, length)):
        raise ValueError('Cylinder density/diameter/length must be positive')
    with np.errstate(over='ignore', invalid='ignore'):
        value = density * math.pi * diameter**2 * length / 4
    if not np.all(np.isfinite(value)):
        raise ValueError('Cylinder mass overflow')
    return float(value) if value.ndim == 0 else value


def cylinder_inertia(mass_kg, diameter_m, length_m):
    """Transverse central inertia of the declared cylinder body."""
    mass, diameter, length = (_array(v, name) for v, name in (
        (mass_kg, 'mass'), (diameter_m, 'diameter'), (length_m, 'length')))
    if any(np.any(x <= 0) for x in (mass, diameter, length)):
        raise ValueError('Cylinder mass/diameter/length must be positive')
    with np.errstate(over='ignore', invalid='ignore'):
        value = mass * (length**2 / 12 + diameter**2 / 16)
    if not np.all(np.isfinite(value)):
        raise ValueError('Cylinder inertia overflow')
    return float(value) if value.ndim == 0 else value


def mass_weights(crown_masses_kg):
    masses = _array(crown_masses_kg, 'crown_masses_kg')
    if masses.ndim != 1 or not len(masses) or np.any(masses < 0) or masses.sum() <= 0:
        raise ValueError('Mass profile requires nonnegative crown masses with positive total')
    return masses / masses.sum()


def triangular_weights(lengths_m, crown_interval_m=None, *, peak_m=None):
    """Exact segment integrals of unit-area triangular density along the stem."""
    lengths = _array(lengths_m, 'lengths_m')
    if lengths.ndim != 1 or not len(lengths) or np.any(lengths <= 0):
        raise ValueError('Triangular profile requires positive segment lengths')
    total = float(lengths.sum())
    bounds = _array((0., total) if crown_interval_m is None else crown_interval_m,
                    'crown_interval_m', (2,))
    start, stop = bounds
    if not 0 <= start < stop <= total:
        raise ValueError('Crown interval must lie within [0,total length]')
    edge = np.concatenate(([0.], np.cumsum(lengths)))
    peak = (start + stop) / 2 if peak_m is None else _number(peak_m, 'peak_m')
    if not start <= peak <= stop:
        raise ValueError('Triangular peak must lie inside its support interval')
    q = np.clip((edge - start) / (stop - start), 0, 1)
    p = (peak - start) / (stop - start)
    if p == 0:
        cdf = 1 - (1 - q)**2
    elif p == 1:
        cdf = q**2
    else:
        cdf = np.where(q <= p, q**2 / p, 1 - (1 - q)**2 / (1 - p))
    return np.diff(cdf)


@dataclass(frozen=True)
class PlanarChain:
    lengths_m: object
    diameters_m: object
    masses_kg: object
    inertias_kg_m2: object
    elastic_modulus_pa: float
    joint_diameters_m: object
    weights: object
    base_m: object = (0.0, 0.0)
    gravity_m_s2: float = 9.81
    air_velocity_m_s: object = (0.0, 0.0)
    ground_z_m: float = 0.0
    support: str = 'fixed_hinge_free_tip'
    effective_body_assumption: str | None = None
    _a: np.ndarray = field(init=False, repr=False, compare=False)
    _s: np.ndarray = field(init=False, repr=False, compare=False)
    _h: np.ndarray = field(init=False, repr=False, compare=False)
    _c: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        lengths = _array(self.lengths_m, 'lengths_m')
        if lengths.ndim != 1 or not 1 <= len(lengths) <= 128 or np.any(lengths <= 0):
            raise ValueError('Require 1..128 positive body lengths')
        if not np.allclose(lengths, lengths[0], rtol=1e-12, atol=0):
            raise ValueError('This model version requires a uniform length grid')
        n = len(lengths)
        object.__setattr__(self, 'lengths_m', lengths)
        for name in ('diameters_m', 'masses_kg', 'inertias_kg_m2'):
            arr = _array(getattr(self, name), name, (n,))
            if np.any(arr <= 0):
                raise ValueError(f'{name} must be positive for every body')
            object.__setattr__(self, name, arr)
        joints = _array(self.joint_diameters_m, 'joint_diameters_m', (n - 1,))
        if np.any(joints <= 0):
            raise ValueError('joint_diameters_m must be positive')
        object.__setattr__(self, 'joint_diameters_m', joints)
        weights = _array(self.weights, 'weights', (n,))
        if np.any(weights < 0) or not math.isclose(float(weights.sum()), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Nonnegative damping weights must sum to one')
        object.__setattr__(self, 'weights', weights)
        object.__setattr__(self, 'base_m', _array(self.base_m, 'base_m', (2,)))
        object.__setattr__(self, 'air_velocity_m_s', _array(self.air_velocity_m_s, 'air_velocity_m_s', (2,)))
        object.__setattr__(self, 'elastic_modulus_pa', _number(self.elastic_modulus_pa, 'E', positive=True))
        object.__setattr__(self, 'gravity_m_s2', _number(self.gravity_m_s2, 'gravity'))
        # A height may legitimately be negative in a translated laboratory frame.
        ground = _array([self.ground_z_m], 'ground_z_m', (1,))[0]
        object.__setattr__(self, 'ground_z_m', float(ground))
        if self.base_m[1] < self.ground_z_m:
            raise ValueError('The fixed hinge must not lie below the ground plane')
        if self.support != 'fixed_hinge_free_tip':
            raise ValueError('Only fixed_hinge_free_tip is reconstructed; no root clamp/undercut model')
        if self.effective_body_assumption is not None and (
                not isinstance(self.effective_body_assumption, str) or not self.effective_body_assumption.strip()):
            raise ValueError('An effective-body assumption must be an explicit nonempty statement')
        a = np.tril(np.broadcast_to(lengths, (n, n)).copy(), -1) + np.diag(lengths / 2)
        s = a.T @ (self.masses_kg[:, None] * a)
        h = a.T @ self.masses_kg
        c = self.elastic_modulus_pa * math.pi * joints**4 / (64 * lengths[0])
        if not all(np.all(np.isfinite(x)) for x in (a, s, h, c)):
            raise ValueError('Body/joint parameters overflow the SI model')
        for name, arr in (('_a', a), ('_s', s), ('_h', h), ('_c', c)):
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)

    @property
    def n(self):
        return len(self.lengths_m)

    @property
    def joint_stiffness_nm(self):
        return self._c.copy()

    def _state(self, theta, omega=None):
        theta = _array(theta, 'theta', (self.n,))
        return theta if omega is None else (theta, _array(omega, 'omega', (self.n,)))

    def kinematics(self, theta, omega=None):
        theta = self._state(theta)
        e = np.stack((np.sin(theta), np.cos(theta)), axis=1)
        u = np.stack((np.cos(theta), -np.sin(theta)), axis=1)
        nodes = np.vstack((self.base_m, self.base_m + np.cumsum(self.lengths_m[:, None] * e, axis=0)))
        com = self.base_m + self._a @ e
        jacobian = self._a[:, :, None] * u[None, :, :]
        velocity = None if omega is None else self._a @ (u * self._state(theta, omega)[1][:, None])
        return nodes, com, jacobian, velocity

    def mass_matrix(self, theta):
        theta = self._state(theta)
        return self._s * np.cos(theta[:, None] - theta[None, :]) + np.diag(self.inertias_kg_m2)

    def forces(self, theta, omega, beta):
        theta, omega = self._state(theta, omega)
        beta = _number(beta, 'beta', minimum=0)
        delta = theta[:, None] - theta[None, :]
        centrifugal = (self._s * np.sin(delta)) @ omega**2
        gravity = self.gravity_m_s2 * self._h * np.sin(theta)
        spring = np.zeros(self.n)
        torque = self._c * np.diff(theta)
        spring[:-1] += torque
        spring[1:] -= torque
        _, _, j, v = self.kinematics(theta, omega)
        relative = v - self.air_velocity_m_s
        drag_forces = -beta * self.weights[:, None] * relative
        drag = np.einsum('kic,kc->i', j, drag_forces)
        return {
            'centrifugal': centrifugal, 'gravity': gravity, 'springs': spring, 'drag': drag,
            'drag_forces_n': drag_forces,
            'relative_dissipation_w': float(beta * np.sum(self.weights[:, None] * relative**2)),
            'air_work_w': float(np.sum(drag_forces * self.air_velocity_m_s)),
        }

    def acceleration(self, theta, omega, beta):
        f = self.forces(theta, omega, beta)
        return _solve_mass(self.mass_matrix(theta), f['gravity'] + f['springs'] + f['drag'] - f['centrifugal'])

    def energy(self, theta, omega):
        theta, omega = self._state(theta, omega)
        _, com, _, _ = self.kinematics(theta)
        return float(.5 * omega @ self.mass_matrix(theta) @ omega
                     + .5 * np.sum(self._c * np.diff(theta)**2)
                     + self.gravity_m_s2 * np.sum(self.masses_kg * (com[:, 1] - self.base_m[1])))


@dataclass(frozen=True)
class SimulationResult:
    times_s: np.ndarray
    theta_rad: np.ndarray
    omega_rad_s: np.ndarray
    nodes_m: np.ndarray
    com_m: np.ndarray
    energy_j: np.ndarray
    dissipation_j: np.ndarray
    air_work_j: np.ndarray
    contact_time_s: float | None
    truncated: bool
    diagnostics: dict


def _solve_mass(matrix, force):
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(force)):
        raise ValueError('Nonfinite mass matrix/force')
    try:
        value = np.linalg.solve(matrix, force)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Degenerate mass matrix; no admissible trajectory') from exc
    if not np.all(np.isfinite(value)):
        raise ValueError('Nonfinite angular acceleration')
    return value


def simulate(chain, theta0, omega0, t, beta, max_step_s, rtol=1e-7, atol=1e-9,
             method='Radau', max_rhs_calls=200000):
    """Integrate until requested end or first centreline-ground contact.

    Result may contain only the pre-contact prefix; observation_coordinates
    refuses this prefix so SSE is never computed over a shorter window.
    """
    if not isinstance(chain, PlanarChain):
        raise ValueError('A validated PlanarChain is required')
    theta0, omega0 = chain._state(theta0, omega0)
    t = _array(t, 'times_s')
    if t.ndim != 1 or not 2 <= len(t) <= 10000 or t[0] != 0 or np.any(np.diff(t) <= 0) or t[-1] > 120:
        raise ValueError('Require 2..10000 increasing times starting at zero, with duration <=120 s')
    beta = _number(beta, 'beta')
    step = _number(max_step_s, 'max_step_s', positive=True)
    rtol = _number(rtol, 'rtol', positive=True)
    atol = _number(atol, 'atol', positive=True)
    if rtol > .01 or atol > .01:
        raise ValueError('Integrator tolerances must be <=0.01')
    if method not in ('Radau', 'BDF', 'DOP853'):
        raise ValueError('Integrator must explicitly be Radau, BDF or DOP853')
    if isinstance(max_rhs_calls, (bool, np.bool_)) or not isinstance(max_rhs_calls, numbers.Integral) or not 10 <= max_rhs_calls <= 1000000:
        raise ValueError('max_rhs_calls must be an integer in [10,1000000]')
    if math.ceil(float(t[-1]) / step) > 500000:
        raise ValueError('Requested integration step exceeds the bounded step budget')
    nodes0 = chain.kinematics(theta0)[0]
    if np.min(nodes0[1:, 1]) <= chain.ground_z_m + 1e-12:
        raise ContactBeforeObservation('Initial state already reaches/passes centreline ground contact')
    initial_condition = float(np.linalg.cond(chain.mass_matrix(theta0)))
    if not math.isfinite(initial_condition) or initial_condition > 1e12:
        raise ValueError('Initial mass matrix condition exceeds the research safety limit 1e12')
    y0 = np.concatenate((theta0, omega0, [0., 0.]))
    calls = 0

    def rhs(_time, y):
        nonlocal calls
        calls += 1
        if calls > max_rhs_calls:
            raise ValueError('Integration RHS budget exceeded; no partial trajectory is admissible')
        theta, omega = y[:chain.n], y[chain.n:2 * chain.n]
        f = chain.forces(theta, omega, beta)
        alpha = _solve_mass(chain.mass_matrix(theta), f['gravity'] + f['springs'] + f['drag'] - f['centrifugal'])
        value = np.concatenate((omega, alpha, [f['relative_dissipation_w'], f['air_work_w']]))
        if not np.all(np.isfinite(value)):
            raise ValueError('Integration became nonfinite')
        return value

    def contact(_time, y):
        return float(np.min(chain.kinematics(y[:chain.n])[0][1:, 1]) - chain.ground_z_m)

    contact.terminal = True
    contact.direction = -1
    started = time.perf_counter()
    solution = solve_ivp(rhs, (0., float(t[-1])), y0, method=method,
                         dense_output=True, max_step=step, rtol=rtol, atol=atol, events=contact)
    elapsed = time.perf_counter() - started
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise ValueError('Integration failed; no partial trajectory is admissible: ' + solution.message)
    # solve_ivp finds sign crossings per accepted step. Reject unresolved angular
    # steps instead of silently claiming that rapid multiple crossings were found.
    widths = np.diff(solution.t)
    midpoint_y = solution.sol((solution.t[1:] + solution.t[:-1]) / 2)
    speed_bound = np.maximum.reduce([
        abs(solution.y[chain.n:2 * chain.n, :-1]),
        abs(solution.y[chain.n:2 * chain.n, 1:]),
        abs(midpoint_y[chain.n:2 * chain.n]),
    ])
    max_angle_step = float(np.max(widths[None, :] * speed_bound)) if len(widths) else 0.
    if max_angle_step > math.pi / 8:
        raise ValueError('Contact angular resolution exceeded pi/8; reduce max_step_s/tolerances')
    contact_time = float(solution.t_events[0][0]) if len(solution.t_events[0]) else None
    if contact_time is not None and contact_time <= 1e-12:
        raise ContactBeforeObservation('Ground contact occurs before any admissible observation interval')
    truncated = contact_time is not None
    requested = t if not truncated else t[t < contact_time - 1e-12]
    if not truncated and solution.t[-1] < t[-1]:
        raise ValueError('Integrator did not reach the complete requested window')
    y = solution.sol(requested).T
    theta, omega = y[:, :chain.n], y[:, chain.n:2 * chain.n]
    nodes, com, energy, conditions = [], [], [], []
    for angles, angular_velocity in zip(theta, omega):
        node, centre, _, _ = chain.kinematics(angles)
        nodes.append(node)
        com.append(centre)
        energy.append(chain.energy(angles, angular_velocity))
        conditions.append(float(np.linalg.cond(chain.mass_matrix(angles))))
    if not all(math.isfinite(value) and value <= 1e12 for value in conditions):
        raise ValueError('Sampled mass matrix condition exceeds the research safety limit 1e12')
    nodes, com = np.asarray(nodes), np.asarray(com)
    direction = np.stack((np.sin(theta), np.cos(theta)), axis=-1)
    lower = com - .5 * chain.lengths_m[None, :, None] * direction
    upper = com + .5 * chain.lengths_m[None, :, None] * direction
    length_error = float(np.max(abs(np.linalg.norm(np.diff(nodes, axis=1), axis=2) - chain.lengths_m)))
    shared_error = float(np.max(np.linalg.norm(upper[:, :-1] - lower[:, 1:], axis=2))) if chain.n > 1 else 0.
    midpoint_error = float(np.max(np.linalg.norm(com - .5 * (nodes[:, :-1] + nodes[:, 1:]), axis=2)))
    energy = np.asarray(energy)
    dissipation, air_work = y[:, -2], y[:, -1]
    initial_energy = chain.energy(theta0, omega0)
    balance = energy - initial_energy + dissipation - air_work
    end = solution.y[:, -1]
    end_energy = chain.energy(end[:chain.n], end[chain.n:2 * chain.n])
    max_balance = max(float(np.max(abs(balance))), abs(end_energy - initial_energy + end[-2] - end[-1]))
    energy_scale = max(1., abs(initial_energy), abs(end_energy), abs(float(end[-2])), abs(float(end[-1])))
    diagnostics = {
        'model_version': MODEL_VERSION, 'integrator_version': INTEGRATOR_VERSION,
        'method': method, 'max_step_s': step, 'rtol': rtol, 'atol': atol,
        'rhs_calls': calls, 'nfev': int(solution.nfev), 'njev': int(solution.njev),
        'nlu': int(solution.nlu), 'accepted_steps': len(solution.t) - 1,
        'elapsed_s': elapsed, 'max_angle_step_rad': max_angle_step,
        'energy_balance_max_abs_j': max_balance, 'energy_balance_relative': max_balance / energy_scale,
        'max_length_error_m': length_error, 'shared_joint_error_m': shared_error,
        'max_com_midpoint_error_m': midpoint_error,
        'initial_mass_matrix_condition': initial_condition,
        'max_sampled_mass_matrix_condition': max(conditions),
        'contact_model': 'zero_width_centreline_first_ground_contact_excluding_fixed_base',
        'effective_body_assumption': chain.effective_body_assumption,
        'experimental_validation': False,
    }
    arrays = [requested.copy(), theta.copy(), omega.copy(), nodes, com,
              energy, dissipation.copy(), air_work.copy()]
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError('Nonfinite trajectory/output')
    for arr in arrays:
        arr.setflags(write=False)
    return SimulationResult(*arrays, contact_time, truncated, diagnostics)


def observation_coordinates(result, kind, indices=None, fractions=None):
    """Select declared physical points; never invent missing observations.

    Indices are zero based: COM/material body 0..N-1, nodes 0..N (base included).
    Material fraction 0 is the body's lower joint, 1 its upper joint.
    """
    if not isinstance(result, SimulationResult):
        raise ValueError('A SimulationResult is required')
    if result.truncated:
        raise ContactBeforeObservation('Requested observations reach/pass first ground contact; shorten the declared window')
    if kind not in ('com', 'nodes', 'material_points'):
        raise ValueError('Observation kind must be com, nodes or material_points')
    source = result.com_m if kind == 'com' else result.nodes_m
    count = result.com_m.shape[1] if kind == 'material_points' else source.shape[1]
    if indices is None:
        if kind == 'material_points':
            raise ValueError('Material-point body indices and fractions are required')
        index = np.arange(count)
    else:
        raw = np.asarray(indices)
        if raw.ndim != 1 or not len(raw) or raw.dtype.kind not in 'iu' or np.any(raw < 0) or np.any(raw >= count) or (kind != 'material_points' and len(np.unique(raw)) != len(raw)):
            raise ValueError('Observation indices must be in-range integers (unique for COM/nodes)')
        index = raw.astype(int)
    if kind != 'material_points':
        if fractions is not None:
            raise ValueError('Fractions apply only to material_points')
        return source[:, index, :].copy()
    fraction = _array(fractions, 'material point fractions', (len(index),))
    if np.any((fraction < 0) | (fraction > 1)):
        raise ValueError('Material fractions must lie in [0,1]')
    return (result.nodes_m[:, index, :] + fraction[None, :, None]
            * (result.nodes_m[:, index + 1, :] - result.nodes_m[:, index, :]))


def uniform_cylinder_chain(length_m, diameter_m, density_kg_m3, elastic_modulus_pa, n,
                           crown_mass_kg=0., profile='triangular', crown_interval_m=None,
                           effective_body_assumption=None, **kwargs):
    """Uniform cylinder mesh with normalized *integrated* crown/damping profile.

    Triangle peak is the interval midpoint; continuous integral is one, so total
    beta and crown mass do not multiply with N. With crown added, using midpoint
    COM and cylinder inertia of combined mass is an explicit effective-body
    approximation, never silently the physical crown inertia.
    """
    length = _number(length_m, 'length_m', positive=True)
    diameter = _number(diameter_m, 'diameter_m', positive=True)
    density = _number(density_kg_m3, 'density_kg_m3', positive=True)
    crown = _number(crown_mass_kg, 'crown_mass_kg')
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, numbers.Integral) or not 1 <= n <= 128:
        raise ValueError('n must be an integer in [1,128]')
    if crown > 0 and (not isinstance(effective_body_assumption, str) or not effective_body_assumption.strip()):
        raise ValueError('Added crown mass requires an explicit effective_body_assumption')
    if profile not in ('mass', 'triangular'):
        raise ValueError('Factory profile must be mass or triangular')
    if profile == 'mass' and crown <= 0:
        raise ValueError('A crown-mass profile requires a positive declared crown mass')
    bounds = _array((0., length) if crown_interval_m is None else crown_interval_m,
                    'crown_interval_m', (2,))
    start, stop = bounds
    if not 0 <= start < stop <= length:
        raise ValueError('Crown interval must lie within [0,length] and have positive length')
    edge = np.linspace(0, length, n + 1)
    q = np.clip((edge - start) / (stop - start), 0, 1)
    # Exact normalized integrals. Mass profile distributes the declared mass
    # uniformly along its support; triangle is a new normalized parametrization.
    cdf = q if profile == 'mass' else np.where(q <= .5, 2 * q**2, 1 - 2 * (1 - q)**2)
    weights = np.diff(cdf) if profile == 'mass' else triangular_weights(np.full(n, length / n), bounds)
    lengths = np.full(n, length / n)
    wood_mass = density * math.pi * diameter**2 * lengths / 4
    masses = wood_mass + crown * weights
    inertia = masses * (lengths**2 / 12 + diameter**2 / 16)
    return PlanarChain(lengths, np.full(n, diameter), masses, inertia, elastic_modulus_pa,
                       np.full(n - 1, diameter), weights,
                       effective_body_assumption=effective_body_assumption, **kwargs)
