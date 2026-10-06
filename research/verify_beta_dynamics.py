"""Reproducible AS-10 numerical checks; all generated observations are SYNTHETIC.

Run: python -m research.verify_beta_dynamics NEW_OUTPUT_DIRECTORY
No network, production, application, database or training operations are used.
The coupled reference uses the same reconstructed physics with tighter numerical
controls, not an independently validated physical tree model. Independent analytic
one/two-body oracles separately check equations. --publish-research-evidence copies
only the generated example, numerical summary and four marked-synthetic PNGs into
research/; the default preserves the repository and never overwrites an output run.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

from .beta_dynamics import run_document, versions
from .beta_inputs import FRAME, MODEL, UNITS, parse_document
from .planar_chain import (
    PlanarChain, cylinder_inertia, cylinder_mass, observation_coordinates,
    simulate, triangular_weights, uniform_cylinder_chain,
)

VERIFICATION_VERSION = 'as10-synthetic-numerical-protocol-v1'
REFERENCE_SCOPE = ('Same reconstructed equations, tighter numerical controls; '
                   'not independent physical truth or experimental validation')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def quantity(value, unit, source, uncertainty=0):
    return {'value': np.asarray(value).tolist(), 'unit': unit, 'source': source,
            'standard_uncertainty': uncertainty}


def example_document(*, n=3):
    """Generate a complete dense pre-contact package, including reference provenance."""
    large = n == 20
    if n not in (3, 20):
        raise ValueError('This fixed verification protocol generates only n=3 or n=20')
    length, diameter, density = (20., .2, 780.) if large else (3., .08, 650.)
    e, true_beta = (1.2e10, 80.) if large else (5e6, 2.4)
    lengths, diameters = np.full(n, length / n), np.full(n, diameter)
    crown = 100 * triangular_weights(lengths, (10., 20.), peak_m=10.) if large else np.array([.5, 1., 1.5])
    masses = cylinder_mass(density, diameters, lengths) + crown
    theta = np.full(n, .1) if large else np.array([.15, .18, .2])
    omega = np.zeros(n) if large else np.array([.2, .1, .05])
    times = np.linspace(0, .05 if large else .4, 6 if large else 17)
    assumption = ('SYNTHETIC effective body: full wood+crown mass is placed at each '
                  'segment midpoint; transverse inertia uses the combined cylinder. '
                  'This is explicitly not measured crown COM or crown inertia.')
    chain = PlanarChain(lengths, diameters, masses, cylinder_inertia(masses, diameters, lengths),
                        e, np.full(n - 1, diameter), crown / crown.sum(),
                        effective_body_assumption=assumption)
    reference_cfg = {'method': 'Radau' if large else 'DOP853',
                     'max_step_s': .00025 if large else .0017,
                     'rtol': 1e-10 if large else 2e-12,
                     'atol': 1e-12 if large else 2e-14}
    reference = simulate(chain, theta, omega, times, true_beta, **reference_cfg)
    points = {'kind': 'com', 'indices': list(range(n))}
    positions = observation_coordinates(reference, **points)
    src = 'Explicit synthetic fixture chosen by verify_beta_dynamics.py; not a measured tree'
    wood_source = ('Correct physical cylinder mass rho*pi*d^2*length/4 plus the '
                   'explicit synthetic crown masses; not the printed dissertation typo')
    parameters = {
        'lengths_m': quantity(lengths, 'm', src),
        'diameters_m': quantity(diameters, 'm', src),
        'joint_diameters_m': quantity(np.full(n - 1, diameter), 'm',
            'Synthetic uniform cylinder: explicitly specified joint diameters, not inferred from midpoint taper'),
        'elastic_modulus_pa': quantity(e, 'Pa', src + ('; source-scale benchmark E=1.2e10, not a species constant' if large else '')),
        'masses_kg': quantity(masses, 'kg', wood_source),
        'density_kg_m3': quantity(density, 'kg/m^3', src),
        'base_position_m': quantity([0., 0.], 'm', 'Synthetic fixed hinge at the world origin'),
        'gravity_m_s2': quantity(9.81, 'm/s^2', 'Declared synthetic constant gravity'),
        'ground_z_m': quantity(0., 'm', 'Synthetic horizontal zero-width centreline contact plane'),
        'air_velocity_m_s': quantity([0., 0.], 'm/s', 'Synthetic still air, explicitly specified'),
        'initial_angles_rad': quantity(theta, 'rad', src),
        'initial_angular_velocities_rad_s': quantity(omega, 'rad/s',
            src + '; supplied state, not inferred or silently defaulted'),
    }
    cfg = {
        'bounds_kg_s': [60., 100.] if large else [0., 6.],
        'bounds_reason': ('Synthetic truth80 lies inside a deliberately narrow benchmark interval; '
                         'not experimentally established bounds' if large else
                         'Synthetic truth2.4 lies inside [0,6]; not a species-based beta lookup'),
        'max_step_s': .001 if large else .012,
        'beta_tolerance_kg_s': .001 if large else .0001,
        'trajectory_tolerance_m': 1e-7,
        'beta_step_convergence_kg_s': .02 if large else .001,
        'residual_tolerance_m': 1e-7,
        'energy_relative_tolerance': 1e-7, 'geometry_tolerance_m': 1e-10,
        'sensitivity_fraction': .05, 'initial_speed_perturbation_rad_s': .01,
        'rtol': 1e-7 if large else 1e-8, 'atol': 1e-9 if large else 1e-10,
        'grid_size': 5 if large else 9, 'max_rhs_calls': 20000,
        'method': 'Radau',
        'tolerance_reason': 'Explicit numerical/synthetic protocol limits; not experimentally proven measurement accuracy',
    }
    # Large-case resolution intentionally exceeds its very short-window signal.
    resolution = 1e-5 if large else 0.
    doc = {
        'schema_version': 2, 'model': MODEL, 'synthetic': True,
        'dataset_id': 'SYNTHETIC-AS10-N20-50MS' if large else 'SYNTHETIC-AS10-THREE-BODY',
        'units': dict(UNITS), 'frame': copy.deepcopy(FRAME), 'support': 'fixed_hinge_free_tip',
        'air_model': 'still', 'initial_velocity_mode': 'supplied',
        'body': {'mode': 'effective_midpoint_cylinders', 'midpoint_assumption': assumption,
                 'effective_body_assumption': assumption},
        'parameters': parameters,
        'load_profile': {'kind': 'crown_mass_fractions', 'source': src,
                         'crown_masses_kg': quantity(crown, 'kg', src)},
        'observations': {
            'time_origin': 'initial_state', 'times_s': quantity(times, 's', src),
            'positions_m': quantity(positions, 'm', 'Synthetic tighter numerical reference: ' + REFERENCE_SCOPE,
                                    uncertainty=resolution),
            'points': points, 'visibility': np.ones(positions.shape[:2], dtype=bool).tolist(),
            'calibration': {'kind': 'synthetic_exact', 'source': 'Generated reference; no video/field measurements',
                'metric_mapping': 'Coordinates are already exact-model world x,z metres',
                'time_origin_description': 't=0 is exactly the supplied initial angle/velocity state',
                'plane_assumption': 'All bodies and COM tracks are in a declared mathematical x,z plane',
                'uncertainty_note': ('1e-5 m is an explicit hypothetical resolution threshold for testing '
                    'weak identifiability of the 50ms window, not measured instrument uncertainty' if large else
                    'No observational noise; discrepancy comes only from numerical reference error')}},
        'identification': cfg,
        'synthetic_truth': {'beta_kg_s': true_beta, 'not_dissertation_beta_max': True,
                            'not_species_parameter': True, 'experimental_validation': False},
        'reference_generation': {'script': 'research/verify_beta_dynamics.py',
            'script_sha256': sha(__file__), 'versions': versions(), 'integrator': reference_cfg,
            'scope': REFERENCE_SCOPE, 'forward_diagnostics': reference.diagnostics,
            'wood_density_kg_m3': density, 'wood_total_mass_kg': float(cylinder_mass(density, diameter, length)),
            'crown_total_mass_kg': float(crown.sum())},
    }
    parse_document(doc)
    return doc


def _fit_summary(result, truth):
    return {'status': result['status'], 'beta_kg_s': result['beta_kg_s'],
            'candidate_beta_kg_s': result['candidate_beta_kg_s'], 'truth_beta_kg_s': truth,
            'absolute_candidate_error_kg_s': abs(result['candidate_beta_kg_s'] - truth),
            'sse_m2': result['sse_m2'], 'rmse_coordinate_m': result['rmse_coordinate_m'],
            'time_refinement': result['time_refinement'], 'diagnostics': result['diagnostics'],
            'forward_diagnostics': result['forward_diagnostics'],
            'forward_evaluations': result['forward_evaluations'], 'wall_time_s': result['wall_time_s'],
            'sensitivity': result['sensitivity'], 'versions': result['versions'],
            'input_sha256': result['input_sha256'], 'experimental_validation': False}


def _cli(input_path, output_path):
    started = time.perf_counter()
    run = subprocess.run([sys.executable, '-m', 'research.beta_dynamics', str(input_path), str(output_path)],
                         capture_output=True, text=True, encoding='utf-8', errors='strict', timeout=900)
    (output_path.parent / (output_path.name + '-cli.stdout.txt')).write_text(run.stdout, encoding='utf-8')
    (output_path.parent / (output_path.name + '-cli.stderr.txt')).write_text(run.stderr, encoding='utf-8')
    if run.returncode != 0:
        raise RuntimeError(f'Full CLI failed ({run.returncode}); inspect {output_path.name}-cli.stderr.txt')
    complete = json.loads((output_path / 'COMPLETE.json').read_text(encoding='utf-8'))
    for name, digest in complete['files_sha256'].items():
        if sha(output_path / name) != digest:
            raise RuntimeError('CLI output manifest hash mismatch: ' + name)
    result = json.loads((output_path / 'result.json').read_text(encoding='utf-8'))
    return result, {'returncode': run.returncode, 'subprocess_wall_time_s': time.perf_counter()-started,
                    'verified_manifest_files': len(complete['files_sha256']),
                    'input_file_sha256': sha(input_path), 'result_file_sha256': sha(output_path/'result.json'),
                    'png_sha256': {name: sha(output_path/name) for name in
                                   ('comparison.png', 'residuals.png', 'objective.png', 'energy.png')}}


def numerical_checks():
    """Independent analytic equations, energy and same-physical-point grid checks."""
    one = PlanarChain([1.], [.02], [1.], [1/12], 1., [], [1.], gravity_m_s2=0.)
    times = np.linspace(0, .4, 9)
    decay = 2 * .25 / (1/3)
    exact_theta = .1 + .4 * (1 - np.exp(-decay * times)) / decay
    exact_omega = .4 * np.exp(-decay * times)
    step_checks = []
    for step in (.2, .1, .05):
        r = simulate(one, [.1], [.4], times, 2., step, rtol=1e-3, atol=1e-5)
        step_checks.append({'max_step_s': step,
            'max_theta_error_rad': float(np.max(abs(r.theta_rad[:, 0]-exact_theta))),
            'max_omega_error_rad_s': float(np.max(abs(r.omega_rad_s[:, 0]-exact_omega)))})
    if not step_checks[2]['max_theta_error_rad'] < step_checks[1]['max_theta_error_rad'] < step_checks[0]['max_theta_error_rad']:
        raise RuntimeError('One-rod step convergence did not pass')
    length, mass, diameter, beta, g = 1.2, 2.3, .08, 2.4, 9.81
    inertia = mass * (length**2 / 12 + diameter**2 / 16)
    pivot = inertia + mass * length**2 / 4
    pendulum_times = np.linspace(0, .3, 13)
    def pendulum_rhs(_t, y):
        angle, speed = y
        return [speed, (mass*g*length/2 * math.sin(angle) - beta*length**2/4 * speed)/pivot]
    pendulum_truth = solve_ivp(pendulum_rhs, (0., .3), [.2, .1], method='DOP853',
        t_eval=pendulum_times, max_step=.0007, rtol=2e-12, atol=2e-14)
    pendulum = PlanarChain([length], [diameter], [mass], [inertia], 1e7, [], [1.])
    pendulum_result = simulate(pendulum, [.2], [.1], pendulum_times, beta, .01, rtol=1e-9, atol=1e-11)
    pendulum_errors = {
        'max_theta_error_rad': float(np.max(abs(pendulum_result.theta_rad[:, 0]-pendulum_truth.y[0]))),
        'max_omega_error_rad_s': float(np.max(abs(pendulum_result.omega_rad_s[:, 0]-pendulum_truth.y[1]))),
        'reference': 'Independent scalar fixed-hinge pendulum ODE, DOP853 rtol2e-12 atol2e-14 max_step0.0007',
        'mass_kg': mass, 'length_m': length, 'central_inertia_kg_m2': inertia,
        'beta_kg_s': beta, 'gravity_m_s2': g, 'duration_s': .3,
    }
    if not pendulum_truth.success or max(pendulum_errors[k] for k in ('max_theta_error_rad', 'max_omega_error_rad_s')) > 5e-10:
        raise RuntimeError('Independent gravity-driven pendulum trajectory did not pass')
    two = PlanarChain([1., 1.], [.02, .02], [1., 1.], [1/12, 1/12], 64/math.pi, [1.], [.5, .5])
    expected = [1.5 + 3*math.pi/8, -1.5 - 3*math.pi/2 + 14.715]
    acceleration_error = float(np.max(abs(two.acceleration([0., math.pi/2], [1., 2.], 0.)-expected)))
    if acceleration_error > 1e-11:
        raise RuntimeError('Two-rod hand-derived acceleration did not pass')
    energies = []
    for beta, air in [(0., (0., 0.)), (.8, (0., 0.)), (.8, (2., -.1))]:
        chain = PlanarChain([1., 1.], [.02, .02], [1., 1.], [1/12, 1/12], 64/math.pi,
                            [1.], [.5, .5], air_velocity_m_s=air)
        r = simulate(chain, [.12, .18], [.1, -.05], np.linspace(0, .2, 21), beta, .005,
                     rtol=2e-9, atol=1e-11)
        balance = float(np.max(abs(r.energy_j-r.energy_j[0]+r.dissipation_j-r.air_work_j)))
        if balance > 2e-8:
            raise RuntimeError('Pre-contact energy/work balance did not pass')
        energies.append({'beta_kg_s': beta, 'air_velocity_m_s': list(air),
            'max_balance_error_j': balance, 'energy_change_j': float(r.energy_j[-1]-r.energy_j[0]),
            'relative_dissipation_j': float(r.dissipation_j[-1]), 'air_work_j': float(r.air_work_j[-1])})
    tips, meshes = [], []
    for n in (4, 8, 16):
        chain = uniform_cylinder_chain(1., .05, 780., 1e5, n)
        r = simulate(chain, np.full(n, .25), np.zeros(n), np.linspace(0, .08, 9), .2,
                     .004, rtol=1e-8, atol=1e-10)
        tips.append(observation_coordinates(r, 'material_points', [n-1], [1.]))
        meshes.append({'n': n, 'total_mass_kg': float(chain.masses_kg.sum()),
                       'sum_weights': float(chain.weights.sum()), 'forward_diagnostics': r.diagnostics})
    deltas = [float(np.max(np.linalg.norm(tips[i]-tips[i+1], axis=2))) for i in (0, 1)]
    if deltas[1] >= deltas[0]/3:
        raise RuntimeError('Physical-tip mesh convergence did not pass')
    return {'one_rod_analytic_reference': {'pivot_inertia_kg_m2': 1/3,
                'decay_rate_per_s': decay, 'step_checks': step_checks},
            'two_rod_hand_derived_acceleration': {'expected_rad_s2': expected,
                'max_error_rad_s2': acceleration_error},
            'gravity_driven_physical_pendulum': pendulum_errors,
            'energy_work_checks': energies,
            'spatial_refinement': {'meaning': 'Same physical tip, not changing COM indices',
                'meshes': meshes, 'adjacent_max_tip_delta_m': deltas}}


def verify(output):
    started = time.perf_counter()
    checks = numerical_checks()
    document = example_document()
    input_path = output / 'beta_chain_synthetic_v2.json'
    write_json(input_path, document)
    clean, clean_cli = _cli(input_path, output/'clean-cli')
    clean_summary = _fit_summary(clean, 2.4)
    if clean_summary['absolute_candidate_error_kg_s'] > .001 or not clean['numerically_converged']:
        raise RuntimeError('Clean synthetic inverse recovery did not pass')
    # Every variant preserves the original generated observations/provenance.
    variants = {}
    noisy = copy.deepcopy(document)
    sigma, seed = 1e-4, 731005
    rng = np.random.default_rng(seed)
    observed = np.asarray(noisy['observations']['positions_m']['value'])
    noisy['observations']['positions_m']['value'] = (observed + rng.normal(0, sigma, observed.shape)).tolist()
    noisy['observations']['positions_m']['standard_uncertainty'] = sigma
    noisy['observations']['positions_m']['source'] += f'; added synthetic independent Gaussian noise seed={seed}, sigma={sigma}m'
    noisy['observations']['calibration']['uncertainty_note'] = 'Artificial independent Gaussian coordinate noise; no field/video measurement'
    noisy['dataset_id'] += '-NOISE'
    noisy['identification']['residual_tolerance_m'] = sigma * 5
    noise_result = run_document(noisy)
    variants['noise'] = {**_fit_summary(noise_result, 2.4), 'seed': seed, 'sigma_m': sigma}
    write_json(output/'noise-input.json', noisy)
    write_json(output/'noise-result.json', noise_result)
    for name, parameter, factor, additive in [
        ('wrong_elastic_modulus', 'elastic_modulus_pa', 1.2, 0.),
        ('wrong_full_masses', 'masses_kg', 1.1, 0.),
        ('wrong_initial_angular_velocity', 'initial_angular_velocities_rad_s', 1., .1)]:
        altered = copy.deepcopy(document)
        altered['parameters'][parameter]['value'] = (np.asarray(altered['parameters'][parameter]['value']) * factor + additive).tolist()
        altered['parameters'][parameter]['source'] = 'Deliberately wrong synthetic model assumption; reference observations retain their original true parameters'
        altered['dataset_id'] += '-' + name.upper()
        result = run_document(altered)
        variants[name] = {**_fit_summary(result, 2.4), 'factor': factor, 'additive': additive,
                          'observations_sha256': hashlib.sha256(np.asarray(observed).tobytes()).hexdigest()}
        write_json(output/(name+'-input.json'), altered)
        write_json(output/(name+'-result.json'), result)
    # Unit conversion is checked against the same physical forward trajectory.
    converted = copy.deepcopy(document)
    for container in [converted['parameters'], converted['observations']]:
        for entry in container.values():
            if isinstance(entry, dict) and entry.get('unit') == 'm':
                entry['value'] = (np.asarray(entry['value']) * 100).tolist()
                if entry['standard_uncertainty'] is not None:
                    entry['standard_uncertainty'] = (np.asarray(entry['standard_uncertainty']) * 100).tolist()
                entry['unit'] = 'cm'
    original_data, converted_data = parse_document(document), parse_document(converted)
    conversion_error = float(np.max(abs(original_data.observed-converted_data.observed)))
    if conversion_error > 1e-14 or not np.allclose(original_data.chain.lengths_m, converted_data.chain.lengths_m, rtol=0, atol=1e-14):
        raise RuntimeError('Unit normalization did not preserve physical input')
    large_document = example_document(n=20)
    write_json(output/'n20-source-scale-input.json', large_document)
    large, large_cli = _cli(output/'n20-source-scale-input.json', output/'n20-cli')
    return {'verification_version': VERIFICATION_VERSION,
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'SYNTHETIC numerical verification only; no real-tree observations or field precision',
            'experimental_validation': False, 'not_original_author_code': True,
            'reference_scope': REFERENCE_SCOPE, 'versions': versions(),
            'plotting_environment': {'matplotlib': package_version('matplotlib'),
                                     'fonttools': package_version('fonttools')},
            'source_identity_note': 'Code SHA256 values identify the tested working-tree sources; Git HEAD alone may predate uncommitted research files',
            'verification_script_sha256': sha(__file__),
            'hardware': {'platform': platform.platform(), 'processor': platform.processor(),
                         'logical_cpu_count': os.cpu_count()},
            'checks': checks, 'clean_inverse': clean_summary, 'clean_full_cli': clean_cli,
            'variants': variants, 'unit_conversion': {'environment': 'v2 parser normalization',
                'original_length_unit': 'm', 'changed_length_unit': 'cm',
                'max_observation_difference_m': conversion_error, 'passed': True},
            'n20_source_scale_inverse': _fit_summary(large, 80.), 'n20_full_cli': large_cli,
            'n20_scope': 'Only 50 ms; deliberately hypothetical 1e-5m resolution exposes weak beta signal; no field accuracy claim',
            'total_wall_time_s': time.perf_counter()-started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_directory', type=Path)
    parser.add_argument('--publish-research-evidence', action='store_true')
    args = parser.parse_args(argv)
    if args.output_directory.exists():
        parser.error('Output directory already exists; previous numerical evidence will not be overwritten')
    args.output_directory.mkdir(parents=True, exist_ok=False)
    try:
        summary = verify(args.output_directory)
        write_json(args.output_directory/'as10_numerical_checks.json', summary)
        if args.publish_research_evidence:
            root = Path(__file__).resolve().parent
            (root/'examples').mkdir(exist_ok=True)
            (root/'evidence').mkdir(exist_ok=True)
            shutil.copyfile(args.output_directory/'beta_chain_synthetic_v2.json', root/'examples'/'beta_chain_synthetic_v2.json')
            shutil.copyfile(args.output_directory/'as10_numerical_checks.json', root/'evidence'/'as10_numerical_checks.json')
            for name in ('comparison', 'residuals', 'objective', 'energy'):
                shutil.copyfile(args.output_directory/'clean-cli'/(name+'.png'), root/'evidence'/('as10_synthetic_'+name+'.png'))
        print(json.dumps({'scope': summary['scope'], 'clean_beta_kg_s': summary['clean_inverse']['beta_kg_s'],
                          'n20_beta_kg_s': summary['n20_source_scale_inverse']['beta_kg_s'],
                          'n20_candidate_kg_s': summary['n20_source_scale_inverse']['candidate_beta_kg_s'],
                          'total_wall_time_s': summary['total_wall_time_s']}, ensure_ascii=False))
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        write_json(args.output_directory/'VERIFICATION_FAILED.json', {
            'status': 'failed', 'error': str(exc), 'verification_version': VERIFICATION_VERSION,
            'experimental_validation': False, 'verification_script_sha256': sha(__file__)})
        print('Numerical verification failed: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
