"""AS-10 research CLI: provenance input -> coupled model -> beta -> diagnostics.

Run `python -m research.beta_dynamics INPUT.json NEW_OUTPUT_DIRECTORY`.
This module has no network, app, database, training, or production integration.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy

from .beta_identification import identify, coordinate_objective
from .beta_inputs import parse_document, MODEL, SCHEMA_VERSION
from .planar_chain import simulate, observation_coordinates, INTEGRATOR_VERSION

WORKFLOW_VERSION = 'coupled-beta-workflow-v1'


def canonical_hash(document):
    canonical = json.dumps(document, sort_keys=True, separators=(',', ':'),
                           allow_nan=False, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


def versions():
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                                  capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    files = ['beta_dynamics.py', 'beta_inputs.py', 'planar_chain.py', 'beta_identification.py']
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'git_revision': revision, 'code_sha256': {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in files}}


def run_document(document):
    """Identify one conditional beta with known initial speeds, never field validation.

    An inadmissible/contact/divergent trial aborts explicitly; we do not fabricate
    positions or silently score a different observation window. Tighten justified
    bounds or supply a consistently pre-contact observation package in that case.
    """
    started = time.perf_counter()
    data = parse_document(document)
    cfg, chain = data.config, data.chain
    cache = OrderedDict()
    calls = 0
    rhs_calls = 0
    def forward_for(active_chain, active_omega):
        local_cache = cache if active_chain is chain and active_omega is data.omega0 else OrderedDict()
        def forward(beta, times, step):
            nonlocal calls, rhs_calls
            key = (float(beta), float(step))
            if key not in local_cache:
                if calls >= cfg['max_forward_evaluations']:
                    raise ValueError('Workflow forward-evaluation budget exceeded')
                if time.perf_counter()-started > cfg['max_wall_time_s']:
                    raise ValueError('Workflow wall-time budget exceeded')
                remaining_rhs = cfg['max_total_rhs_calls'] - rhs_calls
                if remaining_rhs < 10:
                    raise ValueError('Workflow total RHS budget exceeded')
                calls += 1
                sim = simulate(active_chain, data.theta0, active_omega, times, beta, step,
                               rtol=cfg['rtol'], atol=cfg['atol'], method=cfg['method'],
                               max_rhs_calls=min(cfg['max_rhs_calls'], remaining_rhs))
                rhs_calls += sim.diagnostics['rhs_calls']
                predicted = observation_coordinates(sim, **data.points)
                local_cache[key] = (sim, predicted)
                # Bound research cache memory instead of retaining every full trajectory.
                if len(local_cache) > 16:
                    local_cache.popitem(last=False)
            local_cache.move_to_end(key)
            return local_cache[key][1]
        return forward
    forward = forward_for(chain, data.omega0)
    # The supplied state must be the same physical points as the observed t=0.
    initial_prediction = forward(cfg['bounds_kg_s'][0], data.times, cfg['max_step_s'])[0]
    initial_error = float(np.max(np.linalg.norm(initial_prediction - data.observed[0], axis=1)))
    refinements = []
    fit = None
    for factor in [1, 2, 4]:
        step = cfg['max_step_s'] / factor
        fit = identify(data.times, data.observed, forward,
                       bounds_kg_s=cfg['bounds_kg_s'], bounds_reason=cfg['bounds_reason'],
                       forward_version=MODEL, max_step_s=step,
                       beta_tolerance_kg_s=cfg['beta_tolerance_kg_s'],
                       trajectory_tolerance_m=cfg['trajectory_tolerance_m'], grid_size=cfg['grid_size'])
        refinements.append({'max_step_s': step, 'candidate_beta_kg_s': fit['candidate_beta_kg_s'],
                            'beta_kg_s': fit['beta_kg_s'], 'sse_m2': fit['sse_m2'],
                            'trajectory_deltas_m': fit['step_halving_max_delta_m'],
                            'diagnostics': fit['diagnostics']})
    beta = fit['candidate_beta_kg_s']
    final_step = cfg['max_step_s'] / 4
    forward(beta, data.times, final_step)
    final_sim, prediction = cache[(float(beta), final_step)]
    flags = list(fit['diagnostics'])
    deltas = [abs(refinements[i+1]['candidate_beta_kg_s'] - refinements[i]['candidate_beta_kg_s'])
              for i in [0, 1]]
    if max(deltas) > cfg['beta_step_convergence_kg_s']:
        flags.append('fitted_beta_time_refinement_not_converged')
    if initial_error > cfg['residual_tolerance_m']:
        flags.append('observed_initial_state_inconsistent')
    if fit['rmse_coordinate_m'] > cfg['residual_tolerance_m']:
        flags.append('residual_exceeds_declared_tolerance')
    diag = final_sim.diagnostics
    energy_error = float(diag['energy_balance_relative'])
    geometry_error = float(diag['max_length_error_m'])
    if energy_error > cfg['energy_relative_tolerance']:
        flags.append('energy_balance_tolerance_not_met')
    if geometry_error > cfg['geometry_tolerance_m']:
        flags.append('geometry_tolerance_not_met')
    # Conditional sensitivities, not a joint optimisation or a confidence interval.
    fraction = cfg['sensitivity_fraction']
    perturbed = {
        'elastic_modulus': (replace(chain, elastic_modulus_pa=chain.elastic_modulus_pa * (1+fraction)),
                           data.omega0, fraction),
        'full_body_masses_and_inertias': (replace(chain, masses_kg=chain.masses_kg * (1+fraction),
                                                 inertias_kg_m2=chain.inertias_kg_m2 * (1+fraction)),
                                        data.omega0, fraction),
        'initial_angular_velocity': (chain, data.omega0 + cfg['initial_speed_perturbation_rad_s'],
                                     cfg['initial_speed_perturbation_rad_s']),
    }
    lo, hi = cfg['bounds_kg_s']
    beta_increment = max(cfg['beta_tolerance_kg_s'] * 10, (hi-lo)*.01)
    beta_low, beta_high = max(lo, beta-beta_increment), min(hi, beta+beta_increment)
    beta_effect = (forward(beta_high, data.times, final_step) -
                   forward(beta_low, data.times, final_step)).ravel()
    scaled_beta_effect = beta_effect * ((hi-lo)/(beta_high-beta_low))
    sensitivities, columns = {}, [scaled_beta_effect]
    for name, (changed_chain, changed_omega, increment) in perturbed.items():
        altered = forward_for(changed_chain, changed_omega)(beta, data.times, final_step)
        effect = (altered - prediction).ravel()
        norm = float(np.linalg.norm(effect))
        beta_norm = float(np.linalg.norm(scaled_beta_effect))
        cosine = None if norm == 0 or beta_norm == 0 else float(np.dot(effect, scaled_beta_effect)/(norm*beta_norm))
        score, _ = coordinate_objective(data.observed, altered)
        sensitivities[name] = {'increment': increment,
                               'increment_unit': 'rad/s' if name == 'initial_angular_velocity' else 'fraction',
                               'max_point_shift_m': float(np.max(np.linalg.norm(altered-prediction, axis=2))),
                               'sse_at_same_beta_m2': score,
                               'absolute_cosine_with_beta_effect': None if cosine is None else abs(cosine)}
        columns.append(effect)
        if cosine is not None and abs(cosine) > .98:
            flags.append('weak_joint_distinguishability_beta_vs_' + name)
    singular = np.linalg.svd(np.column_stack(columns), compute_uv=False)
    ratio = None if singular[0] == 0 else float(singular[-1]/singular[0])
    if ratio is None or ratio < 1e-6:
        flags.append('nuisance_sensitivity_matrix_ill_conditioned')
    low_prediction = forward(lo, data.times, final_step)
    high_prediction = forward(hi, data.times, final_step)
    bound_signal = float(np.sqrt(np.mean((high_prediction-low_prediction)**2)))
    uncertainty = data.normalized['positions_m']['standard_uncertainty']
    noise_scale = cfg['residual_tolerance_m'] if uncertainty is None else float(np.max(uncertainty))
    if bound_signal <= noise_scale:
        flags.append('beta_signal_below_declared_coordinate_resolution')
    if uncertainty is None:
        flags.append('coordinate_uncertainty_unknown_no_confidence_interval')
    flags.append('initial_angular_velocities_supplied_not_jointly_estimated')
    flags = list(dict.fromkeys(flags))
    rejected = any(f in flags for f in ['beta_not_identifiable_flat_profile',
        'beta_signal_below_declared_coordinate_resolution', 'observed_initial_state_inconsistent',
        'residual_exceeds_declared_tolerance', 'optimum_at_search_boundary'])
    converged = fit['numerically_converged'] and not any(
        f in flags for f in ['fitted_beta_time_refinement_not_converged',
                            'energy_balance_tolerance_not_met', 'geometry_tolerance_not_met'])
    fit.update(schema_version=SCHEMA_VERSION, workflow_version=WORKFLOW_VERSION,
               status='conditional_research_candidate' if not rejected and converged else 'not_established',
               beta_kg_s=fit['beta_kg_s'] if not rejected and converged else None,
               interpretation='Conditional on supplied material, mass, support and initial state; not a validated tree parameter',
               synthetic=document['synthetic'], dataset_id=document['dataset_id'],
               input=document, input_sha256=canonical_hash(document), normalized_inputs=data.normalized,
               point_specification=data.points, load_profile={
                   'kind': document['load_profile']['kind'], 'source': document['load_profile']['source'],
                   'weights': chain.weights.tolist(), 'sum_weights': float(chain.weights.sum()),
                   'element_beta_kg_s': (beta*chain.weights).tolist(),
                   'meaning': 'total_beta_kg_s_times_normalized_weights_not_dissertation_beta_max'},
               integrator={'version': INTEGRATOR_VERSION, 'method': cfg['method'],
                           'rtol': cfg['rtol'], 'atol': cfg['atol'], 'max_step_s': final_step,
                           'scipy': scipy.__version__}, versions=versions(),
               time_refinement={'fits': refinements, 'beta_deltas_kg_s': deltas,
                                'allowed_beta_delta_kg_s': cfg['beta_step_convergence_kg_s']},
               sensitivity={'one_at_a_time': sensitivities,
                            'scaled_singular_values_m': singular.tolist(),
                            'smallest_to_largest_singular_ratio': ratio,
                            'trajectory_rms_across_search_bounds_m': bound_signal,
                            'coordinate_resolution_for_warning_m': noise_scale,
                            'joint_fit_performed': False, 'cosine_warning_threshold': .98,
                            'singular_ratio_warning_threshold': 1e-6,
                            'confidence_interval': None, 'noise_model': 'not_assumed'},
               forward_diagnostics=diag,
               energy={'energy_j': final_sim.energy_j.tolist(),
                       'dissipation_j': final_sim.dissipation_j.tolist(),
                       'air_work_j': final_sim.air_work_j.tolist()},
               initial_state_max_point_error_m=initial_error,
               contact_time_s=final_sim.contact_time_s, forward_evaluations=calls,
               total_rhs_calls=rhs_calls, resource_limits={k: cfg[k] for k in [
                   'max_forward_evaluations', 'max_rhs_calls', 'max_total_rhs_calls', 'max_wall_time_s']},
               wall_time_s=time.perf_counter()-started, diagnostics=flags,
               numerically_converged=converged, experimental_validation=False,
               confidence_interval=None, uniqueness_proven=False,
               scope='synthetic_coupled_chain' if document['synthetic'] else 'unvalidated_dynamic_experiment')
    return fit


def plots(result, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times = np.asarray(result['times_s'])
    observed, predicted = np.asarray(result['observed_m']), np.asarray(result['predicted_m'])
    label = 'SYNTHETIC' if result['synthetic'] else 'EXPERIMENT — UNVALIDATED'
    selected = sorted(set([0, len(observed[0])//2, len(observed[0])-1]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for i in selected:
        for component in [0, 1]:
            line, = axes[component].plot(times, predicted[:, i, component], label=f'point {i}: model')
            axes[component].plot(times, observed[:, i, component], '.', color=line.get_color(),
                                 label=f'point {i}: observations')
    for ax, name in zip(axes, ['x', 'z']):
        ax.set(xlabel='Time [s]', ylabel=f'{name} [m]'); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.suptitle(label + ' — selected observed points (full tracks in CSV)')
    fig.savefig(directory/'comparison.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    residual = observed-predicted
    for i in selected:
        ax.plot(times, np.linalg.norm(residual[:, i], axis=1), label=f'point {i}')
    ax.plot(times, np.linalg.norm(residual, axis=2).mean(axis=1), '--', label='mean of all points')
    ax.set(xlabel='Time [s]', ylabel='Coordinate deviation [m]', title=label+' — residuals')
    ax.grid(alpha=.3); ax.legend(); fig.savefig(directory/'residuals.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    profile = result['profile']
    ax.plot([p['beta_kg_s'] for p in profile], [p['sse_m2'] for p in profile], 'o-', label='profile grid')
    ax.axvline(result['candidate_beta_kg_s'], color='darkred', linestyle='--', label='numerical candidate')
    ax.set(xlabel='Total beta [kg/s]', ylabel='Coordinate SSE [m²]', title=label+' — objective')
    ax.grid(alpha=.3); ax.legend(); fig.savefig(directory/'objective.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    energy = result['energy']
    e, d, work = (np.asarray(energy[k]) for k in ['energy_j', 'dissipation_j', 'air_work_j'])
    ax.plot(times, e-e[0], label='E(t)−E(0)')
    ax.plot(times, -d+work, '--', label='−relative dissipation + air work')
    ax.set(xlabel='Time [s]', ylabel='Energy change [J]', title=label+' — energy balance')
    ax.grid(alpha=.3); ax.legend(); fig.savefig(directory/'energy.png', dpi=160); plt.close(fig)


def save_result(document, result, directory):
    for name, obj in [('input.json', document), ('result.json', result)]:
        with (directory/name).open('x', encoding='utf-8') as stream:
            json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
    with (directory/'trajectories.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_s', 'selected_point', 'model_point_index', 'observed_x_m', 'observed_z_m',
                         'predicted_x_m', 'predicted_z_m', 'residual_x_m', 'residual_z_m'])
        indices = result['point_specification']['indices']
        for t, observed, predicted in zip(result['times_s'], result['observed_m'], result['predicted_m']):
            for index, (a, b) in enumerate(zip(observed, predicted)):
                writer.writerow([t, index, indices[index], *a, *b, a[0]-b[0], a[1]-b[1]])
    plots(result, directory)
    with (directory/'COMPLETE.json').open('x', encoding='utf-8') as stream:
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(directory.iterdir()) if p.is_file() and p.name != 'COMPLETE.json'}
        json.dump({'input_sha256': result['input_sha256'], 'files_sha256': files}, stream, indent=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path); parser.add_argument('output_directory', type=Path)
    args = parser.parse_args(argv)
    if args.output_directory.exists():
        parser.error('Output directory already exists; refusing to overwrite a previous calculation')
    try:
        if args.input.stat().st_size > 10_000_000:
            parser.error('Input document exceeds 10 MB research contract limit')
        raw = args.input.read_bytes()
        document = json.loads(raw.decode('utf-8-sig'))
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error('Cannot read a valid JSON input: ' + str(exc))
    args.output_directory.mkdir(parents=True, exist_ok=False)
    try:
        result = run_document(document)
        result['input_file_sha256'] = hashlib.sha256(raw).hexdigest()
        save_result(document, result, args.output_directory)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        failure = {'status': 'failed', 'workflow_version': WORKFLOW_VERSION,
                   'input_file_sha256': hashlib.sha256(raw).hexdigest(),
                   'error': str(exc), 'experimental_validation': False}
        with (args.output_directory/'FAILED.json').open('x', encoding='utf-8') as stream:
            json.dump(failure, stream, indent=2, ensure_ascii=False, allow_nan=False)
        print(json.dumps(failure, ensure_ascii=False)); return 2
    print(json.dumps({'status': result['status'], 'beta_kg_s': result['beta_kg_s'],
                      'candidate_beta_kg_s': result['candidate_beta_kg_s'],
                      'synthetic': result['synthetic'], 'wall_time_s': result['wall_time_s'],
                      'diagnostics': result['diagnostics']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
