"""Independent analytic and energy oracles for the research chain reconstruction."""
import json
import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from research.planar_chain import (
    ContactBeforeObservation, PlanarChain, cylinder_inertia, cylinder_mass,
    mass_weights, observation_coordinates, simulate, triangular_weights,
    uniform_cylinder_chain,
)


def chain(n=1, *, gravity=0., air=(0., 0.), weights=None, ground=0.):
    return PlanarChain(np.ones(n), np.full(n, .02), np.ones(n), np.full(n, 1 / 12),
                       64 / math.pi, np.ones(n - 1),
                       np.full(n, 1 / n) if weights is None else weights,
                       gravity_m_s2=gravity, air_velocity_m_s=air, ground_z_m=ground)


def test_cylinder_mass_diameter_correction_and_explicit_joint_diameter():
    assert cylinder_mass(780, .2, 1) == pytest.approx(780 * math.pi * .2**2 / 4)
    mass = cylinder_mass(780, [.2, .1], [1., 1.])
    np.testing.assert_allclose(cylinder_inertia(mass, [.2, .1], 1), mass * (1 / 12 + np.array([.2, .1])**2 / 16))
    model = PlanarChain([1, 1], [.2, .1], mass, cylinder_inertia(mass, [.2, .1], 1),
                        1.2e10, [.15], [.5, .5])
    assert model.joint_stiffness_nm[0] == pytest.approx(1.2e10 * math.pi * .15**4 / 64)
    assert model.joint_stiffness_nm[0] != pytest.approx(1.2e10 * math.pi * .1**4 / 64)


def test_one_rod_exact_damped_rotation_without_gravity():
    model = chain()
    t = np.linspace(0, .8, 17)
    beta, theta0, omega0 = 2., .1, .4
    # Pivot inertia A=I+mL²/4=1/3. COM drag torque=-beta*L²*omega/4.
    decay = beta * .25 / (1 / 3)
    truth_omega = omega0 * np.exp(-decay * t)
    truth_theta = theta0 + omega0 * (1 - np.exp(-decay * t)) / decay
    result = simulate(model, [theta0], [omega0], t, beta, .1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.theta_rad[:, 0], truth_theta, atol=2e-11)
    np.testing.assert_allclose(result.omega_rad_s[:, 0], truth_omega, atol=3e-11)
    np.testing.assert_allclose(result.com_m[:, 0], .5 * np.stack([np.sin(truth_theta), np.cos(truth_theta)], axis=1), atol=2e-11)
    assert result.diagnostics['experimental_validation'] is False
    assert not result.truncated and result.contact_time_s is None


def test_one_rod_gravity_sign_and_air_force_are_independent_oracle():
    model = chain(gravity=9.81, air=(2., -.3))
    theta, omega, beta = .4, .7, 3.
    # All terms independently projected onto e'(theta), without calling forces.
    torque = (9.81 * .5 * math.sin(theta) - beta * .25 * omega
              + beta * .5 * (2 * math.cos(theta) + .3 * math.sin(theta)))
    assert model.acceleration([theta], [omega], beta)[0] == pytest.approx(torque / (1 / 3))
    upright = chain(gravity=9.81)
    assert upright.acceleration([.001], [0], 0)[0] > 0
    assert upright.acceleration([0], [0], 99)[0] == 0


def test_gravity_driven_physical_pendulum_trajectory_against_independent_scalar_ode():
    length, mass, diameter, beta, g = 1.2, 2.3, .08, 2.4, 9.81
    inertia = mass * (length**2 / 12 + diameter**2 / 16)
    pivot = inertia + mass * length**2 / 4
    times = np.linspace(0, .3, 13)
    # Independent scalar equation about a fixed hinge, not the chain mass/force helpers.
    def oracle(_t, y):
        theta, omega = y
        return [omega, (mass*g*length/2 * math.sin(theta) - beta*length**2/4 * omega) / pivot]
    truth = solve_ivp(oracle, (0., .3), [.2, .1], method='DOP853', t_eval=times,
                      max_step=.0007, rtol=2e-12, atol=2e-14)
    model = PlanarChain([length], [diameter], [mass], [inertia], 1e7, [], [1.], gravity_m_s2=g)
    result = simulate(model, [.2], [.1], times, beta, .01, rtol=1e-9, atol=1e-11)
    assert truth.success and not result.truncated
    np.testing.assert_allclose(result.theta_rad[:, 0], truth.y[0], atol=5e-10)
    np.testing.assert_allclose(result.omega_rad_s[:, 0], truth.y[1], atol=5e-10)
    exact_com = length/2 * np.stack((np.sin(truth.y[0]), np.cos(truth.y[0])), axis=1)
    np.testing.assert_allclose(result.com_m[:, 0], exact_com, atol=5e-10)


def test_two_rod_mass_centrifugal_gravity_and_pair_spring_oracle():
    model = chain(2, gravity=9.81)
    angles, speed = np.array([0., math.pi / 2]), np.array([1., 2.])
    # m=L=1,I=1/12: M11=4/3,M22=1/3,M12=.5cos(delta).
    np.testing.assert_allclose(model.mass_matrix(angles), [[4 / 3, 0], [0, 1 / 3]], atol=1e-15)
    f = model.forces(angles, speed, 0)
    np.testing.assert_allclose(f['centrifugal'], [-2., .5], atol=1e-14)
    np.testing.assert_allclose(f['gravity'], [0., 9.81 / 2], atol=1e-14)
    np.testing.assert_allclose(f['springs'], [math.pi / 2, -math.pi / 2], atol=1e-14)
    expected = [1.5 + 3 * math.pi / 8, -1.5 - 3 * math.pi / 2 + 14.715]
    np.testing.assert_allclose(model.acceleration(angles, speed, 0), expected, atol=1e-12)
    assert f['springs'].sum() == 0  # No invented spring to the ground.


def test_mass_matrix_against_independent_com_jacobians_and_energy_identity():
    model = PlanarChain([.7] * 3, [.08, .06, .04], [2., 3., 4.], [.04, .07, .09],
                        1e5, [.07, .05], [.2, .3, .5], gravity_m_s2=0)
    angles, speed = np.array([.2, -.1, .4]), np.array([.4, -.3, .2])
    independent = np.diag([.04, .07, .09])
    for k, mass in enumerate([2., 3., 4.]):
        coefficients = np.array([.7 if i < k else .35 if i == k else 0 for i in range(3)])
        jacobian = np.stack([coefficients * np.cos(angles), -coefficients * np.sin(angles)])
        independent += mass * jacobian.T @ jacobian
    np.testing.assert_allclose(model.mass_matrix(angles), independent, rtol=1e-14)
    assert np.min(np.linalg.eigvalsh(independent)) > 0
    eps = 1e-6
    derivative = (model.mass_matrix(angles + eps * speed) - model.mass_matrix(angles - eps * speed)) / (2 * eps)
    assert speed @ model.forces(angles, speed, 0)['centrifugal'] == pytest.approx(.5 * speed @ derivative @ speed, abs=1e-10)


@pytest.mark.parametrize('beta,air', [(0., (0., 0.)), (.8, (0., 0.)), (.8, (2., -.1))])
def test_precontact_energy_balance_with_dissipation_and_air_work(beta, air):
    model = chain(2, gravity=9.81, air=air)
    result = simulate(model, [.12, .18], [.1, -.05], np.linspace(0, .2, 21), beta, .005,
                      rtol=2e-9, atol=1e-11)
    balance = result.energy_j - result.energy_j[0] + result.dissipation_j - result.air_work_j
    assert np.max(abs(balance)) < 2e-8
    if beta == 0:
        assert np.ptp(result.energy_j) < 2e-8
    elif air == (0., 0.):
        assert np.all(np.diff(result.energy_j) <= 1e-10)
        assert result.dissipation_j[-1] > 0
    else:
        assert abs(result.air_work_j[-1]) > .001
    assert result.diagnostics['energy_balance_relative'] < 1e-9
    json.dumps(result.diagnostics, allow_nan=False)


def test_wind_can_increase_energy_and_power_identity_holds():
    model = chain(air=(3., 0.))
    result = simulate(model, [.2], [.3], [0, .02, .04], 2., .002)
    assert result.energy_j[-1] > result.energy_j[0]
    f = model.forces([.2], [.3], 2.)
    v = .5 * .3 * np.array([math.cos(.2), -math.sin(.2)])
    assert np.sum(f['drag_forces_n'] * v) == pytest.approx(-f['relative_dissipation_w'] + f['air_work_w'])
    assert np.max(abs(result.energy_j - result.energy_j[0] + result.dissipation_j - result.air_work_j)) < 1e-8


def test_contact_stops_and_adapter_never_shortens_observation_sse():
    model = chain()
    result = simulate(model, [.2], [3.], np.linspace(0, 1, 21), 0., .01)
    assert result.contact_time_s == pytest.approx((math.pi / 2 - .2) / 3, abs=1e-9)
    assert result.truncated and result.times_s[-1] < result.contact_time_s
    assert np.all(result.nodes_m[:, 1:, 1] > 0)
    with pytest.raises(ContactBeforeObservation, match='window'):
        observation_coordinates(result, 'com')
    with pytest.raises(ContactBeforeObservation, match='Initial'):
        simulate(model, [math.pi], [0.], [0, .1], 0., .01)
    # The fixed base is on the plane and must not trigger an immediate event.
    safe = simulate(model, [.2], [.1], [0, .1], 0., .01)
    assert safe.contact_time_s is None


def test_observable_points_are_not_silently_com_or_nodes():
    result = simulate(chain(2), [.1, .2], [0, 0], [0, .01], 0., .002)
    points = observation_coordinates(result, 'material_points', [1, 1], [.25, .75])
    np.testing.assert_allclose(points[:, 0], .75 * result.nodes_m[:, 1] + .25 * result.nodes_m[:, 2])
    np.testing.assert_allclose(observation_coordinates(result, 'com', [1])[:, 0],
                               .5 * (result.nodes_m[:, 1] + result.nodes_m[:, 2]))
    np.testing.assert_allclose(observation_coordinates(result, 'nodes', [2])[:, 0], result.nodes_m[:, 2])
    assert not np.allclose(points[:, 0], result.com_m[:, 1])
    for kind, indices, fractions in [('com', [0, 0], None), ('nodes', [3], None),
                                     ('material_points', [0], [1.1]), ('com', [0], [.5])]:
        with pytest.raises(ValueError):
            observation_coordinates(result, kind, indices, fractions)


def test_strict_validation_and_immutable_input_copies():
    lengths = [1., 1.]
    model = PlanarChain(lengths, [.1, .1], [1, 1], [.1, .1], 1e5, [.1], [.5, .5])
    lengths[0] = 9
    assert model.lengths_m[0] == 1
    with pytest.raises(ValueError):
        model.masses_kg[0] = 999
    bad = [([1., 2.], [.1, .1], [1, 1], [.1, .1], [.1], [.5, .5]),
           ([1., 1.], [.1, .1], [True, 1], [.1, .1], [.1], [.5, .5]),
           ([1., 1.], [.1, .1], [1, 1], [.1, .1], [], [.5, .5]),
           ([1., 1.], [.1, .1], [1, 1], [.1, .1], [.1], [.5, .6])]
    for length, diameter, mass, inertia, joint, weight in bad:
        with pytest.raises(ValueError):
            PlanarChain(length, diameter, mass, inertia, 1e5, joint, weight)
    for t, omega in [([0, 0], [0]), ([0, .1], [None]), ([0, .1], ['0'])]:
        with pytest.raises(ValueError):
            simulate(chain(), [.1], omega, t, 1., .01)
    with pytest.raises(ValueError):
        simulate(chain(), [.1], [0], [0, .1], False, .01)


def test_normalized_profiles_conserve_total_mass_and_beta_under_refinement():
    np.testing.assert_allclose(mass_weights([0, 1, 3]), [0, .25, .75])
    for bad in ([0, 0], [1, -1], [None, 1]):
        with pytest.raises(ValueError):
            mass_weights(bad)
    for peak in (4., 7., 10.):
        coarse = triangular_weights(np.ones(10), (4, 10), peak_m=peak)
        fine = triangular_weights(np.full(20, .5), (4, 10), peak_m=peak)
        np.testing.assert_allclose(coarse, fine.reshape(10, 2).sum(axis=1), atol=5e-16)
        assert coarse.sum() == pytest.approx(1)
        assert np.all(coarse[:4] == 0)
    # Descending triangle integrates 3/4 in the first half; no arbitrary /2.
    np.testing.assert_allclose(triangular_weights([1., 1.], (0., 2.), peak_m=0), [.75, .25])
    for n in (2, 4, 20):
        model = uniform_cylinder_chain(10., .2, 780., 1.2e10, n, crown_mass_kg=25.,
            crown_interval_m=(4., 10.), effective_body_assumption='Synthetic midpoint effective cylinder, not measured crown inertia')
        assert model.masses_kg.sum() == pytest.approx(cylinder_mass(780, .2, 10) + 25)
        assert model.weights.sum() == pytest.approx(1)
    with pytest.raises(ValueError, match='assumption'):
        uniform_cylinder_chain(10, .2, 780, 1e10, 20, crown_mass_kg=25)
    with pytest.raises(ValueError, match='positive declared crown'):
        uniform_cylinder_chain(10, .2, 780, 1e10, 20, profile='mass')


def test_step_refinement_converges_to_independent_one_rod_oracle():
    model = chain()
    times = np.linspace(0, .4, 9)
    decay = 2 * .25 / (1 / 3)
    theta_exact = .1 + .4 * (1 - np.exp(-decay * times)) / decay
    errors = []
    for step in (.2, .1, .05):
        result = simulate(model, [.1], [.4], times, 2., step, rtol=1e-3, atol=1e-5)
        errors.append(np.max(abs(result.theta_rad[:, 0] - theta_exact)))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-7


def test_bounded_solver_rejects_budget_exhaustion_and_large_contact_steps():
    with pytest.raises(ValueError, match='RHS budget'):
        simulate(chain(2), [.2, .3], [.1, .2], [0, 1], 1., .01, max_rhs_calls=10)
    with pytest.raises(ValueError, match='step budget'):
        simulate(chain(), [.1], [0], [0, 1], 0., 1e-8)
    with pytest.raises(ValueError, match='Integrator'):
        simulate(chain(), [.1], [0], [0, 1], 0., .01, method='RK45')
    with pytest.raises(ValueError, match='angular resolution'):
        simulate(chain(), [.1], [100.], [0, .1], 0., .1)


def test_mesh_refinement_compares_same_physical_tip_not_different_com_indices():
    tips = []
    for n in (4, 8, 16):
        model = uniform_cylinder_chain(1., .05, 780., 1e5, n)
        result = simulate(model, np.full(n, .25), np.zeros(n), np.linspace(0, .08, 9), .2,
                          .004, rtol=1e-8, atol=1e-10)
        # The physical endpoint is unchanged across meshes; COM indices are not.
        tips.append(observation_coordinates(result, 'material_points', [n - 1], [1.]))
        assert model.masses_kg.sum() == pytest.approx(cylinder_mass(780, .05, 1))
        assert model.weights.sum() == pytest.approx(1)
    deltas = [np.max(np.linalg.norm(tips[i] - tips[i + 1], axis=2)) for i in (0, 1)]
    assert deltas[1] < deltas[0] / 3
    assert deltas[1] < .0002


def test_twenty_element_stiff_source_scale_is_integrated_with_recorded_budget():
    model = uniform_cylinder_chain(20., .2, 780., 1.2e10, 20, crown_mass_kg=100.,
        crown_interval_m=(10., 20.),
        effective_body_assumption='Synthetic midpoint combined-cylinder inertia, not measured crown inertia')
    result = simulate(model, np.full(20, .1), np.zeros(20), np.linspace(0, .05, 6), 80.,
                      .001, rtol=1e-7, atol=1e-9, max_rhs_calls=10000)
    assert result.theta_rad.shape == (6, 20)
    assert not result.truncated
    assert 0 < result.diagnostics['rhs_calls'] <= 10000
    assert result.diagnostics['energy_balance_relative'] < 1e-8
    assert result.diagnostics['elapsed_s'] > 0
    assert result.diagnostics['experimental_validation'] is False
    assert result.diagnostics['max_length_error_m'] < 1e-12
    assert result.diagnostics['shared_joint_error_m'] < 1e-12
    assert result.diagnostics['max_com_midpoint_error_m'] < 1e-12
    assert result.diagnostics['max_sampled_mass_matrix_condition'] < 1e12


def test_numerically_degenerate_mass_is_rejected_before_integration():
    model = PlanarChain([1, 1], [.01, .01], [1e-30, 1], [1e-30, 1e-30],
                        1., [.01], [.5, .5])
    with pytest.raises(ValueError, match='condition'):
        simulate(model, [0, 0], [0, 0], [0, .1], 0., .001)
    with pytest.raises(ValueError, match='Degenerate mass'):
        model.acceleration([0, 0], [0, 0], 0.)
    with pytest.raises(ValueError, match='below the ground'):
        PlanarChain([1], [.01], [1], [.1], 1., [], [1], base_m=(0, -1), ground_z_m=0)


def test_energy_diagnostic_is_invariant_to_world_origin_translation():
    original = chain(2, gravity=9.81)
    moved = replace(original, base_m=(123., 987.), ground_z_m=987.)
    angles, speed = [.12, .18], [.1, -.05]
    # Changing coordinate zero adds no physical energy or torque. A large constant
    # gravity datum must not make the relative conservation check artificially easy.
    assert moved.energy(angles, speed) == pytest.approx(original.energy(angles, speed), abs=2e-12)
    np.testing.assert_allclose(moved.acceleration(angles, speed, .8),
                               original.acceleration(angles, speed, .8), atol=1e-13)
    reference = simulate(original, angles, speed, np.linspace(0, .2, 11), .8, .005,
                         rtol=1e-9, atol=1e-11)
    shifted = simulate(moved, angles, speed, np.linspace(0, .2, 11), .8, .005,
                       rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(shifted.energy_j, reference.energy_j, atol=3e-12)
    assert shifted.diagnostics['energy_balance_relative'] < 1e-9
