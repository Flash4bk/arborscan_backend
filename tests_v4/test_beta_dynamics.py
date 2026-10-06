"""AS-10 contract/integration controls, never experimental tree validation.

Truth trajectories use direct, tighter DOP853 integration of the coupled
equations, rather than the production simulation/identification wrapper. This
tests integrator and inverse-pipeline consistency, not independent field truth.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from research.beta_inputs import FRAME, MODEL, UNITS, parse_document


ROOT = Path(__file__).resolve().parents[1]


def quantity(value, unit, uncertainty=0):
    return {"value": value, "unit": unit, "source": "Independent synthetic AS-10 test fixture",
            "standard_uncertainty": uncertainty}


def document(n=2):
    """A short, pre-contact, deliberately flexing effective-body example."""
    times = np.linspace(0, .55, 12).tolist()
    return {
        "schema_version": 2, "model": MODEL, "synthetic": True,
        "dataset_id": "AS10-independent-DOP853-control", "units": deepcopy(UNITS),
        "frame": deepcopy(FRAME), "support": "fixed_hinge_free_tip",
        "air_model": "still", "initial_velocity_mode": "supplied",
        "body": {"mode": "effective_midpoint_cylinders",
                 "midpoint_assumption": "Synthetic combined masses lie at each rod midpoint",
                 "effective_body_assumption": "Crown is included in declared full masses; its inertia is an explicit effective cylinder approximation"},
        "parameters": {
            "lengths_m": quantity([1.] * n, "m"),
            "diameters_m": quantity([.1, .09, .08][:n], "m"),
            "joint_diameters_m": quantity([.095, .085][:n-1], "m"),
            "elastic_modulus_pa": quantity(2e7, "Pa"),
            "masses_kg": quantity([2.4, 1.8, 1.2][:n], "kg"),
            "base_position_m": quantity([0., 0.], "m"),
            "gravity_m_s2": quantity(9.81, "m/s^2"),
            "ground_z_m": quantity(0., "m"),
            "air_velocity_m_s": quantity([0., 0.], "m/s"),
            "initial_angles_rad": quantity([.20, .27, .19][:n], "rad"),
            "initial_angular_velocities_rad_s": quantity([.14, .12, .10][:n], "rad/s"),
        },
        "load_profile": {"kind": "crown_mass_fractions", "source": "Synthetic chosen crown masses, not a species constant",
                         "crown_masses_kg": quantity([.2, .5, .7][:n], "kg")},
        "observations": {
            "times_s": quantity(times, "s"),
            "positions_m": quantity(np.zeros((len(times), n, 2)).tolist(), "m"),
            "points": {"kind": "com", "indices": list(range(n))},
            "time_origin": "initial_state",
            "calibration": {"kind": "synthetic_exact", "source": "Direct tighter DOP853 control",
                "metric_mapping": "Exact SI coordinates from synthetic chain equations",
                "time_origin_description": "Supplied state is t=0",
                "plane_assumption": "Exactly planar synthetic experiment",
                "uncertainty_note": "No experimental calibration claim; uncertainty explicitly zero"},
        },
        "identification": {
            "bounds_kg_s": [0., 7.], "bounds_reason": "Synthetic control bracket, not species prior",
            "max_step_s": .04, "beta_tolerance_kg_s": 2e-5,
            "trajectory_tolerance_m": 1e-6, "beta_step_convergence_kg_s": 2e-4,
            "residual_tolerance_m": 5e-5, "energy_relative_tolerance": 2e-6,
            "geometry_tolerance_m": 1e-10, "sensitivity_fraction": .05,
            "initial_speed_perturbation_rad_s": .02,
            "rtol": 2e-7, "atol": 2e-9, "grid_size": 9,
            "max_rhs_calls": 200000, "method": "DOP853",
            "tolerance_reason": "Synthetic numeric thresholds; not measured field uncertainty",
        },
    }


def independent_positions(doc, beta):
    parsed = parse_document(doc)
    n = parsed.chain.n

    def rhs(_, state):
        theta, omega = state[:n], state[n:]
        return np.r_[omega, parsed.chain.acceleration(theta, omega, beta)]

    result = solve_ivp(rhs, (0., float(parsed.times[-1])),
                       np.r_[parsed.theta0, parsed.omega0], t_eval=parsed.times,
                       method="DOP853", max_step=.0015, rtol=2e-12, atol=2e-14)
    assert result.success
    positions = []
    for state in result.y.T:
        nodes, com, _, _ = parsed.chain.kinematics(state[:n])
        assert np.min(nodes[1:, 1]) > parsed.chain.ground_z_m
        positions.append(com)
    return np.asarray(positions)


@pytest.fixture(scope="module")
def controls():
    result = {}
    for n, beta in [(2, 0.), (2, 1.7), (3, 1.7), (3, 5.), (2, 6.)]:
        doc = document(n)
        doc["observations"]["positions_m"]["value"] = independent_positions(doc, beta).tolist()
        result[n, beta] = doc
    return result


def run(doc):
    from research.beta_dynamics import run_document
    return run_document(deepcopy(doc))


def test_units_preserve_original_provenance_and_convert_uncertainty(controls):
    converted = deepcopy(controls[2, 1.7])
    original = deepcopy(converted)
    for key in ["lengths_m", "diameters_m", "joint_diameters_m", "base_position_m"]:
        entry = converted["parameters"][key]
        entry.update(value=(np.asarray(entry["value"]) * 100).tolist(), unit="cm", standard_uncertainty=.2)
    converted["parameters"]["masses_kg"].update(value=[2400., 1800.], unit="g", standard_uncertainty=5.)
    converted["parameters"]["initial_angles_rad"].update(
        value=np.rad2deg([.20, .27]).tolist(), unit="deg", standard_uncertainty=.5)
    converted["parameters"]["initial_angular_velocities_rad_s"].update(
        value=np.rad2deg([.14, .12]).tolist(), unit="deg/s", standard_uncertainty=.25)
    converted["parameters"]["elastic_modulus_pa"].update(value=20., unit="MPa", standard_uncertainty=.1)
    converted["observations"]["times_s"].update(
        value=(np.asarray(original["observations"]["times_s"]["value"]) * 1000).tolist(), unit="ms", standard_uncertainty=1.)
    converted["observations"]["positions_m"].update(
        value=(np.asarray(original["observations"]["positions_m"]["value"]) * 100).tolist(), unit="cm", standard_uncertainty=.1)
    snapshot = deepcopy(converted)
    parsed = parse_document(converted)
    assert converted == snapshot
    assert parsed.normalized["lengths_m"]["standard_uncertainty"] == pytest.approx(.002)
    assert parsed.normalized["positions_m"]["standard_uncertainty"] == pytest.approx(.001)
    assert parsed.normalized["initial_angles_rad"]["standard_uncertainty"] == pytest.approx(np.deg2rad(.5))
    assert parsed.normalized["times_s"]["standard_uncertainty"] == pytest.approx(.001)
    assert parsed.normalized["masses_kg"]["standard_uncertainty"] == pytest.approx(.005)
    assert parsed.normalized["masses_kg"]["input_unit"] == "g"
    np.testing.assert_allclose(parsed.chain.masses_kg, [2.4, 1.8])
    np.testing.assert_allclose(parsed.theta0, [.20, .27])
    np.testing.assert_allclose(parsed.observed, original["observations"]["positions_m"]["value"])
    assert converted["parameters"]["masses_kg"]["source"] == snapshot["parameters"]["masses_kg"]["source"]


def test_measured_inertia_provenance_is_distinct_from_effective_derived_array():
    doc = document()
    doc["body"]["mode"] = "measured_midpoint_inertias"
    entry = quantity([.203, .151], "kg m^2", [.002, .001])
    entry["source"] = "Synthetic separately supplied transverse central inertias"
    doc["parameters"]["inertias_kg_m2"] = entry
    parsed = parse_document(doc)
    assert parsed.normalized["inertias_kg_m2"]["source"] == entry["source"]
    assert parsed.normalized["inertias_kg_m2"]["standard_uncertainty"] == [.002, .001]
    np.testing.assert_allclose(parsed.normalized["effective_body_inertias_kg_m2"], entry["value"])


@pytest.mark.parametrize("case", ["nonuniform", "unsupported_support", "missing_omega", "unknown_uncertainty",
                                  "missing_value", "hidden", "wrong_frame", "wood_ignores_crown",
                                  "missing_effective_assumption", "bad_unit", "wrong_time_origin"])
def test_unsupported_or_missing_measurements_are_rejected(case):
    doc = document()
    if case == "nonuniform":
        doc["parameters"]["lengths_m"]["value"] = [1., 1.01]
    elif case == "unsupported_support":
        doc["support"] = "clamped_root"
    elif case == "missing_omega":
        del doc["parameters"]["initial_angular_velocities_rad_s"]
    elif case == "unknown_uncertainty":
        del doc["parameters"]["masses_kg"]["standard_uncertainty"]
    elif case == "missing_value":
        doc["observations"]["positions_m"]["value"][1][0][0] = None
    elif case == "hidden":
        doc["observations"]["visibility"] = np.ones((12, 2), dtype=bool).tolist()
        doc["observations"]["visibility"][2][1] = False
    elif case == "wrong_frame":
        doc["frame"]["z"] = "down"
    elif case == "wood_ignores_crown":
        doc["body"]["mode"] = "uniform_wood_cylinders"
        doc["parameters"]["density_kg_m3"] = quantity(780., "kg/m^3")
    elif case == "missing_effective_assumption":
        del doc["body"]["effective_body_assumption"]
    elif case == "bad_unit":
        doc["parameters"]["masses_kg"]["unit"] = "lb"
    elif case == "wrong_time_origin":
        doc["observations"]["time_origin"] = "video_start_unknown"
    with pytest.raises(ValueError):
        parse_document(doc)


def test_experimental_provenance_requires_original_assets_and_calibration():
    doc = document()
    doc["synthetic"] = False
    doc["observations"]["calibration"]["kind"] = "calibrated_planar_video"
    with pytest.raises(ValueError, match="asset"):
        parse_document(doc)
    doc["observations"]["calibration"]["assets"] = [{"reference": "test/video.mp4", "sha256": "a" * 64}]
    parsed = parse_document(doc)
    assert parsed.observed.shape == (12, 2, 2)
    del doc["observations"]["calibration"]["metric_mapping"]
    with pytest.raises(ValueError, match="metric_mapping"):
        parse_document(doc)


@pytest.mark.parametrize("block", ["parameters", "body", "load_profile", "observations",
                                    "observations.points", "observations.calibration",
                                    "observations.calibration.assets.0", "identification"])
@pytest.mark.parametrize("bad", [None, []], ids=["null", "array"])
def test_malformed_structural_blocks_fail_with_value_error(block, bad):
    doc = document()
    if ".assets." in block:
        doc["synthetic"] = False
        calibration = doc["observations"]["calibration"]
        calibration["kind"] = "calibrated_planar_video"
        calibration["assets"] = [{"reference": "test/video.mp4", "sha256": "a" * 64}]
    path = block.split(".")
    target = doc
    for part in path[:-1]:
        target = target[int(part) if part.isdigit() else part]
    last = int(path[-1]) if path[-1].isdigit() else path[-1]
    target[last] = bad
    with pytest.raises(ValueError, match="JSON object"):
        parse_document(doc)


@pytest.mark.parametrize("digest", [["a"] * 64, "A" * 64, None], ids=["array", "uppercase", "null"])
def test_experimental_asset_sha_is_a_lowercase_string(digest):
    doc = document()
    doc["synthetic"] = False
    calibration = doc["observations"]["calibration"]
    calibration["kind"] = "calibrated_planar_video"
    calibration["assets"] = [{"reference": "test/video.mp4", "sha256": digest}]
    with pytest.raises(ValueError, match="SHA256"):
        parse_document(doc)


@pytest.mark.parametrize("n,beta", [(2, 1.7), (3, 1.7), (3, 5.)])
def test_recovers_beta_from_tighter_different_integrator_control(controls, n, beta):
    doc = deepcopy(controls[n, beta])
    doc["identification"]["method"] = "Radau"
    result = run(doc)
    assert result["beta_kg_s"] == pytest.approx(beta, abs=2e-4)
    assert result["numerically_converged"]
    assert result["experimental_validation"] is False
    assert result["input"] == doc
    assert len(result["input_sha256"]) == 64
    assert result["schema_version"] == 2
    assert "initial_angular_velocities_supplied_not_jointly_estimated" in result["diagnostics"]
    assert not any("not_converged" in flag or "tolerance_not_met" in flag
                   or flag == "residual_exceeds_declared_tolerance"
                   for flag in result["diagnostics"])
    fits = result["time_refinement"]["fits"]
    assert len(fits) == 3
    assert [item["max_step_s"] for item in fits] == pytest.approx([.04, .02, .01])
    assert max(abs(item["beta_kg_s"] - beta) for item in fits) < 2e-4


def test_true_zero_beta_is_a_boundary_candidate_not_false_positive(controls):
    result = run(controls[2, 0.])
    assert result["candidate_beta_kg_s"] == pytest.approx(0., abs=2e-4)
    assert "optimum_at_search_boundary" in result["diagnostics"]


def test_out_of_bounds_beta_reports_boundary_and_residual(controls):
    doc = deepcopy(controls[2, 6.])
    doc["identification"]["bounds_kg_s"] = [0., 4.]
    result = run(doc)
    assert result["candidate_beta_kg_s"] == pytest.approx(4., abs=2e-4)
    assert "optimum_at_search_boundary" in result["diagnostics"]
    assert result["beta_kg_s"] is None


def test_motionless_upright_chain_cannot_identify_beta():
    doc = document()
    doc["parameters"]["initial_angles_rad"]["value"] = [0., 0.]
    doc["parameters"]["initial_angular_velocities_rad_s"]["value"] = [0., 0.]
    doc["observations"]["positions_m"]["value"] = independent_positions(doc, 3.).tolist()
    result = run(doc)
    assert result["beta_kg_s"] is None
    assert "beta_not_identifiable_flat_profile" in result["diagnostics"]


def test_noise_and_parameter_errors_are_not_experimental_validation(controls):
    exact = deepcopy(controls[3, 1.7])
    clean_result = run(exact)
    noisy = deepcopy(exact)
    rng = np.random.default_rng(20261006)
    measured = np.asarray(noisy["observations"]["positions_m"]["value"])
    measured += rng.normal(0., 1e-3, measured.shape)
    noisy["observations"]["positions_m"]["value"] = measured.tolist()
    noisy["observations"]["positions_m"]["standard_uncertainty"] = .001
    noise_result = run(noisy)
    assert noise_result["experimental_validation"] is False
    assert "residual_exceeds_declared_tolerance" in noise_result["diagnostics"]
    assert noise_result["rmse_coordinate_m"] > 100 * clean_result["rmse_coordinate_m"]
    assert noise_result["candidate_beta_kg_s"] == pytest.approx(1.7, abs=.15)
    for key, wrong in [("elastic_modulus_pa", 4e7), ("masses_kg", [3.6, 2.7, 1.8]),
                       ("initial_angular_velocities_rad_s", [.24, .22, .20])]:
        changed = deepcopy(exact)
        changed["parameters"][key]["value"] = wrong
        result = run(changed)
        assert result["experimental_validation"] is False
        assert "residual_exceeds_declared_tolerance" in result["diagnostics"], (
            f"{key} mismatch must not silently report a clean fit")
        assert result["rmse_coordinate_m"] > 100 * clean_result["rmse_coordinate_m"]
        assert result["candidate_beta_kg_s"] != pytest.approx(clean_result["candidate_beta_kg_s"], abs=1e-3)


def cli(input_file, output):
    return subprocess.run([sys.executable, "-m", "research.beta_dynamics", str(input_file), str(output)],
                          cwd=ROOT, capture_output=True, text=True, timeout=180)


def test_cli_replay_saves_original_input_finite_outputs_and_refuses_overwrite(controls, tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(controls[2, 1.7], ensure_ascii=False, indent=2), encoding="utf-8")
    first, second = tmp_path / "first", tmp_path / "replay"
    for output in [first, second]:
        completed = cli(source, output)
        assert completed.returncode == 0, completed.stderr
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        original = json.loads((output / "input.json").read_text(encoding="utf-8"))
        assert original == controls[2, 1.7] == result["input"]
        assert result["candidate_beta_kg_s"] == pytest.approx(1.7, abs=2e-4)
        for name in ["comparison.png", "residuals.png", "objective.png", "energy.png"]:
            assert (output / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        lines = (output / "trajectories.csv").read_text(encoding="utf-8").splitlines()
        assert len(lines) > 12
        assert "nan" not in "\n".join(lines).lower()
        assert "inf" not in "\n".join(lines).lower()
    a = json.loads((first / "result.json").read_text(encoding="utf-8"))
    b = json.loads((second / "result.json").read_text(encoding="utf-8"))
    assert a["input_sha256"] == b["input_sha256"]
    assert a["candidate_beta_kg_s"] == b["candidate_beta_kg_s"]
    assert (first / "trajectories.csv").read_bytes() == (second / "trajectories.csv").read_bytes()
    before = hashlib.sha256((first / "result.json").read_bytes()).hexdigest()
    refused = cli(source, first)
    assert refused.returncode != 0
    assert hashlib.sha256((first / "result.json").read_bytes()).hexdigest() == before


def test_cli_rejects_missing_state_without_creating_success_artifacts(tmp_path):
    doc = document()
    del doc["parameters"]["initial_angular_velocities_rad_s"]
    source = tmp_path / "missing.json"
    source.write_text(json.dumps(doc), encoding="utf-8")
    output = tmp_path / "failed"
    result = cli(source, output)
    assert result.returncode != 0
    assert not (output / "result.json").exists()


def test_cli_null_parameters_writes_failure_manifest_without_traceback(tmp_path):
    doc = document()
    doc["parameters"] = None
    source = tmp_path / "invalid-structure.json"
    source.write_text(json.dumps(doc), encoding="utf-8")
    output = tmp_path / "failed"
    completed = cli(source, output)
    assert completed.returncode == 2
    assert "Traceback" not in completed.stdout + completed.stderr
    failure = json.loads((output / "FAILED.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["experimental_validation"] is False
    assert failure["input_file_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert not (output / "result.json").exists()
    assert not (output / "COMPLETE.json").exists()


def test_cli_invalid_json_has_clear_input_error_without_success_artifacts(tmp_path):
    source = tmp_path / "broken.json"
    source.write_bytes(b'{"schema_version":2,')
    output = tmp_path / "never-created"
    completed = cli(source, output)
    assert completed.returncode == 2
    assert "Cannot read a valid JSON input" in completed.stderr
    assert "Traceback" not in completed.stdout + completed.stderr
    assert not output.exists()
