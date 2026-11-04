"""Offline benchmark contracts; fake inference is not model-quality evidence."""
import copy
import io
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from tools import benchmark_tree_recognition as benchmark


def png(array):
    output = io.BytesIO()
    Image.fromarray(array).save(output, "PNG")
    return output.getvalue()


def make_bundle(root, *, labelled=True, accepted=True, candidate=False):
    root.mkdir(exist_ok=True)
    photo = png(np.full((10, 12, 3), 90, np.uint8))
    truth = np.zeros((10, 12), np.uint8)
    truth[2:4, 3:5] = 255
    mask = png(truth)
    (root / "private-photo.png").write_bytes(photo)
    (root / "private-mask.png").write_bytes(mask)
    (root / "baseline.pt").write_bytes(b"frozen fake baseline: never loaded by YOLO")
    models = {"baseline": {"path": root / "baseline.pt", "sha256": benchmark.digest((root / "baseline.pt").read_bytes())}}
    if candidate:
        (root / "candidate.pt").write_bytes(b"frozen fake candidate: never loaded by YOLO")
        models["candidate"] = {"path": root / "candidate.pt", "sha256": benchmark.digest((root / "candidate.pt").read_bytes())}
    case = {"id": "private-target", "scene_id": "private-scene", "conditions": ["sparse_crown"],
            "image": {"path": "private-photo.png", "sha256": benchmark.digest(photo)},
            "tap": [.3, .25], "provenance": {"split": "val", "tree_id": "tree-a", "series_id": "series-a"}}
    if labelled:
        case["mask"] = {"path": "private-mask.png", "sha256": benchmark.digest(mask)}
        if accepted:
            case["review"] = {"status": "accepted", "correction_id": "reviewed-revision",
                              "image_sha256": case["image"]["sha256"], "mask_sha256": case["mask"]["sha256"],
                              "decisions": [{"action": "accepted", "actor_id": "private-reviewer", "at": "2026-10-20"}]}
    manifest = {"schema": benchmark.SCHEMA, "purpose": "diagnostic", "cases": [case]}
    return manifest, models, truth


def validated(root, manifest, models):
    path = root / "private-manifest.json"
    raw = benchmark.json_bytes(manifest)
    path.write_bytes(raw)
    return benchmark.validate_inputs(path, benchmark.digest(raw), models)


def snapshot(root, manifest, models):
    options = {"device": "cpu", "threads": 1, "imgsz": 1024, "confidence": .25, "max_seconds": 20}
    return benchmark.prepare_snapshot(root / "new-output", validated(root, manifest, models), options)


def cli_args(root, manifest, models, output=None):
    path = root / "private-manifest.json"
    raw = benchmark.json_bytes(manifest)
    path.write_bytes(raw)
    args = ["--manifest", str(path), "--manifest-sha256", benchmark.digest(raw),
            "--baseline", str(models["baseline"]["path"]), "--baseline-sha256", models["baseline"]["sha256"],
            "--output", str(output or root / "new-output")]
    if "candidate" in models:
        args += ["--candidate", str(models["candidate"]["path"]), "--candidate-sha256", models["candidate"]["sha256"]]
    return args


class FakeRuntime:
    def __init__(self, mask=None, warnings=()):
        self.mask, self.warnings, self.taps = mask, list(warnings), []

    def infer(self, image, tap_x=None, tap_y=None):
        self.taps.append((tap_x, tap_y))
        return SimpleNamespace(detected=self.mask is not None, mask=self.mask,
                               confidence=.8 if self.mask is not None else None,
                               bbox=(3, 2, 5, 3) if self.mask is not None else None,
                               warnings=self.warnings)


@pytest.fixture
def existing_evaluator():
    from arborscan_v4.quality_train import segmentation_metrics
    return segmentation_metrics


def test_check_only_freezes_original_oriented_mask_and_weights_without_model_import(tmp_path, capsys, monkeypatch):
    manifest, models, _ = make_bundle(tmp_path)
    monkeypatch.setattr(benchmark, "launch_worker", lambda *args: pytest.fail("check-only must never infer"))
    before = (tmp_path / "private-photo.png").read_bytes()
    assert benchmark.main(cli_args(tmp_path, manifest, models) + ["--check-only"]) == 0
    output = tmp_path / "new-output"
    frozen = json.loads((output / "frozen.json").read_text())
    assert (output / "originals/0.bin").read_bytes() == before
    assert benchmark.file_hash(output / "weights/baseline.pt") == models["baseline"]["sha256"]
    assert frozen["training_independence"] == {"baseline": "not_supplied"}
    assert frozen["settings"]["warmup_runs"] == 1
    assert frozen["settings"]["timing_repeats"] == 3
    assert frozen["cases"][0]["review_attested"]
    assert not (output / "COMPLETE").exists()
    assert (tmp_path / "private-photo.png").read_bytes() == before
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["status"] == "checked_no_inference" and not summary["activation_eligible"]
    assert "private-" not in captured.out + captured.err and str(tmp_path) not in captured.out + captured.err


@pytest.mark.parametrize("field", ["manifest", "image", "mask", "weights"])
def test_corrupt_checksums_fail_before_creating_output(tmp_path, field, capsys):
    manifest, models, _ = make_bundle(tmp_path)
    args = cli_args(tmp_path, manifest, models)
    if field == "manifest":
        args[args.index("--manifest-sha256") + 1] = "0" * 64
    else:
        path = models["baseline"]["path"] if field == "weights" else tmp_path / manifest["cases"][0][field]["path"]
        path.write_bytes(path.read_bytes() + b"corrupt")
    assert benchmark.main(args + ["--check-only"]) == 2
    assert not (tmp_path / "new-output").exists()
    assert str(tmp_path) not in capsys.readouterr().err


@pytest.mark.parametrize("tap", [[float("nan"), 0], [float("inf"), 0], [True, .2], [-.1, .2], [.2, 1.01], [0], "0,0"])
def test_nonfinite_or_outside_taps_are_rejected(tmp_path, tap):
    manifest, models, _ = make_bundle(tmp_path)
    manifest["cases"][0]["tap"] = tap
    # Nonfinite JSON itself is invalid, while finite invalid taps reach preflight.
    if isinstance(tap, list) and any(isinstance(v, float) and not np.isfinite(v) for v in tap):
        raw = json.dumps(manifest).encode()
        path = tmp_path / "private-manifest.json"
        path.write_bytes(raw)
        with pytest.raises(benchmark.BenchmarkError, match="nonfinite_json"):
            benchmark.validate_inputs(path, benchmark.digest(raw), models)
    else:
        with pytest.raises(benchmark.BenchmarkError, match="invalid_tap"):
            validated(tmp_path, manifest, models)


@pytest.mark.parametrize("bad_mask,code", [(np.zeros((9, 12), np.uint8), "mask_format_or_dimensions"),
                                          (np.full((10, 12), 128, np.uint8), "mask_not_binary"),
                                          (np.zeros((10, 12, 4), np.uint8), "mask_format_or_dimensions")])
def test_mask_alignment_binary_and_encoding_are_enforced(tmp_path, bad_mask, code):
    manifest, models, _ = make_bundle(tmp_path, accepted=False)
    raw = png(bad_mask)
    (tmp_path / "private-mask.png").write_bytes(raw)
    manifest["cases"][0]["mask"]["sha256"] = benchmark.digest(raw)
    with pytest.raises(benchmark.BenchmarkError, match=code):
        validated(tmp_path, manifest, models)


def test_exif_orientation_is_applied_before_mask_dimension_check(tmp_path):
    manifest, models, _ = make_bundle(tmp_path, accepted=False)
    photo = Image.new("RGB", (9, 5), (10, 40, 60))
    exif = photo.getexif()
    exif[274] = 6
    raw = io.BytesIO()
    photo.save(raw, "JPEG", exif=exif)
    (tmp_path / "private-photo.png").write_bytes(raw.getvalue())
    manifest["cases"][0]["image"]["sha256"] = benchmark.digest(raw.getvalue())
    mask = png(np.full((9, 5), 255, np.uint8))
    (tmp_path / "private-mask.png").write_bytes(mask)
    manifest["cases"][0]["mask"]["sha256"] = benchmark.digest(mask)
    result = validated(tmp_path, manifest, models)
    case = result[2][0][0]
    assert (case["width"], case["height"]) == (5, 9)


@pytest.mark.parametrize("shared", ["bytes", "pixels", "scene", "tree", "series"])
def test_known_cross_split_leakage_is_rejected_without_creating_a_new_split(tmp_path, shared):
    manifest, models, _ = make_bundle(tmp_path, labelled=False)
    other = copy.deepcopy(manifest["cases"][0])
    other.update(id="second-target", scene_id="different-scene")
    other["provenance"] = {"split": "test", "tree_id": "other-tree", "series_id": "other-series"}
    if shared not in ("bytes", "pixels"):
        other_raw = png(np.full((10, 12, 3), 40, np.uint8))
        (tmp_path / "other.png").write_bytes(other_raw)
        other["image"] = {"path": "other.png", "sha256": benchmark.digest(other_raw)}
    if shared == "pixels":
        buffer = io.BytesIO()
        Image.open(tmp_path / "private-photo.png").save(buffer, "BMP")
        (tmp_path / "other.bmp").write_bytes(buffer.getvalue())
        other["image"] = {"path": "other.bmp", "sha256": benchmark.digest(buffer.getvalue())}
        assert other["image"]["sha256"] != manifest["cases"][0]["image"]["sha256"]
    if shared == "scene":
        other["scene_id"] = manifest["cases"][0]["scene_id"]
    if shared in ("tree", "series"):
        other["provenance"][shared + "_id"] = manifest["cases"][0]["provenance"][shared + "_id"]
    manifest["cases"].append(other)
    with pytest.raises(benchmark.BenchmarkError, match="cross_split_leakage"):
        validated(tmp_path, manifest, models)


def training_record(models):
    return {"weights_sha256": models["baseline"]["sha256"], "complete": True,
            "image_sha256s": [], "pixel_sha256s": [], "scene_ids": [], "tree_ids": [], "series_ids": []}


@pytest.mark.parametrize("overlap", ["image_sha256s", "scene_ids", "tree_ids", "series_ids"])
def test_supplied_training_provenance_cannot_overlap_holdout(tmp_path, overlap):
    manifest, models, _ = make_bundle(tmp_path)
    record = training_record(models)
    case = manifest["cases"][0]
    value = {"image_sha256s": case["image"]["sha256"], "scene_ids": case["scene_id"],
             "tree_ids": case["provenance"]["tree_id"], "series_ids": case["provenance"]["series_id"]}[overlap]
    record[overlap] = [value]
    manifest["training_provenance"] = {"baseline": record}
    with pytest.raises(benchmark.BenchmarkError, match="training_holdout_leakage"):
        validated(tmp_path, manifest, models)


def test_training_provenance_is_bound_to_exact_weights(tmp_path):
    manifest, models, _ = make_bundle(tmp_path)
    record = training_record(models)
    record["weights_sha256"] = "0" * 64
    manifest["training_provenance"] = {"baseline": record}
    with pytest.raises(benchmark.BenchmarkError, match="training_provenance_binding_invalid"):
        validated(tmp_path, manifest, models)


@pytest.mark.parametrize("missing", ["mask", "review", "holdout", "accepted_decision", "review_hash"])
def test_quality_validation_requires_bound_accepted_masks_and_explicit_holdout(tmp_path, missing):
    manifest, models, _ = make_bundle(tmp_path)
    manifest["purpose"] = "quality_validation"
    case = manifest["cases"][0]
    if missing == "mask":
        case.pop("mask")
        case.pop("review")
    elif missing == "review":
        case.pop("review")
    elif missing == "holdout":
        case["provenance"]["split"] = "train"
    elif missing == "accepted_decision":
        case["review"]["decisions"] = []
    else:
        case["review"]["mask_sha256"] = "0" * 64
    with pytest.raises(benchmark.BenchmarkError):
        validated(tmp_path, manifest, models)


def test_reviewed_quality_data_never_implies_activation_or_verified_training_independence(tmp_path):
    manifest, models, _ = make_bundle(tmp_path)
    manifest["purpose"] = "quality_validation"
    output, frozen = snapshot(tmp_path, manifest, models)
    assert frozen["purpose"] == "quality_validation"
    assert frozen["training_independence"]["baseline"] == "not_supplied"
    assert any("not independently verified" in value for value in frozen["limitations"])
    assert not (output / "COMPLETE").exists()


def test_real_existing_evaluator_is_reused_with_exact_taps_and_raw_selection_warnings(tmp_path, existing_evaluator):
    manifest, models, truth = make_bundle(tmp_path, candidate=True)
    output, frozen = snapshot(tmp_path, manifest, models)
    prediction = truth.copy()
    prediction[0, 0] = 255
    runtimes = {"baseline": FakeRuntime(warnings=["tree_segmentation_returned_no_masks"]),
                "candidate": FakeRuntime(prediction, ["tap_did_not_hit_mask_nearest_tree_selected"])}
    evaluations = []
    def evaluator(*args):
        evaluations.append(args[1][0]["correction_id"])
        return existing_evaluator(*args)
    report = benchmark.evaluate(frozen, output, lambda path, _: runtimes[path.stem], evaluator)
    before = report["models"]["baseline"]["overall"]
    after = report["models"]["candidate"]["overall"]
    assert evaluations == ["private-target", "private-target"]
    assert runtimes["baseline"].taps == runtimes["candidate"].taps == [(.3, .25)]
    assert before["misses"] == 1 and before["mean_iou"] == 0
    assert after["mean_iou"] == pytest.approx(.8) and after["mean_dice"] == pytest.approx(8 / 9)
    assert after["false_positive_pixels"] == 1 and after["detections"] == 1
    assert after["usable_detections"] == 0 and after["target_unresolved"] == 1
    assert report["models"]["candidate"]["by_condition"]["sparse_crown"]["cases"] == 1
    assert report["models"]["candidate"]["by_scene"]["private-scene"]["scene_count"] == 1
    assert report["candidate_minus_baseline"]["mean_iou"] == pytest.approx(.8)
    assert not report["activation_eligible"]
    assert not report["independent_validation_established"] and report["evidence_level"] == "diagnostic_only"
    case = report["models"]["candidate"]["cases"][0]
    assert case["true_positive_pixels"] == 4 and case["false_negative_pixels"] == 0
    assert (output / case["overlay_file"]).is_file()


def test_unlabelled_photo_has_detection_diagnostic_but_no_quality_score(tmp_path):
    manifest, models, _ = make_bundle(tmp_path, labelled=False)
    output, frozen = snapshot(tmp_path, manifest, models)
    def forbidden_evaluation(*args):
        pytest.fail("unlabelled diagnostics have no ground-truth metric")
    report = benchmark.evaluate(frozen, output, lambda *_: FakeRuntime(), forbidden_evaluation)
    model = report["models"]["baseline"]
    assert model["overall"]["unlabelled_diagnostics"] == 1
    assert model["overall"]["mean_iou"] is None and model["overall"]["misses"] == 0
    assert model["cases"][0]["metrics"] is None
    assert model["cases"][0]["evidence"] == "unlabelled_diagnostic"


def test_negative_diagnostic_counts_false_positive_instead_of_silently_averaging_empty_iou(tmp_path, existing_evaluator):
    manifest, models, _ = make_bundle(tmp_path, accepted=False)
    negative = png(np.zeros((10, 12), np.uint8))
    (tmp_path / "private-mask.png").write_bytes(negative)
    manifest["cases"][0]["mask"]["sha256"] = benchmark.digest(negative)
    output, frozen = snapshot(tmp_path, manifest, models)
    prediction = np.zeros((10, 12), np.uint8)
    prediction[0, 0] = 255
    report = benchmark.evaluate(frozen, output, lambda *_: FakeRuntime(prediction), existing_evaluator)
    overall = report["models"]["baseline"]["overall"]
    assert overall["false_positive_negative_cases"] == 1 and overall["false_positive_pixels"] == 1
    assert overall["mean_iou"] is None and overall["misses"] == 0


@pytest.mark.parametrize("corruption", ["weights/baseline.pt", "images/0.png", "masks/0.png"])
def test_frozen_asset_corruption_prevents_inference(tmp_path, corruption, existing_evaluator):
    manifest, models, _ = make_bundle(tmp_path)
    output, frozen = snapshot(tmp_path, manifest, models)
    path = output / corruption
    path.write_bytes(path.read_bytes() + b"changed")
    runtime = FakeRuntime()
    with pytest.raises(benchmark.BenchmarkError, match="frozen_.*corrupt"):
        benchmark.evaluate(frozen, output, lambda *_: runtime, existing_evaluator)
    assert not runtime.taps


@pytest.mark.parametrize("destination", ["existing-directory", "private-photo.png", "private-manifest.json"])
def test_existing_destination_or_source_is_never_overwritten(tmp_path, destination, capsys):
    manifest, models, _ = make_bundle(tmp_path)
    output = tmp_path / destination
    if destination == "existing-directory":
        output.mkdir()
        (output / "preserve").write_bytes(b"preserve")
    args = cli_args(tmp_path, manifest, models, output)
    originals = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert benchmark.main(args + ["--check-only"]) == 2
    assert all(path.read_bytes() == raw for path, raw in originals.items())
    assert "output_must_be_new_non_source_directory" in capsys.readouterr().err


def test_runtime_failure_is_sanitized_and_has_no_completion_receipt(tmp_path, capsys, monkeypatch):
    manifest, models, _ = make_bundle(tmp_path)
    def error(*args):
        raise RuntimeError("private-owner and C:/secret/location with credentials")
    monkeypatch.setattr(benchmark, "launch_worker", error)
    assert benchmark.main(cli_args(tmp_path, manifest, models)) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"status": "failed", "error": "benchmark_failed"}
    assert not captured.out and "secret" not in captured.err and "private-owner" not in captured.err
    assert not (tmp_path / "new-output/COMPLETE").exists()
    assert (tmp_path / "new-output/failure.json").is_file()


def test_child_time_limit_kills_inference_process_and_keeps_logs_local(tmp_path, monkeypatch):
    manifest, models, _ = make_bundle(tmp_path)
    output, _ = snapshot(tmp_path, manifest, models)
    observed = {}
    class Process:
        killed = False
        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("private-runtime", timeout)
            return 1
        def kill(self):
            self.killed = True
    process = Process()
    def start(args, **kwargs):
        observed.update(args=args, **kwargs)
        return process
    monkeypatch.setattr(benchmark.subprocess, "Popen", start)
    with pytest.raises(benchmark.BenchmarkError, match="runtime_time_limit"):
        benchmark.launch_worker(output, 1, 2)
    assert process.killed
    assert observed["stdout"] is observed["stderr"]
    assert observed["env"]["OMP_NUM_THREADS"] == observed["env"]["OPENBLAS_NUM_THREADS"] == "2"
    assert observed["args"][-1] == benchmark.file_hash(output / "frozen.json")
    assert (output / "runtime.log").exists() and not (output / "COMPLETE").exists()


@pytest.mark.parametrize("arguments", [["--device", "cuda"], ["--threads", "0"], ["--threads", "5"],
                                      ["--max-seconds", "1801"], ["--imgsz", "1537"],
                                      ["--confidence", "nan"], ["--candidate", "private-no-hash.pt"],
                                      ["--warmup-runs", "-1"], ["--warmup-runs", "6"],
                                      ["--timing-repeats", "0"], ["--timing-repeats", "11"]])
def test_cli_enforces_explicit_cpu_resource_bounds(tmp_path, arguments, capsys):
    manifest, models, _ = make_bundle(tmp_path)
    assert benchmark.main(cli_args(tmp_path, manifest, models) + arguments + ["--check-only"]) == 2
    assert not (tmp_path / "new-output").exists()
    assert "private-no-hash" not in capsys.readouterr().err


def test_nonfinite_json_and_duplicate_keys_are_rejected():
    with pytest.raises(benchmark.BenchmarkError, match="duplicate_json_key"):
        benchmark.parse_json(b'{"purpose":"diagnostic","purpose":"quality_validation"}')


def test_completed_child_receipt_is_verified_before_reporting_success(tmp_path, monkeypatch):
    manifest, models, _ = make_bundle(tmp_path)
    output, _ = snapshot(tmp_path, manifest, models)
    (output / "report.json").write_bytes(b"{}")
    (output / "COMPLETE").write_bytes(benchmark.json_bytes({"report_sha256": "0" * 64,
                                                         "frozen_sha256": benchmark.file_hash(output / "frozen.json")}))
    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *a, **k: SimpleNamespace(wait=lambda **kw: 0))
    with pytest.raises(benchmark.BenchmarkError, match="completion_checksum_mismatch"):
        benchmark.launch_worker(output, 2, 1)


def test_changed_frozen_manifest_is_rejected_before_ml_dependencies_are_imported(tmp_path):
    manifest, models, _ = make_bundle(tmp_path)
    output, _ = snapshot(tmp_path, manifest, models)
    with pytest.raises(benchmark.BenchmarkError, match="frozen_manifest_corrupt"):
        benchmark.run_worker(output, "0" * 64)


def test_snapshot_creates_runtime_directories_before_ml_imports(tmp_path):
    manifest, models, _ = make_bundle(tmp_path)
    output, _ = snapshot(tmp_path, manifest, models)
    # The real library checks YOLO_CONFIG_DIR's parent. An existing parent keeps
    # its settings inside this run rather than silently falling back to CWD.
    assert (output / "runtime-config").is_dir()
    assert (output / "runtime-cache").is_dir()
    assert not list((output / "runtime-config").iterdir())
    assert not (tmp_path / "Ultralytics").exists()


@pytest.mark.parametrize("path", [r"\\private-host\share\photo.png", "//private-host/share/photo.png", "https://private-host/photo.png"])
def test_remote_asset_paths_fail_before_filesystem_resolution(tmp_path, path, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(Path, "resolve", lambda *args, **kwargs: pytest.fail("remote resolution must not occur"))
        with pytest.raises(benchmark.BenchmarkError, match="invalid_local_file"):
            benchmark.resolve_file(path, tmp_path, benchmark.MAX_ASSET_BYTES)


@pytest.mark.parametrize("names,tree_id", [
    ({0: "tree"}, 0),
    ({0: "tree", 1: "stick"}, 0),
    ({0: "stick", 1: "tree"}, 1),
    ({0: "street", 7: " TREE "}, 7),
    ({4: "дерево", 8: "эталон"}, 4),
])
def test_named_tree_mapping_preserves_class_id_and_excludes_non_tree(names, tree_id):
    runtime = SimpleNamespace(model_names=names, _tree_class_ids=lambda: set(names))
    mapping = benchmark.configure_benchmark_tree_class(runtime)
    assert runtime._tree_class_ids() == {tree_id}
    assert mapping["tree_class_id"] == tree_id
    assert mapping["model_classes"] == {str(key): value for key, value in sorted(names.items())}
    assert mapping["ignored_class_ids"] == sorted(set(names) - {tree_id})
    assert mapping["logical_class"] == "tree"


@pytest.mark.parametrize("names,error", [
    ({0: "person", 1: "potted plant"}, "tree_class_name_required"),
    ({0: "street"}, "tree_class_name_required"),
    ({0: "tree", 1: "дерево"}, "ambiguous_tree_class_schema"),
    ({0: "tree", 1: "Trees"}, "ambiguous_tree_class_schema"),
    ({"1": "tree"}, "invalid_model_class_schema"),
    ({True: "tree"}, "invalid_model_class_schema"),
    ({-1: "tree"}, "invalid_model_class_schema"),
    ({0: 42}, "invalid_model_class_schema"),
    ({}, "invalid_model_class_schema"),
])
def test_unknown_or_ambiguous_model_classes_are_not_renamed_to_tree(names, error):
    runtime = SimpleNamespace(model_names=names)
    with pytest.raises(benchmark.BenchmarkError, match=error):
        benchmark.configure_benchmark_tree_class(runtime)
    assert not hasattr(runtime, "benchmark_class_mapping")


@pytest.mark.parametrize("labelled", [False, True])
def test_warmed_timing_excludes_decode_validation_metrics_and_overlay_io(
        tmp_path, monkeypatch, existing_evaluator, labelled):
    manifest, models, truth = make_bundle(tmp_path, labelled=labelled, candidate=True)
    output, frozen = snapshot(tmp_path, manifest, models)
    frozen["settings"].update(warmup_runs=1, timing_repeats=3)
    import cv2
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock.now)
    read = cv2.imread
    write = cv2.imwrite
    validate = benchmark.validate_prediction
    def expensive_read(*args, **kwargs):
        clock.now += 20
        return read(*args, **kwargs)
    def expensive_write(*args, **kwargs):
        clock.now += 100
        return write(*args, **kwargs)
    def expensive_validation(*args):
        clock.now += 7
        return validate(*args)
    def expensive_thread_check():
        clock.now += 13
        return {"observed": {"torch_threads": 1}}
    monkeypatch.setattr(cv2, "imread", expensive_read)
    monkeypatch.setattr(cv2, "imwrite", expensive_write)
    monkeypatch.setattr(benchmark, "validate_prediction", expensive_validation)
    class ClockRuntime(FakeRuntime):
        def __init__(self, duration):
            super().__init__(truth)
            self.duration, self.images = duration, []
            self.benchmark_thread_policy_check = expensive_thread_check
        def infer(self, image, tap_x=None, tap_y=None):
            clock.now += self.duration
            self.images.append(image)
            return super().infer(image, tap_x, tap_y)
    runtimes = {"baseline": ClockRuntime(2), "candidate": ClockRuntime(5)}
    report = benchmark.evaluate(frozen, output, lambda path, _: runtimes[path.stem], existing_evaluator)
    for name, duration in [("baseline", 2), ("candidate", 5)]:
        runtime = runtimes[name]
        row = report["models"][name]["cases"][0]
        assert len(runtime.taps) == 5  # one quality pass + one warmup + three timed runs
        assert runtime.taps == [(.3, .25)] * 5
        assert all(image is runtime.images[-1] for image in runtime.images[-4:])
        assert row["seconds"] > duration
        assert row["inference_timing"]["samples_seconds"] == [duration] * 3
        assert row["inference_timing"]["mean_seconds"] == duration
        assert row["inference_timing"]["median_seconds"] == duration
        assert report["models"][name]["inference_timing"]["mean_seconds"] == duration
        assert report["models"][name]["inference_timing"]["sample_count"] == 3
        assert report["models"][name]["thread_policy"] == {"observed": {"torch_threads": 1}}
        assert (row["metrics"] is None) == (not labelled)
    assert report["candidate_minus_baseline"]["inference_mean_seconds"] == 3
    assert report["candidate_minus_baseline"]["inference_median_seconds"] == 3
    assert report["timing_protocol"]["excluded"] == [
        "model_load", "warmup", "image_decode", "metric_evaluation",
        "prediction_contract_checks", "file_IO"]


def test_old_frozen_settings_keep_single_inference_and_unmeasured_latency(tmp_path, existing_evaluator):
    manifest, models, truth = make_bundle(tmp_path)
    output, frozen = snapshot(tmp_path, manifest, models)
    runtime = FakeRuntime(truth)
    report = benchmark.evaluate(frozen, output, lambda *_: runtime, existing_evaluator)
    assert runtime.taps == [(.3, .25)]
    assert report["timing_protocol"]["enabled"] is False
    assert report["models"]["baseline"]["inference_timing"] is None
    assert report["models"]["baseline"]["cases"][0]["inference_timing"] is None
    assert report["models"]["baseline"]["thread_policy"] is None


@pytest.mark.parametrize("settings", [{"timing_repeats": True}, {"timing_repeats": 11},
                                     {"timing_repeats": 1, "warmup_runs": 6},
                                     {"timing_repeats": 0, "warmup_runs": 1}])
def test_timing_protocol_rejects_invalid_or_unbounded_repetition(settings):
    with pytest.raises(benchmark.BenchmarkError, match="invalid_timing_bounds"):
        benchmark.timing_protocol(settings)


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf")])
def test_nonfinite_or_negative_latency_is_not_reported(monkeypatch, duration):
    ticks = iter([0.0, duration])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    protocol = benchmark.timing_protocol({"warmup_runs": 0, "timing_repeats": 1})
    with pytest.raises(benchmark.BenchmarkError, match="invalid_runtime_timing"):
        benchmark.measure_inference(FakeRuntime(), np.zeros((2, 2, 3), np.uint8), None, protocol)


def fake_thread_libraries():
    state = SimpleNamespace(torch_threads=8, interop=8, cv=0)
    torch = SimpleNamespace(
        set_num_threads=lambda value: setattr(state, "torch_threads", value),
        set_num_interop_threads=lambda value: setattr(state, "interop", value),
        get_num_threads=lambda: state.torch_threads,
        get_num_interop_threads=lambda: state.interop)
    cv = SimpleNamespace(setNumThreads=lambda value: setattr(state, "cv", value),
                         getNumThreads=lambda: state.cv)
    return state, torch, cv, SimpleNamespace(NUM_THREADS=8), SimpleNamespace(NUM_THREADS=8)


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_worker_pins_both_ultralytics_constants_and_observes_native_counts(threads):
    state, torch, cv, utils, torch_utils = fake_thread_libraries()
    check = benchmark.configure_runtime_threads(threads, torch, cv, utils, torch_utils)
    # Simulate the real select_device CPU reset after the initial setting.
    torch.set_num_threads(torch_utils.NUM_THREADS)
    assert check()["observed"] == {
        "torch_threads": threads, "torch_interop_threads": threads,
        "opencv_threads": threads, "ultralytics_threads": threads,
        "ultralytics_select_device_threads": threads}
    assert check()["requested_threads"] == threads
    state.torch_threads = 8
    with pytest.raises(benchmark.BenchmarkError, match="runtime_thread_policy_changed"):
        check()


@pytest.mark.parametrize("threads", [0, 5, True, 1.5])
def test_worker_rejects_invalid_thread_policy_before_changing_libraries(threads):
    state, torch, cv, utils, torch_utils = fake_thread_libraries()
    with pytest.raises(benchmark.BenchmarkError, match="invalid_runtime_threads"):
        benchmark.configure_runtime_threads(threads, torch, cv, utils, torch_utils)
    assert state.torch_threads == 8 and utils.NUM_THREADS == 8 and state.cv == 0


@pytest.mark.parametrize("warmup,repeats", [(1, 0), (0, 1)])
def test_native_thread_drift_fails_after_warmup_or_measured_inference(warmup, repeats):
    state, torch, cv, utils, torch_utils = fake_thread_libraries()
    check = benchmark.configure_runtime_threads(1, torch, cv, utils, torch_utils)
    class DriftingRuntime(FakeRuntime):
        benchmark_thread_policy_check = staticmethod(check)
        def infer(self, *args):
            state.torch_threads = 8
            return super().infer(*args)
    with pytest.raises(benchmark.BenchmarkError, match="runtime_thread_policy_changed"):
        benchmark.measure_inference(DriftingRuntime(), np.zeros((2, 2, 3), np.uint8), None,
                                    {"warmup_runs_per_case": warmup,
                                     "timing_repeats_per_case": repeats})


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires actual CPU ML environment")
def test_actual_ultralytics_select_device_keeps_enforced_thread_counts_in_private_child(tmp_path):
    # Import ML libraries only in a fresh, offline process. Both library
    # directories exist first; no user-wide Ultralytics settings are touched.
    config, cache = tmp_path / "config", tmp_path / "cache"
    config.mkdir()
    cache.mkdir()
    env = dict(os.environ)
    env.update(YOLO_CONFIG_DIR=str(config), YOLO_CACHE_DIR=str(cache),
               MODEL_CACHE_DIR=str(cache), MPLCONFIGDIR=str(cache),
               YOLO_AUTOINSTALL="false", CUDA_VISIBLE_DEVICES="-1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", TEMP=str(tmp_path), TMP=str(tmp_path))
    script = '''
import json, socket
def offline(*args, **kwargs):
    raise OSError("offline thread-policy test")
socket.create_connection = socket.getaddrinfo = offline
socket.socket.connect = socket.socket.connect_ex = socket.socket.sendto = offline
import cv2, torch
from ultralytics import utils
from ultralytics.utils import torch_utils
from tools.benchmark_tree_recognition import configure_runtime_threads
utils.NUM_THREADS = torch_utils.NUM_THREADS = 8
torch.set_num_threads(1)
torch_utils.select_device("cpu", verbose=False)
assert torch.get_num_threads() == 8, "regression prerequisite: actual select_device reset"
check = configure_runtime_threads(1, torch, cv2, utils, torch_utils)
torch_utils.select_device("cpu", verbose=False)
torch.mm(torch.ones((16, 16)), torch.ones((16, 16)))
print(json.dumps(check(), sort_keys=True))
'''
    child = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                           env=env, capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    observed = json.loads(child.stdout.splitlines()[-1])
    assert observed["requested_threads"] == 1
    assert set(observed["observed"].values()) == {1}
    assert observed["enforcement"] == "worker_local_verified_after_every_infer"
