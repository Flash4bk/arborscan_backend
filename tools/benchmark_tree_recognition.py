"""Offline, checksum-bound comparison of the existing tree inference runtime.

This tool never trains, contacts a registry, splits a dataset, or activates weights.
Run ``--help`` for the CLI. See MANIFEST_CONTRACT for the local manifest schema.
Library output stays in the new output directory; stdout contains totals only.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

SCHEMA = "arborscan.recognition-benchmark.v1"
MANIFEST_CONTRACT = """
{
  "schema": "arborscan.recognition-benchmark.v1",
  "purpose": "diagnostic|quality_validation",
  "cases": [{
    "id": "target-1", "scene_id": "scene-1", "conditions": ["sparse_crown"],
    "image": {"path": "photo.jpg", "sha256": "64 hex characters"},
    "mask": {"path": "target-mask.png", "sha256": "64 hex characters"},
    "tap": [0.4, 0.2],
    "provenance": {"split": "val|test|train", "tree_id": "tree-1", "series_id": "series-1"},
    "review": {"status": "accepted", "correction_id": "revision-1",
      "image_sha256": "same image hash", "mask_sha256": "same mask hash",
      "decisions": [{"action": "accepted", "actor_id": "reviewer", "at": "timestamp"}]}
  }],
  "training_provenance": {"baseline": {
    "weights_sha256": "same pinned baseline hash", "complete": true,
    "image_sha256s": [], "pixel_sha256s": [], "scene_ids": [], "tree_ids": [], "series_ids": []
  }}
}
Paths are relative to the manifest, or absolute local paths. URLs/UNC paths fail.
Mask, tap, provenance and review are optional for diagnostic cases. Masks must be
binary L/1 PNG aligned to the EXIF-oriented RGB image. Quality validation requires
a nonempty mask, accepted review attestation and explicit val/test split for every
case. Review/training provenance are local attestations, not server verification.
Training provenance is optional per pinned model; absence/incompleteness is explicit
in the report. Known image/pixel/scene/tree/series leakage always fails preflight.
No independent-quality or activation decision follows from this tool's completion.
Whole-tree classes are resolved by one exact tree/trees/дерево/деревья name;
other class IDs (such as stick) are excluded, never collapsed into tree. Pure CPU
inference latency uses decoded BGR input, untimed warmup and repeated calls.
Legacy case seconds includes workflow IO and is not the warmed latency. This
selected-mask benchmark does not calculate instance AP/mAP.
"""
HASH = re.compile(r"[0-9a-f]{64}\Z")
TOKEN = re.compile(r"[A-Za-z0-9_-]{1,120}\Z")
CONDITION = re.compile(r"[a-z][a-z0-9_]{0,47}\Z")
MAX_CASES = 500
MAX_PIXELS = 25_000_000
MAX_ASSET_BYTES = 15 * 1024**2
MAX_TOTAL_BYTES = 512 * 1024**2
MAX_WEIGHT_BYTES = 512 * 1024**2
TREE_NAMES = {"tree", "trees", "дерево", "деревья"}


class BenchmarkError(ValueError):
    """Only fixed error codes may leave the local output directory."""


def fail(code):
    raise BenchmarkError(code)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_hash(value):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        fail("invalid_sha256")
    return value


def file_hash(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2).encode() + b"\n"


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                fail("duplicate_json_key")
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda _: fail("nonfinite_json"))
    except (UnicodeError, json.JSONDecodeError):
        fail("invalid_json")


def token(value):
    if not isinstance(value, str) or not TOKEN.fullmatch(value):
        fail("invalid_identifier")
    return value


def resolve_file(value, base, max_bytes):
    if (not isinstance(value, str) or not value or "://" in value
            or value.startswith(("\\\\", "//"))):
        fail("invalid_local_file")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    try:
        path = path.resolve(strict=True)
        if not path.is_file() or not 0 < path.stat().st_size <= max_bytes:
            fail("input_size_or_type")
    except OSError:
        fail("input_unavailable")
    return path


def asset(spec, base):
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        fail("invalid_asset")
    path = resolve_file(spec["path"], base, MAX_ASSET_BYTES)
    raw = path.read_bytes()
    if digest(raw) != checked_hash(spec["sha256"]):
        fail("asset_checksum_mismatch")
    return path, raw


def oriented_image(raw):
    from PIL import Image, ImageOps
    try:
        with Image.open(io.BytesIO(raw)) as original:
            if original.width * original.height > MAX_PIXELS or getattr(original, "n_frames", 1) != 1:
                fail("image_limits")
            rgb = ImageOps.exif_transpose(original).convert("RGB")
            pixels = rgb.tobytes()
            pixel_hash = digest(str((rgb.height, rgb.width, 3)).encode() + pixels)
            buffer = io.BytesIO()
            rgb.save(buffer, "PNG")
            return buffer.getvalue(), (rgb.width, rgb.height), pixel_hash
    except BenchmarkError:
        raise
    except Exception:
        fail("image_decode_failed")


def checked_mask(raw, dimensions):
    from PIL import Image
    try:
        with Image.open(io.BytesIO(raw)) as original:
            if (original.format != "PNG" or original.mode not in ("1", "L")
                    or original.size != dimensions or original.getexif().get(274, 1) != 1
                    or getattr(original, "n_frames", 1) != 1):
                fail("mask_format_or_dimensions")
            binary = original.convert("L")
            histogram = binary.histogram()
            if sum(histogram[1:255]):
                fail("mask_not_binary")
            return int(histogram[255])
    except BenchmarkError:
        raise
    except Exception:
        fail("mask_decode_failed")


def reviewed(case):
    review = case.get("review")
    if review is None:
        return False
    if (not isinstance(review, dict) or review.get("status") != "accepted"
            or not isinstance(review.get("correction_id"), str)
            or not review["correction_id"] or not case.get("mask")
            or review.get("image_sha256") != case["image"]["sha256"]
            or review.get("mask_sha256") != case["mask"]["sha256"]):
        fail("review_binding_invalid")
    decisions = review.get("decisions", [])
    if not isinstance(decisions, list) or not any(isinstance(d, dict) and d.get("action") == "accepted"
               and d.get("actor_id") and d.get("at") for d in decisions):
        fail("accepted_review_required")
    return True


def provenance_keys(case):
    result = [("image_sha256s", case["image"]["sha256"]),
              ("pixel_sha256s", case["pixel_sha256"]), ("scene_ids", case["scene_id"])]
    provenance = case.get("provenance", {})
    result.extend((plural, provenance[singular]) for singular, plural in
                  (("tree_id", "tree_ids"), ("series_id", "series_ids")) if singular in provenance)
    return result


def check_leakage(cases, training, models):
    seen = {}
    for case in cases:
        split = case.get("provenance", {}).get("split")
        if split:
            for key in provenance_keys(case):
                if key in seen and seen[key] != split:
                    fail("cross_split_leakage")
                seen[key] = split
    if not isinstance(training, dict) or set(training) - set(models):
        fail("invalid_training_provenance")
    status = {}
    fields = {"image_sha256s", "pixel_sha256s", "scene_ids", "tree_ids", "series_ids"}
    for name, model in models.items():
        record = training.get(name)
        if record is None:
            status[name] = "not_supplied"
            continue
        if (not isinstance(record, dict) or record.get("weights_sha256") != model["sha256"]
                or not fields <= set(record) or not isinstance(record.get("complete"), bool)):
            fail("training_provenance_binding_invalid")
        sets = {}
        for field in fields:
            values = record[field]
            if not isinstance(values, list) or len(values) > 100_000:
                fail("invalid_training_provenance")
            sets[field] = {checked_hash(v) if field.endswith("sha256s") else token(v) for v in values}
        for case in cases:
            if case.get("provenance", {}).get("split") in ("val", "test"):
                if any(value in sets[field] for field, value in provenance_keys(case)):
                    fail("training_holdout_leakage")
        status[name] = "complete_local_attestation_no_known_overlap" if record["complete"] else "incomplete_local_attestation"
    return status


def validate_inputs(manifest_path, manifest_sha256, models):
    manifest_path = resolve_file(str(manifest_path), Path.cwd(), 2 * 1024**2)
    raw = manifest_path.read_bytes()
    if digest(raw) != checked_hash(manifest_sha256):
        fail("manifest_checksum_mismatch")
    manifest = parse_json(raw)
    if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA
            or manifest.get("purpose") not in ("diagnostic", "quality_validation")
            or not isinstance(manifest.get("cases"), list)
            or not 1 <= len(manifest["cases"]) <= MAX_CASES):
        fail("invalid_manifest")
    sources = [manifest_path]
    prepared, identifiers, retained = [], set(), 0
    for source in manifest["cases"]:
        if not isinstance(source, dict):
            fail("invalid_case")
        case = dict(source)
        identifier = token(case.get("id"))
        if identifier in identifiers:
            fail("duplicate_case_id")
        identifiers.add(identifier)
        token(case.get("scene_id"))
        conditions = case.get("conditions", [])
        if (not isinstance(conditions, list) or len(conditions) > 32
                or any(not isinstance(v, str) or not CONDITION.fullmatch(v) for v in conditions)):
            fail("invalid_conditions")
        case["conditions"] = sorted(set(conditions))
        provenance = case.get("provenance", {})
        if not isinstance(provenance, dict) or set(provenance) - {"split", "tree_id", "series_id"}:
            fail("invalid_provenance")
        if provenance.get("split") not in (None, "train", "val", "test"):
            fail("invalid_split")
        for field in ("tree_id", "series_id"):
            if field in provenance:
                token(provenance[field])
        tap = case.get("tap")
        if tap is not None and (not isinstance(tap, list) or len(tap) != 2 or any(
                isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) or not 0 <= v <= 1 for v in tap)):
            fail("invalid_tap")
        image_path, image_raw = asset(case.get("image"), manifest_path.parent)
        oriented, dimensions, pixel_hash = oriented_image(image_raw)
        sources.append(image_path)
        mask_raw, foreground = None, None
        if "mask" in case:
            mask_path, mask_raw = asset(case["mask"], manifest_path.parent)
            foreground = checked_mask(mask_raw, dimensions)
            sources.append(mask_path)
        case.update(pixel_sha256=pixel_hash, width=dimensions[0], height=dimensions[1],
                    foreground_pixels=foreground, review_attested=reviewed(case))
        if manifest["purpose"] == "quality_validation" and (
                not foreground or not case["review_attested"] or provenance.get("split") not in ("val", "test")):
            fail("quality_validation_requires_reviewed_holdout_masks")
        retained += len(image_raw) + len(oriented) + len(mask_raw or b"") + dimensions[0] * dimensions[1] * 3
        if retained > MAX_TOTAL_BYTES:
            fail("snapshot_memory_limit")
        prepared.append((case, image_raw, oriented, mask_raw))
    checked_models = {}
    for name, spec in models.items():
        path = resolve_file(str(spec["path"]), Path.cwd(), MAX_WEIGHT_BYTES)
        if path.suffix.lower() != ".pt" or file_hash(path) != checked_hash(spec["sha256"]):
            fail("weights_checksum_or_format")
        checked_models[name] = {"path": path, "sha256": spec["sha256"]}
        sources.append(path)
    independence = check_leakage([row[0] for row in prepared], manifest.get("training_provenance", {}), checked_models)
    return manifest, raw, prepared, checked_models, sources, independence


def write_new(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)


def prepare_snapshot(output, validated, settings):
    manifest, raw, prepared, models, sources, independence = validated
    if str(output).startswith(("\\\\", "//")):
        fail("output_must_be_local")
    output = Path(output).resolve()
    if output.exists() or not output.parent.is_dir() or any(output == p or output in p.parents for p in sources):
        fail("output_must_be_new_non_source_directory")
    output.mkdir(exist_ok=False)
    # Ultralytics checks the parent of its settings directory before creating it.
    # Missing runtime-config would make it fall back to the current working
    # directory instead of keeping this run's settings beside its frozen inputs.
    for name in ("images", "masks", "originals", "weights", "runtime-config", "runtime-cache"):
        (output / name).mkdir()
    write_new(output / "source-manifest.json", raw)
    cases = []
    for number, (case, original, oriented, mask) in enumerate(prepared):
        case = dict(case)
        case["image_file"] = f"images/{number}.png"
        case["original_file"] = f"originals/{number}.bin"
        case["oriented_sha256"] = digest(oriented)
        write_new(output / case["image_file"], oriented)
        write_new(output / case["original_file"], original)
        if mask is not None:
            case["mask_file"] = f"masks/{number}.png"
            write_new(output / case["mask_file"], mask)
        cases.append(case)
    frozen_models = {}
    for name, spec in models.items():
        relative = f"weights/{name}.pt"
        with spec["path"].open("rb") as source, (output / relative).open("xb") as destination:
            shutil.copyfileobj(source, destination, 1024**2)
        if file_hash(output / relative) != spec["sha256"]:
            fail("weights_changed_during_snapshot")
        frozen_models[name] = {"file": relative, "sha256": spec["sha256"]}
    frozen = {"schema": SCHEMA, "purpose": manifest["purpose"], "source_manifest_sha256": digest(raw),
              "cases": cases, "models": frozen_models, "settings": settings,
              "training_independence": independence,
              "limitations": ["Local review/provenance attestations are not independently verified server decisions.",
                              "Known-overlap checks do not detect every unrecorded series or near duplicate.",
                              "Completion does not establish general accuracy or authorize activation.",
                              "Raw server-selection metrics retain masks that the app rejects after a nearest-tree warning.",
                              "Empty-mask diagnostics are excluded from mean IoU/Dice and counted as negative cases.",
                              "Unlabelled cases are diagnostics and have no quality metrics."]}
    write_new(output / "frozen.json", json_bytes(frozen))
    return output, frozen


def configure_benchmark_tree_class(runtime):
    """Map an explicitly named tree class; never collapse stick/COCO classes."""
    names = runtime.model_names
    if (not isinstance(names, dict) or not names or any(
            isinstance(key, bool) or not isinstance(key, int) or key < 0
            or not isinstance(value, str) for key, value in names.items())):
        fail("invalid_model_class_schema")
    selected = [key for key, value in names.items() if value.strip().casefold() in TREE_NAMES]
    if len(selected) != 1:
        fail("tree_class_name_required" if not selected else "ambiguous_tree_class_schema")
    tree_id = selected[0]
    # The application runtime accepts broad substring names. This offline
    # comparison instead permits only the one explicitly mapped whole-tree ID.
    runtime._tree_class_ids = lambda: {tree_id}
    runtime.benchmark_class_mapping = {
        "logical_class": "tree", "tree_class_id": tree_id,
        "model_classes": {str(key): value for key, value in sorted(names.items())},
        "ignored_class_ids": sorted(set(names) - {tree_id}),
        "resolution": "one_exact_tree_class_name",
    }
    return runtime.benchmark_class_mapping


def validate_prediction(prediction, image):
    if prediction.detected:
        import numpy as np
        mask = prediction.mask
        if mask is None or mask.shape != image.shape[:2] or not np.isfinite(mask).all():
            fail("runtime_mask_contract")


def configure_runtime_threads(threads, torch_module, cv2_module, yolo_utils, torch_utils):
    """Pin and observe native thread counts in this disposable worker only.

    Ultralytics' first CPU prediction calls select_device(), which resets Torch
    to its imported NUM_THREADS constant. Pin both copies before prediction,
    rather than merely labelling a requested count as an observed count.
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 4:
        fail("invalid_runtime_threads")
    yolo_utils.NUM_THREADS = torch_utils.NUM_THREADS = threads
    torch_module.set_num_threads(threads)
    torch_module.set_num_interop_threads(threads)
    cv2_module.setNumThreads(threads)

    def check():
        counts = {"torch_threads": torch_module.get_num_threads(),
                  "torch_interop_threads": torch_module.get_num_interop_threads(),
                  "opencv_threads": cv2_module.getNumThreads(),
                  "ultralytics_threads": yolo_utils.NUM_THREADS,
                  "ultralytics_select_device_threads": torch_utils.NUM_THREADS}
        if any(value != threads for value in counts.values()):
            fail("runtime_thread_policy_changed")
        return {"requested_threads": threads, "observed": counts,
                "enforcement": "worker_local_verified_after_every_infer"}
    check()
    return check


def check_runtime_threads(runtime):
    check = getattr(runtime, "benchmark_thread_policy_check", None)
    return check() if check is not None else None


def timing_protocol(settings):
    # Earlier frozen v1 runs and library callers lacked these fields. Do not
    # silently change their inference count or relabel their legacy seconds.
    repeats = settings.get("timing_repeats", 0)
    warmup = settings.get("warmup_runs", 1 if repeats else 0)
    if (isinstance(repeats, bool) or not isinstance(repeats, int) or not 0 <= repeats <= 10
            or isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup <= 5
            or (not repeats and warmup)):
        fail("invalid_timing_bounds")
    return {"enabled": bool(repeats), "warmup_runs_per_case": warmup,
            "timing_repeats_per_case": repeats,
            "scope": "resident_CPU_runtime.infer_on_decoded_BGR_including_selection",
            "excluded": ["model_load", "warmup", "image_decode", "metric_evaluation",
                         "prediction_contract_checks", "file_IO"],
            "thread_policy": "requested_in_settings_actual_in_each_model_thread_policy",
            "legacy_seconds_scope": "one_shot_case_workflow_including_decode_or_evaluator_IO"}


def summarize_timing(samples):
    return {"sample_count": len(samples), "samples_seconds": samples,
            "mean_seconds": statistics.fmean(samples),
            "median_seconds": statistics.median(samples),
            "min_seconds": min(samples), "max_seconds": max(samples)}


def measure_inference(runtime, image, tap, protocol):
    """Time the existing inference call only, with all validation outside it."""
    for _ in range(protocol["warmup_runs_per_case"]):
        prediction = runtime.infer(image, *(tap or (None, None)))
        check_runtime_threads(runtime)
        validate_prediction(prediction, image)
    samples = []
    for _ in range(protocol["timing_repeats_per_case"]):
        started = time.perf_counter()
        prediction = runtime.infer(image, *(tap or (None, None)))
        seconds = time.perf_counter() - started
        if not math.isfinite(seconds) or seconds < 0:
            fail("invalid_runtime_timing")
        check_runtime_threads(runtime)
        validate_prediction(prediction, image)
        samples.append(seconds)
    return summarize_timing(samples)


class RecordingRuntime:
    """Pass the supplied target through the existing selection implementation."""
    def __init__(self, runtime, tap):
        self.runtime, self.tap, self.prediction = runtime, tap, None

    def infer(self, image):
        self.prediction = self.runtime.infer(image, *(self.tap or (None, None)))
        check_runtime_threads(self.runtime)
        validate_prediction(self.prediction, image)
        return self.prediction


def summarize(rows):
    labelled = [r for r in rows if r.get("metrics") is not None]
    positive = [r for r in labelled if r["foreground_pixels"] > 0]
    return {"cases": len(rows), "labelled_cases": len(labelled),
            "unlabelled_diagnostics": len(rows) - len(labelled),
            "review_attested_cases": sum(r["review_attested"] for r in rows),
            "scene_count": len({r["scene_id"] for r in rows}),
            "detections": sum(r["detected"] for r in rows),
            "usable_detections": sum(r["detected"] and not r["selection_warning"] for r in rows),
            "misses": sum(not r["detected"] for r in positive),
            "target_unresolved": sum(not r["detected"] or r["selection_warning"] for r in positive),
            "zero_overlap_detections": sum(r["detected"] and r["true_positive_pixels"] == 0 for r in positive),
            "selection_warnings": sum(bool(r["selection_warning"]) for r in rows),
            "false_positive_pixels": sum(r["false_positive_pixels"] for r in labelled),
            "false_positive_negative_cases": sum(r["foreground_pixels"] == 0 and r["detected"] for r in labelled),
            "correct_negative_cases": sum(r["foreground_pixels"] == 0 and not r["detected"] for r in labelled),
            "mean_iou": sum(r["metrics"]["iou"] for r in positive) / len(positive) if positive else None,
            "mean_dice": sum(r["metrics"]["dice"] for r in positive) / len(positive) if positive else None}


def evaluate(frozen, output, runtime_factory, evaluator):
    """Reuse quality_train.segmentation_metrics, including its actual overlays."""
    import cv2
    import numpy as np
    from PIL import Image
    result = {key: frozen[key] for key in ("schema", "purpose", "source_manifest_sha256", "settings",
                                         "training_independence", "limitations")}
    result.update(activation_eligible=False, independent_validation_established=False,
                  evidence_level="reviewed_holdout_benchmark" if frozen["purpose"] == "quality_validation" else "diagnostic_only",
                  models={})
    protocol = timing_protocol(frozen["settings"])
    result["timing_protocol"] = protocol
    result["limitations"] = list(result["limitations"]) + [
        "Legacy case seconds is not warmed inference latency; use inference_timing when measured.",
        "Settings threads is requested; actual native counts are verified only when model.thread_policy is present.",
        "Single selected-mask metrics do not compute instance AP or mAP.",
    ]
    for name, model in frozen["models"].items():
        weight_path = output / model["file"]
        if file_hash(weight_path) != model["sha256"]:
            fail("frozen_weights_corrupt")
        runtime = runtime_factory(weight_path, frozen["settings"])
        rows = []
        for number, case in enumerate(frozen["cases"]):
            image_path = output / case["image_file"]
            if file_hash(image_path) != case["oriented_sha256"]:
                fail("frozen_image_corrupt")
            if case.get("mask_file") and file_hash(output / case["mask_file"]) != case["mask"]["sha256"]:
                fail("frozen_mask_corrupt")
            folder = output / "diagnostics" / name / str(number)
            folder.mkdir(parents=True, exist_ok=False)
            adapter = RecordingRuntime(runtime, case.get("tap"))
            metrics = None
            started = time.perf_counter()
            if case.get("mask_file"):
                item = {"correction_id": case["id"], "split": case.get("provenance", {}).get("split", "diagnostic"),
                        "files": {case["image_file"]: case["oriented_sha256"], case["mask_file"]: case["mask"]["sha256"]}}
                metrics = evaluator(adapter, [item], output, folder)["cases"][0]
            else:
                adapter.infer(cv2.imread(str(image_path)))
            prediction = adapter.prediction
            seconds = time.perf_counter() - started
            row = {"id": case["id"], "scene_id": case["scene_id"], "conditions": case["conditions"],
                   "image_sha256": case["image"]["sha256"], "pixel_sha256": case["pixel_sha256"],
                   "mask_sha256": case.get("mask", {}).get("sha256"), "tap": case.get("tap"),
                   "provenance": case.get("provenance", {}), "review_attested": case["review_attested"],
                   "foreground_pixels": case["foreground_pixels"], "detected": bool(prediction.detected),
                   "confidence": prediction.confidence, "bbox": prediction.bbox,
                   "warnings": list(prediction.warnings), "seconds": seconds, "metrics": metrics,
                   "selection_warning": "tap_did_not_hit_mask_nearest_tree_selected" in prediction.warnings,
                   "evidence": "review_attested_mask_metrics" if case["review_attested"] else
                               ("unreviewed_mask_diagnostic" if metrics else "unlabelled_diagnostic")}
            predicted = prediction.mask > 127 if prediction.detected else np.zeros((case["height"], case["width"]), bool)
            mask_output = folder / "selected-mask.png"
            Image.fromarray(predicted.astype(np.uint8) * 255).save(mask_output)
            if metrics is not None:
                with Image.open(output / case["mask_file"]) as mask_image:
                    truth = np.asarray(mask_image.convert("L")) > 127
                row.update(true_positive_pixels=int((truth & predicted).sum()),
                           false_positive_pixels=int((~truth & predicted).sum()),
                           false_negative_pixels=int((truth & ~predicted).sum()))
                if not (folder / "0.png").is_file():
                    fail("evaluator_overlay_missing")
                row["overlay_file"] = (folder / "0.png").relative_to(output).as_posix()
            row["prediction_mask_file"] = mask_output.relative_to(output).as_posix()
            row["inference_timing"] = None
            if protocol["enabled"]:
                # Decode and metrics/PNG operations have already completed.
                # This fresh array is reused unchanged for every timed repeat.
                timing_image = cv2.imread(str(image_path))
                if timing_image is None or timing_image.shape[:2] != (case["height"], case["width"]):
                    fail("timing_image_decode_failed")
                row["inference_timing"] = measure_inference(runtime, timing_image, case.get("tap"), protocol)
            rows.append(row)
        scenes = sorted({r["scene_id"] for r in rows})
        conditions = sorted({c for r in rows for c in r["conditions"]})
        by_scene = {scene: summarize([r for r in rows if r["scene_id"] == scene]) for scene in scenes}
        scene_iou = [value["mean_iou"] for value in by_scene.values() if value["mean_iou"] is not None]
        aggregate = summarize(rows)
        aggregate["scene_weighted_mean_iou"] = sum(scene_iou) / len(scene_iou) if scene_iou else None
        result["models"][name] = {"weights_sha256": model["sha256"], "overall": aggregate,
                                 "thread_policy": check_runtime_threads(runtime),
                                 "class_mapping": getattr(runtime, "benchmark_class_mapping", None),
                                 "inference_timing": summarize_timing([
                                     seconds for row in rows
                                     for seconds in (row["inference_timing"] or {}).get("samples_seconds", [])
                                 ]) if protocol["enabled"] else None,
                                 "by_scene": by_scene,
                                 "by_condition": {condition: summarize([r for r in rows if condition in r["conditions"]]) for condition in conditions},
                                 "cases": rows}
    if "candidate" in result["models"]:
        before = result["models"]["baseline"]["overall"]
        after = result["models"]["candidate"]["overall"]
        result["candidate_minus_baseline"] = {key: after[key] - before[key] if after[key] is not None and before[key] is not None else None
                                               for key in ("mean_iou", "mean_dice", "misses", "false_positive_pixels", "scene_weighted_mean_iou")}
        for key in ("mean_seconds", "median_seconds"):
            baseline_time = result["models"]["baseline"]["inference_timing"]
            candidate_time = result["models"]["candidate"]["inference_timing"]
            result["candidate_minus_baseline"]["inference_" + key] = (
                candidate_time[key] - baseline_time[key] if baseline_time and candidate_time else None)
    return result


def run_worker(output, expected_frozen_sha256):
    frozen_raw = (output / "frozen.json").read_bytes()
    if digest(frozen_raw) != checked_hash(expected_frozen_sha256):
        fail("frozen_manifest_corrupt")
    frozen = parse_json(frozen_raw)
    settings = frozen["settings"]
    os.environ.update({"MODEL_DIR": str(output / "weights"), "MODEL_CACHE_DIR": str(output / "runtime-cache"),
                       "YOLO_CONFIG_DIR": str(output / "runtime-config"),
                       "YOLO_CACHE_DIR": str(output / "runtime-cache"), "YOLO_AUTOINSTALL": "false",
                       "MODEL_QUALITY_ENABLED": "0"})
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # This subprocess performs local inference only, including during dependency
    # imports. Do not allow telemetry, installers or model downloads to connect.
    import socket
    def offline(*args, **kwargs):
        raise OSError("Offline benchmark: network is disabled")
    socket.create_connection = socket.getaddrinfo = offline
    socket.socket.connect = socket.socket.connect_ex = offline
    import cv2
    import torch
    from ultralytics import settings as yolo_settings
    from ultralytics import utils as yolo_utils
    from ultralytics.utils import torch_utils
    from arborscan_v4.quality_train import segmentation_metrics
    from arborscan_v4.vision_engine import TreeVisionRuntime
    check_threads = configure_runtime_threads(settings["threads"], torch, cv2, yolo_utils, torch_utils)
    yolo_settings.update({key: False for key in ("sync", "hub", "wandb", "mlflow", "clearml", "comet", "neptune", "dvc", "raytune") if key in yolo_settings})
    def factory(weights, options):
        runtime = TreeVisionRuntime()
        runtime._select_model_path = lambda: (weights, 4)
        runtime.conf, runtime.imgsz = options["confidence"], options["imgsz"]
        runtime.load()
        runtime.benchmark_thread_policy_check = check_threads
        check_runtime_threads(runtime)
        configure_benchmark_tree_class(runtime)
        return runtime
    report = evaluate(frozen, output, factory, segmentation_metrics)
    report_raw = json_bytes(report)
    write_new(output / "report.json", report_raw)
    write_new(output / "COMPLETE", json_bytes({"report_sha256": digest(report_raw),
                                             "frozen_sha256": file_hash(output / "frozen.json")}))


def launch_worker(output, max_seconds, threads):
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = str(threads)
    with (output / "runtime.log").open("xb") as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--_worker", str(output), file_hash(output / "frozen.json")],
                                 stdout=log, stderr=log, env=environment)
        try:
            code = child.wait(timeout=max_seconds)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            fail("runtime_time_limit")
    if code != 0 or not (output / "COMPLETE").is_file():
        fail("runtime_failed_see_local_log")
    receipt = parse_json((output / "COMPLETE").read_bytes())
    if (receipt.get("report_sha256") != file_hash(output / "report.json")
            or receipt.get("frozen_sha256") != file_hash(output / "frozen.json")):
        fail("completion_checksum_mismatch")


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        fail("invalid_arguments_use_help")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 3 and argv[0] == "--_worker":
        run_worker(Path(argv[1]).resolve(), argv[2])
        return 0
    parser = SafeParser(description=__doc__, epilog=MANIFEST_CONTRACT,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="Untimed warmup calls per image/model before latency sampling (0..5)")
    parser.add_argument("--timing-repeats", type=int, default=3,
                        help="Runtime-only latency samples per image/model after warmup (1..10)")
    parser.add_argument("--check-only", action="store_true")
    output = None
    try:
        options = parser.parse_args(argv)
        if (not 1 <= options.threads <= 4 or not 1 <= options.max_seconds <= 1800
                or not 320 <= options.imgsz <= 1536 or not math.isfinite(options.confidence)
                or not .05 <= options.confidence <= .95
                or not 0 <= options.warmup_runs <= 5 or not 1 <= options.timing_repeats <= 10
                or bool(options.candidate) != bool(options.candidate_sha256)):
            fail("invalid_runtime_bounds_or_candidate_pair")
        models = {"baseline": {"path": options.baseline, "sha256": options.baseline_sha256}}
        if options.candidate:
            models["candidate"] = {"path": options.candidate, "sha256": options.candidate_sha256}
        validated = validate_inputs(options.manifest, options.manifest_sha256, models)
        settings = {"device": "cpu", "threads": options.threads, "imgsz": options.imgsz,
                    "confidence": options.confidence, "max_seconds": options.max_seconds,
                    "warmup_runs": options.warmup_runs, "timing_repeats": options.timing_repeats,
                    "preprocessing": "exif_rgb_to_bgr_v1", "selection": "existing_TreeVisionRuntime.infer"}
        output, frozen = prepare_snapshot(options.output, validated, settings)
        if options.check_only:
            summary = {"status": "checked_no_inference", "purpose": frozen["purpose"],
                       "case_count": len(frozen["cases"]), "model_count": len(models),
                       "scene_count": len({case["scene_id"] for case in frozen["cases"]}),
                       "activation_eligible": False, "independent_validation_established": False}
            write_new(output / "preflight.json", json_bytes(summary))
        else:
            launch_worker(output, options.max_seconds, options.threads)
            report = parse_json((output / "report.json").read_bytes())
            summary = {"status": "complete", "purpose": report["purpose"], "activation_eligible": False,
                       "independent_validation_established": False, "evidence_level": report["evidence_level"],
                       "models": {name: value["overall"] for name, value in report["models"].items()}}
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        code = str(error) if isinstance(error, BenchmarkError) else "benchmark_failed"
        if output is not None:
            with contextlib.suppress(OSError):
                write_new(output / "failure.json", json_bytes({"status": "failed", "error": code}))
        print(json.dumps({"status": "failed", "error": code}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
