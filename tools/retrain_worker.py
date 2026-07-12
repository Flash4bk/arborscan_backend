# tools/retrain_worker.py
import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import random
import re
import hashlib
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any

import requests
from supabase import create_client
from ultralytics import YOLO

# -----------------------------
# Defaults / env knobs
# -----------------------------
DEFAULT_BUCKET_VERIFIED = os.getenv("SUPABASE_BUCKET_VERIFIED", "arborscan-verified")
DEFAULT_BUCKET_MODELS = os.getenv("SUPABASE_BUCKET_MODELS", "arborscan-models")
DEFAULT_BUCKET_DATASETS = os.getenv("SUPABASE_BUCKET_DATASETS", "arborscan-datasets")

DEFAULT_MIN_NEW = int(os.getenv("MIN_NEW", "0"))  # allow 0 for manual runs
DEFAULT_EPOCHS = int(os.getenv("EPOCHS", "30"))
DEFAULT_IMGSZ = int(os.getenv("IMGSZ", "1024"))
DEFAULT_BATCH = int(os.getenv("BATCH", "4"))
DEFAULT_INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "60"))

TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", "0.8"))
REPLAY_RATIO = float(os.getenv("REPLAY_RATIO", "0.2"))  # replay = ceil(new * ratio)
MAX_REPLAY = int(os.getenv("MAX_REPLAY", "200"))
MIN_MASK_AREA = float(os.getenv("MIN_MASK_AREA", "100"))
SELECTION_SEED = os.getenv("SELECTION_SEED", "")

# If true, only include samples with complete AR (points_count >= required_points)
REQUIRE_AR_COMPLETE = os.getenv("REQUIRE_AR_COMPLETE", "0") == "1"

MODEL_MIN_SIZE_BYTES = int(os.getenv("MODEL_MIN_SIZE_BYTES", "1000000"))
VERIFY_MODEL_UPLOAD = os.getenv("VERIFY_MODEL_UPLOAD", "true").lower() in {"1", "true", "yes", "on"}
EXPECTED_TREE_TASK = os.getenv("TREE_MODEL_TASK", "segment")


# -----------------------------
# Utils
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def log(msg: str) -> None:
    print(msg, flush=True)

def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def _base_url() -> str:
    return require_env("SUPABASE_URL").rstrip("/")

def _storage_headers() -> dict:
    key = require_env("SUPABASE_SERVICE_KEY")
    return {"Authorization": f"Bearer {key}", "apikey": key}

def make_supabase():
    url = require_env("SUPABASE_URL")
    key = require_env("SUPABASE_SERVICE_KEY")
    return create_client(url, key)



def ar_complete(meta: dict) -> bool:
    if not isinstance(meta, dict):
        return False
    ar = meta.get("ar") if isinstance(meta.get("ar"), dict) else None
    if not ar:
        return False
    pc = ar.get("points_count")
    rp = ar.get("required_points")
    try:
        pc_i = int(pc) if pc is not None else None
        rp_i = int(rp) if rp is not None else None
    except Exception:
        return False
    return bool(pc_i is not None and rp_i is not None and pc_i >= rp_i)
# -----------------------------
# Storage REST helpers
# -----------------------------
def storage_list_objects(bucket: str, prefix: str = "", limit: int = 1000, offset: int = 0) -> List[dict]:
    url = f"{_base_url()}/storage/v1/object/list/{bucket}"
    r = requests.post(url, headers=_storage_headers(), json={"prefix": prefix, "limit": limit, "offset": offset}, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []

def storage_download_bytes(bucket: str, path: str) -> bytes:
    url = f"{_base_url()}/storage/v1/object/authenticated/{bucket}/{path}"
    r = requests.get(url, headers=_storage_headers(), timeout=180)
    r.raise_for_status()
    return r.content

def storage_upload_bytes(bucket: str, path: str, content: bytes, content_type: str) -> None:
    url = f"{_base_url()}/storage/v1/object/{bucket}/{path}"
    headers = dict(_storage_headers())
    headers.update({"Content-Type": content_type, "x-upsert": "true"})
    r = requests.post(url, headers=headers, data=content, timeout=240)
    r.raise_for_status()

def storage_upload_json(bucket: str, path: str, data: dict) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    storage_upload_bytes(bucket, path, payload, "application/json")

# -----------------------------
# training_state helpers
# -----------------------------
def get_training_state(supabase) -> dict:
    return supabase.table("training_state").select("*").eq("id", 1).single().execute().data

def update_training_state(supabase, patch: dict) -> None:
    patch = dict(patch or {})
    if not patch:
        return

    for _ in range(10):
        try:
            supabase.table("training_state").update(patch).eq("id", 1).execute()
            return
        except Exception as e:
            msg = str(e)
            if "PGRST204" in msg and "Could not find the" in msg and "column" in msg:
                m = re.search(r"Could not find the '([^']+)' column", msg)
                if m:
                    col = m.group(1)
                    if col in patch:
                        log(f"[!] training_state: column '{col}' not found, skipping it.")
                        patch.pop(col, None)
                        if not patch:
                            return
                        continue
            raise

def try_acquire_training_lock(supabase) -> bool:
    state = get_training_state(supabase)
    if state.get("training_in_progress"):
        return False
    if not state.get("retrain_requested"):
        return False
    update_training_state(supabase, {"training_in_progress": True, "retrain_requested": False})
    return True

def safe_release_training_lock(supabase, *, success: bool, last_model_version: Optional[int] = None, extra_patch: Optional[dict] = None) -> None:
    patch = {"training_in_progress": False}
    if success:
        patch["last_trained_at"] = utc_now_iso()
        if last_model_version is not None:
            patch["last_model_version"] = int(last_model_version)
    if extra_patch:
        patch.update(extra_patch)
    update_training_state(supabase, patch)

# -----------------------------
# Verified sample discovery
# -----------------------------
def list_verified_analysis_ids(bucket_verified: str) -> List[str]:
    ids: set[str] = set()
    offset = 0
    while True:
        objs = storage_list_objects(bucket_verified, prefix="", limit=1000, offset=offset)
        if not objs:
            break
        for o in objs:
            # Supabase Storage list() may return either:
            #  - file paths like "<analysis_id>/meta_verified.json"
            #  - folder entries like "<analysis_id>" (no slash)
            name = (o.get("name") or "").strip()
            if not name:
                continue
            if name.endswith("/"):
                name = name[:-1]
            aid = name.split("/", 1)[0]
            if aid:
                ids.add(aid)
        if len(objs) < 1000:
            break
        offset += 1000
    return sorted(ids)

def read_meta_verified(bucket_verified: str, aid: str) -> Optional[dict]:
    try:
        raw = storage_download_bytes(bucket_verified, f"{aid}/meta_verified.json")
        return json.loads(raw.decode("utf-8")) if raw else None
    except Exception:
        return None

def discover_new_samples(bucket_verified: str, max_samples: Optional[int] = None) -> List[Tuple[str, dict]]:
    results: List[Tuple[str, dict]] = []
    for aid in list_verified_analysis_ids(bucket_verified):
        meta = read_meta_verified(bucket_verified, aid)
        if not meta:
            continue
        if not meta.get("has_user_mask", False):
            continue
        if meta.get("exclude_from_training", False):
            continue
        if REQUIRE_AR_COMPLETE and not ar_complete(meta):
            continue
        if meta.get("used_for_training", False):
            continue
        results.append((aid, meta))
        if max_samples is not None and len(results) >= max_samples:
            break
    return results

def discover_replay_samples(bucket_verified: str, k: int) -> List[Tuple[str, dict]]:
    if k <= 0:
        return []
    pool: List[Tuple[str, dict]] = []
    for aid in list_verified_analysis_ids(bucket_verified):
        meta = read_meta_verified(bucket_verified, aid)
        if not meta:
            continue
        if not meta.get("has_user_mask", False):
            continue
        if meta.get("exclude_from_training", False):
            continue
        if REQUIRE_AR_COMPLETE and not ar_complete(meta):
            continue
        if not meta.get("used_for_training", False):
            continue
        pool.append((aid, meta))

    if not pool:
        return []

    rng = random.Random()
    if SELECTION_SEED:
        rng.seed(SELECTION_SEED + "|replay|" + str(len(pool)))
    rng.shuffle(pool)
    return pool[: min(k, len(pool))]

def diagnose_candidates(bucket_verified: str, max_examples: int = 20) -> dict:
    counts = {
        "total_folders": 0,
        "no_meta_verified": 0,
        "no_user_mask": 0,
        "excluded": 0,
        "already_used": 0,
        "eligible_new": 0,
        "eligible_replay_pool": 0,
    }
    examples = []

    for aid in list_verified_analysis_ids(bucket_verified):
        counts["total_folders"] += 1
        meta = read_meta_verified(bucket_verified, aid)
        if not meta:
            counts["no_meta_verified"] += 1
            if len(examples) < max_examples:
                examples.append((aid, "no_meta_verified"))
            continue
        if not meta.get("has_user_mask", False):
            counts["no_user_mask"] += 1
            if len(examples) < max_examples:
                examples.append((aid, "no_user_mask"))
            continue
        if meta.get("exclude_from_training", False):
            counts["excluded"] += 1
            if len(examples) < max_examples:
                examples.append((aid, "excluded"))
            continue
        if meta.get("used_for_training", False):
            counts["already_used"] += 1
            counts["eligible_replay_pool"] += 1
            continue
        counts["eligible_new"] += 1

    return {"counts": counts, "examples": examples}

# -----------------------------
# Model paths, validation and versions
# -----------------------------
def resolve_project_layout() -> Tuple[Path, Path, Path]:
    """Return (script_dir, project_root, models_dir).

    The worker may be located either in the project root or in project/tools.
    MODEL_DIR always has priority so the API and the worker can use exactly
    the same directory.
    """
    script_dir = Path(__file__).resolve().parent

    explicit_root = os.getenv("PROJECT_ROOT")
    if explicit_root:
        project_root = Path(explicit_root).expanduser().resolve()
    elif (script_dir / "server.py").exists():
        project_root = script_dir
    elif (script_dir.parent / "server.py").exists():
        project_root = script_dir.parent
    else:
        project_root = script_dir

    explicit_models = os.getenv("MODEL_DIR")
    models_dir = (
        Path(explicit_models).expanduser().resolve()
        if explicit_models
        else (project_root / "models").resolve()
    )
    return script_dir, project_root, models_dir


def model_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_model_sha256(version: int) -> Optional[str]:
    raw = os.getenv(f"MODEL_V{int(version)}_SHA256")
    return raw.strip().lower() if raw and raw.strip() else None


def validate_model_file(
    path: Path,
    *,
    label: str,
    expected_sha256: Optional[str] = None,
    load_with_ultralytics: bool = False,
) -> Dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise RuntimeError(f"{label} is not a file: {path}")

    size = path.stat().st_size
    if size < MODEL_MIN_SIZE_BYTES:
        raise RuntimeError(
            f"{label} is too small ({size} bytes): {path}. "
            "The file may be incomplete or contain an HTTP error response."
        )

    sha256 = model_sha256(path)
    if expected_sha256 and sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, "
            f"got {sha256}"
        )

    task = None
    if load_with_ultralytics:
        model = YOLO(str(path))
        task = getattr(model, "task", None)
        if EXPECTED_TREE_TASK and task and task != EXPECTED_TREE_TASK:
            raise RuntimeError(
                f"{label} task is '{task}', expected '{EXPECTED_TREE_TASK}'"
            )
        del model

    return {
        "path": str(path),
        "size_bytes": size,
        "sha256": sha256,
        "task": task,
    }


def atomic_write_bytes(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _existing_versions_local(models_dir: Path) -> set[int]:
    versions: set[int] = set()
    if not models_dir.exists():
        return versions

    for path in models_dir.glob("model_v*.pt"):
        match = re.fullmatch(r"model_v(\d+)\.pt", path.name)
        if not match:
            continue
        version = int(match.group(1))
        try:
            validate_model_file(
                path,
                label=f"local model v{version}",
                expected_sha256=expected_model_sha256(version),
            )
            versions.add(version)
        except Exception as exc:
            log(f"[!] Ignoring invalid local model {path}: {exc}")
    return versions


def _existing_versions_bucket(bucket_models: str) -> set[int]:
    versions: set[int] = set()
    offset = 0
    while True:
        objects = storage_list_objects(
            bucket_models,
            prefix="",
            limit=1000,
            offset=offset,
        )
        if not objects:
            break
        for item in objects:
            name = (item.get("name") or "").split("/")[-1]
            match = re.fullmatch(r"model_v(\d+)\.pt", name)
            if match:
                versions.add(int(match.group(1)))
        if len(objects) < 1000:
            break
        offset += 1000
    return versions


def ensure_models_dir(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)


def get_base_model_path(models_dir: Path, base_version: int) -> Path:
    if base_version <= 0:
        return models_dir / "base.pt"
    return models_dir / f"model_v{base_version}.pt"


def _download_and_validate_model(
    *,
    bucket_models: str,
    remote_name: str,
    destination: Path,
    label: str,
    expected_sha256: Optional[str],
) -> Path:
    content = storage_download_bytes(bucket_models, remote_name)
    if len(content) < MODEL_MIN_SIZE_BYTES:
        raise RuntimeError(
            f"Downloaded {label} is too small ({len(content)} bytes): "
            f"{bucket_models}/{remote_name}"
        )

    atomic_write_bytes(destination, content)
    try:
        validate_model_file(
            destination,
            label=label,
            expected_sha256=expected_sha256,
            load_with_ultralytics=True,
        )
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    log(
        f"[✓] Downloaded and validated {label}: "
        f"{bucket_models}/{remote_name} -> {destination}"
    )
    return destination


def ensure_base_model_local(
    *,
    models_dir: Path,
    bucket_models: str,
    base_version: int,
) -> Path:
    ensure_models_dir(models_dir)
    base_path = get_base_model_path(models_dir, base_version)
    expected_hash = (
        expected_model_sha256(base_version) if base_version > 0 else None
    )

    if base_path.exists():
        try:
            validate_model_file(
                base_path,
                label=f"base model v{base_version}",
                expected_sha256=expected_hash,
                load_with_ultralytics=True,
            )
            return base_path
        except Exception as exc:
            log(f"[!] Local base model is invalid and will be replaced: {exc}")
            base_path.unlink(missing_ok=True)

    if base_version > 0:
        return _download_and_validate_model(
            bucket_models=bucket_models,
            remote_name=f"model_v{base_version}.pt",
            destination=base_path,
            label=f"base model v{base_version}",
            expected_sha256=expected_hash,
        )

    errors: List[str] = []
    for candidate in (
        "base.pt",
        "yolov8n-seg.pt",
        "yolov8s-seg.pt",
        "yolov8m-seg.pt",
    ):
        try:
            return _download_and_validate_model(
                bucket_models=bucket_models,
                remote_name=candidate,
                destination=base_path,
                label="base segmentation model",
                expected_sha256=None,
            )
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    raise RuntimeError(
        f"Base model not found or invalid: {base_path}. "
        f"Checked bucket '{bucket_models}': {' | '.join(errors)}"
    )


def compute_versions(
    models_dir: Path,
    bucket_models: str,
) -> Tuple[int, int, set[int]]:
    local_set = _existing_versions_local(models_dir)
    bucket_set = _existing_versions_bucket(bucket_models)
    existing = set(local_set) | set(bucket_set)

    base_version = max(existing) if existing else 0
    new_version = base_version + 1
    while new_version in existing:
        new_version += 1
    return base_version, new_version, existing


def diagnose_models(models_dir: Path, bucket_models: str) -> Dict[str, Any]:
    local_versions = sorted(_existing_versions_local(models_dir))
    try:
        remote_versions = sorted(_existing_versions_bucket(bucket_models))
        remote_error = None
    except Exception as exc:
        remote_versions = []
        remote_error = str(exc)

    files = []
    for version in local_versions:
        path = models_dir / f"model_v{version}.pt"
        try:
            info = validate_model_file(
                path,
                label=f"model v{version}",
                expected_sha256=expected_model_sha256(version),
            )
            info["version"] = version
            files.append(info)
        except Exception as exc:
            files.append(
                {
                    "version": version,
                    "path": str(path),
                    "error": str(exc),
                }
            )

    return {
        "models_dir": str(models_dir),
        "local_versions": local_versions,
        "remote_versions": remote_versions,
        "remote_error": remote_error,
        "files": files,
    }

# -----------------------------
# Dataset export (manifest-based)
# -----------------------------
def run_export_script(tools_dir: Path, bucket_verified: str, manifest_in_path: Path, out_dir: Path) -> None:
    script = tools_dir / "export_yolov8_dataset.py"
    if not script.exists():
        raise RuntimeError(f"export script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--bucket-verified",
        bucket_verified,
        "--manifest-in",
        str(manifest_in_path),
        "--out-dir",
        str(out_dir),
        "--min-mask-area",
        str(MIN_MASK_AREA),
    ]
    log("[*] Exporting dataset via export_yolov8_dataset.py (manifest-based) ...")
    log("[*] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(tools_dir))

def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=str(src_dir))
    final_zip = zip_path.with_suffix(".zip")
    if final_zip != zip_path:
        if zip_path.exists():
            zip_path.unlink()
        final_zip.replace(zip_path)

def upload_dataset_snapshot(*, bucket_datasets: str, dataset_version: int, zip_path: Path, manifest_path: Path, data_yaml_path: Path) -> Dict[str, str]:
    prefix = f"dataset_v{dataset_version}"
    storage_upload_bytes(bucket_datasets, f"{prefix}/dataset.zip", zip_path.read_bytes(), "application/zip")
    storage_upload_bytes(bucket_datasets, f"{prefix}/manifest.json", manifest_path.read_bytes(), "application/json")
    storage_upload_bytes(bucket_datasets, f"{prefix}/data.yaml", data_yaml_path.read_bytes(), "text/yaml")
    log(f"[✓] Uploaded dataset snapshot to {bucket_datasets}/{prefix}/")
    return {
        "dataset_zip": f"{prefix}/dataset.zip",
        "dataset_manifest": f"{prefix}/manifest.json",
        "dataset_yaml": f"{prefix}/data.yaml",
    }

# -----------------------------
# Training
# -----------------------------
def run_yolo_train(*, base_model: Path, data_yaml: Path, epochs: int, imgsz: int, batch: int, device: Optional[str], runs_segment_dir: Path, run_name: str) -> Path:
    cmd = [
        "yolo",
        "task=segment",
        "mode=train",
        f"model={str(base_model)}",
        f"data={str(data_yaml)}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"project={str(runs_segment_dir)}",
        f"name={run_name}",
        "exist_ok=True",
    ]
    if device:
        cmd.append(f"device={device}")

    log("[*] Training YOLO segmentation model ...")
    log("[*] " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    best = runs_segment_dir / run_name / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"best.pt not found at: {best}")
    return best

def save_new_model(best_pt: Path, models_dir: Path, new_version: int) -> Path:
    validate_model_file(
        best_pt,
        label="training result best.pt",
        load_with_ultralytics=True,
    )

    destination = models_dir / f"model_v{new_version}.pt"
    temporary = models_dir / f".model_v{new_version}.{uuid.uuid4().hex}.tmp"
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(best_pt, temporary)
        validate_model_file(
            temporary,
            label=f"new model v{new_version}",
            load_with_ultralytics=True,
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)

    validate_model_file(
        destination,
        label=f"saved model v{new_version}",
        load_with_ultralytics=True,
    )
    return destination


def upload_model_to_bucket(bucket_models: str, model_path: Path) -> None:
    local_info = validate_model_file(
        model_path,
        label=f"model upload {model_path.name}",
        load_with_ultralytics=True,
    )
    storage_upload_bytes(
        bucket_models,
        model_path.name,
        model_path.read_bytes(),
        "application/octet-stream",
    )

    if VERIFY_MODEL_UPLOAD:
        remote_content = storage_download_bytes(bucket_models, model_path.name)
        remote_hash = hashlib.sha256(remote_content).hexdigest()
        if len(remote_content) != local_info["size_bytes"]:
            raise RuntimeError(
                f"Uploaded model size mismatch for {model_path.name}: "
                f"local={local_info['size_bytes']}, remote={len(remote_content)}"
            )
        if remote_hash != local_info["sha256"]:
            raise RuntimeError(
                f"Uploaded model SHA-256 mismatch for {model_path.name}: "
                f"local={local_info['sha256']}, remote={remote_hash}"
            )

    log(
        f"[✓] Uploaded and verified model: "
        f"{bucket_models}/{model_path.name}"
    )

# -----------------------------
# Mark samples used
# -----------------------------
def mark_samples_used_for_training(bucket_verified: str, samples_new: List[Tuple[str, dict]], new_version: int) -> None:
    now = utc_now_iso()
    for aid, meta in samples_new:
        meta = dict(meta or {})
        meta["used_for_training"] = True
        meta["used_for_training_at"] = now
        meta["used_in_model_version"] = int(new_version)
        try:
            storage_upload_json(bucket_verified, f"{aid}/meta_verified.json", meta)
        except Exception as e:
            log(f"[!] Failed to mark used_for_training for {aid}: {e}")

# -----------------------------
# Selection + manifest
# -----------------------------
def build_selection(*, bucket_verified: str, new_samples: List[Tuple[str, dict]]) -> Dict[str, Any]:
    new_ids = [aid for aid, _ in new_samples]
    replay_k = int(min(MAX_REPLAY, max(0, (len(new_ids) * REPLAY_RATIO + 0.999999))))  # ceil
    replay_samples = discover_replay_samples(bucket_verified, replay_k)
    replay_ids = [aid for aid, _ in replay_samples]

    all_ids = new_ids + replay_ids
    if len(all_ids) < 2:
        raise RuntimeError("Not enough samples (new + replay) to train (need at least 2).")

    rng = random.Random()
    if SELECTION_SEED:
        rng.seed(SELECTION_SEED + "|split|" + str(len(all_ids)))
    ids_shuffled = all_ids[:]
    rng.shuffle(ids_shuffled)

    split_idx = max(1, int(len(ids_shuffled) * TRAIN_SPLIT))
    split_idx = min(split_idx, len(ids_shuffled) - 1)

    return {
        "new_ids": new_ids,
        "replay_ids": replay_ids,
        "train_ids": ids_shuffled[:split_idx],
        "val_ids": ids_shuffled[split_idx:],
    }

def write_manifest_in(*, path: Path, dataset_version: int, bucket_verified: str, base_model_version: int, new_model_version: int, selection: Dict[str, Any]) -> None:
    obj = {
        "dataset_version": int(dataset_version),
        "created_at_utc": utc_now_iso(),
        "bucket_verified": bucket_verified,
        "base_model_version": int(base_model_version),
        "new_model_version": int(new_model_version),
        "policy": {
            "train_split": TRAIN_SPLIT,
            "replay_ratio": REPLAY_RATIO,
            "max_replay": MAX_REPLAY,
            "min_mask_area": MIN_MASK_AREA,
            "selection_seed": SELECTION_SEED or None,
        },
        "selection": selection,
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="ArborScan retrain worker (real-last + next-free + manifest + snapshot + diag)")
    parser.add_argument("--bucket-verified", default=DEFAULT_BUCKET_VERIFIED)
    parser.add_argument("--bucket-models", default=DEFAULT_BUCKET_MODELS)
    parser.add_argument("--bucket-datasets", default=DEFAULT_BUCKET_DATASETS)
    parser.add_argument("--min-new", type=int, default=DEFAULT_MIN_NEW)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SEC)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--diagnose-models", action="store_true")
    args = parser.parse_args()

    tools_dir, project_root, models_dir = resolve_project_layout()
    runs_segment_dir = Path(
        os.getenv("TRAIN_RUNS_DIR", str(project_root / "runs" / "segment"))
    ).expanduser().resolve()
    dataset_dir = Path(
        os.getenv("TRAIN_DATASET_DIR", str(project_root / "dataset_yolov8"))
    ).expanduser().resolve()
    manifest_in_path = Path(
        os.getenv("TRAIN_MANIFEST_PATH", str(project_root / "manifest_in.json"))
    ).expanduser().resolve()

    ensure_models_dir(models_dir)
    runs_segment_dir.mkdir(parents=True, exist_ok=True)

    log(f"[*] Script directory: {tools_dir}")
    log(f"[*] Project root: {project_root}")
    log(f"[*] Shared model directory: {models_dir}")

    if args.diagnose_models:
        print(
            json.dumps(
                diagnose_models(models_dir, args.bucket_models),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    supabase = make_supabase()

    while True:
        state = get_training_state(supabase)

        if state.get("training_in_progress"):
            log("[*] training_in_progress = TRUE, waiting ...")
            if args.once:
                sys.exit(0)
            time.sleep(args.interval)
            continue

        if not state.get("retrain_requested"):
            log("[*] retrain_requested = FALSE, waiting ...")
            if args.once:
                sys.exit(0)
            time.sleep(args.interval)
            continue

        new_samples = discover_new_samples(args.bucket_verified, max_samples=args.max_samples)

        if args.min_new > 0 and len(new_samples) < args.min_new:
            log(f"[*] Not enough new samples: {len(new_samples)} < {args.min_new}. Resetting retrain_requested to FALSE.")
            update_training_state(supabase, {"retrain_requested": False})
            if args.once:
                sys.exit(0)
            time.sleep(args.interval)
            continue

        if not try_acquire_training_lock(supabase):
            log("[*] Could not acquire training lock, waiting ...")
            if args.once:
                sys.exit(0)
            time.sleep(args.interval)
            continue

        log(f"[*] Acquired training lock. New samples to train on: {len(new_samples)}")
        diag = diagnose_candidates(args.bucket_verified, max_examples=25)
        log(f"[*] Candidate diagnostics: {diag['counts']}")
        if diag.get("examples"):
            log(f"[*] Example skipped: {diag['examples'][:10]}")

        success = False

        try:
            base_version, new_version, existing = compute_versions(models_dir, args.bucket_models)
            log(f"[*] Existing versions: {sorted(existing) if existing else []}")
            log(f"[*] Base model version (real) = v{base_version}; new version will be v{new_version}")

            base_model = ensure_base_model_local(models_dir=models_dir, bucket_models=args.bucket_models, base_version=base_version)

            selection = build_selection(bucket_verified=args.bucket_verified, new_samples=new_samples)
            write_manifest_in(
                path=manifest_in_path,
                dataset_version=new_version,
                bucket_verified=args.bucket_verified,
                base_model_version=base_version,
                new_model_version=new_version,
                selection=selection,
            )
            log(f"[*] Wrote manifest_in: {manifest_in_path}")

            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            run_export_script(tools_dir=tools_dir, bucket_verified=args.bucket_verified, manifest_in_path=manifest_in_path, out_dir=dataset_dir)

            zip_path = tools_dir / "dataset.zip"
            zip_dir(dataset_dir, zip_path)
            log(f"[✓] Zipped dataset snapshot: {zip_path}")

            out_manifest = dataset_dir / "manifest.json"
            out_yaml = dataset_dir / "data.yaml"
            if not out_manifest.exists() or not out_yaml.exists():
                raise RuntimeError("export_yolov8_dataset.py did not produce manifest.json/data.yaml")

            snapshot_paths = upload_dataset_snapshot(
                bucket_datasets=args.bucket_datasets,
                dataset_version=new_version,
                zip_path=zip_path,
                manifest_path=out_manifest,
                data_yaml_path=out_yaml,
            )

            run_name = f"train_v{new_version}"
            best_pt = run_yolo_train(
                base_model=base_model,
                data_yaml=out_yaml,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                runs_segment_dir=runs_segment_dir,
                run_name=run_name,
            )

            new_model_path = save_new_model(best_pt, models_dir, new_version)
            log(f"[✓] Saved new model: {new_model_path}")
            upload_model_to_bucket(args.bucket_models, new_model_path)

            mark_samples_used_for_training(args.bucket_verified, new_samples, new_version)

            max_after = max(existing | {new_version}) if existing else new_version

            safe_release_training_lock(
                supabase,
                success=True,
                last_model_version=max_after,
                extra_patch={
                    "last_dataset_manifest": snapshot_paths.get("dataset_manifest"),
                    "last_dataset_zip": snapshot_paths.get("dataset_zip"),
                    "last_dataset_version": int(new_version),
                    "last_trained_version": int(new_version),
                },
            )

            success = True
            log(f"[✓] Training completed. trained_version=v{new_version} last_model_version=v{max_after}")

        except Exception as e:
            msg = str(e)
            log(f"[!] Training failed: {msg}")

            # don't spam-crash when it's just lack of data
            if "Not enough samples" in msg:
                try:
                    safe_release_training_lock(supabase, success=False)
                except Exception as e2:
                    log(f"[!] Failed to release training lock: {e2}")
                if args.once:
                    sys.exit(1)
                time.sleep(args.interval)
                continue

            try:
                safe_release_training_lock(supabase, success=False)
            except Exception as e2:
                log(f"[!] Failed to release training lock: {e2}")

        if args.once:
            sys.exit(0 if success else 1)

        time.sleep(args.interval)

if __name__ == "__main__":
    main()
