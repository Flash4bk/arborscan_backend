import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any

from supabase import create_client
import requests
import re
import random

# -----------------------------
# Defaults (можно переопределять аргументами CLI)
# -----------------------------
DEFAULT_BUCKET_VERIFIED = "arborscan-verified"
DEFAULT_BUCKET_MODELS = "arborscan-models"
DEFAULT_BUCKET_DATASETS = "arborscan-datasets"

# ВАЖНО: для обучения "из приложения" по кнопке — по умолчанию запускаем даже на малом датасете
DEFAULT_MIN_NEW = 0

DEFAULT_EPOCHS = 30
DEFAULT_IMGSZ = 1024
DEFAULT_BATCH = 4
DEFAULT_INTERVAL_SEC = 60

# Replay policy (anti-forgetting)
DEFAULT_REPLAY_RATIO = float(os.getenv("REPLAY_RATIO", "0.2"))  # % from old set relative to new
DEFAULT_MAX_REPLAY = int(os.getenv("MAX_REPLAY", "200"))
DEFAULT_TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", "0.8"))
DEFAULT_MIN_MASK_AREA = float(os.getenv("MIN_MASK_AREA", "100"))

# Deterministic selection (optional): set SELECTION_SEED for reproducibility across runs
DEFAULT_SELECTION_SEED = os.getenv("SELECTION_SEED", "")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log(msg: str) -> None:
    print(msg, flush=True)


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def make_supabase():
    url = require_env("SUPABASE_URL")
    key = require_env("SUPABASE_SERVICE_KEY")
    return create_client(url, key)


def _base_url() -> str:
    return require_env("SUPABASE_URL").rstrip("/")


def _storage_headers() -> dict:
    key = require_env("SUPABASE_SERVICE_KEY")
    return {
        "Authorization": f"Bearer {key}",
        "apikey": key,
    }


def storage_list_objects(bucket: str, prefix: str = "") -> List[dict]:
    """List objects in a Storage bucket using the REST API."""
    url = f"{_base_url()}/storage/v1/object/list/{bucket}"
    r = requests.post(
        url,
        headers=_storage_headers(),
        json={"prefix": prefix, "limit": 1000, "offset": 0},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def storage_download_bytes(bucket: str, path: str) -> bytes:
    """Download object bytes from a (usually private) bucket."""
    url = f"{_base_url()}/storage/v1/object/authenticated/{bucket}/{path}"
    r = requests.get(url, headers=_storage_headers(), timeout=60)
    r.raise_for_status()
    return r.content


def storage_upload_bytes(bucket: str, path: str, content: bytes, content_type: str) -> None:
    """Upload object bytes with upsert via REST (works across supabase-py versions)."""
    url = f"{_base_url()}/storage/v1/object/{bucket}/{path}"
    headers = dict(_storage_headers())
    headers.update({"Content-Type": content_type, "x-upsert": "true"})
    r = requests.post(url, headers=headers, data=content, timeout=120)
    r.raise_for_status()


def storage_upload_json(bucket: str, path: str, data: dict) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    storage_upload_bytes(bucket, path, payload, "application/json")


def get_training_state(supabase) -> dict:
    return (
        supabase.table("training_state")
        .select("*")
        .eq("id", 1)
        .single()
        .execute()
        .data
    )


def update_training_state(supabase, patch: dict) -> None:
    supabase.table("training_state").update(patch).eq("id", 1).execute()


def try_acquire_training_lock(supabase) -> bool:
    """
    Пытаемся "захватить" обучение:
    - training_in_progress = True
    - retrain_requested = False
    Если кто-то уже поставил training_in_progress=True, воркер должен выйти/ждать.
    """
    state = get_training_state(supabase)
    if state.get("training_in_progress"):
        return False
    if not state.get("retrain_requested"):
        return False

    # best-effort lock
    update_training_state(
        supabase,
        {
            "training_in_progress": True,
            "retrain_requested": False,
        },
    )
    return True


def safe_release_training_lock(
    supabase,
    *,
    success: bool,
    last_model_version: Optional[int] = None,
    extra: Optional[dict] = None,
) -> None:
    patch = {
        "training_in_progress": False,
    }
    if success:
        patch["last_trained_at"] = utc_now_iso()
        if last_model_version is not None:
            patch["last_model_version"] = last_model_version
    if extra:
        patch.update(extra)
    update_training_state(supabase, patch)


def _unique_analysis_ids_from_objects(objs: List[dict]) -> List[str]:
    """
    Supabase Storage list returns objects with names like:
      <analysis_id>/input.jpg
      <analysis_id>/meta_verified.json
    We infer analysis_id by splitting at '/'.
    """
    ids = set()
    for o in objs:
        name = (o.get("name") or "").strip()
        if not name:
            continue
        aid = name.split("/", 1)[0]
        if aid:
            ids.add(aid)
    return sorted(ids)


def load_meta_verified_or_meta(bucket: str, aid: str) -> Optional[dict]:
    """
    Prefer meta_verified.json, fallback to meta.json.
    This makes the worker compatible with older items / admin toggles that might write meta.json only.
    """
    for fname in ("meta_verified.json", "meta.json"):
        try:
            raw = storage_download_bytes(bucket, f"{aid}/{fname}")
            if not raw:
                continue
            meta = json.loads(raw.decode("utf-8"))
            if isinstance(meta, dict):
                return meta
        except Exception:
            continue
    return None


def discover_new_samples(
    bucket: str,
    max_samples: Optional[int] = None,
) -> List[Tuple[str, dict]]:
    """
    Возвращает список (analysis_id, meta) для НОВЫХ примеров,
    где:
      - has_user_mask == True
      - used_for_training == False (или отсутствует)
      - exclude_from_training != True
    """
    results: List[Tuple[str, dict]] = []
    objs = storage_list_objects(bucket, "")
    analysis_ids = _unique_analysis_ids_from_objects(objs)

    for aid in analysis_ids:
        meta = load_meta_verified_or_meta(bucket, aid)
        if not meta:
            continue
        if not meta.get("has_user_mask", False):
            continue
        if meta.get("used_for_training", False):
            continue
        if meta.get("exclude_from_training", False):
            continue

        results.append((aid, meta))
        if max_samples is not None and len(results) >= max_samples:
            break

    return results


def discover_old_samples_for_replay(bucket: str) -> List[str]:
    """
    Возвращает analysis_id старых примеров, которые можно брать в replay:
      - has_user_mask == True
      - used_for_training == True
      - exclude_from_training != True
    """
    objs = storage_list_objects(bucket, "")
    analysis_ids = _unique_analysis_ids_from_objects(objs)

    replay: List[str] = []
    for aid in analysis_ids:
        meta = load_meta_verified_or_meta(bucket, aid)
        if not meta:
            continue
        if not meta.get("has_user_mask", False):
            continue
        if not meta.get("used_for_training", False):
            continue
        if meta.get("exclude_from_training", False):
            continue
        replay.append(aid)
    return replay


def ensure_models_dir(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)


def get_base_model_path(models_dir: Path, last_version: int) -> Path:
    """
    Если last_version == 0 -> models/base.pt
    иначе -> models/model_v{last_version}.pt
    """
    if last_version == 0:
        return models_dir / "base.pt"
    return models_dir / f"model_v{last_version}.pt"


def ensure_base_model_local(bucket_models: str, models_dir: Path, last_version: int) -> Path:
    """Ensure the base model file exists locally; download from models bucket if needed.

    Expected files in bucket:
      - model_v{N}.pt at bucket root (recommended)
      - optionally under 'tree/' prefix (legacy)
    For version 0 we expect:
      - base.pt in repo OR in bucket (base.pt or yolov8n-seg.pt)
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    if last_version == 0:
        base_path = models_dir / "base.pt"
        if base_path.exists():
            return base_path

        # Try download base.pt or yolov8n-seg.pt from bucket
        candidates = ["base.pt", "yolov8n-seg.pt", "yolov8n-seg.pt.pt"]
        for c in candidates:
            try:
                data = storage_download_bytes(bucket_models, c)
                base_path.write_bytes(data)
                log(f"[✓] Downloaded base model from bucket '{bucket_models}': {c} -> {base_path}")
                return base_path
            except Exception:
                continue

        raise RuntimeError(
            f"Base model not found locally ({base_path}) and not found in models bucket '{bucket_models}'. "
            "Upload Ultralytics yolov8n-seg.pt as 'base.pt' (recommended) or 'yolov8n-seg.pt' to the models bucket, "
            "or bake it into the image at models/base.pt."
        )

    # Versioned model
    path = models_dir / f"model_v{last_version}.pt"
    if path.exists():
        return path

    candidates = [
        f"model_v{last_version}.pt",
        f"tree/model_v{last_version}.pt",
        f"tree/model_v{last_version}.pt".replace("model_", "model-"),  # just in case
    ]
    last_err = None
    for c in candidates:
        try:
            data = storage_download_bytes(bucket_models, c)
            path.write_bytes(data)
            log(f"[✓] Downloaded base model from bucket '{bucket_models}': {c} -> {path}")
            return path
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Base model not found: {path}. Tried downloading from bucket '{bucket_models}' with candidates: {candidates}. "
        f"Last error: {last_err}"
    )


def _max_model_version_local(models_dir: Path) -> int:
    mx = 0
    if not models_dir.exists():
        return 0
    for p in models_dir.glob("model_v*.pt"):
        m = re.match(r"model_v(\d+)\.pt$", p.name)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx


def _max_model_version_bucket(bucket_models: str) -> int:
    mx = 0
    try:
        objs = storage_list_objects(bucket_models, prefix="")
        for o in objs:
            name = o.get("name") or ""
            base = name.split("/")[-1]
            m = re.match(r"model_v(\d+)\.pt$", base)
            if m:
                mx = max(mx, int(m.group(1)))
    except Exception as e:
        log(f"[!] Could not list models bucket for version detection: {e}")
    return mx


def upload_model_to_bucket(bucket_models: str, model_path: Path) -> None:
    dst = model_path.name
    content = model_path.read_bytes()
    storage_upload_bytes(bucket_models, dst, content, "application/octet-stream")
    log(f"[✓] Uploaded model to bucket: {bucket_models}/{dst}")


def write_manifest_in(
    *,
    out_path: Path,
    bucket_verified: str,
    base_model_version: int,
    new_model_version: int,
    new_ids: List[str],
    replay_ids: List[str],
    train_ids: List[str],
    val_ids: List[str],
    train_split: float,
    replay_ratio: float,
    min_mask_area: float,
    selection_seed: Optional[str],
) -> None:
    manifest: Dict[str, Any] = {
        "dataset_version": new_model_version,
        "created_at_utc": utc_now_iso(),
        "bucket_verified": bucket_verified,
        "base_model_version": base_model_version,
        "new_model_version": new_model_version,
        "policy": {
            "train_split": float(train_split),
            "replay_ratio": float(replay_ratio),
            "min_mask_area": float(min_mask_area),
            "selection_seed": selection_seed or None,
        },
        "selection": {
            "new_ids": list(new_ids),
            "replay_ids": list(replay_ids),
            "train_ids": list(train_ids),
            "val_ids": list(val_ids),
            "dropped_ids": [],
        },
    }
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_export_script(tools_dir: Path, *, bucket_verified: str, manifest_in: Path, out_dir: Path) -> None:
    """
    Запускает tools/export_yolov8_dataset.py с явным manifest_in.
    """
    script = tools_dir / "export_yolov8_dataset.py"
    if not script.exists():
        raise RuntimeError(f"export script not found: {script}")

    log("[*] Exporting dataset via export_yolov8_dataset.py (manifest-based) ...")
    cmd = [
        sys.executable,
        str(script),
        "--bucket-verified",
        bucket_verified,
        "--manifest-in",
        str(manifest_in),
        "--out-dir",
        str(out_dir),
        "--min-mask-area",
        str(DEFAULT_MIN_MASK_AREA),
    ]
    log("[*] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(tools_dir))


def find_latest_train_dir(runs_segment_dir: Path, name: str) -> Path:
    """
    Итог будет в runs/segment/<name>/weights/best.pt
    """
    out_dir = runs_segment_dir / name
    if not out_dir.exists():
        raise RuntimeError(f"Train output dir not found: {out_dir}")
    return out_dir


def run_yolo_train(
    *,
    base_model: Path,
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: Optional[str],
    runs_segment_dir: Path,
    run_name: str,
) -> Path:
    """
    Запускает обучение и возвращает путь к best.pt
    """
    if not base_model.exists():
        raise RuntimeError(
            f"Base model not found: {base_model}. "
            f"Put yolov8n-seg.pt there as models/base.pt or ensure last model exists."
        )
    if not data_yaml.exists():
        raise RuntimeError(f"data.yaml not found: {data_yaml}")

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

    train_dir = find_latest_train_dir(runs_segment_dir, run_name)
    best = train_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"best.pt not found at: {best}")
    return best


def save_new_model(best_pt: Path, models_dir: Path, new_version: int) -> Path:
    dst = models_dir / f"model_v{new_version}.pt"
    tmp = models_dir / f".model_v{new_version}.pt.tmp"
    shutil.copy2(best_pt, tmp)
    tmp.replace(dst)
    return dst


def mark_samples_used_for_training(
    bucket: str,
    samples: List[Tuple[str, dict]],
    new_version: int,
) -> None:
    """
    Обновляет meta_verified.json (или meta.json если другого нет) в Storage:
      used_for_training: true
      used_for_training_at: <utc iso>
      used_in_model_version: new_version
    """
    now = utc_now_iso()
    for aid, meta in samples:
        meta["used_for_training"] = True
        meta["used_for_training_at"] = now
        meta["used_in_model_version"] = new_version
        try:
            # Prefer updating meta_verified.json if it exists; else meta.json
            # We will try meta_verified.json upload unconditionally (safe).
            storage_upload_json(bucket, f"{aid}/meta_verified.json", meta)
        except Exception as e:
            log(f"[!] Failed to write meta_verified.json for {aid}: {e}")
            try:
                storage_upload_json(bucket, f"{aid}/meta.json", meta)
            except Exception as e2:
                log(f"[!] Failed to write meta.json for {aid}: {e2}")


def try_insert_model_version_row(supabase, new_version: int, model_path: str, dataset_path: str, manifest_path: str) -> None:
    """
    Опциональная запись в таблицу model_versions (если есть соответствующие поля).
    """
    try:
        supabase.table("model_versions").insert(
            {
                "version": new_version,
                "model_path": model_path,
                "dataset_path": dataset_path,
                "manifest_path": manifest_path,
                "created_at": utc_now_iso(),
            }
        ).execute()
    except Exception:
        pass


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_dir():
                continue
            z.write(p, arcname=str(p.relative_to(src_dir)))


def main():
    parser = argparse.ArgumentParser(description="ArborScan retrain worker (manifest + dataset snapshot)")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET_VERIFIED)
    parser.add_argument("--bucket-models", default=os.getenv("SUPABASE_BUCKET_MODELS", DEFAULT_BUCKET_MODELS))
    parser.add_argument("--bucket-datasets", default=os.getenv("SUPABASE_BUCKET_DATASETS", DEFAULT_BUCKET_DATASETS))

    parser.add_argument("--min-new", type=int, default=DEFAULT_MIN_NEW)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu (optional)")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SEC)
    parser.add_argument("--once", action="store_true", help="run once then exit")
    parser.add_argument("--max-samples", type=int, default=None, help="limit number of NEW samples per training run")

    # selection controls
    parser.add_argument("--replay-ratio", type=float, default=DEFAULT_REPLAY_RATIO)
    parser.add_argument("--max-replay", type=int, default=DEFAULT_MAX_REPLAY)
    parser.add_argument("--train-split", type=float, default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--selection-seed", default=DEFAULT_SELECTION_SEED)

    args = parser.parse_args()

    # директории относительно tools/
    tools_dir = Path(__file__).resolve().parent
    project_root = tools_dir.parent
    models_dir = project_root / "models"
    runs_segment_dir = tools_dir / "runs" / "segment"
    dataset_dir = tools_dir / "dataset_yolov8"
    data_yaml = dataset_dir / "data.yaml"
    manifest_in_path = tools_dir / "manifest_in.json"  # will be overwritten for each run
    dataset_zip_path = tools_dir / "dataset.zip"

    ensure_models_dir(models_dir)
    runs_segment_dir.mkdir(parents=True, exist_ok=True)

    supabase = make_supabase()

    while True:
        try:
            state = get_training_state(supabase)
        except Exception as e:
            log(f"[!] Cannot read training_state: {e}")
            if args.once:
                sys.exit(2)
            time.sleep(args.interval)
            continue

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

        # NEW samples
        new_samples = discover_new_samples(
            bucket=args.bucket,
            max_samples=args.max_samples,
        )

        if args.min_new > 0 and len(new_samples) < args.min_new:
            log(f"[*] Not enough new samples: {len(new_samples)} < {args.min_new}. Resetting retrain_requested to FALSE.")
            update_training_state(supabase, {"retrain_requested": False})
            if args.once:
                sys.exit(0)
            time.sleep(args.interval)
            continue

        # Lock
        if not try_acquire_training_lock(supabase):
            log("[*] Could not acquire training lock (someone else?). Waiting ...")
            if args.once:
                sys.exit(0)
            time.sleep(args.interval)
            continue

        log(f"[*] Acquired training lock. New samples to train on: {len(new_samples)}")

        success = False
        new_version: Optional[int] = None

        try:
            # Determine base model version and NEW version (robust)
            state = get_training_state(supabase)
            state_last = int(state.get("last_model_version") or 0)
            local_last = _max_model_version_local(models_dir)
            bucket_last = _max_model_version_bucket(args.bucket_models)

            last_version = max(state_last, local_last, bucket_last)
            base_model = ensure_base_model_local(args.bucket_models, models_dir, last_version)
            new_version = last_version + 1

            # Selection seed
            if args.selection_seed:
                rnd = random.Random(str(args.selection_seed) + f":v{new_version}")
            else:
                rnd = random.Random()

            new_ids = [aid for aid, _ in new_samples]

            # Replay selection
            replay_pool = discover_old_samples_for_replay(args.bucket)
            # Don't replay the ones that are also in new_ids (paranoia)
            replay_pool = [aid for aid in replay_pool if aid not in set(new_ids)]

            desired_replay = int(round(len(new_ids) * float(args.replay_ratio)))
            desired_replay = max(0, min(desired_replay, int(args.max_replay), len(replay_pool)))
            replay_ids = rnd.sample(replay_pool, k=desired_replay) if desired_replay > 0 else []

            # Merge and split deterministically
            all_ids = list(new_ids) + list(replay_ids)
            rnd.shuffle(all_ids)

            if len(all_ids) < 2:
                raise RuntimeError("Not enough samples (new + replay) to train (need at least 2).")

            split_idx = max(1, int(len(all_ids) * float(args.train_split)))
            split_idx = min(split_idx, len(all_ids) - 1)  # ensure at least 1 val if possible
            train_ids = all_ids[:split_idx]
            val_ids = all_ids[split_idx:]

            # Write manifest_in (selection is now fixed & reproducible)
            write_manifest_in(
                out_path=manifest_in_path,
                bucket_verified=args.bucket,
                base_model_version=last_version,
                new_model_version=new_version,
                new_ids=new_ids,
                replay_ids=replay_ids,
                train_ids=train_ids,
                val_ids=val_ids,
                train_split=float(args.train_split),
                replay_ratio=float(args.replay_ratio),
                min_mask_area=float(DEFAULT_MIN_MASK_AREA),
                selection_seed=args.selection_seed or None,
            )
            log(f"[*] Wrote manifest_in: {manifest_in_path}")

            # 1) Export dataset strictly from manifest_in
            run_export_script(
                tools_dir,
                bucket_verified=args.bucket,
                manifest_in=manifest_in_path,
                out_dir=dataset_dir,
            )

            # 1.1) Read manifest_out produced by exporter (contains dropped_ids)
            manifest_out_path = dataset_dir / "manifest.json"
            if not manifest_out_path.exists():
                raise RuntimeError(f"Exporter did not produce manifest.json at {manifest_out_path}")
            manifest_out = json.loads(manifest_out_path.read_text(encoding="utf-8"))
            kept_train = manifest_out.get("selection", {}).get("train_ids", [])
            kept_val = manifest_out.get("selection", {}).get("val_ids", [])
            if len(kept_train) + len(kept_val) < 2:
                raise RuntimeError("After dropping invalid masks, not enough samples remain to train (need at least 2).")

            # 2) ZIP snapshot of the exported dataset (reproducible)
            zip_dir(dataset_dir, dataset_zip_path)
            log(f"[✓] Zipped dataset snapshot: {dataset_zip_path}")

            # 2.1) Upload dataset snapshot to datasets bucket
            dataset_prefix = f"dataset_v{new_version}"
            storage_upload_bytes(args.bucket_datasets, f"{dataset_prefix}/dataset.zip", dataset_zip_path.read_bytes(), "application/zip")
            storage_upload_bytes(args.bucket_datasets, f"{dataset_prefix}/manifest.json", manifest_out_path.read_bytes(), "application/json")
            storage_upload_bytes(args.bucket_datasets, f"{dataset_prefix}/data.yaml", data_yaml.read_bytes(), "text/yaml")
            log(f"[✓] Uploaded dataset snapshot to {args.bucket_datasets}/{dataset_prefix}/")

            # 3) Train (fine-tune from base_model)
            run_name = f"train_v{new_version}"
            best_pt = run_yolo_train(
                base_model=base_model,
                data_yaml=data_yaml,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                runs_segment_dir=runs_segment_dir,
                run_name=run_name,
            )

            # 4) Save model locally
            new_model_path = save_new_model(best_pt, models_dir, new_version)
            log(f"[✓] Saved new model: {new_model_path}")

            # 4.1) Upload model to models bucket
            try:
                upload_model_to_bucket(args.bucket_models, new_model_path)
            except Exception as e:
                log(f"[!] Failed to upload model to bucket: {e}")

            # 5) Mark NEW samples used_for_training (ONLY new_samples, not replay)
            mark_samples_used_for_training(args.bucket, new_samples, new_version)

            # 6) Update training_state + include dataset snapshot pointers (optional fields)
            dataset_zip_remote = f"{args.bucket_datasets}/{dataset_prefix}/dataset.zip"
            manifest_remote = f"{args.bucket_datasets}/{dataset_prefix}/manifest.json"
            safe_release_training_lock(
                supabase,
                success=True,
                last_model_version=new_version,
                extra={
                    "last_dataset_version": new_version,
                    "last_dataset_zip": dataset_zip_remote,
                    "last_dataset_manifest": manifest_remote,
                },
            )

            # 7) Optional model_versions row
            try_insert_model_version_row(
                supabase,
                new_version,
                model_path=f"{args.bucket_models}/{new_model_path.name}",
                dataset_path=dataset_zip_remote,
                manifest_path=manifest_remote,
            )

            success = True
            log(f"[✓] Training completed. last_model_version = {new_version}")

        except Exception as e:
            log(f"[!] Training failed: {e}")
            try:
                safe_release_training_lock(supabase, success=False)
            except Exception as e2:
                log(f"[!] Failed to release training lock: {e2}")

        if args.once:
            sys.exit(0 if success else 1)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
