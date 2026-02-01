import os
import sys
import json
import time
import re
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Optional

from supabase import create_client


# -----------------------------
# Defaults (можно переопределять аргументами CLI / ENV)
# -----------------------------
DEFAULT_BUCKET_VERIFIED = os.getenv("SUPABASE_BUCKET_VERIFIED", "arborscan-verified")
DEFAULT_BUCKET_MODELS = os.getenv("SUPABASE_BUCKET_MODELS", "arborscan-models")

# ВАЖНО: для обучения "из приложения" по кнопке — по умолчанию запускаем даже на малом датасете
DEFAULT_MIN_NEW = 0

DEFAULT_EPOCHS = 30
DEFAULT_IMGSZ = 1024
DEFAULT_BATCH = 4
DEFAULT_INTERVAL_SEC = 60


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


# -----------------------------
# Supabase Storage helpers (без upsert=... в upload)
# -----------------------------
def storage_download_bytes(supabase, bucket: str, path: str) -> Optional[bytes]:
    try:
        res = supabase.storage.from_(bucket).download(path)
        if isinstance(res, (bytes, bytearray)):
            return bytes(res)
        if hasattr(res, "data") and isinstance(res.data, (bytes, bytearray)):
            return bytes(res.data)
        return None
    except Exception:
        return None


def storage_list(supabase, bucket: str, prefix: str = "") -> List[dict]:
    try:
        return supabase.storage.from_(bucket).list(prefix) or []
    except Exception:
        return []


def storage_remove_quiet(supabase, bucket: str, path: str) -> None:
    try:
        # remove ожидает список путей
        supabase.storage.from_(bucket).remove([path])
    except Exception:
        pass


def storage_upload_replace(
    supabase,
    bucket: str,
    path: str,
    content: bytes,
    content_type: str,
) -> None:
    """
    Надёжная замена файла без upsert:
      1) remove (если есть)
      2) upload (без upsert)
    """
    storage_remove_quiet(supabase, bucket, path)
    supabase.storage.from_(bucket).upload(
        path,
        content,
        file_options={"content-type": content_type},
    )


def storage_upload_json_replace(supabase, bucket: str, path: str, data: dict) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    storage_upload_replace(supabase, bucket, path, payload, "application/json")


# -----------------------------
# Supabase DB helpers
# -----------------------------
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
    """
    state = get_training_state(supabase)
    if state.get("training_in_progress"):
        return False
    if not state.get("retrain_requested"):
        return False

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
) -> None:
    patch = {"training_in_progress": False}
    if success:
        patch["last_trained_at"] = utc_now_iso()
        if last_model_version is not None:
            patch["last_model_version"] = last_model_version
    update_training_state(supabase, patch)


# -----------------------------
# Model versioning (по bucket моделей, а не по training_state)
# -----------------------------
def get_max_model_version_in_bucket(supabase, bucket_models: str) -> int:
    """
    Ищем файлы model_vN.pt в корне bucket_models и возвращаем max(N).
    Если нет — 0.
    """
    items = storage_list(supabase, bucket_models, "")
    max_v = 0
    for it in items:
        name = (it.get("name") or "").strip()
        m = re.match(r"^model_v(\d+)\.pt$", name)
        if m:
            max_v = max(max_v, int(m.group(1)))
    return max_v


def ensure_local_model_from_bucket(
    supabase,
    bucket_models: str,
    remote_name: str,
    local_path: Path,
) -> None:
    """
    Скачиваем модель из bucket, если локально её нет.
    """
    if local_path.exists():
        return
    data = storage_download_bytes(supabase, bucket_models, remote_name)
    if not data:
        raise RuntimeError(f"Model not found in bucket: {bucket_models}/{remote_name}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)


def upload_model_to_bucket(
    supabase,
    bucket_models: str,
    local_model_path: Path,
    remote_name: str,
) -> None:
    content = local_model_path.read_bytes()
    storage_upload_replace(
        supabase,
        bucket_models,
        remote_name,
        content,
        "application/octet-stream",
    )


# -----------------------------
# Dataset discovery
# -----------------------------
def discover_verified_samples(
    supabase,
    bucket: str,
    max_samples: Optional[int] = None,
) -> List[Tuple[str, dict]]:
    """
    Возвращает список (analysis_id, meta_verified.json dict) для примеров,
    где:
      - has_user_mask == True
      - used_for_training == False (или отсутствует)
      - exclude_from_training != True
    """
    results: List[Tuple[str, dict]] = []

    top = storage_list(supabase, bucket, "")
    analysis_ids = []
    for obj in top:
        name = obj.get("name", "")
        if name and "/" not in name:
            analysis_ids.append(name)

    for aid in analysis_ids:
        meta_bytes = storage_download_bytes(supabase, bucket, f"{aid}/meta_verified.json")
        if not meta_bytes:
            continue

        try:
            meta = json.loads(meta_bytes.decode("utf-8"))
        except Exception:
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


# -----------------------------
# Train helpers
# -----------------------------
def ensure_models_dir(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)


def run_export_script(tools_dir: Path) -> None:
    script = tools_dir / "export_yolov8_dataset.py"
    if not script.exists():
        raise RuntimeError(f"export script not found: {script}")

    log("[*] Exporting dataset via export_yolov8_dataset.py ...")
    subprocess.run([sys.executable, str(script)], check=True, cwd=str(tools_dir))


def find_train_dir(runs_segment_dir: Path, name: str) -> Path:
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
    if not base_model.exists():
        raise RuntimeError(f"Base model not found: {base_model}")
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

    train_dir = find_train_dir(runs_segment_dir, run_name)
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
    supabase,
    bucket_verified: str,
    samples: List[Tuple[str, dict]],
    new_version: int,
) -> None:
    """
    Обновляет meta_verified.json в Storage:
      used_for_training: true
      used_for_training_at: <utc iso>
      used_in_model_version: new_version
    Делает replace без upsert=...
    """
    now = utc_now_iso()
    for aid, meta in samples:
        meta["used_for_training"] = True
        meta["used_for_training_at"] = now
        meta["used_in_model_version"] = new_version
        try:
            storage_upload_json_replace(supabase, bucket_verified, f"{aid}/meta_verified.json", meta)
        except Exception as e:
            log(f"[!] Failed to mark used_for_training for {aid}: {e}")


def try_insert_model_version_row(supabase, new_version: int, model_storage_path: str) -> None:
    """
    Опционально: таблица model_versions (если есть).
    """
    try:
        supabase.table("model_versions").insert(
            {
                "version": new_version,
                "model_path": model_storage_path,
                "created_at": utc_now_iso(),
            }
        ).execute()
    except Exception:
        pass


# -----------------------------
# Main loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="ArborScan retrain worker")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET_VERIFIED)
    parser.add_argument("--models-bucket", default=DEFAULT_BUCKET_MODELS)
    parser.add_argument("--min-new", type=int, default=DEFAULT_MIN_NEW)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu (optional)")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SEC)
    parser.add_argument("--once", action="store_true", help="run once then exit")
    parser.add_argument("--max-samples", type=int, default=None, help="limit number of samples per training run")
    parser.add_argument("--base-model", default="model_v1.pt", help="fallback base model in models bucket if no model exists")
    args = parser.parse_args()

    tools_dir = Path(__file__).resolve().parent
    project_root = tools_dir.parent

    models_dir = project_root / "models"
    runs_segment_dir = tools_dir / "runs" / "segment"
    dataset_dir = tools_dir / "dataset_yolov8"
    data_yaml = dataset_dir / "data.yaml"

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

        # Собираем список новых примеров
        samples = discover_verified_samples(supabase, bucket=args.bucket, max_samples=args.max_samples)

        if args.min_new > 0 and len(samples) < args.min_new:
            log(f"[*] Not enough new samples: {len(samples)} < {args.min_new}. Reset retrain_requested=FALSE.")
            update_training_state(supabase, {"retrain_requested": False})
            if args.once:
                sys.exit(0)
            time.sleep(args.interval)
            continue

        # Захватываем "лок"
        if not try_acquire_training_lock(supabase):
            log("[*] Could not acquire training lock (someone else?). Waiting ...")
            if args.once:
                sys.exit(0)
            time.sleep(args.interval)
            continue

        log(f"[*] Acquired training lock. New samples to train on: {len(samples)}")

        success = False
        new_version: Optional[int] = None

        try:
            # 1) Экспорт датасета
            run_export_script(tools_dir)

            # 2) Определяем версию по bucket моделей
            max_v = get_max_model_version_in_bucket(supabase, args.models_bucket)
            new_version = max_v + 1

            # 3) Определяем базовую модель: если есть max_v -> model_v{max_v}.pt иначе args.base_model
            if max_v > 0:
                base_remote = f"model_v{max_v}.pt"
                base_local = models_dir / base_remote
            else:
                base_remote = args.base_model
                base_local = models_dir / base_remote

            # скачиваем базовую модель, если её нет локально
            ensure_local_model_from_bucket(supabase, args.models_bucket, base_remote, base_local)

            # 4) Обучаем
            run_name = f"train_v{new_version}"
            best_pt = run_yolo_train(
                base_model=base_local,
                data_yaml=data_yaml,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                runs_segment_dir=runs_segment_dir,
                run_name=run_name,
            )

            # 5) Сохраняем новую модель локально
            new_model_path = save_new_model(best_pt, models_dir, new_version)
            log(f"[✓] Saved new model: {new_model_path}")

            # 6) Загружаем новую модель в bucket моделей (иначе web-сервис её не увидит)
            remote_new = f"model_v{new_version}.pt"
            upload_model_to_bucket(supabase, args.models_bucket, new_model_path, remote_new)
            log(f"[✓] Uploaded model to bucket: {args.models_bucket}/{remote_new}")

            # 7) Помечаем примеры как использованные (без upsert)
            mark_samples_used_for_training(supabase, args.bucket, samples, new_version)

            # 8) Обновляем training_state
            safe_release_training_lock(supabase, success=True, last_model_version=new_version)

            # 9) Опционально — model_versions
            try_insert_model_version_row(supabase, new_version, f"{args.models_bucket}/{remote_new}")

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
