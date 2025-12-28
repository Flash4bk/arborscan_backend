import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

from supabase import create_client


# -----------------------------
# Defaults (можно переопределять аргументами CLI)
# -----------------------------
DEFAULT_BUCKET_VERIFIED = "arborscan-verified"
DEFAULT_MIN_NEW = 10
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


def storage_upload_json(supabase, bucket: str, path: str, data: dict) -> None:
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    supabase.storage.from_(bucket).upload(
        path,
        payload,
        {"content-type": "application/json"},
        upsert=True,
    )


def storage_download_bytes(supabase, bucket: str, path: str) -> bytes:
    return supabase.storage.from_(bucket).download(path)


def storage_list(supabase, bucket: str, prefix: str = "") -> List[dict]:
    """
    Supabase storage list() обычно листит "папку" (prefix).
    Для нашего случая: list("") вернёт верхний уровень (analysis_id папки).
    """
    return supabase.storage.from_(bucket).list(prefix)


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
) -> None:
    patch = {
        "training_in_progress": False,
    }
    if success:
        patch["last_trained_at"] = utc_now_iso()
        if last_model_version is not None:
            patch["last_model_version"] = last_model_version
    update_training_state(supabase, patch)


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
    """
    results: List[Tuple[str, dict]] = []

    # Верхний уровень — "папки" analysis_id
    top = storage_list(supabase, bucket, "")
    analysis_ids = []
    for obj in top:
        name = obj.get("name", "")
        # Supabase Storage list("") может вернуть файлы и "папки";
        # у тебя структура: <analysis_id>/...
        # Обычно "папки" возвращаются как dict с name = "<analysis_id>"
        if name and "/" not in name:
            analysis_ids.append(name)

    for aid in analysis_ids:
        try:
            meta_bytes = storage_download_bytes(
                supabase, bucket, f"{aid}/meta_verified.json"
            )
            meta = json.loads(meta_bytes)
        except Exception:
            continue

        if not meta.get("has_user_mask", False):
            continue
        if meta.get("used_for_training", False):
            continue

        results.append((aid, meta))
        if max_samples is not None and len(results) >= max_samples:
            break

    return results


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


def run_export_script(tools_dir: Path) -> None:
    """
    Запускает tools/export_yolov8_dataset.py в текущем окружении.
    """
    script = tools_dir / "export_yolov8_dataset.py"
    if not script.exists():
        raise RuntimeError(f"export script not found: {script}")

    log("[*] Exporting dataset via export_yolov8_dataset.py ...")
    subprocess.run([sys.executable, str(script)], check=True)


def find_latest_train_dir(runs_segment_dir: Path, name: str) -> Path:
    """
    При использовании Ultralytics с project=runs/segment и name=<name>,
    итог будет в runs/segment/<name>/weights/best.pt
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
    supabase,
    bucket: str,
    samples: List[Tuple[str, dict]],
    new_version: int,
) -> None:
    """
    Обновляет meta_verified.json в Storage:
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
            storage_upload_json(supabase, bucket, f"{aid}/meta_verified.json", meta)
        except Exception as e:
            # Не валим всю тренировку из-за одного мета-файла
            log(f"[!] Failed to mark used_for_training for {aid}: {e}")


def try_insert_model_version_row(supabase, new_version: int, model_path: str) -> None:
    """
    У тебя в Supabase уже есть таблица model_versions.
    Вставка опциональная (если структура отличается — не ломаем процесс).
    """
    try:
        supabase.table("model_versions").insert(
            {
                "version": new_version,
                "model_path": model_path,
                "created_at": utc_now_iso(),
            }
        ).execute()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="ArborScan retrain worker")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET_VERIFIED)
    parser.add_argument("--min-new", type=int, default=DEFAULT_MIN_NEW)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu (optional)")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SEC)
    parser.add_argument("--once", action="store_true", help="run once then exit")
    parser.add_argument("--max-samples", type=int, default=None, help="limit number of samples per training run")

    args = parser.parse_args()

    # директории относительно tools/
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

        # Проверим количество доступных масок (неиспользованных)
        samples = discover_verified_samples(
            supabase,
            bucket=args.bucket,
            max_samples=args.max_samples,
        )
        if len(samples) < args.min_new:
            log(f"[*] Not enough new samples: {len(samples)} < {args.min_new}. Resetting retrain_requested to FALSE.")
            # Снимаем флаг, чтобы не крутиться постоянно (можешь убрать, если хочешь держать флаг)
            update_training_state(supabase, {"retrain_requested": False})
            if args.once:
                sys.exit(0)
            time.sleep(args.interval)
            continue

        # Захватываем "лок" (best-effort)
        if not try_acquire_training_lock(supabase):
            log("[*] Could not acquire training lock (someone else?). Waiting ...")
            if args.once:
                sys.exit(0)
            time.sleep(args.interval)
            continue

        log(f"[*] Acquired training lock. New samples to train on: {len(samples)}")

        success = False
        new_version = None

        try:
            # 1) Экспорт датасета
            run_export_script(tools_dir)

            # 2) Определяем базовую модель
            state = get_training_state(supabase)  # обновим состояние после lock
            last_version = int(state.get("last_model_version") or 0)

            base_model = get_base_model_path(models_dir, last_version)
            new_version = last_version + 1

            # 3) Обучаем (дообучение от base_model)
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

            # 4) Сохраняем новую модель
            new_model_path = save_new_model(best_pt, models_dir, new_version)
            log(f"[✓] Saved new model: {new_model_path}")

            # 5) Помечаем примеры как использованные для обучения
            mark_samples_used_for_training(supabase, args.bucket, samples, new_version)

            # 6) Обновляем training_state
            safe_release_training_lock(supabase, success=True, last_model_version=new_version)

            # 7) Опционально — model_versions
            try_insert_model_version_row(supabase, new_version, str(new_model_path))

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
