import os
import time
import subprocess
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

CHECK_INTERVAL = 60  # секунд
MODELS_DIR = "models"

def get_state():
    return supabase.table("training_state").select("*").eq("id", 1).single().execute().data

def update_state(data):
    supabase.table("training_state").update(data).eq("id", 1).execute()

def main():
    while True:
        state = get_state()

        if state["training_in_progress"]:
            time.sleep(CHECK_INTERVAL)
            continue

        if not state["retrain_requested"]:
            time.sleep(CHECK_INTERVAL)
            continue

        print("[*] Starting retraining...")

        update_state({
            "training_in_progress": True,
            "retrain_requested": False
        })

        # 1. экспорт датасета
        subprocess.run(["python", "export_yolov8_dataset.py"], check=True)

        # 2. определяем модель
        last_version = state["last_model_version"]
        if last_version == 0:
            base_model = "models/base.pt"
        else:
            base_model = f"models/model_v{last_version}.pt"

        new_version = last_version + 1
        new_model = f"models/model_v{new_version}.pt"

        # 3. обучение
        subprocess.run([
            "yolo", "task=segment", "mode=train",
            f"model={base_model}",
            "data=dataset_yolov8/data.yaml",
            "epochs=30",
            "imgsz=1024",
            "batch=4"
        ], check=True)

        # 4. сохраняем модель
        os.rename(
            "runs/segment/train/weights/best.pt",
            new_model
        )

        # 5. отмечаем обучение завершённым
        update_state({
            "training_in_progress": False,
            "last_model_version": new_version,
            "last_trained_at": "now()"
        })

        print(f"[✓] Training completed: {new_model}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
