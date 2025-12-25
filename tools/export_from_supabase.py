# export_from_supabase.py
import os
from supabase import create_client
from tqdm import tqdm

SUPABASE_URL = "https://mfjxhtxwaablygwdjxhx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1manhodHh3YWFibHlnd2RqeGh4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NjM0MzI5MSwiZXhwIjoyMDgxOTE5MjkxfQ.-B5Vo-5ML2o0NerGEu_1l-7-iIe_K-_xre1-9pOWZ_E"
BUCKET = "arborscan-verified"

OUT_DIR = "raw_data"
os.makedirs(OUT_DIR, exist_ok=True)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

folders = supabase.storage.from_(BUCKET).list()

for f in tqdm(folders):
    analysis_id = f["name"]
    local = os.path.join(OUT_DIR, analysis_id)
    os.makedirs(local, exist_ok=True)

    for fname in ["input.jpg", "user_mask.png"]:
        try:
            data = supabase.storage.from_(BUCKET).download(f"{analysis_id}/{fname}")
            with open(os.path.join(local, fname), "wb") as w:
                w.write(data)
        except Exception:
            pass  # если нет маски — пропускаем
