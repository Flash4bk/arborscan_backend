# export_from_supabase.py
import os
from supabase import create_client
from tqdm import tqdm

"""
Export verified samples from Supabase Storage to local disk.

Required environment variables (recommended):
  - SUPABASE_URL
  - SUPABASE_SERVICE_KEY   (service_role key)  OR  SUPABASE_ANON_KEY / SUPABASE_KEY

Optional:
  - SUPABASE_BUCKET        (default: arborscan-verified)
  - OUT_DIR                (default: raw_data)
"""

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_KEY")
    or os.getenv("SUPABASE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
)

BUCKET = os.getenv("SUPABASE_BUCKET", "arborscan-verified")
OUT_DIR = os.getenv("OUT_DIR", "raw_data")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "Missing SUPABASE_URL or SUPABASE key. "
        "Set SUPABASE_URL and SUPABASE_SERVICE_KEY (recommended) in your environment."
    )

os.makedirs(OUT_DIR, exist_ok=True)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

folders = supabase.storage.from_(BUCKET).list()

for f in tqdm(folders, desc=f"Exporting from bucket '{BUCKET}'"):
    analysis_id = f.get("name")
    if not analysis_id:
        continue

    local = os.path.join(OUT_DIR, analysis_id)
    os.makedirs(local, exist_ok=True)

    for fname in ["input.jpg", "user_mask.png", "meta_verified.json", "meta.json", "pred.json"]:
        try:
            data = supabase.storage.from_(BUCKET).download(f"{analysis_id}/{fname}")
            with open(os.path.join(local, fname), "wb") as w:
                w.write(data)
        except Exception:
            # Some files are optional (e.g., user_mask.png)
            pass
