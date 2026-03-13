import os
import sys
import time
import json
import pickle
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd

from config import APPS, DATE_START, DATE_END, DATA_RAW

_output_path = DATA_RAW / "reviews_raw.csv"
PAGES_PER_CHUNK = 20
PYTHON = sys.executable
WORKER_SCRIPT = Path(__file__).parent / "_scrape_chunk.py"

# Targets to Resume/Patch
TARGETS = [
    ("Kredivo", "com.finaccel.android", 3),
    ("Kredivo", "com.finaccel.android", 4),
    ("Kredivo", "com.finaccel.android", 5),
    ("Akulaku", "io.silvrr.installment", 1),
    ("Akulaku", "io.silvrr.installment", 2),
    ("Akulaku", "io.silvrr.installment", 3),
    ("Akulaku", "io.silvrr.installment", 4),
    ("Akulaku", "io.silvrr.installment", 5),
]

def load_existing_db():
    print(f"Loading existing dataset to prevent duplicates from {_output_path}...")
    df = pd.read_csv(_output_path)
    existing = set()
    for _, row in df.iterrows():
        key = f"{row['app_name']}_{row['rating']}_{row['review_date']}_{row['review_text_raw']}"
        existing.add(key)
    print(f"Loaded {len(existing)} unique existing reviews into memory.")
    return existing

def scrape_target(app_name, app_id, star, existing_set):
    print(f"\n{'='*70}")
    print(f"[RESUME] {app_name} — {star}-Star")
    print(f"{'='*70}")
    
    date_start = datetime.strptime(DATE_START, "%Y-%m-%d")
    date_end = datetime.strptime(DATE_END, "%Y-%m-%d")
    
    token_file = DATA_RAW / f"_token_{app_name}_{star}.pkl"
    rows_file = DATA_RAW / f"_rows_{app_name}_{star}.json"
    
    chunk_num = 0
    added_this_session = 0
    consecutive_fails = 0
    
    while True:
        chunk_num += 1
        
        if rows_file.exists():
            rows_file.unlink()
            
        try:
            proc = subprocess.run(
                [PYTHON, str(WORKER_SCRIPT), app_id, str(PAGES_PER_CHUNK),
                 str(token_file), str(rows_file), app_name, str(star)],
                timeout=300, # 5 minute robust timeout
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                print(f"      [WARN] Worker error: {proc.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"      [WARN] Chunk {chunk_num} timed out (300s). Retrying...")
            consecutive_fails += 1
            if consecutive_fails >= 10:
                print(f"      [ERROR] 10 consecutive timeouts. Giving up on {app_name} Star {star} for now.")
                break
            time.sleep(15)
            continue
            
        if not rows_file.exists():
            consecutive_fails += 1
            if consecutive_fails >= 10:
                print(f"      [ERROR] 10 consecutive missing files. Giving up.")
                break
            time.sleep(5)
            continue
            
        consecutive_fails = 0  # Reset on success
        
        try:
            with open(rows_file, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except Exception as e:
            print(f"      [WARN] Failed to read JSON chunk: {e}")
            continue
            
        new_rows_to_save = []
        too_old = 0
        
        for r in rows:
            rd = datetime.strptime(r["review_date"], "%Y-%m-%d")
            if rd < date_start:
                too_old += 1
            elif rd <= date_end:
                # Deduplication Check
                key = f"{r['app_name']}_{r['rating']}_{r['review_date']}_{r['review_text_raw']}"
                if key not in existing_set:
                    existing_set.add(key)
                    new_rows_to_save.append(r)
                    
        # Append immediately
        if new_rows_to_save:
            df_new = pd.DataFrame(new_rows_to_save)
            df_new.to_csv(_output_path, mode='a', index=False, header=False, encoding="utf-8-sig")
            added_this_session += len(new_rows_to_save)
            
        print(f"      Chunk {chunk_num} | Fetched: {len(rows)} | Added New: {len(new_rows_to_save)} | Too Old: {too_old} | Subtotal Added Today: {added_this_session}")
        
        if not rows:
            print(f"      API returned 0 rows. Exhausted.")
            break
            
        # Give up if 80% of chunk is older than date_start
        if too_old > len(rows) * 0.8:
            print(f"      Majority is too old (< {DATE_START}). Done with {app_name} Star {star}.")
            break
            
        time.sleep(2)

def main():
    if not os.path.exists(_output_path):
        print(f"Raw dataset {_output_path} not found! Cannot resume.")
        return
        
    existing_set = load_existing_db()
    
    for app_name, app_id, star in TARGETS:
        scrape_target(app_name, app_id, star, existing_set)
        
    print("\n[RESUME COMPLETE] Patching missing files done!")

if __name__ == "__main__":
    main()
