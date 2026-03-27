import shutil
import random
from pathlib import Path

# ---------------- CONFIG ----------------
RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

SRC_DIR = DATA_DIR / "processed"
TRAIN_OUTDIR = DATA_DIR / "processed_train"
VAL_OUTDIR = DATA_DIR / "processed_val"
TEST_OUTDIR = DATA_DIR / "processed_test"
# ----------------------------------------

def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def split_and_copy_for_action(action: str):
    src_action_dir = SRC_DIR / action
    if not src_action_dir.is_dir():
        return None

    files = sorted([f for f in src_action_dir.iterdir() if f.suffix == ".npy"])
    n_total = len(files)
    
    if n_total < 10:
        print(f"⚠️ [WARNING] {action}: มีข้อมูลน้อยเกินไป ({n_total} ไฟล์) อาจทำให้โมเดลไม่เรียนรู้")
        if n_total == 0: return None

    # ล็อค Seed ให้สุ่มเหมือนเดิมทุกครั้งที่รัน (Reproducibility)
    random.seed(RANDOM_SEED)
    random.shuffle(files)

    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    # 🔥 ป้องกันกรณีข้อมูลน้อยจัด จน Val หรือ Test เป็น 0
    if n_train == 0: n_train = 1
    if n_val == 0 and n_total > 2: n_val = 1

    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    # แสดงผลให้ดูง่ายขึ้น
    print(f"[{action:<12}] Total: {n_total:<4} | Train: {len(train_files):<3} ({len(train_files)/n_total*100:.0f}%) | Val: {len(val_files):<3} | Test: {len(test_files):<3}")

    for category, file_list in [("train", train_files), ("val", val_files), ("test", test_files)]:
        dst_base = TRAIN_OUTDIR if category == "train" else VAL_OUTDIR if category == "val" else TEST_OUTDIR
        target_dir = dst_base / action
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for src_path in file_list:
            shutil.copy2(src_path, target_dir / src_path.name)
            
    return n_total

def main():
    print(f"🚀 เริ่มกระบวนการแบ่งชุดข้อมูล (Ratio: {TRAIN_RATIO*100:.0f}/{VAL_RATIO*100:.0f}/{(1-TRAIN_RATIO-VAL_RATIO)*100:.0f})")
    print(f"📂 กำลังอ่านไฟล์จาก: {SRC_DIR}\n")

    if not SRC_DIR.exists():
        print(f"❌ ไม่พบโฟลเดอร์ต้นทาง: {SRC_DIR}")
        return

    actions = sorted([d.name for d in SRC_DIR.iterdir() if d.is_dir()])
    
    if not actions:
        print("❌ ไม่พบโฟลเดอร์ Class ใน data/processed เลย!")
        return

    print("🧹 เคลียร์โฟลเดอร์เก่า...")
    ensure_clean_dir(TRAIN_OUTDIR)
    ensure_clean_dir(VAL_OUTDIR)
    ensure_clean_dir(TEST_OUTDIR)

    print("-" * 65)
    print(f"{'Class Name':<14} {'Total':<6}   {'Train':<12} {'Val':<5}   {'Test':<5}")
    print("-" * 65)
    
    total_files = 0
    for action in actions:
        count = split_and_copy_for_action(action)
        if count: total_files += count
        
    print("-" * 65)
    print(f"🎯 แบ่งไฟล์สำเร็จทั้งหมด: {total_files} คลิป")

if __name__ == "__main__":
    main()