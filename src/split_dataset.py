import shutil
import random
from pathlib import Path

# ---------------- CONFIG ----------------
RANDOM_SEED = 42

# สัดส่วนการแบ่งข้อมูล (Train 80%, Validation 10%, Test ที่เหลือคือ 10%)
# Train: ให้ AI เรียนรู้
# Val: ให้ AI สอบย่อยระหว่างเรียน (ใช้ทำ EarlyStopping ป้องกัน Overfitting)
# Test: ให้ AI สอบปลายภาค (วัดผลความแม่นยำสุดท้ายแบบที่ AI ไม่เคยเห็นข้อมูลนี้มาก่อน)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

# การจัดการ Path ด้วย pathlib (เป็นวิธีที่ Modern และปลอดภัยกว่า os.path)
# .resolve() หา Path เต็มแบบ Absolute
# .parents[2] ถอยหลังกลับไป 2 ขั้น เพื่อชี้ไปที่โฟลเดอร์ Root ของโปรเจกต์
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# กำหนดโฟลเดอร์ต้นทาง (ข้อมูลดิบที่สกัดจุดมาแล้ว) และปลายทาง (Train/Val/Test)
SRC_DIR = DATA_DIR / "processed"
TRAIN_OUTDIR = DATA_DIR / "processed_train"
VAL_OUTDIR = DATA_DIR / "processed_val"
TEST_OUTDIR = DATA_DIR / "processed_test"
# ----------------------------------------

def ensure_clean_dir(path: Path):
    """
    ฟังก์ชันสำหรับทำความสะอาดโฟลเดอร์ปลายทาง
    หากมีโฟลเดอร์เก่าอยู่ ให้ลบทิ้งทั้งยวง (rmtree) แล้วสร้างใหม่ (mkdir)
    ป้องกันปัญหา "ข้อมูลเก่าตกค้าง" เวลาเรารันสคริปต์นี้ซ้ำหลายๆ รอบ
    """
    if path.exists():
        shutil.rmtree(path)
    # parents=True: สร้างโฟลเดอร์แม่ที่ขาดหายไปให้ด้วย
    # exist_ok=True: ถ้าโฟลเดอร์มีอยู่แล้วก็ไม่ต้องแจ้ง Error
    path.mkdir(parents=True, exist_ok=True)

def split_and_copy_for_action(action: str):
    """
    ฟังก์ชันหลักในการแบ่งข้อมูลของแต่ละ 1 ท่าทาง (Action)
    """
    src_action_dir = SRC_DIR / action
    # ถ้าไม่ใช่โฟลเดอร์ (เช่น อาจจะเป็นไฟล์เผลอหลงเข้ามา) ให้ข้ามไป
    if not src_action_dir.is_dir():
        return None

    # ดึงรายชื่อไฟล์ทั้งหมดที่ลงท้ายด้วย .npy ในโฟลเดอร์ท่าทางนี้ แล้วเรียงลำดับชื่อ
    files = sorted([f for f in src_action_dir.iterdir() if f.suffix == ".npy"])
    n_total = len(files)
    
    # ⚠️ [เซฟตี้ด่านที่ 1] แจ้งเตือนถ้าข้อมูลน้อยเกินไป โมเดล Deep Learning มักต้องการขั้นต่ำหลักร้อยคลิปต่อท่า
    if n_total < 10:
        print(f"⚠️ [WARNING] {action}: มีข้อมูลน้อยเกินไป ({n_total} ไฟล์) อาจทำให้โมเดลไม่เรียนรู้")
        if n_total == 0: return None

    # 🔥 ล็อค Seed ให้การสุ่มเหมือนเดิมทุกครั้งที่รัน (Reproducibility)
    # สำคัญมาก! สมมติเราต้องรันสคริปต์นี้ใหม่ ไฟล์ Test จะต้องเป็นชุดเดิมเสมอ 
    # ไม่งั้นโมเดลอาจจะเผลอเอาไฟล์ Test เก่าไปเทรน ทำให้ผลประเมินโกง (Data Leakage)
    random.seed(RANDOM_SEED)
    random.shuffle(files) # สับไพ่ (สลับลำดับไฟล์) เพื่อกระจายความหลากหลาย

    # คำนวณจำนวนไฟล์ที่จะตกไปอยู่แต่ละถัง (Train / Val / Test)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    # 🔥 [เซฟตี้ด่านที่ 2] ป้องกันกรณีข้อมูลน้อยจัดๆ จนคำนวณสัดส่วนแล้วกลายเป็น 0
    # บังคับให้ Train อย่างน้อยมี 1 ไฟล์ และ Val มี 1 ไฟล์
    if n_train == 0: n_train = 1
    if n_val == 0 and n_total > 2: n_val = 1

    # ตัดแบ่ง List ของไฟล์ออกเป็น 3 ก้อน (Array Slicing)
    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    # แสดงผลให้ดูง่ายขึ้นบนหน้าจอ (จัด Format ด้วย :<12 คือชิดซ้าย 12 ตัวอักษร)
    print(f"[{action:<12}] Total: {n_total:<4} | Train: {len(train_files):<3} ({len(train_files)/n_total*100:.0f}%) | Val: {len(val_files):<3} | Test: {len(test_files):<3}")

    # วนลูปก๊อปปี้ไฟล์จาก List ทั้ง 3 ก้อน ไปยังโฟลเดอร์ปลายทาง
    for category, file_list in [("train", train_files), ("val", val_files), ("test", test_files)]:
        # เลือกโฟลเดอร์ปลายทางตาม category
        dst_base = TRAIN_OUTDIR if category == "train" else VAL_OUTDIR if category == "val" else TEST_OUTDIR
        target_dir = dst_base / action
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for src_path in file_list:
            # ใช้ shutil.copy2 เพื่อก๊อปปี้ไฟล์ โดยจะพยายามรักษา Metadata (เช่น วันที่สร้างไฟล์) ไว้ด้วย
            shutil.copy2(src_path, target_dir / src_path.name)
            
    return n_total

def main():
    """
    ฟังก์ชันจุดตั้งต้นของโปรแกรม จัดการภาพรวมทั้งหมด
    """
    print(f"🚀 เริ่มกระบวนการแบ่งชุดข้อมูล (Ratio: {TRAIN_RATIO*100:.0f}/{VAL_RATIO*100:.0f}/{(1-TRAIN_RATIO-VAL_RATIO)*100:.0f})")
    print(f"📂 กำลังอ่านไฟล์จาก: {SRC_DIR}\n")

    # เช็คว่าโฟลเดอร์ต้นทางมีอยู่จริงไหม ถ้าไม่มียกเลิกการทำงานทันที
    if not SRC_DIR.exists():
        print(f"❌ ไม่พบโฟลเดอร์ต้นทาง: {SRC_DIR}")
        return

    # ดึงชื่อโฟลเดอร์ทั้งหมดใน SRC_DIR (แต่ละโฟลเดอร์คือ 1 Class ท่าทาง) และจัดเรียงตัวอักษร A-Z
    actions = sorted([d.name for d in SRC_DIR.iterdir() if d.is_dir()])
    
    if not actions:
        print("❌ ไม่พบโฟลเดอร์ Class ใน data/processed เลย!")
        return

    # ล้างไพ่ เคลียร์ข้อมูลเก่าทิ้งให้หมดก่อนเริ่มกระบวนการแบ่งใหม่
    print("🧹 เคลียร์โฟลเดอร์เก่า...")
    ensure_clean_dir(TRAIN_OUTDIR)
    ensure_clean_dir(VAL_OUTDIR)
    ensure_clean_dir(TEST_OUTDIR)

    # พิมพ์ Header ของตารางแสดงผล
    print("-" * 65)
    print(f"{'Class Name':<14} {'Total':<6}   {'Train':<12} {'Val':<5}   {'Test':<5}")
    print("-" * 65)
    
    total_files = 0
    # วนลูปจัดการแบ่งไฟล์ทีละท่าทาง
    for action in actions:
        count = split_and_copy_for_action(action)
        if count: total_files += count # เก็บยอดรวมไฟล์ทั้งหมดที่ถูกประมวลผล
        
    print("-" * 65)
    print(f"🎯 แบ่งไฟล์สำเร็จทั้งหมด: {total_files} คลิป")

if __name__ == "__main__":
    main()