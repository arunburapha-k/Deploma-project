import os
import numpy as np
import random
from pathlib import Path
from scipy.interpolate import interp1d

# =====================================================================
# --- 1. จัดการ Path ใหม่ (ถอย 2 ขั้น: src -> storage -> Project Root) ---
# =====================================================================
# Path(__file__).resolve() จะหาที่อยู่ไฟล์ปัจจุบันแบบเต็ม 
# .parents[2] คือการถอยหลังกลับไป 2 โฟลเดอร์ เพื่อให้ชี้ไปที่โฟลเดอร์หลักของโปรเจกต์ (Project Root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# ชี้เป้าไปที่โฟลเดอร์ Train เท่านั้น เพราะเราจะไม่ทำ Augment ข้อมูล Val/Test เด็ดขาด
TRAIN_DIR = str(DATA_DIR / "processed_train")


# ================== CONFIG ทั่วไป ==================
RANDOM_SEED = 42

SEQ_LEN = 30 # จำนวนเฟรมต่อ 1 คลิป (Time steps)
FEAT_DIM = 258 # จำนวนฟีเจอร์ทั้งหมดต่อ 1 เฟรม

# โครงสร้างของ MediaPipe (Pose = ร่างกาย, Hand = มือซ้าย/ขวา)
POSE_LM = 33
POSE_DIM = 4    # มี 4 แกน: x, y, z และ visibility (ความน่าจะเป็นที่กล้องมองเห็นจุดนั้น)
HAND_LM = 21
HAND_DIM = 3    # มี 3 แกน: x, y, z (มือไม่มี visibility)

# คำนวณหาขนาดของแต่ละส่วนเพื่อใช้ในการแบ่ง Index Array
POSE_SIZE = POSE_LM * POSE_DIM  # 33 * 4 = 132
LH_START = POSE_SIZE  # จุดเริ่มต้นของมือซ้าย คือ Index ที่ 132
LH_SIZE = HAND_LM * HAND_DIM  # 21 * 3 = 63
RH_START = LH_START + LH_SIZE  # จุดเริ่มต้นมือขวา คือ Index ที่ 132 + 63 = 195
RH_SIZE = HAND_LM * HAND_DIM  # 21 * 3 = 63
FEATURE_TOTAL = POSE_SIZE + LH_SIZE + RH_SIZE

# เช็คความถูกต้อง (Sanity check) ป้องกันการตั้งค่าขนาดผิด
assert FEATURE_TOTAL == FEAT_DIM, f"FEAT_DIM ต้องเท่ากับ {FEATURE_TOTAL} (Pose132+LH63+RH63)"

# ==========================================================
# Config (Safe Mode: ป้องกันพิกัดระเบิดและรักษารูปทรงกายวิภาค)
# ==========================================================

# 1. Spatial Perturbations (การรบกวนเชิงพื้นที่ - ตำแหน่งของร่างกาย)
NOISE_STD = 0.01          # ค่าเบี่ยงเบนมาตรฐาน (Standard Deviation) สำหรับจำลอง Noise กล้อง
TRANSLATE_STD = 0.05      # ระยะการขยับตัว (Shift) ซ้าย/ขวา/บน/ล่าง
SCALE_RANGE = (0.80, 1.20)# สัดส่วนการขยาย/ย่อตัวละคร (80% ถึง 120%)
JOINT_DROP_PROB = 0.02    # โอกาสที่ 1 ข้อต่อจะหายไป (เซ็ตเป็น 0)

# 2. Temporal Perturbations (การรบกวนเชิงเวลา - ความเร็วและจังหวะ)
MAX_SHIFT_FRAMES = 3      # จำนวนเฟรมสูงสุดที่จะเลื่อนคลิปไปข้างหน้าหรือถอยหลัง
TIME_WARP_RANGE = (0.80, 1.20) # สัดส่วนความเร็วคลิป (0.8=เล่นเร็วขึ้น, 1.2=เล่นช้าลง)
PARTIAL_KEEP_RANGE = (0.85, 0.95) # อัตราส่วนการเก็บเนื้อหาคลิปไว้ (ตัดหัวท้ายทิ้ง)

# 3. Real-world Camera Simulations (จำลองสภาพกล้องมือถือในโลกจริง)
PREFIX_MAX_FRAMES = 2     # จำนวนเฟรมขยะ (no_action) ที่จะสุ่มแปะหัวคลิป
SUFFIX_MAX_FRAMES = 2     # จำนวนเฟรมขยะ (no_action) ที่จะสุ่มแปะท้ายคลิป
LOW_FPS_DROP_RATE = 0.3   # โอกาสที่เฟรมปัจจุบันจะเหมือนเฟรมที่แล้ว (ภาพกระตุก)
NO_ACTION_CLASS_NAME = "no_action" # ชื่อคลาสที่เก็บภาพคนยืนเฉยๆ ไม่ทำอะไร

MAX_YAW_DEG = 15.0        # มุมหมุนกล้องรอบแกน Y (ซ้าย-ขวา) สูงสุด
MAX_MASK_FRAMES = 5       # จำนวนเฟรมสูงสุดที่กล้องจะค้าง (Freeze)
TRACKING_LOSS_PROB = 0.05 # โอกาสที่ MediaPipe จะทำมือหายทั้งข้าง

# เซ็ตค่า Random Seed ให้เหมือนเดิมทุกครั้งที่รัน เพื่อให้ผลลัพธ์สามารถทำซ้ำได้ (Reproducibility)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------- index helper (Updated for 258) -----------------
# np.arange(start, stop, step) เพื่อดึงตำแหน่ง Index ของแกนต่างๆ 
# เช่น POSE_X_IDX จะดึง index ที่ 0, 4, 8, 12... (ข้ามไปทีละ 4 เพราะมี x, y, z, v)
POSE_X_IDX = np.arange(0, POSE_SIZE, 4)
POSE_Y_IDX = np.arange(1, POSE_SIZE, 4)
POSE_Z_IDX = np.arange(2, POSE_SIZE, 4)
POSE_VIS_IDX = np.arange(3, POSE_SIZE, 4)

LH_X_IDX = LH_START + np.arange(0, LH_SIZE, 3) # ของมือข้ามทีละ 3 เพราะมีแค่ x, y, z
LH_Y_IDX = LH_START + np.arange(1, LH_SIZE, 3)
LH_Z_IDX = LH_START + np.arange(2, LH_SIZE, 3)

RH_X_IDX = RH_START + np.arange(0, RH_SIZE, 3)
RH_Y_IDX = RH_START + np.arange(1, RH_SIZE, 3)
RH_Z_IDX = RH_START + np.arange(2, RH_SIZE, 3)

# รวม Index ของแกน X และ Z ทั้งหมดเข้าด้วยกัน (ใช้สำหรับคำนวณ Camera Yaw 3D Rotation)
ALL_X_IDX = np.concatenate([POSE_X_IDX, LH_X_IDX, RH_X_IDX])
ALL_Z_IDX = np.concatenate([POSE_Z_IDX, LH_Z_IDX, RH_Z_IDX])


# ================== UTILITIES ==================

def is_augmented_filename(fname: str) -> bool:
    """ฟังก์ชันเช็คว่าไฟล์นี้ถูกทำ Augment ไปแล้วหรือยัง (ป้องกันการทำซ้ำจนไฟล์ล้น)"""
    name, ext = os.path.splitext(fname)
    if ext != ".npy": return True # ถ้าไม่ใช่ไฟล์ numpy ให้มองข้ามไป
    # รวม Suffix (คำลงท้าย) ที่เราสร้างขึ้นมาทั้งหมด
    suffixes = ["_flip", "_noise1", "_tshift", "_drop", "_st", "_tw", "_ps", "_psna", "_lowfps", "_yaw", "_mask", "_tkloss"]
    # เช็คว่าชื่อไฟล์มีคำลงท้ายเหล่านี้หรือไม่
    return any(name.endswith(suf) for suf in suffixes)

def load_no_action_pool(train_dir: str, class_name: str = "no_action"):
    """ฟังก์ชันโหลดข้อมูลท่าทางคนยืนนิ่งๆ (No Action) เก็บไว้ใน RAM เพื่อเอาไว้ใช้แปะหัว-ท้ายคลิป"""
    pool = []
    na_dir = os.path.join(train_dir, class_name)
    if not os.path.isdir(na_dir):
        return pool
    # ดึงไฟล์ที่ยังไม่ถูก Augment มาเป็น Pool
    files = sorted([f for f in os.listdir(na_dir) if f.endswith(".npy") and not is_augmented_filename(f)])
    for fname in files:
        path = os.path.join(na_dir, fname)
        try:
            seq = np.load(path)
        except: continue
        # เช็คให้ชัวร์ว่าขนาดไฟล์ถูกต้อง [30, 258]
        if seq.ndim == 2 and seq.shape[0] == SEQ_LEN and seq.shape[1] == FEAT_DIM:
            pool.append(seq)
    return pool

# ================== Augmentation Functions ==================

def add_gaussian_noise(seq: np.ndarray, std: float = NOISE_STD) -> np.ndarray:
    """เติม Noise รบกวนเข้าไปในแกน x, y, z จำลองอาการกล้องสั่น"""
    noisy_seq = seq.copy()
    # สุ่มค่า Noise จากการแจกแจงแบบปกติ (Gaussian Normal Distribution)
    noise = np.random.normal(loc=0.0, scale=std, size=seq.shape)
    
    # รวม Index ของ x, y, z ทั้งหมด (ยกเว้นค่า Visibility ที่เราจะไม่กวนมัน)
    xyz_idx = np.concatenate([
        POSE_X_IDX, POSE_Y_IDX, POSE_Z_IDX, 
        LH_X_IDX, LH_Y_IDX, LH_Z_IDX, 
        RH_X_IDX, RH_Y_IDX, RH_Z_IDX
    ])
    
    # บวกค่า Noise เข้าไปในแต่ละเฟรม
    for t in range(noisy_seq.shape[0]):
        noisy_seq[t, xyz_idx] += noise[t, xyz_idx]
    return noisy_seq

def temporal_shift(seq: np.ndarray, max_shift: int = MAX_SHIFT_FRAMES) -> np.ndarray:
    """เลื่อนคลิปไปข้างหน้าหรือถอยหลัง จำลองกรณีคนเริ่มทำท่าไม่พร้อมกัน"""
    T = seq.shape[0]
    if T <= 1 or max_shift <= 0: return seq.copy()
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0: return seq.copy()
    
    shifted = np.empty_like(seq)
    if shift > 0: # เลื่อนไปข้างหน้า
        shifted[shift:] = seq[:-shift]
        shifted[:shift] = seq[0] # ก๊อปปี้เฟรมแรกมาถมช่องว่างด้านหน้า
    else:         # ถอยหลัง
        k = -shift
        shifted[:-k] = seq[k:]
        shifted[-k:] = seq[-1] # ก๊อปปี้เฟรมสุดท้ายมาถมช่องว่างด้านหลัง
    return shifted

def joint_dropout(seq: np.ndarray, drop_prob: float = JOINT_DROP_PROB) -> np.ndarray:
    """สุ่มลบข้อต่อบางจุด (เซ็ตเป็น 0) จำลองกรณี MediaPipe หามือ/ร่างกายบางจุดไม่เจอ"""
    dropped = seq.copy()
    # เช็ค Pose (ลบทีละ 4 ค่า เพราะมี x, y, z, v)
    for j in range(POSE_LM):
        if random.random() < drop_prob:
            base = j * POSE_DIM
            dropped[:, base : base + POSE_DIM] = 0.0
            
    # เช็ค L-Hand (ลบทีละ 3 ค่า x, y, z)
    for j in range(HAND_LM):
        if random.random() < drop_prob:
            base = LH_START + j * HAND_DIM
            dropped[:, base : base + HAND_DIM] = 0.0
            
    # เช็ค R-Hand (ลบทีละ 3 ค่า x, y, z)
    for j in range(HAND_LM):
        if random.random() < drop_prob:
            base = RH_START + j * HAND_DIM
            dropped[:, base : base + HAND_DIM] = 0.0

    return dropped

def scale_translate(seq: np.ndarray, scale_range=SCALE_RANGE, translate_std=TRANSLATE_STD) -> np.ndarray:
    """จำลองการย่อ/ขยาย (ยืนใกล้/ไกลกล้อง) และการย้ายตำแหน่ง (ยืนซ้าย/ขวาจอ)"""
    st = seq.copy()
    T = st.shape[0]

    # ดึง Index แยกตามแกน
    x_idx = np.concatenate([POSE_X_IDX, LH_X_IDX, RH_X_IDX])
    y_idx = np.concatenate([POSE_Y_IDX, LH_Y_IDX, RH_Y_IDX])
    z_idx = np.concatenate([POSE_Z_IDX, LH_Z_IDX, RH_Z_IDX])
    
    # สุ่มค่า Scale (ย่อ/ขยาย) และ Translate (ขยับ)
    scale = np.random.uniform(scale_range[0], scale_range[1])
    tx = np.random.normal(loc=0.0, scale=translate_std)
    ty = np.random.normal(loc=0.0, scale=translate_std)
    tz = np.random.normal(loc=0.0, scale=translate_std)

    # จุดกึ่งกลาง (Origin) ที่ใช้ยึดเพื่อขยาย ซึ่งของ MediaPipe มักอยู่ที่ 0.0
    cx, cy, cz = 0.0, 0.0, 0.0
    for t in range(T):
        # สมการ: จุดใหม่ = (จุดเก่า - จุดศูนย์กลาง) * การย่อขยาย + จุดศูนย์กลาง + การย้ายตำแหน่ง
        st[t, x_idx] = (st[t, x_idx] - cx) * scale + cx + tx
        st[t, y_idx] = (st[t, y_idx] - cy) * scale + cy + ty
        st[t, z_idx] = (st[t, z_idx] - cz) * scale + cz + tz
    return st

def time_warp(seq: np.ndarray, scale_range=TIME_WARP_RANGE) -> np.ndarray:
    """จำลองความเร็วคลิป (เล่นเร็วขึ้น หรือ ช้าลง) โดยใช้ Interpolation"""
    T, F = seq.shape
    scale = np.random.uniform(scale_range[0], scale_range[1])
    
    # คำนวณความยาวใหม่
    new_len = int(round(T * scale))
    new_len = max(2, min(T * 2, new_len)) # ป้องกันไม่ให้สั้นหรือยาวจนเกินไป
    
    # 1. ยืด/หด ตาม scale ด้วยการทำ Linear Interpolation (สมการเส้นตรงสร้างข้อมูลจำลองระหว่างเฟรม)
    old_indices = np.linspace(0, T - 1, num=T)
    temp_indices = np.linspace(0, T - 1, num=new_len)
    interpolator = interp1d(old_indices, seq, axis=0, kind='linear')
    warped_seq = interpolator(temp_indices).astype(np.float32)
    
    # 2. บีบหรือยืดกลับให้เหลือ 30 เฟรมเท่าเดิม เพื่อให้โมเดลรับขนาด [30, 258] ได้
    final_indices = np.linspace(0, warped_seq.shape[0] - 1, num=T)
    final_interpolator = interp1d(np.arange(warped_seq.shape[0]), warped_seq, axis=0, kind='linear')
    
    return final_interpolator(final_indices).astype(np.float32)

def partial_sequence(seq: np.ndarray, keep_range=PARTIAL_KEEP_RANGE) -> np.ndarray:
    """หั่นหัวท้ายคลิปออกเล็กน้อย เพื่อสุ่มดูแค่พาร์ทกลางของการขยับตัว"""
    T, F = seq.shape
    keep_ratio = np.random.uniform(keep_range[0], keep_range[1])
    keep_len = max(2, int(round(T * keep_ratio)))
    if keep_len >= T: return seq.copy()
    
    # สุ่มจุดเริ่มต้นที่จะทำการตัด
    start = np.random.randint(0, T - keep_len + 1)
    sub = seq[start : start + keep_len]

    # ใช้ Interpolation ยืดส่วนที่ถูกตัด ให้กลับมามี 30 เฟรมเท่าเดิม
    old_indices = np.linspace(0, keep_len - 1, num=keep_len)
    new_indices = np.linspace(0, keep_len - 1, num=T)
    interpolator = interp1d(old_indices, sub, axis=0, kind='linear')
    
    return interpolator(new_indices).astype(np.float32)

def prefix_suffix_no_action(seq: np.ndarray, no_action_pool, max_prefix=PREFIX_MAX_FRAMES, max_suffix=SUFFIX_MAX_FRAMES) -> np.ndarray:
    """สุ่มเอาท่าทางคนยืนนิ่งๆ (จากโฟลเดอร์ no_action) มาแปะต่อหัว หรือ ต่อท้ายคลิป"""
    if not no_action_pool: return seq.copy() # ถ้าไม่มี pool ให้ข้าม
    T, F = seq.shape
    na_seq = random.choice(no_action_pool) # สุ่มคลิปยืนนิ่งมา 1 คลิป
    if na_seq.shape != seq.shape: return seq.copy()
    
    prefix_len = np.random.randint(0, max_prefix + 1)
    suffix_len = np.random.randint(0, max_suffix + 1)
    if prefix_len == 0 and suffix_len == 0: return seq.copy()
    
    # ดึงหัว-ท้ายของคลิปยืนนิ่งมา
    prefix = na_seq[:prefix_len] if prefix_len > 0 else np.empty((0, F))
    suffix = na_seq[-suffix_len:] if suffix_len > 0 else np.empty((0, F))

    # จับมาต่อกัน (Concatenate)
    combined = np.concatenate([prefix, seq, suffix], axis=0)
    
    # ใช้ Interpolation บีบความยาวคลิปกลับให้เหลือ 30 เฟรม (T) เหมือนเดิม
    old_indices = np.linspace(0, combined.shape[0] - 1, num=combined.shape[0])
    new_indices = np.linspace(0, combined.shape[0] - 1, num=T)
    interpolator = interp1d(old_indices, combined, axis=0, kind='linear')
    
    return interpolator(new_indices).astype(np.float32)

def simulate_low_fps(seq: np.ndarray, drop_rate: float = LOW_FPS_DROP_RATE) -> np.ndarray:
    """จำลองกล้องมือถือราคาถูกที่เฟรมเรตตก โดยการทำให้ภาพปัจจุบันค้างเหมือนภาพที่แล้ว"""
    T = seq.shape[0]
    new_seq = seq.copy()
    for t in range(1, T):
        if np.random.rand() < drop_rate:
            new_seq[t] = new_seq[t-1] # ก๊อปปี้ภาพเก่ามาวางทับภาพปัจจุบัน
    return new_seq

def simulate_camera_yaw(seq: np.ndarray, max_angle_deg: float = MAX_YAW_DEG) -> np.ndarray:
    """จำลองผู้ใช้วางกล้องเอียงไปทางซ้าย/ขวา (หมุนแกน Y แบบ 3 มิติ)"""
    rotated_seq = seq.copy()
    
    # สุ่มมุมที่จะหมุน แล้วแปลงหน่วยจาก องศา (Degree) เป็น เรเดียน (Radians)
    theta = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # ดึงค่าพิกัด X และ Z ออกมาทั้งหมด
    X = rotated_seq[:, ALL_X_IDX]
    Z = rotated_seq[:, ALL_Z_IDX]
    
    # คำนวณพิกัดใหม่ตามสูตร 3D Rotation Matrix (หมุนรอบแกน Y)
    X_new = X * cos_t + Z * sin_t
    Z_new = -X * sin_t + Z * cos_t
    
    # อัปเดตค่าพิกัดใหม่กลับเข้าไปใน Array เดิม
    rotated_seq[:, ALL_X_IDX] = X_new
    rotated_seq[:, ALL_Z_IDX] = Z_new
    
    return rotated_seq

def temporal_masking(seq: np.ndarray, max_mask_frames: int = MAX_MASK_FRAMES) -> np.ndarray:
    """จำลองสถานการณ์ CPU มือถือทำงานหนักจนกล้องค้างไปหลายเฟรมติดต่อกัน (Freeze block)"""
    masked_seq = seq.copy()
    T = masked_seq.shape[0]
    
    if max_mask_frames <= 0: return masked_seq
    # สุ่มว่าจะค้างกี่เฟรม
    mask_len = np.random.randint(1, max_mask_frames + 1)
    
    # ป้องกันไม่ให้ค้างยาวเกินไปจนทับคลิปทั้งหมด (จำกัดให้ค้างไม่เกิน 1/3 ของคลิป)
    if mask_len >= T - 5: 
        mask_len = T // 3
        
    # สุ่มจุดเริ่มต้นที่กล้องจะค้าง
    start_idx = np.random.randint(0, T - mask_len)
    
    # จำภาพสุดท้ายก่อนค้างไว้
    freeze_frame = masked_seq[start_idx]
    # ถมเฟรมที่ค้างทั้งหมดด้วยภาพสุดท้ายนั้น
    for i in range(start_idx, start_idx + mask_len):
        masked_seq[i] = freeze_frame
        
    return masked_seq

def mediapipe_tracking_loss(seq: np.ndarray, loss_prob: float = TRACKING_LOSS_PROB) -> np.ndarray:
    """จำลองจังหวะที่มือขยับเร็วมากๆ จน MediaPipe หลุดโฟกัส และมือหายไปทั้งก้อน"""
    dropped_seq = seq.copy()
    T = dropped_seq.shape[0]
    
    for t in range(T):
        if random.random() < loss_prob: # ถ้าแจ็คพอตแตกมือหลุด
            # สุ่มเลือกว่าจะทำมือซ้าย หรือ มือขวา หาย (50/50)
            if random.random() < 0.5:
                # เซ็ตข้อมูลพิกัดของมือซ้ายทั้งหมดให้เป็น 0
                dropped_seq[t, LH_START : LH_START + LH_SIZE] = 0.0
            else:
                # เซ็ตข้อมูลพิกัดของมือขวาทั้งหมดให้เป็น 0
                dropped_seq[t, RH_START : RH_START + RH_SIZE] = 0.0
                
    return dropped_seq

# ================== MAIN AUGMENT FOR ONE FILE ==================

def augment_file(action_dir: str, fname: str, no_action_pool, is_no_action_class: bool):
    """ฟังก์ชันหลักที่รับไฟล์ต้นฉบับ 1 ไฟล์ แล้วแตกหน่อออกมาเป็น 10 ไฟล์ Augment"""
    path = os.path.join(action_dir, fname)
    try: 
        seq = np.load(path) # พยายามโหลดไฟล์ npy
    except Exception as e:
        print(f"[SKIP] {path}: {e}")
        return

    # เช็คขนาดให้เป๊ะ ป้องกัน Error ตอนเทรน
    if seq.ndim != 2 or seq.shape[1] != FEAT_DIM:
        print(f"[WARN] shape mismatch (Expected T,{FEAT_DIM}): {path}, shape={seq.shape}")
        return

    # แยกชื่อไฟล์ออกจากนามสกุล (เช่น "clip_01" กับ ".npy")
    base_name, _ = os.path.splitext(fname)
    
    # ทำการแปลงไฟล์และเซฟลงฮาร์ดดิสก์ (แต่ละบรรทัดคือการทำ 1 เทคนิค)
    np.save(os.path.join(action_dir, f"{base_name}_noise1.npy"), add_gaussian_noise(seq, NOISE_STD))
    np.save(os.path.join(action_dir, f"{base_name}_tshift.npy"), temporal_shift(seq, MAX_SHIFT_FRAMES))
    np.save(os.path.join(action_dir, f"{base_name}_drop.npy"), joint_dropout(seq, JOINT_DROP_PROB))
    np.save(os.path.join(action_dir, f"{base_name}_st.npy"), scale_translate(seq, SCALE_RANGE, TRANSLATE_STD))
    np.save(os.path.join(action_dir, f"{base_name}_tw.npy"), time_warp(seq, TIME_WARP_RANGE))
    np.save(os.path.join(action_dir, f"{base_name}_ps.npy"), partial_sequence(seq, PARTIAL_KEEP_RANGE))
    np.save(os.path.join(action_dir, f"{base_name}_yaw.npy"), simulate_camera_yaw(seq, MAX_YAW_DEG))
    np.save(os.path.join(action_dir, f"{base_name}_mask.npy"), temporal_masking(seq, MAX_MASK_FRAMES))
    np.save(os.path.join(action_dir, f"{base_name}_tkloss.npy"), mediapipe_tracking_loss(seq, TRACKING_LOSS_PROB))
    
    # เทคนิคการแปะ no_action จะทำเฉพาะกับคลิปที่ไม่ใช่คลาส no_action เท่านั้น
    if (not is_no_action_class) and no_action_pool:
        np.save(os.path.join(action_dir, f"{base_name}_psna.npy"), prefix_suffix_no_action(seq, no_action_pool))
        
    np.save(os.path.join(action_dir, f"{base_name}_lowfps.npy"), simulate_low_fps(seq, LOW_FPS_DROP_RATE))
    
    print(f"  -> Augmented {base_name}")

# ================== MAIN ==================

def main():
    """ฟังก์ชันจุดตั้งต้นของสคริปต์ ควบคุมการวนลูปผ่านโฟลเดอร์ทั้งหมด"""
    # เช็คก่อนว่ามีโฟลเดอร์ Train ไหม ถ้าไม่มีให้เตือน
    if not os.path.isdir(TRAIN_DIR):
        print(f"[ERROR] ไม่พบโฟลเดอร์ TRAIN_DIR: {TRAIN_DIR}")
        print("กรุณารันไฟล์ split_dataset.py เพื่อสร้างโฟลเดอร์นี้ก่อนครับ")
        return

    # โหลดคลิปยืนนิ่งมาเก็บไว้ใน RAM ก่อน
    no_action_pool = load_no_action_pool(TRAIN_DIR, NO_ACTION_CLASS_NAME)

    # ดึงรายชื่อโฟลเดอร์ (ซึ่งก็คือชื่อ Action ต่างๆ) ใน TRAIN_DIR
    actions = [
        d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))
    ]

    # วนลูปเข้าไปในแต่ละโฟลเดอร์ท่าทาง
    for action in actions:
        action_dir = os.path.join(TRAIN_DIR, action)
        print(f"\n=== Action: {action} ===")
        # ดึงเฉพาะไฟล์นามสกุล .npy และต้อง 'ไม่เคย' ถูกพ่วงชื่อเป็นไฟล์ Augment มาก่อน
        files = sorted([f for f in os.listdir(action_dir) if f.endswith(".npy") and not is_augmented_filename(f)])
        
        if not files:
            print("  (ไม่มีไฟล์ใหม่ให้ augment หรือไฟล์ทั้งหมดถูก augment แล้ว)")
            continue
            
        print(f"  จำนวนไฟล์ต้นฉบับ: {len(files)}")

        # ตัวแปรเช็คว่าตอนนี้กำลังอยู่ในโฟลเดอร์ยืนนิ่ง (no_action) หรือเปล่า
        is_no_action_class = action == NO_ACTION_CLASS_NAME
        
        # วนลูปไฟล์ต้นฉบับทีละไฟล์ แล้วส่งเข้าโรงงานทำ Augment
        for fname in files:
            augment_file(action_dir, fname, no_action_pool, is_no_action_class)


# คำสั่งเพื่อให้ Python รู้ว่าถ้ารันไฟล์นี้โดยตรง ให้เริ่มต้นทำงานที่ฟังก์ชัน main()
if __name__ == "__main__":
    main()