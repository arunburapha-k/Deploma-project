"""
augment_offline.py (Version: 258 Features)

สคริปต์สำหรับทำ Offline Data Augmentation
เฉพาะ train set (เช่น data/processed_train)

⚠️ เวอร์ชันนี้ถูกปรับให้ตรงกับโครงสร้างฟีเจอร์ที่มี Z กลับมาแล้ว:
    - Pose   : 33 จุด × 4 ค่า (x, y, z, visibility) = 132
    - L-Hand : 21 จุด × 3 ค่า (x, y, z)           = 63
    - R-Hand : 21 จุด × 3 ค่า (x, y, z)           = 63
    รวมทั้งหมด                                  = 258 ค่า ต่อเฟรม
"""

import os
import numpy as np
import random
from pathlib import Path
from scipy.interpolate import interp1d

# =====================================================================
# --- 1. จัดการ Path ใหม่ (ถอย 2 ขั้น: src -> storage -> Project Root) ---
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# ชี้เป้าไปที่โฟลเดอร์ Train เท่านั้น
TRAIN_DIR = str(DATA_DIR / "processed_train")


# ================== CONFIG ทั่วไป ==================
RANDOM_SEED = 42

SEQ_LEN = 30
FEAT_DIM = 258 

POSE_LM = 33
POSE_DIM = 4    # x, y, z, vis
HAND_LM = 21
HAND_DIM = 3    # x, y, z

POSE_SIZE = POSE_LM * POSE_DIM  # 132
LH_START = POSE_SIZE  # 132
LH_SIZE = HAND_LM * HAND_DIM  # 63
RH_START = LH_START + LH_SIZE  # 195
RH_SIZE = HAND_LM * HAND_DIM  # 63
FEATURE_TOTAL = POSE_SIZE + LH_SIZE + RH_SIZE

assert FEATURE_TOTAL == FEAT_DIM, f"FEAT_DIM ต้องเท่ากับ {FEATURE_TOTAL} (Pose132+LH63+RH63)"

# ==========================================================
# Config (Safe Mode: ป้องกันพิกัดระเบิดและรักษารูปทรงกายวิภาค)
# ==========================================================
NOISE_STD = 0.02          
MAX_SHIFT_FRAMES = 3      
JOINT_DROP_PROB = 0.05    
SCALE_RANGE = (0.80, 1.20)
TRANSLATE_STD = 0.03      

TIME_WARP_RANGE = (0.85, 1.15) 
PARTIAL_KEEP_RANGE = (0.85, 0.95) 
PREFIX_MAX_FRAMES = 2     
SUFFIX_MAX_FRAMES = 2     
LOW_FPS_DROP_RATE = 0.3   
NO_ACTION_CLASS_NAME = "no_action"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------- index helper (Updated for 258) -----------------
POSE_X_IDX = np.arange(0, POSE_SIZE, 4)
POSE_Y_IDX = np.arange(1, POSE_SIZE, 4)
POSE_Z_IDX = np.arange(2, POSE_SIZE, 4)
POSE_VIS_IDX = np.arange(3, POSE_SIZE, 4)

LH_X_IDX = LH_START + np.arange(0, LH_SIZE, 3)
LH_Y_IDX = LH_START + np.arange(1, LH_SIZE, 3)
LH_Z_IDX = LH_START + np.arange(2, LH_SIZE, 3)

RH_X_IDX = RH_START + np.arange(0, RH_SIZE, 3)
RH_Y_IDX = RH_START + np.arange(1, RH_SIZE, 3)
RH_Z_IDX = RH_START + np.arange(2, RH_SIZE, 3)

# Pair สำหรับ Flip สลับซ้ายขวา
POSE_FLIP_PAIRS = np.array([
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
])
POSE_FLIP_INDICES = []
for l_idx, r_idx in POSE_FLIP_PAIRS:
    for d in range(POSE_DIM): 
        POSE_FLIP_INDICES.append((l_idx * POSE_DIM + d, r_idx * POSE_DIM + d))


# ================== UTILITIES ==================

def is_augmented_filename(fname: str) -> bool:
    name, ext = os.path.splitext(fname)
    if ext != ".npy": return True
    suffixes = ["_flip", "_noise1", "_tshift", "_drop", "_st", "_tw", "_ps", "_psna", "_lowfps"]
    return any(name.endswith(suf) for suf in suffixes)

def load_no_action_pool(train_dir: str, class_name: str = "no_action"):
    pool = []
    na_dir = os.path.join(train_dir, class_name)
    if not os.path.isdir(na_dir):
        return pool
    files = sorted([f for f in os.listdir(na_dir) if f.endswith(".npy") and not is_augmented_filename(f)])
    for fname in files:
        path = os.path.join(na_dir, fname)
        try:
            seq = np.load(path)
        except: continue
        if seq.ndim == 2 and seq.shape[0] == SEQ_LEN and seq.shape[1] == FEAT_DIM:
            pool.append(seq)
    return pool

# ================== Augmentation Functions ==================

def flip_keypoints_frame(keypoints: np.ndarray) -> np.ndarray:
    flipped = np.copy(keypoints)
    
    flipped[POSE_X_IDX] = -flipped[POSE_X_IDX]
    flipped[LH_X_IDX] = -flipped[LH_X_IDX]
    flipped[RH_X_IDX] = -flipped[RH_X_IDX]

    for l_flat, r_flat in POSE_FLIP_INDICES:
        flipped[l_flat], flipped[r_flat] = flipped[r_flat], flipped[l_flat]

    lh_block = np.copy(flipped[LH_START : LH_START + LH_SIZE])
    rh_block = np.copy(flipped[RH_START : RH_START + RH_SIZE])
    flipped[LH_START : LH_START + LH_SIZE] = rh_block
    flipped[RH_START : RH_START + RH_SIZE] = lh_block

    return flipped

def horizontal_flip_sequence(seq: np.ndarray) -> np.ndarray:
    flipped_seq = np.empty_like(seq)
    for t in range(seq.shape[0]):
        flipped_seq[t] = flip_keypoints_frame(seq[t])
    return flipped_seq

def add_gaussian_noise(seq: np.ndarray, std: float = NOISE_STD) -> np.ndarray:
    noisy_seq = seq.copy()
    noise = np.random.normal(loc=0.0, scale=std, size=seq.shape)
    
    xyz_idx = np.concatenate([
        POSE_X_IDX, POSE_Y_IDX, POSE_Z_IDX, 
        LH_X_IDX, LH_Y_IDX, LH_Z_IDX, 
        RH_X_IDX, RH_Y_IDX, RH_Z_IDX
    ])
    
    for t in range(noisy_seq.shape[0]):
        noisy_seq[t, xyz_idx] += noise[t, xyz_idx]
        
    return noisy_seq

def temporal_shift(seq: np.ndarray, max_shift: int = MAX_SHIFT_FRAMES) -> np.ndarray:
    T = seq.shape[0]
    if T <= 1 or max_shift <= 0: return seq.copy()
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0: return seq.copy()
    shifted = np.empty_like(seq)
    if shift > 0:
        shifted[shift:] = seq[:-shift]
        shifted[:shift] = seq[0]
    else:
        k = -shift
        shifted[:-k] = seq[k:]
        shifted[-k:] = seq[-1]
    return shifted

def joint_dropout(seq: np.ndarray, drop_prob: float = JOINT_DROP_PROB) -> np.ndarray:
    dropped = seq.copy()
    
    for j in range(POSE_LM):
        if random.random() < drop_prob:
            base = j * POSE_DIM
            dropped[:, base : base + POSE_DIM] = 0.0
            
    for j in range(HAND_LM):
        if random.random() < drop_prob:
            base = LH_START + j * HAND_DIM
            dropped[:, base : base + HAND_DIM] = 0.0
            
    for j in range(HAND_LM):
        if random.random() < drop_prob:
            base = RH_START + j * HAND_DIM
            dropped[:, base : base + HAND_DIM] = 0.0

    return dropped

def scale_translate(seq: np.ndarray, scale_range=SCALE_RANGE, translate_std=TRANSLATE_STD) -> np.ndarray:
    st = seq.copy()
    T = st.shape[0]

    x_idx = np.concatenate([POSE_X_IDX, LH_X_IDX, RH_X_IDX])
    y_idx = np.concatenate([POSE_Y_IDX, LH_Y_IDX, RH_Y_IDX])
    z_idx = np.concatenate([POSE_Z_IDX, LH_Z_IDX, RH_Z_IDX])
    
    scale = np.random.uniform(scale_range[0], scale_range[1])
    tx = np.random.normal(loc=0.0, scale=translate_std)
    ty = np.random.normal(loc=0.0, scale=translate_std)
    tz = np.random.normal(loc=0.0, scale=translate_std)

    cx, cy, cz = 0.0, 0.0, 0.0
    for t in range(T):
        st[t, x_idx] = (st[t, x_idx] - cx) * scale + cx + tx
        st[t, y_idx] = (st[t, y_idx] - cy) * scale + cy + ty
        st[t, z_idx] = (st[t, z_idx] - cz) * scale + cz + tz

    return st

def time_warp(seq: np.ndarray, scale_range=TIME_WARP_RANGE) -> np.ndarray:
    T, F = seq.shape
    scale = np.random.uniform(scale_range[0], scale_range[1])
    
    # คำนวณความยาวใหม่ก่อนถูกบังคับให้เหลือ T
    new_len = int(round(T * scale))
    new_len = max(2, min(T * 2, new_len)) # ป้องกันสั้น/ยาวเกินไป
    
    # 1. ยืด/หด ตาม scale ด้วย Interpolation
    old_indices = np.linspace(0, T - 1, num=T)
    temp_indices = np.linspace(0, T - 1, num=new_len)
    interpolator = interp1d(old_indices, seq, axis=0, kind='linear')
    warped_seq = interpolator(temp_indices).astype(np.float32)
    
    # 2. ตัดหรือซูมให้กลับมาเหลือความยาว T เท่าเดิม (เพื่อให้โมเดลรับได้)
    final_indices = np.linspace(0, warped_seq.shape[0] - 1, num=T)
    final_interpolator = interp1d(np.arange(warped_seq.shape[0]), warped_seq, axis=0, kind='linear')
    
    return final_interpolator(final_indices).astype(np.float32)

def partial_sequence(seq: np.ndarray, keep_range=PARTIAL_KEEP_RANGE) -> np.ndarray:
    T, F = seq.shape
    keep_ratio = np.random.uniform(keep_range[0], keep_range[1])
    keep_len = max(2, int(round(T * keep_ratio)))
    if keep_len >= T: return seq.copy()
    
    start = np.random.randint(0, T - keep_len + 1)
    sub = seq[start : start + keep_len]

    old_indices = np.linspace(0, keep_len - 1, num=keep_len)
    new_indices = np.linspace(0, keep_len - 1, num=T)
    interpolator = interp1d(old_indices, sub, axis=0, kind='linear')
    
    return interpolator(new_indices).astype(np.float32)

def prefix_suffix_no_action(seq: np.ndarray, no_action_pool, max_prefix=PREFIX_MAX_FRAMES, max_suffix=SUFFIX_MAX_FRAMES) -> np.ndarray:
    if not no_action_pool: return seq.copy()
    T, F = seq.shape
    na_seq = random.choice(no_action_pool)
    if na_seq.shape != seq.shape: return seq.copy()
    
    prefix_len = np.random.randint(0, max_prefix + 1)
    suffix_len = np.random.randint(0, max_suffix + 1)
    if prefix_len == 0 and suffix_len == 0: return seq.copy()
    
    prefix = na_seq[:prefix_len] if prefix_len > 0 else np.empty((0, F))
    suffix = na_seq[-suffix_len:] if suffix_len > 0 else np.empty((0, F))

    combined = np.concatenate([prefix, seq, suffix], axis=0)
    
    # 🔥 ใช้ Interpolation บีบกลับให้เหลือ T เฟรม
    old_indices = np.linspace(0, combined.shape[0] - 1, num=combined.shape[0])
    new_indices = np.linspace(0, combined.shape[0] - 1, num=T)
    interpolator = interp1d(old_indices, combined, axis=0, kind='linear')
    
    return interpolator(new_indices).astype(np.float32)

def simulate_low_fps(seq: np.ndarray, drop_rate: float = LOW_FPS_DROP_RATE) -> np.ndarray:
    T = seq.shape[0]
    new_seq = seq.copy()
    for t in range(1, T):
        if np.random.rand() < drop_rate:
            new_seq[t] = new_seq[t-1]
    return new_seq

# ================== MAIN AUGMENT FOR ONE FILE ==================

def augment_file(action_dir: str, fname: str, no_action_pool, is_no_action_class: bool):
    path = os.path.join(action_dir, fname)
    try: seq = np.load(path)
    except Exception as e:
        print(f"[SKIP] {path}: {e}")
        return

    if seq.ndim != 2 or seq.shape[1] != FEAT_DIM:
        print(f"[WARN] shape mismatch (Expected T,{FEAT_DIM}): {path}, shape={seq.shape}")
        return

    base_name, _ = os.path.splitext(fname)
    
    # np.save(os.path.join(action_dir, f"{base_name}_flip.npy"), horizontal_flip_sequence(seq))
    np.save(os.path.join(action_dir, f"{base_name}_noise1.npy"), add_gaussian_noise(seq, NOISE_STD))
    np.save(os.path.join(action_dir, f"{base_name}_tshift.npy"), temporal_shift(seq, MAX_SHIFT_FRAMES))
    np.save(os.path.join(action_dir, f"{base_name}_drop.npy"), joint_dropout(seq, JOINT_DROP_PROB))
    np.save(os.path.join(action_dir, f"{base_name}_st.npy"), scale_translate(seq, SCALE_RANGE, TRANSLATE_STD))
    np.save(os.path.join(action_dir, f"{base_name}_tw.npy"), time_warp(seq, TIME_WARP_RANGE))
    np.save(os.path.join(action_dir, f"{base_name}_ps.npy"), partial_sequence(seq, PARTIAL_KEEP_RANGE))
    
    if (not is_no_action_class) and no_action_pool:
        np.save(os.path.join(action_dir, f"{base_name}_psna.npy"), prefix_suffix_no_action(seq, no_action_pool))
        
    np.save(os.path.join(action_dir, f"{base_name}_lowfps.npy"), simulate_low_fps(seq, LOW_FPS_DROP_RATE))
    
    print(f"  -> Augmented {base_name}")

# ================== MAIN ==================

def main():
    if not os.path.isdir(TRAIN_DIR):
        print(f"[ERROR] ไม่พบโฟลเดอร์ TRAIN_DIR: {TRAIN_DIR}")
        print("กรุณารันไฟล์ split_dataset.py เพื่อสร้างโฟลเดอร์นี้ก่อนครับ")
        return

    no_action_pool = load_no_action_pool(TRAIN_DIR, NO_ACTION_CLASS_NAME)

    actions = [
        d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))
    ]

    for action in actions:
        action_dir = os.path.join(TRAIN_DIR, action)
        print(f"\n=== Action: {action} ===")
        files = sorted([f for f in os.listdir(action_dir) if f.endswith(".npy") and not is_augmented_filename(f)])
        
        if not files:
            print("  (ไม่มีไฟล์ใหม่ให้ augment หรือไฟล์ทั้งหมดถูก augment แล้ว)")
            continue
            
        print(f"  จำนวนไฟล์ต้นฉบับ: {len(files)}")

        is_no_action_class = action == NO_ACTION_CLASS_NAME
        for fname in files:
            augment_file(action_dir, fname, no_action_pool, is_no_action_class)


if __name__ == "__main__":
    main()