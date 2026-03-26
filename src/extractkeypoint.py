import os
import cv2
import numpy as np
import mediapipe as mp
import concurrent.futures
from scipy.interpolate import interp1d

# =====================================================================
# --- 1. จัดการ Path ใหม่ (รองรับโครงสร้าง storage/src และ data) ---
# =====================================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))       # อยู่ที่ storage/src
STORAGE_DIR = os.path.dirname(SRC_DIR)                     # ถอยมาที่ storage
PROJECT_ROOT = os.path.dirname(STORAGE_DIR)                # ถอยมานอกสุด (ระดับเดียวกับ data)

# ชี้ไปยังโฟลเดอร์หลัก
MODEL_DIR = os.path.join(STORAGE_DIR, "models")
LOG_DIR = os.path.join(STORAGE_DIR, "logs")                
DATA_DIR = os.path.join(PROJECT_ROOT, "data")              # พุ่งเป้าไปที่โฟลเดอร์ data ด้านนอก

# ตัวแปรย่อยสำหรับโฟลเดอร์ Data
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed")

# =====================================================================
# --- 2. CONFIG หลัก ---
# =====================================================================
SEQ_LEN = 30
FEAT_DIM = 258
MAX_CORES = 6  # 🔥 บังคับใช้ 6 Cores 

actions = [
    "anxiety",
    "fever",
    "feverish",
    "insomnia",
    "itching",
    "no_action",
    "pain",
    "polyuria",
    "suffocated",
    "wounded"
]

# =====================================================================
# --- 3. Helper Functions (ต้องอยู่นอกสุดเพื่อให้ Multiprocessing มองเห็น) ---
# =====================================================================
def extract_keypoints(results, prev_lh=None, prev_rh=None):
    """ดึงค่าพิกัดสัมพัทธ์ Dimension: 258 พร้อมระบบ Forward Fill"""
    ref_x, ref_y, ref_z = 0.5, 0.5, 0.0
    body_size = 1.0

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        ref_x = (landmarks[11].x + landmarks[12].x) / 2
        ref_y = (landmarks[11].y + landmarks[12].y) / 2
        ref_z = (landmarks[11].z + landmarks[12].z) / 2

        dist_x = landmarks[11].x - landmarks[12].x
        dist_y = landmarks[11].y - landmarks[12].y
        body_size = np.sqrt(dist_x**2 + dist_y**2)

        if body_size < 0.001:
            body_size = 1.0

    def get_relative_coords(landmarks_obj, is_pose=False, prev_state=None):
        if not landmarks_obj:
            if prev_state is not None and np.any(prev_state != 0):
                return prev_state
            return np.zeros(33 * 4) if is_pose else np.zeros(21 * 3)

        data = []
        for res in landmarks_obj.landmark:
            rel_x = (res.x - ref_x) / body_size
            rel_y = (res.y - ref_y) / body_size
            rel_z = (res.z - ref_z) / body_size

            if is_pose:
                data.append([rel_x, rel_y, rel_z, res.visibility])
            else:
                data.append([rel_x, rel_y, rel_z])

        return np.array(data).flatten()

    pose = get_relative_coords(results.pose_landmarks, is_pose=True)
    lh = get_relative_coords(results.left_hand_landmarks, is_pose=False, prev_state=prev_lh)
    rh = get_relative_coords(results.right_hand_landmarks, is_pose=False, prev_state=prev_rh)

    return np.concatenate([pose, lh, rh]), lh, rh


# =====================================================================
# --- 4. Worker Function (หน้าที่ของแต่ละ Core ที่จะประมวลผลวิดีโอ 1 คลิป) ---
# =====================================================================
def process_single_video(args):
    """ฟังก์ชันนี้จะถูกเรียกใช้โดย CPU แต่ละ Core"""
    video_path, save_path, action_name, video_filename = args
    
    # 🔥 สร้าง Holistic ใหม่ 1 ตัว ต่อ 1 คลิปเสมอ ป้องกันเส้นพิกัดกระชาก
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        
        cap = cv2.VideoCapture(video_path)
        all_frames_data = []
        prev_lh = np.zeros(21 * 3)
        prev_rh = np.zeros(21 * 3)

        while True:
            success, frame = cap.read()
            if not success:
                break
                
            # 🔥 (เปิดไว้) พลิกภาพเป็นกระจกเงา หากคลิปถูกถ่ายด้วยกล้องหน้า
            frame = cv2.flip(frame, 1)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            
            keypoints, prev_lh, prev_rh = extract_keypoints(results, prev_lh, prev_rh)
            all_frames_data.append(keypoints)

        cap.release()

    total_extracted = len(all_frames_data)
    
    if total_extracted < SEQ_LEN:
        return f"⚠️ [SKIP] {action_name}/{video_filename} สั้นเกินไป ({total_extracted} เฟรม)"

    # 🔥 ทำการบีบอัด/ยืดเฟรมด้วยสมการคณิตศาสตร์ (Interpolation)
    raw_seq = np.array(all_frames_data) # Shape: (total_extracted, 258)
    
    old_indices = np.linspace(0, total_extracted - 1, num=total_extracted)
    new_indices = np.linspace(0, total_extracted - 1, num=SEQ_LEN)
    
    interpolator = interp1d(old_indices, raw_seq, axis=0, kind='linear')
    sequence_data = interpolator(new_indices).astype(np.float32)

    # เซฟเป็น .npy
    np.save(save_path, sequence_data)
    return f"✅ [OK] {action_name}/{video_filename} (แปลง {total_extracted} -> 30 เฟรม)"


# =====================================================================
# --- 5. Main Execution (จุดปล่อยงานให้ CPU 6 Cores) ---
# =====================================================================
def main():
    print(f"--- 🚀 เริ่มกระบวนการสกัดจุดด้วย CPU {MAX_CORES} Cores ---")
    
    # เช็กโฟลเดอร์ให้เรียบร้อย
    for action in actions:
        os.makedirs(os.path.join(PROCESSED_DATA_PATH, action), exist_ok=True)
    
    # จัดเตรียม Task (ตารางงาน) ทั้งหมด
    tasks = []
    for action in actions:
        action_raw_path = os.path.join(RAW_DATA_PATH, action)
        action_processed_path = os.path.join(PROCESSED_DATA_PATH, action)

        if not os.path.exists(action_raw_path):
            continue

        video_files = [f for f in os.listdir(action_raw_path) if f.endswith((".mp4", ".avi", ".mov"))]
        
        for sequence_idx, video_file in enumerate(video_files):
            video_path = os.path.join(action_raw_path, video_file)
            # ตั้งชื่อไฟล์ .npy ตามชื่อคลาสและลำดับ
            npy_path = os.path.join(action_processed_path, f"{action}_{sequence_idx}.npy")
            
            # ยัดงานลงกล่อง (video_path, save_path, action_name, video_filename)
            tasks.append((video_path, npy_path, action, video_file))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print(f"❌ ไม่พบไฟล์วิดีโอในโฟลเดอร์ {RAW_DATA_PATH} เลยครับ")
        return

    print(f"📂 พบวิดีโอทั้งหมด: {total_tasks} คลิป")

    processed_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_CORES) as executor:
        # สั่งรันและรอรับผลลัพธ์
        results = executor.map(process_single_video, tasks)
        
        for result_msg in results:
            processed_count += 1
            print(f"\r⏳ ความคืบหน้า: [{processed_count}/{total_tasks}] {result_msg}", end="", flush=True)

    print("\n\n🎉 --- สกัดจุดเสร็จสมบูรณ์ 100% เตรียมนำไปเข้ากระบวนการ Augment ต่อได้เลย! ---")


if __name__ == '__main__':
    main()