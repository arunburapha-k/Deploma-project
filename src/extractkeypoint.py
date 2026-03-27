import os
import cv2
import numpy as np
import mediapipe as mp
import concurrent.futures
from scipy.interpolate import interp1d

# =====================================================================
# --- 1. จัดการ Path ใหม่ (รองรับโครงสร้าง storage/src และ data) ---
# =====================================================================
# os.path.abspath(__file__) หาที่อยู่เต็มของไฟล์นี้ 
# os.path.dirname() คือการถอยกลับไป 1 โฟลเดอร์แม่
SRC_DIR = os.path.dirname(os.path.abspath(__file__))       # อยู่ที่ storage/src
STORAGE_DIR = os.path.dirname(SRC_DIR)                     # ถอยมาที่ storage
PROJECT_ROOT = os.path.dirname(STORAGE_DIR)                # ถอยมานอกสุด (ระดับเดียวกับ data)

# ชี้ไปยังโฟลเดอร์หลักต่างๆ
MODEL_DIR = os.path.join(STORAGE_DIR, "models")
LOG_DIR = os.path.join(STORAGE_DIR, "logs")                
DATA_DIR = os.path.join(PROJECT_ROOT, "data")              # พุ่งเป้าไปที่โฟลเดอร์ data ด้านนอก

# ตัวแปรย่อยสำหรับโฟลเดอร์ Data
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw")              # เก็บวิดีโอ .mp4 ดิบๆ
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed")  # เก็บไฟล์ .npy ที่สกัดแล้ว

# =====================================================================
# --- 2. CONFIG หลัก ---
# =====================================================================
SEQ_LEN = 30     # ความยาว Sequence มาตรฐานที่ AI ต้องการ (Time steps)
FEAT_DIM = 258   # จำนวนฟีเจอร์ (Pose 132 + Left Hand 63 + Right Hand 63)

# 🔥 บังคับใช้ 6 Cores: การสกัดจุดด้วย MediaPipe เป็นงานที่หนัก CPU (CPU-bound)
# การระบุ Cores ป้องกันไม่ให้ CPU ทำงาน 100% จนเครื่องค้าง (เหลือ 2 Cores ไว้รัน OS)
MAX_CORES = 6  

actions = [
    "anxiety", "fever", "feverish", "insomnia", "itching",
    "no_action", "pain", "polyuria", "suffocated", "wounded"
]

# =====================================================================
# --- 3. Helper Functions (ต้องอยู่นอกสุดเพื่อให้ Multiprocessing มองเห็น) ---
# =====================================================================
def extract_keypoints(results, prev_lh=None, prev_rh=None):
    """
    ดึงค่าพิกัดสัมพัทธ์ (Relative Coordinates) Dimension: 258 
    พร้อมระบบ Forward Fill สำหรับมือที่หายไป
    """
    ref_x, ref_y, ref_z = 0.5, 0.5, 0.0 # ค่าเริ่มต้นของจุดศูนย์กลาง (กรณีหาตัวคนไม่เจอ)
    body_size = 1.0                     # ค่าเริ่มต้นของขนาดตัว

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 📐 1. หาจุดศูนย์กลาง (Origin): ใช้จุดกึ่งกลางระหว่างไหล่ซ้าย(11) และไหล่ขวา(12)
        # ทำให้ AI ไม่สนใจว่าคนจะยืนอยู่ซ้ายหรือขวาของกล้อง (Translation Invariance)
        ref_x = (landmarks[11].x + landmarks[12].x) / 2
        ref_y = (landmarks[11].y + landmarks[12].y) / 2
        ref_z = (landmarks[11].z + landmarks[12].z) / 2

        # 📐 2. หาขนาดตัว (Scale): ใช้ระยะห่างระหว่างไหล่สองข้าง (Euclidean Distance)
        # ทำให้ AI ไม่สนใจว่าคนจะยืนใกล้หรือไกลกล้อง (Scale Invariance)
        dist_x = landmarks[11].x - landmarks[12].x
        dist_y = landmarks[11].y - landmarks[12].y
        body_size = np.sqrt(dist_x**2 + dist_y**2)

        # ป้องกัน Error กรณีคนหันข้างจนไหล่ทับกัน (หารด้วย 0)
        if body_size < 0.001:
            body_size = 1.0

    def get_relative_coords(landmarks_obj, is_pose=False, prev_state=None):
        """ฟังก์ชันย่อยสำหรับแปลงพิกัดดิบ ให้เป็นพิกัดสัมพัทธ์"""
        if not landmarks_obj:
            # 🛡️ Forward Fill: ถ้า MediaPipe หามือไม่เจอ ให้ใช้พิกัดของเฟรมที่แล้วแทน
            # ป้องกันข้อมูลกระตุก (Spike) กลายเป็น 0 กะทันหัน
            if prev_state is not None and np.any(prev_state != 0):
                return prev_state
            # ถ้าเป็นเฟรมแรกแล้วหาไม่เจอเลย ก็ต้องยอมให้เป็น 0
            return np.zeros(33 * 4) if is_pose else np.zeros(21 * 3)

        data = []
        for res in landmarks_obj.landmark:
            # สมการ: จุดสัมพัทธ์ = (พิกัดดิบ - จุดกึ่งกลางไหล่) / ความกว้างไหล่
            rel_x = (res.x - ref_x) / body_size
            rel_y = (res.y - ref_y) / body_size
            rel_z = (res.z - ref_z) / body_size

            if is_pose:
                data.append([rel_x, rel_y, rel_z, res.visibility]) # Pose เก็บค่า V ด้วย
            else:
                data.append([rel_x, rel_y, rel_z])

        return np.array(data).flatten() # ตีแบน Array ย่อยๆ ให้เป็นเวกเตอร์เส้นตรงยาว 1 มิติ

    pose = get_relative_coords(results.pose_landmarks, is_pose=True)
    lh = get_relative_coords(results.left_hand_landmarks, is_pose=False, prev_state=prev_lh)
    rh = get_relative_coords(results.right_hand_landmarks, is_pose=False, prev_state=prev_rh)

    # นำเวกเตอร์ทั้ง 3 ส่วนมาต่อกัน (Concatenate) จะได้ความยาว 132 + 63 + 63 = 258 พอดี
    # และรีเทิร์น lh, rh กลับไปด้วยเพื่อใช้เป็น prev_state ในเฟรมถัดไป
    return np.concatenate([pose, lh, rh]), lh, rh


# =====================================================================
# --- 4. Worker Function (หน้าที่ของแต่ละ Core ที่จะประมวลผลวิดีโอ 1 คลิป) ---
# =====================================================================
def process_single_video(args):
    """
    ฟังก์ชันนี้จะถูกส่งไปให้ CPU แต่ละ Core รันแบบอิสระขนานกัน (Parallel)
    """
    video_path, save_path, action_name, video_filename = args
    
    # 🔥 หัวใจสำคัญ: ต้องประกาศ Model ของ MediaPipe "ภายใน" ฟังก์ชันนี้
    # เพราะโมเดล AI มักจะเกิด Error (Segmentation Fault) ถ้าเราแชร์มันข้าม Process
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        static_image_mode=False,     # False = โหมดวิดีโอ (ใช้ระบบ Tracking ช่วยให้แม่นและเร็วขึ้น)
        model_complexity=2,          # 2 = โมเดลใหญ่สุด แม่นยำสุด
        smooth_landmarks=True,       # ช่วยลดการสั่น (Jitter) ของเส้นโครงกระดูก
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        
        cap = cv2.VideoCapture(video_path)
        all_frames_data = []
        
        # ตัวแปรจำมือของเฟรมที่แล้ว เริ่มต้นเป็น 0
        prev_lh = np.zeros(21 * 3)
        prev_rh = np.zeros(21 * 3)

        while True:
            success, frame = cap.read() # อ่านภาพทีละเฟรม
            if not success:
                break
                
            # 🔥 พลิกภาพเป็นกระจกเงา (Mirror) 
            # สำคัญมากสำหรับภาษามือ เพราะวิดีโอบางคลิปถ่ายกล้องหน้าซ้าย/ขวาจะสลับกัน 
            frame = cv2.flip(frame, 1)

            # OpenCV อ่านภาพมาเป็นโหมด BGR แต่ MediaPipe ต้องการ RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ปิดการเขียนทับภาพเพื่อเพิ่มประสิทธิภาพ (Performance Optimization) ของหน่วยความจำ
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            
            # ส่งเข้าโรงงานสกัดจุด
            keypoints, prev_lh, prev_rh = extract_keypoints(results, prev_lh, prev_rh)
            all_frames_data.append(keypoints)

        cap.release()

    total_extracted = len(all_frames_data)
    
    # ⚠️ [เซฟตี้] ถ้าคลิปสั้นกว่า 30 เฟรม การทำ Interpolation ยืดคลิปอาจจะทำให้ท่าทางผิดเพี้ยน ให้ทิ้งไปเลย
    if total_extracted < SEQ_LEN:
        return f"⚠️ [SKIP] {action_name}/{video_filename} สั้นเกินไป ({total_extracted} เฟรม)"

    # 🔥 ทำการบีบอัด/ยืดเฟรมด้วยสมการคณิตศาสตร์ (Linear Interpolation)
    # สมมติคลิปมี 60 เฟรม เราต้องการลดเหลือ 30 เฟรม 
    # ระบบจะสร้างสมการเส้นตรงเชื่อมทุกจุดใน 60 เฟรม แล้วสุ่มจิ้มดึงข้อมูลออกมา 30 จุดให้ห่างเท่าๆ กัน
    raw_seq = np.array(all_frames_data) # Shape: (total_extracted, 258)
    
    old_indices = np.linspace(0, total_extracted - 1, num=total_extracted)
    new_indices = np.linspace(0, total_extracted - 1, num=SEQ_LEN)
    
    interpolator = interp1d(old_indices, raw_seq, axis=0, kind='linear')
    sequence_data = interpolator(new_indices).astype(np.float32)

    # เซฟเป็นไฟล์ Numpy Binary (.npy) ซึ่งอ่าน/เขียนเร็ว และกินพื้นที่น้อยกว่า .csv หลายเท่า
    np.save(save_path, sequence_data)
    
    # รีเทิร์นข้อความเพื่อแจ้งให้ Main Process ทราบว่าทำงานเสร็จแล้ว
    return f"✅ [OK] {action_name}/{video_filename} (แปลง {total_extracted} -> 30 เฟรม)"


# =====================================================================
# --- 5. Main Execution (จุดปล่อยงานให้ CPU 6 Cores) ---
# =====================================================================
def main():
    print(f"--- 🚀 เริ่มกระบวนการสกัดจุดด้วย CPU {MAX_CORES} Cores ---")
    
    # เช็ก/สร้าง โฟลเดอร์ปลายทาง (processed) ให้ครบทุกคลาส
    for action in actions:
        os.makedirs(os.path.join(PROCESSED_DATA_PATH, action), exist_ok=True)
    
    # จัดเตรียม Task (ตารางงาน) ทั้งหมดเพื่อส่งให้ CPU
    tasks = []
    for action in actions:
        action_raw_path = os.path.join(RAW_DATA_PATH, action)
        action_processed_path = os.path.join(PROCESSED_DATA_PATH, action)

        if not os.path.exists(action_raw_path):
            continue

        # ดึงรายชื่อวิดีโอ (รองรับทั้ง mp4, avi, mov)
        video_files = [f for f in os.listdir(action_raw_path) if f.endswith((".mp4", ".avi", ".mov"))]
        
        for sequence_idx, video_file in enumerate(video_files):
            video_path = os.path.join(action_raw_path, video_file)
            # ตั้งชื่อไฟล์ .npy ใหม่ให้เป็นมาตรฐาน (เช่น fever_0.npy, fever_1.npy)
            npy_path = os.path.join(action_processed_path, f"{action}_{sequence_idx}.npy")
            
            # ยัดงานลงกล่องเพื่อรอแจกจ่าย
            tasks.append((video_path, npy_path, action, video_file))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print(f"❌ ไม่พบไฟล์วิดีโอในโฟลเดอร์ {RAW_DATA_PATH} เลยครับ")
        return

    print(f"📂 พบวิดีโอทั้งหมด: {total_tasks} คลิป")

    processed_count = 0
    # 🔥 พระเอกของงานนี้: ProcessPoolExecutor
    # ใช้ "Process" แทน "Thread" เพื่อหลบเลี่ยง Python GIL (Global Interpreter Lock)
    # ทำให้ CPU 6 คอร์สามารถคำนวณเลขหนักๆ ได้พร้อมกัน 100% เร็วกว่ารันปกติเกือบ 6 เท่า!
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_CORES) as executor:
        # สั่งรันและรอรับผลลัพธ์แบบ Asynchronous (คอร์ไหนเสร็จก่อน ก็ส่งผลลัพธ์ออกมาก่อน)
        results = executor.map(process_single_video, tasks)
        
        for result_msg in results:
            processed_count += 1
            # พิมพ์ทับบรรทัดเดิม (\r) เพื่อทำแถบสถานะความคืบหน้าแบบประหยัดบรรทัด
            print(f"\r⏳ ความคืบหน้า: [{processed_count}/{total_tasks}] {result_msg}", end="", flush=True)

    print("\n\n🎉 --- สกัดจุดเสร็จสมบูรณ์ 100% เตรียมนำไปเข้ากระบวนการ Augment ต่อได้เลย! ---")


if __name__ == '__main__':
    # ต้องมี if __name__ == '__main__': เสมอเมื่อใช้ ProcessPoolExecutor บน Windows 
    # ไม่งั้นจะเกิดอาการ Recursive Spawn (เปิดโปรแกรมซ้อนกันไม่สิ้นสุด)
    main()