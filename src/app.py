import os, json, collections
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import threading
import time
import queue # 🔥 คิวสำหรับรับส่งข้อมูลข้าม Thread อย่างปลอดภัย (Thread-safe)

# ========== CONFIG ==========
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.dirname(SRC_DIR)
MODEL_DIR = os.path.join(STORAGE_DIR, "models")

# ใช้งานไฟล์ INT8 ซึ่งโหลดเร็วสุดและกินแรมน้อยสุด (เหมาะกับ Edge Simulation)
TFLITE_MODEL = os.path.join(MODEL_DIR, "model_int8.tflite")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")
THRESH_PATH = os.path.join(MODEL_DIR, "thresholds.json")

SEQ_LEN = 30
FEAT_DIM = 258

# --- ตัวตั้งค่าเพื่อความเสถียรของ UI (Anti-flicker) ---
PROCESS_EVERY_N = 1     # รันโมเดลทุกๆ กี่เฟรม (ถ้าเครื่องช้ามาก อาจจะตั้งเป็น 2 เพื่อรันแบบเฟรมเว้นเฟรม)
ALPHA_EMA = 0.15        # สัมประสิทธิ์ Exponential Moving Average (ยิ่งน้อย กราฟยิ่งสมูทแต่ตอบสนองช้า)
DEFAULT_THRESH = 0.70   # เกณฑ์ความมั่นใจขั้นต่ำที่ AI จะกล้าฟันธง
TOP2_MARGIN = 0.20      # ระยะห่างความมั่นใจระหว่างอันดับ 1 กับ 2 (ป้องกัน AI ลังเล)
MIN_COVERAGE = 0.50     # เฟรมต้องไม่เป็น 0 (หาคนไม่เจอ) เกินกว่า 50% ของคลิป
STABLE_FRAMES = 5       # ต้องทายท่าเดียวกันติดกัน 5 เฟรม ถึงจะแสดงผลขึ้นจอ (กันข้อความกระพริบ)

CAM_INDEX = 0           # กล้อง WebCam (0 คือกล้องตัวแรกของเครื่อง)
FRAME_W, FRAME_H = 1280, 720
MODEL_COMPLEXITY = 1    # ความละเอียด MediaPipe (ลดจาก 2 เป็น 1 เพื่อเน้น FPS ล้วนๆ)

# ========== Utils (ฟังก์ชันคณิตศาสตร์และการวาด) ==========
def nonzero_frames_ratio(seq30x258: np.ndarray) -> float:
    """เช็คว่าใน 30 เฟรม มีเฟรมที่เห็นคน (ไม่เป็น 0 ล้วน) กี่เปอร์เซ็นต์"""
    if seq30x258.shape != (SEQ_LEN, FEAT_DIM):
        return 0.0
    return float(np.any(seq30x258 != 0.0, axis=1).sum()) / float(SEQ_LEN)

def extract_258(results, prev_lh=None, prev_rh=None):
    """(เหมือนเดิม) สกัดพิกัดสัมพัทธ์และใช้ Forward Fill ตอนมือหาย"""
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
        return np.array(data, dtype=np.float32).flatten()

    pose = get_relative_coords(results.pose_landmarks, is_pose=True)
    lh = get_relative_coords(results.left_hand_landmarks, is_pose=False, prev_state=prev_lh)
    rh = get_relative_coords(results.right_hand_landmarks, is_pose=False, prev_state=prev_rh)
    return np.concatenate([pose, lh, rh]), lh, rh

def draw_header(image, label_text, conf, fps):
    """วาดแถบดำด้านบนจอ พร้อมข้อความและตัววัด FPS"""
    H, W = image.shape[:2]
    cv2.rectangle(image, (0, 0), (W, 80), (0, 0, 0), -1)
    
    cv2.putText(image, f"FPS: {fps:.1f}", (W - 150, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    cv2.putText(
        image, f"{label_text}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
        (0, 255, 0) if conf > 0 else (200, 200, 200), 2, cv2.LINE_AA,
    )
    if conf > 0:
        cv2.putText(
            image, f"Conf: {conf*100:.1f}%", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

def draw_topk_bars(image, labels, probs, k=3, origin=(20, 100)):
    """วาดกราฟแท่ง (Bar Chart) 3 อันดับแรก เพื่อให้เห็นความลังเลของโมเดล"""
    H, W = image.shape[:2]
    x0, y0 = origin
    bar_w = int(W * 0.85)
    bar_h = 20
    gap = 15
    idxs = np.argsort(probs)[-k:][::-1]

    cv2.putText(image, "Top Predictions:", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    for i, idx in enumerate(idxs):
        p = float(probs[idx])
        w = int(bar_w * p)
        y = y0 + i * (bar_h + gap)
        cv2.rectangle(image, (x0, y), (x0 + bar_w, y + bar_h), (50, 50, 50), 1)
        color = (0, 255, 255) if i == 0 and p > 0.5 else (100, 100, 100)
        cv2.rectangle(image, (x0, y), (x0 + w, y + bar_h), color, -1)
        cv2.putText(
            image, f"{labels[idx]}: {p*100:.1f}%", (x0 + 5, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255) if p > 0.5 else (180, 180, 180), 1, cv2.LINE_AA,
        )

# ========== Load Config & TFLite ==========
print("TF:", tf.__version__)

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)
labels = [label_map[str(i)] for i in range(len(label_map))]

# โหลด Thresholds ที่คำนวณไว้ตอนเทรนมาเป็นมาตรฐานชี้วัด
per_th = {c: DEFAULT_THRESH for c in labels}
if os.path.exists(THRESH_PATH):
    try:
        with open(THRESH_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for k, v in raw.items():
            cname = labels[int(k)] if k.isdigit() and int(k) < len(labels) else k
            if cname in per_th:
                per_th[cname] = float(v["threshold"] if isinstance(v, dict) else v)
    except: pass

# โหลด TFLite (ใช้ 4 Thread ฝั่ง CPU สำหรับประมวลผลเมทริกซ์)
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL, num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_tflite(x_in):
    """ยัด Data ลงช่อง Input ของ TFLite และดึง Output ออกมา"""
    x_in = x_in.astype(input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], x_in)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0].astype(np.float32)

# ========== 🔥 Background ML Worker (Multithreading ตัวจริง) ==========
# สร้าง Queue เล็กๆ (maxsize=2) ป้องกันภาพเก่าค้างในคิว ถ้ากล้องส่งมาเร็วกว่าที่ ML ประมวลผลทัน
frame_queue = queue.Queue(maxsize=2)

# ดิกชันนารีแชร์ข้อมูลข้าม Thread (เพราะ Dictionary ใน Python เปลี่ยนค่าข้าม Thread ได้ปลอดภัย)
result_data = {
    "probs": None, "shown_label": "Scanning...", "shown_conf": 0.0
}

def ml_worker():
    """ทำงานเป็นคนงานหลังบ้าน คอยดึงรูปจากคิวมาหาร่าง MediaPipe และเรียก AI ทำนาย"""
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False, model_complexity=MODEL_COMPLEXITY, 
        smooth_landmarks=True, min_detection_confidence=0.4, min_tracking_confidence=0.4,
        # 🔥 ปิด Face Mesh (468 จุดบนหน้า) ทิ้ง เพราะเราไม่ได้ใช้ในโมเดล 258 Features ของเราเลย!
        # ช่วยประหยัด CPU มือถือไปได้มหาศาล และเพิ่ม FPS ได้อย่างชัดเจน
        refine_face_landmarks=False 
    )
    
    # วงแหวนเก็บข้อมูล (Ring Buffer) 30 เฟรม พอครบ 30 อันที่ 31 มา อันแรกจะโดนถีบออกไปเอง
    seq_buf = collections.deque(maxlen=SEQ_LEN)
    prev_lh_state, prev_rh_state = np.zeros(21*3, dtype=np.float32), np.zeros(21*3, dtype=np.float32)
    prev_probs = None
    frame_count = 0
    candidate_label, candidate_streak = None, 0

    while True:
        try:
            # ดึงภาพจากคิวแบบไม่บล็อก (ถ้าไม่มีภาพให้ข้ามลูปนี้ไปก่อน)
            frame_rgb = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
            
        if frame_rgb is None: # สัญญาณให้ปิด Thread (รับมาจากคำสั่งปิดหน้าต่าง)
            break

        # รัน MediaPipe
        res = holistic.process(frame_rgb)
        features, prev_lh_state, prev_rh_state = extract_258(res, prev_lh_state, prev_rh_state)
        seq_buf.append(features)
        frame_count += 1

        # รัน Bi-GRU แบบ Sliding Window (ถ้าบัฟเฟอร์มีครบ 30 เฟรมแล้ว)
        if len(seq_buf) == SEQ_LEN and (frame_count % PROCESS_EVERY_N == 0):
            # เสริมมิติ Batch เข้าไปให้กลายเป็น [1, 30, 258]
            x = np.array(seq_buf, dtype=np.float32)[None, ...]
            probs = run_tflite(x)

            # ใช้วิธี EMA (Exponential Moving Average) ในการเกลี่ยความน่าจะเป็นให้สมูทขึ้น
            # ป้องกันอาการ 'แท่งกราฟดีดขึ้นลง' ทุกเฟรมจนผู้ใช้อ่านไม่ทัน
            smoothed = probs if prev_probs is None else (ALPHA_EMA * probs + (1 - ALPHA_EMA) * prev_probs)
            prev_probs = smoothed

            # 🧠 Logic การตัดสินใจขั้นสูง (Heuristic Filters)
            top3 = np.argsort(smoothed)[-3:][::-1]
            top, second = int(top3[0]), int(top3[1])
            name_top, conf_top, conf_second = labels[top], float(smoothed[top]), float(smoothed[second])
            
            margin = conf_top - conf_second # ห่างกันเท่าไหร่? (มั่นใจมากแค่ไหนว่าไม่ใช่ท่าย่อยอีกอัน)
            need = per_th.get(name_top, DEFAULT_THRESH)
            cover = nonzero_frames_ratio(x[0]) # กล้องเห็นคนตลอดไหม?

            # ต้องผ่านเกณฑ์ 3 ข้อถึงจะยอมรับคำตอบ
            passed = (conf_top >= need) and (margin >= TOP2_MARGIN) and (cover >= MIN_COVERAGE)

            if passed:
                if candidate_label == name_top: candidate_streak += 1
                else: candidate_label, candidate_streak = name_top, 1
                
                # ถ้ายืนยันคำตอบเดิมได้ครบ 5 เฟรม ถึงจะเอาขึ้นจอ (Anti-flicker UI)
                if candidate_streak >= STABLE_FRAMES:
                    result_data["shown_label"] = name_top
                    result_data["shown_conf"] = conf_top
            else:
                candidate_label, candidate_streak = None, 0
                result_data["shown_label"] = "Scanning..."
                result_data["shown_conf"] = 0.0

            # อัปเดตกราฟแท่งเสมอ แม้จะยังไม่ผ่านเกณฑ์ก็ตาม
            result_data["probs"] = smoothed

# เริ่ม Thread แบคกราวด์ (daemon=True หมายถึงถ้าโปรแกรมหลักปิด ไอ้หมอนี่ต้องโดนฆ่าทิ้งทันที)
ml_thread = threading.Thread(target=ml_worker, daemon=True)
ml_thread.start()

# ========== Main UI Thread (หน้าที่: ดึงภาพจากกล้อง โยนลงคิว และโชว์ UI ให้เร็วที่สุด) ==========
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
real_w, real_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ครอปภาพให้เป็นแนวตั้ง (อัตราส่วน 9:16) เพื่อจำลองหน้าจอมือถือ
target_w = int(real_h * (9 / 16)) 
start_x = (real_w - target_w) // 2
end_x = start_x + target_w

print(f"Started UI. Crop Region: x={start_x} to {end_x}")

prev_time = time.time()
fps = 0

while True:
    ret, raw_frame = cap.read()
    if not ret: break

    # คำนวณ FPS บนหน้าจอ (เนื่องจาก Thread นี้ทำหน้าที่แค่วาดภาพ FPS ตรงนี้ควรจะวิ่งเต็ม 30-60 FPS เสมอ)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # ครอปและพลิกภาพ
    raw_frame = cv2.flip(raw_frame, 1)
    frame = raw_frame[:, start_x:end_x]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # โยนภาพลงคิวให้ลูกน้อง (ML Worker) ไปจัดการต่อ
    if not frame_queue.full():
        frame_queue.put(rgb)

    # ดึงผลลัพธ์ล่าสุดจากตัวแปรแชร์ มาวาดทับลงบนภาพปัจจุบัน (ภาพลื่นไหล แต่กราฟแท่งจะอัปเดตตามสปีดของ ML)
    if result_data["probs"] is not None:
        draw_topk_bars(frame, labels, result_data["probs"], k=3, origin=(20, 130))
    
    draw_header(frame, result_data["shown_label"], result_data["shown_conf"], fps)

    # ปรับขนาดหน้าต่างแอปจำลองให้สูง 800px จะได้ไม่ล้นจอคอม
    disp_h = 800
    disp_w = int(target_w * (disp_h / real_h))
    cv2.imshow("App Simulator (FPS Optimized)", cv2.resize(frame, (disp_w, disp_h)))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup หลังกดปิดโปรแกรม
frame_queue.put(None) # ส่งยาพิษไปฆ่า ML Worker ในลูป
cap.release()
cv2.destroyAllWindows()