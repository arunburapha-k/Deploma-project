import cv2
import os
import time
import numpy as np

# --- 1. การตั้งค่า (Config) ---
RAW_DATA_PATH = os.path.join('data', 'raw') 

# ชื่อท่าทางที่ต้องการเก็บ
actions = np.array(['suffocated_r2']) 

no_sequences = 20     
sequence_length = 90  
start_delay = 2       

# สร้างโฟลเดอร์
for action in actions:
    try:
        os.makedirs(os.path.join(RAW_DATA_PATH, action))
    except:
        pass

# --- 2. เริ่มเปิดกล้อง ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# ตั้งค่ากล้องให้ละเอียดที่สุดในแนวนอน (เพื่อให้ Crop แล้วยังชัด)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# อ่านค่าจริงที่กล้องให้มา
real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Original Camera Resolution: {real_w}x{real_h}")

# --- 🔥 คำนวณขนาดการ Crop (9:16) 🔥 ---
# เราจะใช้ความสูงเต็ม (h) แล้วคำนวณความกว้างใหม่ (new_w)
# สูตร: new_w = h * (9/16)
target_h = real_h
target_w = int(real_h * (9/16)) 

# คำนวณจุดเริ่มต้นตัดภาพ (เพื่อให้ภาพอยู่ตรงกลาง)
start_x = (real_w - target_w) // 2
end_x = start_x + target_w

print(f"Target Video Resolution (Cropped): {target_w}x{target_h}")

print("--- Starting Data Collection (Center Crop 9:16) ---")
print("Press 'q' to quit early.")

for action in actions:
    print(f"Collecting data for action: {action}")
    print("Get Ready! Starting in 4 seconds...")
    time.sleep(4) 
    
    for sequence in range(no_sequences):
        save_path = os.path.join(RAW_DATA_PATH, action, f'{action}_{sequence}.mp4')
        
        # ตั้งค่า VideoWriter เป็นขนาดที่ Crop แล้ว
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 30.0, (target_w, target_h))
        
        frames_captured = 0
        while frames_captured < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- 🔥 ตัดภาพ (Crop) แทนการหมุน 🔥 ---
            # ตัดเอาเฉพาะตรงกลางภาพ
            frame_vertical = frame[:, start_x:end_x]
            
            # (Optional) กลับด้านซ้ายขวาให้เหมือนกระจก (ถ้าต้องการ)
            frame_vertical = cv2.flip(frame_vertical, 1)

            # เขียนเฟรมลงไฟล์
            out.write(frame_vertical)
            
            # --- ส่วนแสดงผล (GUI) ---
            display_frame = frame_vertical.copy()
            
            cv2.putText(display_frame, f'Action: {action}', (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f'Video: {sequence}/{no_sequences}', (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f'Frame: {frames_captured}/{sequence_length}', (20, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            
            # แสดงภาพ
            cv2.imshow('Data Collection (Cropped 9:16)', display_frame)
            
            frames_captured += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                out.release()
                cap.release()
                cv2.destroyAllWindows()
                exit()
        
        out.release()
        
        # --- ช่วงพัก (Break) ---
        start_time = time.time()
        while (time.time() - start_time) < start_delay:
            ret, frame = cap.read()
            # Crop ภาพตอนพักด้วย
            frame_vertical = frame[:, start_x:end_x]
            
            cv2.putText(frame_vertical, 'WAIT...', (50, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(frame_vertical, f'Next: {action}', (50, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Data Collection (Cropped 9:16)', frame_vertical)
            cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
print("All data collected!")