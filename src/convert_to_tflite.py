import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
import tensorflow.keras.backend as K
import time

# ==========================================================
# 🔥 1. จัดการ Path ให้ถูกต้อง
# ==========================================================
# ถอยหลังโฟลเดอร์เพื่อหา Project Root (เหมือนกับไฟล์อื่นๆ ในโปรเจกต์)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))        
STORAGE_DIR = os.path.dirname(SRC_DIR)                     
PROJECT_ROOT = os.path.dirname(STORAGE_DIR)                

MODEL_DIR = os.path.join(STORAGE_DIR, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "processed_train")      

# ตั้งชื่อไฟล์ผลลัพธ์ที่จะได้หลังจากการแปลง (มี 3 แบบตามระดับการบีบอัด)
KERAS_MODEL = os.path.join(MODEL_DIR, "final_model.keras") 
OUT_FP32 = os.path.join(MODEL_DIR, "model_fp32.tflite")    # ไม่บีบอัด
OUT_FP16 = os.path.join(MODEL_DIR, "model_fp16.tflite")    # บีบอัดเหลือครึ่งนึง (เหมาะกับ GPU มือถือ)
OUT_INT8 = os.path.join(MODEL_DIR, "model_int8.tflite")    # บีบอัด 4 เท่า (เหมาะกับ CPU มือถือ)

print(f"TF Version: {tf.__version__}")
tf.keras.backend.clear_session() # ล้างหน่วยความจำกราฟเก่าๆ ทิ้งก่อนเริ่มงาน

# ----------------- 3. โหลดโมเดลพร้อม Custom Objects -----------------
print("\n[INFO] Loading Keras model...")
try:
    # 🔥 หัวใจสำคัญ: ใช้ compile=False 
    # ตอนเราเอาโมเดลไปรันบนมือถือ เราไม่ได้จะ 'เทรน' มันแล้ว (ไม่ต้องการ Optimizer อย่าง Adam หรือ Loss)
    # การโหลดแบบ compile=False จะช่วยลดขนาด Memory และเลี่ยง Error ตัวแปรแปลกๆ ได้
    model = tf.keras.models.load_model(KERAS_MODEL, compile=False)
    print("[INFO] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    print("[ADVICE] หากยังพบ Error เรื่อง Variables ให้กลับไปตรวจสอบว่าคลาส Attention ตอนเทรนใช้ชื่อ Weight เดียวกันหรือไม่")
    exit(1)

# ----------------- 4. เตรียมฟังก์ชัน Serving -----------------
# 🔥 นี่คือสุดยอด Best Practice: การตรึง Shape (Freezing)
# มือถือรันแอปแบบ Real-time ทีละคลิป ดังนั้น Batch Size ต้องเป็น '1' เสมอ
# การบังคับ Shape [1, 30, 258] ตรงนี้ จะทำให้ TFLite รู้ตัวล่วงหน้าและจัดสรร Memory ได้เร็วที่สุด 
# (ถ้าปล่อยให้เป็น [None, 30, 258] มือถือจะต้องเสียเวลา Allocate Memory ใหม่ทุกเฟรม แอปจะกระตุก)
@tf.function(
    input_signature=[tf.TensorSpec(shape=[1, 30, 258], dtype=tf.float32, name="input")]
)
def serving(x):
    # training=False สำคัญมาก เพราะจะทำให้ Dropout ปิดตัวลง และ BatchNorm ใช้ค่า Mean/Variance คงที่
    return model(x, training=False)

# ดึงกราฟคณิตศาสตร์แบบสมบูรณ์ (Concrete Function) ออกมาจาก Keras Model
concrete_func = serving.get_concrete_function()

# ----------------- 5. ฟังก์ชันแปลง TFLite (Optimized) -----------------
def convert_and_save(converter, output_path, mode_name):
    """ฟังก์ชันผู้ช่วยสำหรับสั่ง Convert และ Save ไฟล์ลงฮาร์ดดิสก์"""
    print(f"\n[INFO] Converting to {mode_name}...")
    try:
        tflite_model = converter.convert()
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        print(f"[OK] Saved {mode_name} to: {output_path}")
        return output_path
    except Exception as e:
        print(f"[ERROR] {mode_name} Conversion failed: {e}")
        return None

# --- Setup Converter ---
# ใช้ from_concrete_functions เพราะเสถียรที่สุดสำหรับโมเดลที่มี RNN/LSTM/GRU
conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
# อนุญาตให้ใช้ Operation พื้นฐานของ TFLite (C++ Kernels ที่ถูกปรับแต่งมาเพื่อมือถือ)
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# แบบที่ 1. FP32 (Original)
# ขนาดไฟล์ใหญ่สุด (เท่าต้นฉบับ) ความแม่นยำ 100% แต่บนมือถืออาจจะรันช้ากว่าแบบอื่น
convert_and_save(conv, OUT_FP32, "FP32")

# แบบที่ 2. FP16 (แนะนำสุดสำหรับ Mobile)
# แปลงค่าน้ำหนักจาก Float32 -> Float16 ไฟล์เล็กลง 50% ความแม่นยำแทบไม่ตกเลย
# และมือถือรุ่นใหม่ๆ ที่มี NPU/GPU จะประมวลผล Float16 ได้ไวกว่า Float32 มาก
conv_fp16 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
conv_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
conv_fp16.target_spec.supported_types = [tf.float16]
convert_and_save(conv_fp16, OUT_FP16, "FP16")

# แบบที่ 3. Dynamic Range (INT8 Weights) 
# บีบค่าน้ำหนักให้เป็นจำนวนเต็ม 8-bit (เล็กลง 4 เท่า!) แต่ตอนรันจริงจะตีกลับเป็น Float โง่ๆ (Dynamic)
# วิธีนี้ "ปลอดภัยที่สุดสำหรับโมเดลสาย RNN" เพราะถ้าบีบเป็น INT8 100% ค่า Hidden State ในลูปมันมักจะเพี้ยนจนทายผิดหมด
conv_dyn = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
conv_dyn.optimizations = [tf.lite.Optimize.DEFAULT]
convert_and_save(conv_dyn, OUT_INT8, "INT8 Dynamic")

# ----------------- 6. ตรวจสอบไฟล์ผลลัพธ์ -----------------
print("\n[INFO] Validating generated TFLite models...")
# ลองโหลดไฟล์ TFLite ขึ้นมาจริงๆ ถ้าระบบ Allocate Tensors ผ่าน แปลว่าเอาไปยัดลง Android/iOS แล้วแอปจะไม่เด้ง Crash แน่นอน
for p in [OUT_FP32, OUT_FP16, OUT_INT8]:
    if os.path.exists(p):
        try:
            interp = tf.lite.Interpreter(model_path=p)
            interp.allocate_tensors()
            print(f"  [OK] {os.path.basename(p):<18} | Status: Ready for Mobile Deployment")
        except Exception as e:
            print(f"  [WARN] {os.path.basename(p):<18} | Status: Potential Issue ({e})")


# ==========================================================
# 7. Sanity Check: ทดสอบความเร็วและความแม่นยำจำลองบน TFLite
# ==========================================================
def run_tflite_inference(tflite_path, input_data):
    """โหลดโมเดลเข้า Interpreter เพื่อจำลองสภาพแวดล้อมการทำงานบนมือถือ และจับเวลา"""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # หาตำแหน่งช่องเสียบข้อมูล (Input/Output Index)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # ยัดข้อมูลจำลองเข้าไปในโมเดล
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # จับเวลาและสั่งให้โมเดลประมวลผล (Inference)
    start_time = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start_time) * 1000 # แปลงหน่วยวินาทีเป็น Milliseconds (ms)
    
    # ดึงคำตอบความน่าจะเป็นของแต่ละคลาสออกมา
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data, inference_time

print("\n[INFO] Running Sanity Check (Speed & Prediction matching)...")
# สร้างข้อมูลจำลอง (Dummy Sequence) ขนาด Batch=1, Seq=30, Features=258 เป็นค่าสุ่ม
dummy_input = np.random.randn(1, 30, 258).astype(np.float32)

# ทดสอบรันกับ Keras ต้นฉบับเพื่อเอาผลลัพธ์มาเป็น "มาตรฐาน (Ground Truth)"
keras_pred = model.predict(dummy_input, verbose=0)
print(f" -> Keras Original Output : {np.argmax(keras_pred)} (Confidence: {np.max(keras_pred):.4f})")

# ทดสอบรัน TFLite แต่ละเวอร์ชัน เพื่อดูว่าการบีบอัดทำให้มัน "โง่ลง" หรือ "ช้าลง" ไหม
for tflite_file in [OUT_FP32, OUT_FP16, OUT_INT8]:
    if os.path.exists(tflite_file):
        # 🔥 วอร์มอัปเครื่องก่อนจับเวลา: โมเดล AI มักจะรันช้าในเฟรมแรกเสมอ (Initialization overhead) 
        # การรันทิ้งเปล่า 1 รอบ จะช่วยให้การจับเวลารอบถัดไป สะท้อนความเร็วตอนใช้งานจริงได้อย่างแม่นยำ
        _, _ = run_tflite_inference(tflite_file, dummy_input) 
        
        # ทดสอบจริง (จับเวลาที่แท้จริง)
        tflite_pred, latency = run_tflite_inference(tflite_file, dummy_input)
        
        # หาว่ามันทายตรงกับ Keras ต้นฉบับไหม (Mean Absolute Error)
        # ถ้ายิ่งเข้าใกล้ 0 แปลว่าการบีบอัดไฟล์แทบไม่ส่งผลเสียต่อความแม่นยำเลย
        error_diff = np.mean(np.abs(keras_pred - tflite_pred))
        
        print(f" -> {os.path.basename(tflite_file):<18} | Pred: {np.argmax(tflite_pred)} | Diff Error: {error_diff:.6f} | Latency: {latency:.2f} ms")

# คำแนะนำปิดท้ายสำหรับวิศวกรฝั่ง Application
print("\n[ADVICE] หาก Diff Error ของ INT8 สูงเกินไป (เช่น > 0.1) แนะนำให้ใช้ FP16 ในการ Deploy ครับ")
print("\n--- All Processes Completed ---")