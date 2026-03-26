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
SRC_DIR = os.path.dirname(os.path.abspath(__file__))        
STORAGE_DIR = os.path.dirname(SRC_DIR)                     
PROJECT_ROOT = os.path.dirname(STORAGE_DIR)                

MODEL_DIR = os.path.join(STORAGE_DIR, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "processed_train")      

KERAS_MODEL = os.path.join(MODEL_DIR, "final_model.keras") 
OUT_FP32 = os.path.join(MODEL_DIR, "model_fp32.tflite")
OUT_FP16 = os.path.join(MODEL_DIR, "model_fp16.tflite")
OUT_INT8 = os.path.join(MODEL_DIR, "model_int8.tflite")

print(f"TF Version: {tf.__version__}")
tf.keras.backend.clear_session()
# ----------------- 3. โหลดโมเดลพร้อม Custom Objects -----------------
print("\n[INFO] Loading Keras model...")
try:
    # 🔥 หัวใจสำคัญ: ใช้ compile=False เพื่อเลี่ยง Error ของ Adam Optimizer
    model = tf.keras.models.load_model(
        KERAS_MODEL,
    )
    print("[INFO] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    print("[ADVICE] หากยังพบ Error เรื่อง Variables ให้กลับไปตรวจสอบว่าคลาส Attention ตอนเทรนใช้ชื่อ Weight เดียวกันหรือไม่")
    exit(1)

# ----------------- 4. เตรียมฟังก์ชัน Serving -----------------
# บังคับ Batch Size เป็น 1 เพื่อให้เหมาะกับการใช้งานจริงบนมือถือ (Real-time)
@tf.function(
    input_signature=[tf.TensorSpec(shape=[1, 30, 258], dtype=tf.float32, name="input")]
)
def serving(x):
    return model(x, training=False)

concrete_func = serving.get_concrete_function()

# ----------------- 5. ฟังก์ชันแปลง TFLite (Optimized) -----------------
def convert_and_save(converter, output_path, mode_name):
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
conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# 1. FP32 (Original)
convert_and_save(conv, OUT_FP32, "FP32")

# 2. FP16 (แนะนำสำหรับมือถือรุ่นใหม่ๆ จะรันเร็วมากบน GPU Delegate)
conv_fp16 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
conv_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
conv_fp16.target_spec.supported_types = [tf.float16]
convert_and_save(conv_fp16, OUT_FP16, "FP16")

# 3. Dynamic Range (INT8 Weights) - ปลอดภัยที่สุดสำหรับ RNN/GRU
conv_dyn = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
conv_dyn.optimizations = [tf.lite.Optimize.DEFAULT]
convert_and_save(conv_dyn, OUT_INT8, "INT8 Dynamic")

# ----------------- 6. ตรวจสอบไฟล์ผลลัพธ์ -----------------
print("\n[INFO] Validating generated TFLite models...")
for p in [OUT_FP32, OUT_FP16, OUT_INT8]:
    if os.path.exists(p):
        try:
            interp = tf.lite.Interpreter(model_path=p)
            interp.allocate_tensors()
            print(f" [OK] {os.path.basename(p):<18} | Status: Ready for Mobile Deployment")
        except Exception as e:
            print(f" [WARN] {os.path.basename(p):<18} | Status: Potential Issue ({e})")


# ==========================================================
# 7. Sanity Check: ทดสอบความเร็วและความแม่นยำจำลองบน TFLite
# ==========================================================
def run_tflite_inference(tflite_path, input_data):
    # โหลดโมเดลจำลองสภาพแวดล้อมมือถือ
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # หาตำแหน่งช่องเสียบข้อมูล (Input/Output Index)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # ยัดข้อมูลจำลอง (Dummy data) เข้าไป
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # จับเวลาและสั่งให้โมเดลประมวลผล (Inference)
    start_time = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start_time) * 1000 # หน่วยเป็น Milliseconds
    
    # ดึงคำตอบออกมา
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data, inference_time

print("\n[INFO] Running Sanity Check (Speed & Prediction matching)...")
# สร้างข้อมูลจำลอง (Dummy Sequence) ขนาด Batch=1, Seq=30, Features=258
dummy_input = np.random.randn(1, 30, 258).astype(np.float32)

# ทดสอบ Keras ต้นฉบับ
keras_pred = model.predict(dummy_input, verbose=0)
print(f" -> Keras Original Output : {np.argmax(keras_pred)} (Confidence: {np.max(keras_pred):.4f})")

# ทดสอบ TFLite แต่ละเวอร์ชัน
for tflite_file in [OUT_FP32, OUT_FP16, OUT_INT8]:
    if os.path.exists(tflite_file):
        # วอร์มอัปเครื่องก่อนจับเวลา (เหมือนแอปเพิ่งเปิด)
        _, _ = run_tflite_inference(tflite_file, dummy_input) 
        
        # ทดสอบจริง
        tflite_pred, latency = run_tflite_inference(tflite_file, dummy_input)
        
        # หาว่ามันทายตรงกับ Keras ต้นฉบับไหม (Mean Absolute Error ควรเข้าใกล้ 0)
        error_diff = np.mean(np.abs(keras_pred - tflite_pred))
        
        print(f" -> {os.path.basename(tflite_file):<18} | Pred: {np.argmax(tflite_pred)} | Diff Error: {error_diff:.6f} | Latency: {latency:.2f} ms")

print("\n[ADVICE] หาก Diff Error ของ INT8 สูงเกินไป (เช่น > 0.1) แนะนำให้ใช้ FP16 ในการ Deploy ครับ")

print("\n--- All Processes Completed ---")
