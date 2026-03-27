import os
import numpy as np
import random
import json
import math

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Conv1D,
    Bidirectional,
    GRU,
    LSTM,
    BatchNormalization,
    MaxPooling1D,
    SpatialDropout1D,
    GaussianNoise,
    Layer,
    GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping
)

# ---------------- 0) EXPERIMENT CONFIG ----------------
# ตั้งชื่อการทดลองให้สื่อความหมาย เพื่อความง่ายในการดูผลลัพธ์ใน TensorBoard
EXPERIMENT_NAME = "bi-gru-v1.2.5"
RNN_TYPE = "gru" # เลือกว่าจะใช้ GRU หรือ LSTM

# ตั้งค่า Hyperparameters หลักของโมเดล
CONV_FILTERS = 64  # จำนวนฟิลเตอร์ใน Convolution layer (สกัดฟีเจอร์เบื้องต้น)
CONV_KERNEL = 3    # ขนาดของหน้าต่าง Convolution (กวาดดูทีละ 3 เฟรม)
RNN_UNITS = 64     # จำนวน Cell ใน RNN (ความจำระยะสั้น-ยาว)
DENSE_UNITS1 = 64  # จำนวน Node ใน Fully Connected layer ก่อน Output
BATCH_SIZE = 64    # จำนวนคลิปวิดีโอต่อ 1 รอบการปรับน้ำหนัก (1 Batch)

SPATIAL_DROPOUT_RATE = 0.5 # โอกาสสุ่มปิดฟีเจอร์ทั้งเส้นข้ามเวลา (ใช้เฉพาะตอนเทรน)
DROPOUT_RATE = 0.5         # โอกาสสุ่มปิดโหนดธรรมดาก่อนออกผลลัพธ์

LEARNING_RATE = 1e-3       # ความเร็วในการเรียนรู้ (ก้าวเดินของ Optimizer)
NUM_EPOCHS = 100           # จำนวนรอบสูงสุดที่จะให้โมเดลดูข้อมูลทั้งหมด

# เปิดโหมด Class Balancing เพื่อแก้ปัญหาข้อมูลแต่ละท่ามีจำนวนคลิปไม่เท่ากัน
USE_BALANCED_SAMPLING = True

sequence_length = 30 # จำนวนเฟรมที่โมเดลรับเข้าไป
num_features = 258   # จำนวนพิกัด X,Y,Z ต่อเฟรม (Pose+LH+RH)

# ---------------- 1) CONFIG พื้นฐาน (🔥 ปรับ Path ใหม่) ----------------
# ใช้ระบบ Relative Path แบบถอยหลัง 2 ชั้น เพื่อหาโฟลเดอร์ Root ของโปรเจกต์
SRC_DIR = os.path.dirname(os.path.abspath(__file__))  # พิกัดปัจจุบัน (storage/src)
STORAGE_DIR = os.path.dirname(SRC_DIR)  # ถอย 1 ขั้นมาที่ (storage)
PROJECT_ROOT = os.path.dirname(STORAGE_DIR)  # ถอยอีก 1 ขั้นมาที่ Root

# ชี้เป้าไปที่โฟลเดอร์ Data ที่ถูกแบ่งสัดส่วนไว้แล้ว
DATA_DIR = os.path.join(PROJECT_ROOT, "data") 
TRAIN_DIR = os.path.join(DATA_DIR, "processed_train")
VAL_DIR = os.path.join(DATA_DIR, "processed_val")
TEST_DIR = os.path.join(DATA_DIR, "processed_test")

# เตรียมโฟลเดอร์สำหรับเซฟโมเดล (.keras) และเซฟผลการเทรน (TensorBoard)
MODEL_DIR = os.path.join(STORAGE_DIR, "models")  
LOG_BASE_DIR = os.path.join(PROJECT_ROOT, "logs")  

# รายชื่อท่าทางทั้งหมดที่โมเดลต้องเรียนรู้ (ลำดับตรงนี้จะกลายเป็น Index 0-9)
actions = np.array(
    [
        "anxiety",
        "fever",
        "feverish",
        "insomnia",
        "itching",
        "no_action",
        # "pain",  # โดนคอมเมนต์ทิ้ง แปลว่าท่านี้จะไม่ถูกนำมาเทรน
        "polyuria",
        "suffocated",
        "wounded",
    ]
)

# ล็อค Random Seed ทุกจุดที่เป็นไปได้ เพื่อให้ผลลัพธ์การรัน 
# ออกมาเหมือนเดิมทุกครั้ง (Reproducibility) ง่ายต่อการ Debug
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ---------------- 2) Helper Functions (โหลดข้อมูล) ----------------
def load_split(split_dir):
    """
    ฟังก์ชันสำหรับอ่านไฟล์ .npy จากโฟลเดอร์ย่อย (เช่น เข้าโฟลเดอร์ anxiety ดึงทุกคลิป)
    พร้อมกับแปลงชื่อคลาสให้กลายเป็นตัวเลข (Label Encoding) และ One-hot Encoding
    """
    sequences, labels = [], []
    action_map = {action: idx for idx, action in enumerate(actions)}
    print(f"\nLoading split from: {split_dir}")
    
    for action in actions:
        action_path = os.path.join(split_dir, action)
        if not os.path.isdir(action_path):
            print(f"  [WARN] Missing folder for action '{action}': {action_path}")
            continue
            
        # ดึงไฟล์ .npy ทั้งหมดและเรียงลำดับให้เป็นระเบียบ
        npy_files = [f for f in os.listdir(action_path) if f.endswith(".npy")]
        npy_files.sort()
        
        for npy_file in npy_files:
            npy_path = os.path.join(action_path, npy_file)
            res = np.load(npy_path)
            # ⚠️ เช็คความถูกต้องของ Shape ก่อนเอาเข้าโมเดล
            if res.shape == (sequence_length, num_features):
                sequences.append(res)
                labels.append(action_map[action])
                
    X = np.array(sequences)
    y = np.array(labels)
    # แปลงตัวเลข เช่น คลาส 2 ให้กลายเป็นอาเรย์ [0, 0, 1, 0, 0, ...] (One-hot)
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=len(actions))
    print(f"  -> Loaded {X.shape[0]} sequences from {split_dir}")
    return X, y_one_hot

print("Loading datasets (train / val / test)...")
# โหลดข้อมูลทั้ง 3 ส่วนขึ้นมาเก็บไว้ใน RAM ก่อนเลย
X_train, y_train = load_split(TRAIN_DIR)
X_val, y_val = load_split(VAL_DIR)
X_test, y_test = load_split(TEST_DIR)


# ---------------- 2.1) Data Generators ----------------
def data_generator(X_data, y_data, batch_size=32):
    """
    Generator ธรรมดา: สับไพ่และหยิบข้อมูลออกมาทีละ batch_size แบบสุ่มปกติ
    (ใช้สำหรับ Validation หรือ Test ที่ไม่ต้องการปรับสมดุลข้อมูล)
    """
    num_samples = X_data.shape[0]
    indices = np.arange(num_samples)
    while True:
        np.random.shuffle(indices) # สับไพ่ทุกๆ Epoch
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            # yield คือการพ่นข้อมูลออกไปให้โมเดล แล้วหยุดรอจนกว่าโมเดลจะขอใหม่
            yield X_data[batch_indices], y_data[batch_indices]


def balanced_data_generator(X_data, y_data, batch_size=32):
    """
    Balanced Generator: พระเอกของเรา!
    ใช้สำหรับ Train Set ป้องกันโมเดลอคติ (Bias) ไปหาคลาสที่มีคลิปเยอะ
    โดยจะบังคับหยิบคลาสขึ้นมาเฉลี่ยๆ กัน แล้วค่อยไปสุ่มดึงคลิปในคลาสนั้นมาอีกที
    """
    num_classes = y_data.shape[1]
    y_int = np.argmax(y_data, axis=1) # แปลง One-hot กลับเป็นตัวเลขเดี่ยว
    
    # ดึง Index ของข้อมูลแยกตามตะกร้า (แยกตามคลาส) เก็บไว้ล่วงหน้าเพื่อความเร็ว
    class_indices = [np.where(y_int == c)[0] for c in range(num_classes)]
    
    # กรองคลาสที่ไม่มีข้อมูลออก ป้องกัน Error ตอนสุ่ม (เช่น คลาส pain ที่โดนคอมเมนต์ไป)
    valid_classes = [c for c in range(num_classes) if len(class_indices[c]) > 0]
    
    while True:
        # 1. สุ่มเลือก "คลาส" มาให้ครบตามจำนวน Batch Size แบบให้ซ้ำได้ (replace=True)
        # วิธีนี้รับประกันว่าคลาสที่มีคลิปน้อย จะถูกสุ่มขึ้นมาบ่อยขึ้นเท่าเทียมกับคลาสคลิปเยอะ
        chosen_classes = np.random.choice(valid_classes, size=batch_size, replace=True)
        
        # 2. จากตะกร้าคลาสที่เลือกมา สุ่มจิ้มเอาคลิปขึ้นมา 1 คลิปต่อตะกร้า
        batch_idx = [np.random.choice(class_indices[c]) for c in chosen_classes]
        
        yield X_data[batch_idx], y_data[batch_idx]


# ---------------- 4) Model ----------------
def build_model():
    """
    สร้างสถาปัตยกรรม (Architecture) ของโมเดลด้วย Functional API
    1D-CNN (ดึงฟีเจอร์) -> Bi-GRU (จำลำดับเวลา) -> Dense (ตัดสินใจ)
    """
    # ระบุ Shape ของข้อมูลเข้าให้ชัดเจน (Batch size เป็น None คือรับกี่ก้อนก็ได้)
    inputs = Input(shape=(sequence_length, num_features), name="input_layer")

    # 1. Gaussian Noise: ฉีด Noise ปลอมๆ เข้าไปกวนโมเดล 
    # (ผมแนะนำให้พิจารณาเอาออกตามที่เคยคุยกันเรื่อง TFLite Compatibility นะครับ)
    x = GaussianNoise(0.05, name="noise_injection")(inputs)

    # 2. Conv1D Block: ทำหน้าที่เป็น "แว่นขยาย" สแกนดูการขยับสั้นๆ ติดๆ กันทีละ 3 เฟรม
    x = Conv1D(
        filters=CONV_FILTERS,
        kernel_size=CONV_KERNEL,
        activation="relu",
        padding="same", # ให้ Output มีความยาวเฟรม 30 เท่าเดิม (ไม่หด)
        kernel_regularizer=regularizers.l2(0.002), # ป้องกัน Overfitting (L2 Penalty)
        name="conv1d_feature",
        use_bias=False # ปิด bias เพราะเราจะใช้ BatchNorm ในชั้นถัดไปอยู่แล้ว (Best practice)
    )(x)
    # BatchNorm ช่วยปรับสเกลข้อมูลให้กลางๆ โมเดลจะเทรนได้เร็วและเสถียรขึ้น
    x = BatchNormalization(name="batch_norm")(x)
    # SpatialDropout ปิดฟีเจอร์ทั้งเส้นข้าม 30 เฟรม บังคับโมเดลไม่ให้พึ่งฟีเจอร์ใดฟีเจอร์หนึ่งมากไป
    x = SpatialDropout1D(SPATIAL_DROPOUT_RATE, name="spatial_dropout")(x)
    # MaxPooling1D(2) ยุบความยาวเวลาลงครึ่งหนึ่ง (จาก 30 เฟรม เหลือ 15 เฟรม) เพื่อสกัดแต่ข้อมูลเด่นๆ
    x = MaxPooling1D(pool_size=2, name="max_pool")(x)

    # 3. RNN Block: ใช้ Bidirectional ทำให้โมเดลเห็นทั้ง "อดีต->อนาคต" และ "อนาคต->อดีต"
    rnn_layer_cls = GRU if RNN_TYPE.lower() == "gru" else LSTM
    # return_sequences=True เพราะเรายังอยากได้ข้อมูลที่เป็นมิติเวลาอยู่ (ยังไม่สรุปผล)
    x = Bidirectional(rnn_layer_cls(RNN_UNITS, return_sequences=True, unroll=False), name="bi_rnn")(x)

    # 4. GlobalAveragePooling1D: ยุบแกนเวลา (Time) ทั้งหมดทิ้งไป เอาค่าเฉลี่ยมา 
    # กลายเป็น Vector เดี่ยวๆ 1 เส้น เพื่อส่งให้ Dense layer ถัดไป
    x = GlobalAveragePooling1D(name="global_avg_pool")(x)

    # 5. Dense Block: ทำหน้าที่ "สรุปผล" จาก Vector ที่ได้
    x = Dense(
        DENSE_UNITS1,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.002),
        name="dense_hidden",
    )(x)
    # Dropout ตัดการเชื่อมต่อประสาทแบบสุ่ม ป้องกันการท่องจำ
    x = Dropout(DROPOUT_RATE, name="dropout_final")(x)

    # Output Layer: แปลงผลลัพธ์สุดท้ายให้เป็นความน่าจะเป็น (Probability) 0-1 รวมกันได้ 1 (Softmax)
    outputs = Dense(len(actions), activation="softmax", dtype="float32", name="output_layer")(x)

    # ห่อทุกอย่างเป็น Model Object
    model = Model(inputs=inputs, outputs=outputs, name=EXPERIMENT_NAME)

    # Loss: ใช้ Crossentropy แต่ผสม label_smoothing=0.1 
    # หมายถึงมันจะไม่บังคับให้ทายถูกเป๊ะ 100% (เป้าหมาย 1 กลายเป็น 0.9) ช่วยลด Overfitting ได้ดีมาก
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    
    # ผูก Optimizer (Adam) เข้ากับโมเดล พร้อมตั้งเป้าดู Accuracy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model


# ---------------- 5) Training Process ----------------
print(f"\n--- Starting Training (Experiment: {EXPERIMENT_NAME}) ---")
print(f"Config: Filters={CONV_FILTERS}, RNN={RNN_UNITS}, Dense={DENSE_UNITS1}, Drop={DROPOUT_RATE}")

# เรียกคำสั่งสร้างโมเดล แล้วปรินท์โครงสร้าง (Shape ย่อ-ขยายยังไง) ให้ดู
model = build_model()
model.summary()

# Setup Paths & Callbacks
log_dir = os.path.join(LOG_BASE_DIR, EXPERIMENT_NAME)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")

# วาดรูปแผนผังสถาปัตยกรรมเก็บไว้
print("\n[INFO] Plotting model architecture...")
try:
    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(MODEL_DIR, "model_architecture.png"),
        show_shapes=True,       # สำคัญ: ทำให้เห็นเลยว่า Tensor แปลงร่างจาก 3D เป็น 2D ตรงไหน
        show_layer_names=True,  
        dpi=96                  
    )
    print(f"[OK] Model architecture saved to {MODEL_DIR}\\model_architecture.png")
except Exception as e:
    # ถ้าขึ้นเตือนตรงนี้ มักแปลว่าคอมคุณยังไม่ลงโปรแกรม Graphviz (ข้ามได้ ไม่กระทบการเทรน)
    print(f"[WARN] Could not plot model. Ensure Graphviz is in PATH. Error: {e}")

# สร้าง Callbacks 4 ตัวคอยดูแลการเทรน
callbacks_list = [
    # TensorBoard: อัดกราฟผลลัพธ์เอาไปโชว์สวยๆ
    TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,       
        write_graph=True,       
        write_images=True,      
        update_freq='epoch',    
        profile_batch=0         
    ),
    # ModelCheckpoint: คอยจับตาดูว่า val_accuracy รอบไหนสูงสุด ก็เซฟทับไฟล์เดิม (เก็บแต่ตัวเก่งสุด)
    ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max",
    ),
    # ReduceLROnPlateau: ถ้าเทรนไป 5 รอบแล้วยังไม่เก่งขึ้น (Plateau) ให้ลดก้าวเดิน (LR) ลงครึ่งนึง (0.5)
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    ),
    # EarlyStopping: ถ้ารอมา 10 รอบแล้ว Loss ไม่ลดเลย ให้หยุดเทรนเลย ประหยัดไฟ!
    EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    ),
]


# 🔥 Function สำหรับหุ้ม Generator ให้เป็น tf.data.Dataset
def get_dataset_pipeline(X, y, is_training=True, batch_size=32):
    """
    แปลง Generator ฝั่ง Python ให้กลายเป็นท่อข้อมูล (Pipeline) ฝั่ง C++ (TensorFlow)
    เพื่อดึงประสิทธิภาพ Multithreading ออกมาสูงสุด
    """
    # กำหนดลักษณะของข้อมูลที่ท่อนี้จะพ่นออกมา (เพื่อไม่ให้ TensorFlow สับสน)
    output_sig = (
        tf.TensorSpec(shape=(None, sequence_length, num_features), dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(actions)), dtype=tf.float32),
    )

    # เลือกใช้ Balanced ตอนเทรน นอกนั้น (Val) ใช้สุ่มธรรมดา
    gen_func = (
        balanced_data_generator
        if (is_training and USE_BALANCED_SAMPLING)
        else data_generator
    )

    # หุ้ม Generator สร้างเป็น Dataset
    ds = tf.data.Dataset.from_generator(
        lambda: gen_func(X, y, batch_size), output_signature=output_sig
    )
    
    # ถ้าเป็นรอบ Train ให้เติม Augment เล็กๆ (ฉีด Noise สด)
    if is_training:
        def soft_augment(features, labels):
            noise = tf.random.normal(shape=tf.shape(features), mean=0.0, stddev=0.005, dtype=tf.float32)
            augmented_features = features + noise
            return augmented_features, labels
            
        # ใช้ AUTOTUNE เพื่อให้ TF ไปคำนวณเอาเองว่าต้องแบ่งไปให้ CPU กี่ Core ทำงานถึงจะเร็วสุด
        ds = ds.map(soft_augment, num_parallel_calls=tf.data.AUTOTUNE)

    # 🔥 Prefetch (เตรียมข้อมูลรอไว้ใน Buffer)
    # สมมติ GPU กำลังเรียน Batch 1 อยู่, CPU จะเตรียมดึง Batch 2 รอไว้เลย ทำให้ GPU ไม่ต้องว่างงาน
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

print(f"[INFO] Preparing {'BALANCED' if USE_BALANCED_SAMPLING else 'NORMAL'} training pipeline...")
train_ds = get_dataset_pipeline(X_train, y_train, is_training=True, batch_size=BATCH_SIZE)
val_ds = get_dataset_pipeline(X_val, y_val, is_training=False, batch_size=BATCH_SIZE)

# คำนวณว่า 1 Epoch ต้องเดินกี่ก้าว (Steps)
steps_per_epoch = max(1, math.ceil(len(X_train) / BATCH_SIZE))
validation_steps = max(1, math.ceil(len(X_val) / BATCH_SIZE))

# 🚀 ปล่อยจรวด! เริ่มเทรน (ครอบ Try-Except ไว้ เผื่อเรากด Ctrl+C กลางคัน จะได้เซฟโมเดลไว้ทัน)
try:
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=NUM_EPOCHS,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1,
    )
except Exception as e:
    print(f"\n[ERROR] Training interrupted: {e}")
    # กู้ภัย: เซฟตัวล่าสุดไว้ก่อนพัง
    model.save(os.path.join(MODEL_DIR, "interrupted_model.keras"))


# ---------------- 6) Evaluation & Save ----------------
print("\nEvaluating Model on TEST set...")
# เอาข้อสอบปลายภาค (Test Set) มาให้โมเดลทำ
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)
print(f"Test loss = {test_loss:.4f}, Test accuracy = {test_acc:.4f}")

# เซฟโมเดลสุดท้าย
model.save(os.path.join(MODEL_DIR, "final_model.keras"))
print(f"Final model saved to {os.path.join(MODEL_DIR, 'final_model.keras')}")

# Save Label Map (เพื่อให้แอป Android/iOS รู้ว่า 0 คืออะไร 1 คืออะไร)
label_map = {i: action for i, action in enumerate(actions)}
with open(os.path.join(MODEL_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

# Threshold Calibration (จูนค่าความมั่นใจ)
print("\nCalibrating thresholds...")
# สั่งทำนาย Validation Set ทั้งยวง แล้วเอาความน่าจะเป็น (probs) มาวิเคราะห์หาจุดตัด (Threshold) ที่ดีที่สุด
probs = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
y_true = np.argmax(y_val, axis=1)

thresholds = {}
for c, name in enumerate(actions):
    y_true_c = (y_true == c).astype(int)
    best_f1, best_th = -1, 0.5
    
    # ลองสแกนค่า Threshold ตั้งแต่ 0.3 ถึง 0.9 (สเตปละ 0.01)
    for th in np.linspace(0.3, 0.9, 61):
        # ถ้าความน่าจะเป็น มากกว่า Threshold ถือว่าทายว่าเป็นคลาสนี้ (1)
        y_pred_c = (probs[:, c] >= th).astype(int)
        
        # คำนวณตัวชี้วัด
        tp = np.sum((y_pred_c == 1) & (y_true_c == 1)) # ทายว่าใช่ และของจริงก็ใช่
        fp = np.sum((y_pred_c == 1) & (y_true_c == 0)) # ทายว่าใช่ แต่ของจริงไม่ใช่ (มั่ว)
        fn = np.sum((y_pred_c == 0) & (y_true_c == 1)) # ทายว่าไม่ใช่ แต่ของจริงดันใช่ (หลุดรอด)
        
        # ค้นหาค่า F1-Score (จุดสมดุลระหว่างความแม่นยำและการกวาดจับ) สูงสุด
        f1 = 0.0 if (tp + fp == 0 or tp + fn == 0) else (2 * tp) / (2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_th = f1, th
            
    # บันทึกค่า Thresh ที่ดีที่สุดของแต่ละท่าเก็บไว้
    thresholds[name] = {"threshold": float(best_th), "f1": float(best_f1)}
    print(f"  {name}: Th={best_th:.2f}, F1={best_f1:.4f}")

# เซฟค่า Threshold ลงไฟล์ JSON เพื่อส่งให้ฝั่งแอปพลิเคชันไปตั้งค่ากรองผลลัพธ์
with open(os.path.join(MODEL_DIR, "thresholds.json"), "w", encoding="utf-8") as f:
    json.dump(thresholds, f, ensure_ascii=False, indent=4)
print("Done.")