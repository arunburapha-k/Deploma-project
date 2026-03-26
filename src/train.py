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
EXPERIMENT_NAME = "bi-gru-v1.2.4"
RNN_TYPE = "gru"

# --- FIXED HYPERPARAMETERS (🔥 ปรับลดขนาดสมองตาม Requirement 5) ---
CONV_FILTERS = 128     # คงไว้เท่าเดิม ประหยัดแบตเตอรี่ตอนกรองข้อมูล
CONV_KERNEL = 3       
RNN_UNITS = 64        # 🔥 เพิ่มความจำ (จาก 32 -> 64) คุ้มมาก!
DENSE_UNITS1 = 32     # คงไว้ ป้องกันสมองบวมท้ายๆ
SPATIAL_DROPOUT_RATE = 0.5
DROPOUT_RATE = 0.5

LEARNING_RATE = 1e-3 
NUM_EPOCHS = 300       # 🔥 แนะนำให้เพิ่ม Epoch เพราะโมเดลสมองใหญ่ขึ้น ต้องใช้เวลาเทรนขึ้นอีกนิด
BATCH_SIZE = 128

# class balancing
USE_BALANCED_SAMPLING = True

sequence_length = 30
num_features = 258

# ---------------- 1) CONFIG พื้นฐาน (🔥 ปรับ Path ใหม่) ----------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))  # storage/src
STORAGE_DIR = os.path.dirname(SRC_DIR)  # storage
PROJECT_ROOT = os.path.dirname(STORAGE_DIR)  # Project Root (ระดับเดียวกับ data)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")  # พุ่งไปที่ data ด้านนอก
TRAIN_DIR = os.path.join(DATA_DIR, "processed_train")
VAL_DIR = os.path.join(DATA_DIR, "processed_val")
TEST_DIR = os.path.join(DATA_DIR, "processed_test")

MODEL_DIR = os.path.join(STORAGE_DIR, "models")  # เก็บโมเดลไว้ใน storage/models
LOG_BASE_DIR = os.path.join(PROJECT_ROOT, "logs")  # เก็บ logs ไว้ใน storage/logs

actions = np.array(
    [
        "anxiety",
        "fever",
        "feverish",
        "insomnia",
        "itching",
        "no_action",
        "pain",
        "polyuria",
        "suffocated",
        "wounded",
    ]
)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------- 2) Helper Functions (โหลดข้อมูล) ----------------
def load_split(split_dir):
    sequences, labels = [], []
    action_map = {action: idx for idx, action in enumerate(actions)}
    print(f"\nLoading split from: {split_dir}")
    for action in actions:
        action_path = os.path.join(split_dir, action)
        if not os.path.isdir(action_path):
            print(f"  [WARN] Missing folder for action '{action}': {action_path}")
            continue
        npy_files = [f for f in os.listdir(action_path) if f.endswith(".npy")]
        npy_files.sort()
        for npy_file in npy_files:
            npy_path = os.path.join(action_path, npy_file)
            res = np.load(npy_path)
            if res.shape == (sequence_length, num_features):
                sequences.append(res)
                labels.append(action_map[action])
    X = np.array(sequences)
    y = np.array(labels)
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=len(actions))
    print(f"  -> Loaded {X.shape[0]} sequences from {split_dir}")
    return X, y_one_hot


print("Loading datasets (train / val / test)...")
X_train, y_train = load_split(TRAIN_DIR)
X_val, y_val = load_split(VAL_DIR)
X_test, y_test = load_split(TEST_DIR)


# ---------------- 2.1) Data Generators ----------------
def data_generator(X_data, y_data, batch_size=32):
    num_samples = X_data.shape[0]
    indices = np.arange(num_samples)
    while True:
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            yield X_data[batch_indices], y_data[batch_indices]


def balanced_data_generator(X_data, y_data, batch_size=32):
    num_classes = y_data.shape[1]
    y_int = np.argmax(y_data, axis=1)
    
    # ดึง Index ของแต่ละคลาสเก็บไว้ล่วงหน้า
    class_indices = [np.where(y_int == c)[0] for c in range(num_classes)]
    # กรองคลาสที่ไม่มีข้อมูลออก ป้องกัน Error
    valid_classes = [c for c in range(num_classes) if len(class_indices[c]) > 0]
    
    while True:
        # 1. สุ่มเลือก Class มาให้ครบตามจำนวน Batch Size แบบมีสับเปลี่ยน
        chosen_classes = np.random.choice(valid_classes, size=batch_size, replace=True)
        
        # 2. สุ่ม Index ของข้อมูลในแต่ละ Class ที่ถูกเลือก (Vectorized approach)
        # รวดเร็วกว่าการใช้ List .append() หลายเท่าตัว
        batch_idx = [np.random.choice(class_indices[c]) for c in chosen_classes]
        
        yield X_data[batch_idx], y_data[batch_idx]
# ---------------- 4)Model ----------------
def build_model():
    # ระบุ Shape ให้ชัดเจน (None สำหรับ Batch Size, ที่เหลือต้องระบุตายตัว)
    inputs = Input(shape=(sequence_length, num_features), name="input_layer")

    # 1. Gaussian Noise (ป้องกัน Overfitting จาก Sensor Noise)
    x = GaussianNoise(0.05, name="noise_injection")(inputs)

    # 2. Conv1D Block
    x = Conv1D(
        filters=CONV_FILTERS,
        kernel_size=CONV_KERNEL,
        activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l2(0.001),
        name="conv1d_feature",
    )(x)
    x = BatchNormalization(name="batch_norm")(x)
    x = SpatialDropout1D(SPATIAL_DROPOUT_RATE, name="spatial_dropout")(x)
    x = MaxPooling1D(pool_size=2, name="max_pool")(x)

    # 3. RNN Block (Bi-GRU)
    rnn_layer_cls = GRU if RNN_TYPE.lower() == "gru" else LSTM
    x = Bidirectional(rnn_layer_cls(RNN_UNITS, return_sequences=True), name="bi_rnn")(x)

    # 4. GlobalAveragePooling1D
    x = GlobalAveragePooling1D(name="global_avg_pool")(x)

    # 5. Dense Block
    x = Dense(
        DENSE_UNITS1,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.001),
        name="dense_hidden",
    )(x)
    x = Dropout(DROPOUT_RATE, name="dropout_final")(x)

    # Output Layer
    outputs = Dense(len(actions), activation="softmax", name="output_layer")(x)

    # สร้าง Model
    model = Model(inputs=inputs, outputs=outputs, name=EXPERIMENT_NAME)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model


# ---------------- 5) Training Process ----------------
print(f"\n--- Starting Training (Experiment: {EXPERIMENT_NAME}) ---")
print(
    f"Config: Filters={CONV_FILTERS}, RNN={RNN_UNITS}, Dense={DENSE_UNITS1}, Drop={DROPOUT_RATE}"
)

# สร้างโมเดลจาก Functional API ที่เราแก้กันไว้
model = build_model()
model.summary()

# Setup Paths & Callbacks
log_dir = os.path.join(LOG_BASE_DIR, EXPERIMENT_NAME)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")
# 
print("\n[INFO] Plotting model architecture...")
try:
    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(MODEL_DIR, "model_architecture.png"),
        show_shapes=True,       # โชว์ขนาด Input/Output (สำคัญมากสำหรับวิเคราะห์ Shape)
        show_layer_names=True,  # โชว์ชื่อเลเยอร์
        dpi=96                  # ความคมชัดของภาพ
    )
    print(f"[OK] Model architecture saved to {MODEL_DIR}\\model_architecture.png")
except Exception as e:
    print(f"[WARN] Could not plot model. Ensure Graphviz is in PATH. Error: {e}")

callbacks_list = [
    TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,       # [สำคัญมาก] บันทึกค่า Weights และ Biases ทุกๆ 1 Epoch
        write_graph=True,       # วาดแผนผังโครงสร้างสถาปัตยกรรมโมเดล (Model Graph)
        write_images=True,      # แปลงค่าน้ำหนัก (Weights) ออกมาเป็นภาพขาวดำให้ดูง่ายขึ้น
        update_freq='epoch',    # อัปเดตข้อมูลบันทึกลงไฟล์ทุกๆ สิ้น Epoch (ป้องกันเครื่องค้าง)
        profile_batch=0         # ปิดระบบ Profiler ไว้ก่อนเพื่อลด Error ยิบย่อยเรื่อง Permission บนระบบปฏิบัติการ
    ),
    ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max",
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    ),
    EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    ),
]


# 🔥 Function สำหรับหุ้ม Generator ให้เป็น tf.data.Dataset เพื่อความปลอดภัยและรวดเร็ว
def get_dataset_pipeline(X, y, is_training=True, batch_size=32):
    # กำหนดลักษณะของข้อมูล (Output Signature)
    output_sig = (
        tf.TensorSpec(shape=(None, sequence_length, num_features), dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(actions)), dtype=tf.float32),
    )

    # เลือกใช้ Generator ตามที่ตั้งค่าไว้
    gen_func = (
        balanced_data_generator
        if (is_training and USE_BALANCED_SAMPLING)
        else data_generator
    )

    ds = tf.data.Dataset.from_generator(
        lambda: gen_func(X, y, batch_size), output_signature=output_sig
    )
    
    if is_training:
        def soft_augment(features, labels):
            # สุ่มเติม Gaussian Noise บางๆ มากๆ (std=0.005) 
            # เพื่อจำลองอาการ MediaPipe มือสั่นในที่มืด
            noise = tf.random.normal(shape=tf.shape(features), mean=0.0, stddev=0.005, dtype=tf.float32)
            augmented_features = features + noise
            return augmented_features, labels
            
        # ใช้ tf.data.AUTOTUNE ให้มันจัดการ Thread อัตโนมัติ (ไม่กวน GPU)
        ds = ds.map(soft_augment, num_parallel_calls=tf.data.AUTOTUNE)

    # 🔥 Prefetch คือหัวใจ: เตรียมข้อมูลรอไว้ใน Buffer ล่วงหน้า
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


# เตรียม Data Pipelines
print(
    f"[INFO] Preparing {'BALANCED' if USE_BALANCED_SAMPLING else 'NORMAL'} training pipeline..."
)
train_ds = get_dataset_pipeline(
    X_train, y_train, is_training=True, batch_size=BATCH_SIZE
)
val_ds = get_dataset_pipeline(X_val, y_val, is_training=False, batch_size=BATCH_SIZE)

# คำนวณ Steps per epoch
steps_per_epoch = max(1, math.ceil(len(X_train) / BATCH_SIZE))
validation_steps = max(1, math.ceil(len(X_val) / BATCH_SIZE))

# เริ่มเทรนด้วยความมั่นใจ
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
    # ในกรณีเกิด Error ระหว่างเทรน พยายามเซฟโมเดลล่าสุดไว้กันพลาด
    model.save(os.path.join(MODEL_DIR, "interrupted_model.keras"))


# ---------------- 6) Evaluation & Save ----------------
print("\nEvaluating Model on TEST set...")
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)
print(f"Test loss = {test_loss:.4f}, Test accuracy = {test_acc:.4f}")

model.save(os.path.join(MODEL_DIR, "final_model.keras"))
print(f"Final model saved to {os.path.join(MODEL_DIR, 'final_model.keras')}")

# Save Label Map
label_map = {i: action for i, action in enumerate(actions)}
with open(os.path.join(MODEL_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

# Threshold Calibration
print("\nCalibrating thresholds...")
probs = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
y_true = np.argmax(y_val, axis=1)
thresholds = {}
for c, name in enumerate(actions):
    y_true_c = (y_true == c).astype(int)
    best_f1, best_th = -1, 0.5
    for th in np.linspace(0.3, 0.9, 61):
        y_pred_c = (probs[:, c] >= th).astype(int)
        tp = np.sum((y_pred_c == 1) & (y_true_c == 1))
        fp = np.sum((y_pred_c == 1) & (y_true_c == 0))
        fn = np.sum((y_pred_c == 0) & (y_true_c == 1))
        f1 = 0.0 if (tp + fp == 0 or tp + fn == 0) else (2 * tp) / (2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    thresholds[name] = {"threshold": float(best_th), "f1": float(best_f1)}
    print(f"  {name}: Th={best_th:.2f}, F1={best_f1:.4f}")

with open(os.path.join(MODEL_DIR, "thresholds.json"), "w", encoding="utf-8") as f:
    json.dump(thresholds, f, ensure_ascii=False, indent=4)
print("Done.")
