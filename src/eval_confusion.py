import os
# ==============================================================================
# สับสวิตช์เปิด Legacy Mode (Keras 2) ก่อนที่จะมีการ Import TensorFlow 
# เพื่อให้สามารถโหลด Weights ของ Custom Layers ที่เทรนจาก TF เวอร์ชันเก่าได้ 100%
# และปิดการแสดงผล Log ที่ไม่จำเป็นของระดับ C++ (เช่น แจ้งเตือนเรื่อง CPU instruction)
# ==============================================================================
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json, argparse, random
import numpy as np
import pandas as pd
import tensorflow as tf

# เนื่องจากใช้ Legacy Mode เราจะใช้ tf.keras แทนเพื่อความเสถียร
from tensorflow.keras.layers import Layer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

# บังคับใช้ Backend "Agg" สำหรับ Matplotlib เพื่อให้เซฟรูปได้โดยไม่ต้องเปิดหน้าต่าง UI (ลด Error บน Server)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- Setup Directories -----------------
# จัดการ Path แบบถอยหลังเพื่อหาโฟลเดอร์ Root ของโปรเจกต์
SRC_DIR = os.path.dirname(os.path.abspath(__file__))  # พิกัดไฟล์ปัจจุบัน (storage/src)
STORAGE_DIR = os.path.dirname(SRC_DIR)                # ถอย 1 ขั้นมา (storage)
PROJECT_ROOT = os.path.dirname(STORAGE_DIR)           # ถอยอีก 1 ขั้นมา Root
MODEL_DIR = os.path.join(STORAGE_DIR, "models")       # โฟลเดอร์เก็บโมเดล
DATA_DIR = os.path.join(PROJECT_ROOT, "data")         # โฟลเดอร์เก็บข้อมูล

KERAS_MODEL = os.path.join(MODEL_DIR, "best_model.keras")  # ไฟล์โมเดลที่ต้องการประเมิน
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json") # ไฟล์แปลรหัสตัวเลขเป็นชื่อท่าทาง

# ==============================================================================

# ----------------- CLI Arguments -----------------
# สร้างระบบรับคำสั่งผ่าน Command Line Terminal (เช่น python eval_confusion.py --batch 64)
parser = argparse.ArgumentParser(description="Evaluate best_model.keras and export confusion matrix & metrics.")
parser.add_argument("--data-dir",  default=os.path.join(DATA_DIR,"processed_test"), help="Processed .npy root dir")
parser.add_argument("--subset",    choices=["test","all"], default="all", help="Evaluate on test split or all data")
parser.add_argument("--test-size", type=float, default=0.2, help="Test size for split when subset=test")
parser.add_argument("--seed",      type=int, default=42, help="Random seed")
parser.add_argument("--batch",     type=int, default=32, help="Prediction batch size")
parser.add_argument("--out",       default=os.path.join(STORAGE_DIR, "reports"), help="Output directory for reports") # ชี้ไปที่ storage/reports
args = parser.parse_args()

# ล็อค Seed เพื่อให้การสุ่มต่างๆ ให้ผลลัพธ์คงที่ (Reproducibility)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
random.seed(args.seed)

SEQ_LEN, NUM_FEAT = 30, 258
# สร้างโฟลเดอร์เก็บ Report ถ้ายังไม่มี
os.makedirs(args.out, exist_ok=True)

# ----------------- Load labels -----------------
def load_labels():
    """
    ฟังก์ชันโหลดชื่อคลาส 
    ถ้ามีไฟล์ label_map.json ให้ใช้ไฟล์นั้น (ชัวร์ที่สุด) 
    ถ้าไม่มี ให้ไปอ่านชื่อโฟลเดอร์ใน data_dir เอาเอง
    """
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            lm = json.load(f)
        # เรียงลำดับ Index ให้ถูกต้อง (0, 1, 2, ...) แล้วดึงชื่อออกมา
        idxs = sorted(int(k) for k in lm.keys())
        labels = [lm[str(i)] for i in idxs]
        return labels
        
    # กรณีหาไฟล์ไม่เจอ (Fallback)
    classes = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    classes.sort() # สำคัญ! ต้องเรียง A-Z เพื่อให้ Index ตรงกับตอนเทรน
    return classes

labels = load_labels()
num_classes = len(labels)
print(f"[INFO] Classes ({num_classes}):", labels)

# ----------------- Load dataset -----------------
X_list, y_list = [], []
print(f"[INFO] Loading data from: {args.data_dir}")

# 🔥 [แก้ไข Bug] ปรับลูปการอ่านไฟล์ให้ถูกต้องและมีระเบียบ
for ci, cname in enumerate(labels):
    cdir = os.path.join(args.data_dir, cname)
    if not os.path.isdir(cdir):
        print(f"[WARN] Missing class folder: {cdir} (skip this class)")
        continue
    
    files = sorted([f for f in os.listdir(cdir) if f.endswith(".npy")])
    count = 0
    for f in files:
        file_path = os.path.join(cdir, f)
        try:
            arr = np.load(file_path)
        except Exception as e:
            print(f"[ERROR] Cannot load {file_path}: {e}")
            continue
            
        # ตรวจสอบ Shape ของ Data ให้เป๊ะ (Batch_size, 30, 258)
        if arr.shape != (SEQ_LEN, NUM_FEAT):
            print(f"[SKIP] {cname}/{f} shape mismatch: {arr.shape}")
            continue
            
        X_list.append(arr.astype(np.float32)) # แปลงเป็น Float32 ลดขนาด Memory
        y_list.append(ci)                     # เก็บคำตอบที่ถูกต้อง (Ground truth)
        count += 1
        
    print(f"  - {cname:<12}: {count} samples loaded")

# แปลง List ให้เป็น Numpy Array ก้อนเดียว
X = np.asarray(X_list, dtype=np.float32)
y = np.asarray(y_list, dtype=np.int32)
print(f"[INFO] Total dataset: X={X.shape}, y={y.shape}")

if len(X) == 0:
    print("[ERROR] No data found! Please check data path.")
    exit()

# ----------------- Split -----------------
# ระบบช่วยหั่นข้อมูล เผื่อเราโหลดโฟลเดอร์ 'processed_train' มาประเมิน
if args.subset == "test":
    # stratify=y คือการบังคับให้อัตราส่วนของแต่ละคลาสใน Train/Test เท่าเดิม
    _, X_eval, _, y_eval = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"[INFO] Using SPLIT (subset=test): {X_eval.shape[0]} samples for eval")
else:
    # ถ้า subset=all (ค่า Default) แปลว่าเราโหลด processed_test มาแล้ว ก็ใช้ทั้งหมดเลย
    X_eval, y_eval = X, y
    print(f"[INFO] Using ALL loaded data (subset=all): {X_eval.shape[0]} samples")

# ----------------- Load model -----------------
print(f"\n[INFO] Loading model: {KERAS_MODEL}")
try:
    # โหลดโมเดล โดยถ้ามี Custom Layers (เช่น Attention) ให้ปลดคอมเมนต์ออก
    model = tf.keras.models.load_model(
        KERAS_MODEL,
        # compile=False  # ปลดคอมเมนต์ตัวนี้หากเจอ Error เกี่ยวกับ Optimizer state
    )
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit()

# ----------------- Predict & Metrics -----------------
print("\n[INFO] Predicting...")
# สั่งโมเดลทำนายข้อมูลทั้งหมด ผลลัพธ์จะเป็นความน่าจะเป็น (Probability) ของแต่ละคลาส
probs = model.predict(X_eval, batch_size=args.batch, verbose=1)
# เลือกคลาสที่มีความน่าจะเป็นสูงสุดมาเป็นคำตอบ (Predicted Class)
y_pred = np.argmax(probs, axis=1)

# สร้างตาราง Confusion Matrix แบบนับจำนวน (Counts)
cm = confusion_matrix(y_eval, y_pred, labels=list(range(num_classes)))
# สร้างตาราง Confusion Matrix แบบเปอร์เซ็นต์ (Row-normalized) 
# โดยหารด้วยจำนวนข้อมูลจริงในคลาสนั้นๆ (บวก np.maximum ป้องกันการหารด้วย 0)
cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

# สร้างตารางสรุปผล Precision, Recall, F1-Score ในรูปแบบ Dictionary
report = classification_report(y_eval, y_pred, target_names=labels, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()

# สกัดค่ารายคลาสออกมาเพื่อทำตารางจัดอันดับท่าที่แย่ที่สุด
p, r, f1, support = precision_recall_fscore_support(y_eval, y_pred, labels=list(range(num_classes)), zero_division=0)
per_class_df = pd.DataFrame({
    "class": labels, "precision": p, "recall": r, "f1": f1, "support": support
}).sort_values("class")

# ----------------- Save CSVs & Plots -----------------
print("[INFO] Saving reports and plots...")
# ส่งออกตารางลงไฟล์ CSV (encoding="utf-8-sig" ช่วยให้อ่านใน Excel ได้ไม่เพี้ยน)
pd.DataFrame(cm, index=labels, columns=labels).to_csv(os.path.join(args.out, "confusion_counts.csv"), encoding="utf-8-sig")
pd.DataFrame(cm_norm, index=labels, columns=labels).round(4).to_csv(os.path.join(args.out, "confusion_norm.csv"), encoding="utf-8-sig")
report_df.to_csv(os.path.join(args.out, "classification_report.csv"), encoding="utf-8-sig", float_format="%.4f")
per_class_df.to_csv(os.path.join(args.out, "per_class_table.csv"), index=False, encoding="utf-8-sig", float_format="%.4f")

def plot_cm(M, title, fn, normalize=False):
    """
    ฟังก์ชันวาดรูปตารางสี่เหลี่ยมสีฟ้า (Confusion Matrix) 
    M = Matrix ข้อมูล, fn = ชื่อไฟล์รูปภาพ
    """
    # ปรับขนาดรูปภาพให้อ่านง่ายขึ้น (ยิ่งคลาสเยอะ รูปยิ่งใหญ่)
    plt.figure(figsize=(max(8, 0.8*num_classes), max(7, 0.8*num_classes)))
    # วาดรูปสี่เหลี่ยม ไล่สีฟ้า (Blues) ยิ่งค่าเยอะ สีจะยิ่งเข้ม
    plt.imshow(M, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar() # แถบสเกลสีด้านข้าง
    
    # วางชื่อคลาสกำกับแกน X และ Y (หมุนแกน X 90 องศาเพื่อไม่ให้ชื่อทับกัน)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    
    # พิมพ์ตัวเลขกำกับลงในแต่ละช่องสี่เหลี่ยม
    fmt = ".2f" if normalize else "d"
    thresh = (M.max()/2.0) if M.size > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            val = M[i, j]
            s = f"{val:{fmt}}"
            # ถ้าช่องนั้นสีเข้ม ให้ใช้ตัวหนังสือสีขาว ไม่งั้นใช้สีดำ
            plt.text(j, i, s,
                     horizontalalignment="center",
                     color="white" if val > thresh else "black", fontsize=8)
                     
    plt.ylabel("True label (ท่ายากจริง)")
    plt.xlabel("Predicted label (AI ทายว่า)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, fn), dpi=180) # ความละเอียดสูง 180 DPI
    plt.close()

# เรียกฟังก์ชันวาดรูปทั้งแบบดิบ และแบบเปอร์เซ็นต์
plot_cm(cm, "Confusion Matrix (Counts)", "confusion_counts.png", normalize=False)
plot_cm(cm_norm, "Confusion Matrix (Row-normalized)", "confusion_norm.png", normalize=True)

# ----------------- Summary -----------------
# คำนวณความแม่นยำภาพรวม (ค่าตรงแกนทแยงมุมรวมกัน / ข้อมูลทั้งหมด)
overall_acc = (y_pred == y_eval).mean()

# สรุปผลไฮไลต์ 3 ท่าที่ทายแย่สุด (เอาไว้ไปทำ Augment เพิ่ม) และ 3 ท่าที่ทายเก่งสุด
summary = [
    f"Samples eval: {X_eval.shape[0]}",
    f"Overall accuracy: {overall_acc:.4f}",
    "", "Per-class (worst 3 by F1):"
]
worst3 = per_class_df.sort_values("f1").head(3)
best3  = per_class_df.sort_values("f1", ascending=False).head(3)

for _, row in worst3.iterrows():
    summary.append(f"  - {row['class']}: F1={row['f1']:.3f}, P={row['precision']:.3f}, R={row['recall']:.3f}, n={int(row['support'])}")
summary.append("\nPer-class (best 3 by F1):")
for _, row in best3.iterrows():
    summary.append(f"  + {row['class']}: F1={row['f1']:.3f}, P={row['precision']:.3f}, R={row['recall']:.3f}, n={int(row['support'])}")

# เซฟบทสรุปเป็น Text File
with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary))

print("\n" + "="*40)
print("\n".join(summary))
print("="*40)
print(f"\n[OK] Saved reports to: {os.path.abspath(args.out)}")