import os
# ==============================================================================
# สับสวิตช์เปิด Legacy Mode (Keras 2) ก่อนที่จะมีการ Import TensorFlow 
# เพื่อให้สามารถโหลด Weights ของ Custom Layers ที่เทรนจากเวอร์ชันเก่าได้ 100%
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- Setup Directories -----------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))  # storage/src
STORAGE_DIR = os.path.dirname(SRC_DIR)  # storage
PROJECT_ROOT = os.path.dirname(STORAGE_DIR)  # Project Root
MODEL_DIR = os.path.join(STORAGE_DIR, "models")

KERAS_MODEL = os.path.join(MODEL_DIR, "best_model.keras")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")
DATA_DIR = os.path.join(PROJECT_ROOT, "data") 

# ==============================================================================

# ----------------- CLI Arguments -----------------
parser = argparse.ArgumentParser(description="Evaluate best_model.keras and export confusion matrix & metrics.")
parser.add_argument("--data-dir",  default=os.path.join(DATA_DIR,"processed_test"), help="Processed .npy root dir")
parser.add_argument("--subset",    choices=["test","all"], default="all", help="Evaluate on test split or all data")
parser.add_argument("--test-size", type=float, default=0.2, help="Test size for split when subset=test")
parser.add_argument("--seed",      type=int, default=42, help="Random seed")
parser.add_argument("--batch",     type=int, default=32, help="Prediction batch size")
parser.add_argument("--out",       default="reports", help="Output directory for reports")
args = parser.parse_args()

np.random.seed(args.seed)
tf.random.set_seed(args.seed)
random.seed(args.seed)

SEQ_LEN, NUM_FEAT = 30, 258
os.makedirs(args.out, exist_ok=True)

# ----------------- Load labels -----------------
def load_labels():
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            lm = json.load(f)
        idxs = sorted(int(k) for k in lm.keys())
        labels = [lm[str(i)] for i in idxs]
        return labels
    classes = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    classes.sort()
    return classes

labels = load_labels()
num_classes = len(labels)
print(f"[INFO] Classes ({num_classes}):", labels)

# ----------------- Load dataset -----------------
X, y = [], []
print(f"[INFO] Loading data from: {args.data_dir}")

for ci, cname in enumerate(labels):
    cdir = os.path.join(args.data_dir, cname)
    if not os.path.isdir(cdir):
        print(f"[WARN] Missing class folder: {cdir} (skip this class)")
        continue
    
    files = [f for f in os.listdir(cdir) if f.endswith(".npy")]
    files.sort()
    count = 0
    for f in files:
        arr = np.load(os.path.join(cdir, f))
        if arr.shape != (SEQ_LEN, NUM_FEAT):
            print(f"[SKIP] {cname}/{f} shape={arr.shape}")
            continue
        X.append(arr.astype(np.float32))
        y.append(ci)
        count += 1
    print(f"  - {cname}: {count} samples")

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)
print(f"[INFO] Total dataset: X={X.shape}, y={y.shape}")

if len(X) == 0:
    print("[ERROR] No data found! Please check data path.")
    exit()

# ----------------- Split -----------------
if args.subset == "test":
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"[INFO] Using SPLIT (subset=test): {X_eval.shape[0]} samples for eval")
else:
    X_eval, y_eval = X, y
    print(f"[INFO] Using ALL loaded data (subset=all): {X_eval.shape[0]} samples")

# ----------------- Load model -----------------
print(f"[INFO] Loading model: {KERAS_MODEL}")
try:
    model = tf.keras.models.load_model(
        KERAS_MODEL,
        # custom_objects={
        #     'Attention': Attention,
        #     'VelocityFeatureLayer': VelocityFeatureLayer
        # },
        # compile=False
    )
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit()

# ----------------- Predict & Metrics -----------------
print("[INFO] Predicting...")
probs = model.predict(X_eval, batch_size=args.batch, verbose=1)
y_pred = np.argmax(probs, axis=1)

cm = confusion_matrix(y_eval, y_pred, labels=list(range(num_classes)))
cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

report = classification_report(y_eval, y_pred, target_names=labels, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()

p, r, f1, support = precision_recall_fscore_support(y_eval, y_pred, labels=list(range(num_classes)), zero_division=0)
per_class_df = pd.DataFrame({
    "class": labels, "precision": p, "recall": r, "f1": f1, "support": support
}).sort_values("class")

# ----------------- Save CSVs & Plots -----------------
pd.DataFrame(cm, index=labels, columns=labels).to_csv(os.path.join(args.out, "confusion_counts.csv"), encoding="utf-8-sig")
pd.DataFrame(cm_norm, index=labels, columns=labels).round(4).to_csv(os.path.join(args.out, "confusion_norm.csv"), encoding="utf-8-sig")
report_df.to_csv(os.path.join(args.out, "classification_report.csv"), encoding="utf-8-sig", float_format="%.4f")
per_class_df.to_csv(os.path.join(args.out, "per_class_table.csv"), index=False, encoding="utf-8-sig", float_format="%.4f")

def plot_cm(M, title, fn, normalize=False):
    plt.figure(figsize=(max(6, 0.6*num_classes), max(5, 0.6*num_classes)))
    plt.imshow(M, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    fmt = ".2f" if normalize else "d"
    thresh = (M.max()/2.0) if M.size > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            val = M[i, j]
            s = f"{val:{fmt}}"
            plt.text(j, i, s,
                     horizontalalignment="center",
                     color="white" if val > thresh else "black", fontsize=8)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, fn), dpi=180)
    plt.close()

plot_cm(cm, "Confusion Matrix (Counts)", "confusion_counts.png", normalize=False)
plot_cm(cm_norm, "Confusion Matrix (Row-normalized)", "confusion_norm.png", normalize=True)

# ----------------- Summary -----------------
overall_acc = (y_pred == y_eval).mean()
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

with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary))

print("\n".join(summary))
print(f"\n[OK] Saved reports to: {os.path.abspath(args.out)}")