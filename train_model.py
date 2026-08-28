import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "gesture_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "gesture_model.joblib")

print("\n[1/5] Checking dataset...")

if not os.path.exists(DATA_FILE):
    print(f"[-] Error: '{DATA_FILE}' not found! Run collect_data.py first.")
    exit(1)

# Load CSV
df = pd.read_csv(DATA_FILE, header=None)
if df.empty:
    print("[-] Error: 'gesture_data.csv' is empty.")
    exit(1)

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
classes = sorted(list(set(y)))

print(f"[2/5] Loaded {len(df)} samples across {len(classes)} classes: {classes}")

if len(classes) < 2:
    print("[-] Error: Need at least 2 distinct classes to train.")
    exit(1)

# Train/Test Split
print("[3/5] Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Model
print("[4/5] Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n================ Metrics ================")
print(f"Accuracy: {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# Save
joblib.dump(clf, MODEL_FILE)
print(f"[5/5] Success! Model saved to: {MODEL_FILE}\n")
