import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score


df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# Remove duplicates
df = df.drop_duplicates()

# -----------------------------
# Split features & target
# -----------------------------
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

# -----------------------------
# 🔄 Encoding
# -----------------------------
encoders = {}

for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# -----------------------------
# 📊 Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# ⚖️ Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# ⚙️ Hyperparameter Tuning (Random Forest)
# =====================================================
params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20]
}

grid = GridSearchCV(RandomForestClassifier(), params, cv=3)
grid.fit(X_train, y_train)

rf_best = grid.best_estimator_

print("🔧 Best RF Parameters:", grid.best_params_)

# =====================================================
# 🧪 Cross Validation
# =====================================================
cv_score = cross_val_score(rf_best, X_train, y_train, cv=5).mean()
print("📊 Cross Validation Score (RF):", cv_score)

# -----------------------------
# 🤖 Models (with tuned RF)
# -----------------------------
models = {
    "RandomForest_Tuned": rf_best,
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

best_model = None
best_accuracy = 0
best_name = ""

# -----------------------------
# 🚀 Train & Compare
# -----------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"{name} Accuracy: {acc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

# -----------------------------
# 🏆 Save Best Model
# -----------------------------
print(f"\n🏆 Best Model: {best_name}")
print(f"🎯 Accuracy: {best_accuracy:.4f}")

joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("✅ Best model saved successfully!")