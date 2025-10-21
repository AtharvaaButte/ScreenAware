import os
from math import sqrt
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    silhouette_score
)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans

# paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "digital_diet_mental_health.csv")
models_dir = os.path.join(base_dir, "models")
backend_models_dir = os.path.abspath(os.path.join(base_dir, "..", "backend", "app", "models"))

os.makedirs(models_dir, exist_ok=True)
os.makedirs(backend_models_dir, exist_ok=True)

# 1. Load data
print("Loading dataset...")
df = pd.read_csv(data_path)
print("Dataset shape:", df.shape)

# 2. Compute strain score & assign risk categories
# Screen-to-sleep displacement
disp = df["daily_screen_time_hours"] / (df["sleep_duration_hours"] + 0.1)
score_disp = np.clip((disp - 0.4) / (1.6 - 0.4) * 10, 0, 10)

# Stress & poor sleep burden (1 to 10 scale)
score_stress_sleep = (df["stress_level"] + (11 - df["sleep_quality"])) / 2

# Inactivity deficit
score_inact = np.clip(
    (df["daily_screen_time_hours"] / (df["physical_activity_hours_per_week"] + 1)) * 2,
    0,
    10
)

# Composite Digital Strain Index (0 to 10)
df["cdsi"] = 0.40 * score_disp + 0.40 * score_stress_sleep + 0.20 * score_inact

def assign_risk(score):
    if score >= 6.2:
        return "High"
    elif score <= 4.2:
        return "Low"
    else:
        return "Medium"

df["risk_level"] = df["cdsi"].apply(assign_risk)

print("\nRisk distribution:")
print(df["risk_level"].value_counts())

# 3. Train risk classification model (XGBoost)
print("\nTraining risk classifier...")
clf_features = [
    "daily_screen_time_hours",
    "sleep_duration_hours",
    "stress_level",
    "sleep_quality",
    "physical_activity_hours_per_week"
]

X_clf = df[clf_features]
y_clf = df["risk_level"]

le = LabelEncoder()
y_clf_encoded = le.fit_transform(y_clf)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf_encoded, test_size=0.2, random_state=42, stratify=y_clf_encoded
)

scaler_clf = StandardScaler()
X_train_c_scaled = scaler_clf.fit_transform(X_train_c)
X_test_c_scaled = scaler_clf.transform(X_test_c)

clf_model = XGBClassifier(
    n_estimators=180,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    eval_metric="mlogloss"
)
clf_model.fit(X_train_c_scaled, y_train_c)

# 5-fold cross validation check
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf_model, X_train_c_scaled, y_train_c, cv=cv, scoring="accuracy")
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f}")

# Test set evaluation
y_pred_c = clf_model.predict(X_test_c_scaled)
print(f"Test Accuracy: {accuracy_score(y_test_c, y_pred_c):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_c, target_names=le.classes_))

# 4. Train mood regression model
print("Training mood regression model...")
# Feature engineering (ratios and wellness score)
df["screen_sleep_ratio"] = df["daily_screen_time_hours"] / (df["sleep_duration_hours"] + 1)
df["stress_x_sleep"] = df["stress_level"] * df["sleep_quality"]
df["activity_balance"] = df["physical_activity_hours_per_week"] / (df["daily_screen_time_hours"] + 1)
df["wellness_score"] = (
    (df["sleep_quality"] + df["physical_activity_hours_per_week"]) / 2
    - (df["stress_level"] + df["daily_screen_time_hours"] / 2)
)

reg_features = clf_features + [
    "screen_sleep_ratio",
    "stress_x_sleep",
    "activity_balance",
    "wellness_score"
]

X_reg = df[reg_features]
y_reg = df["mood_rating"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_reg = StandardScaler()
X_train_r_scaled = scaler_reg.fit_transform(X_train_r)
X_test_r_scaled = scaler_reg.transform(X_test_r)

reg_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
reg_model.fit(X_train_r_scaled, y_train_r)

y_pred_r = reg_model.predict(X_test_r_scaled)
print(f"Regression MAE: {mean_absolute_error(y_test_r, y_pred_r):.4f}")
print(f"Regression RMSE: {sqrt(mean_squared_error(y_test_r, y_pred_r)):.4f}")

# Min/max bounds for 1-5 mood mapping in backend
all_preds = reg_model.predict(scaler_reg.transform(X_reg))
mood_min_range = float(np.percentile(all_preds, 1))
mood_max_range = float(np.percentile(all_preds, 99))

# 5. Usage Clustering (KMeans)
print("\nClustering user usage patterns...")
cluster_features = [
    "social_media_hours",
    "gaming_hours",
    "entertainment_hours",
    "work_related_hours"
]

X_cluster = df[cluster_features]
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

km = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = km.fit_predict(X_cluster_scaled)
print(f"KMeans (k=5) Silhouette Score: {silhouette_score(X_cluster_scaled, labels):.4f}")

# Assign names to cluster centroids
centers = scaler_cluster.inverse_transform(km.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=cluster_features)

cluster_name_map = {}
for idx, row in centers_df.iterrows():
    dom = row.idxmax()
    if dom == "social_media_hours":
        cluster_name_map[idx] = "Social Media Dominant"
    elif dom == "gaming_hours":
        cluster_name_map[idx] = "Gaming Heavy"
    elif dom == "entertainment_hours":
        cluster_name_map[idx] = "Entertainment Focused" if row["work_related_hours"] >= 1.0 else "Casual Leisure"
    elif dom == "work_related_hours":
        cluster_name_map[idx] = "Work Oriented"
    else:
        cluster_name_map[idx] = "Balanced Usage"

print("Identified personas:")
for idx, name in cluster_name_map.items():
    print(f"  Cluster {idx}: {name}")

# 6. Save models and artifacts
print("\nSaving model files...")
artifacts = {
    "clf_features": clf_features,
    "reg_features": reg_features,
    "cluster_features": cluster_features,
    "best_cluster_method": "KMeans(k=5)",
    "cluster_name_map": cluster_name_map,
    "clf_target_labels": list(le.classes_),
    "mood_min_range": mood_min_range,
    "mood_max_range": mood_max_range
}

files = {
    "risk_model.pkl": clf_model,
    "scaler_clf.pkl": scaler_clf,
    "label_encoder_clf.pkl": le,
    "mood_model.pkl": reg_model,
    "scaler_reg.pkl": scaler_reg,
    "cluster_model.pkl": km,
    "scaler_cluster.pkl": scaler_cluster,
    "artifacts.pkl": artifacts
}

for name, obj in files.items():
    joblib.dump(obj, os.path.join(models_dir, name))
    joblib.dump(obj, os.path.join(backend_models_dir, name))
    print(f"Saved {name}")

print("\nDone! All models trained and saved successfully.")
