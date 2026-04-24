from dotenv import load_dotenv
import os
load_dotenv()

import hopsworks
import joblib
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_recall_curve
)

# ────────────────────────────────────────────
# 1. Hopsworks verbinden
# ────────────────────────────────────────────
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs      = project.get_feature_store()

# ────────────────────────────────────────────
# 2. Feature Group laden - Immer neueste Version !
# ────────────────────────────────────────────
fg = fs.get_feature_group(
    name="weather_features_multiregion"
)
print(f"Using Feature Group version {fg.version}")

# ────────────────────────────────────────────
# 3. Feature View erstellen / laden
# ────────────────────────────────────────────
query = fg.select([
    "location",
    "temp",
    "humidity",
    "dew_point",
    "cloud_cover",
    "cloud_cover_low",
    "pressure",
    "precip",
    "wind_gusts",
    "weather_code",
    "humidity_avg_24h",
    "temp_avg_24h",
    "pressure_avg_24h",
    "precip_avg_24h",
    "wind_gusts_avg_24h",
    "strong_wind_warning"
])

try:
    feature_view = fs.get_feature_view(
        name="wind_classification_feature_view"
        # version nicht angeben
    )
    print(f"Using Feature View version {feature_view.version}")
except Exception as e:
    print(f"Feature View nicht gefunden ({e}), wird neu erstellt...")
    feature_view = fs.create_feature_view(
        name="wind_classification_feature_view",
        version=1,
        query=query,
        labels=["strong_wind_warning"]
    )
    print("Feature View neu erstellt.")

assert feature_view is not None, "feature_view ist None – Abbruch"

# ────────────────────────────────────────────
# 4. Training Dataset laden
# ────────────────────────────────────────────
X_train, X_test, y_train, y_test = feature_view.train_test_split(
    test_size=0.2,
    description="Strong wind classification training dataset",
    compute_statistics=False
)

y_train_s = y_train.iloc[:, 0]
y_test_s  = y_test.iloc[:, 0]

print(f"Train: {X_train.shape[0]} Zeilen | Test: {X_test.shape[0]} Zeilen")
vc = y_train_s.value_counts()
print(f"Klassenverteilung (Train):")
print(f"  0 (kein starker Wind): {vc.get(0, 0)}")
print(f"  1 (starker Wind):      {vc.get(1, 0)}")
print()

# ────────────────────────────────────────────
# 4b. Encoding + Spalten-Alignment
# ────────────────────────────────────────────
# Nicht-numerische Spalten ausser location + weather_code droppen
drop_cols = [
    c for c in X_train.columns
    if X_train[c].dtype == object and c not in ["location", "weather_code"]
]
if drop_cols:
    print(f"Dropping non-numeric columns: {drop_cols}")
    X_train = X_train.drop(columns=drop_cols)
    X_test  = X_test.drop(columns=drop_cols)

X_train["weather_code"] = X_train["weather_code"].astype(str)
X_test["weather_code"]  = X_test["weather_code"].astype(str)

X_train = pd.get_dummies(X_train, columns=["weather_code", "location"], dtype=int)
X_test  = pd.get_dummies(X_test,  columns=["weather_code", "location"], dtype=int)

# Spalten angleichen: Test bekommt fehlende Spalten aus Train mit 0
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

print(f"Features nach Encoding: {X_train.shape[1]} Spalten")
print()

# ────────────────────────────────────────────
# 5. Klassengewicht berechnen fuer XGBoost
# ────────────────────────────────────────────
n_neg = (y_train_s == 0).sum()
n_pos = (y_train_s == 1).sum()
scale_pos_weight = round(n_neg / n_pos, 2)
print(f"scale_pos_weight: {scale_pos_weight}  (neg={n_neg}, pos={n_pos})")
print()

# ────────────────────────────────────────────
# 6. Modell trainieren (XGBoost)
# ────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,   # kompensiert Klassenungleichgewicht
    eval_metric="aucpr",                 # Area under Precision-Recall – besser bei Imbalance
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train_s,
    eval_set=[(X_test, y_test_s)],
    verbose=False
)
print("Modell trainiert")
print()

# ────────────────────────────────────────────
# 7. Threshold-Optimierung (F1-maximierend)
# ────────────────────────────────────────────
y_pred_prob = model.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test_s, y_pred_prob)

# F1 fuer jeden Threshold berechnen, Division durch 0 abfangen
f1_scores = np.where(
    (precisions[:-1] + recalls[:-1]) == 0,
    0,
    2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1])
)

best_idx       = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1        = f1_scores[best_idx]

print(f"Optimaler Threshold: {best_threshold:.3f}  (F1={best_f1:.4f})")
print()

# ────────────────────────────────────────────
# 8. Evaluation mit optimiertem Threshold
# ────────────────────────────────────────────
y_pred = (y_pred_prob >= best_threshold).astype(int)
auc    = roc_auc_score(y_test_s, y_pred_prob)
f1     = f1_score(y_test_s, y_pred)

print("=" * 50)
print("MODELL-EVALUATION")
print("=" * 50)
print(classification_report(
    y_test_s, y_pred,
    target_names=["kein starker Wind", "starker Wind"]
))
print(f"ROC-AUC:  {auc:.4f}")
print(f"F1-Score: {f1:.4f}")
print()

print("Confusion Matrix:")
cm = confusion_matrix(y_test_s, y_pred)
print(pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Pred 0", "Pred 1"]
))
print()

# Feature Importance (Top 10)
fi_df = (
    pd.DataFrame({
        "feature":    X_train.columns.tolist(),
        "importance": model.feature_importances_
    })
    .sort_values("importance", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
print("Top-10 Feature Importances:")
print(fi_df.to_string(index=False))
print()

# ────────────────────────────────────────────
# 9. Lokal speichern
# ────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

model_path = "model/wind_classifier.pkl"
joblib.dump(model, model_path)
print(f"Modell gespeichert: {model_path}")

config = {
    "model_type":      "XGBClassifier",
    "label":           "strong_wind_warning",
    "threshold_kmh":   50,
    "threshold_prob":  round(float(best_threshold), 4),
    "n_estimators":    500,
    "scale_pos_weight": scale_pos_weight,
    "f1":              round(f1, 4),
    "roc_auc":         round(auc, 4),
    "feature_columns": X_train.columns.tolist(),
    "top_features":    fi_df.to_dict(orient="records")
}

config_path = "model/model_config.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"Config gespeichert: {config_path}")
print()

# ────────────────────────────────────────────
# 10. In Hopsworks Model Registry hochladen
# ────────────────────────────────────────────
mr = project.get_model_registry()

try:
    existing = mr.get_best_model("wind_speed_classifier", metric="f1", direction="max")
    version  = existing.version + 1
except Exception:
    version = 1

hw_model = mr.sklearn.create_model(
    name="wind_speed_classifier",
    version=version,
    metrics={
        "f1":        round(f1, 4),
        "roc_auc":   round(auc, 4),
        "threshold": round(float(best_threshold), 4),
    },
    description="XGBoost Klassifikation: Starker Wind (>=50 km/h), threshold-optimiert",
    input_example=X_train.iloc[0:1],
    feature_view=feature_view
)

hw_model.save("model")
print(f"Modell in Model Registry hochgeladen (Version {version}).")
