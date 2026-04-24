from dotenv import load_dotenv
import os
load_dotenv()

import hopsworks
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ────────────────────────────────────────────
# 1. Hopsworks verbinden
# ────────────────────────────────────────────
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
mr = project.get_model_registry()

# ────────────────────────────────────────────
# 2. Feature Group laden
# ────────────────────────────────────────────
fg = fs.get_feature_group(name="weather_features_multiregion", version=1)
print("✅ Feature Group geladen")

# ────────────────────────────────────────────
# 3. Bestes Modell aus Model Registry laden
# ────────────────────────────────────────────
hw_model = mr.get_best_model(
    name="wind_speed_classifier",
    metric="f1",
    direction="max"
)

model_dir = hw_model.download()
print(f"✅ Modell heruntergeladen (Version {hw_model.version})")

model = joblib.load(f"{model_dir}/wind_classifier.pkl")

with open(f"{model_dir}/model_config.json", "r") as f:
    config = json.load(f)

threshold = config["threshold_prob"]
feature_columns = config["feature_columns"]  # alle Spalten inkl. Dummies aus Training
print(f"✅ Threshold geladen: {threshold}")
print(f"✅ Anzahl Feature-Spalten: {len(feature_columns)}")
print()

# ────────────────────────────────────────────
# 4. Konfiguration
# ────────────────────────────────────────────
TARGET_LOCATION = "zurich"
TIMESTAMP_COL   = "timestamp"
latest_ts       = "N/A"

RAW_FEATURE_COLUMNS = [
    "temp", "humidity", "dew_point", "cloud_cover",
    "cloud_cover_low", "pressure", "precip",
    "wind_gusts", "weather_code",
    "humidity_avg_24h", "temp_avg_24h", "pressure_avg_24h",
    "precip_avg_24h", "wind_gusts_avg_24h"
]

# ────────────────────────────────────────────
# 5. Neueste Zeile aus Feature Group holen
# ────────────────────────────────────────────
try:
    all_df = fg.read()

    if all_df is None or len(all_df) == 0:
        raise ValueError("Feature Group ist leer")

    loc_df = all_df[all_df["location"] == TARGET_LOCATION].copy()

    if len(loc_df) == 0:
        raise ValueError(f"Keine Zeilen fuer Location '{TARGET_LOCATION}' gefunden")

    loc_df[TIMESTAMP_COL] = pd.to_datetime(loc_df[TIMESTAMP_COL])

    features_df = (
        loc_df
        .sort_values(TIMESTAMP_COL, ascending=True)
        .tail(1)
        .reset_index(drop=True)
    )

    latest_ts = features_df[TIMESTAMP_COL].iloc[0]
    print(f"✅ Features geladen – Location: {TARGET_LOCATION} | Timestamp: {latest_ts}")

except Exception as e:
    raise RuntimeError(f"Feature-Laden fehlgeschlagen: {e}")

# ────────────────────────────────────────────
# 6. Validieren & Encoding (identisch zu Training)
# ────────────────────────────────────────────
missing = [c for c in RAW_FEATURE_COLUMNS if c not in features_df.columns]
if missing:
    raise ValueError(f"Fehlende Feature-Spalten im DataFrame: {missing}")

X_raw = features_df[RAW_FEATURE_COLUMNS + ["location"]].copy()

# Encoding wie im Training
X_raw["weather_code"] = X_raw["weather_code"].astype(str)
X_encoded = pd.get_dummies(X_raw, columns=["weather_code", "location"], dtype=int)

# Spalten auf Training-Stand bringen: fehlende mit 0, ueberschuessige droppen
X_encoded = X_encoded.reindex(columns=feature_columns, fill_value=0)

print()
print("Inferenz-Features (erste 10 Spalten):")
print(X_encoded.iloc[:, :10].to_string(index=False))
print()

# ────────────────────────────────────────────
# 7. Prediction
# ────────────────────────────────────────────
y_pred_proba = model.predict_proba(X_encoded)[:, 1][0]
y_pred = int(y_pred_proba >= threshold)

# ────────────────────────────────────────────
# 8. Ergebnis ausgeben
# ────────────────────────────────────────────
print("=" * 70)
print("STARKWIND-VORHERSAGE (3h Horizont, >= 50 km/h)")
print("=" * 70)
print(f"Location                    : {TARGET_LOCATION}")
print(f"Starkwind-Wahrscheinlichkeit: {y_pred_proba * 100:.1f}%")
print(f"Decision Threshold          : {threshold}")
print()

if y_pred == 1:
    print("VORHERSAGE: STARKER WIND WAHRSCHEINLICH → Vorsicht!")
else:
    print("VORHERSAGE: KEIN STARKER WIND → Alles ruhig")

print()
print(f"Feature Timestamp : {latest_ts}")
print(f"Inference Time    : {datetime.now().isoformat()}")
print(f"Model Version     : {hw_model.version}")
print("=" * 70)