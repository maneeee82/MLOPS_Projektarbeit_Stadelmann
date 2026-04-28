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
LOCATIONS = ["zurich", "basel", "bern", "geneva", "lugano"]
TIMESTAMP_COL = "timestamp"

RAW_FEATURE_COLUMNS = [
    "temp", "humidity", "dew_point", "cloud_cover",
    "cloud_cover_low", "pressure", "precip",
    "wind_gusts", "weather_code",
    "humidity_avg_24h", "temp_avg_24h", "pressure_avg_24h",
    "precip_avg_24h", "wind_gusts_avg_24h"
]

# ────────────────────────────────────────────
# 5. Alle Daten einmal laden
# ────────────────────────────────────────────
all_df = fg.read()

if all_df is None or len(all_df) == 0:
    raise ValueError("Feature Group ist leer")

all_df[TIMESTAMP_COL] = pd.to_datetime(all_df[TIMESTAMP_COL])

# ────────────────────────────────────────────
# 6–8. Pro Location: Encoding, Prediction, Ausgabe
# ────────────────────────────────────────────
print("=" * 70)
print("STARKWIND-VORHERSAGE (3h Horizont, >= 50 km/h)")
print(f"Inference Time : {datetime.now().isoformat()}")
print(f"Model Version  : {hw_model.version}")
print("=" * 70)

for TARGET_LOCATION in LOCATIONS:
    try:
        loc_df = all_df[all_df["location"] == TARGET_LOCATION].copy()

        if len(loc_df) == 0:
            print(f"⚠️  {TARGET_LOCATION}: Keine Daten vorhanden")
            continue

        features_df = (
            loc_df
            .sort_values(TIMESTAMP_COL, ascending=True)
            .tail(1)
            .reset_index(drop=True)
        )

        latest_ts = features_df[TIMESTAMP_COL].iloc[0]

        missing = [c for c in RAW_FEATURE_COLUMNS if c not in features_df.columns]
        if missing:
            print(f"⚠️  {TARGET_LOCATION}: Fehlende Spalten {missing}")
            continue

        X_raw = features_df[RAW_FEATURE_COLUMNS + ["location"]].copy()
        X_raw["weather_code"] = X_raw["weather_code"].astype(str)
        X_encoded = pd.get_dummies(X_raw, columns=["weather_code", "location"], dtype=int)
        X_encoded = X_encoded.reindex(columns=feature_columns, fill_value=0)

        assert X_encoded.shape[1] == len(feature_columns)
        assert not X_encoded.isna().any().any()

        y_pred_proba = model.predict_proba(X_encoded)[:, 1][0]
        y_pred = int(y_pred_proba >= threshold)

        warnung = "⚠️  STARKER WIND WAHRSCHEINLICH → Vorsicht!" if y_pred == 1 else "✅ Kein starker Wind"

        print(f"{TARGET_LOCATION:<10} | {y_pred_proba*100:5.1f}% | Threshold: {threshold:.4f} | {warnung}  | TS: {latest_ts}")

    except Exception as e:
        print(f"❌ {TARGET_LOCATION}: Fehler – {e}")

print("=" * 70)