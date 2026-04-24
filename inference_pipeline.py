from dotenv import load_dotenv
import os
load_dotenv()

import hopsworks
import joblib
import json
import pandas as pd
from datetime import datetime

# ────────────────────────────────────────────
# 1. Hopsworks verbinden
# ────────────────────────────────────────────
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
mr = project.get_model_registry()

# ────────────────────────────────────────────
# 2. Feature Group direkt laden (kein Feature View)
# ────────────────────────────────────────────
fg = fs.get_feature_group(name="weather_features_multiregion", version=1)
print("✅ Feature Group geladen")

# ────────────────────────────────────────────
# 3. Modell aus Model Registry laden
# ────────────────────────────────────────────
hw_model = mr.get_model(
    name="rain_classifier",
    version=2
)

model_dir = hw_model.download()
print("✅ Modell heruntergeladen")

model = joblib.load(f"{model_dir}/rain_classifier.pkl")

with open(f"{model_dir}/model_config.json", "r") as f:
    config = json.load(f)

threshold = config["threshold"]
print(f"✅ Threshold geladen: {threshold}")
print()

# ────────────────────────────────────────────
# 4. Feature-Spalten definieren
# ────────────────────────────────────────────
FEATURE_COLUMNS = [
    "temp", "humidity", "dew_point", "cloud_cover",
    "cloud_cover_low", "pressure", "wind_speed",
    "humidity_avg_24h", "temp_avg_24h", "pressure_avg_24h"
]

TARGET_LOCATION = "zurich"   # anpassen falls nötig
TIMESTAMP_COL   = "timestamp"
latest_ts       = "N/A"

# ────────────────────────────────────────────
# 5. Neueste Zeile aus Feature Group holen
# ────────────────────────────────────────────
try:
    all_df = fg.read()

    if all_df is None or len(all_df) == 0:
        raise ValueError("Feature Group ist leer")

    # Nur gewünschte Location
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
# 6. Feature-Spalten extrahieren & validieren
# ────────────────────────────────────────────
missing = [c for c in FEATURE_COLUMNS if c not in features_df.columns]
if missing:
    raise ValueError(f"Fehlende Feature-Spalten im DataFrame: {missing}")

X_inference = features_df[FEATURE_COLUMNS]

print()
print("Inferenz-Features:")
print(X_inference.to_string(index=False))
print()

# ────────────────────────────────────────────
# 7. Prediction
# ────────────────────────────────────────────
y_pred_proba = model.predict_proba(X_inference)[:, 1][0]
y_pred = int(y_pred_proba >= threshold)

# ────────────────────────────────────────────
# 8. Ergebnis ausgeben
# ────────────────────────────────────────────
print("=" * 70)
print("REGEN-VORHERSAGE (2h Horizont)")
print("=" * 70)
print(f"Location                 : {TARGET_LOCATION}")
print(f"Regen-Wahrscheinlichkeit : {y_pred_proba * 100:.1f}%")
print(f"Decision Threshold       : {threshold}")
print()

if y_pred == 1:
    print("VORHERSAGE: REGEN WAHRSCHEINLICH → Schirm einpacken!")
else:
    print("VORHERSAGE: KEIN REGEN → Trocken bleiben")

print()
print(f"Feature Timestamp : {latest_ts}")
print(f"Inference Time    : {datetime.now().isoformat()}")
print(f"Model Version     : 2")
print("=" * 70)