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
project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

# ────────────────────────────────────────────
# 2. Feature View laden
# ────────────────────────────────────────────
feature_view = fs.get_feature_view(
    name="weather_feature_view",
    version=1
)
print("✅ Feature View geladen")

# ────────────────────────────────────────────
# 3. Modell aus Model Registry laden
# ────────────────────────────────────────────
#Hole Modell-Objekt aus Hopsworks
hw_model = mr.get_model( 
    name="rain_classifier",
    version=2
)

# Download zu lokal (Ordner wird erstellt)
model_dir = hw_model.download()
print(f"✅ Modell heruntergeladen")

# Modell laden
model = joblib.load(f"{model_dir}/rain_classifier.pkl")

# Config mit Threshold laden
with open(f"{model_dir}/model_config.json", 'r') as f:
    config = json.load(f)

#Grenze, ab wann "Regen"
threshold = config['threshold']
print(f"✅ Threshold geladen: {threshold}")
print()

# ────────────────────────────────────────────
# 4. Neueste Features laden
# ────────────────────────────────────────────

# OPTION A: Aus Feature View die neuesten Features
# (Wenn deine Feature Pipeline regelmaessig laeuft)
try:
    features_df = feature_view.get_latest_feature_values()
    print("✅ Live-Features aus Feature View geladen")
except:
    print("⚠️  Feature View hat keine Features, nutze Test-Daten")
    # OPTION B: Fallback - aus Test-Daten einen Sample nehmen
    X_test, _ = feature_view.train_test_split(test_size=0.2)
    features_df = X_test.sample(1).reset_index(drop=True)
    print("✅ Features aus Test-Daten geladen")

print(f"Anzahl Zeilen: {len(features_df)}")
print()

# ────────────────────────────────────────────
# 5. Feature Columns definieren (OHNE Label!)
# ────────────────────────────────────────────
feature_columns = [
    "temp", "humidity", "dew_point", "cloud_cover", 
    "cloud_cover_low", "pressure", "wind_speed",
    "humidity_avg_24h", "temp_avg_24h", "pressure_avg_24h"
]

X_inference = features_df[feature_columns]

print("Inferenz-Features:")
print(X_inference)
print()

# ────────────────────────────────────────────
# 6. Prediction durchfuehren
# ────────────────────────────────────────────
y_pred_proba = model.predict_proba(X_inference)[:, 1][0]
y_pred = 1 if y_pred_proba >= threshold else 0

# ────────────────────────────────────────────
# 7. Ergebnis ausgeben
# ────────────────────────────────────────────
print("=" * 70)
print("🌧️  REGEN-VORHERSAGE (2h Horizont)")
print("=" * 70)
print(f"Regen-Wahrscheinlichkeit: {y_pred_proba*100:.1f}%")
print(f"Decision Threshold: {threshold}")
print()

if y_pred == 1:
    print("⚠️  VORHERSAGE: REGEN WAHRSCHEINLICH")
    print("    → Schirm einpacken!")
else:
    print("☀️  VORHERSAGE: KEIN REGEN")
    print("    → Trocken bleiben")

print()
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"Model Version: 2")
print("=" * 70)
