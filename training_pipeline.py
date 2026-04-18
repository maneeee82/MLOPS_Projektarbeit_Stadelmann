from dotenv import load_dotenv
import os
load_dotenv()

import hopsworks
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# ────────────────────────────────────────────
# 1. Hopsworks verbinden
# ────────────────────────────────────────────
project = hopsworks.login()
fs = project.get_feature_store()

# ────────────────────────────────────────────
# 2. Feature Group laden
# ────────────────────────────────────────────
fg = fs.get_feature_group(
    name="weather_features_multiregion",
    version=1
)

# ────────────────────────────────────────────
# 3. Feature View erstellen
# ────────────────────────────────────────────
query = fg.select([
    # RT-Features (aktuell, zum Inferenzzeitpunkt bekannt)
    "temp",           # Temperatur in °C
    "humidity",       # relative Luftfeuchtigkeit in %
    "dew_point",      # Taupunkt – hoher Wert = Luft nahe Sättigung
    "cloud_cover",    # Gesamtbewölkung in %
    "cloud_cover_low",# Tiefe Wolken – besonders relevant für Regen
    "pressure",       # Luftdruck in hPa – fallender Druck = Schlechtwetter
    "wind_speed",     # Windgeschwindigkeit in km/h
    # Aggregierte Features (Rolling Window 24h)
    "humidity_avg_24h",  # Durchschnittliche Luftfeuchtigkeit der letzten 24h
    "temp_avg_24h",      # Durchschnittliche Temperatur der letzten 24h
    "pressure_avg_24h",  # Durchschnittlicher Luftdruck der letzten 24h – Trend erkennbar
    # Label
    "will_rain_in_2h"    # Zielgrösse: Regnet es in den nächsten 2 Stunden? (0/1)
])

try:
    feature_view = fs.get_feature_view(
        name="weather_feature_view",
        version=1
    )
    if feature_view is None:
        raise ValueError("feature_view ist None")
    print("Feature View bereits vorhanden.")

except Exception as e:
    print(f"Feature View nicht gefunden ({e}), wird neu erstellt...")
    feature_view = fs.create_feature_view(
        name="weather_feature_view",
        version=1,
        query=query,
        labels=["will_rain_in_2h"]
    )
    print("Feature View neu erstellt.")

# Sicherheitscheck vor weiterarbeiten
assert feature_view is not None, "feature_view ist None – Abbruch"

# ────────────────────────────────────────────
# 4. Training Dataset laden
# ────────────────────────────────────────────
# X_train: Features zum Trainieren (80%)
# X_test:  Features zum Testen (20%)
# y_train: Labels zum Trainieren (80%)
# y_test:  Labels zum Testen (20%)
X_train, X_test, y_train, y_test = feature_view.train_test_split(
    test_size=0.2,
    description="Rain prediction training dataset"
)

print(f"Train: {X_train.shape[0]} Zeilen | Test: {X_test.shape[0]} Zeilen")
print(f"Label-Verteilung (Train):\n{y_train.value_counts()}")
print()

# ────────────────────────────────────────────
# 5. Modell trainieren
# ────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,       # 100 Entscheidungsbäume werden gebaut
    class_weight="balanced", # gleicht ungleiche Klassen aus (z.B. wenig Regen-Tage)
    random_state=42          # Reproduzierbarkeit – gleicher Seed = gleiche Ergebnisse
)
model.fit(X_train, y_train.values.ravel())
print("✅ Modell trainiert")
print()

# ────────────────────────────────────────────
# 6. ⭐ THRESHOLD-OPTIMIERUNG ⭐
# ────────────────────────────────────────────

# Standard-Vorhersage (mit threshold 0.5)
y_pred_standard = model.predict(X_test)
print("=" * 70)
print("STANDARD-MODELL (threshold 0.5):")
print("=" * 70)
print(classification_report(y_test, y_pred_standard))
print()

# Wahrscheinlichkeiten für REGEN (Klasse 1)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Verschiedene Thresholds testen
thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

print("=" * 70)
print("THRESHOLD-OPTIMIERUNG")
print("=" * 70)
print("Threshold | Precision | Recall | F1-Score")
print("─" * 70)

results = []

for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)
    report = classification_report(
        y_test, 
        y_pred, 
        output_dict=True,
        zero_division=0
    )
    
    prec = report['1']['precision']
    rec = report['1']['recall']
    f1 = report['1']['f1-score']
    
    print(f"{threshold:.2f}      | {prec:.2f}       | {rec:.2f}    | {f1:.2f}")
    
    results.append({
        'threshold': threshold,
        'precision': prec,
        'recall': rec,
        'f1': f1
    })

print()

# ────────────────────────────────────────────
# 7. Besten Threshold wählen
# ────────────────────────────────────────────

results_df = pd.DataFrame(results)

# Besten Recall (wichtig für Regenwarnungen!)
best_recall_idx = results_df['recall'].idxmax()
best_threshold_recall = results_df.loc[best_recall_idx, 'threshold']

# Besten F1 (Balance)
best_f1_idx = results_df['f1'].idxmax()
best_threshold_f1 = results_df.loc[best_f1_idx, 'threshold']

print("=" * 70)
print("BESTE THRESHOLDS")
print("=" * 70)
print(f"Maximaler Recall: {best_threshold_recall}")
print(f"  → Precision: {results_df.loc[best_recall_idx, 'precision']:.2f}")
print(f"  → Recall: {results_df.loc[best_recall_idx, 'recall']:.2f}")
print(f"  → F1: {results_df.loc[best_recall_idx, 'f1']:.2f}")
print()

print(f"Maximaler F1: {best_threshold_f1}")
print(f"  → Precision: {results_df.loc[best_f1_idx, 'precision']:.2f}")
print(f"  → Recall: {results_df.loc[best_f1_idx, 'recall']:.2f}")
print(f"  → F1: {results_df.loc[best_f1_idx, 'f1']:.2f}")
print()

# ⭐ WÄHLE HIER: Recall ist wichtiger für Regenwarnungen!
OPTIMAL_THRESHOLD = best_threshold_recall

print(f"🎯 GEWÄHLTER THRESHOLD: {OPTIMAL_THRESHOLD}")
print()

# ────────────────────────────────────────────
# 8. Mit optimalem Threshold neu evaluieren
# ────────────────────────────────────────────

y_pred_optimal = (y_pred_proba >= OPTIMAL_THRESHOLD).astype(int)

print("=" * 70)
print(f"FINALES MODELL (threshold {OPTIMAL_THRESHOLD})")
print("=" * 70)
print(classification_report(y_test, y_pred_optimal))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_optimal)
print(cm)
print(f"True Negatives (TN):  {cm[0, 0]}")
print(f"False Positives (FP): {cm[0, 1]}")
print(f"False Negatives (FN): {cm[1, 0]}")
print(f"True Positives (TP):  {cm[1, 1]}")

accuracy_optimal = (cm[0, 0] + cm[1, 1]) / cm.sum()
print(f"Accuracy: {accuracy_optimal:.4f}")
print()

# ────────────────────────────────────────────
# 9. Lokal speichern
# ────────────────────────────────────────────

os.makedirs("model", exist_ok=True)

# Modell speichern
model_path = "model/rain_classifier.pkl"
joblib.dump(model, model_path)
print(f"✅ Modell gespeichert: {model_path}")

# Config mit optimalem Threshold speichern
config = {
    'model_type': 'RandomForestClassifier',
    'threshold': float(OPTIMAL_THRESHOLD),
    'n_estimators': 100,
    'class_weight': 'balanced',
    'accuracy': float(accuracy_optimal),
    'notes': 'Optimiert für hohen Recall (Regenwarnungen wichtiger)',
    'threshold_analysis': results_df.to_dict(orient='records')
}

config_path = "model/model_config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"✅ Config gespeichert: {config_path}")
print()

# ────────────────────────────────────────────
# 10. In Hopsworks Model Registry hochladen
# ────────────────────────────────────────────

mr = project.get_model_registry()

hw_model = mr.sklearn.create_model(
    name="rain_classifier",
    version=2,
    metrics={
        "accuracy": round(accuracy_optimal, 4),
        "threshold": round(OPTIMAL_THRESHOLD, 2)
    },
    description=f"Random Forest: Regenvorhersage 2h Horizont (Threshold: {OPTIMAL_THRESHOLD})",
    input_example=X_train.iloc[0:1],
    feature_view=feature_view
)

hw_model.save("model")
print("✅ Modell in Model Registry hochgeladen.")
