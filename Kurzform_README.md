# MLOps Projektarbeit – Wetter Windböen Vorhersage

## Überblick

FTI-Architektur (Feature-Training-Inference) zur Vorhersage von Windböen >= 50 km/h in den nächsten 3 Stunden für fünf Schweizer Städte (Zürich, Basel, Bern, Genf, Lugano) via Hopsworks Feature Store.

**Ziel:** Binäre Klassifikation – `strong_wind_warning`
- 1 = Windböe >= 50 km/h tritt irgendwann in den nächsten 3 Stunden auf (t+1h, t+2h oder t+3h)
- 0 = kein starker Wind

**Datenquelle:** [Open-Meteo Archive API](https://archive-api.open-meteo.com/) – letzte 365 Tage, stündlich, keine Authentifizierung

---

## Architektur

1. **Open-Meteo API** → Feature Pipeline
2. **Feature Pipeline** → Hopsworks Feature Store
3. **Hopsworks Feature Store** → Training Pipeline → Hopsworks Model Registry
4. **Hopsworks Model Registry** → Inference Pipeline → Windwarnung (0/1)

---

## Features

**Rolling Features (24h-Mittelwert):** Temperatur, Luftfeuchtigkeit, Luftdruck, Niederschlag, Windböen

**Aktuelle Features (Inferenzzeit):** Bewölkung, Taupunkt, Windböen, Niederschlag, Wetter-Code (One-Hot), Standort (One-Hot)

**Label:** `strong_wind_warning` – gesetzt via Shift auf t+1h/t+2h/t+3h

---

## Pipelines

**`feature_pipeline.py`** – Daten abrufen, Rolling-Features berechnen, Label setzen, in Hopsworks schreiben

**`training_pipeline.py`** – Daten laden, XGBoost Classifier trainieren (80/20 Split), Threshold via F1 optimieren, Modell versioniert in Registry speichern

**`inference_pipeline.py`** – Neueste Features laden, Modell aus Registry laden, Wahrscheinlichkeit berechnen, Warnung ausgeben

### Modell: XGBoost Classifier

Sequenzielles Ensemble aus Entscheidungsbäumen mit expliziter Regularisierung.

Konfiguration: 200 Bäume, max. Tiefe 6, Learning Rate 0.05, `scale_pos_weight` für Klassenungleichgewicht, Threshold-Optimierung via F1-Score.

---

## Setup

```bash
pip install -r requirements.txt

python feature_pipeline.py   # 1. Features berechnen
python training_pipeline.py  # 2. Modell trainieren
python inference_pipeline.py # 3. Inferenz
```

`.env` benötigt:

```
HOPSWORKS_API_KEY=<dein_api_key>
HOPSWORKS_PROJECT=<dein_projektname>
```

---

## Limitationen

| Problem | Bemerkung |
|---|---|
| Kein Scheduler | Feature Pipeline muss manuell ausgeführt werden |
| Klassenungleichgewicht | Adressiert via `scale_pos_weight` + Threshold-Optimierung, nicht vollständig gelöst |
| Kein zeitlicher Train/Test-Split | Split ist zufällig, nicht chronologisch |
| Keine echten RT-Daten | Archive API liefert keine Echtzeit-Daten; RT-Features werden simuliert |