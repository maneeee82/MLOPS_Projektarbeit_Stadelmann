# MLOps Projektarbeit – Wetter Windböen Vorhersage

## Überblick

Dieses Projekt implementiert eine FTI-Architektur (Feature-Training-Inference)
zur Vorhersage von starken Windböen in den nächsten 3 Stunden für fünf Schweizer Städte.
Als Feature Store wird Hopsworks verwendet.

**Ziel:** Binäre Klassifikation – `strong_wind_warning`
- 1 = Windböe >= 50 km/h tritt irgendwann in den nächsten 3 Stunden auf (t+1h, t+2h oder t+3h), 
- 0 = kein starker Wind


## Datenquelle

- **API:** [Open-Meteo Archive API](https://archive-api.open-meteo.com/)
- **Locations:** Zürich, Basel, Bern, Genf, Lugano
- **Zeitraum:** Letzte 365 Tage bis vor 5 Tagen (stündliche Auflösung)
- **Keine Authentifizierung erforderlich**

## Architektur (FTI)

[Open-Meteo API]
|
v
[Feature Pipeline]

Rohdaten abrufen
Rolling-Features berechnen (24h)
Label setzen (shift -1/-2/-3)
Schreiben in Hopsworks Feature Store
|
v
[Hopsworks Feature Store]
Feature Group: zentrale Datenhaltung
Feature View: definiert Feature-Auswahl für Training
|
v
[Training Pipeline]
Daten laden via Feature View
Modell trainieren (XGBoost Classifier)
Evaluation (Accuracy, F1, Precision, Recall, Threshold)
Speichern in Hopsworks Model Registry (versioniert)
|
v
[Hopsworks Model Registry]
Versioniertes Modell mit Metriken
|
v
[Inference Pipeline]
Aktuelle Features laden (neueste Zeile aus Feature View)
Modell aus Registry laden
Wahrscheinlichkeit berechnen, mit Threshold vergleichen
Ausgabe: Windwarnung ja/nein
---
##=========================================================================================
## Features Pipeline
##=========================================================================================

#### Aggregierte Features – Batch / NRT (Near Real-Time)
Berechnet als Rolling Mean über ein 24h-Fenster:

| Feature | Beschreibung |
|---|---|
| `humidity_avg_24h` | Durchschnittliche relative Luftfeuchtigkeit der letzten 24h |
| `temp_avg_24h` | Durchschnittliche Temperatur der letzten 24h |
| `pressure_avg_24h` | Durchschnittlicher Luftdruck der letzten 24h |
| `precip_avg_24h` | Durchschnittlicher Niederschlag der letzten 24h |
| `wind_gusts_avg_24h` | Durchschnittliche Windböen der letzten 24h |

### Aktuelle Features – RT (Real-Time)
Werte der aktuellen Stunde, erst zur Inferenzzeit bekannt:

| Feature | Beschreibung |
|---|---|
| `cloud_cover` | Gesamtbewölkung in % |
| `cloud_cover_low` | Tiefbewölkung in % |
| `temp` | Aktuelle Temperatur in °C |
| `humidity` | Aktuelle relative Luftfeuchtigkeit in % |
| `dew_point` | Taupunkt in °C |
| `pressure` | Aktueller Luftdruck in hPa |
| `wind_gusts` | Maximale Windböe der vergangenen Stunde in km/h |
| `weather_code` | WMO Weather Code (kategorisch, One-Hot-enkodiert) |
| `precip` | Aktueller Niederschlag in mm |
| `location` | Standort (kategorisch, One-Hot-enkodiert) |

### Label

| Feature | Beschreibung |
|---|---|
| `strong_wind_warning` | 1 = maximale Windböe >= 50 km/h in t+1h, t+2h oder t+3h; 0 = kein starker Wind |

---

## Feature Pipeline

**Datei:** `feature_pipeline.py`

### Ablauf

1. Rohdaten stündlich für 5 Standorte via Open-Meteo API abrufen
2. Rolling-Features über 24h-Fenster berechnen (`.rolling(24).mean()`)
3. Label `strong_wind_warning` via Shift berechnen:
   - Wenn `wind_gusts` in t+1h, t+2h **oder** t+3h >= 50 km/h → Label = 1
4. DataFrame mit Features + Label erstellen
5. Feature Group in Hopsworks erstellen oder laden (Primary Key: `location` + `timestamp`)
6. Daten in Feature Group schreiben

---

##=========================================================================================
## Training Pipeline
##=========================================================================================

**Datei:** `training_pipeline.py`

### Ablauf

1. Verbindung zu Hopsworks via API Key (`.env`)
2. Feature Group laden
3. Feature View erstellen oder laden
4. Train/Test-Split (80/20, zeitlich gemischt)
5. Encoding: `weather_code` und `location` werden One-Hot-enkodiert,
   Spalten zwischen Train und Test angeglichen
6. Modell trainieren: **XGBoost Classifier**
7. Optimalen Klassifikations-Threshold bestimmen (anhand F1-Score)
8. Evaluation auf dem Test-Set
9. Modell + Threshold lokal speichern und in Hopsworks Model Registry hochladen

### Modell: XGBoost Classifier

XGBoost (Extreme Gradient Boosting) ist ein Ensemble-Verfahren, das sequenziell
viele schwache Entscheidungsbäume trainiert. Jeder neue Baum korrigiert die
Fehler des vorherigen (Gradient Boosting). Die finale Vorhersage ist:

$$\hat{y} = \sum_{k=1}^{K} f_k(x)$$

Gegenüber klassischem Gradient Boosting fügt XGBoost explizite Regularisierung
hinzu, um Overfitting zu reduzieren:

$$\mathcal{L} = \sum_i l(\hat{y}_i, y_i) + \sum_k \left(\gamma T_k + \frac{1}{2}\lambda \|w_k\|^2\right)$$

Dabei ist $$T_k$$ die Anzahl Blätter und $$w_k$$ die Blattgewichte des $$k$$-ten Baums.

**Konfiguration:**

| Parameter | Wert | Bedeutung |
|---|---|---|
| `n_estimators` | 200 | Anzahl Bäume |
| `max_depth` | 6 | Maximale Baumtiefe |
| `learning_rate` | 0.05 | Schrittgrösse pro Baum |
| `scale_pos_weight` | berechnet | Korrektur für Klassenungleichgewicht |
| `eval_metric` | `logloss` | Verlustfunktion |
| `use_label_encoder` | False | Kein internes Encoding |

### Threshold-Optimierung

Da starke Windböen selten sind (Klassenungleichgewicht), wird der
Klassifikations-Threshold nicht fix auf 0.5 gesetzt, sondern anhand
des F1-Scores auf dem Test-Set optimiert. Der optimale Threshold wird
zusammen mit dem Modell in der Registry gespeichert.

### Evaluation

| Metrik | Beschreibung |
|---|---|
| Accuracy | Anteil korrekt klassifizierter Samples |
| F1-Score | Harmonisches Mittel aus Precision und Recall |
| Precision | Anteil echter Warnungen unter allen ausgegebenen Warnungen |
| Recall | Anteil erkannter echter Warnungen |
| Optimaler Threshold | Schwellwert für die Klassenentscheidung |

### Versionierung

Beim Upload in die Model Registry wird die Version automatisch inkrementiert:
- Existiert bereits ein Modell `wind_gust_classifier`, wird die nächste Version vergeben
- Existiert kein Modell, startet die Versionierung bei Version 1
- Frühere Versionen werden nie überschrieben und bleiben in der Registry nachvollziehbar
- Gespeicherte Metriken pro Version: Accuracy, F1, Precision, Recall, Threshold

---

## Inference Pipeline

**Datei:** `inference_pipeline.py`

### Ablauf

1. Verbindung zu Hopsworks via API Key (`.env`)
2. Feature View laden
3. Neueste verfügbare Features aus dem Feature Store laden
4. Modell und Threshold aus der Model Registry laden (neueste Version)
5. One-Hot-Encoding der kategorischen Features (`weather_code`, `location`),
   Spalten auf Trainingsschema angleichen
6. Wahrscheinlichkeit berechnen (`predict_proba`)
7. Schwellwert-Vergleich: `P >= threshold` → Windwarnung = 1
8. Ausgabe: Windwarnung für jeden Standort (0 oder 1)

---

## Setup und Ausführung

### Voraussetzungen

- Python < 3.14 (Hopsworks-Anforderung)
- Hopsworks Account auf [hopsworks.ai](https://hopsworks.ai)
- `.env`-Datei mit folgendem Inhalt:

## Setup und Ausführung

### Voraussetzungen

- Python < 3.14 (Hopsworks-Anforderung)
- Hopsworks Account auf [hopsworks.ai](https://hopsworks.ai)
- `.env`-Datei mit folgendem Inhalt:

HOPSWORKS_API_KEY=<dein_api_key>
HOPSWORKS_PROJECT=<dein_projektname>


### Installation

```bash
pip install -r requirements.txt

Reihenfolge der Ausführung

# 1. Features berechnen und in Hopsworks schreiben
python feature_pipeline.py

# 2. Modell trainieren und in Registry speichern
python training_pipeline.py

# 3. Inferenz durchführen
python inference_pipeline.py

Abhängigkeiten
Siehe requirements.txt. Wichtigste Pakete:

Paket	Zweck
hopsworks	Feature Store und Model Registry
xgboost	Klassifikationsmodell
scikit-learn	Preprocessing, Metriken, train_test_split
pandas	Datenverarbeitung
requests	API-Abruf
python-dotenv	Laden der .env-Datei
==============================================================================
Limitationen und Reflexion
Feature Pipeline läuft nicht automatisch: Die Features werden nicht
automatisch aktualisiert. Für produktiven Einsatz müsste die Pipeline
per Scheduler (z.B. Cron, GitHub Actions) regelmässig ausgeführt werden.
Im aktuellen Zustand sind die Daten maximal einige Tage alt.

Klassenungleichgewicht: Starke Windböen (>= 50 km/h) sind selten,
was zu einem unbalancierten Datensatz führt. Dies wird mit
scale_pos_weight und Threshold-Optimierung adressiert, aber nicht vollständig gelöst.

Kein Point-in-Time-Split: Train/Test-Split ist zeitlich gemischt (random),
nicht strikt zeitlich getrennt. In einem produktiven System sollte
der Test-Split immer die zeitlich späteren Daten umfassen.

Simulierte RT-Features: Da die Open-Meteo Archive API keine echten
Echtzeit-Daten liefert, werden die RT-Features aus historischen Daten
simuliert (neueste verfügbare Zeile).