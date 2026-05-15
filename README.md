# MLOps Projektarbeit – Wetter Windböen Vorhersage
# Manuel Stadelmann, 28.04.2026

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
- **Zeitraum:** stündliche Historiendaten, ca. 1 Jahr Rückblick 
- **Keine Authentifizierung erforderlich**


## Architektur
1. **Open-Meteo API** → Feature Pipeline
2. **Feature Pipeline** → Hopsworks Feature Store
3. **Hopsworks Feature Store** → Training Pipeline → Hopsworks Model Registry
4. **Hopsworks Model Registry** → Inference Pipeline → Windwarnung (0/1)

## Features (Überblick)
- **Aggregierte Features (Batch/NRT):** humidity_avg_24h, temp_avg_24h, pressure_avg_24h, precip_avg_24h, wind_gusts_avg_24h
- **Echtzeitfeatures (RT):** temp, humidity, dew_point, cloud_cover, cloud_cover_low, pressure, precip, wind_gusts, weather_code
- **Kategorische Features:** location, weather_code (beide One-Hot-Encoded)
- **Label:** strong_wind_warning (basierend auf Windboeen in t+1h, t+2h, t+3h)

## Features (Detailliert)

### Aggregierte Features (Batch/NRT)
Diese Features werden **ueber mehrere Timesteps** (24h Rolling Window) aggregiert:

- `humidity_avg_24h` – Durchschnittliche Luftfeuchtigkeit der letzten 24h
- `temp_avg_24h` – Durchschnittliche Temperatur der letzten 24h
- `pressure_avg_24h` – Durchschnittlicher Luftdruck der letzten 24h
- `precip_avg_24h` – Durchschnittliche Niederschlagsmenge der letzten 24h
- `wind_gusts_avg_24h` – Durchschnittliche Windböen der letzten 24h

**Charakter:** Diese Features werden **aus historischen Daten** im Feature Store 
berechnet (Rolling Window über die letzten 24h) und enthalten **keine Zukunftsinformationen**. 
Sie sind daher punkt-in-der-zeit korrekt und können offline (Batch) vorab berechnet werden.
Sie repräsentieren die **Vergangenheit** und sind damit determinstisch verfügbar.

### Echtzeitfeatures (RT – Real-Time)
Diese Features sind **zum Inferenzzeitpunkt verfügbar**:

- `temp` – Aktuelle Temperatur (°C)
- `humidity` – Aktuelle Luftfeuchte (%)
- `dew_point` – Aktueller Taupunkt (°C)
- `cloud_cover` – Bewoelkungsgrad (%)
- `cloud_cover_low` – Bewoelkung niedrige Schichten (%)
- `pressure` – Aktueller Luftdruck (hPa)
- `precip` – Aktueller Niederschlag (mm)
- `wind_gusts` – Aktuelle Windboeen (km/h)
- `weather_code` – WMO Wetterkode (kategorisch)

**Charakter:** Diese Features sind **zum Inferenzzeitpunkt bekannt** und werden 
direkt aus dem Feature Store fuer den aktuellen Datenpunkt geladen. Im Gegensatz 
zu den aggregierten Features hängen sie nicht von der Vergangenheit ab und werden 
(in einer echten Produktionsumgebung) erst zur Inferenzzeit live aktualisiert. 
Sie repraesentieren den **Jetzt-Zustand**.

### Limitation: Datenquelle
Die Open-Meteo Archive API liefert nur historische Daten. Um echte Echtzeitdaten 
zu simulieren, wird bei der Inferenz der **letzte verfuegbare Datenpunkt** aus 
dem Feature Store als "aktueller" Datenpunkt verwendet. In einer Produktionsumgebung 
würde hier die Open-Meteo **Forecast API** verwendet, um echte RT-Daten zu erhalten.


## Pipelines

### **`feature_pipeline.py`** 
- Daten abrufen (Open-Meteo Archive API)
- 24h Rolling-Features berechnen 
- Label setzen (starker Wind in t+1h, t+2h, t+3h?)
- In Hopsworks Feature Store schreiben
- Excel fuer Datenkontrolle generieren

### **`training_pipeline.py`**
- Daten aus Feature Store laden
- Feature Engineering: One-Hot-Encoding fuer weather_code, location
- XGBoost Classifier trainieren (80/20 zufaelliger Split)
- Threshold via F1-Score optimieren
- Modell + Threshold in Hopsworks Model Registry speichern
  (registriert mit `metric=f1` fuer spätere Versionsselektion)

### **`inference_pipeline.py`**
- Beste Modellversion aus Registry laden (`metric=f1, direction=max`)
- **Aggregierte Features (Batch/NRT):** humidity_avg_24h, temp_avg_24h, pressure_avg_24h, precip_avg_24h, wind_gusts_avg_24h
- **Echtzeitfeatures (RT):** cloud_cover, wind_gusts, precip, weather_code, temp, humidity, dew_point, cloud_cover_low, pressure
- One-Hot-Encoding anwenden (gleich wie Training)
- Wahrscheinlichkeit berechnen
- Warnung ausgeben (0/1)


### Modell: XGBoost Classifier
Mehrere Entscheidungsbäume werden nacheinander trainiert, wobei jeder Baum die Fehler des vorherigen korrigiert.

**Konfiguration:** 
- Max. 500 Bäume (Early Stopping nach 30 Runden ohne Verbesserung)
- Max. Tiefe 6
- Learning Rate 0.05
- Subsampling 0.8
- `scale_pos_weight` fuer Klassenungleichgewicht
- Eval-Metrik AUCPR (robuster bei Imbalance)
- Threshold-Optimierung via F1-Score



## Ausführungsreihenfolge SETUP

Getestet mit Python 3.13.12

1. Repository klonen
```bash
   git clone https://github.com/maneeee82/MLOPS_Projektarbeit_Stadelmann.git
   cd MLOPS_Projektarbeit_Stadelmann
```
2. Python venv erstellen und aktivieren
```bash
   python3 -m venv .venv
```  
   Linux:
```bash
   source .venv/bin/activate
```
3. Dependencies installieren
```bash
   pip install -r requirements.txt
```
4. in Hopsworks einloggen Projekt erstellen und API-Key erstellen (full scope)
5. .env Datei konfigurieren
```bash
   cp .env.example .env
```
   Dann .env mit Hopsworks API Key eintragen



## Ausführungsreihenfolge Pipelines

```bash
python feature_pipeline.py   # 1. Features berechnen und in Featurestore schreiben (Dauert ca. 3-4min)
python training_pipeline.py  # 2. Modell trainieren (Dauert ca. 3-4min)
python inference_pipeline.py # 3. Inferenz durchführen
```

## Limitationen

- **Kein Scheduler:** Feature Pipeline muss manuell ausgeführt werden. 
  In der Praxis würde ein Scheduler (z.B. Apache Airflow, cron) regelmäßig neue Daten einspeisen.

- **Klassenungleichgewicht:** Adressiert via `scale_pos_weight` + Threshold-Optimierung, 
  nicht vollständig gelöst. Ein Oversampling oder dedizierte Strategie könnte hier helfen.

- **Kein zeitlicher Train/Test-Split:** Split ist zufällig, nicht chronologisch. 
  Dies führt zu Data Leakage (das Modell "sieht" zufällig auch zukünftige Daten im Training). 
  In der Praxis sollte der Split **zeitbasiert** sein: z.B. Trainingsdaten = erste 80%, 
  Testdaten = letzte 20%.

- **Keine echten RT-Daten:** Die Open-Meteo Archive API liefert nur historische Daten. 
  Bei der Inferenz wird der **letzte verfügbare Datenpunkt** als "aktueller" Punkt verwendet. 
  In einer Produktionsumgebung würde die Open-Meteo **Forecast API** echte Live-Daten liefern.