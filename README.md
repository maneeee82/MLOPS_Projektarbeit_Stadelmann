# MLOPS_Projektarbeit_Stadelmann

# Wetter Windböen Vorhersage

## Überblick

Dieses Projekt implementiert eine FTI-Architektur (Feature-Training-Inference)
zur Vorhersage von Windböen in den nächsten 3 Stunden für fünf Schweizer Städte.
Als Feature Store wird Hopsworks verwendet.

**Ziel:** Klassifikation – `strong_wind_warning` (binär: 1 = Böe >= 50 km/h, 0 = kein starker Wind)
**Konkret:** Gegeben die aktuellen Wetterbedingungen und der 24h-Historie, wird vorhergesagt ob in den nächsten 3 Stunden (t+1h, t+2h, t+3h) eine Windböe >= 50 km/h auftritt.

---

## Datenquelle

- **API:** [Open-Meteo Archive API](https://archive-api.open-meteo.com/)
- **Locations:** Zürich, Basel, Bern, Genf, Lugano
- **Zeitraum:** Letzte 365 Tage bis vor 5 Tagen (stündliche Auflösung)
- **Keine Authentifizierung erforderlich**

---
##=========================================================================================
## Features Pipeline
##=========================================================================================

### Aggregierte Features: Batch / NRT (Near Real-Time)
Berechnet als Rolling Mean über ein 24h-Fenster:

| Feature | Beschreibung |
|---|---|
| `humidity_avg_24h` | Durchschnittliche relative Luftfeuchtigkeit der letzten 24h |
| `temp_avg_24h` | Durchschnittliche Temperatur der letzten 24h |
| `pressure_avg_24h` | Durchschnittlicher Luftdruck der letzten 24h |
| `precip_avg_24h` | Durchschnittlicher Niederschlag der letzten 24h |
| `wind_gusts_avg_24h` | Durchschnittliche Windböen der letzten 24h |

### Aktuelle Features: RT (Real-Time)
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
| `weather_code` | WMO Weather Code |
| `precip` | Aktueller Niederschlag in mm |

### Label

| Feature | Beschreibung |
|---|---|
| `strong_wind_warning` | 1 = maximale Windböe >= 50 km/h in t+1h, t+2h oder t+3h; 0 = kein starker Wind (Klassifikations-Target) |

---

## Architektur

[Open-Meteo API]
|
v
[Feature Pipeline] <- berechnet Rolling-Features (24h), setzt Label (shift -1/-2/-3), schreibt in Hopsworks
|
v
[Hopsworks Feature Store] <- zentrale Datenhaltung (Feature Group + Feature View)
|
v
[Training Pipeline] <- lädt Daten, trainiert Modell, speichert in Model Registry
|
v
[Hopsworks Model Registry]<- versioniertes Modell mit Metriken (Accuracy, F1, Precision, Recall)
|
v
[Inference Pipeline] <- lädt Modell, holt aktuelle Features, gibt Windwarnung aus (0 oder 1)

---
##=========================================================================================
## Training Pipeline
##=========================================================================================

**Datei:** `training_pipeline.py`

### Ablauf

1. **Verbindung** zu Hopsworks via API Key (`.env`)
2. **Feature Group laden** – gespeicherte Wetterdaten aus dem Feature Store
3. **Feature View erstellen oder laden** – definiert welche Features ins Modell fliessen
4. **Train/Test-Split** – 80% Training, 20% Test (zeitlich gemischt)
5. **Encoding** – `weather_code` und `location` werden One-Hot-enkodiert, Spalten zwischen Train/Test angeglichen
6. **Modell trainieren** – Random Forest Regressor
7. **Evaluation** – MAE, RMSE, R² auf dem Test-Set
8. **Speichern** – Modell lokal und in der Hopsworks Model Registry (Versionierung automatisch)

### Feature-Auswahl

Folgende Features fliessen ins Modell:

| Feature | Typ | Begründung |
|---|---|---|
| `temp` | RT | Temperatur beeinflusst atmosphärische Instabilität |
| `humidity` | RT | Hohe Luftfeuchtigkeit bremst Wind durch Grenzschichteffekte |
| `dew_point` | RT | Indikator für Feuchte- und Stabilitätszustand der Luft |
| `cloud_cover` | RT | Bewölkung korreliert mit Frontdurchgängen |
| `cloud_cover_low` | RT | Tiefe Wolken deuten auf bodennahe Instabilität hin |
| `pressure` | RT | Druckgradient ist direkte Ursache von Wind |
| `precip` | RT | Niederschlag tritt oft gemeinsam mit starkem Wind auf |
| `wind_speed` | RT | Aktueller Wind als Basis für die Vorhersage |
| `weather_code` | RT | WMO-Code als kategorisches Signal für Wetterregime |
| `location` | KAT | Geografische Unterschiede im Windverhalten |
| `humidity_avg_24h` | NRT | Feuchtigkeitstrend der letzten 24h |
| `temp_avg_24h` | NRT | Temperaturtrend der letzten 24h |
| `pressure_avg_24h` | NRT | Drucktrend erkennbar (steigend/fallend) |
| `precip_avg_24h` | NRT | Niederschlagstrend als Proxy für Frontaktivität |
| `wind_speed_avg_24h` | NRT | Windtrend der letzten 24h |

**Label:** `wind_speed_next_3h_avg` – durchschnittliche Windgeschwindigkeit der nächsten 3 Stunden

### Modell

- **Algorithmus:** `RandomForestRegressor` (scikit-learn)
- **`n_estimators=100`:** 100 Entscheidungsbäume, Durchschnittswert als Ausgabe
- **`n_jobs=-1`:** Alle verfügbaren CPU-Kerne werden genutzt
- **`random_state=42`:** Reproduzierbarkeit

### Evaluation

Ausgabe nach dem Training:
- **MAE** (Mean Absolute Error) – mittlerer absoluter Fehler in km/h
- **RMSE** (Root Mean Squared Error) – bestraft grosse Ausreisser stärker
- **R²** – Anteil der erklärten Varianz (1.0 = perfekt)
- **Top-10 Feature Importances** – welche Features den grössten Einfluss haben

Alle drei Metriken werden in der Hopsworks Model Registry gespeichert.

### Versionierung

Beim Upload in die Model Registry wird die Version automatisch bestimmt:
- Existiert bereits ein Modell `wind_speed_regressor`, wird die Version des besten Modells (nach **R²**) abgerufen und um 1 erhöht.
- Existiert noch kein Modell, startet die Versionierung bei **Version 1**.

> Dadurch werden frühere Modellversionen nie überschrieben und sind jederzeit
> in der Registry nachvollziehbar.