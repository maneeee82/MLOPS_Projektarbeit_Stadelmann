# MLOPS_Projektarbeit_Stadelmann

# Wetter Regen Vorhersage

## Überblick

Dieses Projekt implementiert eine FTI-Architektur (Feature-Training-Inference)
zur Vorhersage von Regen in den nächsten 2 Stunden für fünf Schweizer Städte.
Als Feature Store wird Hopsworks verwendet.

**Ziel:** Binäre Klassifikation – `will_rain_in_2h` (1 = Regen, 0 = kein Regen)

---

## Datenquelle

- **API:** [Open-Meteo Archive API](https://archive-api.open-meteo.com/)
- **Locations:** Zürich, Basel, Bern, Genf, Lugano
- **Zeitraum:** Letzte 365 Tage bis vor 5 Tagen (stündliche Auflösung)
- **Keine Authentifizierung erforderlich**

---

## Features

### Aggregierte Features: Batch / NRT (Near Real-Time)
Berechnet als Rolling Mean über ein 24h-Fenster:

| Feature | Beschreibung |
|---|---|
| `humidity_avg_24h` | Durchschnittliche relative Luftfeuchtigkeit der letzten 24h |
| `temp_avg_24h` | Durchschnittliche Temperatur der letzten 24h |
| `pressure_avg_24h` | Durchschnittlicher Luftdruck der letzten 24h |
| `precip_avg_24h` | Durchschnittlicher Niederschlag der letzten 24h |

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
| `wind_speed` | Windgeschwindigkeit in km/h |
| `weather_code` | WMO Weather Code |
| `precip` | Aktueller Niederschlag in mm |

### Label

| Feature | Beschreibung |
|---|---|
| `will_rain_in_2h` | 1 = Niederschlag ≥ 0.1mm in t+1h oder t+2h, sonst 0 |

---

## Architektur

```
[Open-Meteo API]
      │
      ▼
[Feature Pipeline]        ← berechnet Rolling-Features, setzt Label, schreibt in Hopsworks
      │
      ▼
[Hopsworks Feature Store] ← zentrale Datenhaltung (Feature Group + Feature View)
      │
      ▼
[Training Pipeline]       ← lädt Daten, trainiert Modell, speichert in Model Registry
      │
      ▼
[Hopsworks Model Registry]← versioniertes Modell mit Metriken
      │
      ▼
[Inference Pipeline]      ← lädt Modell, holt aktuelle Features, gibt Vorhersage aus
```

---

## Training Pipeline

**Datei:** `training_pipeline.py`

### Ablauf

1. **Verbindung** zu Hopsworks via API Key (`.env`)
2. **Feature Group laden** – gespeicherte Wetterdaten aus dem Feature Store
3. **Feature View erstellen oder laden** – definiert welche Features ins Modell fliessen
4. **Train/Test-Split** – 80% Training, 20% Test (zeitlich gemischt)
5. **Modell trainieren** – Random Forest Classifier
6. **Evaluation** – Classification Report auf dem Test-Set
7. **Speichern** – Modell lokal und in der Hopsworks Model Registry

### Feature-Auswahl

Folgende Features fliessen ins Modell:

| Feature | Typ | Begruendung |
|---|---|---|
| `temp` | RT | Temperatur beeinflusst Kondensation |
| `humidity` | RT | Hohe Luftfeuchtigkeit = Vorbote von Regen |
| `dew_point` | RT | Kleiner Abstand Taupunkt/Temperatur = Luft nahe Sättigung |
| `cloud_cover` | RT | Bewölkung direkt mit Niederschlag korreliert |
| `cloud_cover_low` | RT | Tiefe Wolken sind besonders regenrelevant |
| `pressure` | RT | Fallender Luftdruck = Schlechtwetterfront |
| `wind_speed` | RT | Frontdurchgang oft mit Wind verbunden |
| `humidity_avg_24h` | NRT | Feuchtigkeitstrend der letzten 24h |
| `temp_avg_24h` | NRT | Temperaturtrend der letzten 24h |
| `pressure_avg_24h` | NRT | Drucktrend erkennbar (steigend/fallend) |

**Bewusst ausgeschlossen:**

| Feature | Grund |
|---|---|
| `precip` | Aktueller Niederschlag ist quasi das Label selbst → Data Leakage |
| `precip_avg_24h` | Gleiche Problematik, zu stark mit Label korreliert |
| `weather_code` | WMO-Code kodiert oft direkt ob es regnet → Data Leakage |

### Modell

- **Algorithmus:** `RandomForestClassifier` (scikit-learn)
- **`class_weight="balanced"`:** Regen ist seltener als kein Regen. Ohne diese Einstellung lernt das Modell einfach "immer kein Regen" und hat trotzdem hohe Accuracy – was wertlos ist.
- **`n_estimators=100`:** 100 Entscheidungsbäume, Mehrheitsentscheid
- **`random_state=42`:** Reproduzierbarkeit

### Evaluation

Ausgabe nach dem Training:
- **Classification Report** (Precision, Recall, F1 pro Klasse)
- **Accuracy** wird als Metrik in der Model Registry gespeichert

> Hinweis: Bei unbalancierten Klassen ist **Recall für Klasse 1** (Regen) die
> wichtigste Metrik. Ein Modell das Regen oft verpasst ist schlechter als eines
> das gelegentlich falschen Alarm schlägt.
