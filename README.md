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

### Aggregierte Features: Batch / NRT (Near Real-Time) )
Berechnet als Rolling Mean über ein 24h-Fenster:

| Feature | Beschreibung |
| `humidity_avg_24h` | Durchschnittliche relative Luftfeuchtigkeit der letzten 24h |
| `temp_avg_24h` | Durchschnittliche Temperatur der letzten 24h |
| `pressure_avg_24h` | Durchschnittlicher Luftdruck der letzten 24h |
| `precip_avg_24h` | Durchschnittlicher Niederschlag der letzten 24h |

### Aktuelle Features: RT (Real-Time)
Werte der aktuellen Stunde, erst zur Inferenzzeit bekannt:

| Feature | Beschreibung |
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
| `will_rain_in_2h` | 1 = Niederschlag ≥ 0.1mm in t+1h oder t+2h, sonst 0 |

---

## Architektur