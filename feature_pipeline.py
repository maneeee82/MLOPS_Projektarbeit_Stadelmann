import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
import hopsworks

#Login
load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))


LOCATIONS = {
    "zurich":  {"lat": 47.38, "lon": 8.54},
    "basel":   {"lat": 47.56, "lon": 7.59},
    "bern":    {"lat": 46.95, "lon": 7.45},
    "geneva":  {"lat": 46.20, "lon": 6.14},
    "lugano":  {"lat": 46.01, "lon": 8.96},
}

def push_to_featurestore(df: pd.DataFrame):
    fs = project.get_feature_store()  # nutzt das bereits erstellte project-Objekt

    fg = fs.get_or_create_feature_group(
        name="weather_features_multiregion",
        version=1,
        description="Stundliche Wetterdaten fuer mehrere Orte",
        primary_key=["location", "timestamp"],
        event_time="timestamp"
    )

    fg.insert(df, write_options={"wait_for_job": True})
    print(f"Feature Group aktualisiert: {len(df)} Zeilen eingefuegt.")


def run_all_locations():
    all_features = []

    end   = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")

    for loc_name, coords in LOCATIONS.items():
        print(f"\nLade Daten fuer: {loc_name}")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude":   coords["lat"],
            "longitude":  coords["lon"],
            "start_date": start,
            "end_date":   end,
            "hourly":     "temperature_2m,relativehumidity_2m,cloudcover,surface_pressure,precipitation",
            "timezone":   "Europe/Zurich"
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()["hourly"]

        df = pd.DataFrame(data)
        df.rename(columns={
            "time":                  "timestamp",
            "temperature_2m":        "temp",
            "relativehumidity_2m":   "humidity",
            "cloudcover":            "cloud_cover",
            "surface_pressure":      "pressure",
            "precipitation":         "precip"
        }, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["location"]  = loc_name

        df = df.sort_values("timestamp").reset_index(drop=True)
        df["humidity_avg_24h"] = df["humidity"].rolling(window=24, min_periods=1).mean()
        df["temp_avg_24h"]     = df["temp"].rolling(window=24, min_periods=1).mean()
        df["pressure_avg_24h"] = df["pressure"].rolling(window=24, min_periods=1).mean()
        df["rain_t1"]          = df["precip"].shift(-1)
        df["rain_t2"]          = df["precip"].shift(-2)
        df["will_rain_in_2h"]  = ((df["rain_t1"] > 0.1) | (df["rain_t2"] > 0.1)).astype(int)

        features_df = df[[
            "timestamp", "location",
            "humidity_avg_24h", "temp_avg_24h", "pressure_avg_24h",
            "cloud_cover", "humidity", "will_rain_in_2h"
        ]].copy()
        features_df.rename(columns={"humidity": "humidity_current"}, inplace=True)
        features_df = features_df.dropna()

        all_features.append(features_df)
        print(f"  → {len(features_df)} Zeilen vorbereitet")

    combined_df = pd.concat(all_features, ignore_index=True)
    print(f"\nTotal: {len(combined_df)} Zeilen fuer {len(LOCATIONS)} Orte")
    push_to_featurestore(combined_df)


if __name__ == "__main__":
    run_all_locations()