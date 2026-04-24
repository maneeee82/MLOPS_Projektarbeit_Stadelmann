import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
import hopsworks

# Login
load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))

THRESHOLD_KMH = 50  # Beaufort 7 – Starker Wind

LOCATIONS = {
    "zurich":  {"lat": 47.38, "lon": 8.54},
    "basel":   {"lat": 47.56, "lon": 7.59},
    "bern":    {"lat": 46.95, "lon": 7.45},
    "geneva":  {"lat": 46.20, "lon": 6.14},
    "lugano":  {"lat": 46.01, "lon": 8.96},
}


def push_to_featurestore(df: pd.DataFrame):
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="weather_features_multiregion",
        version=1,
        description="Stündliche Wetterdaten fuer mehrere Orte",
        primary_key=["location", "timestamp"],
        event_time="timestamp"
    )

    fg.insert(df, write_options={"wait_for_job": True})
    print(f"Feature Group aktualisiert: {len(df)} Zeilen eingefügt.")


def run_all_locations():
    all_features = []

    end   = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")

    for loc_name, coords in LOCATIONS.items():
        print(f"\nLade Daten fuer: {loc_name}")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude":  coords["lat"],
            "longitude": coords["lon"],
            "start_date": start,
            "end_date":   end,
            "hourly": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "cloud_cover",
                "cloud_cover_low",
                "surface_pressure",
                "precipitation",
                "wind_gusts_10m", 
                "weather_code"
            ]),
            "timezone": "Europe/Zurich"
        }

        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()["hourly"]

        df = pd.DataFrame(data)
        df.rename(columns={
            "time":                 "timestamp",
            "temperature_2m":       "temp",
            "relative_humidity_2m": "humidity",
            "dew_point_2m":         "dew_point",
            "cloud_cover":          "cloud_cover",
            "cloud_cover_low":      "cloud_cover_low",
            "surface_pressure":     "pressure",
            "precipitation":        "precip",
            "wind_gusts_10m":       "wind_gusts",  
            "weather_code":         "weather_code"
        }, inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["location"] = loc_name
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Rolling averages
        for col, new_col in [
            ("humidity",    "humidity_avg_24h"),
            ("temp",        "temp_avg_24h"),
            ("pressure",    "pressure_avg_24h"),
            ("precip",      "precip_avg_24h"),
            ("wind_gusts",  "wind_gusts_avg_24h"),   
        ]:
            df[new_col] = df[col].rolling(window=24, min_periods=24).mean()

        df = df.dropna(subset=["wind_gusts_avg_24h"])   

        # --- LABEL: Klassifikation ---
        # Max-Böen in den nächsten 3h berechnen
        df["wind_t1"] = df["wind_gusts"].shift(-1)  
        df["wind_t2"] = df["wind_gusts"].shift(-2)  
        df["wind_t3"] = df["wind_gusts"].shift(-3)   

        wind_max_next_3h = df[["wind_t1", "wind_t2", "wind_t3"]].max(axis=1)

        # Binaeres Label: 1 = Starker Wind (>= 50 km/h), 0 = kein starker Wind
        df["strong_wind_warning"] = (wind_max_next_3h >= THRESHOLD_KMH).astype(int)

        df.drop(columns=["wind_t1", "wind_t2", "wind_t3"], inplace=True)

        df = df.dropna(subset=["strong_wind_warning"])

        # Feature-Auswahl
        features_df = df[[
            "timestamp",
            "location",
            "temp",
            "humidity",
            "dew_point",
            "cloud_cover",
            "cloud_cover_low",
            "pressure",
            "precip",
            "wind_gusts",          
            "weather_code",
            "humidity_avg_24h",
            "temp_avg_24h",
            "pressure_avg_24h",
            "precip_avg_24h",
            "wind_gusts_avg_24h",   
            "strong_wind_warning",
        ]].copy()

        all_features.append(features_df)
        print(f"  → {len(features_df)} Zeilen vorbereitet")

        counts = features_df["strong_wind_warning"].value_counts()
        pct = features_df["strong_wind_warning"].mean() * 100
        print(f"     Starker Wind (1): {counts.get(1,0)} ({pct:.1f}%)  |  Kein starker Wind (0): {counts.get(0,0)}")

    combined_df = pd.concat(all_features, ignore_index=True)
    print(f"\nTotal: {len(combined_df)} Zeilen fuer {len(LOCATIONS)} Orte")

    combined_df.to_excel("weather_data_check.xlsx", index=False)
    print("Excel-Datei gespeichert: weather_data_check.xlsx")

    push_to_featurestore(combined_df)


if __name__ == "__main__":
    run_all_locations()
