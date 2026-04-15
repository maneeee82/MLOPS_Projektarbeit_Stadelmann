import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
import hopsworks

# Login
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
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="weather_features_multiregion",
        version=1,
        description="Stündliche Wetterdaten fuer mehrere Orte",
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
                "wind_speed_10m",
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
            "precipitation":        "precip", #Niederschlag
            "wind_speed_10m":       "wind_speed",
            "weather_code":         "weather_code"
        }, inplace=True)

        # Timestamp korrekt parsen (API liefert lokale Zeit Europe/Zurich)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["location"] = loc_name

        df = df.sort_values("timestamp").reset_index(drop=True)

        # Rolling averages (groupby hier technisch redundant, aber konsistent fuer spaeteres concat)
        for col, new_col in [
            ("humidity", "humidity_avg_24h"),
            ("temp",     "temp_avg_24h"),
            ("pressure", "pressure_avg_24h"),
            ("precip",   "precip_avg_24h"),
        ]:
            df[new_col] = df[col].rolling(window=24, min_periods=1).mean()

        # Label: Regen in den nächsten 2 Stunden (Shift um 1 und 2 Zeilen)
        df["rain_t1"] = df["precip"].shift(-1)
        df["rain_t2"] = df["precip"].shift(-2)

        df["will_rain_in_2h"] = (
            (df["rain_t1"] >= 0.1) | (df["rain_t2"] >= 0.1) #wenn Regen in 1 oder 2h dann 1=Regen
        ).where(df["rain_t1"].notna() & df["rain_t2"].notna()).astype("Int64")

        # Hilfsspalten entfernen
        df.drop(columns=["rain_t1", "rain_t2"], inplace=True)

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
            "wind_speed",
            "weather_code",
            "humidity_avg_24h",
            "temp_avg_24h",
            "pressure_avg_24h",
            "precip_avg_24h",
            "will_rain_in_2h"
        ]].copy()

        # Zeilen mit NaN entfernen (v.a. Label-NaN der letzten 2 Zeilen)
        features_df = features_df.dropna()

        all_features.append(features_df)
        print(f"  → {len(features_df)} Zeilen vorbereitet")

    combined_df = pd.concat(all_features, ignore_index=True)
    print(f"\nTotal: {len(combined_df)} Zeilen fuer {len(LOCATIONS)} Orte")
    
    # Speichern in Excel zur Kontrolle
    combined_df.to_excel("weather_data_check.xlsx", index=False)
    print("Excel-Datei gespeichert: weather_data_check.xlsx")
    
    push_to_featurestore(combined_df)


if __name__ == "__main__":
    run_all_locations()