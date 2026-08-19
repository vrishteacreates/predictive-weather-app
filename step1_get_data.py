import pandas as pd
import requests

# Location: Mumbai / Mira Bhayandar coordinates
LATITUDE = 19.2812
LONGITUDE = 72.8561

URL = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
    ],
    "timezone": "auto",
}

print("Downloading real weather data...")
res = requests.get(URL, params=params)
data = res.json()["hourly"]

# Build table and save
df = pd.DataFrame({
    "Date": data["time"],
    "Temperature": data["temperature_2m"],
    "Humidity": data["relative_humidity_2m"],
    "Pressure": data["surface_pressure"],
    "Wind_Speed": data["wind_speed_10m"],
}).dropna()

df.to_csv("weather.csv", index=False)
print("SUCCESS: 'weather.csv' created with", len(df), "rows!")