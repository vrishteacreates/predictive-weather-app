import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


# 1. Define same architecture
class WeatherLSTM(nn.Module):

  def __init__(self):
    super().__init__()
    self.lstm = nn.LSTM(
        input_size=4, hidden_size=64, num_layers=2, batch_first=True
    )
    self.fc = nn.Linear(64, 4)

  def forward(self, x):
    out, _ = self.lstm(x)
    return self.fc(out[:, -1, :])


# 2. Load model and scaler
model = WeatherLSTM()
model.load_state_dict(
    torch.load("weather_model.pth", map_location=torch.device("cpu"))
)
model.eval()
scaler = joblib.load("scaler.gz")

# 3. Web UI
st.set_page_config(page_title="AI Weather Forecaster", layout="centered")
st.title("🌦️ AI Weather Prediction & Analysis")

st.markdown("### Enter Current Weather Conditions:")
col1, col2 = st.columns(2)
with col1:
  temp = st.slider("Temperature (°C)", -5.0, 45.0, 28.0)
  hum = st.slider("Humidity (%)", 10.0, 100.0, 75.0)
with col2:
  pres = st.slider("Pressure (hPa)", 950.0, 1050.0, 1008.0)
  wind = st.slider("Wind Speed (km/h)", 0.0, 60.0, 15.0)

# Forecast Multi-step Future Hours
hours_ahead = st.selectbox("Predict Future Ahead:", [6, 12, 24])

if st.button("Run Future Analysis", use_container_width=True):
  # Construct initial 24-step sequence
  initial_window = np.tile([temp, hum, pres, wind], (24, 1))
  current_seq = scaler.transform(initial_window)

  predictions = []
  curr_tensor = (
      torch.tensor(current_seq, dtype=torch.float32)
      .unsqueeze(0)
  )

  with torch.no_grad():
    for _ in range(hours_ahead):
      next_step_scaled = model(curr_tensor).numpy()  # shape (1, 4)
      predictions.append(next_step_scaled[0])

      # Roll window forward: drop oldest, append prediction
      new_window = np.vstack([curr_tensor.numpy()[0][1:], next_step_scaled])
      curr_tensor = (
          torch.tensor(new_window, dtype=torch.float32)
          .unsqueeze(0)
      )

  # Invert scale back to real units
  results = scaler.inverse_transform(predictions)
  pred_df = pd.DataFrame(
      results,
      columns=["Temperature", "Humidity", "Pressure", "Wind_Speed"],
      index=[f"+{i+1} hr" for i in range(hours_ahead)],
  )

  st.divider()
  st.subheader(f"📊 Forecast Analysis for Next {hours_ahead} Hours")

  # Metrics for next immediate hour
  c1, c2, c3, c4 = st.columns(4)
  c1.metric("Temp (+1h)", f"{results[0][0]:.1f} °C")
  c2.metric("Humidity (+1h)", f"{results[0][1]:.1f} %")
  c3.metric("Pressure (+1h)", f"{results[0][2]:.1f} hPa")
  c4.metric("Wind (+1h)", f"{results[0][3]:.1f} km/h")

  # Trend Charts
  st.line_chart(pred_df[["Temperature", "Humidity"]])
  st.line_chart(pred_df[["Pressure", "Wind_Speed"]])
  st.dataframe(pred_df)