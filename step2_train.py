import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Load the CSV
df = pd.read_csv("weather.csv")
features = ["Temperature", "Humidity", "Pressure", "Wind_Speed"]
data = df[features].values

# 2. Normalize values between 0 and 1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 3. Create sequence windows (Past 24 hours -> Next 1 hour)
X, y = [], []
for i in range(len(scaled_data) - 24):
  X.append(scaled_data[i : i + 24])
  y.append(scaled_data[i + 24])

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)


# 4. Define the LSTM Brain
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


model = WeatherLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
criterion = nn.MSELoss()

# 5. Train
print("Training the neural network...")
for epoch in range(15):
  for batch_x, batch_y in loader:
    optimizer.zero_grad()
    loss = criterion(model(batch_x), batch_y)
    loss.backward()
    optimizer.step()
  print(f"Epoch {epoch+1}/15 completed.")

# 6. Save the .pth and scaler
torch.save(model.state_dict(), "weather_model.pth")
joblib.dump(scaler, "scaler.gz")
print(
    "SUCCESS: Saved 'weather_model.pth' and 'scaler.gz' ready for your app!"
)