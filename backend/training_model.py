import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("../data/shipment.csv")
df = df.dropna()

# Adjust columns as per dataset
X = df[['distance', 'carrier_rating', 'weather_score']]
y = df['delay_days']

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model trained")
