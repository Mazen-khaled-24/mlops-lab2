import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("data/processed/train.csv")

X = train.iloc[:, :-1]
y = train.iloc[:, -1]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")