import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("data/processed/train.csv")

X = train.iloc[:, :-1]
y = train.iloc[:, -1]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, "model.pkl")