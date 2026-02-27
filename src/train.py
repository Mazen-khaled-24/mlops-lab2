import pandas as pd
import joblib

train = pd.read_csv("data/processed/train.csv")

X = train.iloc[:, :-1]
y = train.iloc[:, -1]

