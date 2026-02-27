import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("data/bank.csv")
df = df.dropna()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

os.makedirs("data/processed", exist_ok=True)

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)