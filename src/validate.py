import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

model = joblib.load("model.pkl")
test = pd.read_csv("data/processed/test.csv")

X = test.iloc[:, :-1]
y = test.iloc[:, -1]

pred = model.predict(X)

acc = accuracy_score(y, pred)

# save metrics
with open("metrics.json", "w") as f:
    json.dump({"accuracy": float(acc)}, f)

# save confusion matrix
cm = confusion_matrix(y, pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")