import pickle
import time
import pandas as pd
from sklearn.metrics import accuracy_score
from day5.演習1.main import prepare_data

model_path = "day5/演習1/models/titanic_model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

_, X_test, _, y_test = prepare_data(test_size=0.2, random_state=42)

start = time.time()
preds = model.predict(X_test)
elapsed = time.time() - start

accuracy = accuracy_score(y_test, preds)

print(f"Inference time: {elapsed:.4f} sec")
print(f"Accuracy: {accuracy:.4f}")

assert elapsed < 1.0, "Inference is too slow"
assert accuracy > 0.85, "Accuracy is below expected threshold"
