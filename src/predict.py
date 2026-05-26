import mlflow.sklearn
import numpy as np

# --- Load latest model from MLflow Registry ---
mlflow.set_tracking_uri("http://localhost:5000")

model_uri = "models:/RidgeRegressionModel/latest"
model = mlflow.sklearn.load_model(model_uri)
print("Model S3 artifact store)")


X_new = np.array([
    [1.5, -0.5, 2],
    [0.0,  1.0, 4],
])
predictions = model.predict(X_new)
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: predicted = {pred:.4f}")