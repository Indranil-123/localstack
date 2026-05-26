import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tabular-regression")


np.random.seed(42)
n = 300
df = pd.DataFrame({
    "feature_1": np.random.randn(n),
    "feature_2": np.random.randn(n),
    "feature_3": np.random.randint(0, 5, n),
})


df["target"] = 3*df["feature_1"] - 2*df["feature_2"] + df["feature_3"] + np.random.randn(n)*0.5

X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="ridge-regression") as run:

    alpha = 1.0
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # Log params & metrics
    mlflow.log_param("model_type", "Ridge")
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_metric("rmse", rmse)

    # Log model to S3 artifact store
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="ridge-model",
        registered_model_name="RidgeRegressionModel"  # registers in Model Registry
    )

    print(f" Model registered as 'RidgeRegressionModel'")
    print(f"View at: http://localhost:5000")