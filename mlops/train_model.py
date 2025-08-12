import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import mlflow

def train():
    df = pd.read_parquet("data/rides.parquet")
    X = df[['ride_id']]
    y = df['duration']

    model = LinearRegression()
    model.fit(X, y)

    # Start MLflow run
    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("feature_count", X.shape[1])
        mlflow.log_metric("coef", model.coef_[0])
        mlflow.log_metric("intercept", model.intercept_)

        # Save model artifact inside MLflow
        joblib.dump(model, "model.joblib")
        mlflow.log_artifact("model.joblib")
        from mlflow import MlflowClient

        # Inside your mlflow.start_run() block, after logging artifact:
        client = MlflowClient()
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model.joblib"

        # Create or get registered model
        model_name = "RideDurationModel"
        try:
            client.get_registered_model(model_name)
        except Exception:
            client.create_registered_model(model_name)

        # Create a new model version linked to this run
        client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=mlflow.active_run().info.run_id,
        )

        print(f"Registered model version under name '{model_name}'")


    print(f"Model coef: {model.coef_[0]:.4f}, intercept: {model.intercept_:.4f}")
    print("Model saved and logged with MLflow")

if __name__ == "__main__":
    train()
