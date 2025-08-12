from mlflow.tracking import MlflowClient

def transition_model_stage(model_name, version, stage):
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True  # archives other versions in that stage
    )
    print(f"Model version {version} transitioned to stage '{stage}'")

if __name__ == "__main__":
    model_name = "RideDurationModel"
    version = 1  # update if you have more versions
    new_stage = "Staging"  # options: None, Staging, Production, Archived

    transition_model_stage(model_name, version, new_stage)
