import mlflow
mlflow.set_tracking_uri("sqlite:////home/achazhoor/Documents/2026/Code/cifar_10/mlflow.db")

client = mlflow.tracking.MlflowClient()

# Find your best run automatically
runs = client.search_runs(
    experiment_ids=["1"],  # "1" = training_pipeline
    filter_string="status = 'FINISHED'",
    order_by=["metrics.val_accuracy DESC"],
    max_results=1,
)
best_run_id = runs[0].info.run_id
print(f"Best run: {best_run_id}, val_accuracy: {runs[0].data.metrics.get('val_accuracy')}")

# Register it
result = mlflow.register_model(f"runs:/{best_run_id}/model", "cifar10-resnet18")

# Tag as champion
client.set_registered_model_alias("cifar10-resnet18", "champion", result.version)
print("Done — model registered and tagged as champion")