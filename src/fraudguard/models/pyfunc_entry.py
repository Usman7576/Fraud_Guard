"""MLflow "models from code" entry point: MLflow runs this file instead of unpickling an object."""

from mlflow.models import set_model

from fraudguard.models.bundle import FraudGuardPyfunc

set_model(FraudGuardPyfunc())
