from src.evaluation.comparison import ModelComparator
from tests.unit.test_comparison_extended import MockModel

c = ModelComparator()
m1 = MockModel()
m1.training_metrics = {"rmse": 1.0, "mae": 0.5}
m1.validation_metrics = {"rmse": 1.2, "mae": 0.6}
print("is_fitted:", m1.is_fitted)
art = m1.get_artifact()
print("artifact train metrics:", art.training_metrics)
print("artifact val metrics:", art.validation_metrics)
c.add_model(m1, "model1")
print("artifacts:", c.artifacts)
df = c.compare_metrics(["rmse"])
print("df:\n", df)
