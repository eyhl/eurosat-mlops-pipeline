import json
import pytest
from src import evaluate
import math
@pytest.mark.slow
def test_eval_smoke(trained_run_dir):
    run_dir = trained_run_dir

    evaluate.run(run_dir=run_dir)
    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path) as f:
        metrics = json.load(f)
    assert "test" in metrics
    loss = metrics["test"]["loss"]
    acc = metrics["test"]["accuracy"]
    assert isinstance(loss, (int, float))
    assert isinstance(acc, (int, float))
    assert math.isfinite(loss)  # not NaN
    assert math.isfinite(acc)  # not NaN
    assert 0.0 <= acc <= 1.0
