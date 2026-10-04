import importlib.util
from pathlib import Path

import numpy as np


def test_tolerances_do_not_hide_tracking_or_nonfinite_errors(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/validate_parity.py"
    spec = importlib.util.spec_from_file_location("validate_parity", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference_path, actual_path = tmp_path / "reference.npz", tmp_path / "actual.npz"
    np.savez(reference_path, depth=[1.0], track=[100.0], vis=[0.01], conf=[0.5])
    np.savez(actual_path, depth=[1.0001], track=[100.5], vis=[0.02], conf=[np.nan])
    with np.load(reference_path) as reference, np.load(actual_path) as actual:
        metrics = module.compare(
            reference, actual, nrmse_limit=1e-3, track_rmse_limit=1
        )
        assert metrics["depth"]["passed"]
        assert metrics["track"]["passed"]
        assert metrics["vis"]["passed"]
        assert not metrics["conf"]["passed"]
        strict = module.compare(
            reference,
            actual,
            nrmse_limit=1e-3,
            track_rmse_limit=0.05,
            strict_tracking=True,
        )
        assert not strict["track"]["passed"]
        assert not strict["vis"]["passed"]
