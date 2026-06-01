"""Model pipeline implementations.

Every pipeline in here implements the same contract (see
:mod:`src.models.base`) so it can be evaluated and logged with the
shared scaffolding in :mod:`src.eval` and :mod:`src.mlflow_utils`.
"""

from src.models.base import (
    PipelineResult,
    predict_per_asset_type,
    run_pipeline,
)

__all__ = ["PipelineResult", "predict_per_asset_type", "run_pipeline"]
