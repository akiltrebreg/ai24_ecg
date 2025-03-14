from pydantic import BaseModel


class ExperimentMetrics(BaseModel):
    accuracy: float
    f1: float


class ExperimentCurves(BaseModel):
    epochs: list
    training_scores: list
    validation_scores: list
