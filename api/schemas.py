from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    n_split: int = Field(ge=1, default=1)
    n_nearest: int = Field(ge=2, default=2)
    data: list[list]


class ModelInput(BaseModel):
    data: list[list]


class ProcessDetails(BaseModel):
    details: str
    in_progress: bool | None = None
    prod_model: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    error_time: str | None = None
    error: str | None = None


class ModelPrediction(BaseModel):
    model: str
    prediction: list[int | float]
