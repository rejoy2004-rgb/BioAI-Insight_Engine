"""Typed request/response schemas for the HTTP API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import config

_F = dict(ge=1, le=10, allow_inf_nan=False)


class PatientFeatures(BaseModel):
    """Nine ordinal cytology scores, each a whole number 1-10 (Wisconsin biopsy scheme)."""
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    Cl_thickness: float = Field(..., alias="Cl.thickness", **_F)
    Cell_size: float = Field(..., alias="Cell.size", **_F)
    Cell_shape: float = Field(..., alias="Cell.shape", **_F)
    Marg_adhesion: float = Field(..., alias="Marg.adhesion", **_F)
    Epith_c_size: float = Field(..., alias="Epith.c.size", **_F)
    Bare_nuclei: float = Field(..., alias="Bare.nuclei", **_F)
    Bl_cromatin: float = Field(..., alias="Bl.cromatin", **_F)
    Normal_nucleoli: float = Field(..., alias="Normal.nucleoli", **_F)
    Mitoses: float = Field(..., alias="Mitoses", **_F)

    @field_validator("*")
    @classmethod
    def _whole_number(cls, v: float) -> float:
        if v != int(v):
            raise ValueError("must be a whole number between 1 and 10")
        return v

    def as_row(self) -> dict[str, float]:
        return self.model_dump(by_alias=True)


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patients: list[PatientFeatures] = Field(..., min_length=1, max_length=config.MAX_BATCH_ROWS)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., max_length=config.MAX_QUERY_CHARS * 4)  # real limit enforced after normalisation
    k: int = Field(3, ge=1, le=10)


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(config.SEED, ge=1, le=100000)


class Prediction(BaseModel):
    prediction: Literal[0, 1]
    prediction_label: str
    malignancy_score: float
    threshold: float
    primary_model: str
    model_scores: dict[str, float]
    explanation: dict[str, Any] | None = None


class BatchPredictionResponse(BaseModel):
    total_records: int
    predictions: list[Prediction]
