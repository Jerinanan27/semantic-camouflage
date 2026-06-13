"""
Request and response shapes for the API.

These pydantic models do three jobs at once:
  1. Validate incoming JSON (reject malformed requests automatically).
  2. Document the API (FastAPI turns them into interactive docs at /docs).
  3. Serialize our internal objects (Prediction, StageTrace) into clean JSON.
"""

from pydantic import BaseModel, Field


# --- Requests --------------------------------------------------------------

class CodeRequest(BaseModel):
    code: str = Field(..., description="The function source code to analyze.")
    use_llm: bool = Field(
        False, description="Use FLAN-T5 for a stronger (slower) attack.")


# --- Building blocks -------------------------------------------------------

class PredictionOut(BaseModel):
    is_vulnerable: bool
    label: str
    confidence: float
    prob_vulnerable: float


class StageOut(BaseModel):
    name: str
    removed: list[str]
    kept: int


class Stage(BaseModel):
    """One condition in the pipeline: the code at that point + its verdict."""
    code: str
    prediction: PredictionOut


# --- Endpoint responses ----------------------------------------------------

class AttackOut(BaseModel):
    injected_code: str
    before: PredictionOut
    after: PredictionOut
    flipped: bool                     # did the verdict change?


class DefendOut(BaseModel):
    defended_code: str
    prediction: PredictionOut
    stages: list[StageOut]
    total_removed: int


class PipelineOut(BaseModel):
    """The full story in one response: clean -> injected -> defended."""
    clean: Stage
    injected: Stage
    defended: Stage
    stages: list[StageOut]            # per-stage defense trace
    attack_flipped: bool              # attack changed the clean verdict
    defense_restored: bool            # defense brought the verdict back


class HealthOut(BaseModel):
    status: str
    detector_model: str
    detector_loaded: bool
