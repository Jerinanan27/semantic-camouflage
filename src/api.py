"""
The web service.

Exposes the detector + attack + defense as HTTP endpoints:

  GET  /health    — is the service up? which model is configured?
  POST /predict   — classify code (clean)
  POST /attack    — camouflage code, show the verdict before vs after
  POST /defend    — filter injected code, show what each stage removed
  POST /pipeline  — the whole story (clean -> injected -> defended) in one call

Run locally:  uvicorn src.api:app --reload
Interactive docs:  http://127.0.0.1:8000/docs

The three heavy objects (detector, attacker, defense) are created once and
reused, via cached provider functions. Endpoints receive them through FastAPI's
`Depends`, which means tests can inject fakes with `app.dependency_overrides`.
"""

from functools import lru_cache
import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import config, schemas
from .attack import Attacker
from .defense import CommentDefensePipeline, DefenseResult, load_default_embedder
from .detector import Prediction, VulnerabilityDetector


# --- Singletons (created once, reused across requests) --------------------

@lru_cache
def get_detector() -> VulnerabilityDetector:
    return VulnerabilityDetector()


@lru_cache
def get_attacker() -> Attacker:
    return Attacker()


@lru_cache
def get_defense() -> CommentDefensePipeline:
    return CommentDefensePipeline(embedder=load_default_embedder())


# --- Small converters: internal objects -> API schemas --------------------

def _pred_out(p: Prediction) -> schemas.PredictionOut:
    return schemas.PredictionOut(
        is_vulnerable=p.is_vulnerable, label=p.label,
        confidence=round(p.confidence, 4),
        prob_vulnerable=round(p.prob_vulnerable, 4))


def _stages_out(result: DefenseResult) -> list[schemas.StageOut]:
    return [schemas.StageOut(name=s.name, removed=s.removed, kept=s.kept)
            for s in result.stages]


# --- The app ---------------------------------------------------------------

app = FastAPI(
    title="Semantic Camouflage",
    description="Attack and defend a code vulnerability detector.",
    version="0.1.0",
)

# Allow a browser frontend (served from a different origin) to call the API.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


_FRONTEND = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "frontend", "index.html")


@app.get("/")
def home():
    """Serve the demo web page (fixes the bare-root 'Not Found')."""
    return FileResponse(_FRONTEND)


@app.get("/health", response_model=schemas.HealthOut)
def health(detector: VulnerabilityDetector = Depends(get_detector)):
    return schemas.HealthOut(
        status="ok",
        detector_model=detector.model_name,
        detector_loaded=detector.is_loaded,
    )


@app.post("/predict", response_model=schemas.PredictionOut)
def predict(req: schemas.CodeRequest,
            detector: VulnerabilityDetector = Depends(get_detector)):
    return _pred_out(detector.predict(req.code))


@app.post("/attack", response_model=schemas.AttackOut)
def attack(req: schemas.CodeRequest,
           detector: VulnerabilityDetector = Depends(get_detector),
           attacker: Attacker = Depends(get_attacker)):
    before = detector.predict(req.code)
    injected = attacker.inject(req.code, use_llm=req.use_llm)
    after = detector.predict(injected)
    return schemas.AttackOut(
        injected_code=injected,
        before=_pred_out(before), after=_pred_out(after),
        flipped=(before.is_vulnerable != after.is_vulnerable),
    )


@app.post("/defend", response_model=schemas.DefendOut)
def defend(req: schemas.CodeRequest,
           detector: VulnerabilityDetector = Depends(get_detector),
           defense: CommentDefensePipeline = Depends(get_defense)):
    result = defense.defend(req.code)
    return schemas.DefendOut(
        defended_code=result.defended_code,
        prediction=_pred_out(detector.predict(result.defended_code)),
        stages=_stages_out(result),
        total_removed=result.total_removed,
    )


@app.post("/pipeline", response_model=schemas.PipelineOut)
def pipeline(req: schemas.CodeRequest,
             detector: VulnerabilityDetector = Depends(get_detector),
             attacker: Attacker = Depends(get_attacker),
             defense: CommentDefensePipeline = Depends(get_defense)):
    """Run the whole story in one call — this is what the demo UI hits."""
    clean_pred = detector.predict(req.code)

    injected_code = attacker.inject(req.code, use_llm=req.use_llm)
    injected_pred = detector.predict(injected_code)

    result = defense.defend(injected_code)
    defended_pred = detector.predict(result.defended_code)

    return schemas.PipelineOut(
        clean=schemas.Stage(code=req.code, prediction=_pred_out(clean_pred)),
        injected=schemas.Stage(code=injected_code, prediction=_pred_out(injected_pred)),
        defended=schemas.Stage(code=result.defended_code, prediction=_pred_out(defended_pred)),
        stages=_stages_out(result),
        attack_flipped=(clean_pred.is_vulnerable != injected_pred.is_vulnerable),
        defense_restored=(clean_pred.is_vulnerable == defended_pred.is_vulnerable),
    )
