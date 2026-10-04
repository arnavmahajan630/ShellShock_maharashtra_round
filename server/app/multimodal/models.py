from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class MMSignals(BaseModel):
    response_ms: int = 0
    hedge_count: int = 0
    speech_rate_wps: Optional[float] = None
    pause_count: Optional[int] = None


class MMResponse(BaseModel):
    path: Optional[List[str]] = None          # flowchart level: node ids in visit order
    predicted_output: Optional[str] = None    # flowchart level
    explanation_text: Optional[str] = None    # typed one-liner or edited transcript
    transcript: Optional[str] = None          # voice level
    transcript_edited: bool = False


class MMAttemptRequest(BaseModel):
    learner_id: str = "pilot"
    problem_id: str
    modality: Literal["image", "voice", "flowchart_trace", "voice_trace"]
    response: MMResponse
    signals: MMSignals = Field(default_factory=MMSignals)


class Cue(BaseModel):
    misconception: str          # e.g. "M01", "M06", "M07"
    strength: float             # 0.0 - 1.0
    quote: str                  # verbatim substring from learner


class Claim(BaseModel):
    kind: Literal[
        "iteration_count", "final_value", "branch_taken",
        "eval_order", "terminates", "other"
    ]
    value: Optional[str] = None    # normalized, e.g. "4", "true"
    quote: str = ""


class JudgeEvidence(BaseModel):
    rubric: List[Dict[str, Any]] = Field(default_factory=list)
    cues: List[Cue] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)
    verdict: Literal["correct", "partial", "incorrect", "unscorable"] = "unscorable"
    confidence: float = 0.5
    raw_response: Optional[str] = None
