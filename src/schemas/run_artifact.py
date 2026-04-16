# src/schemas/run_artifact.py
from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.schemas.plan import AutoMLPlan


class AgentCall(BaseModel):
    agent_name: str
    input_summary: str
    output_summary: str
    duration_seconds: float
    timestamp: datetime


class IterationRecord(BaseModel):
    iteration_number: int
    plan: AutoMLPlan
    results: dict
    evaluation: str
    solved: bool
    agent_calls: list[AgentCall]


class RunArtifact(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    dataset_path: str
    problem: str
    schema_version: str = "1.0"
    llm_config: dict  # serialized via LLMConfig.to_loggable_dict() — no key value
    random_seed: int
    iterations: list[IterationRecord]
    solved: bool
    total_duration_seconds: float