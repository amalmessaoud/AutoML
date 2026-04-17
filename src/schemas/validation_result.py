# src/schemas/validation_result.py
from pydantic import BaseModel


class ValidationResult(BaseModel):
    passed: bool
    issues: list[str]  # hard failures — block execution, force re-plan
    warnings: list[str]  # soft flags — log but don't block
