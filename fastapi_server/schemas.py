"""
Shared Pydantic schemas used across the API.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class DiffInput(BaseModel):
    repo: str = Field(..., example="acme/backend")
    pr_number: int = Field(..., example=42)
    author: str = Field(..., example="jane-dev")
    language: str = Field(..., example="python")
    diff: str = Field(..., description="Raw unified diff (changed lines only)")
    files: List[str] = Field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0


class CodeIssue(BaseModel):
    file: str
    line: int
    severity: Literal["low", "medium", "high", "critical"]
    type: Literal["bug", "security", "performance", "style", "maintainability"]
    description: str
    suggested_fix: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)


class ReviewOutput(BaseModel):
    issues: List[CodeIssue] = Field(default_factory=list)
    overall_score: float = Field(ge=0.0, le=10.0)
    summary: Optional[str] = None
    latency_ms: Optional[int] = None
    cached: bool = False
    model_version: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    redis_connected: bool