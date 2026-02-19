from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class NormalizedIntent(BaseModel):
    """
    Canonical intent for a Yelp-like local lifestyle recommendation.
    NOTE: For Google Places, origin_latlng is ideal (e.g., "47.61,-122.33"). If absent, we will
    still run text search using query + city, but travel-time filtering will be approximate.
    """
    activity_type: str
    city: str
    time_window: Dict[str, str]  # {"day":"Sunday","start_local":"14:00","end_local":"17:00"}
    origin_latlng: Optional[str] = None  # "lat,lng"
    max_travel_minutes: int = 30
    party_size: int = 1
    budget_level: Literal["low", "medium", "high"] = "medium"
    preferences: Dict[str, Any] = Field(default_factory=dict)
    hard_constraints: Dict[str, Any] = Field(default_factory=lambda: {"must_be_open": True})
    output_requirements: Dict[str, Any] = Field(default_factory=lambda: {"num_backups": 3, "detail_level": "medium"})


class ToolCall(BaseModel):
    tool: Literal["google_places_textsearch", "google_places_details"]
    args: Dict[str, Any] = Field(default_factory=dict)


class ExecutableMCP(BaseModel):
    tool_calls: List[ToolCall] = Field(default_factory=list)
    selection_policy: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None


class CandidateVenue(BaseModel):
    venue_id: str
    name: str
    category: str = "unknown"
    address: str
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    price_level: Optional[int] = None  # 0-4 in Google Places (when present)
    place_id: Optional[str] = None
    latlng: Optional[str] = None  # "lat,lng"


class ToolResult(BaseModel):
    tool: str
    ok: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class PlanOption(BaseModel):
    venue_id: str
    name: str
    address: str
    rationale: List[str] = Field(default_factory=list)


class FinalPlan(BaseModel):
    primary: PlanOption
    backups: List[PlanOption] = Field(default_factory=list)
    schedule: Dict[str, str] = Field(default_factory=dict)
    tips: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    ok: bool
    hard_violations: List[str] = Field(default_factory=list)
    score_breakdown: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    replan_suggestions: List[str] = Field(default_factory=list)


class RenderedOutput(BaseModel):
    markdown: str
