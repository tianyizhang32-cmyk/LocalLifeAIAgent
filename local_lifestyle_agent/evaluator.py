from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional, Tuple

from .schemas import CandidateVenue, EvaluationReport, NormalizedIntent
from .infrastructure.validator import DataValidator
from .infrastructure.logger import StructuredLogger
from .infrastructure.metrics import MetricsCollector
from .infrastructure.error_handler import ErrorHandler, ErrorResponse, ErrorCode


class Evaluator:
    """
    Deterministic evaluator: hard constraints + scoring.
    
    Integrated features:
      - Data validation (DataValidator)
      - Logging (StructuredLogger)
      - Metrics collection (MetricsCollector)
      - Error handling (ErrorHandler)
    
    Validates: Requirements 6.6
    """
    def __init__(
        self,
        min_rating: float = 4.0,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None,
        error_handler: Optional[ErrorHandler] = None
    ):
        """Initialize Evaluator
        
        Args:
            min_rating: Minimum rating requirement (default 4.0)
            logger: Logger instance (optional)
            metrics: Metrics collector (optional)
            error_handler: Error handler (optional)
        """
        self.min_rating = min_rating
        self.logger = logger
        self.metrics = metrics
        self.error_handler = error_handler or ErrorHandler()
        self.validator = DataValidator()

    def evaluate(
        self,
        intent: NormalizedIntent,
        candidates: List[CandidateVenue],
        rejected_ids: List[str]
    ) -> Tuple[EvaluationReport, List[Tuple[CandidateVenue, Dict[str, float]]]] | ErrorResponse:
        """Evaluate and rank candidate venues
        
        Integrated features:
        - Candidate venue data validation
        - Logging
        - Metrics collection
        - Error handling
        
        Args:
            intent: Normalized user intent
            candidates: List of candidate venues
            rejected_ids: List of rejected venue IDs
        
        Returns:
            (EvaluationReport, ranked candidate list) or ErrorResponse
        
        Validates: Requirements 6.6
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Set request ID
            if self.logger:
                self.logger.set_request_id(request_id)
                self.logger.info(
                    "Starting candidate evaluation",
                    candidates_count=len(candidates) if candidates else 0,
                    rejected_count=len(rejected_ids) if rejected_ids else 0,
                    min_rating=self.min_rating
                )
            # 1. Validate candidate venue data
            invalid_candidates = []
            for idx, candidate in enumerate(candidates):
                validation_result = self.validator.validate_candidate_venue(
                    candidate.model_dump()
                )
                if not validation_result.valid:
                    invalid_candidates.append({
                        "index": idx,
                        "venue_id": candidate.venue_id,
                        "errors": validation_result.errors
                    })
            
            if invalid_candidates:
                if self.logger:
                    self.logger.warning(
                        "Some candidates have validation errors",
                        invalid_count=len(invalid_candidates),
                        invalid_candidates=invalid_candidates
                    )
                # Continue processing valid candidates, but log warning
            
            # 2. Evaluate and rank candidate venues
            ranked: List[Tuple[CandidateVenue, Dict[str, float]]] = []

            for c in candidates:
                # Skip rejected venues
                if c.venue_id in rejected_ids:
                    if self.logger:
                        self.logger.debug(
                            "Skipping rejected venue",
                            venue_id=c.venue_id,
                            name=c.name
                        )
                    continue
                
                # Skip low-rated venues
                if c.rating is not None and float(c.rating) < self.min_rating:
                    if self.logger:
                        self.logger.debug(
                            "Skipping low-rated venue",
                            venue_id=c.venue_id,
                            name=c.name,
                            rating=c.rating
                        )
                    continue

                # Calculate scores
                rating = float(c.rating or 0.0)
                reviews = float(c.user_ratings_total or 0.0)
                price = c.price_level if c.price_level is not None else 2

                score_rating = max(min((rating - 4.0) / 1.0, 1.0), 0.0)
                score_popularity = min(reviews / 1200.0, 1.0)
                score_price = 1.0 - min(abs(price - 2) / 2.0, 1.0)

                pref_bonus = 0.0
                if intent.preferences.get("quiet"):
                    if "lodging" in (c.category or "") or "tea" in (c.category or ""):
                        pref_bonus += 0.15

                total = 0.45 * score_rating + 0.30 * score_popularity + 0.15 * score_price + pref_bonus
                
                ranked.append((c, {
                    "total": total,
                    "rating": score_rating,
                    "popularity": score_popularity,
                    "price_fit": score_price,
                    "pref_bonus": pref_bonus,
                }))
                
                if self.logger:
                    self.logger.debug(
                        "Scored venue",
                        venue_id=c.venue_id,
                        name=c.name,
                        total_score=round(total, 3),
                        rating_score=round(score_rating, 3),
                        popularity_score=round(score_popularity, 3),
                        price_score=round(score_price, 3),
                        pref_bonus=round(pref_bonus, 3)
                    )

            # 3. Sort
            ranked.sort(key=lambda x: x[1]["total"], reverse=True)

            # 4. Generate evaluation report
            if not ranked:
                if self.logger:
                    self.logger.warning(
                        "No candidates passed evaluation",
                        original_count=len(candidates),
                        rejected_count=len(rejected_ids)
                    )
                
                rep = EvaluationReport(
                    ok=False,
                    hard_violations=["no_candidates_pass_hard_constraints"],
                    replan_suggestions=["broaden_queries", "expand_radius_bias", "relax_min_rating"],
                )
                
                duration = time.time() - start_time
                if self.logger:
                    self.logger.info(
                        "Evaluation completed with no results",
                        duration_ms=round(duration * 1000, 2)
                    )
                if self.metrics:
                    self.metrics.request_duration_seconds.observe(duration)
                
                return rep, []

            breakdown = {cv.venue_id: comp for cv, comp in ranked[:20]}
            rep = EvaluationReport(ok=True, score_breakdown=breakdown)
            
            # 5. Log success
            duration = time.time() - start_time
            if self.logger:
                self.logger.info(
                    "Evaluation completed successfully",
                    duration_ms=round(duration * 1000, 2),
                    ranked_count=len(ranked),
                    top_score=round(ranked[0][1]["total"], 3) if ranked else 0,
                    top_venue=ranked[0][0].name if ranked else None
                )
            if self.metrics:
                self.metrics.request_duration_seconds.observe(duration)
            
            return rep, ranked
            
        except Exception as error:
            # Error handling
            duration = time.time() - start_time
            
            if self.logger:
                self.logger.log_error(
                    error,
                    context={
                        "operation": "evaluate",
                        "duration_ms": round(duration * 1000, 2),
                        "candidates_count": len(candidates) if candidates else 0
                    }
                )
            
            if self.metrics:
                self.metrics.record_error(type(error).__name__)
            
            # Return structured error response
            error_response = self.error_handler.handle_api_error(
                error,
                context={"operation": "evaluate"},
                request_id=request_id
            )
            
            return error_response
