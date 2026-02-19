from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .schemas import FinalPlan, NormalizedIntent, PlanOption, ExecutableMCP
from .planner import Planner
from .executor import Executor
from .evaluator import Evaluator
from .infrastructure.logger import StructuredLogger
from .infrastructure.metrics import MetricsCollector
from .infrastructure.error_handler import ErrorHandler, ErrorResponse, ErrorCode


@dataclass
class RunContext:
    max_tool_calls: int = 6
    max_iterations: int = 3


class Orchestrator:
    """Main orchestration loop module
    
    Coordinates Planner, Executor, and Evaluator modules to implement the complete recommendation flow:
    normalize -> plan -> execute -> evaluate -> replan (bounded) -> assemble
    
    Integrated features:
      - Request ID generation and propagation
      - Global error handling
      - Logging (StructuredLogger)
      - Metrics collection (MetricsCollector)
      - Fallback strategies (when LLM is unavailable)
    
    Validates: Requirements 1.9
    """
    
    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        evaluator: Evaluator,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None,
        error_handler: Optional[ErrorHandler] = None
    ):
        """Initialize Orchestrator
        
        Args:
            planner: Planner module
            executor: Executor module
            evaluator: Evaluator module
            logger: Logger instance (optional)
            metrics: Metrics collector (optional)
            error_handler: Error handler (optional)
        """
        self.planner = planner
        self.executor = executor
        self.evaluator = evaluator
        self.logger = logger
        self.metrics = metrics
        self.error_handler = error_handler or ErrorHandler()

        self.preference_signals: Dict[str, Any] = {}
        self.rejected_options: List[str] = []

    def run(self, user_prompt: str, ctx: RunContext = RunContext()) -> Dict[str, Any]:
        """Run the complete recommendation flow
        
        Flow:
        1. Normalize user intent
        2. Generate execution plan
        3. Execute tool calls
        4. Evaluate candidate venues
        5. Replan if needed
        6. Assemble final recommendation
        
        Integrated features:
        - Request ID generation and propagation
        - Global error handling
        - Logging
        - Metrics collection
        - Fallback strategies
        
        Args:
            user_prompt: User input prompt
            ctx: Run context (max tool calls, max iterations)
        
        Returns:
            Dictionary containing intent, executable, candidates, eval_report, plan
            or dictionary containing error if an error occurs
        
        Validates: Requirements 1.9
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Set request ID
        if self.logger:
            self.logger.set_request_id(request_id)
            self.logger.info(
                "Starting orchestration",
                user_prompt_length=len(user_prompt),
                max_iterations=ctx.max_iterations,
                max_tool_calls=ctx.max_tool_calls
            )
        
        # Update active requests count
        if self.metrics:
            self.metrics.active_requests.inc()
        
        try:
            # 1. Normalize user intent
            if self.logger:
                self.logger.debug("Step 1: Normalizing user intent")
            
            intent = self.planner.normalize(user_prompt)
            
            # Check if error response returned
            if isinstance(intent, ErrorResponse):
                if self.logger:
                    self.logger.error(
                        "Intent normalization failed",
                        error_code=intent.error_code
                    )
                return self._handle_error_response(intent, request_id, start_time)
            
            if self.logger:
                self.logger.info(
                    "Intent normalized successfully",
                    city=intent.city,
                    party_size=intent.party_size,
                    budget_level=intent.budget_level
                )

            last_eval = None
            last_candidates = []
            ranked = []

            # 2-5. Iteration: plan -> execute -> evaluate -> replan
            for it in range(1, ctx.max_iterations + 1):
                if self.logger:
                    self.logger.info(
                        f"Starting iteration {it}/{ctx.max_iterations}",
                        iteration=it
                    )
                
                runtime_context = {
                    "iteration": it,
                    "max_tool_calls": ctx.max_tool_calls,
                    "rejected_options": self.rejected_options,
                    "preference_signals": self.preference_signals,
                    "last_eval": last_eval.model_dump() if last_eval else None,
                }

                # 2. Generate execution plan
                if self.logger:
                    self.logger.debug("Step 2: Generating execution plan")
                
                executable = self.planner.plan(intent, runtime_context)
                
                # Check if error response returned
                if isinstance(executable, ErrorResponse):
                    if self.logger:
                        self.logger.warning(
                            "Plan generation failed, attempting fallback",
                            error_code=executable.error_code,
                            iteration=it
                        )
                    
                    # Fallback strategy: use default query parameters
                    executable = self._fallback_plan(intent, runtime_context)
                    
                    if executable is None:
                        if self.logger:
                            self.logger.error("Fallback plan generation failed")
                        return self._handle_error_response(executable, request_id, start_time)
                
                # 3. Execute tool calls
                if self.logger:
                    self.logger.debug(
                        "Step 3: Executing tool calls",
                        tool_calls_count=len(executable.tool_calls)
                    )
                
                exec_out = self.executor.execute(executable, intent)
                
                # Check if error response returned
                if isinstance(exec_out, ErrorResponse):
                    if self.logger:
                        self.logger.warning(
                            "Tool execution failed",
                            error_code=exec_out.error_code,
                            iteration=it
                        )
                    
                    # Fallback strategy: return empty candidate list
                    exec_out = {"tool_results": [], "candidates": []}

                candidates = exec_out["candidates"]
                last_candidates = candidates
                
                if self.logger:
                    self.logger.info(
                        "Tool execution completed",
                        candidates_count=len(candidates)
                    )

                # 4. Evaluate candidate venues
                if self.logger:
                    self.logger.debug("Step 4: Evaluating candidates")
                
                eval_result = self.evaluator.evaluate(intent, candidates, self.rejected_options)
                
                # Check if error response returned
                if isinstance(eval_result, ErrorResponse):
                    if self.logger:
                        self.logger.warning(
                            "Evaluation failed",
                            error_code=eval_result.error_code,
                            iteration=it
                        )
                    
                    # Fallback strategy: return unsorted candidate list
                    eval_report = None
                    ranked = [(c, {"total": 0.0}) for c in candidates]
                else:
                    eval_report, ranked = eval_result
                    last_eval = eval_report
                
                if self.logger:
                    self.logger.info(
                        "Evaluation completed",
                        ranked_count=len(ranked),
                        evaluation_ok=eval_report.ok if eval_report else False
                    )

                # Check if automatic rating lowering retry is needed
                if eval_report and not eval_report.ok:
                    if "no_candidates_pass_hard_constraints" in eval_report.hard_violations:
                        # Check if rating can be lowered
                        current_rating = self.evaluator.min_rating
                        if current_rating > 2.0:
                            # Lower rating (0.5 each time, minimum 2.0)
                            new_rating = max(current_rating - 0.5, 2.0)
                            
                            if self.logger:
                                self.logger.info(
                                    "No candidates found, lowering min_rating",
                                    current_rating=current_rating,
                                    new_rating=new_rating,
                                    iteration=it
                                )
                            
                            # Update evaluator's min_rating
                            self.evaluator.min_rating = new_rating
                            
                            # Re-evaluate (does not count towards iteration limit)
                            if self.logger:
                                self.logger.debug("Re-evaluating with lower rating threshold")
                            
                            eval_result = self.evaluator.evaluate(intent, candidates, self.rejected_options)
                            
                            if isinstance(eval_result, ErrorResponse):
                                eval_report = None
                                ranked = [(c, {"total": 0.0}) for c in candidates]
                            else:
                                eval_report, ranked = eval_result
                                last_eval = eval_report
                            
                            if self.logger:
                                self.logger.info(
                                    "Re-evaluation completed",
                                    ranked_count=len(ranked),
                                    evaluation_ok=eval_report.ok if eval_report else False
                                )

                # Check if satisfactory results found
                if eval_report and eval_report.ok:
                    # 6. Assemble final recommendation
                    if self.logger:
                        self.logger.debug("Step 6: Assembling final plan")
                    
                    plan = self._assemble(
                        intent,
                        ranked,
                        num_backups=int(intent.output_requirements.get("num_backups", 3))
                    )
                    
                    # Log success
                    duration = time.time() - start_time
                    if self.logger:
                        self.logger.info(
                            "Orchestration completed successfully",
                            duration_ms=round(duration * 1000, 2),
                            iterations=it,
                            primary_venue=plan.primary.name if plan else None
                        )
                    if self.metrics:
                        self.metrics.record_request(duration, 200)
                    
                    # Calculate cost summary
                    cost_summary = self._calculate_cost_summary()
                    
                    return {
                        "intent": intent,
                        "executable": executable,
                        "candidates": candidates,
                        "eval_report": eval_report,
                        "plan": plan,
                        "request_id": request_id,
                        "cost_summary": cost_summary
                    }

                # 5. Replan
                if eval_report:
                    if self.logger:
                        self.logger.debug(
                            "Step 5: Applying replan suggestions",
                            suggestions=eval_report.replan_suggestions
                        )
                    self._apply_replan(intent, eval_report.replan_suggestions)

            # All iterations failed
            duration = time.time() - start_time
            if self.logger:
                self.logger.warning(
                    "Orchestration completed without satisfactory results",
                    duration_ms=round(duration * 1000, 2),
                    iterations=ctx.max_iterations,
                    last_candidates_count=len(last_candidates)
                )
            if self.metrics:
                self.metrics.record_request(duration, 200)  # Still a successful request, just no satisfactory results
            
            return {
                "intent": intent,
                "executable": None,
                "candidates": last_candidates,
                "eval_report": last_eval,
                "plan": None,
                "request_id": request_id
            }
        
        except Exception as error:
            # Global error handling
            duration = time.time() - start_time
            
            if self.logger:
                self.logger.log_error(
                    error,
                    context={
                        "operation": "orchestrate",
                        "duration_ms": round(duration * 1000, 2),
                        "user_prompt_length": len(user_prompt)
                    }
                )
            
            if self.metrics:
                self.metrics.record_error(type(error).__name__)
                self.metrics.record_request(duration, 500)
            
            # Return structured error response
            error_response = self.error_handler.handle_api_error(
                error,
                context={"operation": "orchestrate"},
                request_id=request_id
            )
            
            return {
                "error": error_response,
                "request_id": request_id
            }
        
        finally:
            # Update active requests count
            if self.metrics:
                self.metrics.active_requests.dec()
    
    def _handle_error_response(
        self,
        error_response: ErrorResponse,
        request_id: str,
        start_time: float
    ) -> Dict[str, Any]:
        """Handle error response
        
        Args:
            error_response: Error response object
            request_id: Request ID
            start_time: Request start time
        
        Returns:
            Dictionary containing error
        """
        duration = time.time() - start_time
        
        if self.metrics:
            self.metrics.record_error(error_response.error_code)
            self.metrics.record_request(duration, 500)
        
        return {
            "error": error_response,
            "request_id": request_id
        }
    
    def _fallback_plan(
        self,
        intent: NormalizedIntent,
        runtime_context: Dict[str, Any]
    ) -> Optional[ExecutableMCP]:
        """Fallback strategy: generate default execution plan
        
        When LLM is unavailable, use rule engine to generate default execution plan.
        
        Args:
            intent: Normalized user intent
            runtime_context: Runtime context
        
        Returns:
            ExecutableMCP or None if unable to generate
        
        Validates: Requirements 1.9
        """
        try:
            if self.logger:
                self.logger.info("Using fallback plan generation (rule-based)")
            
            # Build default query
            query = f"{intent.activity_type} in {intent.city}"
            
            # Build default tool calls
            from .schemas import ToolCall
            
            tool_calls = [
                ToolCall(
                    tool="google_places_textsearch",
                    args={
                        "query": query,
                        "max_results": 10
                    }
                )
            ]
            
            # Create ExecutableMCP
            executable = ExecutableMCP(
                tool_calls=tool_calls,
                selection_policy={"strategy": "fallback"},
                notes="Generated by fallback rule engine (LLM unavailable)"
            )
            
            if self.logger:
                self.logger.info(
                    "Fallback plan generated successfully",
                    query=query,
                    tool_calls_count=len(tool_calls)
                )
            
            return executable
        
        except Exception as error:
            if self.logger:
                self.logger.error(
                    "Fallback plan generation failed",
                    error_type=type(error).__name__,
                    error_message=str(error)
                )
            return None

    def _assemble(self, intent: NormalizedIntent, ranked, num_backups: int) -> FinalPlan:
        top = ranked[: max(num_backups + 1, 2)]
        primary_cv, primary_comp = top[0]
        backups = top[1: 1 + num_backups]

        def mk_option(cv, comp) -> PlanOption:
            rationale = []
            if comp.get("rating", 0) > 0.4:
                rationale.append("Strong ratings signal")
            if comp.get("popularity", 0) > 0.6:
                rationale.append("Popular spot with lots of reviews")
            if comp.get("price_fit", 0) > 0.6:
                rationale.append("Matches your budget preference")
            if comp.get("pref_bonus", 0) > 0:
                rationale.append("Likely matches your preference signals (e.g., quieter vibe)")
            return PlanOption(venue_id=cv.venue_id, name=cv.name, address=cv.address, rationale=rationale[:3])

        primary = mk_option(primary_cv, primary_comp)
        backup_opts = [mk_option(cv, comp) for cv, comp in backups]

        start = intent.time_window.get("start_local", "14:00")
        end = intent.time_window.get("end_local", "17:00")

        return FinalPlan(
            primary=primary,
            backups=backup_opts,
            schedule={"arrive_at": start, "leave_at": end},
            tips=[
                "Sunday afternoons can be busy—consider booking ahead if possible.",
                "If you prefer quiet, ask for a corner seat or off-peak slot.",
                "If the primary is full, use a backup with similar vibe/price."
            ],
            assumptions=[
                "Opening hours and reservation status can change; verify before going.",
                "This plan optimizes for your stated time window and preferences under a bounded search budget."
            ]
        )

    def _apply_replan(self, intent: NormalizedIntent, suggestions: List[str]) -> None:
        if "expand_radius_bias" in suggestions:
            intent.max_travel_minutes = min(intent.max_travel_minutes + 10, 60)

    def _calculate_cost_summary(self) -> dict:
        """Calculate total cost summary for LLM and API calls.
        
        Returns:
            Dict with cost breakdown
        """
        # Get LLM usage stats
        llm_stats = self.planner.llm.get_usage_stats()
        
        # Google Places API pricing: $0.017 per Text Search request
        # Estimate based on typical usage (assume executor tracks calls)
        places_calls = getattr(self.executor, 'api_call_count', 0)
        places_cost = places_calls * 0.017
        
        total_cost = llm_stats['estimated_cost_usd'] + places_cost
        
        return {
            "llm": {
                "prompt_tokens": llm_stats['prompt_tokens'],
                "completion_tokens": llm_stats['completion_tokens'],
                "total_tokens": llm_stats['total_tokens'],
                "cost_usd": llm_stats['estimated_cost_usd']
            },
            "google_places": {
                "api_calls": places_calls,
                "cost_usd": round(places_cost, 6)
            },
            "total_cost_usd": round(total_cost, 6)
        }
