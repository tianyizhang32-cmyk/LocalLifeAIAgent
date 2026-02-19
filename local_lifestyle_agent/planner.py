from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Optional

from .schemas import NormalizedIntent, ExecutableMCP
from .llm_client import LLMClient
from .infrastructure.validator import DataValidator, ValidationResult
from .infrastructure.logger import StructuredLogger
from .infrastructure.metrics import MetricsCollector
from .infrastructure.error_handler import ErrorHandler, ErrorResponse, ErrorCode

EXECUTABLE_MCP_SCHEMA = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "tool_calls": {
      "type": "array",
      "items": {
        "anyOf": [
          {
            "type": "object",
            "additionalProperties": False,
            "properties": {
              "tool": {"type": "string", "const": "google_places_textsearch"},
              "args": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                  "query": {"type": "string"}
                },
                "required": ["query"]
              }
            },
            "required": ["tool", "args"]
          },
          {
            "type": "object",
            "additionalProperties": False,
            "properties": {
              "tool": {"type": "string", "const": "google_places_details"},
              "args": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                  "place_id": {"type": "string"}
                },
                "required": ["place_id"]
              }
            },
            "required": ["tool", "args"]
          }
        ]
      }
    },
    "selection_policy": {"type": "object", "additionalProperties": True},
    "notes": {"type": ["string", "null"]}
  },
  "required": ["tool_calls", "selection_policy", "notes"]
}

class Planner:
    """
    Planner:
      - normalize: user prompt -> NormalizedIntent
      - plan: intent + runtime context -> ExecutableMCP (tool plan)
    
    Integrated features:
      - Data validation (DataValidator)
      - Logging (StructuredLogger)
      - Metrics collection (MetricsCollector)
      - Error handling (ErrorHandler)
    
    Validates: Requirements 1.6, 6.1, 6.2, 6.3
    """
    def __init__(
        self,
        llm: LLMClient,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None,
        error_handler: Optional[ErrorHandler] = None
    ):
        """Initialize Planner
        
        Args:
            llm: LLM client
            logger: Logger instance (optional)
            metrics: Metrics collector (optional)
            error_handler: Error handler (optional)
        """
        self.llm = llm
        self.logger = logger
        self.metrics = metrics
        self.error_handler = error_handler or ErrorHandler()
        self.validator = DataValidator()

    def normalize(self, user_prompt: str) -> NormalizedIntent | ErrorResponse:
        """Normalize user prompt to NormalizedIntent
        
        Integrated features:
        - Input validation and sanitization
        - Logging
        - Metrics collection
        - Error handling
        
        Args:
            user_prompt: User input prompt
        
        Returns:
            NormalizedIntent or ErrorResponse
        
        Validates: Requirements 1.6, 6.1, 6.2, 6.3, 6.7, 6.8
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # 设置请求 ID
        if self.logger:
            self.logger.set_request_id(request_id)
            self.logger.info("Starting intent normalization", user_prompt_length=len(user_prompt))
        
        try:
            # 1. Input validation and sanitization
            # Validate input length
            length_validation = self.validator.validate_input_length(user_prompt)
            if not length_validation.valid:
                if self.logger:
                    self.logger.warning(
                        "Input validation failed",
                        errors=length_validation.errors
                    )
                return ErrorResponse(
                    error_code=ErrorCode.INVALID_INPUT,
                    error_message="User prompt is too long",
                    details={"errors": length_validation.errors},
                    request_id=request_id
                )
            
            # Detect malicious content
            malicious_check = self.validator.detect_malicious_content(user_prompt)
            if not malicious_check.valid:
                if self.logger:
                    self.logger.warning(
                        "Malicious content detected",
                        errors=malicious_check.errors
                    )
                return ErrorResponse(
                    error_code=ErrorCode.INVALID_INPUT,
                    error_message="User prompt contains malicious content",
                    details={"errors": malicious_check.errors},
                    request_id=request_id
                )
            
            # Sanitize user input
            sanitized_prompt = self.validator.sanitize_user_input(user_prompt)
            
            # 2. Call LLM for normalization
            NORMALIZED_INTENT_SCHEMA = {
              "type": "object",
              "additionalProperties": False,
              "properties": {
                "activity_type": {"type": "string"},
                "city": {"type": "string"},
                "time_window": {
                  "type": "object",
                  "additionalProperties": False,
                  "properties": {
                    "day": {"type": "string"},
                    "start_local": {"type": "string"},
                    "end_local": {"type": "string"}
                  },
                  "required": ["day", "start_local", "end_local"]
                },
                "origin_latlng": {"type": ["string", "null"]},
                "max_travel_minutes": {"type": "integer", "minimum": 5, "maximum": 120},
                "party_size": {"type": "integer", "minimum": 1, "maximum": 12},
                "budget_level": {"type": "string", "enum": ["low", "medium", "high"]},
                "preferences": {"type": "object", "additionalProperties": True},
                "hard_constraints": {"type": "object", "additionalProperties": True},
                "output_requirements": {"type": "object", "additionalProperties": True}
              },
              "required": [
                "activity_type",
                "city",
                "time_window",
                "origin_latlng",
                "max_travel_minutes",
                "party_size",
                "budget_level",
                "preferences",
                "hard_constraints",
                "output_requirements"
              ]
            }
            schema = NORMALIZED_INTENT_SCHEMA
            system = (
                "Normalize a local-lifestyle request into NormalizedIntent JSON.\n"
                "- Extract the activity_type from user query (e.g., 'afternoon_tea', 'brunch', 'dinner', 'coffee').\n"
                "- Do NOT invent specific venues.\n"
                "- Provide explicit defaults: time_window, max_travel_minutes, budget_level.\n"
                "- If origin lat/lng is not provided, leave origin_latlng null.\n"
            )
            user = f"User prompt:\n{sanitized_prompt}\n\nReturn ONLY NormalizedIntent JSON."
            
            if self.logger:
                self.logger.debug("Calling LLM for intent normalization")
            
            out = self.llm.json_schema(
                system=system,
                user=user,
                schema=schema,
                schema_name="NormalizedIntent"
            )
            
            # 3. Validate LLM output
            validation_result = self.validator.validate_normalized_intent(out)
            
            if not validation_result.valid:
                if self.logger:
                    self.logger.error(
                        "Intent validation failed",
                        errors=validation_result.errors
                    )
                if self.metrics:
                    self.metrics.record_error("VALIDATION_ERROR")
                
                return ErrorResponse(
                    error_code=ErrorCode.VALIDATION_ERROR,
                    error_message="Failed to normalize user intent",
                    details={"errors": validation_result.errors},
                    request_id=request_id
                )
            
            # 4. Create NormalizedIntent object
            intent = NormalizedIntent.model_validate(out)
            
            # 5. Log success
            duration = time.time() - start_time
            if self.logger:
                self.logger.info(
                    "Intent normalization completed",
                    duration_ms=round(duration * 1000, 2),
                    city=intent.city,
                    party_size=intent.party_size
                )
            if self.metrics:
                self.metrics.request_duration_seconds.observe(duration)
            
            return intent
            
        except Exception as error:
            # Error handling
            duration = time.time() - start_time
            
            if self.logger:
                self.logger.log_error(
                    error,
                    context={
                        "operation": "normalize",
                        "duration_ms": round(duration * 1000, 2)
                    }
                )
            
            if self.metrics:
                self.metrics.record_error(type(error).__name__)
            
            # Return structured error response
            error_response = self.error_handler.handle_api_error(
                error,
                context={"operation": "normalize", "api": "openai"},
                request_id=request_id
            )
            
            return error_response

    def plan(self, intent: NormalizedIntent, runtime_context: Dict[str, Any]) -> ExecutableMCP | ErrorResponse:
        """Generate executable tool call plan
        
        Integrated features:
        - Input validation
        - Logging
        - Metrics collection
        - Error handling
        
        Args:
            intent: Normalized user intent
            runtime_context: Runtime context
        
        Returns:
            ExecutableMCP or ErrorResponse
        
        Validates: Requirements 1.6, 6.4
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # 设置请求 ID
        if self.logger:
            self.logger.set_request_id(request_id)
            self.logger.info(
                "Starting plan generation",
                city=intent.city,
                max_tool_calls=runtime_context.get("max_tool_calls", "unknown")
            )
        
        try:
            # 1. Call LLM to generate plan
            schema = EXECUTABLE_MCP_SCHEMA
            system = (
                "You are a domain planner for a Yelp-like local recommendation orchestrator.\n"
                "Output ExecutableMCP JSON with tool_calls.\n"
                "Constraints:\n"
                "- Keep tool_calls <= runtime_context.max_tool_calls\n"
                "- Prefer Google Places Text Search + optional Details\n"
                "- Avoid venue_ids in runtime_context.rejected_options\n"
            )
            payload = {"intent": intent.model_dump(), "runtime_context": runtime_context}
            user = json.dumps(payload, ensure_ascii=False)
            
            if self.logger:
                self.logger.debug("Calling LLM for plan generation")
            
            out = self.llm.json_schema(
                system=system,
                user=user,
                schema=schema,
                schema_name="ExecutableMCP"
            )
            
            # 2. Validate LLM output
            validation_result = self.validator.validate_executable_mcp(out)
            
            if not validation_result.valid:
                if self.logger:
                    self.logger.error(
                        "Plan validation failed",
                        errors=validation_result.errors
                    )
                if self.metrics:
                    self.metrics.record_error("VALIDATION_ERROR")
                
                return ErrorResponse(
                    error_code=ErrorCode.VALIDATION_ERROR,
                    error_message="Failed to generate execution plan",
                    details={"errors": validation_result.errors},
                    request_id=request_id
                )
            
            # 3. Create ExecutableMCP object
            executable = ExecutableMCP.model_validate(out)
            
            # 4. Log success
            duration = time.time() - start_time
            if self.logger:
                self.logger.info(
                    "Plan generation completed",
                    duration_ms=round(duration * 1000, 2),
                    tool_calls_count=len(executable.tool_calls)
                )
            if self.metrics:
                self.metrics.request_duration_seconds.observe(duration)
            
            return executable
            
        except Exception as error:
            # Error handling
            duration = time.time() - start_time
            
            if self.logger:
                self.logger.log_error(
                    error,
                    context={
                        "operation": "plan",
                        "duration_ms": round(duration * 1000, 2)
                    }
                )
            
            if self.metrics:
                self.metrics.record_error(type(error).__name__)
            
            # Return structured error response
            error_response = self.error_handler.handle_api_error(
                error,
                context={"operation": "plan", "api": "openai"},
                request_id=request_id
            )
            
            return error_response
