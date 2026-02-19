from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Optional

from openai import OpenAI

from local_lifestyle_agent.infrastructure.cache import Cache
from local_lifestyle_agent.infrastructure.config import Config
from local_lifestyle_agent.infrastructure.error_handler import ErrorHandler
from local_lifestyle_agent.infrastructure.logger import StructuredLogger
from local_lifestyle_agent.infrastructure.metrics import MetricsCollector


def _enforce_no_additional_properties(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    OpenAI Structured Outputs requires that object schemas explicitly set:
      "additionalProperties": false
    Recursively applies this to nested objects and arrays.
    """
    if not isinstance(schema, dict):
        return schema

    # If schema is a union/anyOf/oneOf/allOf, recurse into branches
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema and isinstance(schema[key], list):
            schema[key] = [_enforce_no_additional_properties(s) for s in schema[key]]

    t = schema.get("type")

    if t == "object":
        # Must explicitly specify and set to false
        schema["additionalProperties"] = False

        # Recurse into properties
        props = schema.get("properties", {})
        if isinstance(props, dict):
            for k, v in props.items():
                props[k] = _enforce_no_additional_properties(v)
            schema["properties"] = props

        # Recurse into patternProperties / definitions if present
        for k in ("$defs", "definitions"):
            if k in schema and isinstance(schema[k], dict):
                schema[k] = {dk: _enforce_no_additional_properties(dv) for dk, dv in schema[k].items()}

    elif t == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            schema["items"] = _enforce_no_additional_properties(items)

    return schema


class LLMClient:
    """Enhanced LLM Client with retry, caching, logging, and metrics.
    
    Integrates infrastructure components for production-ready API calls:
    - ErrorHandler: Retry logic with exponential backoff
    - Cache: Response caching to reduce API calls
    - Logger: Structured logging for all operations
    - Metrics: Performance and usage metrics collection
    
    Validates: Requirements 5.1, 5.2, 5.3, 5.4
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        config: Optional[Config] = None,
        cache: Optional[Cache] = None,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None,
        error_handler: Optional[ErrorHandler] = None
    ):
        """Initialize enhanced LLM client.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4o-mini, supports structured outputs)
            config: Configuration object (optional, uses defaults if not provided)
            cache: Cache instance (optional, creates new if not provided)
            logger: Logger instance (optional, creates new if not provided)
            metrics: Metrics collector (optional, creates new if not provided)
            error_handler: Error handler (optional, creates new if not provided)
        """
        self.api_key = api_key
        self.model = model
        
        # Initialize infrastructure components
        # Config must be initialized first as other components may depend on it
        self.config = config or Config()
        
        # Use provided components or create new ones with config values
        if cache is not None:
            self.cache = cache
        else:
            self.cache = Cache(
                max_size=self.config.cache_max_size,
                ttl=self.config.cache_ttl
            )
        
        if logger is not None:
            self.logger = logger
        else:
            self.logger = StructuredLogger("llm_client", log_level=self.config.log_level)
        
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = MetricsCollector()
        
        if error_handler is not None:
            self.error_handler = error_handler
        else:
            self.error_handler = ErrorHandler(
                max_retries=self.config.max_retries,
                base_delay=self.config.retry_base_delay,
                max_delay=self.config.retry_max_delay,
                exponential_base=self.config.retry_exponential_base
            )
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            timeout=self.config.openai_timeout
        )
        
        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        
        self.logger.info("LLM Client initialized", model=model)

    def json_schema(
        self,
        *,
        system: str,
        user: str,
        schema: Dict[str, Any],
        schema_name: str = "out",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Call OpenAI API with JSON schema response format.
        
        Implements caching, retry logic, and comprehensive logging/metrics.
        
        Args:
            system: System prompt
            user: User prompt
            schema: JSON schema for response validation
            schema_name: Schema name for OpenAI API
            strict: Whether to enforce strict schema validation
            
        Returns:
            Parsed JSON response matching the schema
            
        Raises:
            Exception: If all retry attempts fail
            
        Validates: Requirements 5.1, 5.2, 5.3, 5.4
        """
        # 1. Generate cache key
        cache_key = self._generate_cache_key(system, user, schema, schema_name)
        
        # 2. Check cache
        if self.config.cache_enabled:
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.logger.info("Cache hit", cache_key=cache_key[:16])
                self.metrics.record_cache_hit()
                return cached
            else:
                self.metrics.record_cache_miss()
        
        # 3. Call API with retry
        result = self._call_with_retry(system, user, schema, schema_name, strict)
        
        # 4. Cache result
        if self.config.cache_enabled:
            self.cache.set(cache_key, result)
            self.metrics.update_cache_size(len(self.cache))
        
        return result
    
    def _call_with_retry(
        self,
        system: str,
        user: str,
        schema: Dict[str, Any],
        schema_name: str,
        strict: bool
    ) -> Dict[str, Any]:
        """Call API with retry logic and exponential backoff.
        
        Args:
            system: System prompt
            user: User prompt
            schema: JSON schema
            schema_name: Schema name
            strict: Strict validation flag
            
        Returns:
            Parsed JSON response
            
        Raises:
            Exception: If all retry attempts fail
            
        Validates: Requirements 5.1, 5.2
        """
        # Sanitize schema for Structured Outputs requirements
        schema = _enforce_no_additional_properties(schema)
        
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                
                # Make API call
                resp = self.client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": schema_name,
                            "schema": schema,
                            "strict": strict,
                        }
                    },
                )
                
                duration = time.time() - start_time
                
                # Log and record metrics
                self.logger.log_api_call("openai", "POST", duration, 200)
                self.metrics.record_api_call("openai", duration, 200)
                
                # Extract and parse response
                text = getattr(resp, "output_text", None)
                if not text:
                    raise ValueError("No output_text from model response")
                
                result = json.loads(text)
                
                # Track token usage if available
                if hasattr(resp, 'usage'):
                    usage = resp.usage
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(usage, 'completion_tokens', 0)
                    total_tokens = getattr(usage, 'total_tokens', 0)
                    
                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    self.total_tokens += total_tokens
                
                # Validate response structure
                self._validate_response(result, schema)
                
                return result
                
            except Exception as e:
                last_error = e
                duration = time.time() - start_time
                
                # Log error
                self.logger.log_error(e, {
                    "attempt": attempt,
                    "max_retries": self.config.max_retries,
                    "duration": duration
                })
                
                # Record error metrics
                self.metrics.record_api_call("openai", duration, 500)
                self.metrics.record_error(type(e).__name__)
                
                # Check if should retry
                if not self.error_handler.should_retry(e) or attempt == self.config.max_retries:
                    self.logger.error(
                        "API call failed after all retries",
                        attempt=attempt,
                        error=str(e)
                    )
                    raise
                
                # Calculate retry delay
                delay = self.error_handler.get_retry_delay(attempt)
                self.logger.warning(
                    f"Retrying after {delay}s",
                    attempt=attempt + 1,
                    max_retries=self.config.max_retries,
                    delay=delay
                )
                time.sleep(delay)
        
        # Should not reach here, but just in case
        raise last_error or Exception("Max retries exceeded")
    
    def _generate_cache_key(
        self,
        system: str,
        user: str,
        schema: Dict[str, Any],
        schema_name: str
    ) -> str:
        """Generate cache key from request parameters.
        
        Uses SHA256 hash of concatenated parameters for consistent caching.
        
        Args:
            system: System prompt
            user: User prompt
            schema: JSON schema
            schema_name: Schema name
            
        Returns:
            Cache key (SHA256 hex digest)
            
        Validates: Requirement 5.2
        """
        # Concatenate all parameters that affect the response
        content = "|".join([
            system,
            user,
            json.dumps(schema, sort_keys=True),
            schema_name,
            self.model
        ])
        
        # Generate SHA256 hash
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _validate_response(self, response: Dict[str, Any], schema: Dict[str, Any]):
        """Validate response structure against schema.
        
        Performs basic validation to ensure response matches expected structure.
        
        Args:
            response: API response to validate
            schema: Expected JSON schema
            
        Raises:
            ValueError: If response doesn't match schema
            
        Validates: Requirement 5.3
        """
        # Basic validation: check if response is a dict
        if not isinstance(response, dict):
            raise ValueError(f"Response must be a dict, got {type(response)}")
        
        # Check required properties if specified in schema
        if "required" in schema and isinstance(schema["required"], list):
            for required_field in schema["required"]:
                if required_field not in response:
                    raise ValueError(f"Missing required field: {required_field}")
        
        # Check properties match expected types (basic validation)
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                if prop_name in response:
                    expected_type = prop_schema.get("type")
                    actual_value = response[prop_name]
                    
                    # Basic type checking
                    if expected_type == "string" and not isinstance(actual_value, str):
                        raise ValueError(
                            f"Field '{prop_name}' should be string, got {type(actual_value)}"
                        )
                    elif expected_type == "integer" and not isinstance(actual_value, int):
                        raise ValueError(
                            f"Field '{prop_name}' should be integer, got {type(actual_value)}"
                        )
                    elif expected_type == "number" and not isinstance(actual_value, (int, float)):
                        raise ValueError(
                            f"Field '{prop_name}' should be number, got {type(actual_value)}"
                        )
                    elif expected_type == "boolean" and not isinstance(actual_value, bool):
                        raise ValueError(
                            f"Field '{prop_name}' should be boolean, got {type(actual_value)}"
                        )
                    elif expected_type == "array" and not isinstance(actual_value, list):
                        raise ValueError(
                            f"Field '{prop_name}' should be array, got {type(actual_value)}"
                        )
                    elif expected_type == "object" and not isinstance(actual_value, dict):
                        raise ValueError(
                            f"Field '{prop_name}' should be object, got {type(actual_value)}"
                        )
    
    def get_usage_stats(self) -> dict:
        """Get token usage statistics.
        
        Returns:
            Dict with token counts and estimated costs
        """
        # Pricing for gpt-4o-mini: $0.150/1M input, $0.600/1M output
        input_cost = (self.total_prompt_tokens / 1_000_000) * 0.150
        output_cost = (self.total_completion_tokens / 1_000_000) * 0.600
        total_cost = input_cost + output_cost
        
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(total_cost, 6)
        }
