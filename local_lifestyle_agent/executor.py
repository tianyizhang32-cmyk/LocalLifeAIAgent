from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from .schemas import CandidateVenue, ExecutableMCP, NormalizedIntent, ToolCall, ToolResult
from .adapters.google_places import GooglePlacesAdapter
from .infrastructure.validator import DataValidator
from .infrastructure.logger import StructuredLogger
from .infrastructure.metrics import MetricsCollector
from .infrastructure.error_handler import ErrorHandler, ErrorResponse, ErrorCode


def _latlng_from_geometry(obj: Dict[str, Any]) -> Optional[str]:
    try:
        loc = obj["geometry"]["location"]
        return f"{loc['lat']},{loc['lng']}"
    except Exception:
        return None


class Executor:
    """
    Executes Planner tool calls (MCP).
    
    Integrated features:
      - Data validation (DataValidator)
      - Logging (StructuredLogger)
      - Metrics collection (MetricsCollector)
      - Error handling (ErrorHandler)
      - Response data sanitization
    
    Validates: Requirements 6.4, 6.5
    """
    def __init__(
        self,
        places: GooglePlacesAdapter,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None,
        error_handler: Optional[ErrorHandler] = None
    ):
        """Initialize Executor
        
        Args:
            places: Google Places API adapter
            logger: Logger instance (optional)
            metrics: Metrics collector (optional)
            error_handler: Error handler (optional)
        """
        self.places = places
        self.logger = logger
        self.metrics = metrics
        self.error_handler = error_handler or ErrorHandler()
        self.validator = DataValidator()
        self.api_call_count = 0  # Track API calls for cost calculation

    def execute(self, executable: ExecutableMCP, intent: NormalizedIntent) -> Dict[str, Any] | ErrorResponse:
        """Execute tool call plan
        
        Integrated features:
        - Parameter validation
        - Response data sanitization
        - Logging
        - Metrics collection
        - Error handling
        
        Args:
            executable: Executable tool call plan
            intent: Normalized user intent
        
        Returns:
            Dictionary containing tool_results and candidates, or ErrorResponse
        
        Validates: Requirements 6.4, 6.5
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # 设置请求 ID
        if self.logger:
            self.logger.set_request_id(request_id)
            self.logger.info(
                "Starting tool execution",
                tool_calls_count=len(executable.tool_calls)
            )
        
        try:
            # 1. Validate ExecutableMCP parameters
            validation_result = self.validator.validate_executable_mcp(
                executable.model_dump()
            )
            
            if not validation_result.valid:
                if self.logger:
                    self.logger.warning(
                        "ExecutableMCP validation failed",
                        errors=validation_result.errors
                    )
                return ErrorResponse(
                    error_code=ErrorCode.VALIDATION_ERROR,
                    error_message="Invalid tool call parameters",
                    details={"errors": validation_result.errors},
                    request_id=request_id
                )
            
            # 2. Execute tool calls
            tool_results: List[ToolResult] = []
            candidates: Dict[str, CandidateVenue] = {}

            for call in executable.tool_calls:
                if self.logger:
                    self.logger.debug(
                        "Executing tool call",
                        tool=call.tool,
                        args=call.args
                    )
                
                if call.tool == "google_places_textsearch":
                    self.api_call_count += 1  # Track API call
                    tr = self._do_textsearch(call, intent)
                    tool_results.append(tr)
                    
                    if tr.ok:
                        # Sanitize response data
                        cleaned_data = self._clean_response_data(tr.data)
                        
                        for item in cleaned_data.get("results", []):
                            place_id = item.get("place_id")
                            if not place_id:
                                continue
                            
                            venue = CandidateVenue(
                                venue_id=place_id,
                                place_id=place_id,
                                name=item.get("name", "Unknown"),
                                address=item.get("formatted_address", ""),
                                rating=item.get("rating"),
                                user_ratings_total=item.get("user_ratings_total"),
                                price_level=item.get("price_level"),
                                latlng=_latlng_from_geometry(item),
                                category=(item.get("types") or ["unknown"])[0],
                            )
                            candidates[venue.venue_id] = venue

                elif call.tool == "google_places_details":
                    self.api_call_count += 1  # Track API call
                    tr = self._do_details(call)
                    tool_results.append(tr)
                    
                    if tr.ok:
                        # Sanitize response data
                        cleaned_data = self._clean_response_data(tr.data)
                        res = cleaned_data.get("result", {})
                        pid = res.get("place_id")
                        
                        if pid and pid in candidates:
                            c = candidates[pid]
                            c.price_level = res.get("price_level", c.price_level)
                            c.rating = res.get("rating", c.rating)
                            c.user_ratings_total = res.get("user_ratings_total", c.user_ratings_total)
                            c.address = res.get("formatted_address", c.address)
                            c.latlng = _latlng_from_geometry(res) or c.latlng
                else:
                    tool_results.append(ToolResult(tool=call.tool, ok=False, error="unknown_tool"))
            
            # 3. Log success
            duration = time.time() - start_time
            if self.logger:
                self.logger.info(
                    "Tool execution completed",
                    duration_ms=round(duration * 1000, 2),
                    candidates_count=len(candidates),
                    successful_calls=sum(1 for tr in tool_results if tr.ok)
                )
            if self.metrics:
                self.metrics.request_duration_seconds.observe(duration)

            return {"tool_results": tool_results, "candidates": list(candidates.values())}
            
        except Exception as error:
            # Error handling
            duration = time.time() - start_time
            
            if self.logger:
                self.logger.log_error(
                    error,
                    context={
                        "operation": "execute",
                        "duration_ms": round(duration * 1000, 2)
                    }
                )
            
            if self.metrics:
                self.metrics.record_error(type(error).__name__)
            
            # Return structured error response
            error_response = self.error_handler.handle_api_error(
                error,
                context={"operation": "execute", "api": "google_places"},
                request_id=request_id
            )
            
            return error_response

    def _do_textsearch(self, call: ToolCall, intent: NormalizedIntent) -> ToolResult:
        """Execute text search tool call
        
        Args:
            call: Tool call
            intent: Normalized user intent
        
        Returns:
            ToolResult: Tool execution result
        """
        try:
            # Validate parameters
            query = call.args.get("query")
            if not query:
                if self.logger:
                    self.logger.warning("Text search missing query parameter")
                return ToolResult(tool=call.tool, ok=False, error="missing_query")

            origin = intent.origin_latlng
            radius_m = call.args.get("radius_m")
            if radius_m is None:
                radius_m = int(intent.max_travel_minutes * 800)  # heuristic

            if self.logger:
                self.logger.debug(
                    "Calling Google Places text search",
                    query=query,
                    origin=origin,
                    radius_m=radius_m
                )

            data = self.places.text_search(
                query=query,
                location_latlng=origin,
                radius_m=radius_m if origin else None,
                max_results=int(call.args.get("max_results", 10)),
            )
            
            if self.logger:
                self.logger.debug(
                    "Text search completed",
                    results_count=len(data.get("results", []))
                )
            
            return ToolResult(tool=call.tool, ok=True, data=data)
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Text search failed",
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
            return ToolResult(tool=call.tool, ok=False, error=f"{type(e).__name__}: {e}")

    def _do_details(self, call: ToolCall) -> ToolResult:
        """Execute place details tool call
        
        Args:
            call: Tool call
        
        Returns:
            ToolResult: Tool execution result
        """
        try:
            # Validate parameters
            place_id = call.args.get("place_id")
            if not place_id:
                if self.logger:
                    self.logger.warning("Place details missing place_id parameter")
                return ToolResult(tool=call.tool, ok=False, error="missing_place_id")
            
            if self.logger:
                self.logger.debug(
                    "Calling Google Places details",
                    place_id=place_id
                )
            
            data = self.places.details(place_id=place_id)
            
            if self.logger:
                self.logger.debug("Place details completed")
            
            return ToolResult(tool=call.tool, ok=True, data=data)
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Place details failed",
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
            return ToolResult(tool=call.tool, ok=False, error=f"{type(e).__name__}: {e}")
    
    def _clean_response_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize API response data, remove invalid fields
        
        Sanitization rules:
        - Remove None value fields
        - Remove empty string fields
        - Remove empty list/dict fields
        - Recursively sanitize nested structures
        
        This method is idempotent: clean(clean(data)) == clean(data)
        
        Args:
            data: Raw response data
        
        Returns:
            Sanitized data
        
        Validates: Requirements 6.5
        """
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                # Recursively sanitize nested structures
                cleaned_value = self._clean_response_data(value)
                
                # Skip invalid values
                if cleaned_value is None:
                    continue
                if isinstance(cleaned_value, str) and len(cleaned_value) == 0:
                    continue
                if isinstance(cleaned_value, (list, dict)) and len(cleaned_value) == 0:
                    continue
                
                cleaned[key] = cleaned_value
            
            return cleaned
        
        elif isinstance(data, list):
            # Sanitize each element in the list
            return [
                self._clean_response_data(item)
                for item in data
                if self._clean_response_data(item) is not None
            ]
        
        else:
            # Return basic types directly
            return data
