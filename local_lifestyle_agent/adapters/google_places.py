from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from local_lifestyle_agent.infrastructure.cache import Cache
from local_lifestyle_agent.infrastructure.config import Config
from local_lifestyle_agent.infrastructure.error_handler import ErrorHandler
from local_lifestyle_agent.infrastructure.logger import StructuredLogger
from local_lifestyle_agent.infrastructure.metrics import MetricsCollector


class GooglePlacesAdapter:
    """Enhanced Google Places API adapter with retry, caching, logging, and metrics.
    
    Integrates infrastructure components for production-ready API calls:
    - ErrorHandler: Retry logic with exponential backoff
    - Cache: Response caching to reduce API calls
    - Logger: Structured logging for all operations
    - Metrics: Performance and usage metrics collection
    - Session: Connection pooling for better performance
    
    Validates: Requirements 5.5, 5.6, 5.7
    """
    
    BASE = "https://maps.googleapis.com/maps/api"

    def __init__(
        self,
        api_key: str,
        config: Optional[Config] = None,
        cache: Optional[Cache] = None,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None,
        error_handler: Optional[ErrorHandler] = None
    ):
        """Initialize enhanced Google Places adapter.
        
        Args:
            api_key: Google Places API key
            config: Configuration object (optional, uses defaults if not provided)
            cache: Cache instance (optional, creates new if not provided)
            logger: Logger instance (optional, creates new if not provided)
            metrics: Metrics collector (optional, creates new if not provided)
            error_handler: Error handler (optional, creates new if not provided)
        """
        self.api_key = api_key
        
        # Initialize infrastructure components
        self.config = config or Config()
        
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
            self.logger = StructuredLogger("google_places", log_level=self.config.log_level)
        
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
        
        # Initialize requests session with connection pooling
        self.session = requests.Session()
        
        # Configure connection pool
        adapter = HTTPAdapter(
            pool_connections=self.config.connection_pool_size,
            pool_maxsize=self.config.connection_pool_size,
            max_retries=0  # We handle retries manually
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        self.logger.info("Google Places Adapter initialized")

    def text_search(
        self,
        *,
        query: str,
        location_latlng: Optional[str] = None,  # "lat,lng"
        radius_m: Optional[int] = None,
        language: str = "en",
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """Text search with caching, retry logic, and comprehensive logging/metrics.
        
        Args:
            query: Search query
            location_latlng: Location in "lat,lng" format (optional)
            radius_m: Search radius in meters (optional)
            language: Language code (default: "en")
            max_results: Maximum number of results (default: 10)
            
        Returns:
            API response with search results
            
        Raises:
            Exception: If all retry attempts fail
            
        Validates: Requirements 5.5, 5.6, 5.7
        """
        # 1. Generate cache key
        cache_key = self._generate_cache_key(
            "text_search",
            query=query,
            location_latlng=location_latlng,
            radius_m=radius_m,
            language=language,
            max_results=max_results
        )
        
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
        url = f"{self.BASE}/place/textsearch/json"
        params: Dict[str, Any] = {
            "query": query,
            "key": self.api_key,
            "language": language,
        }
        if location_latlng and radius_m:
            params["location"] = location_latlng
            params["radius"] = radius_m
        
        result = self._call_with_retry(url, params)
        
        # 4. Validate and clean response
        self._validate_response(result, expected_key="results")
        result = self._clean_response_data(result)
        
        # 5. Limit results
        if "results" in result and isinstance(result["results"], list):
            result["results"] = result["results"][:max_results]
        
        # 6. Cache result
        if self.config.cache_enabled:
            self.cache.set(cache_key, result)
            self.metrics.update_cache_size(len(self.cache))
        
        return result

    def details(
        self,
        *,
        place_id: str,
        fields: str = "place_id,name,rating,user_ratings_total,formatted_address,price_level,opening_hours,geometry,url,website",
        language: str = "en",
    ) -> Dict[str, Any]:
        """Get place details with caching, retry logic, and comprehensive logging/metrics.
        
        Args:
            place_id: Google Places place ID
            fields: Comma-separated list of fields to retrieve
            language: Language code (default: "en")
            
        Returns:
            API response with place details
            
        Raises:
            Exception: If all retry attempts fail
            
        Validates: Requirements 5.5, 5.6, 5.7
        """
        # 1. Generate cache key
        cache_key = self._generate_cache_key(
            "details",
            place_id=place_id,
            fields=fields,
            language=language
        )
        
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
        url = f"{self.BASE}/place/details/json"
        params = {
            "place_id": place_id,
            "fields": fields,
            "key": self.api_key,
            "language": language,
        }
        
        result = self._call_with_retry(url, params)
        
        # 4. Validate and clean response
        self._validate_response(result, expected_key="result")
        result = self._clean_response_data(result)
        
        # 5. Cache result
        if self.config.cache_enabled:
            self.cache.set(cache_key, result)
            self.metrics.update_cache_size(len(self.cache))
        
        return result
    
    def _call_with_retry(
        self,
        url: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call API with retry logic and exponential backoff.
        
        Args:
            url: API endpoint URL
            params: Request parameters
            
        Returns:
            Parsed JSON response
            
        Raises:
            Exception: If all retry attempts fail
            
        Validates: Requirements 5.5, 5.6
        """
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                
                # Make API call
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.config.google_places_timeout
                )
                
                duration = time.time() - start_time
                
                # Check HTTP status
                response.raise_for_status()
                
                # Log and record metrics
                self.logger.log_api_call("google_places", "GET", duration, response.status_code)
                self.metrics.record_api_call("google_places", duration, response.status_code)
                
                # Parse JSON response
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON response: {e}")
                
                # Check API status
                api_status = result.get("status")
                if api_status and api_status != "OK" and api_status != "ZERO_RESULTS":
                    raise ValueError(f"API returned error status: {api_status}")
                
                return result
                
            except Exception as e:
                last_error = e
                duration = time.time() - start_time
                
                # Log error
                self.logger.log_error(e, {
                    "attempt": attempt,
                    "max_retries": self.config.max_retries,
                    "duration": duration,
                    "url": url
                })
                
                # Record error metrics
                status_code = getattr(e, "response", None)
                if status_code and hasattr(status_code, "status_code"):
                    status_code = status_code.status_code
                else:
                    status_code = 500
                
                self.metrics.record_api_call("google_places", duration, status_code)
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
    
    def _generate_cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key from request parameters.
        
        Uses SHA256 hash of concatenated parameters for consistent caching.
        
        Args:
            method: API method name ("text_search" or "details")
            **kwargs: Request parameters
            
        Returns:
            Cache key (SHA256 hex digest)
            
        Validates: Requirement 5.6
        """
        # Sort kwargs for consistent ordering
        sorted_params = sorted(kwargs.items())
        
        # Concatenate all parameters
        content = "|".join([
            method,
            *[f"{k}={v}" for k, v in sorted_params]
        ])
        
        # Generate SHA256 hash
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _validate_response(self, response: Dict[str, Any], expected_key: str):
        """Validate response structure.
        
        Performs basic validation to ensure response matches expected structure.
        
        Args:
            response: API response to validate
            expected_key: Expected key in response ("results" or "result")
            
        Raises:
            ValueError: If response doesn't match expected structure
            
        Validates: Requirement 5.7
        """
        # Check if response is a dict
        if not isinstance(response, dict):
            raise ValueError(f"Response must be a dict, got {type(response)}")
        
        # Check status field
        if "status" not in response:
            raise ValueError("Response missing 'status' field")
        
        # Check expected key (unless status is error)
        status = response.get("status")
        if status == "OK" and expected_key not in response:
            raise ValueError(f"Response missing expected key: {expected_key}")
    
    def _clean_response_data(self, data: Any) -> Any:
        """Clean response data by removing invalid or unnecessary fields.
        
        This is an idempotent operation - cleaning twice produces the same result.
        
        Args:
            data: Response data to clean
            
        Returns:
            Cleaned response data
            
        Validates: Requirement 6.5 (idempotence)
        """
        if isinstance(data, dict):
            # Remove None values and empty strings
            cleaned = {}
            for key, value in data.items():
                if value is not None and value != "":
                    cleaned[key] = self._clean_response_data(value)
            return cleaned
        elif isinstance(data, list):
            # Recursively clean list items
            return [self._clean_response_data(item) for item in data]
        else:
            return data
