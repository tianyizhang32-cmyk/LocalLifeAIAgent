"""Unit tests for enhanced LLM Client.

Tests cover:
- Cache integration and hit/miss behavior
- Retry logic with exponential backoff
- Response validation
- Error handling
- Logging and metrics integration

Validates: Requirements 5.1, 5.2, 5.3, 5.4
"""

import json
import time
from unittest.mock import Mock, patch, MagicMock

import pytest

from local_lifestyle_agent.llm_client import LLMClient, _enforce_no_additional_properties
from local_lifestyle_agent.infrastructure.cache import Cache
from local_lifestyle_agent.infrastructure.config import Config
from local_lifestyle_agent.infrastructure.error_handler import ErrorHandler
from local_lifestyle_agent.infrastructure.logger import StructuredLogger
from local_lifestyle_agent.infrastructure.metrics import MetricsCollector


class TestEnforceNoAdditionalProperties:
    """Test schema sanitization function."""
    
    def test_object_schema_gets_additional_properties_false(self):
        """Test that object schemas get additionalProperties: false."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = _enforce_no_additional_properties(schema)
        
        assert result["additionalProperties"] is False
    
    def test_nested_objects_get_additional_properties_false(self):
        """Test that nested objects also get additionalProperties: false."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}}
                }
            }
        }
        result = _enforce_no_additional_properties(schema)
        
        assert result["additionalProperties"] is False
        assert result["properties"]["user"]["additionalProperties"] is False
    
    def test_array_items_processed(self):
        """Test that array item schemas are processed."""
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}}
            }
        }
        result = _enforce_no_additional_properties(schema)
        
        assert result["items"]["additionalProperties"] is False


class TestLLMClientInitialization:
    """Test LLM Client initialization."""
    
    def test_init_with_defaults(self):
        """Test initialization with default components."""
        client = LLMClient(api_key="test-key")
        
        assert client.api_key == "test-key"
        assert client.model == "gpt-5.2"
        assert isinstance(client.config, Config)
        assert isinstance(client.cache, Cache)
        assert isinstance(client.logger, StructuredLogger)
        assert isinstance(client.metrics, MetricsCollector)
        assert isinstance(client.error_handler, ErrorHandler)
    
    def test_init_with_custom_components(self):
        """Test initialization with custom components."""
        config = Config(cache_enabled=False, max_retries=5)
        cache = Cache(max_size=500, ttl=1800)
        logger = StructuredLogger("test", log_level="DEBUG")
        metrics = MetricsCollector()
        error_handler = ErrorHandler(max_retries=5)
        
        client = LLMClient(
            api_key="test-key",
            model="gpt-4",
            config=config,
            cache=cache,
            logger=logger,
            metrics=metrics,
            error_handler=error_handler
        )
        
        assert client.model == "gpt-4"
        assert client.config.cache_enabled is False
        assert client.config.max_retries == 5
        assert client.cache.max_size == 500


class TestCacheIntegration:
    """Test cache integration in LLM Client."""
    
    def test_cache_hit_returns_cached_result(self):
        """Test that cache hit returns cached result without API call."""
        config = Config(cache_enabled=True)
        cache = Cache(max_size=100, ttl=3600)
        metrics = MetricsCollector()
        
        # Create client - it will use the provided cache instance
        client = LLMClient(
            api_key="test-key",
            config=config,
            cache=cache,
            metrics=metrics
        )
        
        # Pre-populate cache using the client's cache instance
        cache_key = client._generate_cache_key(
            "system", "user", {"type": "object"}, "test"
        )
        cached_result = {"cached": True}
        client.cache.set(cache_key, cached_result)
        
        # Mock the API call to ensure it's not called
        with patch.object(client, "_call_with_retry") as mock_call:
            result = client.json_schema(
                system="system",
                user="user",
                schema={"type": "object"},
                schema_name="test"
            )
        
        # Verify cache hit
        assert result == cached_result
        assert mock_call.call_count == 0
        assert metrics.cache_hit_total.get() == 1
    
    def test_cache_miss_calls_api(self):
        """Test that cache miss calls API and caches result."""
        config = Config(cache_enabled=True)
        cache = Cache(max_size=100, ttl=3600)
        metrics = MetricsCollector()
        
        client = LLMClient(
            api_key="test-key",
            config=config,
            cache=cache,
            metrics=metrics
        )
        
        api_result = {"from_api": True}
        
        # Mock the API call
        with patch.object(client, "_call_with_retry", return_value=api_result):
            result = client.json_schema(
                system="system",
                user="user",
                schema={"type": "object"},
                schema_name="test"
            )
        
        # Verify API was called and result cached
        assert result == api_result
        assert metrics.cache_miss_total.get() == 1
        
        # Verify result is now in cache (use client's cache instance)
        cache_key = client._generate_cache_key(
            "system", "user", {"type": "object"}, "test"
        )
        assert client.cache.get(cache_key) == api_result
    
    def test_cache_disabled_always_calls_api(self):
        """Test that disabled cache always calls API."""
        config = Config(cache_enabled=False)
        cache = Cache(max_size=100, ttl=3600)
        
        client = LLMClient(
            api_key="test-key",
            config=config,
            cache=cache
        )
        
        # Pre-populate cache (should be ignored)
        cache_key = client._generate_cache_key(
            "system", "user", {"type": "object"}, "test"
        )
        cache.set(cache_key, {"cached": True})
        
        api_result = {"from_api": True}
        
        # Mock the API call
        with patch.object(client, "_call_with_retry", return_value=api_result):
            result = client.json_schema(
                system="system",
                user="user",
                schema={"type": "object"},
                schema_name="test"
            )
        
        # Verify API was called despite cache
        assert result == api_result


class TestCacheKeyGeneration:
    """Test cache key generation."""
    
    def test_same_inputs_generate_same_key(self):
        """Test that identical inputs generate identical cache keys."""
        client = LLMClient(api_key="test-key")
        
        key1 = client._generate_cache_key(
            "system", "user", {"type": "object"}, "test"
        )
        key2 = client._generate_cache_key(
            "system", "user", {"type": "object"}, "test"
        )
        
        assert key1 == key2
    
    def test_different_inputs_generate_different_keys(self):
        """Test that different inputs generate different cache keys."""
        client = LLMClient(api_key="test-key")
        
        key1 = client._generate_cache_key(
            "system1", "user", {"type": "object"}, "test"
        )
        key2 = client._generate_cache_key(
            "system2", "user", {"type": "object"}, "test"
        )
        
        assert key1 != key2
    
    def test_schema_order_doesnt_affect_key(self):
        """Test that schema property order doesn't affect cache key."""
        client = LLMClient(api_key="test-key")
        
        schema1 = {"type": "object", "properties": {"a": {}, "b": {}}}
        schema2 = {"properties": {"b": {}, "a": {}}, "type": "object"}
        
        key1 = client._generate_cache_key("system", "user", schema1, "test")
        key2 = client._generate_cache_key("system", "user", schema2, "test")
        
        # Keys should be the same due to sort_keys=True in json.dumps
        assert key1 == key2


class TestRetryLogic:
    """Test retry logic with exponential backoff."""
    
    def test_successful_call_no_retry(self):
        """Test that successful call doesn't retry."""
        config = Config(max_retries=3, cache_enabled=False)
        client = LLMClient(api_key="test-key", config=config)
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.output_text = '{"result": "success"}'
        
        mock_create = Mock(return_value=mock_response)
        with patch.object(client.client.responses, "create", mock_create):
            result = client._call_with_retry(
                "system", "user", {"type": "object"}, "test", True
            )
        
        assert result == {"result": "success"}
        assert mock_create.call_count == 1
    
    def test_retries_on_timeout_error(self):
        """Test that timeout errors trigger retry."""
        config = Config(max_retries=2, cache_enabled=False, retry_base_delay=0.1)
        error_handler = ErrorHandler(max_retries=2, base_delay=0.1)
        client = LLMClient(
            api_key="test-key",
            config=config,
            error_handler=error_handler
        )
        
        # Mock: first call fails with timeout, second succeeds
        mock_response = MagicMock()
        mock_response.output_text = '{"result": "success"}'
        
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("API timeout")
            return mock_response
        
        with patch.object(client.client.responses, "create", side_effect=side_effect):
            result = client._call_with_retry(
                "system", "user", {"type": "object"}, "test", True
            )
        
        assert result == {"result": "success"}
        assert call_count == 2  # First failed, second succeeded
    
    def test_fails_after_max_retries(self):
        """Test that call fails after max retries exceeded."""
        config = Config(max_retries=2, cache_enabled=False, retry_base_delay=0.1)
        error_handler = ErrorHandler(max_retries=2, base_delay=0.1)
        client = LLMClient(
            api_key="test-key",
            config=config,
            error_handler=error_handler
        )
        
        # Mock: all calls fail
        mock_create = Mock(side_effect=TimeoutError("API timeout"))
        with patch.object(client.client.responses, "create", mock_create):
            with pytest.raises(TimeoutError):
                client._call_with_retry(
                    "system", "user", {"type": "object"}, "test", True
                )
        
        # Should have tried: initial + 2 retries = 3 times
        assert mock_create.call_count == 3
    
    def test_no_retry_on_authentication_error(self):
        """Test that authentication errors don't trigger retry."""
        config = Config(max_retries=3, cache_enabled=False)
        client = LLMClient(api_key="test-key", config=config)
        
        # Mock authentication error
        auth_error = Exception("401 Unauthorized")
        
        mock_create = Mock(side_effect=auth_error)
        with patch.object(client.client.responses, "create", mock_create):
            with pytest.raises(Exception, match="401"):
                client._call_with_retry(
                    "system", "user", {"type": "object"}, "test", True
                )
        
        # Should only try once (no retry for auth errors)
        assert mock_create.call_count == 1


class TestResponseValidation:
    """Test response validation."""
    
    def test_valid_response_passes(self):
        """Test that valid response passes validation."""
        client = LLMClient(api_key="test-key")
        
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        response = {"name": "Alice", "age": 30}
        
        # Should not raise
        client._validate_response(response, schema)
    
    def test_missing_required_field_fails(self):
        """Test that missing required field fails validation."""
        client = LLMClient(api_key="test-key")
        
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        response = {"name": "Alice"}  # Missing 'age'
        
        with pytest.raises(ValueError, match="Missing required field: age"):
            client._validate_response(response, schema)
    
    def test_wrong_type_fails(self):
        """Test that wrong type fails validation."""
        client = LLMClient(api_key="test-key")
        
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer"}
            }
        }
        response = {"age": "thirty"}  # Should be integer
        
        with pytest.raises(ValueError, match="should be integer"):
            client._validate_response(response, schema)
    
    def test_non_dict_response_fails(self):
        """Test that non-dict response fails validation."""
        client = LLMClient(api_key="test-key")
        
        schema = {"type": "object"}
        response = ["not", "a", "dict"]
        
        with pytest.raises(ValueError, match="Response must be a dict"):
            client._validate_response(response, schema)


class TestMetricsIntegration:
    """Test metrics collection integration."""
    
    def test_successful_call_records_metrics(self):
        """Test that successful API call records metrics."""
        config = Config(cache_enabled=False)
        metrics = MetricsCollector()
        client = LLMClient(api_key="test-key", config=config, metrics=metrics)
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.output_text = '{"result": "success"}'
        
        with patch.object(client.client.responses, "create", return_value=mock_response):
            client._call_with_retry(
                "system", "user", {"type": "object"}, "test", True
            )
        
        # Verify metrics recorded
        assert metrics.api_call_total.get({"api": "openai", "status": "200"}) == 1
        assert metrics.api_call_duration_seconds.get_count({"api": "openai"}) == 1
    
    def test_failed_call_records_error_metrics(self):
        """Test that failed API call records error metrics."""
        config = Config(max_retries=0, cache_enabled=False)
        metrics = MetricsCollector()
        client = LLMClient(api_key="test-key", config=config, metrics=metrics)
        
        # Mock failed API call
        with patch.object(
            client.client.responses,
            "create",
            side_effect=TimeoutError("API timeout")
        ):
            with pytest.raises(TimeoutError):
                client._call_with_retry(
                    "system", "user", {"type": "object"}, "test", True
                )
        
        # Verify error metrics recorded
        assert metrics.api_call_total.get({"api": "openai", "status": "500"}) == 1
        assert metrics.error_total.get({"error_type": "TimeoutError"}) == 1


class TestLoggingIntegration:
    """Test logging integration."""
    
    def test_successful_call_logs_api_call(self):
        """Test that successful call logs API call details."""
        config = Config(cache_enabled=False)
        logger = StructuredLogger("test", log_level="INFO")
        client = LLMClient(api_key="test-key", config=config, logger=logger)
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.output_text = '{"result": "success"}'
        
        with patch.object(client.client.responses, "create", return_value=mock_response):
            with patch.object(logger, "log_api_call") as mock_log:
                client._call_with_retry(
                    "system", "user", {"type": "object"}, "test", True
                )
                
                # Verify API call was logged
                assert mock_log.call_count == 1
                args = mock_log.call_args[0]
                assert args[0] == "openai"
                assert args[1] == "POST"
                assert args[3] == 200
    
    def test_failed_call_logs_error(self):
        """Test that failed call logs error details."""
        config = Config(max_retries=0, cache_enabled=False)
        logger = StructuredLogger("test", log_level="INFO")
        client = LLMClient(api_key="test-key", config=config, logger=logger)
        
        # Mock failed API call
        with patch.object(
            client.client.responses,
            "create",
            side_effect=TimeoutError("API timeout")
        ):
            with patch.object(logger, "log_error") as mock_log_error:
                with pytest.raises(TimeoutError):
                    client._call_with_retry(
                        "system", "user", {"type": "object"}, "test", True
                    )
                
                # Verify error was logged
                assert mock_log_error.call_count == 1
