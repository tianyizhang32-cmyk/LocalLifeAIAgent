"""Unit tests for enhanced Google Places Adapter.

Tests cover:
- Basic functionality (text_search, details)
- Caching behavior
- Retry logic
- Error handling
- Response validation
- Connection pooling

Validates: Requirements 5.5, 5.6, 5.7
"""

import json
import time
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests

from local_lifestyle_agent.adapters.google_places import GooglePlacesAdapter
from local_lifestyle_agent.infrastructure.cache import Cache
from local_lifestyle_agent.infrastructure.config import Config
from local_lifestyle_agent.infrastructure.error_handler import ErrorHandler
from local_lifestyle_agent.infrastructure.logger import StructuredLogger
from local_lifestyle_agent.infrastructure.metrics import MetricsCollector


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(
        google_places_api_key="test_key",
        google_places_timeout=30,
        max_retries=3,
        retry_base_delay=0.1,  # Short delay for tests
        retry_max_delay=1.0,
        cache_enabled=True,
        cache_ttl=3600,
        cache_max_size=100,
        log_level="INFO"
    )


@pytest.fixture
def cache():
    """Create test cache."""
    return Cache(max_size=100, ttl=3600)


@pytest.fixture
def logger():
    """Create test logger."""
    return StructuredLogger("test_google_places", log_level="INFO")


@pytest.fixture
def metrics():
    """Create test metrics collector."""
    return MetricsCollector()


@pytest.fixture
def error_handler():
    """Create test error handler."""
    return ErrorHandler(max_retries=3, base_delay=0.1, max_delay=1.0)


@pytest.fixture
def adapter(config, cache, logger, metrics, error_handler):
    """Create test adapter with all dependencies."""
    return GooglePlacesAdapter(
        api_key="test_key",
        config=config,
        cache=cache,
        logger=logger,
        metrics=metrics,
        error_handler=error_handler
    )


class TestGooglePlacesAdapterInitialization:
    """Test adapter initialization."""
    
    def test_init_with_all_dependencies(self, config, cache, logger, metrics, error_handler):
        """Test initialization with all dependencies provided."""
        adapter = GooglePlacesAdapter(
            api_key="test_key",
            config=config,
            cache=cache,
            logger=logger,
            metrics=metrics,
            error_handler=error_handler
        )
        
        assert adapter.api_key == "test_key"
        assert adapter.config == config
        assert adapter.cache == cache
        assert adapter.logger == logger
        assert adapter.metrics == metrics
        assert adapter.error_handler == error_handler
        assert adapter.session is not None
    
    def test_init_with_defaults(self):
        """Test initialization with default dependencies."""
        adapter = GooglePlacesAdapter(api_key="test_key")
        
        assert adapter.api_key == "test_key"
        assert adapter.config is not None
        assert adapter.cache is not None
        assert adapter.logger is not None
        assert adapter.metrics is not None
        assert adapter.error_handler is not None
        assert adapter.session is not None
    
    def test_session_has_connection_pool(self, adapter):
        """Test that session is configured with connection pooling."""
        # Check that session has adapters mounted
        assert "https://" in adapter.session.adapters
        assert "http://" in adapter.session.adapters


class TestTextSearch:
    """Test text_search method."""
    
    def test_text_search_success(self, adapter):
        """Test successful text search."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "results": [
                {
                    "place_id": "test_id_1",
                    "name": "Test Place 1",
                    "rating": 4.5
                },
                {
                    "place_id": "test_id_2",
                    "name": "Test Place 2",
                    "rating": 4.0
                }
            ]
        }
        
        with patch.object(adapter.session, "get", return_value=mock_response):
            result = adapter.text_search(query="afternoon tea")
        
        assert result["status"] == "OK"
        assert len(result["results"]) == 2
        assert result["results"][0]["name"] == "Test Place 1"
    
    def test_text_search_with_location_and_radius(self, adapter):
        """Test text search with location and radius parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "results": []
        }
        
        with patch.object(adapter.session, "get", return_value=mock_response) as mock_get:
            adapter.text_search(
                query="afternoon tea",
                location_latlng="47.6062,-122.3321",
                radius_m=5000
            )
            
            # Verify location and radius were passed
            call_args = mock_get.call_args
            params = call_args[1]["params"]
            assert params["location"] == "47.6062,-122.3321"
            assert params["radius"] == 5000
    
    def test_text_search_limits_results(self, adapter):
        """Test that text search limits results to max_results."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "results": [{"place_id": f"id_{i}"} for i in range(20)]
        }
        
        with patch.object(adapter.session, "get", return_value=mock_response):
            result = adapter.text_search(query="test", max_results=5)
        
        assert len(result["results"]) == 5
    
    def test_text_search_caching(self, adapter):
        """Test that text search results are cached."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "results": [{"place_id": "test_id"}]
        }
        
        with patch.object(adapter.session, "get", return_value=mock_response) as mock_get:
            # First call - should hit API
            result1 = adapter.text_search(query="afternoon tea")
            assert mock_get.call_count == 1
            
            # Second call with same params - should hit cache
            result2 = adapter.text_search(query="afternoon tea")
            assert mock_get.call_count == 1  # No additional API call
            
            # Results should be identical
            assert result1 == result2
    
    def test_text_search_cache_key_uniqueness(self, adapter):
        """Test that different parameters generate different cache keys."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "results": []
        }
        
        with patch.object(adapter.session, "get", return_value=mock_response) as mock_get:
            # Different queries should not share cache
            adapter.text_search(query="afternoon tea")
            adapter.text_search(query="coffee shop")
            
            assert mock_get.call_count == 2


class TestDetails:
    """Test details method."""
    
    def test_details_success(self, adapter):
        """Test successful place details retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "result": {
                "place_id": "test_id",
                "name": "Test Place",
                "rating": 4.5,
                "formatted_address": "123 Test St"
            }
        }
        
        with patch.object(adapter.session, "get", return_value=mock_response):
            result = adapter.details(place_id="test_id")
        
        assert result["status"] == "OK"
        assert result["result"]["name"] == "Test Place"
    
    def test_details_caching(self, adapter):
        """Test that place details are cached."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "result": {"place_id": "test_id"}
        }
        
        with patch.object(adapter.session, "get", return_value=mock_response) as mock_get:
            # First call - should hit API
            result1 = adapter.details(place_id="test_id")
            assert mock_get.call_count == 1
            
            # Second call with same place_id - should hit cache
            result2 = adapter.details(place_id="test_id")
            assert mock_get.call_count == 1  # No additional API call
            
            # Results should be identical
            assert result1 == result2


class TestRetryLogic:
    """Test retry logic and error handling."""
    
    def test_retry_on_timeout(self, adapter):
        """Test retry on timeout error."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "results": []
        }
        
        with patch.object(adapter.session, "get") as mock_get:
            # First two calls timeout, third succeeds
            mock_get.side_effect = [
                requests.Timeout("Timeout"),
                requests.Timeout("Timeout"),
                mock_response
            ]
            
            result = adapter.text_search(query="test")
            
            # Should have retried twice and succeeded on third attempt
            assert mock_get.call_count == 3
            assert result["status"] == "OK"
    
    def test_retry_on_500_error(self, adapter):
        """Test retry on 5xx server error."""
        mock_error_response = Mock()
        mock_error_response.status_code = 500
        mock_error_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        
        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {
            "status": "OK",
            "results": []
        }
        
        with patch.object(adapter.session, "get") as mock_get:
            # First call fails with 500, second succeeds
            mock_get.side_effect = [
                mock_error_response,
                mock_success_response
            ]
            
            result = adapter.text_search(query="test")
            
            # Should have retried once
            assert mock_get.call_count == 2
            assert result["status"] == "OK"
    
    def test_no_retry_on_400_error(self, adapter):
        """Test no retry on 4xx client error."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = requests.HTTPError("400 Bad Request")
        
        with patch.object(adapter.session, "get", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                adapter.text_search(query="test")
    
    def test_max_retries_exceeded(self, adapter):
        """Test that max retries limit is respected."""
        with patch.object(adapter.session, "get") as mock_get:
            # Always timeout
            mock_get.side_effect = requests.Timeout("Timeout")
            
            with pytest.raises(requests.Timeout):
                adapter.text_search(query="test")
            
            # Should have tried max_retries + 1 times (initial + retries)
            assert mock_get.call_count == adapter.config.max_retries + 1


class TestResponseValidation:
    """Test response validation."""
    
    def test_validate_missing_status(self, adapter):
        """Test validation fails when status field is missing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": []  # Missing status field
        }
        
        with patch.object(adapter.session, "get", return_value=mock_response):
            with pytest.raises(ValueError, match="missing 'status' field"):
                adapter.text_search(query="test")
    
    def test_validate_missing_results(self, adapter):
        """Test validation fails when expected key is missing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK"
            # Missing results field
        }
        
        with patch.object(adapter.session, "get", return_value=mock_response):
            with pytest.raises(ValueError, match="missing expected key"):
                adapter.text_search(query="test")
    
    def test_validate_invalid_json(self, adapter):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        with patch.object(adapter.session, "get", return_value=mock_response):
            with pytest.raises(ValueError, match="Invalid JSON response"):
                adapter.text_search(query="test")


class TestDataCleaning:
    """Test response data cleaning."""
    
    def test_clean_removes_none_values(self, adapter):
        """Test that cleaning removes None values."""
        data = {
            "name": "Test",
            "rating": None,
            "address": "123 Test St"
        }
        
        cleaned = adapter._clean_response_data(data)
        
        assert "name" in cleaned
        assert "rating" not in cleaned
        assert "address" in cleaned
    
    def test_clean_removes_empty_strings(self, adapter):
        """Test that cleaning removes empty strings."""
        data = {
            "name": "Test",
            "description": "",
            "address": "123 Test St"
        }
        
        cleaned = adapter._clean_response_data(data)
        
        assert "name" in cleaned
        assert "description" not in cleaned
        assert "address" in cleaned
    
    def test_clean_is_idempotent(self, adapter):
        """Test that cleaning is idempotent (cleaning twice = cleaning once)."""
        data = {
            "name": "Test",
            "rating": None,
            "description": "",
            "nested": {
                "value": "test",
                "empty": None
            }
        }
        
        cleaned_once = adapter._clean_response_data(data)
        cleaned_twice = adapter._clean_response_data(cleaned_once)
        
        assert cleaned_once == cleaned_twice
    
    def test_clean_handles_nested_structures(self, adapter):
        """Test that cleaning handles nested dicts and lists."""
        data = {
            "name": "Test",
            "nested": {
                "value": "test",
                "empty": None
            },
            "list": [
                {"id": 1, "empty": ""},
                {"id": 2, "value": "test"}
            ]
        }
        
        cleaned = adapter._clean_response_data(data)
        
        assert "empty" not in cleaned["nested"]
        assert "empty" not in cleaned["list"][0]
        assert len(cleaned["list"]) == 2


class TestMetricsAndLogging:
    """Test metrics collection and logging."""
    
    def test_records_api_call_metrics(self, adapter):
        """Test that API calls are recorded in metrics."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "results": []
        }
        
        initial_count = adapter.metrics.api_call_total.get({"api": "google_places", "status": "200"})
        
        with patch.object(adapter.session, "get", return_value=mock_response):
            adapter.text_search(query="test")
        
        final_count = adapter.metrics.api_call_total.get({"api": "google_places", "status": "200"})
        
        assert final_count == initial_count + 1
    
    def test_records_cache_hit_metrics(self, adapter):
        """Test that cache hits are recorded in metrics."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "results": []
        }
        
        with patch.object(adapter.session, "get", return_value=mock_response):
            # First call - cache miss
            initial_hits = adapter.metrics.cache_hit_total.get()
            adapter.text_search(query="test")
            
            # Second call - cache hit
            adapter.text_search(query="test")
            final_hits = adapter.metrics.cache_hit_total.get()
            
            assert final_hits == initial_hits + 1
    
    def test_records_error_metrics(self, adapter):
        """Test that errors are recorded in metrics."""
        initial_count = adapter.metrics.error_total.get({"error_type": "Timeout"})
        
        with patch.object(adapter.session, "get", side_effect=requests.Timeout("Timeout")):
            with pytest.raises(requests.Timeout):
                adapter.text_search(query="test")
        
        final_count = adapter.metrics.error_total.get({"error_type": "Timeout"})
        
        # Should have recorded error for each retry attempt
        assert final_count > initial_count


class TestCacheKeyGeneration:
    """Test cache key generation."""
    
    def test_cache_key_is_consistent(self, adapter):
        """Test that same parameters generate same cache key."""
        key1 = adapter._generate_cache_key(
            "text_search",
            query="afternoon tea",
            language="en",
            max_results=10
        )
        
        key2 = adapter._generate_cache_key(
            "text_search",
            query="afternoon tea",
            language="en",
            max_results=10
        )
        
        assert key1 == key2
    
    def test_cache_key_differs_for_different_params(self, adapter):
        """Test that different parameters generate different cache keys."""
        key1 = adapter._generate_cache_key(
            "text_search",
            query="afternoon tea"
        )
        
        key2 = adapter._generate_cache_key(
            "text_search",
            query="coffee shop"
        )
        
        assert key1 != key2
    
    def test_cache_key_differs_for_different_methods(self, adapter):
        """Test that different methods generate different cache keys."""
        key1 = adapter._generate_cache_key(
            "text_search",
            query="test"
        )
        
        key2 = adapter._generate_cache_key(
            "details",
            place_id="test"
        )
        
        assert key1 != key2
