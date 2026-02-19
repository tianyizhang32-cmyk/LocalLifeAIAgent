"""Unit tests for error handler module."""

import pytest
import time
from local_lifestyle_agent.infrastructure.error_handler import (
    ErrorHandler,
    ErrorResponse,
    ErrorCode
)


class TestErrorHandler:
    """Test suite for ErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = ErrorHandler(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=2
        )
    
    def test_handle_api_timeout_error(self):
        """Test handling of API timeout errors."""
        error = TimeoutError("API request timed out")
        context = {"api": "openai", "method": "POST"}
        
        response = self.handler.handle_api_error(error, context, request_id="test_123")
        
        assert response.error_code == ErrorCode.API_TIMEOUT
        assert "timed out" in response.error_message.lower()
        assert response.request_id == "test_123"
        assert response.retry_after is not None
        assert response.details["api"] == "openai"
    
    def test_handle_api_connection_error(self):
        """Test handling of API connection errors."""
        error = ConnectionError("Failed to connect")
        context = {"api": "google_places", "method": "GET"}
        
        response = self.handler.handle_api_error(error, context, request_id="test_456")
        
        assert response.error_code == ErrorCode.API_CONNECTION_ERROR
        assert "connect" in response.error_message.lower()
        assert response.request_id == "test_456"
        assert response.retry_after is not None
    
    def test_handle_api_rate_limit_error(self):
        """Test handling of API rate limit errors."""
        error = Exception("429 Too Many Requests")
        context = {"api": "openai"}
        
        response = self.handler.handle_api_error(error, context, request_id="test_789")
        
        assert response.error_code == ErrorCode.API_RATE_LIMIT
        assert "rate limit" in response.error_message.lower()
        assert response.retry_after is not None
    
    def test_handle_api_authentication_error(self):
        """Test handling of API authentication errors."""
        error = Exception("401 Unauthorized")
        context = {"api": "openai"}
        
        response = self.handler.handle_api_error(error, context)
        
        assert response.error_code == ErrorCode.API_AUTHENTICATION_ERROR
        assert "authentication" in response.error_message.lower()
        assert response.retry_after is None  # Should not retry auth errors
    
    def test_handle_api_server_error(self):
        """Test handling of API server errors (5xx)."""
        error = Exception("502 Bad Gateway")
        context = {"api": "google_places"}
        
        response = self.handler.handle_api_error(error, context)
        
        assert response.error_code == ErrorCode.API_SERVER_ERROR
        assert "server error" in response.error_message.lower()
        assert response.retry_after is not None
    
    def test_handle_validation_error(self):
        """Test handling of validation errors."""
        error = ValueError("Invalid input data")
        
        response = self.handler.handle_validation_error(error, request_id="test_val")
        
        assert response.error_code == ErrorCode.VALIDATION_ERROR
        assert "validation" in response.error_message.lower()
        assert response.request_id == "test_val"
        assert response.retry_after is None  # Should not retry validation errors
    
    def test_handle_timeout_error(self):
        """Test handling of timeout errors."""
        error = TimeoutError("Request timeout")
        
        response = self.handler.handle_timeout_error(error, request_id="test_timeout")
        
        assert response.error_code == ErrorCode.API_TIMEOUT
        assert "timed out" in response.error_message.lower()
        assert response.retry_after is not None
    
    def test_handle_rate_limit_error_with_retry_after(self):
        """Test handling of rate limit errors with custom retry-after."""
        error = Exception("Rate limit exceeded")
        
        response = self.handler.handle_rate_limit_error(
            error,
            retry_after=30,
            request_id="test_rate"
        )
        
        assert response.error_code == ErrorCode.API_RATE_LIMIT
        assert response.retry_after == 30
    
    def test_should_retry_timeout_error(self):
        """Test retry decision for timeout errors."""
        error = TimeoutError("Timeout")
        assert self.handler.should_retry(error) is True
    
    def test_should_retry_connection_error(self):
        """Test retry decision for connection errors."""
        error = ConnectionError("Connection failed")
        assert self.handler.should_retry(error) is True
    
    def test_should_retry_rate_limit_error(self):
        """Test retry decision for rate limit errors."""
        error = Exception("429 Too Many Requests")
        assert self.handler.should_retry(error) is True
    
    def test_should_retry_server_error(self):
        """Test retry decision for 5xx server errors."""
        error = Exception("502 Bad Gateway")
        assert self.handler.should_retry(error) is True
    
    def test_should_not_retry_authentication_error(self):
        """Test retry decision for authentication errors."""
        error = Exception("401 Unauthorized")
        assert self.handler.should_retry(error) is False
    
    def test_should_not_retry_bad_request(self):
        """Test retry decision for 4xx client errors."""
        error = Exception("400 Bad Request")
        assert self.handler.should_retry(error) is False
    
    def test_should_not_retry_not_found(self):
        """Test retry decision for 404 errors."""
        error = Exception("404 Not Found")
        assert self.handler.should_retry(error) is False
    
    def test_get_retry_delay_exponential_backoff(self):
        """Test exponential backoff calculation."""
        # Attempt 0: base_delay * (2^0) = 1.0 * 1 = 1.0 (+ jitter)
        delay_0 = self.handler.get_retry_delay(0)
        assert 1.0 <= delay_0 <= 1.5  # 1.0 + 50% jitter
        
        # Attempt 1: base_delay * (2^1) = 1.0 * 2 = 2.0 (+ jitter)
        delay_1 = self.handler.get_retry_delay(1)
        assert 2.0 <= delay_1 <= 3.0  # 2.0 + 50% jitter
        
        # Attempt 2: base_delay * (2^2) = 1.0 * 4 = 4.0 (+ jitter)
        delay_2 = self.handler.get_retry_delay(2)
        assert 4.0 <= delay_2 <= 6.0  # 4.0 + 50% jitter
    
    def test_get_retry_delay_max_cap(self):
        """Test that retry delay is capped at max_delay."""
        # Attempt 10: base_delay * (2^10) = 1.0 * 1024 = 1024.0
        # Should be capped at max_delay = 60.0
        delay = self.handler.get_retry_delay(10)
        assert delay <= 60
    
    def test_get_retry_delay_with_custom_base(self):
        """Test retry delay with custom exponential base."""
        handler = ErrorHandler(base_delay=2.0, exponential_base=3)
        
        # Attempt 0: 2.0 * (3^0) = 2.0 * 1 = 2.0 (+ jitter)
        delay_0 = handler.get_retry_delay(0)
        assert 2.0 <= delay_0 <= 3.0
        
        # Attempt 1: 2.0 * (3^1) = 2.0 * 3 = 6.0 (+ jitter)
        delay_1 = handler.get_retry_delay(1)
        assert 6.0 <= delay_1 <= 9.0


class TestErrorResponse:
    """Test suite for ErrorResponse model."""
    
    def test_error_response_creation(self):
        """Test creating an ErrorResponse."""
        response = ErrorResponse(
            error_code="TEST_ERROR",
            error_message="Test error message",
            request_id="test_123"
        )
        
        assert response.error_code == "TEST_ERROR"
        assert response.error_message == "Test error message"
        assert response.request_id == "test_123"
        assert response.details is None
        assert response.retry_after is None
        assert response.timestamp is not None
    
    def test_error_response_with_details(self):
        """Test ErrorResponse with details."""
        details = {"api": "openai", "method": "POST", "status": 500}
        response = ErrorResponse(
            error_code="API_ERROR",
            error_message="API failed",
            details=details,
            request_id="test_456"
        )
        
        assert response.details == details
        assert response.details["api"] == "openai"
    
    def test_error_response_with_retry_after(self):
        """Test ErrorResponse with retry_after."""
        response = ErrorResponse(
            error_code="RATE_LIMIT",
            error_message="Rate limit exceeded",
            retry_after=30,
            request_id="test_789"
        )
        
        assert response.retry_after == 30
    
    def test_error_response_timestamp_format(self):
        """Test that timestamp is in ISO format."""
        response = ErrorResponse(
            error_code="TEST",
            error_message="Test",
            request_id="test"
        )
        
        # Timestamp should be in format: YYYY-MM-DDTHH:MM:SS.000Z
        assert "T" in response.timestamp
        assert response.timestamp.endswith("Z")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
