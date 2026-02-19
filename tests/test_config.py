"""Unit tests for configuration management module.

Tests configuration loading from environment variables, files, and defaults.
Validates: Requirements 2.1, 2.2, 2.3, 2.4-2.8, 2.9
"""

import os
import json
import tempfile
from pathlib import Path

import pytest

from local_lifestyle_agent.infrastructure.config import Config


class TestConfigFromEnv:
    """Test configuration loading from environment variables."""
    
    def test_from_env_with_api_keys(self, monkeypatch):
        """Test loading API keys from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "test-google-key")
        
        config = Config.from_env()
        
        assert config.openai_api_key == "test-openai-key"
        assert config.google_places_api_key == "test-google-key"
    
    def test_from_env_with_timeouts(self, monkeypatch):
        """Test loading timeout settings from environment variables."""
        monkeypatch.setenv("OPENAI_TIMEOUT", "45")
        monkeypatch.setenv("GOOGLE_PLACES_TIMEOUT", "60")
        
        config = Config.from_env()
        
        assert config.openai_timeout == 45
        assert config.google_places_timeout == 60
    
    def test_from_env_with_retry_config(self, monkeypatch):
        """Test loading retry configuration from environment variables."""
        monkeypatch.setenv("MAX_RETRIES", "5")
        monkeypatch.setenv("RETRY_BASE_DELAY", "2.0")
        monkeypatch.setenv("RETRY_MAX_DELAY", "120.0")
        
        config = Config.from_env()
        
        assert config.max_retries == 5
        assert config.retry_base_delay == 2.0
        assert config.retry_max_delay == 120.0
    
    def test_from_env_with_cache_config(self, monkeypatch):
        """Test loading cache configuration from environment variables."""
        monkeypatch.setenv("CACHE_ENABLED", "false")
        monkeypatch.setenv("CACHE_TTL", "7200")
        monkeypatch.setenv("CACHE_MAX_SIZE", "2000")
        
        config = Config.from_env()
        
        assert config.cache_enabled is False
        assert config.cache_ttl == 7200
        assert config.cache_max_size == 2000
    
    def test_from_env_with_logging_config(self, monkeypatch):
        """Test loading logging configuration from environment variables."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("LOG_FORMAT", "text")
        monkeypatch.setenv("LOG_FILE", "/var/log/agent.log")
        
        config = Config.from_env()
        
        assert config.log_level == "DEBUG"
        assert config.log_format == "text"
        assert config.log_file == "/var/log/agent.log"
    
    def test_from_env_with_rate_limit_config(self, monkeypatch):
        """Test loading rate limit configuration from environment variables."""
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
        monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "120")
        
        config = Config.from_env()
        
        assert config.rate_limit_enabled is False
        assert config.rate_limit_requests_per_minute == 120
    
    def test_from_env_with_performance_config(self, monkeypatch):
        """Test loading performance configuration from environment variables."""
        monkeypatch.setenv("MAX_CONCURRENT_REQUESTS", "20")
        monkeypatch.setenv("CONNECTION_POOL_SIZE", "50")
        
        config = Config.from_env()
        
        assert config.max_concurrent_requests == 20
        assert config.connection_pool_size == 50
    
    def test_from_env_empty_returns_defaults(self):
        """Test that from_env returns defaults when no env vars are set."""
        # Clear any existing env vars
        env_vars = [
            "OPENAI_API_KEY", "GOOGLE_PLACES_API_KEY", "OPENAI_TIMEOUT",
            "MAX_RETRIES", "CACHE_ENABLED", "LOG_LEVEL"
        ]
        for var in env_vars:
            os.environ.pop(var, None)
        
        config = Config.from_env()
        
        # Should have default values
        assert config.openai_timeout == 30
        assert config.max_retries == 3
        assert config.cache_enabled is True
        assert config.log_level == "INFO"


class TestConfigFromFile:
    """Test configuration loading from files."""
    
    def test_from_file_json_nested(self):
        """Test loading configuration from nested JSON file."""
        config_data = {
            "api": {
                "openai": {
                    "api_key": "file-openai-key",
                    "timeout": 45,
                    "model": "gpt-4-turbo"
                },
                "google_places": {
                    "api_key": "file-google-key",
                    "timeout": 60
                }
            },
            "retry": {
                "max_retries": 5,
                "base_delay": 2.0
            },
            "cache": {
                "enabled": False,
                "ttl": 7200
            },
            "logging": {
                "level": "DEBUG",
                "format": "text"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = Config.from_file(temp_path)
            
            assert config.openai_api_key == "file-openai-key"
            assert config.openai_timeout == 45
            assert config.openai_model == "gpt-4-turbo"
            assert config.google_places_api_key == "file-google-key"
            assert config.google_places_timeout == 60
            assert config.max_retries == 5
            assert config.retry_base_delay == 2.0
            assert config.cache_enabled is False
            assert config.cache_ttl == 7200
            assert config.log_level == "DEBUG"
            assert config.log_format == "text"
        finally:
            os.unlink(temp_path)
    
    def test_from_file_json_flat(self):
        """Test loading configuration from flat JSON file."""
        config_data = {
            "openai_api_key": "flat-openai-key",
            "google_places_api_key": "flat-google-key",
            "openai_timeout": 50,
            "max_retries": 4,
            "cache_enabled": True,
            "log_level": "WARNING"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = Config.from_file(temp_path)
            
            assert config.openai_api_key == "flat-openai-key"
            assert config.google_places_api_key == "flat-google-key"
            assert config.openai_timeout == 50
            assert config.max_retries == 4
            assert config.cache_enabled is True
            assert config.log_level == "WARNING"
        finally:
            os.unlink(temp_path)
    
    def test_from_file_not_found(self):
        """Test that from_file raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            Config.from_file("/nonexistent/config.json")
    
    def test_from_file_invalid_json(self):
        """Test that from_file raises ValueError for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                Config.from_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_from_file_unsupported_format(self):
        """Test that from_file raises ValueError for unsupported file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("some text")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported config file format"):
                Config.from_file(temp_path)
        finally:
            os.unlink(temp_path)


class TestConfigLoad:
    """Test configuration loading with priority: env > file > defaults."""
    
    def test_load_defaults_only(self):
        """Test that load returns defaults when no file or env vars."""
        # Clear env vars
        for var in os.environ.copy():
            if var.startswith(("OPENAI_", "GOOGLE_", "MAX_", "CACHE_", "LOG_", "RATE_", "CONNECTION_", "RETRY_")):
                os.environ.pop(var, None)
        
        config = Config.load()
        
        # Should have default values
        assert config.openai_timeout == 30
        assert config.google_places_timeout == 30
        assert config.max_retries == 3
        assert config.retry_base_delay == 1.0
        assert config.cache_enabled is True
        assert config.cache_ttl == 3600
        assert config.log_level == "INFO"
        assert config.log_format == "json"
    
    def test_load_file_overrides_defaults(self):
        """Test that file configuration overrides defaults."""
        config_data = {
            "openai_timeout": 45,
            "max_retries": 5,
            "cache_enabled": False,
            "log_level": "DEBUG"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = Config.load(config_file=temp_path)
            
            # File values should override defaults
            assert config.openai_timeout == 45
            assert config.max_retries == 5
            assert config.cache_enabled is False
            assert config.log_level == "DEBUG"
            
            # Non-specified values should use defaults
            assert config.google_places_timeout == 30
            assert config.retry_base_delay == 1.0
        finally:
            os.unlink(temp_path)
    
    def test_load_env_overrides_file(self, monkeypatch):
        """Test that environment variables override file configuration."""
        # Create config file
        config_data = {
            "openai_timeout": 45,
            "max_retries": 5,
            "log_level": "DEBUG"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Set env vars that conflict with file
            monkeypatch.setenv("OPENAI_TIMEOUT", "60")
            monkeypatch.setenv("LOG_LEVEL", "ERROR")
            
            config = Config.load(config_file=temp_path)
            
            # Env vars should override file
            assert config.openai_timeout == 60
            assert config.log_level == "ERROR"
            
            # File value should be used when no env var
            assert config.max_retries == 5
        finally:
            os.unlink(temp_path)
    
    def test_load_missing_file_uses_defaults(self):
        """Test that load uses defaults when config file is missing."""
        config = Config.load(config_file="/nonexistent/config.json")
        
        # Should use defaults without error
        assert config.openai_timeout == 30
        assert config.max_retries == 3


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_log_level_validation(self):
        """Test that invalid log level raises validation error."""
        with pytest.raises(ValueError, match="log_level must be one of"):
            Config(log_level="INVALID")
    
    def test_log_level_case_insensitive(self):
        """Test that log level is case-insensitive."""
        config = Config(log_level="debug")
        assert config.log_level == "DEBUG"
        
        config = Config(log_level="Info")
        assert config.log_level == "INFO"
    
    def test_log_format_validation(self):
        """Test that invalid log format raises validation error."""
        with pytest.raises(ValueError, match="log_format must be one of"):
            Config(log_format="xml")
    
    def test_log_format_case_insensitive(self):
        """Test that log format is case-insensitive."""
        config = Config(log_format="JSON")
        assert config.log_format == "json"
        
        config = Config(log_format="Text")
        assert config.log_format == "text"
    
    def test_timeout_range_validation(self):
        """Test that timeout values are validated."""
        # Valid timeouts
        config = Config(openai_timeout=1)
        assert config.openai_timeout == 1
        
        config = Config(openai_timeout=300)
        assert config.openai_timeout == 300
        
        # Invalid timeouts
        with pytest.raises(ValueError):
            Config(openai_timeout=0)
        
        with pytest.raises(ValueError):
            Config(openai_timeout=301)
    
    def test_retry_range_validation(self):
        """Test that retry values are validated."""
        # Valid retries
        config = Config(max_retries=0)
        assert config.max_retries == 0
        
        config = Config(max_retries=10)
        assert config.max_retries == 10
        
        # Invalid retries
        with pytest.raises(ValueError):
            Config(max_retries=-1)
        
        with pytest.raises(ValueError):
            Config(max_retries=11)
    
    def test_cache_ttl_range_validation(self):
        """Test that cache TTL is validated."""
        # Valid TTL
        config = Config(cache_ttl=60)
        assert config.cache_ttl == 60
        
        config = Config(cache_ttl=86400)
        assert config.cache_ttl == 86400
        
        # Invalid TTL
        with pytest.raises(ValueError):
            Config(cache_ttl=59)
        
        with pytest.raises(ValueError):
            Config(cache_ttl=86401)


class TestConfigDefaults:
    """Test that all configuration items have appropriate defaults."""
    
    def test_default_values(self):
        """Test that Config has sensible default values."""
        config = Config()
        
        # API defaults
        assert config.openai_timeout == 30
        assert config.google_places_timeout == 30
        assert config.openai_model == "gpt-4"
        
        # Retry defaults
        assert config.max_retries == 3
        assert config.retry_base_delay == 1.0
        assert config.retry_max_delay == 60.0
        assert config.retry_exponential_base == 2
        
        # Cache defaults
        assert config.cache_enabled is True
        assert config.cache_ttl == 3600
        assert config.cache_max_size == 1000
        
        # Logging defaults
        assert config.log_level == "INFO"
        assert config.log_format == "json"
        assert config.log_file is None
        
        # Rate limiting defaults
        assert config.rate_limit_enabled is True
        assert config.rate_limit_requests_per_minute == 60
        
        # Performance defaults
        assert config.max_concurrent_requests == 10
        assert config.connection_pool_size == 20
