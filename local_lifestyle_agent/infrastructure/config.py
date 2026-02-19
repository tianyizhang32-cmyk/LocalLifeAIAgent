"""Configuration management module for production-ready agent.

This module provides centralized configuration management with support for:
- Environment variables (highest priority)
- Configuration files (YAML)
- Default values (fallback)

Validates: Requirements 2.1, 2.2, 2.3, 2.4-2.8, 2.9
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator


class Config(BaseModel):
    """System configuration with environment variable and file support.
    
    Configuration priority:
    1. Environment variables (highest)
    2. Configuration file
    3. Default values (lowest)
    """
    
    # API Configuration
    openai_api_key: str = Field(default="")
    google_places_api_key: str = Field(default="")
    openai_timeout: int = Field(default=30, ge=1, le=300)
    google_places_timeout: int = Field(default=30, ge=1, le=300)
    openai_model: str = Field(default="gpt-4")
    
    # Retry Configuration
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_base_delay: float = Field(default=1.0, ge=0.1, le=10.0)
    retry_max_delay: float = Field(default=60.0, ge=1.0, le=300.0)
    retry_exponential_base: int = Field(default=2, ge=2, le=10)
    
    # Cache Configuration
    cache_enabled: bool = Field(default=True)
    cache_ttl: int = Field(default=3600, ge=60, le=86400)  # 1 hour default, max 24 hours
    cache_max_size: int = Field(default=1000, ge=10, le=100000)
    
    # Logging Configuration
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    log_file: Optional[str] = Field(default=None)
    
    # Rate Limiting Configuration
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests_per_minute: int = Field(default=60, ge=1, le=1000)
    
    # Performance Configuration
    max_concurrent_requests: int = Field(default=10, ge=1, le=100)
    connection_pool_size: int = Field(default=20, ge=1, le=100)
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the allowed values."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v}")
        return v_upper
    
    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format is one of the allowed values."""
        allowed = ["json", "text"]
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"log_format must be one of {allowed}, got {v}")
        return v_lower
    
    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables.
        
        Returns:
            Config instance populated from environment variables
        """
        env_config = {}
        
        # API Configuration
        if api_key := os.environ.get("OPENAI_API_KEY"):
            env_config["openai_api_key"] = api_key
        if api_key := os.environ.get("GOOGLE_PLACES_API_KEY"):
            env_config["google_places_api_key"] = api_key
        if timeout := os.environ.get("OPENAI_TIMEOUT"):
            env_config["openai_timeout"] = int(timeout)
        if timeout := os.environ.get("GOOGLE_PLACES_TIMEOUT"):
            env_config["google_places_timeout"] = int(timeout)
        if model := os.environ.get("OPENAI_MODEL"):
            env_config["openai_model"] = model
        
        # Retry Configuration
        if retries := os.environ.get("MAX_RETRIES"):
            env_config["max_retries"] = int(retries)
        if delay := os.environ.get("RETRY_BASE_DELAY"):
            env_config["retry_base_delay"] = float(delay)
        if max_delay := os.environ.get("RETRY_MAX_DELAY"):
            env_config["retry_max_delay"] = float(max_delay)
        if base := os.environ.get("RETRY_EXPONENTIAL_BASE"):
            env_config["retry_exponential_base"] = int(base)
        
        # Cache Configuration
        if enabled := os.environ.get("CACHE_ENABLED"):
            env_config["cache_enabled"] = enabled.lower() in ("true", "1", "yes")
        if ttl := os.environ.get("CACHE_TTL"):
            env_config["cache_ttl"] = int(ttl)
        if max_size := os.environ.get("CACHE_MAX_SIZE"):
            env_config["cache_max_size"] = int(max_size)
        
        # Logging Configuration
        if level := os.environ.get("LOG_LEVEL"):
            env_config["log_level"] = level
        if fmt := os.environ.get("LOG_FORMAT"):
            env_config["log_format"] = fmt
        if log_file := os.environ.get("LOG_FILE"):
            env_config["log_file"] = log_file
        
        # Rate Limiting Configuration
        if enabled := os.environ.get("RATE_LIMIT_ENABLED"):
            env_config["rate_limit_enabled"] = enabled.lower() in ("true", "1", "yes")
        if rpm := os.environ.get("RATE_LIMIT_REQUESTS_PER_MINUTE"):
            env_config["rate_limit_requests_per_minute"] = int(rpm)
        
        # Performance Configuration
        if max_concurrent := os.environ.get("MAX_CONCURRENT_REQUESTS"):
            env_config["max_concurrent_requests"] = int(max_concurrent)
        if pool_size := os.environ.get("CONNECTION_POOL_SIZE"):
            env_config["connection_pool_size"] = int(pool_size)
        
        return cls(**env_config)
    
    @classmethod
    def from_file(cls, path: str) -> Config:
        """Load configuration from a JSON or YAML file.
        
        Args:
            path: Path to configuration file (JSON or YAML)
            
        Returns:
            Config instance populated from file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file format is invalid
        """
        config_path = Path(path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        # Read file content
        content = config_path.read_text()
        
        # Parse based on file extension
        if config_path.suffix in [".json"]:
            try:
                file_config = json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in config file: {e}")
        elif config_path.suffix in [".yaml", ".yml"]:
            try:
                import yaml
                file_config = yaml.safe_load(content)
            except Exception as e:
                raise ValueError(f"Invalid YAML in config file: {e}")
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Flatten nested structure if needed
        flat_config = cls._flatten_config(file_config)
        
        return cls(**flat_config)
    
    @staticmethod
    def _flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested configuration structure.
        
        Converts nested dict like:
        {"api": {"openai": {"api_key": "xxx"}}}
        
        To flat dict like:
        {"openai_api_key": "xxx"}
        """
        flat = {}
        
        # Handle nested API configuration
        if "api" in config:
            api_config = config["api"]
            if "openai" in api_config:
                openai = api_config["openai"]
                if "api_key" in openai:
                    flat["openai_api_key"] = openai["api_key"]
                if "timeout" in openai:
                    flat["openai_timeout"] = openai["timeout"]
                if "model" in openai:
                    flat["openai_model"] = openai["model"]
            
            if "google_places" in api_config:
                gp = api_config["google_places"]
                if "api_key" in gp:
                    flat["google_places_api_key"] = gp["api_key"]
                if "timeout" in gp:
                    flat["google_places_timeout"] = gp["timeout"]
        
        # Handle nested retry configuration
        if "retry" in config:
            retry = config["retry"]
            if "max_retries" in retry:
                flat["max_retries"] = retry["max_retries"]
            if "base_delay" in retry:
                flat["retry_base_delay"] = retry["base_delay"]
            if "max_delay" in retry:
                flat["retry_max_delay"] = retry["max_delay"]
            if "exponential_base" in retry:
                flat["retry_exponential_base"] = retry["exponential_base"]
        
        # Handle nested cache configuration
        if "cache" in config:
            cache = config["cache"]
            if "enabled" in cache:
                flat["cache_enabled"] = cache["enabled"]
            if "ttl" in cache:
                flat["cache_ttl"] = cache["ttl"]
            if "max_size" in cache:
                flat["cache_max_size"] = cache["max_size"]
        
        # Handle nested logging configuration
        if "logging" in config:
            logging = config["logging"]
            if "level" in logging:
                flat["log_level"] = logging["level"]
            if "format" in logging:
                flat["log_format"] = logging["format"]
            if "file" in logging:
                flat["log_file"] = logging["file"]
        
        # Handle nested rate_limit configuration
        if "rate_limit" in config:
            rate_limit = config["rate_limit"]
            if "enabled" in rate_limit:
                flat["rate_limit_enabled"] = rate_limit["enabled"]
            if "requests_per_minute" in rate_limit:
                flat["rate_limit_requests_per_minute"] = rate_limit["requests_per_minute"]
        
        # Handle nested performance configuration
        if "performance" in config:
            perf = config["performance"]
            if "max_concurrent_requests" in perf:
                flat["max_concurrent_requests"] = perf["max_concurrent_requests"]
            if "connection_pool_size" in perf:
                flat["connection_pool_size"] = perf["connection_pool_size"]
        
        # Also handle flat configuration (direct keys)
        for key, value in config.items():
            if key not in ["api", "retry", "cache", "logging", "rate_limit", "performance"]:
                flat[key] = value
        
        return flat
    
    @classmethod
    def load(cls, config_file: Optional[str] = None) -> Config:
        """Load configuration with priority: env vars > config file > defaults.
        
        Args:
            config_file: Optional path to configuration file
            
        Returns:
            Config instance with merged configuration
        """
        # Start with defaults (implicit in Pydantic model)
        config_dict = {}
        
        # Load from file if provided
        if config_file:
            try:
                file_config = cls.from_file(config_file)
                config_dict = file_config.model_dump()
            except FileNotFoundError:
                # File not found, use defaults and log warning
                import warnings
                warnings.warn(f"Config file not found: {config_file}, using defaults")
            except ValueError as e:
                # Invalid config file, use defaults and log warning
                import warnings
                warnings.warn(f"Invalid config file: {e}, using defaults")
        
        # Load from environment (overrides file config)
        env_config = cls.from_env()
        env_dict = env_config.model_dump()
        
        # Merge: env overrides file overrides defaults
        # Only override if env value is not the default
        defaults = cls().model_dump()
        for key, env_value in env_dict.items():
            # If env value differs from default, use it (env was explicitly set)
            if env_value != defaults[key]:
                config_dict[key] = env_value
            # Otherwise, keep file value if it exists
            elif key not in config_dict:
                config_dict[key] = env_value
        
        return cls(**config_dict)
