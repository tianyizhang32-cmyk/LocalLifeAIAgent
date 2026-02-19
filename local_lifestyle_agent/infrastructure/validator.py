"""Data validation and sanitization module

This module provides data validation and input sanitization functionality to ensure data quality and security.

Features:
- Validate NormalizedIntent data structure
- Validate required fields, numeric ranges, enum values
- Sanitize user input (remove malicious content)
- Limit input length (prevent DoS attacks)

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.6, 6.7, 6.8, 6.9, 6.10
"""

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Validation result model
    
    Attributes:
        valid: Whether validation passed
        errors: List of validation errors
    """
    valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")


class DataValidator:
    """Data validator
    
    Provides data validation and sanitization functionality to ensure data quality and security.
    """
    
    # Input length limits
    MAX_INPUT_LENGTH = 1000
    MAX_CITY_LENGTH = 100
    MAX_QUERY_LENGTH = 500
    
    # Malicious content patterns
    MALICIOUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",  # XSS: script tags
        r"javascript:",  # XSS: javascript protocol
        r"on\w+\s*=",  # XSS: event handlers
        r"<iframe[^>]*>",  # XSS: iframe tags
        r"SELECT\s+.*\s+FROM",  # SQL injection: SELECT
        r"INSERT\s+INTO",  # SQL injection: INSERT
        r"UPDATE\s+.*\s+SET",  # SQL injection: UPDATE
        r"DELETE\s+FROM",  # SQL injection: DELETE
        r"DROP\s+TABLE",  # SQL injection: DROP
        r"\.\./",  # Path traversal
        r"\.\.\\",  # Path traversal (Windows)
    ]
    
    # Valid budget levels
    VALID_BUDGET_LEVELS = ["low", "medium", "high"]
    
    # Valid days
    VALID_DAYS = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]
    
    @classmethod
    def validate_normalized_intent(cls, intent: Dict[str, Any]) -> ValidationResult:
        """Validate NormalizedIntent data structure
        
        Validation rules:
        - Required fields: city, time_window, max_travel_minutes, party_size, budget_level
        - Numeric ranges: max_travel_minutes (5-120), party_size (1-50)
        - Enum values: budget_level (low/medium/high)
        - Input length: city (max 100 characters)
        
        Args:
            intent: NormalizedIntent dictionary
        
        Returns:
            ValidationResult: Validation result
        
        Validates: Requirements 6.1, 6.2, 6.3, 6.8, 6.9
        """
        errors = []
        
        # Validate required fields
        required_fields = [
            "city",
            "time_window",
            "max_travel_minutes",
            "party_size",
            "budget_level"
        ]
        
        for field in required_fields:
            if field not in intent or intent[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate city field
        if "city" in intent:
            city = intent["city"]
            if not isinstance(city, str):
                errors.append(f"city must be a string, got {type(city).__name__}")
            elif len(city) == 0:
                errors.append("city cannot be empty")
            elif len(city) > cls.MAX_CITY_LENGTH:
                errors.append(
                    f"city name too long (max {cls.MAX_CITY_LENGTH} characters), "
                    f"got {len(city)}"
                )
        
        # Validate time_window field
        if "time_window" in intent:
            time_window = intent["time_window"]
            if not isinstance(time_window, dict):
                errors.append(
                    f"time_window must be a dict, got {type(time_window).__name__}"
                )
            else:
                # Validate time_window sub-fields
                if "day" not in time_window:
                    errors.append("time_window.day is required")
                elif time_window["day"] not in cls.VALID_DAYS:
                    errors.append(
                        f"time_window.day must be one of {cls.VALID_DAYS}, "
                        f"got {time_window['day']}"
                    )
                
                if "start_local" not in time_window:
                    errors.append("time_window.start_local is required")
                elif not cls._is_valid_time(time_window["start_local"]):
                    errors.append(
                        f"time_window.start_local must be in HH:MM format, "
                        f"got {time_window['start_local']}"
                    )
                
                if "end_local" not in time_window:
                    errors.append("time_window.end_local is required")
                elif not cls._is_valid_time(time_window["end_local"]):
                    errors.append(
                        f"time_window.end_local must be in HH:MM format, "
                        f"got {time_window['end_local']}"
                    )
        
        # Validate max_travel_minutes field
        if "max_travel_minutes" in intent:
            max_travel = intent["max_travel_minutes"]
            if not isinstance(max_travel, (int, float)):
                errors.append(
                    f"max_travel_minutes must be a number, "
                    f"got {type(max_travel).__name__}"
                )
            elif not (5 <= max_travel <= 120):
                errors.append(
                    f"max_travel_minutes must be between 5 and 120, got {max_travel}"
                )
        
        # Validate party_size field
        if "party_size" in intent:
            party_size = intent["party_size"]
            if not isinstance(party_size, int):
                errors.append(
                    f"party_size must be an integer, got {type(party_size).__name__}"
                )
            elif not (1 <= party_size <= 50):
                errors.append(
                    f"party_size must be between 1 and 50, got {party_size}"
                )
        
        # Validate budget_level field
        if "budget_level" in intent:
            budget_level = intent["budget_level"]
            if not isinstance(budget_level, str):
                errors.append(
                    f"budget_level must be a string, got {type(budget_level).__name__}"
                )
            elif budget_level not in cls.VALID_BUDGET_LEVELS:
                errors.append(
                    f"budget_level must be one of {cls.VALID_BUDGET_LEVELS}, "
                    f"got {budget_level}"
                )
        
        # Validate optional fields
        if "dietary_restrictions" in intent:
            dietary = intent["dietary_restrictions"]
            if not isinstance(dietary, list):
                errors.append(
                    f"dietary_restrictions must be a list, "
                    f"got {type(dietary).__name__}"
                )
            elif any(not isinstance(item, str) for item in dietary):
                errors.append("dietary_restrictions must contain only strings")
        
        if "ambiance_preferences" in intent:
            ambiance = intent["ambiance_preferences"]
            if not isinstance(ambiance, list):
                errors.append(
                    f"ambiance_preferences must be a list, "
                    f"got {type(ambiance).__name__}"
                )
            elif any(not isinstance(item, str) for item in ambiance):
                errors.append("ambiance_preferences must contain only strings")
        
        return ValidationResult(valid=len(errors) == 0, errors=errors)
    
    @classmethod
    def validate_executable_mcp(cls, executable: Dict[str, Any]) -> ValidationResult:
        """Validate ExecutableMCP data structure
        
        Validation rules:
        - Required fields: tool_calls
        - tool_calls must be a list and cannot be empty
        - Each tool_call must contain tool and args
        
        Args:
            executable: ExecutableMCP dictionary
        
        Returns:
            ValidationResult: Validation result
        
        Validates: Requirement 6.4
        """
        errors = []
        
        # Validate required fields
        if "tool_calls" not in executable:
            errors.append("Missing required field: tool_calls")
            return ValidationResult(valid=False, errors=errors)
        
        tool_calls = executable["tool_calls"]
        
        # Validate tool_calls is a list
        if not isinstance(tool_calls, list):
            errors.append(
                f"tool_calls must be a list, got {type(tool_calls).__name__}"
            )
            return ValidationResult(valid=False, errors=errors)
        
        # Validate tool_calls is not empty
        if len(tool_calls) == 0:
            errors.append("tool_calls cannot be empty")
            return ValidationResult(valid=False, errors=errors)
        
        # Validate each tool_call
        for i, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                errors.append(
                    f"tool_calls[{i}] must be a dict, "
                    f"got {type(tool_call).__name__}"
                )
                continue
            
            # Validate tool field
            if "tool" not in tool_call:
                errors.append(f"tool_calls[{i}].tool is required")
            elif not isinstance(tool_call["tool"], str):
                errors.append(
                    f"tool_calls[{i}].tool must be a string, "
                    f"got {type(tool_call['tool']).__name__}"
                )
            
            # Validate args field
            if "args" not in tool_call:
                errors.append(f"tool_calls[{i}].args is required")
            elif not isinstance(tool_call["args"], dict):
                errors.append(
                    f"tool_calls[{i}].args must be a dict, "
                    f"got {type(tool_call['args']).__name__}"
                )
        
        return ValidationResult(valid=len(errors) == 0, errors=errors)
    
    @classmethod
    def validate_candidate_venue(cls, venue: Dict[str, Any]) -> ValidationResult:
        """Validate CandidateVenue data structure
        
        Validation rules:
        - Required fields: venue_id, name, address
        - Optional fields: rating (0-5), price_level (0-4)
        
        Args:
            venue: CandidateVenue dictionary
        
        Returns:
            ValidationResult: Validation result
        
        Validates: Requirement 6.6
        """
        errors = []
        
        # Validate required fields
        required_fields = ["venue_id", "name", "address"]
        for field in required_fields:
            if field not in venue or venue[field] is None:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(venue[field], str):
                errors.append(
                    f"{field} must be a string, got {type(venue[field]).__name__}"
                )
            elif len(venue[field]) == 0:
                errors.append(f"{field} cannot be empty")
        
        # Validate optional fields
        if "rating" in venue and venue["rating"] is not None:
            rating = venue["rating"]
            if not isinstance(rating, (int, float)):
                errors.append(
                    f"rating must be a number, got {type(rating).__name__}"
                )
            elif not (0 <= rating <= 5):
                errors.append(f"rating must be between 0 and 5, got {rating}")
        
        if "price_level" in venue and venue["price_level"] is not None:
            price_level = venue["price_level"]
            if not isinstance(price_level, int):
                errors.append(
                    f"price_level must be an integer, got {type(price_level).__name__}"
                )
            elif not (0 <= price_level <= 4):
                errors.append(
                    f"price_level must be between 0 and 4, got {price_level}"
                )
        
        if "user_ratings_total" in venue and venue["user_ratings_total"] is not None:
            user_ratings = venue["user_ratings_total"]
            if not isinstance(user_ratings, int):
                errors.append(
                    f"user_ratings_total must be an integer, got {type(user_ratings).__name__}"
                )
            elif user_ratings < 0:
                errors.append(
                    f"user_ratings_total must be non-negative, got {user_ratings}"
                )
        
        return ValidationResult(valid=len(errors) == 0, errors=errors)
    
    @classmethod
    def sanitize_user_input(cls, text: str) -> str:
        """Sanitize user input, remove malicious content
        
        Sanitization rules:
        - Remove HTML tags
        - Remove JavaScript code
        - Remove SQL injection attempts
        - Remove path traversal attempts
        - Limit input length
        
        Args:
            text: User input text
        
        Returns:
            Sanitized text
        
        Validates: Requirements 6.7, 6.8
        """
        if not isinstance(text, str):
            return ""
        
        # Limit input length
        if len(text) > cls.MAX_INPUT_LENGTH:
            text = text[:cls.MAX_INPUT_LENGTH]
        
        # Remove malicious content
        for pattern in cls.MALICIOUS_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        return text
    
    @classmethod
    def detect_malicious_content(cls, text: str) -> ValidationResult:
        """Detect malicious content
        
        Detection rules:
        - HTML tags (XSS)
        - JavaScript code (XSS)
        - SQL injection attempts
        - Path traversal attempts
        
        Args:
            text: Text to detect
        
        Returns:
            ValidationResult: Detection result
        
        Validates: Requirement 6.7
        """
        errors = []
        
        if not isinstance(text, str):
            return ValidationResult(valid=True, errors=[])
        
        # Detect malicious patterns
        for pattern in cls.MALICIOUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                errors.append(f"Malicious content detected: pattern '{pattern}'")
        
        return ValidationResult(valid=len(errors) == 0, errors=errors)
    
    @classmethod
    def validate_input_length(cls, text: str, max_length: Optional[int] = None) -> ValidationResult:
        """Validate input length
        
        Args:
            text: Text to validate
            max_length: Maximum length (optional, uses MAX_INPUT_LENGTH by default)
        
        Returns:
            ValidationResult: Validation result
        
        Validates: Requirement 6.8
        """
        errors = []
        
        if not isinstance(text, str):
            errors.append(f"Input must be a string, got {type(text).__name__}")
            return ValidationResult(valid=False, errors=errors)
        
        max_len = max_length or cls.MAX_INPUT_LENGTH
        
        if len(text) > max_len:
            errors.append(
                f"Input too long (max {max_len} characters), got {len(text)}"
            )
        
        return ValidationResult(valid=len(errors) == 0, errors=errors)
    
    @staticmethod
    def _is_valid_time(time_str: str) -> bool:
        """Validate time format (HH:MM)
        
        Args:
            time_str: Time string
        
        Returns:
            Whether valid
        """
        if not isinstance(time_str, str):
            return False
        
        # Match HH:MM format
        pattern = r"^([0-1][0-9]|2[0-3]):[0-5][0-9]$"
        return bool(re.match(pattern, time_str))
