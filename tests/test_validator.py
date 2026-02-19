"""单元测试：数据验证模块

测试 DataValidator 类的各种验证和清洗功能。

验证需求：6.1, 6.2, 6.3, 6.4, 6.6, 6.7, 6.8, 6.9, 6.10
"""

import pytest
from local_lifestyle_agent.infrastructure.validator import (
    DataValidator,
    ValidationResult
)


class TestValidateNormalizedIntent:
    """测试 NormalizedIntent 验证"""
    
    def test_valid_intent(self):
        """测试有效的 intent"""
        intent = {
            "city": "Seattle",
            "time_window": {
                "day": "Sunday",
                "start_local": "14:00",
                "end_local": "17:00"
            },
            "max_travel_minutes": 30,
            "party_size": 2,
            "budget_level": "medium"
        }
        
        result = DataValidator.validate_normalized_intent(intent)
        
        assert result.valid is True
        assert len(result.errors) == 0
    
    def test_missing_required_field(self):
        """测试缺少必填字段"""
        intent = {
            "city": "Seattle",
            # 缺少 time_window
            "max_travel_minutes": 30,
            "party_size": 2,
            "budget_level": "medium"
        }
        
        result = DataValidator.validate_normalized_intent(intent)
        
        assert result.valid is False
        assert any("time_window" in error for error in result.errors)
    
    def test_invalid_city_type(self):
        """测试 city 类型错误"""
        intent = {
            "city": 123,  # 应该是字符串
            "time_window": {
                "day": "Sunday",
                "start_local": "14:00",
                "end_local": "17:00"
            },
            "max_travel_minutes": 30,
            "party_size": 2,
            "budget_level": "medium"
        }
        
        result = DataValidator.validate_normalized_intent(intent)
        
        assert result.valid is False
        assert any("city must be a string" in error for error in result.errors)
    
    def test_empty_city(self):
        """测试空 city"""
        intent = {
            "city": "",
            "time_window": {
                "day": "Sunday",
                "start_local": "14:00",
                "end_local": "17:00"
            },
            "max_travel_minutes": 30,
            "party_size": 2,
            "budget_level": "medium"
        }
        
        result = DataValidator.validate_normalized_intent(intent)
        
        assert result.valid is False
        assert any("city cannot be empty" in error for error in result.errors)
    
    def test_city_too_long(self):
        """测试 city 过长"""
        intent = {
            "city": "A" * 101,  # 超过 100 字符
            "time_window": {
                "day": "Sunday",
                "start_local": "14:00",
                "end_local": "17:00"
            },
            "max_travel_minutes": 30,
            "party_size": 2,
            "budget_level": "medium"
        }
        
        result = DataValidator.validate_normalized_intent(intent)
        
        assert result.valid is False
        assert any("city name too long" in error for error in result.errors)
    
    def test_invalid_max_travel_minutes_range(self):
        """测试 max_travel_minutes 超出范围"""
        intent = {
            "city": "Seattle",
            "time_window": {
                "day": "Sunday",
                "start_local": "14:00",
                "end_local": "17:00"
            },
            "max_travel_minutes": 150,  # 超过 120
            "party_size": 2,
            "budget_level": "medium"
        }
        
        result = DataValidator.validate_normalized_intent(intent)
        
        assert result.valid is False
        assert any("max_travel_minutes must be between 5 and 120" in error for error in result.errors)
    
    def test_invalid_max_travel_minutes_type(self):
        """测试 max_travel_minutes 类型错误"""
        intent = {
            "city": "Seattle",
            "time_window": {
                "day": "Sunday",
                "start_local": "14:00",
                "end_local": "17:00"
            },
            "max_travel_minutes": "30",  # 应该是数字
            "party_size": 2,
            "budget_level": "medium"
        }
        
        result = DataValidator.validate_normalized_intent(intent)
        
        assert result.valid is False
        assert any("max_travel_minutes must be a number" in error for error in result.errors)
    
    def test_invalid_party_size_range(self):
        """测试 party_size 超出范围"""
        intent = {
            "city": "Seattle",
            "time_window": {
                "day": "Sunday",
                "start_local": "14:00",
                "end_local": "17:00"
            },
            "max_travel_minutes": 30,
            "party_size": 100,  # 超过 50
            "budget_level": "medium"
        }
        
        result = DataValidator.validate_normalized_intent(intent)
        
        assert result.valid is False
        assert any("party_size must be between 1 and 50" in error for error in result.errors)
    
    def test_invalid_budget_level(self):
        """测试无效的 budget_level"""
        intent = {
            "city": "Seattle",
            "time_window": {
                "day": "Sunday",
                "start_local": "14:00",
                "end_local": "17:00"
            },
            "max_travel_minutes": 30,
            "party_size": 2,
            "budget_level": "expensive"  # 无效值
        }
        
        result = DataValidator.validate_normalized_intent(intent)
        
        assert result.valid is False
        assert any("budget_level must be one of" in error for error in result.errors)
    
    def test_invalid_time_window_type(self):
        """测试 time_window 类型错误"""
        intent = {
            "city": "Seattle",
            "time_window": "Sunday 14:00-17:00",  # 应该是字典
            "max_travel_minutes": 30,
            "party_size": 2,
            "budget_level": "medium"
        }
        
        result = DataValidator.validate_normalized_intent(intent)
        
        assert result.valid is False
        assert any("time_window must be a dict" in error for error in result.errors)
    
    def test_missing_time_window_day(self):
        """测试缺少 time_window.day"""
        intent = {
            "city": "Seattle",
            "time_window": {
                # 缺少 day
                "start_local": "14:00",
                "end_local": "17:00"
            },
            "max_travel_minutes": 30,
            "party_size": 2,
            "budget_level": "medium"
        }
        
        result = DataValidator.validate_normalized_intent(intent)
        
        assert result.valid is False
        assert any("time_window.day is required" in error for error in result.errors)
    
    def test_invalid_time_format(self):
        """测试无效的时间格式"""
        intent = {
            "city": "Seattle",
            "time_window": {
                "day": "Sunday",
                "start_local": "2pm",  # 无效格式
                "end_local": "17:00"
            },
            "max_travel_minutes": 30,
            "party_size": 2,
            "budget_level": "medium"
        }
        
        result = DataValidator.validate_normalized_intent(intent)
        
        assert result.valid is False
        assert any("start_local must be in HH:MM format" in error for error in result.errors)
    
    def test_valid_optional_fields(self):
        """测试有效的可选字段"""
        intent = {
            "city": "Seattle",
            "time_window": {
                "day": "Sunday",
                "start_local": "14:00",
                "end_local": "17:00"
            },
            "max_travel_minutes": 30,
            "party_size": 2,
            "budget_level": "medium",
            "dietary_restrictions": ["vegetarian", "gluten-free"],
            "ambiance_preferences": ["quiet", "cozy"]
        }
        
        result = DataValidator.validate_normalized_intent(intent)
        
        assert result.valid is True
        assert len(result.errors) == 0
    
    def test_invalid_dietary_restrictions_type(self):
        """测试 dietary_restrictions 类型错误"""
        intent = {
            "city": "Seattle",
            "time_window": {
                "day": "Sunday",
                "start_local": "14:00",
                "end_local": "17:00"
            },
            "max_travel_minutes": 30,
            "party_size": 2,
            "budget_level": "medium",
            "dietary_restrictions": "vegetarian"  # 应该是列表
        }
        
        result = DataValidator.validate_normalized_intent(intent)
        
        assert result.valid is False
        assert any("dietary_restrictions must be a list" in error for error in result.errors)


class TestValidateExecutableMCP:
    """测试 ExecutableMCP 验证"""
    
    def test_valid_executable(self):
        """测试有效的 executable"""
        executable = {
            "tool_calls": [
                {
                    "tool": "google_places_textsearch",
                    "args": {"query": "afternoon tea Seattle"}
                }
            ]
        }
        
        result = DataValidator.validate_executable_mcp(executable)
        
        assert result.valid is True
        assert len(result.errors) == 0
    
    def test_missing_tool_calls(self):
        """测试缺少 tool_calls"""
        executable = {}
        
        result = DataValidator.validate_executable_mcp(executable)
        
        assert result.valid is False
        assert any("tool_calls" in error for error in result.errors)
    
    def test_invalid_tool_calls_type(self):
        """测试 tool_calls 类型错误"""
        executable = {
            "tool_calls": "google_places_textsearch"  # 应该是列表
        }
        
        result = DataValidator.validate_executable_mcp(executable)
        
        assert result.valid is False
        assert any("tool_calls must be a list" in error for error in result.errors)
    
    def test_missing_tool_field(self):
        """测试缺少 tool 字段"""
        executable = {
            "tool_calls": [
                {
                    # 缺少 tool
                    "args": {"query": "afternoon tea"}
                }
            ]
        }
        
        result = DataValidator.validate_executable_mcp(executable)
        
        assert result.valid is False
        assert any("tool_calls[0].tool is required" in error for error in result.errors)
    
    def test_missing_args_field(self):
        """测试缺少 args 字段"""
        executable = {
            "tool_calls": [
                {
                    "tool": "google_places_textsearch"
                    # 缺少 args
                }
            ]
        }
        
        result = DataValidator.validate_executable_mcp(executable)
        
        assert result.valid is False
        assert any("tool_calls[0].args is required" in error for error in result.errors)


class TestValidateCandidateVenue:
    """测试 CandidateVenue 验证"""
    
    def test_valid_venue(self):
        """测试有效的 venue"""
        venue = {
            "venue_id": "venue123",
            "place_id": "ChIJ123",
            "name": "Tea House",
            "address": "123 Main St, Seattle, WA",
            "formatted_address": "123 Main St, Seattle, WA",
            "rating": 4.5,
            "price_level": 2
        }
        
        result = DataValidator.validate_candidate_venue(venue)
        
        assert result.valid is True
        assert len(result.errors) == 0
    
    def test_missing_required_field(self):
        """测试缺少必填字段"""
        venue = {
            "place_id": "ChIJ123",
            # 缺少 name
            "formatted_address": "123 Main St"
        }
        
        result = DataValidator.validate_candidate_venue(venue)
        
        assert result.valid is False
        assert any("name" in error for error in result.errors)
    
    def test_invalid_rating_range(self):
        """测试 rating 超出范围"""
        venue = {
            "place_id": "ChIJ123",
            "name": "Tea House",
            "formatted_address": "123 Main St",
            "rating": 6.0  # 超过 5
        }
        
        result = DataValidator.validate_candidate_venue(venue)
        
        assert result.valid is False
        assert any("rating must be between 0 and 5" in error for error in result.errors)
    
    def test_invalid_price_level_range(self):
        """测试 price_level 超出范围"""
        venue = {
            "place_id": "ChIJ123",
            "name": "Tea House",
            "formatted_address": "123 Main St",
            "price_level": 5  # 超过 4
        }
        
        result = DataValidator.validate_candidate_venue(venue)
        
        assert result.valid is False
        assert any("price_level must be between 0 and 4" in error for error in result.errors)


class TestSanitizeUserInput:
    """测试用户输入清洗"""
    
    def test_clean_input(self):
        """测试干净的输入"""
        text = "Find afternoon tea in Seattle"
        
        result = DataValidator.sanitize_user_input(text)
        
        assert result == text
    
    def test_remove_script_tags(self):
        """测试移除 script 标签"""
        text = "Find tea <script>alert('xss')</script> in Seattle"
        
        result = DataValidator.sanitize_user_input(text)
        
        assert "<script>" not in result
        assert "alert" not in result
    
    def test_remove_javascript_protocol(self):
        """测试移除 javascript 协议"""
        text = "Find tea javascript:alert('xss') in Seattle"
        
        result = DataValidator.sanitize_user_input(text)
        
        assert "javascript:" not in result
    
    def test_remove_sql_injection(self):
        """测试移除 SQL 注入"""
        text = "Find tea'; DROP TABLE users; -- in Seattle"
        
        result = DataValidator.sanitize_user_input(text)
        
        assert "DROP TABLE" not in result
    
    def test_remove_path_traversal(self):
        """测试移除路径遍历"""
        text = "Find tea ../../etc/passwd in Seattle"
        
        result = DataValidator.sanitize_user_input(text)
        
        assert "../" not in result
    
    def test_limit_input_length(self):
        """测试限制输入长度"""
        text = "A" * 2000  # 超过 MAX_INPUT_LENGTH (1000)
        
        result = DataValidator.sanitize_user_input(text)
        
        assert len(result) <= DataValidator.MAX_INPUT_LENGTH
    
    def test_normalize_whitespace(self):
        """测试规范化空白字符"""
        text = "Find   afternoon    tea   in   Seattle"
        
        result = DataValidator.sanitize_user_input(text)
        
        assert result == "Find afternoon tea in Seattle"
    
    def test_non_string_input(self):
        """测试非字符串输入"""
        result = DataValidator.sanitize_user_input(123)
        
        assert result == ""


class TestDetectMaliciousContent:
    """测试恶意内容检测"""
    
    def test_clean_content(self):
        """测试干净的内容"""
        text = "Find afternoon tea in Seattle"
        
        result = DataValidator.detect_malicious_content(text)
        
        assert result.valid is True
        assert len(result.errors) == 0
    
    def test_detect_xss_script(self):
        """测试检测 XSS script"""
        text = "<script>alert('xss')</script>"
        
        result = DataValidator.detect_malicious_content(text)
        
        assert result.valid is False
        assert len(result.errors) > 0
    
    def test_detect_sql_injection(self):
        """测试检测 SQL 注入"""
        text = "'; DROP TABLE users; --"
        
        result = DataValidator.detect_malicious_content(text)
        
        assert result.valid is False
        assert len(result.errors) > 0
    
    def test_detect_path_traversal(self):
        """测试检测路径遍历"""
        text = "../../etc/passwd"
        
        result = DataValidator.detect_malicious_content(text)
        
        assert result.valid is False
        assert len(result.errors) > 0


class TestValidateInputLength:
    """测试输入长度验证"""
    
    def test_valid_length(self):
        """测试有效长度"""
        text = "Find afternoon tea in Seattle"
        
        result = DataValidator.validate_input_length(text)
        
        assert result.valid is True
        assert len(result.errors) == 0
    
    def test_exceed_default_max_length(self):
        """测试超过默认最大长度"""
        text = "A" * 1001  # 超过 MAX_INPUT_LENGTH (1000)
        
        result = DataValidator.validate_input_length(text)
        
        assert result.valid is False
        assert any("Input too long" in error for error in result.errors)
    
    def test_custom_max_length(self):
        """测试自定义最大长度"""
        text = "A" * 101
        
        result = DataValidator.validate_input_length(text, max_length=100)
        
        assert result.valid is False
        assert any("max 100 characters" in error for error in result.errors)
    
    def test_non_string_input(self):
        """测试非字符串输入"""
        result = DataValidator.validate_input_length(123)
        
        assert result.valid is False
        assert any("Input must be a string" in error for error in result.errors)


class TestIsValidTime:
    """测试时间格式验证"""
    
    def test_valid_time(self):
        """测试有效时间"""
        assert DataValidator._is_valid_time("14:00") is True
        assert DataValidator._is_valid_time("00:00") is True
        assert DataValidator._is_valid_time("23:59") is True
    
    def test_invalid_time_format(self):
        """测试无效时间格式"""
        assert DataValidator._is_valid_time("2pm") is False
        assert DataValidator._is_valid_time("14:00:00") is False
        assert DataValidator._is_valid_time("25:00") is False
        assert DataValidator._is_valid_time("14:60") is False
    
    def test_non_string_input(self):
        """测试非字符串输入"""
        assert DataValidator._is_valid_time(1400) is False
        assert DataValidator._is_valid_time(None) is False
