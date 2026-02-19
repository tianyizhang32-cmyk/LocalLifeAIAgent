"""
测试结构化日志模块

验证需求：3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.10
"""

import json
import logging
import tempfile
from pathlib import Path

import pytest

from local_lifestyle_agent.infrastructure.logger import (
    StructuredLogger,
    create_logger
)


class TestStructuredLogger:
    """测试 StructuredLogger 类"""
    
    def test_logger_initialization(self):
        """测试日志记录器初始化"""
        logger = StructuredLogger("test_logger")
        
        assert logger.name == "test_logger"
        assert logger.log_format == "json"
        assert logger.request_id is None
        assert logger.logger.level == logging.INFO
    
    def test_logger_with_custom_level(self):
        """测试自定义日志级别"""
        logger = StructuredLogger("test_logger", log_level="DEBUG")
        
        assert logger.logger.level == logging.DEBUG
    
    def test_set_request_id(self):
        """测试设置请求 ID"""
        logger = StructuredLogger("test_logger")
        
        logger.set_request_id("req_123")
        
        assert logger.request_id == "req_123"
    
    def test_log_with_request_id(self, caplog):
        """测试日志包含请求 ID"""
        logger = StructuredLogger("test_logger", log_format="text")
        logger.set_request_id("req_456")
        
        with caplog.at_level(logging.INFO):
            logger.info("Test message", key="value")
        
        # 验证日志记录
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert hasattr(record, "context")
        assert record.context["request_id"] == "req_456"
    
    def test_log_levels(self, caplog):
        """测试不同日志级别"""
        logger = StructuredLogger("test_logger", log_level="DEBUG", log_format="text")
        
        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
        
        assert len(caplog.records) == 5
        assert caplog.records[0].levelname == "DEBUG"
        assert caplog.records[1].levelname == "INFO"
        assert caplog.records[2].levelname == "WARNING"
        assert caplog.records[3].levelname == "ERROR"
        assert caplog.records[4].levelname == "CRITICAL"
    
    def test_log_api_call(self, caplog):
        """测试记录 API 调用"""
        logger = StructuredLogger("test_logger", log_format="text")
        
        with caplog.at_level(logging.INFO):
            logger.log_api_call(
                api="openai",
                method="POST",
                duration=1.234,
                status=200
            )
        
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.message == "API call completed"
        assert record.context["api"] == "openai"
        assert record.context["method"] == "POST"
        assert record.context["duration_ms"] == 1234.0
        assert record.context["status"] == 200
    
    def test_log_error(self, caplog):
        """测试记录错误"""
        logger = StructuredLogger("test_logger", log_format="text")
        
        try:
            raise ValueError("Test error")
        except ValueError as e:
            with caplog.at_level(logging.ERROR):
                logger.log_error(e, {"context_key": "context_value"})
        
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelname == "ERROR"
        assert record.context["error_type"] == "ValueError"
        assert record.context["error_message"] == "Test error"
        assert "traceback" in record.context
        assert record.context["context_key"] == "context_value"
    
    def test_log_event(self, caplog):
        """测试记录系统事件"""
        logger = StructuredLogger("test_logger", log_format="text")
        
        with caplog.at_level(logging.INFO):
            logger.log_event("startup", "System started", version="1.0.0")
        
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.context["event_type"] == "startup"
        assert record.context["version"] == "1.0.0"
    
    def test_sanitize_api_key(self):
        """测试 API Key 脱敏（需求 3.10, 10.1）"""
        logger = StructuredLogger("test_logger")
        
        data = {
            "api_key": "sk-1234567890abcdef",
            "openai_api_key": "sk-abcdefghijklmnop",
            "google_api_key": "AIza1234567890abcdef"
        }
        
        sanitized = logger.sanitize(data)
        
        # 验证脱敏格式：前4位 + *** + 后4位
        assert sanitized["api_key"] == "sk-1***cdef"
        assert sanitized["openai_api_key"] == "sk-a***mnop"
        assert sanitized["google_api_key"] == "AIza***cdef"
    
    def test_sanitize_short_api_key(self):
        """测试短 API Key 脱敏"""
        logger = StructuredLogger("test_logger")
        
        data = {"api_key": "short"}
        sanitized = logger.sanitize(data)
        
        # 短字符串完全隐藏
        assert sanitized["api_key"] == "***"
    
    def test_sanitize_password(self):
        """测试密码脱敏"""
        logger = StructuredLogger("test_logger")
        
        data = {
            "password": "mypassword123",
            "user_password": "secret456"
        }
        
        sanitized = logger.sanitize(data)
        
        # "mypassword123" 有13个字符，前4后4: "mypa***d123"
        assert sanitized["password"] == "mypa***d123"
        # "secret456" 有9个字符，前4后4: "secr***t456"
        assert sanitized["user_password"] == "secr***t456"
    
    def test_sanitize_nested_dict(self):
        """测试嵌套字典脱敏"""
        logger = StructuredLogger("test_logger")
        
        data = {
            "user": {
                "name": "John",
                "api_key": "sk-1234567890abcdef"
            },
            "config": {
                "timeout": 30,
                "secret_token": "token123456789"
            }
        }
        
        sanitized = logger.sanitize(data)
        
        assert sanitized["user"]["name"] == "John"
        assert sanitized["user"]["api_key"] == "sk-1***cdef"
        assert sanitized["config"]["timeout"] == 30
        assert sanitized["config"]["secret_token"] == "toke***6789"
    
    def test_sanitize_list(self):
        """测试列表脱敏"""
        logger = StructuredLogger("test_logger")
        
        data = {
            "keys": [
                {"api_key": "sk-1234567890abcdef"},
                {"api_key": "sk-abcdefghijklmnop"}
            ]
        }
        
        sanitized = logger.sanitize(data)
        
        assert sanitized["keys"][0]["api_key"] == "sk-1***cdef"
        assert sanitized["keys"][1]["api_key"] == "sk-a***mnop"
    
    def test_sanitize_non_sensitive_data(self):
        """测试非敏感数据不被脱敏"""
        logger = StructuredLogger("test_logger")
        
        data = {
            "name": "John Doe",
            "age": 30,
            "city": "Seattle",
            "preferences": ["tea", "coffee"]
        }
        
        sanitized = logger.sanitize(data)
        
        # 非敏感数据应保持不变
        assert sanitized == data
    
    def test_is_sensitive_field(self):
        """测试敏感字段识别"""
        logger = StructuredLogger("test_logger")
        
        # 敏感字段
        assert logger._is_sensitive_field("api_key")
        assert logger._is_sensitive_field("API_KEY")
        assert logger._is_sensitive_field("openai_api_key")
        assert logger._is_sensitive_field("password")
        assert logger._is_sensitive_field("user_password")
        assert logger._is_sensitive_field("secret")
        assert logger._is_sensitive_field("secret_token")
        assert logger._is_sensitive_field("auth_token")
        
        # 非敏感字段
        assert not logger._is_sensitive_field("name")
        assert not logger._is_sensitive_field("age")
        assert not logger._is_sensitive_field("city")
    
    def test_json_format_output(self, caplog):
        """测试 JSON 格式输出（需求 3.4）"""
        logger = StructuredLogger("test_logger", log_format="json")
        logger.set_request_id("req_789")
        
        with caplog.at_level(logging.INFO):
            logger.info("Test message", key="value")
        
        # 验证 JSON 格式
        assert len(caplog.records) == 1
        record = caplog.records[0]
        
        # 格式化输出应该是有效的 JSON
        formatted = logger.logger.handlers[0].formatter.format(record)
        log_data = json.loads(formatted)
        
        assert "timestamp" in log_data
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_logger"
        assert log_data["message"] == "Test message"
        assert log_data["request_id"] == "req_789"
        assert log_data["key"] == "value"
    
    def test_log_file_creation(self):
        """测试日志文件创建"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = StructuredLogger(
                "test_logger",
                log_file=str(log_file)
            )
            
            logger.info("Test message")
            
            # 验证日志文件已创建
            assert log_file.exists()
            
            # 验证日志内容
            content = log_file.read_text()
            assert "Test message" in content
    
    def test_log_rotation(self):
        """测试日志轮转（需求 3.7）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = StructuredLogger(
                "test_logger",
                log_file=str(log_file),
                max_bytes=100,  # 小文件大小触发轮转
                backup_count=2
            )
            
            # 写入大量日志触发轮转
            for i in range(50):
                logger.info(f"Test message {i}" * 10)
            
            # 验证备份文件已创建
            backup_files = list(Path(tmpdir).glob("test.log.*"))
            assert len(backup_files) > 0
    
    def test_log_level_filtering(self):
        """测试日志级别过滤（需求 3.6）"""
        logger = StructuredLogger("test_logger", log_level="WARNING")
        
        # 创建一个自定义 handler 来捕获日志
        from io import StringIO
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.logger.addHandler(handler)
        
        logger.debug("Debug message")  # 不应记录
        logger.info("Info message")    # 不应记录
        logger.warning("Warning message")  # 应记录
        logger.error("Error message")      # 应记录
        
        output = stream.getvalue()
        assert "Debug message" not in output
        assert "Info message" not in output
        assert "Warning message" in output
        assert "Error message" in output


class TestCreateLogger:
    """测试 create_logger 工厂函数"""
    
    def test_create_logger_default(self):
        """测试默认配置创建日志记录器"""
        logger = create_logger("test_logger")
        
        assert logger.name == "test_logger"
        assert logger.logger.level == logging.INFO
    
    def test_create_logger_with_config(self):
        """测试使用配置创建日志记录器"""
        config = {
            "log_level": "DEBUG",
            "log_format": "text",
            "max_bytes": 5242880,
            "backup_count": 3
        }
        
        logger = create_logger("test_logger", config)
        
        assert logger.logger.level == logging.DEBUG
        assert logger.log_format == "text"
    
    def test_create_logger_with_file(self):
        """测试创建文件日志记录器"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            config = {"log_file": str(log_file)}
            
            logger = create_logger("test_logger", config)
            logger.info("Test message")
            
            assert log_file.exists()


class TestLoggerIntegration:
    """集成测试"""
    
    def test_complete_logging_workflow(self):
        """测试完整的日志记录流程"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "integration.log"
            logger = StructuredLogger(
                "integration_test",
                log_level="DEBUG",
                log_format="json",
                log_file=str(log_file)
            )
            
            # 设置请求 ID
            logger.set_request_id("req_integration_001")
            
            # 记录各种类型的日志
            logger.debug("Debug information", step=1)
            logger.info("Processing started", user="test_user")
            
            # 记录 API 调用
            logger.log_api_call(
                api="openai",
                method="POST",
                duration=1.5,
                status=200,
                model="gpt-4"
            )
            
            # 记录带敏感信息的日志
            logger.info(
                "API configured",
                api_key="sk-1234567890abcdef",
                endpoint="https://api.openai.com"
            )
            
            # 记录错误
            try:
                raise ValueError("Test error")
            except ValueError as e:
                logger.log_error(e, {"operation": "test_operation"})
            
            # 记录系统事件
            logger.log_event("shutdown", "System shutting down", reason="test_complete")
            
            # 验证日志文件
            assert log_file.exists()
            content = log_file.read_text()
            
            # 验证所有日志都已写入
            assert "Debug information" in content
            assert "Processing started" in content
            assert "API call completed" in content
            assert "API configured" in content
            assert "Exception occurred" in content
            assert "System shutting down" in content
            
            # 验证请求 ID 存在
            assert "req_integration_001" in content
            
            # 验证 API Key 已脱敏
            assert "sk-1***cdef" in content
            assert "sk-1234567890abcdef" not in content
            
            # 验证 JSON 格式
            lines = content.strip().split("\n")
            for line in lines:
                log_data = json.loads(line)
                assert "timestamp" in log_data
                assert "level" in log_data
                assert "logger" in log_data
                assert "message" in log_data
