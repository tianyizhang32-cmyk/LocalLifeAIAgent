"""
结构化日志模块

提供结构化日志记录，支持：
- JSON 格式日志输出
- 请求 ID 追踪
- 敏感信息脱敏（API Key、用户数据）
- 日志级别过滤
- 日志轮转

验证需求：3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.10
"""

import json
import logging
import re
import sys
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional
from pathlib import Path


class StructuredLogger:
    """结构化日志记录器
    
    提供结构化日志记录能力，支持 JSON 格式输出、请求追踪和敏感信息脱敏。
    
    Attributes:
        name: 日志记录器名称
        request_id: 当前请求 ID（用于追踪完整请求链路）
        logger: 底层 Python logger 实例
    """
    
    # 敏感字段名称模式（用于自动识别需要脱敏的字段）
    SENSITIVE_FIELD_PATTERNS = [
        r".*api[_-]?key.*",
        r".*password.*",
        r".*secret.*",
        r".*token.*",
        r".*auth.*",
    ]
    
    def __init__(
        self,
        name: str,
        log_level: str = "INFO",
        log_format: str = "json",
        log_file: Optional[str] = None,
        max_bytes: int = 10485760,  # 10MB
        backup_count: int = 5
    ):
        """初始化结构化日志记录器
        
        Args:
            name: 日志记录器名称
            log_level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
            log_format: 日志格式（json 或 text）
            log_file: 日志文件路径（可选，不指定则输出到 stdout）
            max_bytes: 日志文件最大大小（字节）
            backup_count: 保留的备份文件数量
        """
        self.name = name
        self.log_format = log_format
        self.request_id: Optional[str] = None
        
        # 创建 logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除现有 handlers（避免重复）
        self.logger.handlers.clear()
        
        # 添加 handler
        if log_file:
            # 文件 handler（带轮转）
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        else:
            # 控制台 handler
            handler = logging.StreamHandler(sys.stdout)
        
        # 设置 formatter
        if log_format == "json":
            handler.setFormatter(self._JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        self.logger.addHandler(handler)
    
    def set_request_id(self, request_id: str):
        """设置当前请求 ID
        
        Args:
            request_id: 请求 ID（用于追踪完整请求链路）
        """
        self.request_id = request_id
    
    def log(self, level: str, message: str, **kwargs):
        """记录日志
        
        Args:
            level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
            message: 日志消息
            **kwargs: 额外的上下文信息
        """
        # 添加请求 ID
        if self.request_id:
            kwargs["request_id"] = self.request_id
        
        # 脱敏敏感信息
        kwargs = self.sanitize(kwargs)
        
        # 记录日志
        log_method = getattr(self.logger, level.lower())
        log_method(message, extra={"context": kwargs})
    
    def debug(self, message: str, **kwargs):
        """记录 DEBUG 级别日志"""
        self.log("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """记录 INFO 级别日志"""
        self.log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录 WARNING 级别日志"""
        self.log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """记录 ERROR 级别日志"""
        self.log("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """记录 CRITICAL 级别日志"""
        self.log("CRITICAL", message, **kwargs)
    
    def log_api_call(
        self,
        api: str,
        method: str,
        duration: float,
        status: int,
        **kwargs
    ):
        """记录 API 调用
        
        Args:
            api: API 名称（如 "openai", "google_places"）
            method: HTTP 方法（如 "GET", "POST"）
            duration: 调用耗时（秒）
            status: HTTP 状态码
            **kwargs: 额外的上下文信息
        """
        self.info(
            "API call completed",
            api=api,
            method=method,
            duration_ms=round(duration * 1000, 2),
            status=status,
            **kwargs
        )
    
    def log_error(self, error: Exception, context: Optional[Dict] = None):
        """记录错误
        
        Args:
            error: 异常对象
            context: 错误上下文信息
        """
        context = context or {}
        context.update({
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        })
        
        self.error("Exception occurred", **context)
    
    def log_event(self, event_type: str, message: str, **kwargs):
        """记录系统事件
        
        Args:
            event_type: 事件类型（如 "startup", "shutdown", "config_loaded"）
            message: 事件消息
            **kwargs: 额外的上下文信息
        """
        self.info(message, event_type=event_type, **kwargs)
    
    def sanitize(self, data: Any) -> Any:
        """脱敏敏感信息
        
        对 API Key、密码等敏感信息进行脱敏处理。
        API Key 格式：前4位 + *** + 后4位
        
        Args:
            data: 需要脱敏的数据
        
        Returns:
            脱敏后的数据
        """
        if isinstance(data, dict):
            return {
                key: self._sanitize_value(key, value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self.sanitize(item) for item in data]
        else:
            return data
    
    def _sanitize_value(self, key: str, value: Any) -> Any:
        """脱敏单个值
        
        Args:
            key: 字段名
            value: 字段值
        
        Returns:
            脱敏后的值
        """
        # 检查是否是敏感字段
        if self._is_sensitive_field(key):
            if isinstance(value, str) and len(value) >= 8:
                # API Key 格式：前4位 + *** + 后4位
                return f"{value[:4]}***{value[-4:]}"
            elif isinstance(value, str):
                # 短字符串完全隐藏
                return "***"
            else:
                return "***"
        
        # 递归处理嵌套结构
        if isinstance(value, dict):
            return self.sanitize(value)
        elif isinstance(value, list):
            return [self.sanitize(item) for item in value]
        else:
            return value
    
    def _is_sensitive_field(self, field_name: str) -> bool:
        """判断字段是否敏感
        
        Args:
            field_name: 字段名
        
        Returns:
            是否敏感字段
        """
        field_lower = field_name.lower()
        return any(
            re.match(pattern, field_lower, re.IGNORECASE)
            for pattern in self.SENSITIVE_FIELD_PATTERNS
        )
    
    class _JsonFormatter(logging.Formatter):
        """JSON 格式化器"""
        
        def format(self, record: logging.LogRecord) -> str:
            """格式化日志记录为 JSON
            
            Args:
                record: 日志记录
            
            Returns:
                JSON 格式的日志字符串
            """
            log_data = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            
            # 添加上下文信息
            if hasattr(record, "context"):
                log_data.update(record.context)
            
            # 添加异常信息
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
            
            return json.dumps(log_data, ensure_ascii=False)


def create_logger(
    name: str,
    config: Optional[Dict] = None
) -> StructuredLogger:
    """创建结构化日志记录器（工厂函数）
    
    Args:
        name: 日志记录器名称
        config: 配置字典（可选）
    
    Returns:
        StructuredLogger 实例
    """
    config = config or {}
    
    return StructuredLogger(
        name=name,
        log_level=config.get("log_level", "INFO"),
        log_format=config.get("log_format", "json"),
        log_file=config.get("log_file"),
        max_bytes=config.get("max_bytes", 10485760),
        backup_count=config.get("backup_count", 5)
    )
