"""统一错误处理模块

本模块提供集中式错误处理、重试逻辑和结构化错误响应。

验证需求：1.1, 1.2, 1.4, 1.5, 1.9, 1.10
"""

import random
import time
from typing import Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    """错误代码枚举"""
    
    # API 错误
    API_TIMEOUT = "API_TIMEOUT"
    API_CONNECTION_ERROR = "API_CONNECTION_ERROR"
    API_RATE_LIMIT = "API_RATE_LIMIT"
    API_AUTHENTICATION_ERROR = "API_AUTHENTICATION_ERROR"
    API_INVALID_RESPONSE = "API_INVALID_RESPONSE"
    API_SERVER_ERROR = "API_SERVER_ERROR"
    
    # 验证错误
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    
    # 业务错误
    NO_CANDIDATES_FOUND = "NO_CANDIDATES_FOUND"
    EVALUATION_FAILED = "EVALUATION_FAILED"
    
    # 系统错误
    INTERNAL_ERROR = "INTERNAL_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"


class ErrorResponse(BaseModel):
    """结构化错误响应模型
    
    Attributes:
        error_code: 机器可读的错误代码
        error_message: 用户友好的错误消息
        details: 可选的详细错误信息（用于调试）
        retry_after: 可选的重试延迟（秒）
        request_id: 用于追踪的请求 ID
        timestamp: 错误时间戳
    """
    
    error_code: str = Field(..., description="错误代码")
    error_message: str = Field(..., description="用户友好的错误消息")
    details: Optional[Dict[str, Any]] = Field(None, description="详细错误信息")
    retry_after: Optional[int] = Field(None, description="重试延迟（秒）")
    request_id: str = Field(..., description="请求 ID")
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    )


class ErrorHandler:
    """统一错误处理器，支持重试逻辑和指数退避
    
    本类为系统中的所有异常提供集中式错误处理，包括错误分类、重试判断和延迟计算。
    
    重试策略：
        - 指数退避：delay = base_delay * (exponential_base ** attempt) + random_jitter
        - 最大重试次数：可配置（默认 3 次）
        - 可重试的错误：网络超时、5xx 错误、429 速率限制
        - 不可重试的错误：4xx 客户端错误（除 429）、认证失败
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: int = 2
    ):
        """初始化错误处理器
        
        Args:
            max_retries: 最大重试次数
            base_delay: 指数退避的基础延迟（秒）
            max_delay: 重试之间的最大延迟（秒）
            exponential_base: 指数计算的基数（默认：2）
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def handle_api_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        request_id: str = "unknown"
    ) -> ErrorResponse:
        """处理 API 相关错误
        
        根据异常类型和内容，将错误分类并生成结构化的错误响应。
        
        Args:
            error: 发生的异常
            context: 错误上下文（例如：{"api": "openai", "method": "POST"}）
            request_id: 用于追踪的请求 ID
            
        Returns:
            ErrorResponse: 结构化错误响应
            
        验证需求：1.1, 1.2
        """
        error_type = type(error).__name__
        error_str = str(error).lower()
        api_name = context.get("api", "unknown")
        
        # 超时错误
        if "timeout" in error_type.lower() or "timeout" in error_str:
            return ErrorResponse(
                error_code=ErrorCode.API_TIMEOUT,
                error_message=f"{api_name} API request timed out",
                details={
                    "error_type": error_type,
                    "message": str(error),
                    **context
                },
                retry_after=self.get_retry_delay(0),
                request_id=request_id
            )
        
        # 连接错误
        elif "connection" in error_type.lower() or "connection" in error_str:
            return ErrorResponse(
                error_code=ErrorCode.API_CONNECTION_ERROR,
                error_message=f"Failed to connect to {api_name} API",
                details={
                    "error_type": error_type,
                    "message": str(error),
                    **context
                },
                retry_after=self.get_retry_delay(0),
                request_id=request_id
            )
        
        # 速率限制错误（429）
        elif "429" in str(error) or "rate limit" in error_str:
            return ErrorResponse(
                error_code=ErrorCode.API_RATE_LIMIT,
                error_message=f"{api_name} API rate limit exceeded",
                details={
                    "error_type": error_type,
                    "message": str(error),
                    **context
                },
                retry_after=self.get_retry_delay(0),
                request_id=request_id
            )
        
        # 认证错误（401, 403）
        elif any(code in str(error) for code in ["401", "403"]) or "authentication" in error_str:
            return ErrorResponse(
                error_code=ErrorCode.API_AUTHENTICATION_ERROR,
                error_message=f"{api_name} API authentication failed",
                details={
                    "error_type": error_type,
                    "message": str(error),
                    **context
                },
                retry_after=None,  # 不重试认证错误
                request_id=request_id
            )
        
        # 服务器错误（5xx）
        elif any(str(error).startswith(f"{code}") for code in ["500", "502", "503", "504"]):
            return ErrorResponse(
                error_code=ErrorCode.API_SERVER_ERROR,
                error_message=f"{api_name} API server error",
                details={
                    "error_type": error_type,
                    "message": str(error),
                    **context
                },
                retry_after=self.get_retry_delay(0),
                request_id=request_id
            )
        
        # 其他未知错误
        else:
            return ErrorResponse(
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message=f"Unexpected error occurred with {api_name} API",
                details={
                    "error_type": error_type,
                    "message": str(error),
                    **context
                },
                retry_after=None,
                request_id=request_id
            )
    
    def handle_validation_error(
        self,
        error: Exception,
        request_id: str = "unknown"
    ) -> ErrorResponse:
        """处理验证错误
        
        Args:
            error: 验证异常
            request_id: 用于追踪的请求 ID
            
        Returns:
            ErrorResponse: 结构化错误响应
            
        验证需求：1.6
        """
        return ErrorResponse(
            error_code=ErrorCode.VALIDATION_ERROR,
            error_message="Data validation failed",
            details={
                "error_type": type(error).__name__,
                "message": str(error)
            },
            retry_after=None,  # 不重试验证错误
            request_id=request_id
        )
    
    def handle_timeout_error(
        self,
        error: Exception,
        request_id: str = "unknown"
    ) -> ErrorResponse:
        """处理超时错误
        
        Args:
            error: 超时异常
            request_id: 用于追踪的请求 ID
            
        Returns:
            ErrorResponse: 结构化错误响应
            
        验证需求：1.4
        """
        return ErrorResponse(
            error_code=ErrorCode.API_TIMEOUT,
            error_message="API request timed out",
            details={
                "error_type": type(error).__name__,
                "message": str(error)
            },
            retry_after=self.get_retry_delay(0),
            request_id=request_id
        )
    
    def handle_rate_limit_error(
        self,
        error: Exception,
        retry_after: Optional[int] = None,
        request_id: str = "unknown"
    ) -> ErrorResponse:
        """处理速率限制错误
        
        Args:
            error: 速率限制异常
            retry_after: 来自 API 响应的可选 retry-after 值
            request_id: 用于追踪的请求 ID
            
        Returns:
            ErrorResponse: 结构化错误响应
            
        验证需求：1.5
        """
        # 如果提供了 API 的 retry-after，使用它；否则计算指数退避
        delay = retry_after if retry_after is not None else self.get_retry_delay(0)
        
        return ErrorResponse(
            error_code=ErrorCode.API_RATE_LIMIT,
            error_message="API rate limit exceeded",
            details={
                "error_type": type(error).__name__,
                "message": str(error)
            },
            retry_after=delay,
            request_id=request_id
        )
    
    def should_retry(self, error: Exception) -> bool:
        """判断错误是否应该重试
        
        可重试的错误：
            - 网络超时
            - 连接错误
            - 5xx 服务器错误
            - 429 速率限制
        
        不可重试的错误：
            - 4xx 客户端错误（除 429）
            - 认证失败（401, 403）
        
        Args:
            error: 要评估的异常
            
        Returns:
            bool: 如果错误可重试返回 True，否则返回 False
            
        验证需求：1.4, 1.5
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # 可重试的错误模式
        retryable_patterns = [
            "timeout",
            "connection",
            "429",
            "rate limit",
            "500",
            "502",
            "503",
            "504"
        ]
        
        # 不可重试的错误模式
        non_retryable_patterns = [
            "401",
            "403",
            "authentication",
            "400",
            "404"
        ]
        
        # 首先检查不可重试的模式
        for pattern in non_retryable_patterns:
            if pattern in error_str or pattern in error_type:
                return False
        
        # 检查可重试的模式
        for pattern in retryable_patterns:
            if pattern in error_str or pattern in error_type:
                return True
        
        # 默认：不重试未知错误
        return False
    
    def get_retry_delay(self, attempt: int) -> int:
        """使用指数退避和抖动计算重试延迟
        
        公式：delay = base_delay * (exponential_base ** attempt) + random_jitter
        
        抖动范围：0 到指数延迟的 50%
        最大延迟：受 max_delay 限制
        
        Args:
            attempt: 当前重试尝试次数（从 0 开始）
            
        Returns:
            int: 下次重试前的延迟（秒）
            
        验证需求：1.4, 1.5
        """
        # 计算指数退避
        exponential_delay = self.base_delay * (self.exponential_base ** attempt)
        
        # 添加随机抖动（指数延迟的 0-50%）
        jitter = random.uniform(0, exponential_delay * 0.5)
        
        # 计算总延迟
        total_delay = exponential_delay + jitter
        
        # 限制在 max_delay
        capped_delay = min(total_delay, self.max_delay)
        
        return int(capped_delay)
