# -*- coding: utf-8 -*-
"""
自定义异常类

定义应用特定的异常类型。
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入模板库异常
from src.utils.exceptions import (
    BaseCustomException,
    ValidationException,
    AuthenticationException,
    AuthorizationException,
    RateLimitException
)


class ChatGPTBackendException(BaseCustomException):
    """ChatGPT后端基础异常类"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "CHATGPT_BACKEND_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, status_code, details)


class ConversationNotFoundException(ChatGPTBackendException):
    """对话未找到异常"""
    
    def __init__(
        self,
        conversation_id: Optional[int] = None,
        message: str = "对话不存在或无权限访问",
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        
        if conversation_id:
            details["conversation_id"] = conversation_id
        
        super().__init__(
            message=message,
            error_code="CONVERSATION_NOT_FOUND",
            status_code=404,
            details=details
        )


class MessageNotFoundException(ChatGPTBackendException):
    """消息未找到异常"""
    
    def __init__(
        self,
        message_id: Optional[int] = None,
        message: str = "消息不存在或无权限访问",
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        
        if message_id:
            details["message_id"] = message_id
        
        super().__init__(
            message=message,
            error_code="MESSAGE_NOT_FOUND",
            status_code=404,
            details=details
        )


class UserQuotaExceededException(ChatGPTBackendException):
    """用户配额超限异常"""
    
    def __init__(
        self,
        quota_type: str,
        current_usage: int,
        limit: int,
        reset_time: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if message is None:
            message = f"已超过{quota_type}配额限制 ({current_usage}/{limit})"
        
        if details is None:
            details = {}
        
        details.update({
            "quota_type": quota_type,
            "current_usage": current_usage,
            "limit": limit,
            "reset_time": reset_time
        })
        
        super().__init__(
            message=message,
            error_code="USER_QUOTA_EXCEEDED",
            status_code=429,
            details=details
        )


class OpenAIServiceException(ChatGPTBackendException):
    """OpenAI服务异常"""
    
    def __init__(
        self,
        message: str = "OpenAI服务错误",
        openai_error_code: Optional[str] = None,
        openai_error_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        
        if openai_error_code:
            details["openai_error_code"] = openai_error_code
        
        if openai_error_type:
            details["openai_error_type"] = openai_error_type
        
        super().__init__(
            message=message,
            error_code="OPENAI_SERVICE_ERROR",
            status_code=502,
            details=details
        )


class DatabaseConnectionException(ChatGPTBackendException):
    """数据库连接异常"""
    
    def __init__(
        self,
        message: str = "数据库连接失败",
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        
        if operation:
            details["operation"] = operation
        
        super().__init__(
            message=message,
            error_code="DATABASE_CONNECTION_ERROR",
            status_code=503,
            details=details
        )


class InvalidModelException(ChatGPTBackendException):
    """无效模型异常"""
    
    def __init__(
        self,
        model_name: str,
        available_models: Optional[list] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if message is None:
            message = f"不支持的模型: {model_name}"
        
        if details is None:
            details = {}
        
        details["model_name"] = model_name
        if available_models:
            details["available_models"] = available_models
        
        super().__init__(
            message=message,
            error_code="INVALID_MODEL",
            status_code=400,
            details=details
        )


class ConversationLimitExceededException(ChatGPTBackendException):
    """对话数量超限异常"""
    
    def __init__(
        self,
        current_count: int,
        limit: int,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if message is None:
            message = f"对话数量已达上限 ({current_count}/{limit})"
        
        if details is None:
            details = {}
        
        details.update({
            "current_count": current_count,
            "limit": limit
        })
        
        super().__init__(
            message=message,
            error_code="CONVERSATION_LIMIT_EXCEEDED",
            status_code=429,
            details=details
        )


class MessageTooLongException(ChatGPTBackendException):
    """消息过长异常"""
    
    def __init__(
        self,
        message_length: int,
        max_length: int,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if message is None:
            message = f"消息长度超过限制 ({message_length}/{max_length})"
        
        if details is None:
            details = {}
        
        details.update({
            "message_length": message_length,
            "max_length": max_length
        })
        
        super().__init__(
            message=message,
            error_code="MESSAGE_TOO_LONG",
            status_code=400,
            details=details
        )


class TokenLimitExceededException(ChatGPTBackendException):
    """Token数量超限异常"""
    
    def __init__(
        self,
        token_count: int,
        max_tokens: int,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if message is None:
            message = f"Token数量超过限制 ({token_count}/{max_tokens})"
        
        if details is None:
            details = {}
        
        details.update({
            "token_count": token_count,
            "max_tokens": max_tokens
        })
        
        super().__init__(
            message=message,
            error_code="TOKEN_LIMIT_EXCEEDED",
            status_code=400,
            details=details
        )


class ConversationArchivedException(ChatGPTBackendException):
    """对话已归档异常"""
    
    def __init__(
        self,
        conversation_id: int,
        message: str = "对话已归档，无法进行操作",
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        
        details["conversation_id"] = conversation_id
        
        super().__init__(
            message=message,
            error_code="CONVERSATION_ARCHIVED",
            status_code=403,
            details=details
        )


class UserInactiveException(ChatGPTBackendException):
    """用户未激活异常"""
    
    def __init__(
        self,
        user_id: int,
        status: str,
        message: str = "用户账户未激活或已被禁用",
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        
        details.update({
            "user_id": user_id,
            "status": status
        })
        
        super().__init__(
            message=message,
            error_code="USER_INACTIVE",
            status_code=403,
            details=details
        )


class FileUploadException(ChatGPTBackendException):
    """文件上传异常"""
    
    def __init__(
        self,
        filename: Optional[str] = None,
        file_size: Optional[int] = None,
        max_size: Optional[int] = None,
        message: str = "文件上传失败",
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        
        if filename:
            details["filename"] = filename
        if file_size:
            details["file_size"] = file_size
        if max_size:
            details["max_size"] = max_size
        
        super().__init__(
            message=message,
            error_code="FILE_UPLOAD_ERROR",
            status_code=400,
            details=details
        )


class WebSocketConnectionException(ChatGPTBackendException):
    """WebSocket连接异常"""
    
    def __init__(
        self,
        connection_id: Optional[str] = None,
        reason: Optional[str] = None,
        message: str = "WebSocket连接错误",
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        
        if connection_id:
            details["connection_id"] = connection_id
        if reason:
            details["reason"] = reason
        
        super().__init__(
            message=message,
            error_code="WEBSOCKET_CONNECTION_ERROR",
            status_code=400,
            details=details
        )


class ConfigurationException(ChatGPTBackendException):
    """配置异常"""
    
    def __init__(
        self,
        config_key: Optional[str] = None,
        message: str = "配置错误",
        details: Optional[Dict[str, Any]] = None
    ):
        if details is None:
            details = {}
        
        if config_key:
            details["config_key"] = config_key
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            details=details
        )


class ExternalServiceException(ChatGPTBackendException):
    """外部服务异常"""
    
    def __init__(
        self,
        service_name: str,
        service_error: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if message is None:
            message = f"{service_name}服务错误"
        
        if details is None:
            details = {}
        
        details["service_name"] = service_name
        if service_error:
            details["service_error"] = service_error
        
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            details=details
        )


class CacheException(ChatGPTBackendException):
    """缓存异常"""
    
    def __init__(
        self,
        operation: str,
        cache_key: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if message is None:
            message = f"缓存{operation}操作失败"
        
        if details is None:
            details = {}
        
        details["operation"] = operation
        if cache_key:
            details["cache_key"] = cache_key
        
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            status_code=500,
            details=details
        )


class BusinessLogicException(ChatGPTBackendException):
    """业务逻辑异常"""
    
    def __init__(
        self,
        business_rule: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if message is None:
            message = f"违反业务规则: {business_rule}"
        
        if details is None:
            details = {}
        
        details["business_rule"] = business_rule
        
        super().__init__(
            message=message,
            error_code="BUSINESS_LOGIC_ERROR",
            status_code=400,
            details=details
        )


# 异常映射字典，用于快速查找异常类
EXCEPTION_MAP = {
    "CHATGPT_BACKEND_ERROR": ChatGPTBackendException,
    "CONVERSATION_NOT_FOUND": ConversationNotFoundException,
    "MESSAGE_NOT_FOUND": MessageNotFoundException,
    "USER_QUOTA_EXCEEDED": UserQuotaExceededException,
    "OPENAI_SERVICE_ERROR": OpenAIServiceException,
    "DATABASE_CONNECTION_ERROR": DatabaseConnectionException,
    "INVALID_MODEL": InvalidModelException,
    "CONVERSATION_LIMIT_EXCEEDED": ConversationLimitExceededException,
    "MESSAGE_TOO_LONG": MessageTooLongException,
    "TOKEN_LIMIT_EXCEEDED": TokenLimitExceededException,
    "CONVERSATION_ARCHIVED": ConversationArchivedException,
    "USER_INACTIVE": UserInactiveException,
    "FILE_UPLOAD_ERROR": FileUploadException,
    "WEBSOCKET_CONNECTION_ERROR": WebSocketConnectionException,
    "CONFIGURATION_ERROR": ConfigurationException,
    "EXTERNAL_SERVICE_ERROR": ExternalServiceException,
    "CACHE_ERROR": CacheException,
    "BUSINESS_LOGIC_ERROR": BusinessLogicException
}


def get_exception_class(error_code: str) -> type:
    """根据错误代码获取异常类"""
    return EXCEPTION_MAP.get(error_code, ChatGPTBackendException)


def create_exception(error_code: str, message: str, **kwargs) -> ChatGPTBackendException:
    """创建异常实例"""
    exception_class = get_exception_class(error_code)
    return exception_class(message=message, **kwargs)