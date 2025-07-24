# -*- coding: utf-8 -*-
"""
异常处理模块

提供全局异常处理和自定义异常类。
"""

from .handlers import (
    setup_exception_handlers,
    validation_exception_handler,
    authentication_exception_handler,
    authorization_exception_handler,
    rate_limit_exception_handler,
    openai_exception_handler,
    database_exception_handler,
    general_exception_handler
)

from .custom_exceptions import (
    ChatGPTBackendException,
    ConversationNotFoundException,
    MessageNotFoundException,
    UserQuotaExceededException,
    OpenAIServiceException,
    DatabaseConnectionException
)

__all__ = [
    # 异常处理器
    "setup_exception_handlers",
    "validation_exception_handler",
    "authentication_exception_handler",
    "authorization_exception_handler",
    "rate_limit_exception_handler",
    "openai_exception_handler",
    "database_exception_handler",
    "general_exception_handler",
    
    # 自定义异常
    "ChatGPTBackendException",
    "ConversationNotFoundException",
    "MessageNotFoundException",
    "UserQuotaExceededException",
    "OpenAIServiceException",
    "DatabaseConnectionException"
]