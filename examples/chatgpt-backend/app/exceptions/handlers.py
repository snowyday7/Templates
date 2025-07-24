# -*- coding: utf-8 -*-
"""
异常处理器

提供全局异常处理逻辑。
"""

import sys
from pathlib import Path
import traceback
from typing import Dict, Any, Optional
from datetime import datetime

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from pydantic import ValidationError
import openai

# 导入模板库组件
from src.utils.exceptions import (
    BaseCustomException,
    ValidationException,
    AuthenticationException,
    AuthorizationException,
    RateLimitException
)
from src.core.logging import get_logger

# 导入应用组件
from app.exceptions.custom_exceptions import (
    ChatGPTBackendException,
    ConversationNotFoundException,
    MessageNotFoundException,
    UserQuotaExceededException,
    OpenAIServiceException,
    DatabaseConnectionException
)
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


def create_error_response(
    error_code: str,
    message: str,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> JSONResponse:
    """
    创建标准错误响应
    """
    error_data = {
        "error": {
            "code": error_code,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id
        }
    }
    
    if details:
        error_data["error"]["details"] = details
    
    # 在开发环境中包含更多调试信息
    if settings.ENVIRONMENT == "development":
        error_data["error"]["debug"] = {
            "environment": settings.ENVIRONMENT,
            "traceback": traceback.format_exc() if status_code >= 500 else None
        }
    
    return JSONResponse(
        status_code=status_code,
        content=error_data
    )


async def validation_exception_handler(request: Request, exc: ValidationException) -> JSONResponse:
    """
    处理验证异常
    """
    request_id = getattr(request.state, "request_id", None)
    
    logger.warning(
        f"Validation error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return create_error_response(
        error_code=exc.error_code,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
        request_id=request_id
    )


async def authentication_exception_handler(request: Request, exc: AuthenticationException) -> JSONResponse:
    """
    处理认证异常
    """
    request_id = getattr(request.state, "request_id", None)
    
    logger.warning(
        f"Authentication error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "client_ip": _get_client_ip(request)
        }
    )
    
    return create_error_response(
        error_code=exc.error_code,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
        request_id=request_id
    )


async def authorization_exception_handler(request: Request, exc: AuthorizationException) -> JSONResponse:
    """
    处理授权异常
    """
    request_id = getattr(request.state, "request_id", None)
    user_id = getattr(request.state, "user_id", None)
    
    logger.warning(
        f"Authorization error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "request_id": request_id,
            "user_id": user_id,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return create_error_response(
        error_code=exc.error_code,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
        request_id=request_id
    )


async def rate_limit_exception_handler(request: Request, exc: RateLimitException) -> JSONResponse:
    """
    处理限流异常
    """
    request_id = getattr(request.state, "request_id", None)
    
    logger.warning(
        f"Rate limit exceeded: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "client_ip": _get_client_ip(request)
        }
    )
    
    response = create_error_response(
        error_code=exc.error_code,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
        request_id=request_id
    )
    
    # 添加限流相关的响应头
    if hasattr(exc, 'retry_after'):
        response.headers["Retry-After"] = str(exc.retry_after)
    if hasattr(exc, 'limit'):
        response.headers["X-RateLimit-Limit"] = str(exc.limit)
        response.headers["X-RateLimit-Remaining"] = "0"
    
    return response


async def openai_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    处理OpenAI API异常
    """
    request_id = getattr(request.state, "request_id", None)
    
    # 解析OpenAI异常
    if isinstance(exc, openai.APIError):
        error_code = "OPENAI_API_ERROR"
        message = f"OpenAI API错误: {str(exc)}"
        status_code = getattr(exc, 'status_code', 502)
        details = {
            "openai_error_type": type(exc).__name__,
            "openai_error_code": getattr(exc, 'code', None)
        }
    elif isinstance(exc, openai.RateLimitError):
        error_code = "OPENAI_RATE_LIMIT"
        message = "OpenAI API请求频率超限"
        status_code = 429
        details = {"retry_after": 60}
    elif isinstance(exc, openai.AuthenticationError):
        error_code = "OPENAI_AUTH_ERROR"
        message = "OpenAI API认证失败"
        status_code = 502
        details = None
    elif isinstance(exc, openai.APIConnectionError):
        error_code = "OPENAI_CONNECTION_ERROR"
        message = "无法连接到OpenAI API"
        status_code = 502
        details = None
    else:
        error_code = "OPENAI_UNKNOWN_ERROR"
        message = f"OpenAI服务未知错误: {str(exc)}"
        status_code = 502
        details = None
    
    logger.error(
        f"OpenAI API error: {message}",
        extra={
            "error_code": error_code,
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "openai_error": str(exc)
        },
        exc_info=True
    )
    
    return create_error_response(
        error_code=error_code,
        message=message,
        status_code=status_code,
        details=details,
        request_id=request_id
    )


async def database_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """
    处理数据库异常
    """
    request_id = getattr(request.state, "request_id", None)
    
    # 解析数据库异常类型
    if isinstance(exc, IntegrityError):
        error_code = "DATABASE_INTEGRITY_ERROR"
        message = "数据完整性错误"
        status_code = 400
        details = {"constraint_violation": True}
    elif isinstance(exc, OperationalError):
        error_code = "DATABASE_OPERATIONAL_ERROR"
        message = "数据库操作错误"
        status_code = 503
        details = {"database_unavailable": True}
    else:
        error_code = "DATABASE_ERROR"
        message = "数据库错误"
        status_code = 500
        details = None
    
    logger.error(
        f"Database error: {message}",
        extra={
            "error_code": error_code,
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "database_error": str(exc)
        },
        exc_info=True
    )
    
    return create_error_response(
        error_code=error_code,
        message=message,
        status_code=status_code,
        details=details,
        request_id=request_id
    )


async def chatgpt_backend_exception_handler(request: Request, exc: ChatGPTBackendException) -> JSONResponse:
    """
    处理ChatGPT后端自定义异常
    """
    request_id = getattr(request.state, "request_id", None)
    
    # 根据状态码确定日志级别
    if exc.status_code >= 500:
        log_level = "error"
    elif exc.status_code >= 400:
        log_level = "warning"
    else:
        log_level = "info"
    
    getattr(logger, log_level)(
        f"ChatGPT backend error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": exc.status_code,
            "details": exc.details
        },
        exc_info=exc.status_code >= 500
    )
    
    return create_error_response(
        error_code=exc.error_code,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
        request_id=request_id
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    处理HTTP异常
    """
    request_id = getattr(request.state, "request_id", None)
    
    # 映射HTTP状态码到错误代码
    status_code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        422: "UNPROCESSABLE_ENTITY",
        429: "TOO_MANY_REQUESTS",
        500: "INTERNAL_SERVER_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
        504: "GATEWAY_TIMEOUT"
    }
    
    error_code = status_code_map.get(exc.status_code, "HTTP_ERROR")
    
    logger.warning(
        f"HTTP error: {exc.detail}",
        extra={
            "error_code": error_code,
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": exc.status_code
        }
    )
    
    return create_error_response(
        error_code=error_code,
        message=exc.detail,
        status_code=exc.status_code,
        request_id=request_id
    )


async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    处理请求验证异常
    """
    request_id = getattr(request.state, "request_id", None)
    
    # 格式化验证错误
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    logger.warning(
        f"Request validation error: {len(errors)} validation errors",
        extra={
            "error_code": "REQUEST_VALIDATION_ERROR",
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "validation_errors": errors
        }
    )
    
    return create_error_response(
        error_code="REQUEST_VALIDATION_ERROR",
        message="请求参数验证失败",
        status_code=422,
        details={"validation_errors": errors},
        request_id=request_id
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    处理通用异常
    """
    request_id = getattr(request.state, "request_id", None)
    
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "error_code": "INTERNAL_SERVER_ERROR",
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "exception_type": type(exc).__name__
        },
        exc_info=True
    )
    
    # 在生产环境中隐藏详细错误信息
    if settings.ENVIRONMENT == "production":
        message = "服务器内部错误"
        details = None
    else:
        message = f"未处理的异常: {str(exc)}"
        details = {
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc()
        }
    
    return create_error_response(
        error_code="INTERNAL_SERVER_ERROR",
        message=message,
        status_code=500,
        details=details,
        request_id=request_id
    )


def _get_client_ip(request: Request) -> str:
    """
    获取客户端IP地址
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    return request.client.host if request.client else "unknown"


def setup_exception_handlers(app: FastAPI):
    """
    设置全局异常处理器
    """
    # 自定义异常处理器
    app.add_exception_handler(ValidationException, validation_exception_handler)
    app.add_exception_handler(AuthenticationException, authentication_exception_handler)
    app.add_exception_handler(AuthorizationException, authorization_exception_handler)
    app.add_exception_handler(RateLimitException, rate_limit_exception_handler)
    app.add_exception_handler(ChatGPTBackendException, chatgpt_backend_exception_handler)
    
    # 数据库异常处理器
    app.add_exception_handler(SQLAlchemyError, database_exception_handler)
    
    # OpenAI异常处理器
    app.add_exception_handler(openai.APIError, openai_exception_handler)
    app.add_exception_handler(openai.RateLimitError, openai_exception_handler)
    app.add_exception_handler(openai.AuthenticationError, openai_exception_handler)
    app.add_exception_handler(openai.APIConnectionError, openai_exception_handler)
    
    # HTTP异常处理器
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # 请求验证异常处理器
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    
    # 通用异常处理器（必须放在最后）
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("Exception handlers configured")


# 异常处理器映射
EXCEPTION_HANDLERS = {
    ValidationException: validation_exception_handler,
    AuthenticationException: authentication_exception_handler,
    AuthorizationException: authorization_exception_handler,
    RateLimitException: rate_limit_exception_handler,
    ChatGPTBackendException: chatgpt_backend_exception_handler,
    SQLAlchemyError: database_exception_handler,
    HTTPException: http_exception_handler,
    RequestValidationError: request_validation_exception_handler,
    Exception: general_exception_handler
}


def get_exception_handler(exception_type: type):
    """
    获取异常处理器
    """
    return EXCEPTION_HANDLERS.get(exception_type, general_exception_handler)