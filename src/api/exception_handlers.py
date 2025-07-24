#!/usr/bin/env python3
"""
API异常处理器

提供FastAPI应用的异常处理器，包括：
1. 自定义异常处理
2. HTTP异常处理
3. 验证异常处理
4. 数据库异常处理
5. 通用异常处理
"""

import traceback
import logging
from typing import Union, Dict, Any

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.exc import (
    SQLAlchemyError,
    IntegrityError,
    DataError,
    OperationalError,
    TimeoutError as SQLTimeoutError
)
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

from ..core.exceptions import (
    BaseAPIException,
    AuthenticationException,
    AuthorizationException,
    ValidationException,
    ResourceNotFoundException,
    ConflictException,
    RateLimitException,
    ServiceUnavailableException,
    InternalServerException
)
from ..core.responses import ResponseBuilder
from ..utils.logger import get_logger


# =============================================================================
# 日志配置
# =============================================================================

logger = get_logger(__name__)
error_logger = get_logger("error")
security_logger = get_logger("security")


# =============================================================================
# 自定义异常处理器
# =============================================================================

async def base_api_exception_handler(
    request: Request, 
    exc: BaseAPIException
) -> JSONResponse:
    """
    基础API异常处理器
    
    Args:
        request: 请求对象
        exc: 基础API异常
        
    Returns:
        JSON响应
    """
    # 记录异常信息
    error_info = {
        "request_id": getattr(request.state, "request_id", "unknown"),
        "method": request.method,
        "url": str(request.url),
        "client_ip": _get_client_ip(request),
        "error_type": type(exc).__name__,
        "error_code": exc.error_code,
        "error_message": exc.message,
        "status_code": exc.status_code
    }
    
    # 根据异常类型选择日志级别
    if exc.status_code >= 500:
        error_logger.error(f"Server error: {error_info}")
    elif exc.status_code >= 400:
        error_logger.warning(f"Client error: {error_info}")
    else:
        error_logger.info(f"API exception: {error_info}")
    
    # 构建响应
    response_data = ResponseBuilder.error(
        message=exc.message,
        error_code=exc.error_code,
        details=exc.details
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data,
        headers=exc.headers
    )


async def authentication_exception_handler(
    request: Request,
    exc: AuthenticationException
) -> JSONResponse:
    """
    认证异常处理器
    
    Args:
        request: 请求对象
        exc: 认证异常
        
    Returns:
        JSON响应
    """
    # 记录安全日志
    security_info = {
        "request_id": getattr(request.state, "request_id", "unknown"),
        "method": request.method,
        "url": str(request.url),
        "client_ip": _get_client_ip(request),
        "user_agent": request.headers.get("User-Agent", ""),
        "error_message": exc.message,
        "auth_header": request.headers.get("Authorization", "")
    }
    
    security_logger.warning(f"Authentication failed: {security_info}")
    
    response_data = ResponseBuilder.error(
        message=exc.message,
        error_code=exc.error_code
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data,
        headers={"WWW-Authenticate": "Bearer"}
    )


async def authorization_exception_handler(
    request: Request,
    exc: AuthorizationException
) -> JSONResponse:
    """
    授权异常处理器
    
    Args:
        request: 请求对象
        exc: 授权异常
        
    Returns:
        JSON响应
    """
    # 记录安全日志
    security_info = {
        "request_id": getattr(request.state, "request_id", "unknown"),
        "method": request.method,
        "url": str(request.url),
        "client_ip": _get_client_ip(request),
        "user_agent": request.headers.get("User-Agent", ""),
        "error_message": exc.message
    }
    
    security_logger.warning(f"Authorization failed: {security_info}")
    
    response_data = ResponseBuilder.error(
        message=exc.message,
        error_code=exc.error_code
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data
    )


async def rate_limit_exception_handler(
    request: Request,
    exc: RateLimitException
) -> JSONResponse:
    """
    限流异常处理器
    
    Args:
        request: 请求对象
        exc: 限流异常
        
    Returns:
        JSON响应
    """
    # 记录限流日志
    rate_limit_info = {
        "request_id": getattr(request.state, "request_id", "unknown"),
        "method": request.method,
        "url": str(request.url),
        "client_ip": _get_client_ip(request),
        "user_agent": request.headers.get("User-Agent", ""),
        "error_message": exc.message
    }
    
    security_logger.warning(f"Rate limit exceeded: {rate_limit_info}")
    
    response_data = ResponseBuilder.error(
        message=exc.message,
        error_code=exc.error_code
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data,
        headers={
            "Retry-After": "60",
            "X-RateLimit-Limit": "60",
            "X-RateLimit-Remaining": "0"
        }
    )


# =============================================================================
# HTTP异常处理器
# =============================================================================

async def http_exception_handler(
    request: Request,
    exc: HTTPException
) -> JSONResponse:
    """
    HTTP异常处理器
    
    Args:
        request: 请求对象
        exc: HTTP异常
        
    Returns:
        JSON响应
    """
    # 记录异常信息
    error_info = {
        "request_id": getattr(request.state, "request_id", "unknown"),
        "method": request.method,
        "url": str(request.url),
        "client_ip": _get_client_ip(request),
        "status_code": exc.status_code,
        "detail": exc.detail
    }
    
    if exc.status_code >= 500:
        error_logger.error(f"HTTP server error: {error_info}")
    elif exc.status_code >= 400:
        error_logger.warning(f"HTTP client error: {error_info}")
    
    # 映射常见HTTP状态码到友好消息
    status_messages = {
        400: "请求参数错误",
        401: "未授权访问",
        403: "禁止访问",
        404: "资源不存在",
        405: "请求方法不允许",
        406: "不可接受的请求",
        408: "请求超时",
        409: "资源冲突",
        410: "资源已删除",
        413: "请求体过大",
        414: "请求URI过长",
        415: "不支持的媒体类型",
        422: "请求参数验证失败",
        429: "请求过于频繁",
        500: "服务器内部错误",
        501: "功能未实现",
        502: "网关错误",
        503: "服务不可用",
        504: "网关超时"
    }
    
    message = status_messages.get(exc.status_code, str(exc.detail))
    
    response_data = ResponseBuilder.error(
        message=message,
        error_code=f"HTTP_{exc.status_code}",
        details=str(exc.detail) if exc.detail != message else None
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data,
        headers=getattr(exc, "headers", None)
    )


# =============================================================================
# 验证异常处理器
# =============================================================================

async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    请求验证异常处理器
    
    Args:
        request: 请求对象
        exc: 请求验证异常
        
    Returns:
        JSON响应
    """
    # 记录验证错误
    validation_info = {
        "request_id": getattr(request.state, "request_id", "unknown"),
        "method": request.method,
        "url": str(request.url),
        "client_ip": _get_client_ip(request),
        "errors": exc.errors()
    }
    
    error_logger.warning(f"Request validation failed: {validation_info}")
    
    # 格式化验证错误
    formatted_errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        formatted_errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })
    
    response_data = ResponseBuilder.error(
        message="请求参数验证失败",
        error_code="VALIDATION_ERROR",
        details={
            "errors": formatted_errors,
            "error_count": len(formatted_errors)
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_data
    )


async def pydantic_validation_exception_handler(
    request: Request,
    exc: ValidationError
) -> JSONResponse:
    """
    Pydantic验证异常处理器
    
    Args:
        request: 请求对象
        exc: Pydantic验证异常
        
    Returns:
        JSON响应
    """
    # 记录验证错误
    validation_info = {
        "request_id": getattr(request.state, "request_id", "unknown"),
        "method": request.method,
        "url": str(request.url),
        "client_ip": _get_client_ip(request),
        "errors": exc.errors()
    }
    
    error_logger.warning(f"Pydantic validation failed: {validation_info}")
    
    # 格式化验证错误
    formatted_errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        formatted_errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"]
        })
    
    response_data = ResponseBuilder.error(
        message="数据验证失败",
        error_code="VALIDATION_ERROR",
        details={
            "errors": formatted_errors,
            "error_count": len(formatted_errors)
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_data
    )


# =============================================================================
# 数据库异常处理器
# =============================================================================

async def sqlalchemy_exception_handler(
    request: Request,
    exc: SQLAlchemyError
) -> JSONResponse:
    """
    SQLAlchemy异常处理器
    
    Args:
        request: 请求对象
        exc: SQLAlchemy异常
        
    Returns:
        JSON响应
    """
    # 记录数据库错误
    db_error_info = {
        "request_id": getattr(request.state, "request_id", "unknown"),
        "method": request.method,
        "url": str(request.url),
        "client_ip": _get_client_ip(request),
        "error_type": type(exc).__name__,
        "error_message": str(exc)
    }
    
    error_logger.error(f"Database error: {db_error_info}")
    
    # 根据异常类型返回不同的错误信息
    if isinstance(exc, IntegrityError):
        # 完整性约束错误
        message = "数据完整性错误，可能存在重复或关联数据"
        error_code = "INTEGRITY_ERROR"
        status_code = status.HTTP_409_CONFLICT
    elif isinstance(exc, DataError):
        # 数据错误
        message = "数据格式错误"
        error_code = "DATA_ERROR"
        status_code = status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, OperationalError):
        # 操作错误
        message = "数据库操作错误"
        error_code = "OPERATIONAL_ERROR"
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, SQLTimeoutError):
        # 超时错误
        message = "数据库操作超时"
        error_code = "TIMEOUT_ERROR"
        status_code = status.HTTP_504_GATEWAY_TIMEOUT
    else:
        # 通用数据库错误
        message = "数据库错误"
        error_code = "DATABASE_ERROR"
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    response_data = ResponseBuilder.error(
        message=message,
        error_code=error_code
    )
    
    return JSONResponse(
        status_code=status_code,
        content=response_data
    )


# =============================================================================
# Redis异常处理器
# =============================================================================

async def redis_exception_handler(
    request: Request,
    exc: RedisError
) -> JSONResponse:
    """
    Redis异常处理器
    
    Args:
        request: 请求对象
        exc: Redis异常
        
    Returns:
        JSON响应
    """
    # 记录Redis错误
    redis_error_info = {
        "request_id": getattr(request.state, "request_id", "unknown"),
        "method": request.method,
        "url": str(request.url),
        "client_ip": _get_client_ip(request),
        "error_type": type(exc).__name__,
        "error_message": str(exc)
    }
    
    error_logger.error(f"Redis error: {redis_error_info}")
    
    # 根据异常类型返回不同的错误信息
    if isinstance(exc, RedisConnectionError):
        message = "缓存服务连接失败"
        error_code = "CACHE_CONNECTION_ERROR"
    else:
        message = "缓存服务错误"
        error_code = "CACHE_ERROR"
    
    response_data = ResponseBuilder.error(
        message=message,
        error_code=error_code
    )
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=response_data
    )


# =============================================================================
# 通用异常处理器
# =============================================================================

async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    通用异常处理器
    
    Args:
        request: 请求对象
        exc: 通用异常
        
    Returns:
        JSON响应
    """
    # 记录详细错误信息
    error_info = {
        "request_id": getattr(request.state, "request_id", "unknown"),
        "method": request.method,
        "url": str(request.url),
        "client_ip": _get_client_ip(request),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc()
    }
    
    error_logger.error(f"Unhandled exception: {error_info}")
    
    response_data = ResponseBuilder.error(
        message="服务器内部错误",
        error_code="INTERNAL_SERVER_ERROR"
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_data
    )


# =============================================================================
# 辅助函数
# =============================================================================

def _get_client_ip(request: Request) -> str:
    """
    获取客户端IP地址
    
    Args:
        request: 请求对象
        
    Returns:
        客户端IP地址
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    return request.client.host if request.client else "unknown"


# =============================================================================
# 异常处理器设置函数
# =============================================================================

def setup_api_exception_handlers(app: FastAPI):
    """
    设置API异常处理器
    
    Args:
        app: FastAPI应用实例
    """
    # 自定义异常处理器
    app.add_exception_handler(BaseAPIException, base_api_exception_handler)
    app.add_exception_handler(AuthenticationException, authentication_exception_handler)
    app.add_exception_handler(AuthorizationException, authorization_exception_handler)
    app.add_exception_handler(ValidationException, base_api_exception_handler)
    app.add_exception_handler(ResourceNotFoundException, base_api_exception_handler)
    app.add_exception_handler(ConflictException, base_api_exception_handler)
    app.add_exception_handler(RateLimitException, rate_limit_exception_handler)
    app.add_exception_handler(ServiceUnavailableException, base_api_exception_handler)
    app.add_exception_handler(InternalServerException, base_api_exception_handler)
    
    # HTTP异常处理器
    app.add_exception_handler(HTTPException, http_exception_handler)
    
    # 验证异常处理器
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    app.add_exception_handler(ValidationError, pydantic_validation_exception_handler)
    
    # 数据库异常处理器
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
    
    # Redis异常处理器
    app.add_exception_handler(RedisError, redis_exception_handler)
    
    # 通用异常处理器（必须放在最后）
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("API exception handlers setup completed")


def create_error_response(
    message: str,
    error_code: str = "UNKNOWN_ERROR",
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    details: Union[Dict[str, Any], str, None] = None,
    headers: Union[Dict[str, str], None] = None
) -> JSONResponse:
    """
    创建错误响应
    
    Args:
        message: 错误消息
        error_code: 错误代码
        status_code: HTTP状态码
        details: 错误详情
        headers: 响应头
        
    Returns:
        JSON响应
    """
    response_data = ResponseBuilder.error(
        message=message,
        error_code=error_code,
        details=details
    )
    
    return JSONResponse(
        status_code=status_code,
        content=response_data,
        headers=headers
    )