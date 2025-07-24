#!/usr/bin/env python3
"""
异常处理模块

提供统一的异常处理，支持：
1. 自定义异常类
2. 全局异常处理器
3. 错误响应格式化
4. 异常日志记录
5. 错误追踪
6. 用户友好的错误消息
"""

import traceback
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from redis.exceptions import RedisError
from pydantic import ValidationError as PydanticValidationError

from .config import get_settings
from .logging import get_logger


# 配置日志
logger = get_logger(__name__)


class BaseCustomException(Exception):
    """
    自定义异常基类
    
    所有自定义异常都应该继承此类
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.status_code = status_code
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# 业务异常
# =============================================================================

class BusinessException(BaseCustomException):
    """业务逻辑异常"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            **kwargs
        )


class ValidationException(BaseCustomException):
    """数据验证异常"""
    
    def __init__(self, message: str, field_errors: Optional[List[Dict[str, Any]]] = None, **kwargs):
        details = kwargs.get('details', {})
        if field_errors:
            details['field_errors'] = field_errors
        
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
            **kwargs
        )


class AuthenticationException(BaseCustomException):
    """认证异常"""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            **kwargs
        )


class AuthorizationException(BaseCustomException):
    """授权异常"""
    
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            **kwargs
        )


class ResourceNotFoundException(BaseCustomException):
    """资源未找到异常"""
    
    def __init__(self, resource_type: str, resource_id: Any = None, **kwargs):
        if resource_id:
            message = f"{resource_type} with ID '{resource_id}' not found"
        else:
            message = f"{resource_type} not found"
        
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            details={"resource_type": resource_type, "resource_id": resource_id},
            **kwargs
        )


class ConflictException(BaseCustomException):
    """资源冲突异常"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            **kwargs
        )


class RateLimitException(BaseCustomException):
    """限流异常"""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None, **kwargs):
        details = kwargs.get('details', {})
        if retry_after:
            details['retry_after'] = retry_after
        
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details,
            **kwargs
        )


# =============================================================================
# 系统异常
# =============================================================================

class DatabaseException(BaseCustomException):
    """数据库异常"""
    
    def __init__(self, message: str = "Database error occurred", **kwargs):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            **kwargs
        )


class CacheException(BaseCustomException):
    """缓存异常"""
    
    def __init__(self, message: str = "Cache error occurred", **kwargs):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            **kwargs
        )


class ExternalServiceException(BaseCustomException):
    """外部服务异常"""
    
    def __init__(self, service_name: str, message: str = None, **kwargs):
        if not message:
            message = f"External service '{service_name}' error"
        
        super().__init__(
            message=message,
            status_code=status.HTTP_502_BAD_GATEWAY,
            details={"service_name": service_name},
            **kwargs
        )


class ConfigurationException(BaseCustomException):
    """配置异常"""
    
    def __init__(self, message: str = "Configuration error", **kwargs):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            **kwargs
        )


class FileOperationException(BaseCustomException):
    """文件操作异常"""
    
    def __init__(self, operation: str, filename: str, message: str = None, **kwargs):
        if not message:
            message = f"File {operation} failed for '{filename}'"
        
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"operation": operation, "filename": filename},
            **kwargs
        )


# =============================================================================
# 异常处理器
# =============================================================================

class ExceptionHandler:
    """
    异常处理器
    
    负责处理和格式化异常响应
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    def create_error_response(
        self,
        error_code: str,
        message: str,
        status_code: int,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> JSONResponse:
        """
        创建错误响应
        
        Args:
            error_code: 错误代码
            message: 错误消息
            status_code: HTTP状态码
            details: 错误详情
            request_id: 请求ID
        
        Returns:
            JSONResponse: 错误响应
        """
        error_response = {
            "success": False,
            "error": {
                "code": error_code,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
            }
        }
        
        if details:
            error_response["error"]["details"] = details
        
        if request_id:
            error_response["request_id"] = request_id
        
        # 开发环境显示更多信息
        if self.settings.app_environment == "development":
            error_response["error"]["debug"] = {
                "traceback": traceback.format_exc(),
            }
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )
    
    def log_exception(
        self,
        exc: Exception,
        request: Optional[Request] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录异常日志
        
        Args:
            exc: 异常对象
            request: 请求对象
            extra_context: 额外上下文
        """
        context = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
        
        if request:
            context.update({
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "client_ip": request.client.host if request.client else None,
            })
        
        if extra_context:
            context.update(extra_context)
        
        if isinstance(exc, BaseCustomException):
            context.update({
                "error_code": exc.error_code,
                "status_code": exc.status_code,
                "details": exc.details,
            })
        
        # 根据异常类型选择日志级别
        if isinstance(exc, (AuthenticationException, AuthorizationException, ValidationException)):
            logger.warning(f"Client error: {exc}", extra=context)
        elif isinstance(exc, BaseCustomException) and exc.status_code < 500:
            logger.info(f"Client error: {exc}", extra=context)
        else:
            logger.error(f"Server error: {exc}", extra=context, exc_info=True)


# 全局异常处理器实例
exception_handler = ExceptionHandler()


# =============================================================================
# FastAPI异常处理器
# =============================================================================

async def custom_exception_handler(request: Request, exc: BaseCustomException) -> JSONResponse:
    """
    自定义异常处理器
    
    Args:
        request: 请求对象
        exc: 自定义异常
    
    Returns:
        JSONResponse: 错误响应
    """
    # 记录异常日志
    exception_handler.log_exception(exc, request)
    
    # 获取请求ID
    request_id = getattr(request.state, 'request_id', None)
    
    return exception_handler.create_error_response(
        error_code=exc.error_code,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
        request_id=request_id,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    HTTP异常处理器
    
    Args:
        request: 请求对象
        exc: HTTP异常
    
    Returns:
        JSONResponse: 错误响应
    """
    # 记录异常日志
    exception_handler.log_exception(exc, request)
    
    # 获取请求ID
    request_id = getattr(request.state, 'request_id', None)
    
    return exception_handler.create_error_response(
        error_code="HTTP_EXCEPTION",
        message=exc.detail,
        status_code=exc.status_code,
        request_id=request_id,
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    验证异常处理器
    
    Args:
        request: 请求对象
        exc: 验证异常
    
    Returns:
        JSONResponse: 错误响应
    """
    # 记录异常日志
    exception_handler.log_exception(exc, request)
    
    # 格式化验证错误
    field_errors = []
    for error in exc.errors():
        field_errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input"),
        })
    
    # 获取请求ID
    request_id = getattr(request.state, 'request_id', None)
    
    return exception_handler.create_error_response(
        error_code="VALIDATION_ERROR",
        message="Validation failed",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details={"field_errors": field_errors},
        request_id=request_id,
    )


async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """
    SQLAlchemy异常处理器
    
    Args:
        request: 请求对象
        exc: SQLAlchemy异常
    
    Returns:
        JSONResponse: 错误响应
    """
    # 记录异常日志
    exception_handler.log_exception(exc, request)
    
    # 获取请求ID
    request_id = getattr(request.state, 'request_id', None)
    
    # 根据异常类型返回不同的错误消息
    if isinstance(exc, IntegrityError):
        message = "Data integrity constraint violation"
        error_code = "INTEGRITY_ERROR"
    else:
        message = "Database operation failed"
        error_code = "DATABASE_ERROR"
    
    return exception_handler.create_error_response(
        error_code=error_code,
        message=message,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        request_id=request_id,
    )


async def redis_exception_handler(request: Request, exc: RedisError) -> JSONResponse:
    """
    Redis异常处理器
    
    Args:
        request: 请求对象
        exc: Redis异常
    
    Returns:
        JSONResponse: 错误响应
    """
    # 记录异常日志
    exception_handler.log_exception(exc, request)
    
    # 获取请求ID
    request_id = getattr(request.state, 'request_id', None)
    
    return exception_handler.create_error_response(
        error_code="CACHE_ERROR",
        message="Cache operation failed",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        request_id=request_id,
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    通用异常处理器
    
    Args:
        request: 请求对象
        exc: 异常
    
    Returns:
        JSONResponse: 错误响应
    """
    # 记录异常日志
    exception_handler.log_exception(exc, request)
    
    # 获取请求ID
    request_id = getattr(request.state, 'request_id', None)
    
    return exception_handler.create_error_response(
        error_code="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        request_id=request_id,
    )


# =============================================================================
# 异常处理器注册函数
# =============================================================================

def register_exception_handlers(app):
    """
    注册异常处理器
    
    Args:
        app: FastAPI应用实例
    """
    # 自定义异常
    app.add_exception_handler(BaseCustomException, custom_exception_handler)
    
    # HTTP异常
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # 验证异常
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(PydanticValidationError, validation_exception_handler)
    
    # 数据库异常
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
    
    # Redis异常
    app.add_exception_handler(RedisError, redis_exception_handler)
    
    # 通用异常（必须放在最后）
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("Exception handlers registered")


# =============================================================================
# 便捷函数
# =============================================================================

def raise_not_found(resource_type: str, resource_id: Any = None) -> None:
    """
    抛出资源未找到异常
    
    Args:
        resource_type: 资源类型
        resource_id: 资源ID
    """
    raise ResourceNotFoundException(resource_type, resource_id)


def raise_validation_error(message: str, field_errors: Optional[List[Dict[str, Any]]] = None) -> None:
    """
    抛出验证异常
    
    Args:
        message: 错误消息
        field_errors: 字段错误列表
    """
    raise ValidationException(message, field_errors)


def raise_business_error(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """
    抛出业务异常
    
    Args:
        message: 错误消息
        details: 错误详情
    """
    raise BusinessException(message, details=details)


def raise_auth_error(message: str = "Authentication failed") -> None:
    """
    抛出认证异常
    
    Args:
        message: 错误消息
    """
    raise AuthenticationException(message)


def raise_permission_error(message: str = "Access denied") -> None:
    """
    抛出权限异常
    
    Args:
        message: 错误消息
    """
    raise AuthorizationException(message)


def raise_conflict_error(message: str) -> None:
    """
    抛出冲突异常
    
    Args:
        message: 错误消息
    """
    raise ConflictException(message)


def raise_rate_limit_error(message: str = "Rate limit exceeded", retry_after: Optional[int] = None) -> None:
    """
    抛出限流异常
    
    Args:
        message: 错误消息
        retry_after: 重试间隔
    """
    raise RateLimitException(message, retry_after=retry_after)