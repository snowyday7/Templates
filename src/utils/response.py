#!/usr/bin/env python3
"""
响应工具模块

提供标准化的API响应格式
"""

import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from .logger import get_logger
from .helpers import format_datetime


# =============================================================================
# 类型定义和枚举
# =============================================================================

logger = get_logger(__name__)


class ResponseStatus(Enum):
    """响应状态枚举"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class HttpStatus(Enum):
    """HTTP状态码枚举"""
    # 成功响应
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204

    # 重定向
    MOVED_PERMANENTLY = 301
    FOUND = 302
    NOT_MODIFIED = 304

    # 客户端错误
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429

    # 服务器错误
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504


# =============================================================================
# 响应数据类
# =============================================================================

@dataclass
class ApiResponse:
    """标准API响应格式"""
    status: str  # 响应状态
    message: str  # 响应消息
    data: Optional[Any] = None  # 响应数据
    errors: Optional[List[Dict[str, Any]]] = None  # 错误详情
    meta: Optional[Dict[str, Any]] = None  # 元数据
    timestamp: Optional[str] = None  # 时间戳
    request_id: Optional[str] = None  # 请求ID

    def __post_init__(self):
        """初始化后处理"""
        if self.timestamp is None:
            self.timestamp = format_datetime(datetime.now())

        if self.errors is None:
            self.errors = []

        if self.meta is None:
            self.meta = {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)

        # 移除空值
        if not result['errors']:
            del result['errors']

        if not result['meta']:
            del result['meta']

        if result['data'] is None:
            del result['data']

        return result

    def to_json(self, **kwargs) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)


@dataclass
class ErrorDetail:
    """错误详情"""
    code: str  # 错误代码
    message: str  # 错误消息
    field: Optional[str] = None  # 相关字段
    details: Optional[Dict[str, Any]] = None  # 额外详情

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)

        # 移除空值
        if result['field'] is None:
            del result['field']

        if not result['details']:
            del result['details']

        return result


@dataclass
class PaginationMeta:
    """分页元数据"""
    page: int
    size: int
    total: int
    total_pages: int
    has_previous: bool
    has_next: bool

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


# =============================================================================
# 响应构建器
# =============================================================================

class ResponseBuilder:
    """响应构建器"""

    def __init__(self):
        self._status = ResponseStatus.SUCCESS.value
        self._message = ""
        self._data = None
        self._errors = []
        self._meta = {}
        self._request_id = None

    def status(self, status: Union[ResponseStatus, str]) -> 'ResponseBuilder':
        """设置状态"""
        if isinstance(status, ResponseStatus):
            self._status = status.value
        else:
            self._status = status
        return self

    def message(self, message: str) -> 'ResponseBuilder':
        """设置消息"""
        self._message = message
        return self

    def data(self, data: Any) -> 'ResponseBuilder':
        """设置数据"""
        self._data = data
        return self

    def error(self, code: str, message: str, field: Optional[str] = None,
             details: Optional[Dict[str, Any]] = None) -> 'ResponseBuilder':
        """添加错误"""
        error_detail = ErrorDetail(
            code=code,
            message=message,
            field=field,
            details=details
        )
        self._errors.append(error_detail.to_dict())
        return self

    def errors(self, errors: List[Dict[str, Any]]) -> 'ResponseBuilder':
        """设置错误列表"""
        self._errors = errors
        return self

    def meta(self, key: str, value: Any) -> 'ResponseBuilder':
        """添加元数据"""
        self._meta[key] = value
        return self

    def meta_dict(self, meta: Dict[str, Any]) -> 'ResponseBuilder':
        """设置元数据字典"""
        self._meta.update(meta)
        return self

    def pagination(self, pagination_meta: PaginationMeta) -> 'ResponseBuilder':
        """添加分页元数据"""
        self._meta['pagination'] = pagination_meta.to_dict()
        return self

    def request_id(self, request_id: str) -> 'ResponseBuilder':
        """设置请求ID"""
        self._request_id = request_id
        return self

    def build(self) -> ApiResponse:
        """构建响应"""
        return ApiResponse(
            status=self._status,
            message=self._message,
            data=self._data,
            errors=self._errors if self._errors else None,
            meta=self._meta if self._meta else None,
            request_id=self._request_id
        )


# =============================================================================
# 快捷响应函数
# =============================================================================

def success_response(
    message: str = "操作成功",
    data: Optional[Any] = None,
    meta: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> ApiResponse:
    """成功响应"""
    builder = ResponseBuilder()
    builder.status(ResponseStatus.SUCCESS).message(message)

    if data is not None:
        builder.data(data)

    if meta:
        builder.meta_dict(meta)

    if request_id:
        builder.request_id(request_id)

    return builder.build()


def error_response(
    message: str = "操作失败",
    errors: Optional[List[Dict[str, Any]]] = None,
    data: Optional[Any] = None,
    meta: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> ApiResponse:
    """错误响应"""
    builder = ResponseBuilder()
    builder.status(ResponseStatus.ERROR).message(message)

    if errors:
        builder.errors(errors)

    if data is not None:
        builder.data(data)

    if meta:
        builder.meta_dict(meta)

    if request_id:
        builder.request_id(request_id)

    return builder.build()


def validation_error_response(
    message: str = "数据验证失败",
    validation_errors: Optional[Dict[str, List[str]]] = None,
    request_id: Optional[str] = None
) -> ApiResponse:
    """验证错误响应"""
    builder = ResponseBuilder()
    builder.status(ResponseStatus.ERROR).message(message)

    if validation_errors:
        for field, field_errors in validation_errors.items():
            for error_msg in field_errors:
                builder.error(
                    code="VALIDATION_ERROR",
                    message=error_msg,
                    field=field
                )

    if request_id:
        builder.request_id(request_id)

    return builder.build()


def not_found_response(
    message: str = "资源未找到",
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> ApiResponse:
    """资源未找到响应"""
    builder = ResponseBuilder()
    builder.status(ResponseStatus.ERROR).message(message)

    details = {}
    if resource_type:
        details['resource_type'] = resource_type
    if resource_id:
        details['resource_id'] = resource_id

    if details:
        builder.error(
            code="NOT_FOUND",
            message=message,
            details=details
        )

    if request_id:
        builder.request_id(request_id)

    return builder.build()


def unauthorized_response(
    message: str = "未授权访问",
    request_id: Optional[str] = None
) -> ApiResponse:
    """未授权响应"""
    builder = ResponseBuilder()
    builder.status(ResponseStatus.ERROR).message(message)
    builder.error(code="UNAUTHORIZED", message=message)

    if request_id:
        builder.request_id(request_id)

    return builder.build()


def forbidden_response(
    message: str = "禁止访问",
    required_permissions: Optional[List[str]] = None,
    request_id: Optional[str] = None
) -> ApiResponse:
    """禁止访问响应"""
    builder = ResponseBuilder()
    builder.status(ResponseStatus.ERROR).message(message)

    details = {}
    if required_permissions:
        details['required_permissions'] = required_permissions

    builder.error(
        code="FORBIDDEN",
        message=message,
        details=details if details else None
    )

    if request_id:
        builder.request_id(request_id)

    return builder.build()


def rate_limit_response(
    message: str = "请求过于频繁",
    retry_after: Optional[int] = None,
    request_id: Optional[str] = None
) -> ApiResponse:
    """速率限制响应"""
    builder = ResponseBuilder()
    builder.status(ResponseStatus.ERROR).message(message)

    details = {}
    if retry_after:
        details['retry_after'] = retry_after

    builder.error(
        code="RATE_LIMIT_EXCEEDED",
        message=message,
        details=details if details else None
    )

    if request_id:
        builder.request_id(request_id)

    return builder.build()


def server_error_response(
    message: str = "服务器内部错误",
    error_code: Optional[str] = None,
    request_id: Optional[str] = None
) -> ApiResponse:
    """服务器错误响应"""
    builder = ResponseBuilder()
    builder.status(ResponseStatus.ERROR).message(message)

    builder.error(
        code=error_code or "INTERNAL_SERVER_ERROR",
        message=message
    )

    if request_id:
        builder.request_id(request_id)

    return builder.build()


def paginated_response(
    items: List[Any],
    page: int,
    size: int,
    total: int,
    message: str = "获取成功",
    request_id: Optional[str] = None
) -> ApiResponse:
    """分页响应"""
    import math

    total_pages = math.ceil(total / size) if total > 0 else 1
    has_previous = page > 1
    has_next = page < total_pages

    pagination_meta = PaginationMeta(
        page=page,
        size=size,
        total=total,
        total_pages=total_pages,
        has_previous=has_previous,
        has_next=has_next
    )

    builder = ResponseBuilder()
    builder.status(ResponseStatus.SUCCESS).message(message)
    builder.data(items)
    builder.pagination(pagination_meta)

    if request_id:
        builder.request_id(request_id)

    return builder.build()


# =============================================================================
# 响应包装器
# =============================================================================

class ResponseWrapper:
    """响应包装器"""

    def __init__(self, request_id: Optional[str] = None):
        self.request_id = request_id

    def success(
        self,
        message: str = "操作成功",
        data: Optional[Any] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> ApiResponse:
        """成功响应"""
        return success_response(message, data, meta, self.request_id)

    def error(
        self,
        message: str = "操作失败",
        errors: Optional[List[Dict[str, Any]]] = None,
        data: Optional[Any] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> ApiResponse:
        """错误响应"""
        return error_response(message, errors, data, meta, self.request_id)

    def validation_error(
        self,
        message: str = "数据验证失败",
        validation_errors: Optional[Dict[str, List[str]]] = None
    ) -> ApiResponse:
        """验证错误响应"""
        return validation_error_response(message, validation_errors, self.request_id)

    def not_found(
        self,
        message: str = "资源未找到",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None
    ) -> ApiResponse:
        """资源未找到响应"""
        return not_found_response(message, resource_type, resource_id, self.request_id)

    def unauthorized(self, message: str = "未授权访问") -> ApiResponse:
        """未授权响应"""
        return unauthorized_response(message, self.request_id)

    def forbidden(
        self,
        message: str = "禁止访问",
        required_permissions: Optional[List[str]] = None
    ) -> ApiResponse:
        """禁止访问响应"""
        return forbidden_response(message, required_permissions, self.request_id)

    def rate_limit(
        self,
        message: str = "请求过于频繁",
        retry_after: Optional[int] = None
    ) -> ApiResponse:
        """速率限制响应"""
        return rate_limit_response(message, retry_after, self.request_id)

    def server_error(
        self,
        message: str = "服务器内部错误",
        error_code: Optional[str] = None
    ) -> ApiResponse:
        """服务器错误响应"""
        return server_error_response(message, error_code, self.request_id)

    def paginated(
        self,
        items: List[Any],
        page: int,
        size: int,
        total: int,
        message: str = "获取成功"
    ) -> ApiResponse:
        """分页响应"""
        return paginated_response(items, page, size, total, message, self.request_id)


# =============================================================================
# 工具函数
# =============================================================================

def create_response_from_exception(
    exception: Exception,
    request_id: Optional[str] = None,
    include_traceback: bool = False
) -> ApiResponse:
    """从异常创建响应"""
    from ..core.exceptions import (
        BaseAPIException,
        ValidationException,
        AuthenticationException,
        AuthorizationException,
        NotFoundException,
        RateLimitException
    )

    builder = ResponseBuilder()
    builder.status(ResponseStatus.ERROR)

    if request_id:
        builder.request_id(request_id)

    # 根据异常类型设置响应
    if isinstance(exception, ValidationException):
        builder.message("数据验证失败")
        builder.error("VALIDATION_ERROR", str(exception))

    elif isinstance(exception, AuthenticationException):
        builder.message("认证失败")
        builder.error("AUTHENTICATION_ERROR", str(exception))

    elif isinstance(exception, AuthorizationException):
        builder.message("权限不足")
        builder.error("AUTHORIZATION_ERROR", str(exception))

    elif isinstance(exception, NotFoundException):
        builder.message("资源未找到")
        builder.error("NOT_FOUND", str(exception))

    elif isinstance(exception, RateLimitException):
        builder.message("请求过于频繁")
        builder.error("RATE_LIMIT_EXCEEDED", str(exception))

    elif isinstance(exception, BaseAPIException):
        builder.message(exception.message or "操作失败")
        builder.error(exception.error_code or "API_ERROR", str(exception))

    else:
        builder.message("服务器内部错误")
        builder.error("INTERNAL_SERVER_ERROR", str(exception))

    # 添加异常详情
    if include_traceback:
        import traceback
        builder.meta("traceback", traceback.format_exc())

    return builder.build()


def serialize_response_data(data: Any) -> Any:
    """序列化响应数据"""
    if data is None:
        return None

    if isinstance(data, (str, int, float, bool)):
        return data

    if isinstance(data, datetime):
        return format_datetime(data)

    if isinstance(data, dict):
        return {k: serialize_response_data(v) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        return [serialize_response_data(item) for item in data]

    if hasattr(data, 'to_dict'):
        return serialize_response_data(data.to_dict())

    if hasattr(data, '__dict__'):
        return serialize_response_data(data.__dict__)

    # 尝试转换为字符串
    try:
        return str(data)
    except Exception:
        logger.warning(f"Failed to serialize data: {type(data)}")
        return None


def format_response_for_http(
    response: ApiResponse,
    status_code: Optional[int] = None
) -> tuple[Dict[str, Any], int]:
    """格式化响应用于HTTP返回"""
    # 序列化数据
    response_dict = response.to_dict()
    response_dict['data'] = serialize_response_data(response_dict.get('data'))

    # 确定HTTP状态码
    if status_code is None:
        if response.status == ResponseStatus.SUCCESS.value:
            status_code = HttpStatus.OK.value
        elif response.errors:
            # 根据错误类型确定状态码
            first_error = response.errors[0]
            error_code = first_error.get('code', '')

            if 'VALIDATION' in error_code:
                status_code = HttpStatus.UNPROCESSABLE_ENTITY.value
            elif 'AUTHENTICATION' in error_code or 'UNAUTHORIZED' in error_code:
                status_code = HttpStatus.UNAUTHORIZED.value
            elif 'AUTHORIZATION' in error_code or 'FORBIDDEN' in error_code:
                status_code = HttpStatus.FORBIDDEN.value
            elif 'NOT_FOUND' in error_code:
                status_code = HttpStatus.NOT_FOUND.value
            elif 'RATE_LIMIT' in error_code:
                status_code = HttpStatus.TOO_MANY_REQUESTS.value
            else:
                status_code = HttpStatus.INTERNAL_SERVER_ERROR.value
        else:
            status_code = HttpStatus.INTERNAL_SERVER_ERROR.value

    return response_dict, status_code
