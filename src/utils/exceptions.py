#!/usr/bin/env python3
"""
异常处理工具模块

提供统一的异常处理功能
"""

import sys
import traceback
import functools
from typing import Any, Dict, List, Optional, Type, Union, Callable
from datetime import datetime
import inspect

from .logger import get_logger
from .response import (
    create_response_from_exception,
    ApiResponse,
    ResponseStatus
)
from ..core.exceptions import (
    BaseAPIException,
    ValidationException,
    AuthenticationException,
    AuthorizationException,
    NotFoundException,
    RateLimitException,
    BusinessLogicException
)


# =============================================================================
# 配置和全局变量
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# 异常信息类
# =============================================================================

class ExceptionInfo:
    """异常信息类"""
    
    def __init__(
        self,
        exception: Exception,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.exception = exception
        self.exception_type = type(exception).__name__
        self.exception_message = str(exception)
        self.request_id = request_id
        self.user_id = user_id
        self.context = context or {}
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()
        
        # 获取异常发生的位置
        tb = exception.__traceback__
        if tb:
            while tb.tb_next:
                tb = tb.tb_next
            self.filename = tb.tb_frame.f_code.co_filename
            self.line_number = tb.tb_lineno
            self.function_name = tb.tb_frame.f_code.co_name
        else:
            self.filename = None
            self.line_number = None
            self.function_name = None
    
    def to_dict(self, include_traceback: bool = False) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'exception_type': self.exception_type,
            'exception_message': self.exception_message,
            'timestamp': self.timestamp.isoformat(),
            'request_id': self.request_id,
            'user_id': self.user_id,
            'context': self.context
        }
        
        if self.filename:
            result['location'] = {
                'filename': self.filename,
                'line_number': self.line_number,
                'function_name': self.function_name
            }
        
        if include_traceback:
            result['traceback'] = self.traceback
        
        # 移除空值
        return {k: v for k, v in result.items() if v is not None}


# =============================================================================
# 异常处理器
# =============================================================================

class ExceptionHandler:
    """异常处理器"""
    
    def __init__(
        self,
        log_exceptions: bool = True,
        include_traceback: bool = False,
        notify_on_error: bool = False
    ):
        self.log_exceptions = log_exceptions
        self.include_traceback = include_traceback
        self.notify_on_error = notify_on_error
        self.exception_handlers = {}
        self.global_handlers = []
    
    def register_handler(
        self,
        exception_type: Type[Exception],
        handler: Callable[[Exception, ExceptionInfo], Any]
    ):
        """注册异常处理器"""
        self.exception_handlers[exception_type] = handler
    
    def register_global_handler(
        self,
        handler: Callable[[Exception, ExceptionInfo], Any]
    ):
        """注册全局异常处理器"""
        self.global_handlers.append(handler)
    
    def handle_exception(
        self,
        exception: Exception,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ApiResponse:
        """处理异常"""
        # 创建异常信息
        exception_info = ExceptionInfo(
            exception=exception,
            request_id=request_id,
            user_id=user_id,
            context=context
        )
        
        # 记录异常日志
        if self.log_exceptions:
            self._log_exception(exception_info)
        
        # 查找特定的异常处理器
        handler = None
        for exc_type, exc_handler in self.exception_handlers.items():
            if isinstance(exception, exc_type):
                handler = exc_handler
                break
        
        # 执行特定处理器
        if handler:
            try:
                result = handler(exception, exception_info)
                if isinstance(result, ApiResponse):
                    return result
            except Exception as handler_error:
                logger.error(f"Error in exception handler: {handler_error}")
        
        # 执行全局处理器
        for global_handler in self.global_handlers:
            try:
                global_handler(exception, exception_info)
            except Exception as handler_error:
                logger.error(f"Error in global exception handler: {handler_error}")
        
        # 发送通知（如果启用）
        if self.notify_on_error:
            self._notify_error(exception_info)
        
        # 创建标准响应
        return create_response_from_exception(
            exception,
            request_id=request_id,
            include_traceback=self.include_traceback
        )
    
    def _log_exception(self, exception_info: ExceptionInfo):
        """记录异常日志"""
        log_data = exception_info.to_dict(include_traceback=True)
        
        if isinstance(exception_info.exception, BaseAPIException):
            # API异常使用警告级别
            logger.warning(f"API Exception: {exception_info.exception_message}", extra=log_data)
        else:
            # 其他异常使用错误级别
            logger.error(f"Unhandled Exception: {exception_info.exception_message}", extra=log_data)
    
    def _notify_error(self, exception_info: ExceptionInfo):
        """发送错误通知"""
        try:
            # 这里可以集成邮件、短信、Slack等通知服务
            # 示例：发送到监控系统
            logger.info(f"Error notification sent for: {exception_info.exception_type}")
        except Exception as notify_error:
            logger.error(f"Failed to send error notification: {notify_error}")


# 全局异常处理器实例
default_exception_handler = ExceptionHandler()


# =============================================================================
# 异常处理装饰器
# =============================================================================

def handle_exceptions(
    handler: Optional[ExceptionHandler] = None,
    return_response: bool = True,
    log_exceptions: bool = True,
    reraise: bool = False,
    default_return: Any = None
):
    """异常处理装饰器"""
    if handler is None:
        handler = default_exception_handler
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                if inspect.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                # 提取请求上下文
                request_id = kwargs.get('request_id')
                user_id = kwargs.get('user_id')
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],  # 限制长度
                    'kwargs': {k: str(v)[:100] for k, v in kwargs.items()}
                }
                
                if log_exceptions:
                    response = handler.handle_exception(
                        exception=e,
                        request_id=request_id,
                        user_id=user_id,
                        context=context
                    )
                    
                    if return_response:
                        return response
                
                if reraise:
                    raise
                
                return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 提取请求上下文
                request_id = kwargs.get('request_id')
                user_id = kwargs.get('user_id')
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],  # 限制长度
                    'kwargs': {k: str(v)[:100] for k, v in kwargs.items()}
                }
                
                if log_exceptions:
                    response = handler.handle_exception(
                        exception=e,
                        request_id=request_id,
                        user_id=user_id,
                        context=context
                    )
                    
                    if return_response:
                        return response
                
                if reraise:
                    raise
                
                return default_return
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def catch_and_log(
    exceptions: Union[Type[Exception], tuple] = Exception,
    log_level: str = "ERROR",
    reraise: bool = True,
    default_return: Any = None
):
    """捕获并记录异常的装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                if inspect.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except exceptions as e:
                log_message = f"Exception in {func.__name__}: {e}"
                getattr(logger, log_level.lower())(log_message, exc_info=True)
                
                if reraise:
                    raise
                
                return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                log_message = f"Exception in {func.__name__}: {e}"
                getattr(logger, log_level.lower())(log_message, exc_info=True)
                
                if reraise:
                    raise
                
                return default_return
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# =============================================================================
# 异常转换器
# =============================================================================

class ExceptionConverter:
    """异常转换器"""
    
    def __init__(self):
        self.converters = {}
    
    def register_converter(
        self,
        source_exception: Type[Exception],
        target_exception: Type[Exception],
        message_mapper: Optional[Callable[[Exception], str]] = None
    ):
        """注册异常转换器"""
        self.converters[source_exception] = {
            'target': target_exception,
            'message_mapper': message_mapper
        }
    
    def convert_exception(self, exception: Exception) -> Exception:
        """转换异常"""
        for source_type, converter in self.converters.items():
            if isinstance(exception, source_type):
                target_type = converter['target']
                message_mapper = converter['message_mapper']
                
                if message_mapper:
                    message = message_mapper(exception)
                else:
                    message = str(exception)
                
                return target_type(message)
        
        return exception


# 默认异常转换器
default_converter = ExceptionConverter()

# 注册常见的异常转换
default_converter.register_converter(
    ValueError,
    ValidationException,
    lambda e: f"数据验证失败: {str(e)}"
)

default_converter.register_converter(
    KeyError,
    ValidationException,
    lambda e: f"缺少必需的字段: {str(e)}"
)

default_converter.register_converter(
    FileNotFoundError,
    NotFoundException,
    lambda e: f"文件未找到: {str(e)}"
)


# =============================================================================
# 异常聚合器
# =============================================================================

class ExceptionAggregator:
    """异常聚合器"""
    
    def __init__(self):
        self.exceptions = []
    
    def add_exception(self, exception: Exception, context: Optional[str] = None):
        """添加异常"""
        self.exceptions.append({
            'exception': exception,
            'context': context,
            'timestamp': datetime.now()
        })
    
    def has_exceptions(self) -> bool:
        """是否有异常"""
        return len(self.exceptions) > 0
    
    def get_exceptions(self) -> List[Dict[str, Any]]:
        """获取所有异常"""
        return self.exceptions.copy()
    
    def clear(self):
        """清空异常"""
        self.exceptions.clear()
    
    def raise_if_any(self, message: str = "发生多个错误"):
        """如果有异常则抛出聚合异常"""
        if self.has_exceptions():
            error_messages = []
            for exc_info in self.exceptions:
                context = exc_info['context']
                exception = exc_info['exception']
                if context:
                    error_messages.append(f"{context}: {str(exception)}")
                else:
                    error_messages.append(str(exception))
            
            full_message = f"{message}: {'; '.join(error_messages)}"
            raise BusinessLogicException(full_message)


# =============================================================================
# 工具函数
# =============================================================================

def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """安全执行函数"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Error executing {func.__name__}: {e}", exc_info=True)
        return default_return


async def safe_execute_async(
    func: Callable,
    *args,
    default_return: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """安全执行异步函数"""
    try:
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Error executing {func.__name__}: {e}", exc_info=True)
        return default_return


def format_exception_for_user(exception: Exception) -> str:
    """格式化异常信息给用户看"""
    if isinstance(exception, BaseAPIException):
        return exception.message or str(exception)
    elif isinstance(exception, ValidationException):
        return f"数据验证失败: {str(exception)}"
    elif isinstance(exception, AuthenticationException):
        return "认证失败，请重新登录"
    elif isinstance(exception, AuthorizationException):
        return "权限不足，无法执行此操作"
    elif isinstance(exception, NotFoundException):
        return "请求的资源不存在"
    elif isinstance(exception, RateLimitException):
        return "请求过于频繁，请稍后再试"
    else:
        return "系统繁忙，请稍后再试"


def get_exception_details(exception: Exception) -> Dict[str, Any]:
    """获取异常详细信息"""
    return {
        'type': type(exception).__name__,
        'message': str(exception),
        'module': getattr(exception, '__module__', None),
        'traceback': traceback.format_exc(),
        'args': exception.args
    }


def is_client_error(exception: Exception) -> bool:
    """判断是否为客户端错误"""
    client_error_types = (
        ValidationException,
        AuthenticationException,
        AuthorizationException,
        NotFoundException,
        RateLimitException
    )
    return isinstance(exception, client_error_types)


def is_server_error(exception: Exception) -> bool:
    """判断是否为服务器错误"""
    return not is_client_error(exception)


def create_error_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    method: Optional[str] = None,
    **extra_context
) -> Dict[str, Any]:
    """创建错误上下文"""
    context = {
        'request_id': request_id,
        'user_id': user_id,
        'endpoint': endpoint,
        'method': method,
        'timestamp': datetime.now().isoformat()
    }
    
    context.update(extra_context)
    
    # 移除空值
    return {k: v for k, v in context.items() if v is not None}


# =============================================================================
# 异常监控
# =============================================================================

class ExceptionMonitor:
    """异常监控器"""
    
    def __init__(self):
        self.exception_counts = {}
        self.exception_history = []
        self.max_history = 1000
    
    def record_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None):
        """记录异常"""
        exception_type = type(exception).__name__
        
        # 更新计数
        if exception_type not in self.exception_counts:
            self.exception_counts[exception_type] = 0
        self.exception_counts[exception_type] += 1
        
        # 添加到历史记录
        self.exception_history.append({
            'type': exception_type,
            'message': str(exception),
            'timestamp': datetime.now(),
            'context': context
        })
        
        # 限制历史记录大小
        if len(self.exception_history) > self.max_history:
            self.exception_history = self.exception_history[-self.max_history:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取异常统计信息"""
        total_exceptions = sum(self.exception_counts.values())
        
        return {
            'total_exceptions': total_exceptions,
            'exception_counts': self.exception_counts.copy(),
            'most_common': sorted(
                self.exception_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'recent_exceptions': self.exception_history[-10:]
        }
    
    def clear_statistics(self):
        """清空统计信息"""
        self.exception_counts.clear()
        self.exception_history.clear()


# 全局异常监控器
exception_monitor = ExceptionMonitor()


# =============================================================================
# 注册默认异常处理器
# =============================================================================

def setup_default_handlers():
    """设置默认异常处理器"""
    
    def validation_error_handler(exception: ValidationException, info: ExceptionInfo) -> ApiResponse:
        """验证错误处理器"""
        from .response import validation_error_response
        return validation_error_response(
            message=str(exception),
            request_id=info.request_id
        )
    
    def not_found_handler(exception: NotFoundException, info: ExceptionInfo) -> ApiResponse:
        """资源未找到处理器"""
        from .response import not_found_response
        return not_found_response(
            message=str(exception),
            request_id=info.request_id
        )
    
    def auth_error_handler(exception: AuthenticationException, info: ExceptionInfo) -> ApiResponse:
        """认证错误处理器"""
        from .response import unauthorized_response
        return unauthorized_response(
            message=str(exception),
            request_id=info.request_id
        )
    
    def authz_error_handler(exception: AuthorizationException, info: ExceptionInfo) -> ApiResponse:
        """授权错误处理器"""
        from .response import forbidden_response
        return forbidden_response(
            message=str(exception),
            request_id=info.request_id
        )
    
    def rate_limit_handler(exception: RateLimitException, info: ExceptionInfo) -> ApiResponse:
        """速率限制处理器"""
        from .response import rate_limit_response
        return rate_limit_response(
            message=str(exception),
            request_id=info.request_id
        )
    
    def monitor_handler(exception: Exception, info: ExceptionInfo):
        """监控处理器"""
        exception_monitor.record_exception(exception, info.context)
    
    # 注册处理器
    default_exception_handler.register_handler(ValidationException, validation_error_handler)
    default_exception_handler.register_handler(NotFoundException, not_found_handler)
    default_exception_handler.register_handler(AuthenticationException, auth_error_handler)
    default_exception_handler.register_handler(AuthorizationException, authz_error_handler)
    default_exception_handler.register_handler(RateLimitException, rate_limit_handler)
    
    # 注册全局监控处理器
    default_exception_handler.register_global_handler(monitor_handler)


# 初始化默认处理器
setup_default_handlers()