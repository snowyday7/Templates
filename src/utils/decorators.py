#!/usr/bin/env python3
"""
装饰器工具模块

提供各种常用的装饰器功能
"""

import time
import asyncio
import functools
import inspect
from typing import Any, Callable, Dict, Optional, Union, List
from datetime import datetime, timedelta
import hashlib
import json
from collections import defaultdict
from threading import Lock

from .logger import get_logger
# 定义本地异常类
class RateLimitException(Exception):
    """速率限制异常"""
    pass

class AuthenticationException(Exception):
    """认证异常"""
    pass

class AuthorizationException(Exception):
    """授权异常"""
    pass

class ValidationException(Exception):
    """验证异常"""
    pass


# =============================================================================
# 配置和全局变量
# =============================================================================

logger = get_logger(__name__)

# 速率限制存储
rate_limit_storage = defaultdict(list)
rate_limit_lock = Lock()

# 缓存存储
cache_storage = {}
cache_lock = Lock()


# =============================================================================
# 重试装饰器
# =============================================================================

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff: 退避倍数
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    if inspect.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:  # 不是最后一次尝试
                        if on_retry:
                            try:
                                if inspect.iscoroutinefunction(on_retry):
                                    await on_retry(attempt + 1, e, current_delay)
                                else:
                                    on_retry(attempt + 1, e, current_delay)
                            except Exception as callback_error:
                                logger.error(f"Error in retry callback: {callback_error}")
                        
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay} seconds..."
                        )
                        
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:  # 不是最后一次尝试
                        if on_retry:
                            try:
                                on_retry(attempt + 1, e, current_delay)
                            except Exception as callback_error:
                                logger.error(f"Error in retry callback: {callback_error}")
                        
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay} seconds..."
                        )
                        
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            
            raise last_exception
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# =============================================================================
# 速率限制装饰器
# =============================================================================

def rate_limit(
    max_calls: int = 100,
    time_window: int = 60,
    key_func: Optional[Callable] = None,
    error_message: str = "请求过于频繁，请稍后再试"
):
    """
    速率限制装饰器
    
    Args:
        max_calls: 时间窗口内最大调用次数
        time_window: 时间窗口（秒）
        key_func: 生成限制键的函数
        error_message: 超限时的错误消息
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 生成限制键
            if key_func:
                try:
                    if inspect.iscoroutinefunction(key_func):
                        limit_key = await key_func(*args, **kwargs)
                    else:
                        limit_key = key_func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error generating rate limit key: {e}")
                    limit_key = f"{func.__name__}:default"
            else:
                limit_key = f"{func.__name__}:default"
            
            current_time = time.time()
            
            with rate_limit_lock:
                # 清理过期的调用记录
                cutoff_time = current_time - time_window
                rate_limit_storage[limit_key] = [
                    call_time for call_time in rate_limit_storage[limit_key]
                    if call_time > cutoff_time
                ]
                
                # 检查是否超过限制
                if len(rate_limit_storage[limit_key]) >= max_calls:
                    logger.warning(f"Rate limit exceeded for key: {limit_key}")
                    raise RateLimitException(error_message)
                
                # 记录当前调用
                rate_limit_storage[limit_key].append(current_time)
            
            # 执行原函数
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 生成限制键
            if key_func:
                try:
                    limit_key = key_func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error generating rate limit key: {e}")
                    limit_key = f"{func.__name__}:default"
            else:
                limit_key = f"{func.__name__}:default"
            
            current_time = time.time()
            
            with rate_limit_lock:
                # 清理过期的调用记录
                cutoff_time = current_time - time_window
                rate_limit_storage[limit_key] = [
                    call_time for call_time in rate_limit_storage[limit_key]
                    if call_time > cutoff_time
                ]
                
                # 检查是否超过限制
                if len(rate_limit_storage[limit_key]) >= max_calls:
                    logger.warning(f"Rate limit exceeded for key: {limit_key}")
                    raise RateLimitException(error_message)
                
                # 记录当前调用
                rate_limit_storage[limit_key].append(current_time)
            
            # 执行原函数
            return func(*args, **kwargs)
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# =============================================================================
# 缓存装饰器
# =============================================================================

def cache_result(
    ttl: int = 300,
    key_func: Optional[Callable] = None,
    ignore_args: Optional[List[str]] = None
):
    """
    结果缓存装饰器
    
    Args:
        ttl: 缓存生存时间（秒）
        key_func: 生成缓存键的函数
        ignore_args: 忽略的参数名列表
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                try:
                    if inspect.iscoroutinefunction(key_func):
                        cache_key = await key_func(*args, **kwargs)
                    else:
                        cache_key = key_func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error generating cache key: {e}")
                    cache_key = _generate_default_cache_key(func, args, kwargs, ignore_args)
            else:
                cache_key = _generate_default_cache_key(func, args, kwargs, ignore_args)
            
            current_time = time.time()
            
            with cache_lock:
                # 检查缓存
                if cache_key in cache_storage:
                    cached_data, cached_time = cache_storage[cache_key]
                    if current_time - cached_time < ttl:
                        logger.debug(f"Cache hit for key: {cache_key}")
                        return cached_data
                    else:
                        # 缓存过期，删除
                        del cache_storage[cache_key]
            
            # 执行原函数
            if inspect.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # 存储到缓存
            with cache_lock:
                cache_storage[cache_key] = (result, current_time)
                logger.debug(f"Cache stored for key: {cache_key}")
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                try:
                    cache_key = key_func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error generating cache key: {e}")
                    cache_key = _generate_default_cache_key(func, args, kwargs, ignore_args)
            else:
                cache_key = _generate_default_cache_key(func, args, kwargs, ignore_args)
            
            current_time = time.time()
            
            with cache_lock:
                # 检查缓存
                if cache_key in cache_storage:
                    cached_data, cached_time = cache_storage[cache_key]
                    if current_time - cached_time < ttl:
                        logger.debug(f"Cache hit for key: {cache_key}")
                        return cached_data
                    else:
                        # 缓存过期，删除
                        del cache_storage[cache_key]
            
            # 执行原函数
            result = func(*args, **kwargs)
            
            # 存储到缓存
            with cache_lock:
                cache_storage[cache_key] = (result, current_time)
                logger.debug(f"Cache stored for key: {cache_key}")
            
            return result
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def _generate_default_cache_key(
    func: Callable,
    args: tuple,
    kwargs: dict,
    ignore_args: Optional[List[str]] = None
) -> str:
    """
    生成默认的缓存键
    """
    try:
        # 获取函数签名
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # 过滤忽略的参数
        if ignore_args:
            filtered_args = {
                k: v for k, v in bound_args.arguments.items()
                if k not in ignore_args
            }
        else:
            filtered_args = bound_args.arguments
        
        # 生成键
        key_data = {
            'function': f"{func.__module__}.{func.__name__}",
            'args': filtered_args
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
        
    except Exception as e:
        logger.error(f"Error generating default cache key: {e}")
        return f"{func.__name__}:{hash(args)}:{hash(tuple(sorted(kwargs.items())))}"


# =============================================================================
# 执行时间记录装饰器
# =============================================================================

def log_execution_time(
    log_level: str = "INFO",
    include_args: bool = False,
    threshold: Optional[float] = None
):
    """
    记录函数执行时间的装饰器
    
    Args:
        log_level: 日志级别
        include_args: 是否包含参数信息
        threshold: 只记录超过阈值的执行时间（秒）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                if inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                
                # 检查阈值
                if threshold is None or execution_time >= threshold:
                    log_message = f"{func.__name__} executed in {execution_time:.4f} seconds"
                    
                    if include_args:
                        log_message += f" with args: {args}, kwargs: {kwargs}"
                    
                    getattr(logger, log_level.lower())(log_message)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"{func.__name__} failed after {execution_time:.4f} seconds: {e}"
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # 检查阈值
                if threshold is None or execution_time >= threshold:
                    log_message = f"{func.__name__} executed in {execution_time:.4f} seconds"
                    
                    if include_args:
                        log_message += f" with args: {args}, kwargs: {kwargs}"
                    
                    getattr(logger, log_level.lower())(log_message)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"{func.__name__} failed after {execution_time:.4f} seconds: {e}"
                )
                raise
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# =============================================================================
# 输入验证装饰器
# =============================================================================

def validate_input(
    validators: Dict[str, Callable],
    raise_on_error: bool = True
):
    """
    输入验证装饰器
    
    Args:
        validators: 验证器映射 {参数名: 验证函数}
        raise_on_error: 验证失败时是否抛出异常
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 获取函数签名
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            validation_errors = {}
            
            # 验证参数
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    param_value = bound_args.arguments[param_name]
                    
                    try:
                        if inspect.iscoroutinefunction(validator):
                            is_valid, error_message = await validator(param_value)
                        else:
                            is_valid, error_message = validator(param_value)
                        
                        if not is_valid:
                            validation_errors[param_name] = error_message
                            
                    except Exception as e:
                        logger.error(f"Error validating parameter {param_name}: {e}")
                        validation_errors[param_name] = f"验证失败: {str(e)}"
            
            # 处理验证错误
            if validation_errors:
                if raise_on_error:
                    raise ValidationException(f"参数验证失败: {validation_errors}")
                else:
                    logger.warning(f"Validation errors in {func.__name__}: {validation_errors}")
            
            # 执行原函数
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 获取函数签名
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            validation_errors = {}
            
            # 验证参数
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    param_value = bound_args.arguments[param_name]
                    
                    try:
                        is_valid, error_message = validator(param_value)
                        
                        if not is_valid:
                            validation_errors[param_name] = error_message
                            
                    except Exception as e:
                        logger.error(f"Error validating parameter {param_name}: {e}")
                        validation_errors[param_name] = f"验证失败: {str(e)}"
            
            # 处理验证错误
            if validation_errors:
                if raise_on_error:
                    raise ValidationException(f"参数验证失败: {validation_errors}")
                else:
                    logger.warning(f"Validation errors in {func.__name__}: {validation_errors}")
            
            # 执行原函数
            return func(*args, **kwargs)
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# =============================================================================
# 认证和授权装饰器
# =============================================================================

def require_auth(auth_func: Optional[Callable] = None):
    """
    要求认证的装饰器
    
    Args:
        auth_func: 自定义认证函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 默认认证逻辑（可以根据实际需求修改）
            if auth_func:
                try:
                    if inspect.iscoroutinefunction(auth_func):
                        is_authenticated = await auth_func(*args, **kwargs)
                    else:
                        is_authenticated = auth_func(*args, **kwargs)
                    
                    if not is_authenticated:
                        raise AuthenticationException("认证失败")
                        
                except Exception as e:
                    if isinstance(e, AuthenticationException):
                        raise
                    else:
                        logger.error(f"Error in authentication: {e}")
                        raise AuthenticationException("认证过程中发生错误")
            else:
                # 默认认证逻辑：检查是否有用户信息
                # 这里需要根据实际的认证机制来实现
                logger.warning("No authentication function provided")
            
            # 执行原函数
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 默认认证逻辑
            if auth_func:
                try:
                    is_authenticated = auth_func(*args, **kwargs)
                    
                    if not is_authenticated:
                        raise AuthenticationException("认证失败")
                        
                except Exception as e:
                    if isinstance(e, AuthenticationException):
                        raise
                    else:
                        logger.error(f"Error in authentication: {e}")
                        raise AuthenticationException("认证过程中发生错误")
            else:
                logger.warning("No authentication function provided")
            
            # 执行原函数
            return func(*args, **kwargs)
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def require_permission(
    required_permissions: Union[str, List[str]],
    permission_check_func: Optional[Callable] = None
):
    """
    要求权限的装饰器
    
    Args:
        required_permissions: 所需权限
        permission_check_func: 自定义权限检查函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 标准化权限列表
            if isinstance(required_permissions, str):
                permissions = [required_permissions]
            else:
                permissions = required_permissions
            
            # 权限检查
            if permission_check_func:
                try:
                    if inspect.iscoroutinefunction(permission_check_func):
                        has_permission = await permission_check_func(permissions, *args, **kwargs)
                    else:
                        has_permission = permission_check_func(permissions, *args, **kwargs)
                    
                    if not has_permission:
                        raise AuthorizationException(f"缺少所需权限: {permissions}")
                        
                except Exception as e:
                    if isinstance(e, AuthorizationException):
                        raise
                    else:
                        logger.error(f"Error in permission check: {e}")
                        raise AuthorizationException("权限检查过程中发生错误")
            else:
                # 默认权限检查逻辑
                logger.warning(f"No permission check function provided for permissions: {permissions}")
            
            # 执行原函数
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 标准化权限列表
            if isinstance(required_permissions, str):
                permissions = [required_permissions]
            else:
                permissions = required_permissions
            
            # 权限检查
            if permission_check_func:
                try:
                    has_permission = permission_check_func(permissions, *args, **kwargs)
                    
                    if not has_permission:
                        raise AuthorizationException(f"缺少所需权限: {permissions}")
                        
                except Exception as e:
                    if isinstance(e, AuthorizationException):
                        raise
                    else:
                        logger.error(f"Error in permission check: {e}")
                        raise AuthorizationException("权限检查过程中发生错误")
            else:
                logger.warning(f"No permission check function provided for permissions: {permissions}")
            
            # 执行原函数
            return func(*args, **kwargs)
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# =============================================================================
# 异常处理装饰器
# =============================================================================

def handle_exceptions(
    exceptions: Union[Exception, tuple] = Exception,
    default_return: Any = None,
    log_error: bool = True,
    reraise: bool = False
):
    """
    异常处理装饰器
    
    Args:
        exceptions: 要处理的异常类型
        default_return: 异常时的默认返回值
        log_error: 是否记录错误日志
        reraise: 是否重新抛出异常
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                if inspect.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except exceptions as e:
                if log_error:
                    logger.error(f"Exception in {func.__name__}: {e}")
                
                if reraise:
                    raise
                else:
                    return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
                
            except exceptions as e:
                if log_error:
                    logger.error(f"Exception in {func.__name__}: {e}")
                
                if reraise:
                    raise
                else:
                    return default_return
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# =============================================================================
# 工具函数
# =============================================================================

def clear_cache(pattern: Optional[str] = None):
    """
    清理缓存
    
    Args:
        pattern: 缓存键模式（支持通配符）
    """
    with cache_lock:
        if pattern is None:
            cache_storage.clear()
            logger.info("All cache cleared")
        else:
            import fnmatch
            keys_to_remove = [
                key for key in cache_storage.keys()
                if fnmatch.fnmatch(key, pattern)
            ]
            
            for key in keys_to_remove:
                del cache_storage[key]
            
            logger.info(f"Cleared {len(keys_to_remove)} cache entries matching pattern: {pattern}")


def get_cache_stats() -> Dict[str, Any]:
    """
    获取缓存统计信息
    
    Returns:
        缓存统计信息
    """
    with cache_lock:
        current_time = time.time()
        total_entries = len(cache_storage)
        
        # 计算过期条目
        expired_entries = 0
        for cached_data, cached_time in cache_storage.values():
            # 这里假设默认TTL为300秒，实际应该记录每个条目的TTL
            if current_time - cached_time > 300:
                expired_entries += 1
        
        return {
            'total_entries': total_entries,
            'expired_entries': expired_entries,
            'active_entries': total_entries - expired_entries
        }


def get_rate_limit_stats() -> Dict[str, Any]:
    """
    获取速率限制统计信息
    
    Returns:
        速率限制统计信息
    """
    with rate_limit_lock:
        current_time = time.time()
        stats = {}
        
        for key, call_times in rate_limit_storage.items():
            # 清理过期记录（假设时间窗口为60秒）
            active_calls = [
                call_time for call_time in call_times
                if current_time - call_time <= 60
            ]
            
            stats[key] = {
                'total_calls': len(call_times),
                'active_calls': len(active_calls),
                'last_call': max(call_times) if call_times else None
            }
        
        return stats