#!/usr/bin/env python3
"""
日志配置模块

提供统一的日志管理，支持：
1. 结构化日志
2. 多种输出格式
3. 日志轮转
4. 异步日志
5. 分布式追踪集成
6. 性能监控
"""

import asyncio
import json
import logging
import logging.config
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextvars import ContextVar

import structlog
from structlog.stdlib import LoggerFactory
from pythonjsonlogger import jsonlogger
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .config import get_settings


# 上下文变量
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)


class StructuredFormatter(jsonlogger.JsonFormatter):
    """
    结构化日志格式化器
    
    将日志记录格式化为JSON格式，包含上下文信息
    """
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """添加自定义字段"""
        super().add_fields(log_record, record, message_dict)
        
        # 添加时间戳
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # 添加日志级别
        log_record['level'] = record.levelname
        
        # 添加模块信息
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # 添加进程和线程信息
        log_record['process_id'] = record.process
        log_record['thread_id'] = record.thread
        log_record['thread_name'] = record.threadName
        
        # 添加上下文信息
        if request_id_var.get():
            log_record['request_id'] = request_id_var.get()
        if user_id_var.get():
            log_record['user_id'] = user_id_var.get()
        if session_id_var.get():
            log_record['session_id'] = session_id_var.get()
        
        # 添加追踪信息
        span = trace.get_current_span()
        if span and span.is_recording():
            span_context = span.get_span_context()
            log_record['trace_id'] = format(span_context.trace_id, '032x')
            log_record['span_id'] = format(span_context.span_id, '016x')
        
        # 添加异常信息
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info),
            }


class ColoredFormatter(logging.Formatter):
    """
    彩色日志格式化器
    
    为控制台输出添加颜色
    """
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m',       # 重置
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 添加颜色
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # 格式化消息
        formatted = super().format(record)
        
        # 添加上下文信息
        context_parts = []
        if request_id_var.get():
            context_parts.append(f"req:{request_id_var.get()[:8]}")
        if user_id_var.get():
            context_parts.append(f"user:{user_id_var.get()}")
        
        if context_parts:
            context = f"[{' '.join(context_parts)}] "
        else:
            context = ""
        
        return f"{color}{context}{formatted}{reset}"


class AsyncFileHandler(logging.Handler):
    """
    异步文件处理器
    
    异步写入日志文件，避免阻塞主线程
    """
    
    def __init__(self, filename: str, mode: str = 'a', encoding: str = 'utf-8'):
        super().__init__()
        self.filename = filename
        self.mode = mode
        self.encoding = encoding
        self._queue = asyncio.Queue()
        self._task = None
        self._running = False
    
    def emit(self, record: logging.LogRecord) -> None:
        """发送日志记录"""
        if self._running:
            try:
                formatted = self.format(record)
                asyncio.create_task(self._queue.put(formatted))
            except Exception:
                self.handleError(record)
    
    async def start(self) -> None:
        """启动异步写入任务"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._write_loop())
    
    async def stop(self) -> None:
        """停止异步写入任务"""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            await self._task
    
    async def _write_loop(self) -> None:
        """异步写入循环"""
        try:
            with open(self.filename, self.mode, encoding=self.encoding) as f:
                while self._running:
                    try:
                        # 等待日志记录
                        formatted = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                        f.write(formatted + '\n')
                        f.flush()
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"Error writing log: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error opening log file {self.filename}: {e}", file=sys.stderr)


class LoggingManager:
    """
    日志管理器
    
    负责配置和管理应用日志
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._async_handlers = []
        self._configured = False
    
    def configure(self) -> None:
        """
        配置日志系统
        """
        if self._configured:
            return
        
        # 创建日志目录
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 配置structlog
        self._configure_structlog()
        
        # 配置标准库日志
        self._configure_stdlib_logging()
        
        self._configured = True
    
    def _configure_structlog(self) -> None:
        """
        配置structlog
        """
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        
        # 开发环境使用彩色输出
        if self.settings.app_environment == "development":
            processors.append(structlog.dev.ConsoleRenderer())
        else:
            processors.append(structlog.processors.JSONRenderer())
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    def _configure_stdlib_logging(self) -> None:
        """
        配置标准库日志
        """
        # 日志配置
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'detailed': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s (%(filename)s:%(lineno)d)',
                    'datefmt': '%Y-%m-%d %H:%M:%S',
                },
                'json': {
                    '()': StructuredFormatter,
                    'format': '%(asctime)s %(name)s %(levelname)s %(message)s',
                },
                'colored': {
                    '()': ColoredFormatter,
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    'datefmt': '%H:%M:%S',
                },
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': self.settings.app_log_level,
                    'formatter': 'colored' if self.settings.app_environment == 'development' else 'json',
                    'stream': 'ext://sys.stdout',
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'json',
                    'filename': 'logs/app.log',
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'encoding': 'utf-8',
                },
                'error_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'ERROR',
                    'formatter': 'json',
                    'filename': 'logs/error.log',
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'encoding': 'utf-8',
                },
            },
            'loggers': {
                '': {  # root logger
                    'level': self.settings.app_log_level,
                    'handlers': ['console', 'file', 'error_file'],
                    'propagate': False,
                },
                'uvicorn': {
                    'level': 'INFO',
                    'handlers': ['console', 'file'],
                    'propagate': False,
                },
                'uvicorn.access': {
                    'level': 'INFO',
                    'handlers': ['file'],
                    'propagate': False,
                },
                'sqlalchemy': {
                    'level': 'WARNING',
                    'handlers': ['file'],
                    'propagate': False,
                },
                'alembic': {
                    'level': 'INFO',
                    'handlers': ['console', 'file'],
                    'propagate': False,
                },
                'celery': {
                    'level': 'INFO',
                    'handlers': ['console', 'file'],
                    'propagate': False,
                },
                'redis': {
                    'level': 'WARNING',
                    'handlers': ['file'],
                    'propagate': False,
                },
            },
        }
        
        # 生产环境添加更多处理器
        if self.settings.app_environment == 'production':
            # 添加系统日志处理器
            if sys.platform != 'win32':
                config['handlers']['syslog'] = {
                    'class': 'logging.handlers.SysLogHandler',
                    'level': 'INFO',
                    'formatter': 'json',
                    'address': '/dev/log',
                }
                config['loggers']['']['handlers'].append('syslog')
        
        # 应用配置
        logging.config.dictConfig(config)
    
    async def start_async_handlers(self) -> None:
        """
        启动异步处理器
        """
        for handler in self._async_handlers:
            await handler.start()
    
    async def stop_async_handlers(self) -> None:
        """
        停止异步处理器
        """
        for handler in self._async_handlers:
            await handler.stop()
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        获取日志记录器
        
        Args:
            name: 记录器名称
        
        Returns:
            logging.Logger: 日志记录器
        """
        if not self._configured:
            self.configure()
        
        return logging.getLogger(name)
    
    def get_structlog_logger(self, name: str) -> structlog.BoundLogger:
        """
        获取structlog记录器
        
        Args:
            name: 记录器名称
        
        Returns:
            structlog.BoundLogger: structlog记录器
        """
        if not self._configured:
            self.configure()
        
        return structlog.get_logger(name)


# 全局日志管理器实例
logging_manager = LoggingManager()


# 便捷函数
def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 记录器名称
    
    Returns:
        logging.Logger: 日志记录器
    """
    return logging_manager.get_logger(name)


def get_structlog_logger(name: str) -> structlog.BoundLogger:
    """
    获取structlog记录器
    
    Args:
        name: 记录器名称
    
    Returns:
        structlog.BoundLogger: structlog记录器
    """
    return logging_manager.get_structlog_logger(name)


# 上下文管理器
class LogContext:
    """
    日志上下文管理器
    
    用于设置请求级别的日志上下文
    """
    
    def __init__(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.request_id = request_id
        self.user_id = user_id
        self.session_id = session_id
        self._tokens = []
    
    def __enter__(self):
        if self.request_id:
            self._tokens.append(request_id_var.set(self.request_id))
        if self.user_id:
            self._tokens.append(user_id_var.set(self.user_id))
        if self.session_id:
            self._tokens.append(session_id_var.set(self.session_id))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for token in reversed(self._tokens):
            token.var.reset(token)


# 装饰器
def log_function_call(logger: Optional[logging.Logger] = None, level: str = "DEBUG"):
    """
    函数调用日志装饰器
    
    Args:
        logger: 日志记录器
        level: 日志级别
    
    Returns:
        装饰器函数
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            
            # 记录函数调用开始
            logger.log(
                getattr(logging, level.upper()),
                f"Calling function {func.__name__}",
                extra={
                    'function': func.__name__,
                    'module': func.__module__,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs),
                    'start_time': start_time.isoformat(),
                }
            )
            
            try:
                result = func(*args, **kwargs)
                
                # 记录函数调用成功
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                
                logger.log(
                    getattr(logging, level.upper()),
                    f"Function {func.__name__} completed successfully",
                    extra={
                        'function': func.__name__,
                        'module': func.__module__,
                        'duration': duration,
                        'end_time': end_time.isoformat(),
                        'success': True,
                    }
                )
                
                return result
                
            except Exception as e:
                # 记录函数调用失败
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                
                logger.error(
                    f"Function {func.__name__} failed with error: {str(e)}",
                    extra={
                        'function': func.__name__,
                        'module': func.__module__,
                        'duration': duration,
                        'end_time': end_time.isoformat(),
                        'success': False,
                        'error': str(e),
                        'error_type': type(e).__name__,
                    },
                    exc_info=True
                )
                
                raise
        
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            
            # 记录函数调用开始
            logger.log(
                getattr(logging, level.upper()),
                f"Calling async function {func.__name__}",
                extra={
                    'function': func.__name__,
                    'module': func.__module__,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs),
                    'start_time': start_time.isoformat(),
                    'async': True,
                }
            )
            
            try:
                result = await func(*args, **kwargs)
                
                # 记录函数调用成功
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                
                logger.log(
                    getattr(logging, level.upper()),
                    f"Async function {func.__name__} completed successfully",
                    extra={
                        'function': func.__name__,
                        'module': func.__module__,
                        'duration': duration,
                        'end_time': end_time.isoformat(),
                        'success': True,
                        'async': True,
                    }
                )
                
                return result
                
            except Exception as e:
                # 记录函数调用失败
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                
                logger.error(
                    f"Async function {func.__name__} failed with error: {str(e)}",
                    extra={
                        'function': func.__name__,
                        'module': func.__module__,
                        'duration': duration,
                        'end_time': end_time.isoformat(),
                        'success': False,
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'async': True,
                    },
                    exc_info=True
                )
                
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


# 初始化函数
def init_logging() -> None:
    """
    初始化日志系统
    """
    logging_manager.configure()


async def start_async_logging() -> None:
    """
    启动异步日志处理器
    """
    await logging_manager.start_async_handlers()


async def stop_async_logging() -> None:
    """
    停止异步日志处理器
    """
    await logging_manager.stop_async_handlers()