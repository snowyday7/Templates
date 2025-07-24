#!/usr/bin/env python3
"""
日志工具模块

提供完整的日志记录功能
"""

import os
import sys
import json
import logging
import logging.handlers
from typing import Any, Dict, Optional, Union
from datetime import datetime
from pathlib import Path
import threading
from contextlib import contextmanager


# =============================================================================
# 配置和常量
# =============================================================================

# 默认日志配置
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# 日志文件配置
LOG_DIR = Path("logs")
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5

# 线程本地存储
local_data = threading.local()


# =============================================================================
# 自定义格式化器
# =============================================================================

class JsonFormatter(logging.Formatter):
    """JSON格式化器"""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为JSON"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if self.include_extra:
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'getMessage', 'exc_info',
                    'exc_text', 'stack_info'
                }
            }
            
            if extra_fields:
                log_data['extra'] = extra_fields
        
        # 添加上下文信息
        context = getattr(local_data, 'context', None)
        if context:
            log_data['context'] = context
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class ColoredFormatter(logging.Formatter):
    """彩色格式化器（用于控制台输出）"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录并添加颜色"""
        # 获取颜色
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # 格式化消息
        formatted = super().format(record)
        
        # 添加颜色
        return f"{color}{formatted}{reset}"


# =============================================================================
# 日志过滤器
# =============================================================================

class LevelFilter(logging.Filter):
    """级别过滤器"""
    
    def __init__(self, min_level: int, max_level: int):
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level
    
    def filter(self, record: logging.LogRecord) -> bool:
        """过滤日志记录"""
        return self.min_level <= record.levelno <= self.max_level


class SensitiveDataFilter(logging.Filter):
    """敏感数据过滤器"""
    
    SENSITIVE_FIELDS = {
        'password', 'passwd', 'pwd', 'secret', 'token', 'key',
        'authorization', 'auth', 'credential', 'api_key',
        'access_token', 'refresh_token', 'session_id'
    }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """过滤敏感数据"""
        # 检查消息中的敏感信息
        message = record.getMessage().lower()
        
        # 简单的敏感词检测
        for sensitive_field in self.SENSITIVE_FIELDS:
            if sensitive_field in message:
                # 替换敏感信息
                record.msg = self._mask_sensitive_data(str(record.msg))
                break
        
        return True
    
    def _mask_sensitive_data(self, text: str) -> str:
        """遮蔽敏感数据"""
        import re
        
        # 遮蔽可能的密码、令牌等
        patterns = [
            (r'(password|passwd|pwd)\s*[=:]\s*["\']?([^\s"\',}]+)', r'\1=***'),
            (r'(token|key|secret)\s*[=:]\s*["\']?([^\s"\',}]+)', r'\1=***'),
            (r'(authorization|auth)\s*[=:]\s*["\']?([^\s"\',}]+)', r'\1=***'),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text


# =============================================================================
# 日志处理器工厂
# =============================================================================

class LogHandlerFactory:
    """日志处理器工厂"""
    
    @staticmethod
    def create_console_handler(
        level: int = logging.INFO,
        use_colors: bool = True,
        format_string: Optional[str] = None
    ) -> logging.StreamHandler:
        """创建控制台处理器"""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        if format_string is None:
            format_string = DEFAULT_LOG_FORMAT
        
        if use_colors and sys.stdout.isatty():
            formatter = ColoredFormatter(format_string, DEFAULT_DATE_FORMAT)
        else:
            formatter = logging.Formatter(format_string, DEFAULT_DATE_FORMAT)
        
        handler.setFormatter(formatter)
        
        # 添加敏感数据过滤器
        handler.addFilter(SensitiveDataFilter())
        
        return handler
    
    @staticmethod
    def create_file_handler(
        filename: str,
        level: int = logging.INFO,
        max_size: int = MAX_LOG_SIZE,
        backup_count: int = BACKUP_COUNT,
        use_json: bool = False,
        format_string: Optional[str] = None
    ) -> logging.handlers.RotatingFileHandler:
        """创建文件处理器"""
        # 确保日志目录存在
        log_path = LOG_DIR / filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        handler.setLevel(level)
        
        if use_json:
            formatter = JsonFormatter()
        else:
            if format_string is None:
                format_string = DEFAULT_LOG_FORMAT
            formatter = logging.Formatter(format_string, DEFAULT_DATE_FORMAT)
        
        handler.setFormatter(formatter)
        
        # 添加敏感数据过滤器
        handler.addFilter(SensitiveDataFilter())
        
        return handler
    
    @staticmethod
    def create_error_file_handler(
        filename: str = "error.log",
        level: int = logging.ERROR,
        max_size: int = MAX_LOG_SIZE,
        backup_count: int = BACKUP_COUNT
    ) -> logging.handlers.RotatingFileHandler:
        """创建错误文件处理器"""
        handler = LogHandlerFactory.create_file_handler(
            filename=filename,
            level=level,
            max_size=max_size,
            backup_count=backup_count,
            use_json=True
        )
        
        # 只记录错误和严重错误
        handler.addFilter(LevelFilter(logging.ERROR, logging.CRITICAL))
        
        return handler
    
    @staticmethod
    def create_access_log_handler(
        filename: str = "access.log",
        level: int = logging.INFO,
        max_size: int = MAX_LOG_SIZE,
        backup_count: int = BACKUP_COUNT
    ) -> logging.handlers.RotatingFileHandler:
        """创建访问日志处理器"""
        return LogHandlerFactory.create_file_handler(
            filename=filename,
            level=level,
            max_size=max_size,
            backup_count=backup_count,
            use_json=True
        )


# =============================================================================
# 日志管理器
# =============================================================================

class LoggerManager:
    """日志管理器"""
    
    def __init__(self):
        self._loggers = {}
        self._configured = False
        self._default_level = DEFAULT_LOG_LEVEL
    
    def configure(
        self,
        level: Union[int, str] = DEFAULT_LOG_LEVEL,
        console_output: bool = True,
        file_output: bool = True,
        json_format: bool = False,
        log_dir: Optional[str] = None
    ):
        """配置日志系统"""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        self._default_level = level
        
        # 设置全局日志目录
        if log_dir:
            global LOG_DIR
            LOG_DIR = Path(log_dir)
        
        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 添加控制台处理器
        if console_output:
            console_handler = LogHandlerFactory.create_console_handler(
                level=level,
                use_colors=True
            )
            root_logger.addHandler(console_handler)
        
        # 添加文件处理器
        if file_output:
            # 主日志文件
            file_handler = LogHandlerFactory.create_file_handler(
                filename="app.log",
                level=level,
                use_json=json_format
            )
            root_logger.addHandler(file_handler)
            
            # 错误日志文件
            error_handler = LogHandlerFactory.create_error_file_handler()
            root_logger.addHandler(error_handler)
        
        self._configured = True
    
    def get_logger(
        self,
        name: str,
        level: Optional[int] = None
    ) -> logging.Logger:
        """获取日志器"""
        if not self._configured:
            self.configure()
        
        if name not in self._loggers:
            logger = logging.getLogger(name)
            
            if level is not None:
                logger.setLevel(level)
            elif logger.level == logging.NOTSET:
                logger.setLevel(self._default_level)
            
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def create_access_logger(self, name: str = "access") -> logging.Logger:
        """创建访问日志器"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # 防止重复添加处理器
        if not logger.handlers:
            handler = LogHandlerFactory.create_access_log_handler()
            logger.addHandler(handler)
            
            # 不传播到父日志器
            logger.propagate = False
        
        return logger
    
    def set_level(self, name: str, level: Union[int, str]):
        """设置日志器级别"""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        logger = self.get_logger(name)
        logger.setLevel(level)
    
    def add_handler(self, name: str, handler: logging.Handler):
        """为日志器添加处理器"""
        logger = self.get_logger(name)
        logger.addHandler(handler)
    
    def remove_handler(self, name: str, handler: logging.Handler):
        """从日志器移除处理器"""
        if name in self._loggers:
            self._loggers[name].removeHandler(handler)


# 全局日志管理器实例
logger_manager = LoggerManager()


# =============================================================================
# 便捷函数
# =============================================================================

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """获取日志器"""
    return logger_manager.get_logger(name, level)


def configure_logging(
    level: Union[int, str] = DEFAULT_LOG_LEVEL,
    console_output: bool = True,
    file_output: bool = True,
    json_format: bool = False,
    log_dir: Optional[str] = None
):
    """配置日志系统"""
    logger_manager.configure(
        level=level,
        console_output=console_output,
        file_output=file_output,
        json_format=json_format,
        log_dir=log_dir
    )


def set_log_level(name: str, level: Union[int, str]):
    """设置日志级别"""
    logger_manager.set_level(name, level)


# =============================================================================
# 上下文管理
# =============================================================================

@contextmanager
def log_context(**context):
    """日志上下文管理器"""
    # 保存原有上下文
    old_context = getattr(local_data, 'context', {})
    
    # 设置新上下文
    new_context = old_context.copy()
    new_context.update(context)
    local_data.context = new_context
    
    try:
        yield
    finally:
        # 恢复原有上下文
        local_data.context = old_context


def set_log_context(**context):
    """设置日志上下文"""
    if not hasattr(local_data, 'context'):
        local_data.context = {}
    
    local_data.context.update(context)


def clear_log_context():
    """清除日志上下文"""
    if hasattr(local_data, 'context'):
        local_data.context.clear()


def get_log_context() -> Dict[str, Any]:
    """获取日志上下文"""
    return getattr(local_data, 'context', {}).copy()


# =============================================================================
# 结构化日志记录
# =============================================================================

class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log(
        self,
        level: int,
        message: str,
        **fields
    ):
        """记录结构化日志"""
        # 创建额外字段
        extra = {'structured_fields': fields}
        
        # 记录日志
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **fields):
        """记录调试日志"""
        self.log(logging.DEBUG, message, **fields)
    
    def info(self, message: str, **fields):
        """记录信息日志"""
        self.log(logging.INFO, message, **fields)
    
    def warning(self, message: str, **fields):
        """记录警告日志"""
        self.log(logging.WARNING, message, **fields)
    
    def error(self, message: str, **fields):
        """记录错误日志"""
        self.log(logging.ERROR, message, **fields)
    
    def critical(self, message: str, **fields):
        """记录严重错误日志"""
        self.log(logging.CRITICAL, message, **fields)


def get_structured_logger(name: str) -> StructuredLogger:
    """获取结构化日志记录器"""
    logger = get_logger(name)
    return StructuredLogger(logger)


# =============================================================================
# 性能日志记录
# =============================================================================

class PerformanceLogger:
    """性能日志记录器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    @contextmanager
    def log_execution_time(
        self,
        operation: str,
        level: int = logging.INFO,
        threshold: Optional[float] = None,
        **context
    ):
        """记录执行时间"""
        import time
        
        start_time = time.time()
        
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            
            # 检查阈值
            if threshold is None or execution_time >= threshold:
                self.logger.log(
                    level,
                    f"Operation '{operation}' completed in {execution_time:.4f} seconds",
                    extra={
                        'operation': operation,
                        'execution_time': execution_time,
                        'performance_context': context
                    }
                )
    
    def log_memory_usage(
        self,
        operation: str,
        level: int = logging.DEBUG
    ):
        """记录内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.logger.log(
                level,
                f"Memory usage for '{operation}': {memory_info.rss / 1024 / 1024:.2f} MB",
                extra={
                    'operation': operation,
                    'memory_rss': memory_info.rss,
                    'memory_vms': memory_info.vms
                }
            )
        except ImportError:
            self.logger.warning("psutil not available for memory logging")


def get_performance_logger(name: str) -> PerformanceLogger:
    """获取性能日志记录器"""
    logger = get_logger(name)
    return PerformanceLogger(logger)


# =============================================================================
# 审计日志
# =============================================================================

class AuditLogger:
    """审计日志记录器"""
    
    def __init__(self, name: str = "audit"):
        self.logger = logger_manager.get_logger(name)
        
        # 确保有专门的审计日志处理器
        if not any(isinstance(h, logging.handlers.RotatingFileHandler) 
                  for h in self.logger.handlers):
            handler = LogHandlerFactory.create_file_handler(
                filename="audit.log",
                level=logging.INFO,
                use_json=True
            )
            self.logger.addHandler(handler)
            self.logger.propagate = False
    
    def log_user_action(
        self,
        user_id: str,
        action: str,
        resource: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        **extra_data
    ):
        """记录用户操作"""
        audit_data = {
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'resource_id': resource_id,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'timestamp': datetime.now().isoformat(),
            **extra_data
        }
        
        # 移除空值
        audit_data = {k: v for k, v in audit_data.items() if v is not None}
        
        self.logger.info(
            f"User {user_id} performed {action}",
            extra={'audit_data': audit_data}
        )
    
    def log_system_event(
        self,
        event_type: str,
        description: str,
        severity: str = "info",
        **extra_data
    ):
        """记录系统事件"""
        event_data = {
            'event_type': event_type,
            'description': description,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            **extra_data
        }
        
        level = getattr(logging, severity.upper(), logging.INFO)
        
        self.logger.log(
            level,
            f"System event: {event_type} - {description}",
            extra={'system_event': event_data}
        )


def get_audit_logger() -> AuditLogger:
    """获取审计日志记录器"""
    return AuditLogger()


# =============================================================================
# 初始化
# =============================================================================

# 自动配置日志系统（如果还未配置）
if not logger_manager._configured:
    # 从环境变量读取配置
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    console_output = os.getenv('LOG_CONSOLE', 'true').lower() == 'true'
    file_output = os.getenv('LOG_FILE', 'true').lower() == 'true'
    json_format = os.getenv('LOG_JSON', 'false').lower() == 'true'
    log_dir = os.getenv('LOG_DIR')
    
    configure_logging(
        level=log_level,
        console_output=console_output,
        file_output=file_output,
        json_format=json_format,
        log_dir=log_dir
    )