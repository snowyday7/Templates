"""分布式追踪系统

提供完整的分布式追踪功能，包括：
- OpenTelemetry集成
- 自定义追踪器
- 跨服务追踪
- 性能监控
- 错误追踪
"""

import time
import uuid
import threading
import contextvars
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.zipkin.json import ZipkinExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


class SpanKind(str, Enum):
    """Span类型"""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    """Span状态"""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SpanContext:
    """Span上下文"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanContext":
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            baggage=data.get("baggage", {})
        )


@dataclass
class Span:
    """追踪Span"""
    context: SpanContext
    operation_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    kind: SpanKind = SpanKind.INTERNAL
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def set_tag(self, key: str, value: Any):
        """设置标签"""
        self.tags[key] = value
    
    def set_tags(self, tags: Dict[str, Any]):
        """批量设置标签"""
        self.tags.update(tags)
    
    def log(self, message: str, level: str = "info", **kwargs):
        """记录日志"""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def set_status(self, status: SpanStatus, message: Optional[str] = None):
        """设置状态"""
        self.status = status
        if message:
            self.set_tag("status_message", message)
    
    def finish(self):
        """结束Span"""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "context": self.context.to_dict(),
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "status": self.status.value,
            "kind": self.kind.value,
            "tags": self.tags,
            "logs": self.logs
        }


class TraceConfig(BaseSettings):
    """追踪配置"""
    enabled: bool = True
    service_name: str = "unknown-service"
    service_version: str = "1.0.0"
    environment: str = "development"
    
    # 采样配置
    sampling_rate: float = Field(1.0, ge=0.0, le=1.0)
    
    # 导出器配置
    exporter_type: str = "console"  # console, jaeger, zipkin
    jaeger_endpoint: Optional[str] = None
    zipkin_endpoint: Optional[str] = None
    
    # 批处理配置
    batch_size: int = 512
    batch_timeout: int = 5000  # milliseconds
    
    # 资源限制
    max_spans_per_trace: int = 1000
    max_attributes_per_span: int = 128
    
    class Config:
        env_prefix = "TRACE_"


class BaseTracer(ABC):
    """追踪器基类"""
    
    @abstractmethod
    def start_span(self, operation_name: str, parent_context: Optional[SpanContext] = None, 
                   kind: SpanKind = SpanKind.INTERNAL, **kwargs) -> Span:
        """开始一个新的Span"""
        pass
    
    @abstractmethod
    def finish_span(self, span: Span):
        """结束Span"""
        pass
    
    @abstractmethod
    def inject_context(self, span_context: SpanContext, carrier: Dict[str, str]):
        """注入追踪上下文"""
        pass
    
    @abstractmethod
    def extract_context(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """提取追踪上下文"""
        pass


class CustomTracer(BaseTracer):
    """自定义追踪器"""
    
    def __init__(self, config: TraceConfig):
        self.config = config
        self.spans: Dict[str, Span] = {}
        self._lock = threading.Lock()
        self._span_processors: List[Callable[[Span], None]] = []
    
    def add_span_processor(self, processor: Callable[[Span], None]):
        """添加Span处理器"""
        self._span_processors.append(processor)
    
    def start_span(self, operation_name: str, parent_context: Optional[SpanContext] = None,
                   kind: SpanKind = SpanKind.INTERNAL, **kwargs) -> Span:
        """开始一个新的Span"""
        # 生成追踪ID和Span ID
        if parent_context:
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
        else:
            trace_id = str(uuid.uuid4()).replace("-", "")
            parent_span_id = None
        
        span_id = str(uuid.uuid4()).replace("-", "")[:16]
        
        # 创建Span上下文
        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id
        )
        
        # 创建Span
        span = Span(
            context=context,
            operation_name=operation_name,
            kind=kind
        )
        
        # 设置默认标签
        span.set_tags({
            "service.name": self.config.service_name,
            "service.version": self.config.service_version,
            "environment": self.config.environment,
            **kwargs
        })
        
        # 存储Span
        with self._lock:
            self.spans[span_id] = span
        
        return span
    
    def finish_span(self, span: Span):
        """结束Span"""
        span.finish()
        
        # 处理Span
        for processor in self._span_processors:
            try:
                processor(span)
            except Exception as e:
                print(f"Span processor error: {e}")
        
        # 从存储中移除
        with self._lock:
            self.spans.pop(span.context.span_id, None)
    
    def inject_context(self, span_context: SpanContext, carrier: Dict[str, str]):
        """注入追踪上下文到HTTP头"""
        carrier["X-Trace-Id"] = span_context.trace_id
        carrier["X-Span-Id"] = span_context.span_id
        if span_context.parent_span_id:
            carrier["X-Parent-Span-Id"] = span_context.parent_span_id
    
    def extract_context(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """从HTTP头提取追踪上下文"""
        trace_id = carrier.get("X-Trace-Id")
        span_id = carrier.get("X-Span-Id")
        
        if not trace_id or not span_id:
            return None
        
        return SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=carrier.get("X-Parent-Span-Id")
        )


class OpenTelemetryTracer(BaseTracer):
    """OpenTelemetry追踪器"""
    
    def __init__(self, config: TraceConfig):
        if not OPENTELEMETRY_AVAILABLE:
            raise ImportError("OpenTelemetry not available")
        
        self.config = config
        self._setup_tracer()
    
    def _setup_tracer(self):
        """设置OpenTelemetry追踪器"""
        # 创建TracerProvider
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        
        # 配置导出器
        if self.config.exporter_type == "jaeger" and self.config.jaeger_endpoint:
            exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
        elif self.config.exporter_type == "zipkin" and self.config.zipkin_endpoint:
            exporter = ZipkinExporter(
                endpoint=self.config.zipkin_endpoint
            )
        else:
            # 默认控制台导出器
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            exporter = ConsoleSpanExporter()
        
        # 添加批处理器
        processor = BatchSpanProcessor(
            exporter,
            max_queue_size=self.config.batch_size,
            schedule_delay_millis=self.config.batch_timeout
        )
        provider.add_span_processor(processor)
        
        # 获取追踪器
        self.tracer = trace.get_tracer(
            self.config.service_name,
            self.config.service_version
        )
    
    def start_span(self, operation_name: str, parent_context: Optional[SpanContext] = None,
                   kind: SpanKind = SpanKind.INTERNAL, **kwargs) -> Span:
        """开始OpenTelemetry Span"""
        # 这里需要适配OpenTelemetry的API
        # 简化实现，实际应该使用OpenTelemetry的原生Span
        otel_span = self.tracer.start_span(operation_name)
        
        # 转换为自定义Span格式
        context = SpanContext(
            trace_id=format(otel_span.get_span_context().trace_id, '032x'),
            span_id=format(otel_span.get_span_context().span_id, '016x')
        )
        
        span = Span(
            context=context,
            operation_name=operation_name,
            kind=kind
        )
        
        # 设置属性
        for key, value in kwargs.items():
            otel_span.set_attribute(key, value)
        
        # 存储OpenTelemetry span引用
        span._otel_span = otel_span
        
        return span
    
    def finish_span(self, span: Span):
        """结束OpenTelemetry Span"""
        if hasattr(span, '_otel_span'):
            span._otel_span.end()
        span.finish()
    
    def inject_context(self, span_context: SpanContext, carrier: Dict[str, str]):
        """注入OpenTelemetry上下文"""
        # 使用OpenTelemetry的上下文传播
        from opentelemetry.propagate import inject
        inject(carrier)
    
    def extract_context(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """提取OpenTelemetry上下文"""
        from opentelemetry.propagate import extract
        context = extract(carrier)
        
        # 转换为自定义SpanContext格式
        span_context = trace.get_current_span(context).get_span_context()
        if span_context.is_valid:
            return SpanContext(
                trace_id=format(span_context.trace_id, '032x'),
                span_id=format(span_context.span_id, '016x')
            )
        
        return None


class TracingManager:
    """追踪管理器"""
    
    def __init__(self, config: TraceConfig):
        self.config = config
        self.tracer = self._create_tracer()
        self._current_span: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar(
            'current_span', default=None
        )
    
    def _create_tracer(self) -> BaseTracer:
        """创建追踪器"""
        if OPENTELEMETRY_AVAILABLE and self.config.exporter_type in ["jaeger", "zipkin"]:
            return OpenTelemetryTracer(self.config)
        else:
            return CustomTracer(self.config)
    
    @contextmanager
    def start_span(self, operation_name: str, kind: SpanKind = SpanKind.INTERNAL, **kwargs):
        """开始Span上下文管理器"""
        parent_context = self.get_current_span_context()
        span = self.tracer.start_span(operation_name, parent_context, kind, **kwargs)
        
        # 设置为当前Span
        token = self._current_span.set(span)
        
        try:
            yield span
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.log(f"Exception: {e}", level="error")
            raise
        finally:
            self.tracer.finish_span(span)
            self._current_span.reset(token)
    
    def get_current_span(self) -> Optional[Span]:
        """获取当前Span"""
        return self._current_span.get()
    
    def get_current_span_context(self) -> Optional[SpanContext]:
        """获取当前Span上下文"""
        span = self.get_current_span()
        return span.context if span else None
    
    def inject_trace_context(self, headers: Dict[str, str]):
        """注入追踪上下文到HTTP头"""
        context = self.get_current_span_context()
        if context:
            self.tracer.inject_context(context, headers)
    
    def extract_trace_context(self, headers: Dict[str, str]) -> Optional[SpanContext]:
        """从HTTP头提取追踪上下文"""
        return self.tracer.extract_context(headers)
    
    def setup_auto_instrumentation(self):
        """设置自动埋点"""
        if not OPENTELEMETRY_AVAILABLE:
            return
        
        # FastAPI自动埋点
        FastAPIInstrumentor().instrument()
        
        # SQLAlchemy自动埋点
        SQLAlchemyInstrumentor().instrument()
        
        # Redis自动埋点
        RedisInstrumentor().instrument()
        
        # Requests自动埋点
        RequestsInstrumentor().instrument()


# 全局追踪管理器
_tracing_manager: Optional[TracingManager] = None


def initialize_tracing(config: TraceConfig) -> TracingManager:
    """初始化追踪系统"""
    global _tracing_manager
    _tracing_manager = TracingManager(config)
    _tracing_manager.setup_auto_instrumentation()
    return _tracing_manager


def get_tracer() -> Optional[TracingManager]:
    """获取全局追踪器"""
    return _tracing_manager


# 装饰器
def trace_function(operation_name: Optional[str] = None, kind: SpanKind = SpanKind.INTERNAL):
    """函数追踪装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            if not tracer or not tracer.config.enabled:
                return func(*args, **kwargs)
            
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_span(name, kind) as span:
                # 添加函数信息
                span.set_tags({
                    "function.name": func.__name__,
                    "function.module": func.__module__,
                    "function.args_count": len(args),
                    "function.kwargs_count": len(kwargs)
                })
                
                try:
                    result = func(*args, **kwargs)
                    span.set_tag("function.result_type", type(result).__name__)
                    return result
                except Exception as e:
                    span.set_status(SpanStatus.ERROR, str(e))
                    raise
        
        return wrapper
    return decorator


# 便捷函数
def start_span(operation_name: str, kind: SpanKind = SpanKind.INTERNAL, **kwargs):
    """开始Span的便捷函数"""
    tracer = get_tracer()
    if tracer:
        return tracer.start_span(operation_name, kind, **kwargs)
    return None


def get_current_span() -> Optional[Span]:
    """获取当前Span的便捷函数"""
    tracer = get_tracer()
    return tracer.get_current_span() if tracer else None


def inject_trace_context(headers: Dict[str, str]):
    """注入追踪上下文的便捷函数"""
    tracer = get_tracer()
    if tracer:
        tracer.inject_trace_context(headers)


def extract_trace_context(headers: Dict[str, str]) -> Optional[SpanContext]:
    """提取追踪上下文的便捷函数"""
    tracer = get_tracer()
    return tracer.extract_trace_context(headers) if tracer else None