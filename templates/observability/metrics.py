"""指标收集系统

提供完整的指标收集功能，包括：
- Prometheus集成
- 自定义指标
- 业务指标
- 性能指标
- 告警规则
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server, push_to_gateway
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricType(str, Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


class MetricUnit(str, Enum):
    """指标单位"""
    NONE = ""
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    MICROSECONDS = "microseconds"
    BYTES = "bytes"
    KILOBYTES = "kilobytes"
    MEGABYTES = "megabytes"
    GIGABYTES = "gigabytes"
    PERCENT = "percent"
    COUNT = "count"
    RATE = "rate"
    RATIO = "ratio"


@dataclass
class MetricDefinition:
    """指标定义"""
    name: str
    description: str
    metric_type: MetricType
    unit: MetricUnit = MetricUnit.NONE
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # 用于Histogram
    quantiles: Optional[List[float]] = None  # 用于Summary
    
    def __post_init__(self):
        # 验证指标名称
        if not self.name.replace('_', '').replace(':', '').isalnum():
            raise ValueError(f"Invalid metric name: {self.name}")
        
        # 设置默认buckets
        if self.metric_type == MetricType.HISTOGRAM and self.buckets is None:
            self.buckets = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
        
        # 设置默认quantiles
        if self.metric_type == MetricType.SUMMARY and self.quantiles is None:
            self.quantiles = [0.5, 0.9, 0.95, 0.99]


@dataclass
class MetricSample:
    """指标样本"""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp
        }


class MetricsConfig(BaseSettings):
    """指标配置"""
    enabled: bool = True
    
    # Prometheus配置
    prometheus_enabled: bool = True
    prometheus_port: int = 8000
    prometheus_path: str = "/metrics"
    prometheus_gateway_url: Optional[str] = None
    prometheus_job_name: str = "python-app"
    
    # 指标收集配置
    collection_interval: int = 15  # seconds
    retention_period: int = 3600  # seconds
    max_samples_per_metric: int = 1000
    
    # 标签配置
    default_labels: Dict[str, str] = Field(default_factory=dict)
    
    # 性能配置
    enable_runtime_metrics: bool = True
    enable_gc_metrics: bool = True
    enable_process_metrics: bool = True
    
    class Config:
        env_prefix = "METRICS_"


class BaseMetric(ABC):
    """指标基类"""
    
    def __init__(self, definition: MetricDefinition):
        self.definition = definition
        self.samples: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    @abstractmethod
    def record(self, value: float, labels: Optional[Dict[str, str]] = None):
        """记录指标值"""
        pass
    
    @abstractmethod
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> float:
        """获取指标值"""
        pass
    
    def get_samples(self) -> List[MetricSample]:
        """获取所有样本"""
        with self._lock:
            return list(self.samples)
    
    def clear_samples(self):
        """清空样本"""
        with self._lock:
            self.samples.clear()


class CounterMetric(BaseMetric):
    """计数器指标"""
    
    def __init__(self, definition: MetricDefinition):
        super().__init__(definition)
        self._values: Dict[str, float] = defaultdict(float)
    
    def record(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """增加计数"""
        if value < 0:
            raise ValueError("Counter value must be non-negative")
        
        labels = labels or {}
        label_key = self._get_label_key(labels)
        
        with self._lock:
            self._values[label_key] += value
            sample = MetricSample(
                name=self.definition.name,
                value=self._values[label_key],
                labels=labels
            )
            self.samples.append(sample)
    
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> float:
        """获取计数值"""
        labels = labels or {}
        label_key = self._get_label_key(labels)
        return self._values.get(label_key, 0.0)
    
    def _get_label_key(self, labels: Dict[str, str]) -> str:
        """生成标签键"""
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))


class GaugeMetric(BaseMetric):
    """仪表盘指标"""
    
    def __init__(self, definition: MetricDefinition):
        super().__init__(definition)
        self._values: Dict[str, float] = {}
    
    def record(self, value: float, labels: Optional[Dict[str, str]] = None):
        """设置仪表值"""
        labels = labels or {}
        label_key = self._get_label_key(labels)
        
        with self._lock:
            self._values[label_key] = value
            sample = MetricSample(
                name=self.definition.name,
                value=value,
                labels=labels
            )
            self.samples.append(sample)
    
    def increment(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """增加仪表值"""
        labels = labels or {}
        label_key = self._get_label_key(labels)
        
        with self._lock:
            current = self._values.get(label_key, 0.0)
            self.record(current + value, labels)
    
    def decrement(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """减少仪表值"""
        self.increment(-value, labels)
    
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> float:
        """获取仪表值"""
        labels = labels or {}
        label_key = self._get_label_key(labels)
        return self._values.get(label_key, 0.0)
    
    def _get_label_key(self, labels: Dict[str, str]) -> str:
        """生成标签键"""
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))


class HistogramMetric(BaseMetric):
    """直方图指标"""
    
    def __init__(self, definition: MetricDefinition):
        super().__init__(definition)
        self.buckets = definition.buckets or []
        self._bucket_counts: Dict[str, Dict[float, int]] = defaultdict(lambda: defaultdict(int))
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
    
    def record(self, value: float, labels: Optional[Dict[str, str]] = None):
        """记录观测值"""
        labels = labels or {}
        label_key = self._get_label_key(labels)
        
        with self._lock:
            # 更新桶计数
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[label_key][bucket] += 1
            
            # 更新总和和计数
            self._sums[label_key] += value
            self._counts[label_key] += 1
            
            sample = MetricSample(
                name=self.definition.name,
                value=value,
                labels=labels
            )
            self.samples.append(sample)
    
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> float:
        """获取平均值"""
        labels = labels or {}
        label_key = self._get_label_key(labels)
        
        count = self._counts.get(label_key, 0)
        if count == 0:
            return 0.0
        
        return self._sums[label_key] / count
    
    def get_bucket_counts(self, labels: Optional[Dict[str, str]] = None) -> Dict[float, int]:
        """获取桶计数"""
        labels = labels or {}
        label_key = self._get_label_key(labels)
        return dict(self._bucket_counts[label_key])
    
    def get_sum(self, labels: Optional[Dict[str, str]] = None) -> float:
        """获取总和"""
        labels = labels or {}
        label_key = self._get_label_key(labels)
        return self._sums[label_key]
    
    def get_count(self, labels: Optional[Dict[str, str]] = None) -> int:
        """获取计数"""
        labels = labels or {}
        label_key = self._get_label_key(labels)
        return self._counts[label_key]
    
    def _get_label_key(self, labels: Dict[str, str]) -> str:
        """生成标签键"""
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))


class MetricsRegistry:
    """指标注册表"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.metrics: Dict[str, BaseMetric] = {}
        self._lock = threading.Lock()
        
        # Prometheus注册表
        if PROMETHEUS_AVAILABLE and config.prometheus_enabled:
            self.prometheus_registry = CollectorRegistry()
            self.prometheus_metrics: Dict[str, Any] = {}
    
    def register_metric(self, definition: MetricDefinition) -> BaseMetric:
        """注册指标"""
        with self._lock:
            if definition.name in self.metrics:
                return self.metrics[definition.name]
            
            # 创建自定义指标
            if definition.metric_type == MetricType.COUNTER:
                metric = CounterMetric(definition)
            elif definition.metric_type == MetricType.GAUGE:
                metric = GaugeMetric(definition)
            elif definition.metric_type == MetricType.HISTOGRAM:
                metric = HistogramMetric(definition)
            else:
                raise ValueError(f"Unsupported metric type: {definition.metric_type}")
            
            self.metrics[definition.name] = metric
            
            # 注册到Prometheus
            if PROMETHEUS_AVAILABLE and self.config.prometheus_enabled:
                self._register_prometheus_metric(definition)
            
            return metric
    
    def _register_prometheus_metric(self, definition: MetricDefinition):
        """注册Prometheus指标"""
        kwargs = {
            'name': definition.name,
            'documentation': definition.description,
            'labelnames': definition.labels,
            'registry': self.prometheus_registry
        }
        
        if definition.metric_type == MetricType.COUNTER:
            prometheus_metric = Counter(**kwargs)
        elif definition.metric_type == MetricType.GAUGE:
            prometheus_metric = Gauge(**kwargs)
        elif definition.metric_type == MetricType.HISTOGRAM:
            if definition.buckets:
                kwargs['buckets'] = definition.buckets
            prometheus_metric = Histogram(**kwargs)
        elif definition.metric_type == MetricType.SUMMARY:
            if definition.quantiles:
                kwargs['quantiles'] = definition.quantiles
            prometheus_metric = Summary(**kwargs)
        elif definition.metric_type == MetricType.INFO:
            prometheus_metric = Info(**kwargs)
        else:
            return
        
        self.prometheus_metrics[definition.name] = prometheus_metric
    
    def get_metric(self, name: str) -> Optional[BaseMetric]:
        """获取指标"""
        return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, BaseMetric]:
        """获取所有指标"""
        return self.metrics.copy()
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """记录指标值"""
        metric = self.get_metric(name)
        if metric:
            # 添加默认标签
            if labels:
                labels.update(self.config.default_labels)
            else:
                labels = self.config.default_labels.copy()
            
            metric.record(value, labels)
            
            # 同步到Prometheus
            if PROMETHEUS_AVAILABLE and self.config.prometheus_enabled:
                self._record_prometheus_metric(name, value, labels)
    
    def _record_prometheus_metric(self, name: str, value: float, labels: Optional[Dict[str, str]]):
        """记录Prometheus指标"""
        prometheus_metric = self.prometheus_metrics.get(name)
        if not prometheus_metric:
            return
        
        labels = labels or {}
        
        if isinstance(prometheus_metric, Counter):
            prometheus_metric.labels(**labels).inc(value)
        elif isinstance(prometheus_metric, Gauge):
            prometheus_metric.labels(**labels).set(value)
        elif isinstance(prometheus_metric, (Histogram, Summary)):
            prometheus_metric.labels(**labels).observe(value)
    
    def get_prometheus_metrics(self) -> str:
        """获取Prometheus格式的指标"""
        if not PROMETHEUS_AVAILABLE or not self.config.prometheus_enabled:
            return ""
        
        return generate_latest(self.prometheus_registry).decode('utf-8')
    
    def start_prometheus_server(self):
        """启动Prometheus HTTP服务器"""
        if not PROMETHEUS_AVAILABLE or not self.config.prometheus_enabled:
            return
        
        start_http_server(self.config.prometheus_port, registry=self.prometheus_registry)
    
    def push_to_gateway(self):
        """推送指标到Prometheus Gateway"""
        if not PROMETHEUS_AVAILABLE or not self.config.prometheus_gateway_url:
            return
        
        push_to_gateway(
            self.config.prometheus_gateway_url,
            job=self.config.prometheus_job_name,
            registry=self.prometheus_registry
        )


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, registry: MetricsRegistry):
        self.registry = registry
        self.config = registry.config
        self._collectors: List[Callable[[], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def add_collector(self, collector: Callable[[], None]):
        """添加收集器"""
        self._collectors.append(collector)
    
    def start(self):
        """启动收集器"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """停止收集器"""
        self._running = False
        if self._thread:
            self._thread.join()
    
    def _collect_loop(self):
        """收集循环"""
        while self._running:
            try:
                for collector in self._collectors:
                    collector()
            except Exception as e:
                print(f"Metrics collection error: {e}")
            
            time.sleep(self.config.collection_interval)
    
    def collect_runtime_metrics(self):
        """收集运行时指标"""
        if not self.config.enable_runtime_metrics:
            return
        
        import psutil
        import gc
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent()
        self.registry.record_metric("runtime_cpu_usage_percent", cpu_percent)
        
        # 内存使用
        memory = psutil.virtual_memory()
        self.registry.record_metric("runtime_memory_usage_bytes", memory.used)
        self.registry.record_metric("runtime_memory_usage_percent", memory.percent)
        
        # 垃圾回收
        if self.config.enable_gc_metrics:
            gc_stats = gc.get_stats()
            for i, stats in enumerate(gc_stats):
                labels = {"generation": str(i)}
                self.registry.record_metric("runtime_gc_collections_total", stats["collections"], labels)
                self.registry.record_metric("runtime_gc_collected_total", stats["collected"], labels)
                self.registry.record_metric("runtime_gc_uncollectable_total", stats["uncollectable"], labels)


class MetricsManager:
    """指标管理器"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.registry = MetricsRegistry(config)
        self.collector = MetricsCollector(self.registry)
        
        # 注册默认指标
        self._register_default_metrics()
        
        # 添加默认收集器
        self.collector.add_collector(self.collector.collect_runtime_metrics)
    
    def _register_default_metrics(self):
        """注册默认指标"""
        # HTTP请求指标
        self.registry.register_metric(MetricDefinition(
            name="http_requests_total",
            description="Total number of HTTP requests",
            metric_type=MetricType.COUNTER,
            labels=["method", "endpoint", "status_code"]
        ))
        
        self.registry.register_metric(MetricDefinition(
            name="http_request_duration_seconds",
            description="HTTP request duration in seconds",
            metric_type=MetricType.HISTOGRAM,
            unit=MetricUnit.SECONDS,
            labels=["method", "endpoint"]
        ))
        
        # 数据库指标
        self.registry.register_metric(MetricDefinition(
            name="database_connections_active",
            description="Number of active database connections",
            metric_type=MetricType.GAUGE
        ))
        
        self.registry.register_metric(MetricDefinition(
            name="database_query_duration_seconds",
            description="Database query duration in seconds",
            metric_type=MetricType.HISTOGRAM,
            unit=MetricUnit.SECONDS,
            labels=["query_type"]
        ))
        
        # 缓存指标
        self.registry.register_metric(MetricDefinition(
            name="cache_hits_total",
            description="Total number of cache hits",
            metric_type=MetricType.COUNTER,
            labels=["cache_name"]
        ))
        
        self.registry.register_metric(MetricDefinition(
            name="cache_misses_total",
            description="Total number of cache misses",
            metric_type=MetricType.COUNTER,
            labels=["cache_name"]
        ))
        
        # 运行时指标
        if self.config.enable_runtime_metrics:
            self.registry.register_metric(MetricDefinition(
                name="runtime_cpu_usage_percent",
                description="CPU usage percentage",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.PERCENT
            ))
            
            self.registry.register_metric(MetricDefinition(
                name="runtime_memory_usage_bytes",
                description="Memory usage in bytes",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.BYTES
            ))
            
            self.registry.register_metric(MetricDefinition(
                name="runtime_memory_usage_percent",
                description="Memory usage percentage",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.PERCENT
            ))
        
        # 垃圾回收指标
        if self.config.enable_gc_metrics:
            self.registry.register_metric(MetricDefinition(
                name="runtime_gc_collections_total",
                description="Total number of garbage collections",
                metric_type=MetricType.COUNTER,
                labels=["generation"]
            ))
            
            self.registry.register_metric(MetricDefinition(
                name="runtime_gc_collected_total",
                description="Total number of objects collected",
                metric_type=MetricType.COUNTER,
                labels=["generation"]
            ))
    
    def start(self):
        """启动指标收集"""
        if not self.config.enabled:
            return
        
        self.collector.start()
        
        # 启动Prometheus服务器
        if PROMETHEUS_AVAILABLE and self.config.prometheus_enabled:
            self.registry.start_prometheus_server()
    
    def stop(self):
        """停止指标收集"""
        self.collector.stop()
    
    def record(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """记录指标"""
        self.registry.record_metric(name, value, labels)
    
    def increment(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """增加计数器"""
        self.record(name, value, labels)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """设置仪表值"""
        self.record(name, value, labels)
    
    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """观测直方图/摘要"""
        self.record(name, value, labels)
    
    def get_metrics_data(self) -> str:
        """获取指标数据"""
        return self.registry.get_prometheus_metrics()


# 全局指标管理器
_metrics_manager: Optional[MetricsManager] = None


def initialize_metrics(config: MetricsConfig) -> MetricsManager:
    """初始化指标系统"""
    global _metrics_manager
    _metrics_manager = MetricsManager(config)
    _metrics_manager.start()
    return _metrics_manager


def get_metrics() -> Optional[MetricsManager]:
    """获取全局指标管理器"""
    return _metrics_manager


# 装饰器
def measure_time(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """测量执行时间装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            if not metrics:
                return func(*args, **kwargs)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metrics.observe(metric_name, duration, labels)
        
        return wrapper
    return decorator


def count_calls(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """计数函数调用装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            if metrics:
                metrics.increment(metric_name, 1.0, labels)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# 便捷函数
def record_metric(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """记录指标的便捷函数"""
    metrics = get_metrics()
    if metrics:
        metrics.record(name, value, labels)


def increment_counter(name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
    """增加计数器的便捷函数"""
    metrics = get_metrics()
    if metrics:
        metrics.increment(name, value, labels)


def set_gauge_value(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """设置仪表值的便捷函数"""
    metrics = get_metrics()
    if metrics:
        metrics.set_gauge(name, value, labels)


def observe_histogram(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """观测直方图的便捷函数"""
    metrics = get_metrics()
    if metrics:
        metrics.observe(name, value, labels)