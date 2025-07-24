# -*- coding: utf-8 -*-
"""
可观测性模块
提供监控、追踪、指标收集和告警功能
"""

import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from contextlib import contextmanager, asynccontextmanager
from collections import defaultdict, deque
import psutil
import traceback
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型枚举"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """指标数据结构"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    help_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
            "help": self.help_text
        }


@dataclass
class TraceSpan:
    """追踪跨度"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"

    @property
    def duration(self) -> Optional[float]:
        """获取持续时间"""
        if self.end_time:
            return self.end_time - self.start_time
        return None

    def finish(self, status: str = "ok"):
        """结束跨度"""
        self.end_time = time.time()
        self.status = status

    def add_tag(self, key: str, value: Any):
        """添加标签"""
        self.tags[key] = value

    def add_log(self, message: str, level: str = "info", **kwargs):
        """添加日志"""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status
        }


@dataclass
class Alert:
    """告警数据结构"""
    id: str
    name: str
    level: AlertLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None

    def resolve(self):
        """解决告警"""
        self.resolved = True
        self.resolved_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at
        }


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def counter(self, name: str, value: float = 1, labels: Dict[str, str] = None):
        """记录计数器指标"""
        with self._lock:
            key = self._make_key(name, labels or {})
            self.counters[key] += value
            self.metrics[key] = Metric(
                name=name,
                value=self.counters[key],
                metric_type=MetricType.COUNTER,
                labels=labels or {}
            )

    def gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """记录仪表盘指标"""
        with self._lock:
            key = self._make_key(name, labels or {})
            self.gauges[key] = value
            self.metrics[key] = Metric(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                labels=labels or {}
            )

    def histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """记录直方图指标"""
        with self._lock:
            key = self._make_key(name, labels or {})
            self.histograms[key].append(value)
            # 保留最近1000个值
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]

            # 计算统计值
            values = self.histograms[key]
            self.metrics[key] = Metric(
                name=name,
                value={
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "p50": self._percentile(values, 0.5),
                    "p95": self._percentile(values, 0.95),
                    "p99": self._percentile(values, 0.99)
                },
                metric_type=MetricType.HISTOGRAM,
                labels=labels or {}
            )

    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        """生成指标键"""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _percentile(self, values: List[float], p: float) -> float:
        """计算百分位数"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def get_metrics(self) -> List[Metric]:
        """获取所有指标"""
        with self._lock:
            return list(self.metrics.values())

    def clear_metrics(self):
        """清空指标"""
        with self._lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()


class Tracer:
    """分布式追踪器"""

    def __init__(self):
        self.spans: Dict[str, TraceSpan] = {}
        self.active_spans: Dict[str, str] = {}  # thread_id -> span_id
        self._lock = threading.Lock()

    def start_span(
        self,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> TraceSpan:
        """开始一个新的跨度"""
        import uuid
        thread_id = str(threading.get_ident())

        if not trace_id:
            if parent_span_id and parent_span_id in self.spans:
                trace_id = self.spans[parent_span_id].trace_id
            else:
                trace_id = str(uuid.uuid4())

        span_id = str(uuid.uuid4())
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time()
        )

        with self._lock:
            self.spans[span_id] = span
            self.active_spans[thread_id] = span_id

        return span

    def get_active_span(self) -> Optional[TraceSpan]:
        """获取当前活跃的跨度"""
        thread_id = str(threading.get_ident())
        span_id = self.active_spans.get(thread_id)
        if span_id:
            return self.spans.get(span_id)
        return None

    def finish_span(self, span_id: str, status: str = "ok"):
        """结束跨度"""
        with self._lock:
            if span_id in self.spans:
                self.spans[span_id].finish(status)
                # 清理活跃跨度
                thread_id = str(threading.get_ident())
                if self.active_spans.get(thread_id) == span_id:
                    del self.active_spans[thread_id]

    @contextmanager
    def span(self, operation_name: str, **tags):
        """跨度上下文管理器"""
        active_span = self.get_active_span()
        parent_span_id = active_span.span_id if active_span else None

        span = self.start_span(operation_name, parent_span_id)
        for key, value in tags.items():
            span.add_tag(key, value)

        try:
            yield span
            self.finish_span(span.span_id, "ok")
        except Exception as e:
            span.add_tag("error", True)
            span.add_tag("error.message", str(e))
            span.add_log(f"Exception: {e}", "error", traceback=traceback.format_exc())
            self.finish_span(span.span_id, "error")
            raise

    def get_traces(self, trace_id: Optional[str] = None) -> List[TraceSpan]:
        """获取追踪数据"""
        with self._lock:
            if trace_id:
                return [span for span in self.spans.values() if span.trace_id == trace_id]
            return list(self.spans.values())


class SystemMonitor:
    """系统监控器"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    async def start_monitoring(self, interval: int = 30):
        """开始系统监控"""
        self._monitoring = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(interval)
        )

    async def stop_monitoring(self):
        """停止系统监控"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self, interval: int):
        """监控循环"""
        while self._monitoring:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(interval)

    async def _collect_system_metrics(self):
        """收集系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics_collector.gauge("system_cpu_usage_percent", cpu_percent)

        # 内存使用情况
        memory = psutil.virtual_memory()
        self.metrics_collector.gauge("system_memory_usage_percent", memory.percent)
        self.metrics_collector.gauge("system_memory_used_bytes", memory.used)
        self.metrics_collector.gauge("system_memory_available_bytes", memory.available)

        # 磁盘使用情况
        disk = psutil.disk_usage('/')
        self.metrics_collector.gauge("system_disk_usage_percent", 
                                   (disk.used / disk.total) * 100)
        self.metrics_collector.gauge("system_disk_used_bytes", disk.used)
        self.metrics_collector.gauge("system_disk_free_bytes", disk.free)

        # 网络IO
        net_io = psutil.net_io_counters()
        self.metrics_collector.counter("system_network_bytes_sent", net_io.bytes_sent)
        self.metrics_collector.counter("system_network_bytes_recv", net_io.bytes_recv)

        # 进程信息
        process = psutil.Process()
        self.metrics_collector.gauge("process_cpu_percent", process.cpu_percent())
        self.metrics_collector.gauge("process_memory_rss_bytes", process.memory_info().rss)
        self.metrics_collector.gauge("process_open_fds", process.num_fds())


class AlertManager:
    """告警管理器"""

    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.rules: List[Callable] = []
        self.handlers: List[Callable] = []
        self._lock = threading.Lock()

    def add_rule(self, rule_func: Callable[[List[Metric]], List[Alert]]):
        """添加告警规则"""
        self.rules.append(rule_func)

    def add_handler(self, handler_func: Callable[[Alert], None]):
        """添加告警处理器"""
        self.handlers.append(handler_func)

    def create_alert(
        self,
        alert_id: str,
        name: str,
        level: AlertLevel,
        message: str,
        labels: Dict[str, str] = None
    ) -> Alert:
        """创建告警"""
        alert = Alert(
            id=alert_id,
            name=name,
            level=level,
            message=message,
            labels=labels or {}
        )

        with self._lock:
            self.alerts[alert_id] = alert

        # 触发处理器
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

        return alert

    def resolve_alert(self, alert_id: str):
        """解决告警"""
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolve()

    def evaluate_rules(self, metrics: List[Metric]):
        """评估告警规则"""
        for rule in self.rules:
            try:
                alerts = rule(metrics)
                for alert in alerts:
                    if alert.id not in self.alerts:
                        self.create_alert(
                            alert.id, alert.name, alert.level,
                            alert.message, alert.labels
                        )
            except Exception as e:
                logger.error(f"Error evaluating alert rule: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        with self._lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]


class ObservabilityManager:
    """可观测性管理器"""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.tracer = Tracer()
        self.system_monitor = SystemMonitor(self.metrics_collector)
        self.alert_manager = AlertManager()
        self._setup_default_alert_rules()

    def _setup_default_alert_rules(self):
        """设置默认告警规则"""
        def high_cpu_rule(metrics: List[Metric]) -> List[Alert]:
            alerts = []
            for metric in metrics:
                if (metric.name == "system_cpu_usage_percent" and 
                    metric.value > 80):
                    alerts.append(Alert(
                        id="high_cpu_usage",
                        name="High CPU Usage",
                        level=AlertLevel.WARNING,
                        message=f"CPU usage is {metric.value}%"
                    ))
            return alerts

        def high_memory_rule(metrics: List[Metric]) -> List[Alert]:
            alerts = []
            for metric in metrics:
                if (metric.name == "system_memory_usage_percent" and 
                    metric.value > 90):
                    alerts.append(Alert(
                        id="high_memory_usage",
                        name="High Memory Usage",
                        level=AlertLevel.ERROR,
                        message=f"Memory usage is {metric.value}%"
                    ))
            return alerts

        self.alert_manager.add_rule(high_cpu_rule)
        self.alert_manager.add_rule(high_memory_rule)

    async def start(self):
        """启动可观测性服务"""
        await self.system_monitor.start_monitoring()
        logger.info("Observability manager started")

    async def stop(self):
        """停止可观测性服务"""
        await self.system_monitor.stop_monitoring()
        logger.info("Observability manager stopped")

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        metrics = self.metrics_collector.get_metrics()
        active_alerts = self.alert_manager.get_active_alerts()
        
        # 计算健康分数
        health_score = 100
        for alert in active_alerts:
            if alert.level == AlertLevel.CRITICAL:
                health_score -= 30
            elif alert.level == AlertLevel.ERROR:
                health_score -= 20
            elif alert.level == AlertLevel.WARNING:
                health_score -= 10

        health_score = max(0, health_score)
        
        status = "healthy"
        if health_score < 50:
            status = "unhealthy"
        elif health_score < 80:
            status = "degraded"

        return {
            "status": status,
            "health_score": health_score,
            "metrics_count": len(metrics),
            "active_alerts": len(active_alerts),
            "timestamp": time.time()
        }


# 装饰器支持
def monitor_performance(operation_name: str = None):
    """性能监控装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # 获取全局观测器（需要在应用中设置）
            observability = getattr(wrapper, '_observability', None)
            if not observability:
                return func(*args, **kwargs)

            with observability.tracer.span(name) as span:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    observability.metrics_collector.histogram(
                        f"{name}_duration_seconds", duration
                    )
                    observability.metrics_collector.counter(
                        f"{name}_total", labels={"status": "success"}
                    )
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    observability.metrics_collector.histogram(
                        f"{name}_duration_seconds", duration
                    )
                    observability.metrics_collector.counter(
                        f"{name}_total", labels={"status": "error"}
                    )
                    raise
        return wrapper
    return decorator


# 使用示例
if __name__ == "__main__":
    async def example_usage():
        # 创建可观测性管理器
        obs_manager = ObservabilityManager()
        
        # 启动监控
        await obs_manager.start()
        
        try:
            # 记录一些指标
            obs_manager.metrics_collector.counter("requests_total", 1, 
                                                 {"method": "GET", "status": "200"})
            obs_manager.metrics_collector.gauge("active_connections", 42)
            obs_manager.metrics_collector.histogram("request_duration_seconds", 0.123)
            
            # 使用追踪
            with obs_manager.tracer.span("example_operation") as span:
                span.add_tag("user_id", "12345")
                span.add_log("Processing request")
                await asyncio.sleep(0.1)
                span.add_log("Request processed")
            
            # 获取健康状态
            health = obs_manager.get_health_status()
            print(f"Health status: {health}")
            
            # 等待一段时间让监控收集数据
            await asyncio.sleep(5)
            
        finally:
            await obs_manager.stop()
    
    # 运行示例
    asyncio.run(example_usage())