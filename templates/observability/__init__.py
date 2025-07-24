"""可观测性模块

提供企业级可观测性功能，包括：
- 分布式追踪
- 指标收集和监控
- 告警系统
- 日志聚合
- 性能分析
"""

# 分布式追踪
from .tracing import (
    TracingManager,
    Tracer,
    Span,
    SpanContext,
    TraceConfig,
    trace_function,
    start_span,
    get_current_span,
    inject_trace_context,
    extract_trace_context,
)

# 指标收集
from .metrics import (
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
    Summary,
    MetricRegistry,
    BusinessMetrics,
    SystemMetrics,
    ApplicationMetrics,
    collect_metrics,
    export_metrics,
)

# 告警系统
from .alerting import (
    AlertManager,
    Alert,
    AlertRule,
    AlertChannel,
    AlertSeverity,
    EmailChannel,
    SlackChannel,
    WebhookChannel,
    create_alert,
    send_alert,
    manage_alerts,
)

# 日志聚合
from .log_aggregation import (
    LogAggregator,
    LogProcessor,
    LogFilter,
    LogEnricher,
    LogForwarder,
    ElasticsearchForwarder,
    SplunkForwarder,
    process_logs,
    forward_logs,
)

# 性能分析
from .profiling import (
    Profiler,
    CPUProfiler,
    MemoryProfiler,
    IOProfiler,
    NetworkProfiler,
    ProfileReport,
    profile_application,
    generate_report,
    analyze_performance,
)

__all__ = [
    # 分布式追踪
    "TracingManager",
    "Tracer",
    "Span",
    "SpanContext",
    "TraceConfig",
    "trace_function",
    "start_span",
    "get_current_span",
    "inject_trace_context",
    "extract_trace_context",
    
    # 指标收集
    "MetricsCollector",
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    "MetricRegistry",
    "BusinessMetrics",
    "SystemMetrics",
    "ApplicationMetrics",
    "collect_metrics",
    "export_metrics",
    
    # 告警系统
    "AlertManager",
    "Alert",
    "AlertRule",
    "AlertChannel",
    "AlertSeverity",
    "EmailChannel",
    "SlackChannel",
    "WebhookChannel",
    "create_alert",
    "send_alert",
    "manage_alerts",
    
    # 日志聚合
    "LogAggregator",
    "LogProcessor",
    "LogFilter",
    "LogEnricher",
    "LogForwarder",
    "ElasticsearchForwarder",
    "SplunkForwarder",
    "process_logs",
    "forward_logs",
    
    # 性能分析
    "Profiler",
    "CPUProfiler",
    "MemoryProfiler",
    "IOProfiler",
    "NetworkProfiler",
    "ProfileReport",
    "profile_application",
    "generate_report",
    "analyze_performance",
]