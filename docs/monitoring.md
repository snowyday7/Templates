# 监控日志模块使用指南

监控日志模块提供了完整的应用监控、日志管理、健康检查、错误追踪和性能分析功能。

## 📋 目录

- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [基础使用](#基础使用)
- [高级功能](#高级功能)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

## 🚀 快速开始

### 安装依赖

```bash
pip install structlog prometheus-client sentry-sdk psutil
```

### 基础配置

```python
from templates.monitoring import LoggingConfig, StructuredLogger, HealthCheckManager

# 创建日志配置
logging_config = LoggingConfig(
    LOG_LEVEL="INFO",
    LOG_FORMAT="json",
    LOG_FILE="app.log",
    ENABLE_CONSOLE_LOGGING=True,
    ENABLE_FILE_LOGGING=True
)

# 创建结构化日志器
logger = StructuredLogger(logging_config)

# 创建健康检查管理器
health_manager = HealthCheckManager()
```

## ⚙️ 配置说明

### LoggingConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `LOG_LEVEL` | str | "INFO" | 日志级别 |
| `LOG_FORMAT` | str | "json" | 日志格式(json/text) |
| `LOG_FILE` | str | "app.log" | 日志文件路径 |
| `LOG_MAX_SIZE` | int | 10485760 | 日志文件最大大小(字节) |
| `LOG_BACKUP_COUNT` | int | 5 | 日志文件备份数量 |
| `ENABLE_CONSOLE_LOGGING` | bool | True | 启用控制台日志 |
| `ENABLE_FILE_LOGGING` | bool | True | 启用文件日志 |
| `ENABLE_SYSLOG` | bool | False | 启用系统日志 |
| `SYSLOG_HOST` | str | "localhost" | 系统日志主机 |
| `SYSLOG_PORT` | int | 514 | 系统日志端口 |
| `LOG_CORRELATION_ID` | bool | True | 启用关联ID |

### 监控配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `METRICS_ENABLED` | bool | True | 启用指标收集 |
| `METRICS_PORT` | int | 8000 | 指标服务端口 |
| `HEALTH_CHECK_INTERVAL` | int | 30 | 健康检查间隔(秒) |
| `ERROR_TRACKING_ENABLED` | bool | True | 启用错误追踪 |
| `SENTRY_DSN` | str | None | Sentry DSN |
| `PERFORMANCE_MONITORING` | bool | True | 启用性能监控 |
| `TRACE_SAMPLING_RATE` | float | 0.1 | 追踪采样率 |

### 环境变量配置

创建 `.env` 文件：

```env
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=logs/app.log
ENABLE_CONSOLE_LOGGING=true
ENABLE_FILE_LOGGING=true

METRICS_ENABLED=true
METRICS_PORT=8000
HEALTH_CHECK_INTERVAL=30
ERROR_TRACKING_ENABLED=true
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
PERFORMANCE_MONITORING=true
TRACE_SAMPLING_RATE=0.1
```

## 💻 基础使用

### 1. 结构化日志

```python
from templates.monitoring import StructuredLogger
import structlog

# 创建日志器
logger = StructuredLogger()

# 基础日志记录
logger.info("User login", user_id=123, username="john_doe")
logger.warning("High memory usage", memory_percent=85.2)
logger.error("Database connection failed", error="Connection timeout", retry_count=3)

# 带上下文的日志
with logger.bind(request_id="req-123", user_id=456):
    logger.info("Processing request")
    logger.debug("Validating input data")
    logger.info("Request completed", duration_ms=250)

# 异常日志
try:
    result = risky_operation()
except Exception as e:
    logger.exception("Operation failed", operation="risky_operation", input_data=data)

# 性能日志
with logger.timer("database_query"):
    users = db.query(User).all()
    logger.info("Query completed", result_count=len(users))
```

### 2. 健康检查

```python
from templates.monitoring import HealthCheck, HealthCheckManager
from fastapi import APIRouter

# 定义健康检查
class DatabaseHealthCheck(HealthCheck):
    def __init__(self, db_manager):
        self.db_manager = db_manager
        super().__init__(name="database", timeout=5)
    
    async def check(self) -> dict:
        try:
            with self.db_manager.get_session() as session:
                session.execute("SELECT 1")
            return {
                "status": "healthy",
                "message": "Database connection successful"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Database connection failed: {str(e)}"
            }

class RedisHealthCheck(HealthCheck):
    def __init__(self, redis_manager):
        self.redis_manager = redis_manager
        super().__init__(name="redis", timeout=3)
    
    async def check(self) -> dict:
        try:
            await self.redis_manager.ping()
            return {
                "status": "healthy",
                "message": "Redis connection successful"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Redis connection failed: {str(e)}"
            }

# 注册健康检查
health_manager.register_check(DatabaseHealthCheck(db_manager))
health_manager.register_check(RedisHealthCheck(redis_manager))

# 创建健康检查端点
router = APIRouter()

@router.get("/health")
async def health_check():
    results = await health_manager.check_all()
    
    overall_status = "healthy" if all(
        check["status"] == "healthy" for check in results.values()
    ) else "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": results
    }

@router.get("/health/{service}")
async def service_health_check(service: str):
    result = await health_manager.check_service(service)
    if result is None:
        raise HTTPException(status_code=404, detail="Service not found")
    
    return result
```

### 3. 指标收集

```python
from templates.monitoring import MetricsCollector, PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge

# 创建指标收集器
metrics = PrometheusMetrics()

# 定义指标
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

active_users = Gauge(
    'active_users_total',
    'Number of active users'
)

database_connections = Gauge(
    'database_connections_active',
    'Active database connections'
)

# 在API中使用指标
from fastapi import Request, Response
import time

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # 记录请求指标
    duration = time.time() - start_time
    
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response

# 业务指标
@router.post("/users")
async def create_user(user_data: UserCreate):
    user = await user_service.create_user(user_data)
    
    # 更新活跃用户数
    active_count = await user_service.get_active_user_count()
    active_users.set(active_count)
    
    return user

# 系统指标
import psutil

def collect_system_metrics():
    # CPU使用率
    cpu_percent = psutil.cpu_percent()
    metrics.set_gauge('system_cpu_percent', cpu_percent)
    
    # 内存使用率
    memory = psutil.virtual_memory()
    metrics.set_gauge('system_memory_percent', memory.percent)
    metrics.set_gauge('system_memory_used_bytes', memory.used)
    
    # 磁盘使用率
    disk = psutil.disk_usage('/')
    metrics.set_gauge('system_disk_percent', disk.percent)
    
    # 网络IO
    network = psutil.net_io_counters()
    metrics.set_counter('system_network_bytes_sent', network.bytes_sent)
    metrics.set_counter('system_network_bytes_recv', network.bytes_recv)
```

## 🔧 高级功能

### 1. 分布式追踪

```python
from templates.monitoring import TracingManager, TraceContext
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# 配置追踪
tracing_manager = TracingManager(
    service_name="my-api",
    jaeger_endpoint="http://localhost:14268/api/traces",
    sampling_rate=0.1
)

# 手动创建span
async def process_order(order_id: int):
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span("process_order") as span:
        span.set_attribute("order.id", order_id)
        
        # 验证订单
        with tracer.start_as_current_span("validate_order") as validate_span:
            is_valid = await validate_order(order_id)
            validate_span.set_attribute("order.valid", is_valid)
            
            if not is_valid:
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Invalid order"))
                return False
        
        # 处理支付
        with tracer.start_as_current_span("process_payment") as payment_span:
            payment_result = await process_payment(order_id)
            payment_span.set_attribute("payment.success", payment_result.success)
            payment_span.set_attribute("payment.amount", payment_result.amount)
        
        # 更新库存
        with tracer.start_as_current_span("update_inventory") as inventory_span:
            await update_inventory(order_id)
        
        span.set_attribute("order.processed", True)
        return True

# 自动追踪装饰器
from templates.monitoring import trace_function

@trace_function("user_service")
async def get_user_profile(user_id: int):
    # 自动创建span并记录参数
    user = await db.get_user(user_id)
    return user

# 追踪HTTP请求
@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span(
        f"{request.method} {request.url.path}"
    ) as span:
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.url", str(request.url))
        span.set_attribute("http.user_agent", request.headers.get("user-agent", ""))
        
        response = await call_next(request)
        
        span.set_attribute("http.status_code", response.status_code)
        
        if response.status_code >= 400:
            span.set_status(trace.Status(trace.StatusCode.ERROR))
        
        return response
```

### 2. 错误追踪和告警

```python
from templates.monitoring import ErrorTracker, AlertManager
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

# 配置Sentry
sentry_sdk.init(
    dsn="https://your-sentry-dsn@sentry.io/project-id",
    integrations=[
        FastApiIntegration(auto_enabling_integrations=False),
        SqlalchemyIntegration(),
    ],
    traces_sample_rate=0.1,
    environment="production"
)

# 错误追踪器
class CustomErrorTracker(ErrorTracker):
    def __init__(self):
        self.error_counts = {}
        self.alert_manager = AlertManager()
    
    async def track_error(self, error: Exception, context: dict = None):
        error_type = type(error).__name__
        
        # 记录错误
        logger.error(
            "Error occurred",
            error_type=error_type,
            error_message=str(error),
            context=context or {},
            exc_info=error
        )
        
        # 发送到Sentry
        with sentry_sdk.push_scope() as scope:
            if context:
                for key, value in context.items():
                    scope.set_tag(key, value)
            sentry_sdk.capture_exception(error)
        
        # 错误计数和告警
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        if self.error_counts[error_type] > 10:  # 错误次数超过阈值
            await self.alert_manager.send_alert(
                level="critical",
                message=f"High error rate for {error_type}: {self.error_counts[error_type]} occurrences",
                context=context
            )

error_tracker = CustomErrorTracker()

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    await error_tracker.track_error(exc, {
        "request_path": request.url.path,
        "request_method": request.method,
        "user_agent": request.headers.get("user-agent")
    })
    
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "request_id": request.state.request_id}
    )

# 业务错误追踪
async def process_payment(order_id: int, amount: float):
    try:
        result = await payment_service.charge(order_id, amount)
        return result
    except PaymentError as e:
        await error_tracker.track_error(e, {
            "order_id": order_id,
            "amount": amount,
            "payment_method": "credit_card"
        })
        raise
    except Exception as e:
        await error_tracker.track_error(e, {
            "order_id": order_id,
            "amount": amount,
            "operation": "payment_processing"
        })
        raise
```

### 3. 性能监控

```python
from templates.monitoring import PerformanceMonitor, ProfilerManager
import cProfile
import pstats
from io import StringIO

# 性能监控器
class ApplicationPerformanceMonitor(PerformanceMonitor):
    def __init__(self):
        self.slow_queries = []
        self.slow_requests = []
        self.memory_usage = []
    
    async def monitor_database_query(self, query: str, duration: float):
        if duration > 1.0:  # 慢查询阈值1秒
            self.slow_queries.append({
                "query": query,
                "duration": duration,
                "timestamp": datetime.utcnow()
            })
            
            logger.warning(
                "Slow database query detected",
                query=query[:100],  # 截断长查询
                duration=duration
            )
    
    async def monitor_request(self, path: str, method: str, duration: float):
        if duration > 2.0:  # 慢请求阈值2秒
            self.slow_requests.append({
                "path": path,
                "method": method,
                "duration": duration,
                "timestamp": datetime.utcnow()
            })
            
            logger.warning(
                "Slow request detected",
                path=path,
                method=method,
                duration=duration
            )
    
    def get_performance_report(self) -> dict:
        return {
            "slow_queries_count": len(self.slow_queries),
            "slow_requests_count": len(self.slow_requests),
            "avg_query_time": self._calculate_avg_query_time(),
            "avg_request_time": self._calculate_avg_request_time()
        }

perf_monitor = ApplicationPerformanceMonitor()

# 数据库查询监控
class MonitoredDatabase:
    def __init__(self, db_manager, perf_monitor):
        self.db = db_manager
        self.perf_monitor = perf_monitor
    
    async def execute_query(self, query: str, params=None):
        start_time = time.time()
        try:
            with self.db.get_session() as session:
                result = session.execute(query, params)
                return result
        finally:
            duration = time.time() - start_time
            await self.perf_monitor.monitor_database_query(query, duration)

# 代码性能分析
class CodeProfiler:
    def __init__(self):
        self.profiler = cProfile.Profile()
    
    def start_profiling(self):
        self.profiler.enable()
    
    def stop_profiling(self) -> str:
        self.profiler.disable()
        
        # 生成性能报告
        s = StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # 显示前20个最耗时的函数
        
        return s.getvalue()
    
    def profile_function(self, func):
        def wrapper(*args, **kwargs):
            self.start_profiling()
            try:
                return func(*args, **kwargs)
            finally:
                report = self.stop_profiling()
                logger.info("Function profiling report", 
                           function=func.__name__, 
                           report=report)
        return wrapper

profiler = CodeProfiler()

# 使用性能分析
@profiler.profile_function
def expensive_computation(data):
    # 耗时计算
    result = complex_algorithm(data)
    return result
```

### 4. 日志聚合和分析

```python
from templates.monitoring import LogAggregator, LogAnalyzer
from elasticsearch import Elasticsearch

# 日志聚合器
class ElasticsearchLogAggregator(LogAggregator):
    def __init__(self, es_hosts):
        self.es = Elasticsearch(es_hosts)
        self.index_name = "application-logs"
    
    async def send_log(self, log_entry: dict):
        try:
            await self.es.index(
                index=self.index_name,
                body=log_entry
            )
        except Exception as e:
            # 避免日志发送失败影响主业务
            print(f"Failed to send log to Elasticsearch: {e}")
    
    async def search_logs(self, query: dict, size: int = 100):
        try:
            response = await self.es.search(
                index=self.index_name,
                body=query,
                size=size
            )
            return response['hits']['hits']
        except Exception as e:
            logger.error("Failed to search logs", error=str(e))
            return []

# 日志分析器
class LogAnalyzer:
    def __init__(self, log_aggregator):
        self.aggregator = log_aggregator
    
    async def analyze_error_patterns(self, time_range: str = "1h"):
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"level": "ERROR"}},
                        {"range": {
                            "timestamp": {
                                "gte": f"now-{time_range}"
                            }
                        }}
                    ]
                }
            },
            "aggs": {
                "error_types": {
                    "terms": {
                        "field": "error_type.keyword",
                        "size": 10
                    }
                },
                "error_timeline": {
                    "date_histogram": {
                        "field": "timestamp",
                        "interval": "5m"
                    }
                }
            }
        }
        
        results = await self.aggregator.search_logs(query)
        return self._process_error_analysis(results)
    
    async def analyze_performance_trends(self, endpoint: str = None):
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"exists": {"field": "duration_ms"}}
                    ]
                }
            },
            "aggs": {
                "avg_duration": {
                    "avg": {"field": "duration_ms"}
                },
                "max_duration": {
                    "max": {"field": "duration_ms"}
                },
                "duration_percentiles": {
                    "percentiles": {
                        "field": "duration_ms",
                        "percents": [50, 90, 95, 99]
                    }
                }
            }
        }
        
        if endpoint:
            query["query"]["bool"]["must"].append(
                {"term": {"endpoint.keyword": endpoint}}
            )
        
        results = await self.aggregator.search_logs(query)
        return self._process_performance_analysis(results)

log_aggregator = ElasticsearchLogAggregator(["localhost:9200"])
log_analyzer = LogAnalyzer(log_aggregator)
```

## 📝 最佳实践

### 1. 日志结构化

```python
# 标准化日志字段
class LogFields:
    REQUEST_ID = "request_id"
    USER_ID = "user_id"
    SESSION_ID = "session_id"
    OPERATION = "operation"
    DURATION_MS = "duration_ms"
    ERROR_CODE = "error_code"
    ERROR_TYPE = "error_type"
    ENDPOINT = "endpoint"
    METHOD = "method"
    STATUS_CODE = "status_code"

# 日志上下文管理
class LogContext:
    def __init__(self):
        self._context = {}
    
    def set(self, key: str, value):
        self._context[key] = value
    
    def get(self, key: str, default=None):
        return self._context.get(key, default)
    
    def clear(self):
        self._context.clear()
    
    def to_dict(self) -> dict:
        return self._context.copy()

# 全局日志上下文
log_context = LogContext()

# 中间件设置上下文
@app.middleware("http")
async def logging_context_middleware(request: Request, call_next):
    # 生成请求ID
    request_id = str(uuid.uuid4())
    log_context.set(LogFields.REQUEST_ID, request_id)
    log_context.set(LogFields.ENDPOINT, request.url.path)
    log_context.set(LogFields.METHOD, request.method)
    
    # 从请求头获取用户信息
    user_id = request.headers.get("X-User-ID")
    if user_id:
        log_context.set(LogFields.USER_ID, user_id)
    
    try:
        response = await call_next(request)
        log_context.set(LogFields.STATUS_CODE, response.status_code)
        return response
    finally:
        log_context.clear()

# 使用上下文记录日志
def log_with_context(level: str, message: str, **kwargs):
    context = log_context.to_dict()
    context.update(kwargs)
    
    getattr(logger, level)(message, **context)
```

### 2. 监控告警策略

```python
# 告警规则定义
class AlertRule:
    def __init__(self, name: str, condition, threshold, duration: int = 300):
        self.name = name
        self.condition = condition
        self.threshold = threshold
        self.duration = duration  # 持续时间(秒)
        self.triggered_at = None
    
    def check(self, value) -> bool:
        if self.condition(value, self.threshold):
            if self.triggered_at is None:
                self.triggered_at = time.time()
            elif time.time() - self.triggered_at >= self.duration:
                return True
        else:
            self.triggered_at = None
        
        return False

# 告警管理器
class AlertManager:
    def __init__(self):
        self.rules = []
        self.alert_channels = []
    
    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)
    
    def add_channel(self, channel):
        self.alert_channels.append(channel)
    
    async def check_alerts(self, metrics: dict):
        for rule in self.rules:
            metric_value = metrics.get(rule.name)
            if metric_value is not None and rule.check(metric_value):
                await self._send_alert(rule, metric_value)
    
    async def _send_alert(self, rule: AlertRule, value):
        alert_message = f"Alert: {rule.name} = {value} (threshold: {rule.threshold})"
        
        for channel in self.alert_channels:
            try:
                await channel.send(alert_message)
            except Exception as e:
                logger.error("Failed to send alert", channel=channel.__class__.__name__, error=str(e))

# 告警通道
class SlackAlertChannel:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def send(self, message: str):
        payload = {
            "text": message,
            "username": "MonitorBot",
            "icon_emoji": ":warning:"
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(self.webhook_url, json=payload)

class EmailAlertChannel:
    def __init__(self, smtp_config: dict, recipients: list):
        self.smtp_config = smtp_config
        self.recipients = recipients
    
    async def send(self, message: str):
        # 实现邮件发送逻辑
        pass

# 配置告警
alert_manager = AlertManager()

# 添加告警规则
alert_manager.add_rule(AlertRule(
    name="cpu_usage",
    condition=lambda x, threshold: x > threshold,
    threshold=80,
    duration=300  # 5分钟
))

alert_manager.add_rule(AlertRule(
    name="error_rate",
    condition=lambda x, threshold: x > threshold,
    threshold=0.05,  # 5%错误率
    duration=60  # 1分钟
))

# 添加告警通道
alert_manager.add_channel(SlackAlertChannel("https://hooks.slack.com/..."))
alert_manager.add_channel(EmailAlertChannel(smtp_config, ["admin@example.com"]))
```

### 3. 性能优化

```python
# 异步日志写入
import asyncio
from queue import Queue
from threading import Thread

class AsyncLogWriter:
    def __init__(self, log_file: str, batch_size: int = 100):
        self.log_file = log_file
        self.batch_size = batch_size
        self.log_queue = Queue()
        self.writer_thread = Thread(target=self._write_logs, daemon=True)
        self.writer_thread.start()
    
    def write_log(self, log_entry: dict):
        self.log_queue.put(log_entry)
    
    def _write_logs(self):
        batch = []
        
        while True:
            try:
                log_entry = self.log_queue.get(timeout=1)
                batch.append(log_entry)
                
                if len(batch) >= self.batch_size:
                    self._flush_batch(batch)
                    batch = []
                    
            except:
                if batch:
                    self._flush_batch(batch)
                    batch = []
    
    def _flush_batch(self, batch: list):
        with open(self.log_file, 'a') as f:
            for entry in batch:
                f.write(json.dumps(entry) + '\n')

# 采样日志
class SamplingLogger:
    def __init__(self, logger, sample_rate: float = 0.1):
        self.logger = logger
        self.sample_rate = sample_rate
    
    def should_log(self) -> bool:
        return random.random() < self.sample_rate
    
    def debug(self, message: str, **kwargs):
        if self.should_log():
            self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        # INFO级别总是记录
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        # WARNING级别总是记录
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        # ERROR级别总是记录
        self.logger.error(message, **kwargs)

# 条件日志
class ConditionalLogger:
    def __init__(self, logger):
        self.logger = logger
        self.conditions = {}
    
    def add_condition(self, level: str, condition):
        self.conditions[level] = condition
    
    def log(self, level: str, message: str, **kwargs):
        condition = self.conditions.get(level)
        
        if condition is None or condition(kwargs):
            getattr(self.logger, level)(message, **kwargs)

# 使用条件日志
conditional_logger = ConditionalLogger(logger)

# 只记录慢查询
conditional_logger.add_condition(
    'debug',
    lambda kwargs: kwargs.get('duration_ms', 0) > 100
)

# 只记录特定用户的日志
conditional_logger.add_condition(
    'info',
    lambda kwargs: kwargs.get('user_id') in ['admin', 'test_user']
)
```

## ❓ 常见问题

### Q: 如何处理大量日志数据？

A: 使用日志轮转、压缩和归档：

```python
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import gzip
import os

# 按大小轮转
rotating_handler = RotatingFileHandler(
    'app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)

# 按时间轮转
timed_handler = TimedRotatingFileHandler(
    'app.log',
    when='midnight',
    interval=1,
    backupCount=30
)

# 自动压缩旧日志
def compress_old_logs(log_dir: str):
    for filename in os.listdir(log_dir):
        if filename.endswith('.log') and filename != 'app.log':
            filepath = os.path.join(log_dir, filename)
            
            with open(filepath, 'rb') as f_in:
                with gzip.open(f'{filepath}.gz', 'wb') as f_out:
                    f_out.writelines(f_in)
            
            os.remove(filepath)
```

### Q: 如何监控微服务架构？

A: 使用分布式追踪和服务网格：

```python
# 服务间调用追踪
class ServiceTracer:
    def __init__(self, service_name: str):
        self.service_name = service_name
    
    async def call_service(self, service_url: str, data: dict, trace_id: str = None):
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        headers = {
            'X-Trace-ID': trace_id,
            'X-Service-Name': self.service_name
        }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(service_url, json=data, headers=headers) as response:
                    duration = time.time() - start_time
                    
                    logger.info(
                        "Service call completed",
                        trace_id=trace_id,
                        target_service=service_url,
                        duration_ms=duration * 1000,
                        status_code=response.status
                    )
                    
                    return await response.json()
        
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(
                "Service call failed",
                trace_id=trace_id,
                target_service=service_url,
                duration_ms=duration * 1000,
                error=str(e)
            )
            
            raise
```

### Q: 如何设置合适的监控指标？

A: 遵循RED和USE方法论：

```python
# RED指标 (Rate, Errors, Duration)
class REDMetrics:
    def __init__(self):
        self.request_rate = Counter('requests_per_second')
        self.error_rate = Counter('errors_per_second')
        self.request_duration = Histogram('request_duration_seconds')
    
    def record_request(self, duration: float, is_error: bool = False):
        self.request_rate.inc()
        self.request_duration.observe(duration)
        
        if is_error:
            self.error_rate.inc()

# USE指标 (Utilization, Saturation, Errors)
class USEMetrics:
    def __init__(self):
        self.cpu_utilization = Gauge('cpu_utilization_percent')
        self.memory_utilization = Gauge('memory_utilization_percent')
        self.disk_utilization = Gauge('disk_utilization_percent')
        self.queue_saturation = Gauge('queue_length')
        self.system_errors = Counter('system_errors_total')
    
    def update_system_metrics(self):
        # 更新系统指标
        self.cpu_utilization.set(psutil.cpu_percent())
        self.memory_utilization.set(psutil.virtual_memory().percent)
        self.disk_utilization.set(psutil.disk_usage('/').percent)
```

## 📚 相关文档

- [API开发模块使用指南](api.md)
- [数据库模块使用指南](database.md)
- [缓存消息模块使用指南](cache.md)
- [部署配置模块使用指南](deployment.md)
- [性能优化建议](best-practices/performance.md)
- [安全开发指南](best-practices/security.md)

---

如有其他问题，请查看 [GitHub Issues](https://github.com/your-username/python-backend-templates/issues) 或提交新的问题。