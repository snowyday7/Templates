"""性能优化模块

提供企业级性能优化功能，包括：
- 连接池管理
- 缓存策略
- 异步处理优化
- 性能监控
- 资源管理
"""

# 连接池管理
from .connection_pool import (
    ConnectionPoolManager,
    DatabaseConnectionPool,
    RedisConnectionPool,
    HTTPConnectionPool,
    get_db_pool,
    get_redis_pool,
    get_http_pool,
)

# 缓存策略
from .cache_strategy import (
    CacheStrategy,
    LRUCache,
    TTLCache,
    MultiLevelCache,
    DistributedCache,
    CacheManager,
    cache_result,
    invalidate_cache,
    warm_cache,
)

# 异步处理
from .async_optimization import (
    AsyncTaskManager,
    TaskQueue,
    WorkerPool,
    BatchProcessor,
    AsyncLimiter,
    async_retry,
    batch_process,
    parallel_execute,
)

# 性能监控
from .performance_monitor import (
    PerformanceProfiler,
    QueryProfiler,
    MemoryProfiler,
    CPUProfiler,
    IOProfiler,
    profile_function,
    monitor_performance,
    get_performance_stats,
)

# 资源管理
from .resource_manager import (
    ResourceManager,
    MemoryManager,
    FileHandleManager,
    ThreadManager,
    ProcessManager,
    cleanup_resources,
    monitor_resources,
    optimize_memory,
)

__all__ = [
    # 连接池管理
    "ConnectionPoolManager",
    "DatabaseConnectionPool",
    "RedisConnectionPool",
    "HTTPConnectionPool",
    "get_db_pool",
    "get_redis_pool",
    "get_http_pool",
    
    # 缓存策略
    "CacheStrategy",
    "LRUCache",
    "TTLCache",
    "MultiLevelCache",
    "DistributedCache",
    "CacheManager",
    "cache_result",
    "invalidate_cache",
    "warm_cache",
    
    # 异步处理
    "AsyncTaskManager",
    "TaskQueue",
    "WorkerPool",
    "BatchProcessor",
    "AsyncLimiter",
    "async_retry",
    "batch_process",
    "parallel_execute",
    
    # 性能监控
    "PerformanceProfiler",
    "QueryProfiler",
    "MemoryProfiler",
    "CPUProfiler",
    "IOProfiler",
    "profile_function",
    "monitor_performance",
    "get_performance_stats",
    
    # 资源管理
    "ResourceManager",
    "MemoryManager",
    "FileHandleManager",
    "ThreadManager",
    "ProcessManager",
    "cleanup_resources",
    "monitor_resources",
    "optimize_memory",
]