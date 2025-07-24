# 性能优化建议

本文档提供了Python后端应用性能优化的最佳实践和具体实施方案，帮助开发者构建高性能、可扩展的应用程序。

## 📋 目录

- [性能优化原则](#性能优化原则)
- [数据库优化](#数据库优化)
- [缓存策略](#缓存策略)
- [异步编程](#异步编程)
- [内存优化](#内存优化)
- [网络优化](#网络优化)
- [代码优化](#代码优化)
- [监控与分析](#监控与分析)
- [部署优化](#部署优化)
- [性能测试](#性能测试)

## 🎯 性能优化原则

### 1. 性能优化的黄金法则

```python
# 性能优化优先级
class PerformanceOptimizationPrinciples:
    """
    性能优化原则
    1. 测量优先 - 先测量，再优化
    2. 瓶颈识别 - 找到真正的性能瓶颈
    3. 渐进优化 - 逐步优化，避免过度优化
    4. 权衡取舍 - 平衡性能、可维护性和开发成本
    """
    
    @staticmethod
    def measure_first():
        """测量优先原则"""
        return {
            "profile_before_optimize": "使用性能分析工具测量当前性能",
            "identify_bottlenecks": "识别真正的性能瓶颈",
            "set_performance_goals": "设定明确的性能目标",
            "measure_after_optimize": "优化后重新测量验证效果"
        }
    
    @staticmethod
    def optimization_priorities():
        """优化优先级"""
        return [
            "1. 算法和数据结构优化",
            "2. 数据库查询优化",
            "3. 缓存策略实施",
            "4. 异步处理优化",
            "5. 内存使用优化",
            "6. 网络传输优化",
            "7. 代码级别优化"
        ]

# 性能监控装饰器
import time
import functools
import logging
from typing import Callable, Any

def performance_monitor(func_name: str = None):
    """性能监控装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                logger = logging.getLogger('performance')
                logger.info(
                    f"Function {func_name or func.__name__} "
                    f"executed in {execution_time:.4f} seconds"
                )
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                logger = logging.getLogger('performance')
                logger.info(
                    f"Function {func_name or func.__name__} "
                    f"executed in {execution_time:.4f} seconds"
                )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# 使用示例
@performance_monitor("user_data_processing")
async def process_user_data(user_id: int):
    # 处理用户数据的逻辑
    pass
```

### 2. 性能基准测试

```python
import asyncio
import time
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from statistics import mean, median, stdev

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    function_name: str
    total_runs: int
    total_time: float
    avg_time: float
    median_time: float
    min_time: float
    max_time: float
    std_dev: float
    requests_per_second: float

class PerformanceBenchmark:
    """性能基准测试工具"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    async def benchmark_async_function(
        self, 
        func: Callable, 
        args: tuple = (), 
        kwargs: dict = None, 
        runs: int = 100
    ) -> BenchmarkResult:
        """异步函数基准测试"""
        kwargs = kwargs or {}
        execution_times = []
        
        # 预热
        for _ in range(min(10, runs // 10)):
            await func(*args, **kwargs)
        
        # 正式测试
        start_total = time.time()
        for _ in range(runs):
            start_time = time.time()
            await func(*args, **kwargs)
            execution_times.append(time.time() - start_time)
        total_time = time.time() - start_total
        
        # 计算统计数据
        result = BenchmarkResult(
            function_name=func.__name__,
            total_runs=runs,
            total_time=total_time,
            avg_time=mean(execution_times),
            median_time=median(execution_times),
            min_time=min(execution_times),
            max_time=max(execution_times),
            std_dev=stdev(execution_times) if len(execution_times) > 1 else 0,
            requests_per_second=runs / total_time
        )
        
        self.results.append(result)
        return result
    
    def benchmark_sync_function(
        self, 
        func: Callable, 
        args: tuple = (), 
        kwargs: dict = None, 
        runs: int = 100
    ) -> BenchmarkResult:
        """同步函数基准测试"""
        kwargs = kwargs or {}
        execution_times = []
        
        # 预热
        for _ in range(min(10, runs // 10)):
            func(*args, **kwargs)
        
        # 正式测试
        start_total = time.time()
        for _ in range(runs):
            start_time = time.time()
            func(*args, **kwargs)
            execution_times.append(time.time() - start_time)
        total_time = time.time() - start_total
        
        # 计算统计数据
        result = BenchmarkResult(
            function_name=func.__name__,
            total_runs=runs,
            total_time=total_time,
            avg_time=mean(execution_times),
            median_time=median(execution_times),
            min_time=min(execution_times),
            max_time=max(execution_times),
            std_dev=stdev(execution_times) if len(execution_times) > 1 else 0,
            requests_per_second=runs / total_time
        )
        
        self.results.append(result)
        return result
    
    def print_results(self):
        """打印测试结果"""
        for result in self.results:
            print(f"\n=== {result.function_name} ===")
            print(f"总运行次数: {result.total_runs}")
            print(f"总时间: {result.total_time:.4f}s")
            print(f"平均时间: {result.avg_time:.4f}s")
            print(f"中位数时间: {result.median_time:.4f}s")
            print(f"最小时间: {result.min_time:.4f}s")
            print(f"最大时间: {result.max_time:.4f}s")
            print(f"标准差: {result.std_dev:.4f}s")
            print(f"每秒请求数: {result.requests_per_second:.2f} RPS")

# 使用示例
async def example_usage():
    benchmark = PerformanceBenchmark()
    
    # 测试异步函数
    async def async_task():
        await asyncio.sleep(0.01)
        return "done"
    
    result = await benchmark.benchmark_async_function(async_task, runs=50)
    benchmark.print_results()
```

## 🗄️ 数据库优化

### 1. 查询优化

```python
from sqlalchemy import text, func, and_, or_
from sqlalchemy.orm import joinedload, selectinload, subqueryload
from typing import List, Optional, Dict, Any

class DatabaseOptimization:
    """数据库优化工具类"""
    
    def __init__(self, session):
        self.session = session
    
    def optimized_pagination(
        self, 
        model, 
        page: int = 1, 
        per_page: int = 20,
        filters: Dict[str, Any] = None
    ):
        """优化的分页查询"""
        query = self.session.query(model)
        
        # 应用过滤条件
        if filters:
            for key, value in filters.items():
                if hasattr(model, key):
                    query = query.filter(getattr(model, key) == value)
        
        # 使用offset和limit进行分页
        offset = (page - 1) * per_page
        
        # 获取总数（优化：只在需要时计算）
        total = query.count()
        
        # 获取数据
        items = query.offset(offset).limit(per_page).all()
        
        return {
            'items': items,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page
        }
    
    def cursor_based_pagination(
        self, 
        model, 
        cursor_field: str = 'id',
        cursor_value: Any = None,
        limit: int = 20,
        direction: str = 'next'
    ):
        """基于游标的分页（适合大数据集）"""
        query = self.session.query(model)
        
        if cursor_value is not None:
            cursor_column = getattr(model, cursor_field)
            if direction == 'next':
                query = query.filter(cursor_column > cursor_value)
            else:
                query = query.filter(cursor_column < cursor_value)
        
        # 排序
        cursor_column = getattr(model, cursor_field)
        if direction == 'next':
            query = query.order_by(cursor_column.asc())
        else:
            query = query.order_by(cursor_column.desc())
        
        items = query.limit(limit + 1).all()
        
        has_more = len(items) > limit
        if has_more:
            items = items[:-1]
        
        next_cursor = None
        if items and has_more:
            next_cursor = getattr(items[-1], cursor_field)
        
        return {
            'items': items,
            'next_cursor': next_cursor,
            'has_more': has_more
        }
    
    def bulk_operations(self, model, data_list: List[Dict[str, Any]]):
        """批量操作优化"""
        # 批量插入
        if data_list:
            self.session.bulk_insert_mappings(model, data_list)
            self.session.commit()
    
    def optimized_joins(self, model, relationships: List[str]):
        """优化的关联查询"""
        query = self.session.query(model)
        
        for relationship in relationships:
            # 使用joinedload进行预加载，减少N+1查询问题
            query = query.options(joinedload(relationship))
        
        return query
    
    def raw_sql_optimization(self, sql: str, params: Dict[str, Any] = None):
        """原生SQL优化查询"""
        # 对于复杂查询，使用原生SQL可能更高效
        result = self.session.execute(text(sql), params or {})
        return result.fetchall()

# 数据库连接池优化
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

class OptimizedDatabaseManager:
    """优化的数据库管理器"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            # 连接池配置
            poolclass=QueuePool,
            pool_size=20,          # 连接池大小
            max_overflow=30,       # 最大溢出连接数
            pool_timeout=30,       # 获取连接超时时间
            pool_recycle=3600,     # 连接回收时间
            pool_pre_ping=True,    # 连接前ping检查
            
            # 查询优化
            echo=False,            # 生产环境关闭SQL日志
            future=True,           # 使用新的API
            
            # 连接参数
            connect_args={
                "connect_timeout": 10,
                "application_name": "MyApp",
            }
        )
    
    def get_session(self):
        """获取数据库会话"""
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=self.engine)
        return Session()

# 查询缓存
from functools import lru_cache
import hashlib
import json

class QueryCache:
    """查询结果缓存"""
    
    def __init__(self, redis_client=None, default_ttl: int = 300):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.local_cache = {}
    
    def _generate_cache_key(self, query_str: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        cache_data = {
            'query': query_str,
            'params': params
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return f"query_cache:{hashlib.md5(cache_str.encode()).hexdigest()}"
    
    async def get_cached_result(self, query_str: str, params: Dict[str, Any]):
        """获取缓存结果"""
        cache_key = self._generate_cache_key(query_str, params)
        
        if self.redis:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        else:
            return self.local_cache.get(cache_key)
        
        return None
    
    async def cache_result(
        self, 
        query_str: str, 
        params: Dict[str, Any], 
        result: Any, 
        ttl: int = None
    ):
        """缓存查询结果"""
        cache_key = self._generate_cache_key(query_str, params)
        ttl = ttl or self.default_ttl
        
        # 序列化结果
        serialized_result = json.dumps(result, default=str)
        
        if self.redis:
            await self.redis.setex(cache_key, ttl, serialized_result)
        else:
            self.local_cache[cache_key] = result
    
    def cache_query(self, ttl: int = None):
        """查询缓存装饰器"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = self._generate_cache_key(
                    func.__name__, 
                    {'args': args, 'kwargs': kwargs}
                )
                
                # 尝试从缓存获取
                cached_result = await self.get_cached_result(
                    func.__name__, 
                    {'args': args, 'kwargs': kwargs}
                )
                
                if cached_result is not None:
                    return cached_result
                
                # 执行查询
                result = await func(*args, **kwargs)
                
                # 缓存结果
                await self.cache_result(
                    func.__name__, 
                    {'args': args, 'kwargs': kwargs}, 
                    result, 
                    ttl
                )
                
                return result
            return wrapper
        return decorator
```

### 2. 索引优化

```python
from sqlalchemy import Index, text
from typing import List, Dict, Any

class IndexOptimization:
    """索引优化工具"""
    
    def __init__(self, session):
        self.session = session
    
    def analyze_query_performance(self, sql: str, params: Dict[str, Any] = None):
        """分析查询性能"""
        # PostgreSQL查询计划分析
        explain_sql = f"EXPLAIN ANALYZE {sql}"
        result = self.session.execute(text(explain_sql), params or {})
        
        execution_plan = []
        for row in result:
            execution_plan.append(row[0])
        
        return execution_plan
    
    def suggest_indexes(self, table_name: str, query_patterns: List[Dict[str, Any]]):
        """建议索引"""
        suggestions = []
        
        for pattern in query_patterns:
            where_columns = pattern.get('where_columns', [])
            order_columns = pattern.get('order_columns', [])
            join_columns = pattern.get('join_columns', [])
            
            # 建议复合索引
            if len(where_columns) > 1:
                suggestions.append({
                    'type': 'composite_index',
                    'columns': where_columns,
                    'reason': 'Multiple WHERE conditions'
                })
            
            # 建议排序索引
            if order_columns:
                suggestions.append({
                    'type': 'order_index',
                    'columns': order_columns,
                    'reason': 'ORDER BY optimization'
                })
            
            # 建议连接索引
            if join_columns:
                suggestions.append({
                    'type': 'join_index',
                    'columns': join_columns,
                    'reason': 'JOIN optimization'
                })
        
        return suggestions
    
    def create_optimized_indexes(self, model, index_definitions: List[Dict[str, Any]]):
        """创建优化索引"""
        for index_def in index_definitions:
            columns = [getattr(model, col) for col in index_def['columns']]
            
            index = Index(
                f"idx_{model.__tablename__}_{'_'.join(index_def['columns'])}",
                *columns,
                unique=index_def.get('unique', False)
            )
            
            # 在生产环境中，应该通过迁移脚本创建索引
            # 这里只是示例
            print(f"建议创建索引: {index}")

# 数据库监控
class DatabaseMonitoring:
    """数据库性能监控"""
    
    def __init__(self, session):
        self.session = session
    
    def get_slow_queries(self, threshold_ms: int = 1000):
        """获取慢查询"""
        # PostgreSQL慢查询监控
        sql = """
        SELECT query, mean_time, calls, total_time
        FROM pg_stat_statements
        WHERE mean_time > :threshold
        ORDER BY mean_time DESC
        LIMIT 20
        """
        
        result = self.session.execute(text(sql), {'threshold': threshold_ms})
        return result.fetchall()
    
    def get_table_stats(self, table_name: str):
        """获取表统计信息"""
        sql = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples,
            last_vacuum,
            last_autovacuum,
            last_analyze,
            last_autoanalyze
        FROM pg_stat_user_tables
        WHERE tablename = :table_name
        """
        
        result = self.session.execute(text(sql), {'table_name': table_name})
        return result.fetchone()
    
    def get_index_usage(self, table_name: str):
        """获取索引使用情况"""
        sql = """
        SELECT 
            indexrelname as index_name,
            idx_tup_read as index_reads,
            idx_tup_fetch as index_fetches,
            idx_scan as index_scans
        FROM pg_stat_user_indexes
        WHERE relname = :table_name
        ORDER BY idx_scan DESC
        """
        
        result = self.session.execute(text(sql), {'table_name': table_name})
        return result.fetchall()
```

## 🚀 缓存策略

### 1. 多层缓存架构

```python
import asyncio
import json
import time
from typing import Any, Optional, Dict, Union
from abc import ABC, abstractmethod
from enum import Enum

class CacheLevel(Enum):
    """缓存级别"""
    L1_MEMORY = "l1_memory"      # 内存缓存
    L2_REDIS = "l2_redis"        # Redis缓存
    L3_DATABASE = "l3_database"  # 数据库缓存

class CacheStrategy(Enum):
    """缓存策略"""
    WRITE_THROUGH = "write_through"    # 写穿透
    WRITE_BACK = "write_back"          # 写回
    WRITE_AROUND = "write_around"      # 写绕过

class CacheBackend(ABC):
    """缓存后端抽象类"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        pass

class MemoryCache(CacheBackend):
    """内存缓存实现"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            item = self.cache[key]
            if item['expires_at'] is None or item['expires_at'] > time.time():
                self.access_times[key] = time.time()
                return item['value']
            else:
                await self.delete(key)
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        # LRU淘汰策略
        if len(self.cache) >= self.max_size and key not in self.cache:
            lru_key = min(self.access_times.keys(), key=self.access_times.get)
            await self.delete(lru_key)
        
        expires_at = None
        if ttl:
            expires_at = time.time() + ttl
        
        self.cache[key] = {
            'value': value,
            'expires_at': expires_at
        }
        self.access_times[key] = time.time()
        return True
    
    async def delete(self, key: str) -> bool:
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        return key in self.cache

class RedisCache(CacheBackend):
    """Redis缓存实现"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def get(self, key: str) -> Optional[Any]:
        value = await self.redis.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value.decode() if isinstance(value, bytes) else value
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        try:
            serialized_value = json.dumps(value)
        except (TypeError, ValueError):
            serialized_value = str(value)
        
        if ttl:
            return await self.redis.setex(key, ttl, serialized_value)
        else:
            return await self.redis.set(key, serialized_value)
    
    async def delete(self, key: str) -> bool:
        return await self.redis.delete(key) > 0
    
    async def exists(self, key: str) -> bool:
        return await self.redis.exists(key) > 0

class MultiLevelCache:
    """多层缓存管理器"""
    
    def __init__(self):
        self.backends: Dict[CacheLevel, CacheBackend] = {}
        self.strategy = CacheStrategy.WRITE_THROUGH
    
    def add_backend(self, level: CacheLevel, backend: CacheBackend):
        """添加缓存后端"""
        self.backends[level] = backend
    
    async def get(self, key: str) -> Optional[Any]:
        """多层缓存获取"""
        # 按优先级顺序查找
        for level in [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]:
            if level in self.backends:
                value = await self.backends[level].get(key)
                if value is not None:
                    # 回填到更高级别的缓存
                    await self._backfill_cache(key, value, level)
                    return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """多层缓存设置"""
        success = True
        
        if self.strategy == CacheStrategy.WRITE_THROUGH:
            # 写穿透：同时写入所有层级
            for backend in self.backends.values():
                result = await backend.set(key, value, ttl)
                success = success and result
        
        elif self.strategy == CacheStrategy.WRITE_BACK:
            # 写回：只写入最高级别缓存
            if CacheLevel.L1_MEMORY in self.backends:
                success = await self.backends[CacheLevel.L1_MEMORY].set(key, value, ttl)
        
        return success
    
    async def delete(self, key: str) -> bool:
        """删除所有层级的缓存"""
        success = True
        for backend in self.backends.values():
            result = await backend.delete(key)
            success = success and result
        return success
    
    async def _backfill_cache(self, key: str, value: Any, found_level: CacheLevel):
        """回填缓存到更高级别"""
        levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DATABASE]
        found_index = levels.index(found_level)
        
        # 回填到更高级别的缓存
        for i in range(found_index):
            level = levels[i]
            if level in self.backends:
                await self.backends[level].set(key, value)

# 缓存装饰器
class CacheDecorator:
    """缓存装饰器"""
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
    
    def cached(self, ttl: int = 300, key_prefix: str = ""):
        """缓存装饰器"""
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = self._generate_cache_key(func, args, kwargs, key_prefix)
                
                # 尝试从缓存获取
                cached_result = await self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # 执行函数
                result = await func(*args, **kwargs)
                
                # 缓存结果
                await self.cache.set(cache_key, result, ttl)
                
                return result
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # 同步版本的实现
                cache_key = self._generate_cache_key(func, args, kwargs, key_prefix)
                
                # 这里需要在事件循环中运行异步代码
                loop = asyncio.get_event_loop()
                
                cached_result = loop.run_until_complete(self.cache.get(cache_key))
                if cached_result is not None:
                    return cached_result
                
                result = func(*args, **kwargs)
                loop.run_until_complete(self.cache.set(cache_key, result, ttl))
                
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def _generate_cache_key(self, func, args, kwargs, prefix: str) -> str:
        """生成缓存键"""
        import hashlib
        
        key_data = {
            'function': func.__name__,
            'args': args,
            'kwargs': kwargs
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"{prefix}:{func.__name__}:{key_hash}" if prefix else f"{func.__name__}:{key_hash}"

# 使用示例
async def setup_cache_system():
    # 创建多层缓存
    cache = MultiLevelCache()
    
    # 添加内存缓存
    memory_cache = MemoryCache(max_size=1000)
    cache.add_backend(CacheLevel.L1_MEMORY, memory_cache)
    
    # 添加Redis缓存
    import aioredis
    redis_client = aioredis.from_url("redis://localhost")
    redis_cache = RedisCache(redis_client)
    cache.add_backend(CacheLevel.L2_REDIS, redis_cache)
    
    # 创建缓存装饰器
    cache_decorator = CacheDecorator(cache)
    
    # 使用缓存装饰器
    @cache_decorator.cached(ttl=600, key_prefix="user_data")
    async def get_user_data(user_id: int):
        # 模拟数据库查询
        await asyncio.sleep(0.1)
        return {"user_id": user_id, "name": f"User {user_id}"}
    
    # 测试缓存
    result1 = await get_user_data(123)  # 从数据库获取
    result2 = await get_user_data(123)  # 从缓存获取
    
    return cache
```

### 2. 缓存失效策略

```python
import asyncio
import time
from typing import Set, Dict, List, Callable, Any
from dataclasses import dataclass
from enum import Enum

class InvalidationStrategy(Enum):
    """缓存失效策略"""
    TTL = "ttl"                    # 基于时间的失效
    TAG_BASED = "tag_based"        # 基于标签的失效
    DEPENDENCY = "dependency"      # 基于依赖的失效
    MANUAL = "manual"              # 手动失效

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: float
    ttl: int
    tags: Set[str]
    dependencies: Set[str]
    access_count: int = 0
    last_access: float = 0

class CacheInvalidationManager:
    """缓存失效管理器"""
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self.entries: Dict[str, CacheEntry] = {}
        self.tag_mapping: Dict[str, Set[str]] = {}  # tag -> keys
        self.dependency_mapping: Dict[str, Set[str]] = {}  # dependency -> keys
        self.invalidation_callbacks: List[Callable] = []
    
    async def set_with_metadata(
        self, 
        key: str, 
        value: Any, 
        ttl: int = None,
        tags: Set[str] = None,
        dependencies: Set[str] = None
    ):
        """设置缓存并记录元数据"""
        tags = tags or set()
        dependencies = dependencies or set()
        
        # 创建缓存条目
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl=ttl,
            tags=tags,
            dependencies=dependencies
        )
        
        # 存储到缓存
        await self.cache.set(key, value, ttl)
        
        # 记录元数据
        self.entries[key] = entry
        
        # 更新标签映射
        for tag in tags:
            if tag not in self.tag_mapping:
                self.tag_mapping[tag] = set()
            self.tag_mapping[tag].add(key)
        
        # 更新依赖映射
        for dep in dependencies:
            if dep not in self.dependency_mapping:
                self.dependency_mapping[dep] = set()
            self.dependency_mapping[dep].add(key)
    
    async def invalidate_by_tag(self, tag: str):
        """根据标签失效缓存"""
        if tag in self.tag_mapping:
            keys_to_invalidate = self.tag_mapping[tag].copy()
            
            for key in keys_to_invalidate:
                await self._invalidate_key(key)
            
            # 清理标签映射
            del self.tag_mapping[tag]
    
    async def invalidate_by_dependency(self, dependency: str):
        """根据依赖失效缓存"""
        if dependency in self.dependency_mapping:
            keys_to_invalidate = self.dependency_mapping[dependency].copy()
            
            for key in keys_to_invalidate:
                await self._invalidate_key(key)
            
            # 清理依赖映射
            del self.dependency_mapping[dependency]
    
    async def invalidate_expired(self):
        """失效过期的缓存"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.entries.items():
            if entry.ttl and (entry.created_at + entry.ttl) < current_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            await self._invalidate_key(key)
    
    async def _invalidate_key(self, key: str):
        """失效单个缓存键"""
        # 从缓存中删除
        await self.cache.delete(key)
        
        # 清理元数据
        if key in self.entries:
            entry = self.entries[key]
            
            # 清理标签映射
            for tag in entry.tags:
                if tag in self.tag_mapping:
                    self.tag_mapping[tag].discard(key)
                    if not self.tag_mapping[tag]:
                        del self.tag_mapping[tag]
            
            # 清理依赖映射
            for dep in entry.dependencies:
                if dep in self.dependency_mapping:
                    self.dependency_mapping[dep].discard(key)
                    if not self.dependency_mapping[dep]:
                        del self.dependency_mapping[dep]
            
            del self.entries[key]
        
        # 调用失效回调
        for callback in self.invalidation_callbacks:
            try:
                await callback(key)
            except Exception as e:
                print(f"Invalidation callback error: {e}")
    
    def add_invalidation_callback(self, callback: Callable):
        """添加失效回调"""
        self.invalidation_callbacks.append(callback)
    
    async def start_cleanup_task(self, interval: int = 60):
        """启动清理任务"""
        while True:
            try:
                await self.invalidate_expired()
                await asyncio.sleep(interval)
            except Exception as e:
                print(f"Cache cleanup error: {e}")
                await asyncio.sleep(interval)

# 智能缓存预热
class CacheWarmup:
    """缓存预热管理器"""
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self.warmup_tasks: List[Dict[str, Any]] = []
    
    def add_warmup_task(
        self, 
        func: Callable, 
        args: tuple = (), 
        kwargs: dict = None,
        priority: int = 1,
        schedule: str = "startup"  # startup, periodic, on_demand
    ):
        """添加预热任务"""
        self.warmup_tasks.append({
            'func': func,
            'args': args,
            'kwargs': kwargs or {},
            'priority': priority,
            'schedule': schedule
        })
    
    async def execute_warmup(self, schedule_type: str = "startup"):
        """执行预热任务"""
        tasks = [
            task for task in self.warmup_tasks 
            if task['schedule'] == schedule_type
        ]
        
        # 按优先级排序
        tasks.sort(key=lambda x: x['priority'], reverse=True)
        
        for task in tasks:
            try:
                func = task['func']
                args = task['args']
                kwargs = task['kwargs']
                
                if asyncio.iscoroutinefunction(func):
                    await func(*args, **kwargs)
                else:
                    func(*args, **kwargs)
                    
            except Exception as e:
                print(f"Warmup task error: {e}")
    
    async def intelligent_warmup(self, access_patterns: Dict[str, int]):
        """基于访问模式的智能预热"""
        # 根据访问频率预热热点数据
        sorted_patterns = sorted(
            access_patterns.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for cache_key, access_count in sorted_patterns[:10]:  # 预热前10个热点
            # 这里需要根据实际业务逻辑来重新生成数据
            print(f"Warming up cache key: {cache_key} (accessed {access_count} times)")

# 使用示例
async def cache_invalidation_example():
    # 设置缓存系统
    cache = MultiLevelCache()
    memory_cache = MemoryCache()
    cache.add_backend(CacheLevel.L1_MEMORY, memory_cache)
    
    # 创建失效管理器
    invalidation_manager = CacheInvalidationManager(cache)
    
    # 设置带标签的缓存
    await invalidation_manager.set_with_metadata(
        "user:123",
        {"name": "John", "email": "john@example.com"},
        ttl=3600,
        tags={"user", "profile"},
        dependencies={"user_table"}
    )
    
    # 根据标签失效
    await invalidation_manager.invalidate_by_tag("user")
    
    # 根据依赖失效
    await invalidation_manager.invalidate_by_dependency("user_table")
    
    # 启动清理任务
    asyncio.create_task(invalidation_manager.start_cleanup_task())
```