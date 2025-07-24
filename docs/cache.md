# 缓存消息模块使用指南

缓存消息模块提供了Redis缓存管理、Celery异步任务队列、消息队列等功能，帮助提升应用性能和处理能力。

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
pip install redis celery[redis] kombu
```

### 基础配置

```python
from templates.cache import RedisManager, CeleryManager, CacheConfig

# 创建缓存配置
cache_config = CacheConfig(
    REDIS_HOST="localhost",
    REDIS_PORT=6379,
    REDIS_DB=0,
    REDIS_PASSWORD=None,
    CACHE_DEFAULT_TIMEOUT=300
)

# 创建Redis管理器
redis_manager = RedisManager(cache_config)

# 创建Celery管理器
celery_manager = CeleryManager(
    broker_url="redis://localhost:6379/0",
    result_backend="redis://localhost:6379/0"
)
```

## ⚙️ 配置说明

### CacheConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `REDIS_HOST` | str | "localhost" | Redis主机地址 |
| `REDIS_PORT` | int | 6379 | Redis端口 |
| `REDIS_DB` | int | 0 | Redis数据库编号 |
| `REDIS_PASSWORD` | str | None | Redis密码 |
| `REDIS_MAX_CONNECTIONS` | int | 20 | 最大连接数 |
| `REDIS_SOCKET_TIMEOUT` | int | 5 | 套接字超时时间(秒) |
| `REDIS_CONNECTION_TIMEOUT` | int | 5 | 连接超时时间(秒) |
| `CACHE_DEFAULT_TIMEOUT` | int | 300 | 默认缓存超时时间(秒) |
| `CACHE_KEY_PREFIX` | str | "myapp" | 缓存键前缀 |
| `CACHE_VERSION` | int | 1 | 缓存版本 |

### Celery配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `CELERY_BROKER_URL` | str | 必填 | 消息代理URL |
| `CELERY_RESULT_BACKEND` | str | 必填 | 结果后端URL |
| `CELERY_TASK_SERIALIZER` | str | "json" | 任务序列化格式 |
| `CELERY_RESULT_SERIALIZER` | str | "json" | 结果序列化格式 |
| `CELERY_ACCEPT_CONTENT` | list | ["json"] | 接受的内容类型 |
| `CELERY_TIMEZONE` | str | "UTC" | 时区 |
| `CELERY_ENABLE_UTC` | bool | True | 启用UTC |
| `CELERY_TASK_ROUTES` | dict | {} | 任务路由配置 |
| `CELERY_WORKER_CONCURRENCY` | int | 4 | 工作进程并发数 |

### 环境变量配置

创建 `.env` 文件：

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_MAX_CONNECTIONS=20
CACHE_DEFAULT_TIMEOUT=300
CACHE_KEY_PREFIX=myapp

CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
CELERY_WORKER_CONCURRENCY=4
CELERY_TIMEZONE=Asia/Shanghai
```

## 💻 基础使用

### 1. Redis缓存操作

```python
from templates.cache import RedisManager
import json
from datetime import timedelta

# 基础缓存操作
async def cache_examples():
    # 设置缓存
    await redis_manager.set("user:1", {"name": "John", "age": 30}, timeout=3600)
    
    # 获取缓存
    user_data = await redis_manager.get("user:1")
    print(user_data)  # {'name': 'John', 'age': 30}
    
    # 检查键是否存在
    exists = await redis_manager.exists("user:1")
    print(exists)  # True
    
    # 删除缓存
    await redis_manager.delete("user:1")
    
    # 批量操作
    await redis_manager.set_many({
        "user:1": {"name": "John"},
        "user:2": {"name": "Jane"},
        "user:3": {"name": "Bob"}
    }, timeout=1800)
    
    users = await redis_manager.get_many(["user:1", "user:2", "user:3"])
    print(users)  # [{'name': 'John'}, {'name': 'Jane'}, {'name': 'Bob'}]
    
    # 删除匹配模式的键
    await redis_manager.delete_pattern("user:*")

# 缓存装饰器
from templates.cache import cache_result

@cache_result(timeout=600, key_prefix="api_data")
async def get_expensive_data(user_id: int, category: str):
    """模拟耗时的数据获取操作"""
    await asyncio.sleep(2)  # 模拟耗时操作
    return {
        "user_id": user_id,
        "category": category,
        "data": f"expensive data for user {user_id}",
        "timestamp": datetime.utcnow().isoformat()
    }

# 使用缓存装饰器
result = await get_expensive_data(123, "products")
# 第一次调用会执行函数，后续调用会从缓存返回
```

### 2. Celery异步任务

```python
from templates.cache import CeleryManager
from celery import Celery

# 创建Celery应用
celery_app = celery_manager.create_app()

# 定义任务
@celery_app.task(bind=True, max_retries=3)
def send_email_task(self, to_email: str, subject: str, content: str):
    """发送邮件任务"""
    try:
        # 模拟发送邮件
        import time
        time.sleep(2)  # 模拟邮件发送时间
        
        print(f"Email sent to {to_email}: {subject}")
        return {"status": "success", "email": to_email}
        
    except Exception as exc:
        # 重试机制
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60, exc=exc)
        else:
            return {"status": "failed", "error": str(exc)}

@celery_app.task
def process_image_task(image_path: str, filters: list):
    """图片处理任务"""
    # 模拟图片处理
    processed_path = f"processed_{image_path}"
    return {
        "original_path": image_path,
        "processed_path": processed_path,
        "filters_applied": filters
    }

@celery_app.task
def generate_report_task(user_id: int, report_type: str):
    """生成报告任务"""
    # 模拟报告生成
    import time
    time.sleep(10)  # 模拟长时间处理
    
    return {
        "user_id": user_id,
        "report_type": report_type,
        "report_url": f"/reports/{user_id}_{report_type}.pdf",
        "generated_at": datetime.utcnow().isoformat()
    }

# 在API中使用任务
from fastapi import APIRouter, BackgroundTasks

router = APIRouter(prefix="/tasks", tags=["tasks"])

@router.post("/send-email")
async def send_email(to_email: str, subject: str, content: str):
    # 异步执行任务
    task = send_email_task.delay(to_email, subject, content)
    
    return {
        "task_id": task.id,
        "status": "queued",
        "message": "Email task queued successfully"
    }

@router.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    # 获取任务状态
    task = celery_app.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {
            "task_id": task_id,
            "state": task.state,
            "status": "Task is waiting to be processed"
        }
    elif task.state == 'PROGRESS':
        response = {
            "task_id": task_id,
            "state": task.state,
            "current": task.info.get('current', 0),
            "total": task.info.get('total', 1),
            "status": task.info.get('status', '')
        }
    elif task.state == 'SUCCESS':
        response = {
            "task_id": task_id,
            "state": task.state,
            "result": task.result
        }
    else:  # FAILURE
        response = {
            "task_id": task_id,
            "state": task.state,
            "error": str(task.info)
        }
    
    return response
```

### 3. 消息队列

```python
from templates.cache import MessageQueue
import asyncio

# 创建消息队列
message_queue = MessageQueue(
    broker_url="redis://localhost:6379/0",
    queue_name="notifications"
)

# 发布消息
async def publish_notification(user_id: int, message: str, notification_type: str):
    notification = {
        "user_id": user_id,
        "message": message,
        "type": notification_type,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await message_queue.publish("user.notification", notification)
    print(f"Notification published for user {user_id}")

# 消费消息
async def consume_notifications():
    async def handle_notification(message):
        notification = message['body']
        print(f"Processing notification: {notification}")
        
        # 处理通知逻辑
        if notification['type'] == 'email':
            await send_email_notification(notification)
        elif notification['type'] == 'push':
            await send_push_notification(notification)
        elif notification['type'] == 'sms':
            await send_sms_notification(notification)
    
    await message_queue.consume("user.notification", handle_notification)

# 启动消费者
async def start_consumer():
    await consume_notifications()

# 在后台任务中运行
if __name__ == "__main__":
    asyncio.run(start_consumer())
```

## 🔧 高级功能

### 1. 分布式锁

```python
from templates.cache import DistributedLock
import asyncio

# 分布式锁使用
async def process_user_data(user_id: int):
    lock_key = f"process_user:{user_id}"
    
    async with DistributedLock(redis_manager, lock_key, timeout=30) as lock:
        if lock.acquired:
            print(f"Processing user {user_id}...")
            
            # 执行需要互斥的操作
            await update_user_balance(user_id)
            await send_notification(user_id)
            
            print(f"User {user_id} processing completed")
        else:
            print(f"Could not acquire lock for user {user_id}")

# 手动锁管理
async def manual_lock_example():
    lock = DistributedLock(redis_manager, "my_resource", timeout=60)
    
    try:
        acquired = await lock.acquire()
        if acquired:
            # 执行需要锁保护的操作
            await critical_operation()
        else:
            print("Could not acquire lock")
    finally:
        await lock.release()
```

### 2. 缓存策略

```python
from templates.cache import CacheStrategy, LRUCache, TTLCache

# LRU缓存策略
lru_cache = LRUCache(max_size=1000)

@lru_cache.cached
async def get_user_profile(user_id: int):
    # 从数据库获取用户资料
    return await db.get_user(user_id)

# TTL缓存策略
ttl_cache = TTLCache(default_ttl=300)  # 5分钟过期

@ttl_cache.cached
async def get_api_data(endpoint: str):
    # 从外部API获取数据
    return await external_api.get(endpoint)

# 多级缓存
class MultiLevelCache:
    def __init__(self):
        self.l1_cache = {}  # 内存缓存
        self.l2_cache = redis_manager  # Redis缓存
    
    async def get(self, key: str):
        # 先检查L1缓存
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # 再检查L2缓存
        value = await self.l2_cache.get(key)
        if value is not None:
            # 回填L1缓存
            self.l1_cache[key] = value
            return value
        
        return None
    
    async def set(self, key: str, value, timeout: int = 300):
        # 同时设置L1和L2缓存
        self.l1_cache[key] = value
        await self.l2_cache.set(key, value, timeout)

multi_cache = MultiLevelCache()
```

### 3. 任务调度

```python
from templates.cache import TaskScheduler
from celery.schedules import crontab

# 定期任务配置
celery_app.conf.beat_schedule = {
    'cleanup-expired-sessions': {
        'task': 'tasks.cleanup_expired_sessions',
        'schedule': crontab(minute=0, hour=2),  # 每天凌晨2点
    },
    'send-daily-reports': {
        'task': 'tasks.send_daily_reports',
        'schedule': crontab(minute=0, hour=9),  # 每天上午9点
    },
    'backup-database': {
        'task': 'tasks.backup_database',
        'schedule': crontab(minute=0, hour=3, day_of_week=1),  # 每周一凌晨3点
    },
    'process-analytics': {
        'task': 'tasks.process_analytics',
        'schedule': 30.0,  # 每30秒
    },
}

# 动态任务调度
class DynamicTaskScheduler:
    def __init__(self, celery_app):
        self.celery_app = celery_app
        self.scheduled_tasks = {}
    
    def schedule_task(self, task_name: str, task_func, schedule, **kwargs):
        """动态调度任务"""
        self.scheduled_tasks[task_name] = {
            'task': task_func,
            'schedule': schedule,
            'kwargs': kwargs
        }
        
        # 更新Celery配置
        self.celery_app.conf.beat_schedule[task_name] = {
            'task': task_func,
            'schedule': schedule,
            **kwargs
        }
    
    def remove_task(self, task_name: str):
        """移除调度任务"""
        if task_name in self.scheduled_tasks:
            del self.scheduled_tasks[task_name]
            del self.celery_app.conf.beat_schedule[task_name]

scheduler = DynamicTaskScheduler(celery_app)

# 使用动态调度
scheduler.schedule_task(
    'user-reminder',
    'tasks.send_user_reminder',
    crontab(minute=0, hour=18),  # 每天下午6点
    args=[user_id]
)
```

### 4. 消息路由和过滤

```python
from templates.cache import MessageRouter, MessageFilter

# 消息路由器
class NotificationRouter(MessageRouter):
    def __init__(self):
        super().__init__()
        self.routes = {
            'user.email': 'email_queue',
            'user.sms': 'sms_queue',
            'user.push': 'push_queue',
            'admin.alert': 'admin_queue'
        }
    
    def route_message(self, routing_key: str, message: dict):
        queue_name = self.routes.get(routing_key, 'default_queue')
        return queue_name

# 消息过滤器
class PriorityFilter(MessageFilter):
    def filter(self, message: dict) -> bool:
        priority = message.get('priority', 'normal')
        return priority in ['high', 'urgent']

class UserTypeFilter(MessageFilter):
    def __init__(self, allowed_user_types: list):
        self.allowed_user_types = allowed_user_types
    
    def filter(self, message: dict) -> bool:
        user_type = message.get('user_type', 'regular')
        return user_type in self.allowed_user_types

# 使用路由和过滤
router = NotificationRouter()
priority_filter = PriorityFilter()
vip_filter = UserTypeFilter(['vip', 'premium'])

async def send_filtered_notification(message: dict):
    # 应用过滤器
    if priority_filter.filter(message) and vip_filter.filter(message):
        # 确定路由
        queue = router.route_message(message['type'], message)
        
        # 发送到指定队列
        await message_queue.publish_to_queue(queue, message)
```

## 📝 最佳实践

### 1. 缓存键设计

```python
# 好的缓存键设计
class CacheKeyBuilder:
    @staticmethod
    def user_profile(user_id: int) -> str:
        return f"user:profile:{user_id}"
    
    @staticmethod
    def user_permissions(user_id: int) -> str:
        return f"user:permissions:{user_id}"
    
    @staticmethod
    def api_response(endpoint: str, params: dict) -> str:
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"api:{endpoint}:{hashlib.md5(param_str.encode()).hexdigest()}"
    
    @staticmethod
    def session_data(session_id: str) -> str:
        return f"session:{session_id}"

# 缓存版本控制
class VersionedCache:
    def __init__(self, redis_manager, version: int = 1):
        self.redis = redis_manager
        self.version = version
    
    def _versioned_key(self, key: str) -> str:
        return f"v{self.version}:{key}"
    
    async def get(self, key: str):
        return await self.redis.get(self._versioned_key(key))
    
    async def set(self, key: str, value, timeout: int = 300):
        return await self.redis.set(self._versioned_key(key), value, timeout)
    
    async def invalidate_version(self):
        """使当前版本的所有缓存失效"""
        pattern = f"v{self.version}:*"
        await self.redis.delete_pattern(pattern)
        self.version += 1
```

### 2. 错误处理和重试

```python
# 缓存降级策略
class CacheWithFallback:
    def __init__(self, primary_cache, fallback_cache):
        self.primary = primary_cache
        self.fallback = fallback_cache
    
    async def get(self, key: str):
        try:
            return await self.primary.get(key)
        except Exception as e:
            logger.warning(f"Primary cache failed: {e}, using fallback")
            try:
                return await self.fallback.get(key)
            except Exception as e2:
                logger.error(f"Fallback cache also failed: {e2}")
                return None
    
    async def set(self, key: str, value, timeout: int = 300):
        tasks = []
        try:
            tasks.append(self.primary.set(key, value, timeout))
        except Exception as e:
            logger.warning(f"Primary cache set failed: {e}")
        
        try:
            tasks.append(self.fallback.set(key, value, timeout))
        except Exception as e:
            logger.warning(f"Fallback cache set failed: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

# 任务重试策略
@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def robust_task(self, data):
    try:
        # 执行任务逻辑
        result = process_data(data)
        return result
    except TemporaryError as exc:
        # 临时错误，重试
        logger.warning(f"Temporary error in task: {exc}")
        raise self.retry(countdown=60)
    except PermanentError as exc:
        # 永久错误，不重试
        logger.error(f"Permanent error in task: {exc}")
        raise exc
```

### 3. 性能监控

```python
from templates.cache import CacheMetrics
import time

class MonitoredCache:
    def __init__(self, cache, metrics):
        self.cache = cache
        self.metrics = metrics
    
    async def get(self, key: str):
        start_time = time.time()
        try:
            result = await self.cache.get(key)
            
            if result is not None:
                self.metrics.record_hit(key)
            else:
                self.metrics.record_miss(key)
            
            return result
        finally:
            duration = time.time() - start_time
            self.metrics.record_operation_time('get', duration)
    
    async def set(self, key: str, value, timeout: int = 300):
        start_time = time.time()
        try:
            return await self.cache.set(key, value, timeout)
        finally:
            duration = time.time() - start_time
            self.metrics.record_operation_time('set', duration)

# 缓存统计
class CacheStats:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.operations = {}
    
    def record_hit(self, key: str):
        self.hits += 1
    
    def record_miss(self, key: str):
        self.misses += 1
    
    def record_operation_time(self, operation: str, duration: float):
        if operation not in self.operations:
            self.operations[operation] = []
        self.operations[operation].append(duration)
    
    def get_hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0
    
    def get_avg_operation_time(self, operation: str) -> float:
        times = self.operations.get(operation, [])
        return sum(times) / len(times) if times else 0
```

## ❓ 常见问题

### Q: 如何处理缓存雪崩？

A: 使用随机过期时间和缓存预热：

```python
import random

async def set_with_jitter(key: str, value, base_timeout: int = 300):
    # 添加随机抖动，避免同时过期
    jitter = random.randint(0, base_timeout // 10)
    timeout = base_timeout + jitter
    await redis_manager.set(key, value, timeout)

# 缓存预热
async def warm_up_cache():
    critical_keys = ['popular_products', 'featured_content', 'user_preferences']
    
    for key in critical_keys:
        if not await redis_manager.exists(key):
            data = await fetch_data_for_key(key)
            await set_with_jitter(key, data, 3600)
```

### Q: 如何处理缓存穿透？

A: 使用布隆过滤器和空值缓存：

```python
from templates.cache import BloomFilter

bloom_filter = BloomFilter(capacity=1000000, error_rate=0.1)

async def get_with_bloom_filter(key: str):
    # 先检查布隆过滤器
    if not bloom_filter.might_contain(key):
        return None  # 肯定不存在
    
    # 检查缓存
    cached_value = await redis_manager.get(key)
    if cached_value is not None:
        return cached_value
    
    # 从数据库获取
    db_value = await get_from_database(key)
    
    if db_value is not None:
        await redis_manager.set(key, db_value, 3600)
        bloom_filter.add(key)
    else:
        # 缓存空值，防止穿透
        await redis_manager.set(key, "NULL", 300)
    
    return db_value
```

### Q: 如何监控Celery任务？

A: 使用Flower和自定义监控：

```bash
# 安装Flower
pip install flower

# 启动Flower监控
celery -A your_app flower --port=5555
```

```python
# 自定义任务监控
class TaskMonitor:
    def __init__(self, celery_app):
        self.celery_app = celery_app
    
    def get_active_tasks(self):
        inspect = self.celery_app.control.inspect()
        return inspect.active()
    
    def get_scheduled_tasks(self):
        inspect = self.celery_app.control.inspect()
        return inspect.scheduled()
    
    def get_worker_stats(self):
        inspect = self.celery_app.control.inspect()
        return inspect.stats()

monitor = TaskMonitor(celery_app)
```

## 📚 相关文档

- [API开发模块使用指南](api.md)
- [数据库模块使用指南](database.md)
- [监控日志模块使用指南](monitoring.md)
- [性能优化建议](best-practices/performance.md)
- [测试策略指南](best-practices/testing.md)

---

如有其他问题，请查看 [GitHub Issues](https://github.com/your-username/python-backend-templates/issues) 或提交新的问题。