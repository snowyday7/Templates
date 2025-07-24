"""API网关

提供完整的API网关功能，包括：
- 路由管理
- 负载均衡
- 认证授权
- 限流控制
- 请求转换
- 响应聚合
- 监控日志
- 缓存管理
"""

import time
import json
import asyncio
import hashlib
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from urllib.parse import urlparse, urljoin

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

try:
    import aiohttp
    import requests
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RouteMethod(str, Enum):
    """HTTP方法"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    ANY = "*"


class LoadBalanceStrategy(str, Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    IP_HASH = "ip_hash"
    CONSISTENT_HASH = "consistent_hash"


class AuthenticationType(str, Enum):
    """认证类型"""
    NONE = "none"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    CUSTOM = "custom"


class RateLimitStrategy(str, Enum):
    """限流策略"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class Backend:
    """后端服务"""
    id: str
    host: str
    port: int
    weight: int = 1
    health_check_path: str = "/health"
    timeout: int = 30
    max_connections: int = 100
    
    # 状态信息
    is_healthy: bool = True
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    last_health_check: float = 0.0
    
    # 性能指标
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def url(self) -> str:
        """获取后端URL"""
        return f"http://{self.host}:{self.port}"
    
    @property
    def success_rate(self) -> float:
        """获取成功率"""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests
    
    @property
    def average_response_time(self) -> float:
        """获取平均响应时间"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def record_request(self, response_time: float, success: bool):
        """记录请求"""
        self.total_requests += 1
        self.response_times.append(response_time)
        if not success:
            self.failed_requests += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "weight": self.weight,
            "url": self.url,
            "is_healthy": self.is_healthy,
            "active_connections": self.active_connections,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "average_response_time": self.average_response_time
        }


@dataclass
class RouteConfig:
    """路由配置"""
    path: str
    methods: List[RouteMethod] = field(default_factory=lambda: [RouteMethod.ANY])
    backends: List[Backend] = field(default_factory=list)
    load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    
    # 认证配置
    authentication: AuthenticationType = AuthenticationType.NONE
    auth_config: Dict[str, Any] = field(default_factory=dict)
    
    # 限流配置
    rate_limit_enabled: bool = False
    rate_limit_strategy: RateLimitStrategy = RateLimitStrategy.FIXED_WINDOW
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # 缓存配置
    cache_enabled: bool = False
    cache_ttl: int = 300  # seconds
    cache_key_pattern: Optional[str] = None
    
    # 重试配置
    retry_enabled: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # 超时配置
    timeout: int = 30
    
    # 请求转换
    request_transformers: List[str] = field(default_factory=list)
    response_transformers: List[str] = field(default_factory=list)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def matches(self, path: str, method: str) -> bool:
        """检查路由是否匹配"""
        # 简单的路径匹配（可以扩展为正则表达式）
        path_matches = self.path == path or self.path.endswith('*') and path.startswith(self.path[:-1])
        method_matches = RouteMethod.ANY in self.methods or RouteMethod(method.upper()) in self.methods
        return path_matches and method_matches


class Middleware(ABC):
    """中间件基类"""
    
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理请求"""
        pass
    
    @abstractmethod
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """处理响应"""
        pass


class AuthenticationMiddleware(Middleware):
    """认证中间件"""
    
    def __init__(self, auth_type: AuthenticationType, config: Dict[str, Any]):
        self.auth_type = auth_type
        self.config = config
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理请求认证"""
        if self.auth_type == AuthenticationType.NONE:
            return request
        
        headers = request.get('headers', {})
        
        if self.auth_type == AuthenticationType.API_KEY:
            api_key = headers.get('X-API-Key') or request.get('query_params', {}).get('api_key')
            if not api_key or not self._validate_api_key(api_key):
                raise Exception("Invalid API key")
        
        elif self.auth_type == AuthenticationType.JWT:
            auth_header = headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                raise Exception("Missing or invalid JWT token")
            token = auth_header[7:]
            if not self._validate_jwt_token(token):
                raise Exception("Invalid JWT token")
        
        elif self.auth_type == AuthenticationType.BASIC:
            auth_header = headers.get('Authorization', '')
            if not auth_header.startswith('Basic '):
                raise Exception("Missing or invalid basic auth")
            # 这里可以添加基本认证验证逻辑
        
        return request
    
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """处理响应"""
        return response
    
    def _validate_api_key(self, api_key: str) -> bool:
        """验证API密钥"""
        valid_keys = self.config.get('valid_keys', [])
        return api_key in valid_keys
    
    def _validate_jwt_token(self, token: str) -> bool:
        """验证JWT令牌"""
        # 这里应该实现真正的JWT验证逻辑
        return len(token) > 10  # 简单示例


class RateLimiter(Middleware):
    """限流中间件"""
    
    def __init__(self, strategy: RateLimitStrategy, requests: int, window: int):
        self.strategy = strategy
        self.requests = requests
        self.window = window
        self.counters = defaultdict(deque)
        self.tokens = defaultdict(lambda: {'tokens': requests, 'last_refill': time.time()})
        self._lock = threading.Lock()
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理请求限流"""
        client_id = self._get_client_id(request)
        
        if self.strategy == RateLimitStrategy.FIXED_WINDOW:
            if not self._check_fixed_window(client_id):
                raise Exception("Rate limit exceeded")
        elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            if not self._check_sliding_window(client_id):
                raise Exception("Rate limit exceeded")
        elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            if not self._check_token_bucket(client_id):
                raise Exception("Rate limit exceeded")
        
        return request
    
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """处理响应"""
        return response
    
    def _get_client_id(self, request: Dict[str, Any]) -> str:
        """获取客户端ID"""
        # 可以基于IP、用户ID、API密钥等
        return request.get('client_ip', 'unknown')
    
    def _check_fixed_window(self, client_id: str) -> bool:
        """检查固定窗口限流"""
        current_window = int(time.time()) // self.window
        key = f"{client_id}:{current_window}"
        
        with self._lock:
            count = len(self.counters[key])
            if count >= self.requests:
                return False
            self.counters[key].append(time.time())
            return True
    
    def _check_sliding_window(self, client_id: str) -> bool:
        """检查滑动窗口限流"""
        now = time.time()
        window_start = now - self.window
        
        with self._lock:
            # 清理过期记录
            while self.counters[client_id] and self.counters[client_id][0] < window_start:
                self.counters[client_id].popleft()
            
            if len(self.counters[client_id]) >= self.requests:
                return False
            
            self.counters[client_id].append(now)
            return True
    
    def _check_token_bucket(self, client_id: str) -> bool:
        """检查令牌桶限流"""
        now = time.time()
        
        with self._lock:
            bucket = self.tokens[client_id]
            
            # 补充令牌
            time_passed = now - bucket['last_refill']
            tokens_to_add = time_passed * (self.requests / self.window)
            bucket['tokens'] = min(self.requests, bucket['tokens'] + tokens_to_add)
            bucket['last_refill'] = now
            
            if bucket['tokens'] >= 1:
                bucket['tokens'] -= 1
                return True
            
            return False


class LoggingMiddleware(Middleware):
    """日志中间件"""
    
    def __init__(self, log_requests: bool = True, log_responses: bool = True):
        self.log_requests = log_requests
        self.log_responses = log_responses
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """记录请求日志"""
        if self.log_requests:
            print(f"Request: {request.get('method')} {request.get('path')} from {request.get('client_ip')}")
        
        request['start_time'] = time.time()
        return request
    
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """记录响应日志"""
        if self.log_responses:
            duration = time.time() - response.get('start_time', 0)
            print(f"Response: {response.get('status_code')} in {duration:.3f}s")
        
        return response


class Route:
    """路由"""
    
    def __init__(self, config: RouteConfig):
        self.config = config
        self.middlewares: List[Middleware] = []
        self._current_backend_index = 0
        self._lock = threading.Lock()
        
        # 设置默认中间件
        if config.authentication != AuthenticationType.NONE:
            self.middlewares.append(AuthenticationMiddleware(config.authentication, config.auth_config))
        
        if config.rate_limit_enabled:
            self.middlewares.append(RateLimiter(
                config.rate_limit_strategy,
                config.rate_limit_requests,
                config.rate_limit_window
            ))
        
        self.middlewares.append(LoggingMiddleware())
    
    def add_middleware(self, middleware: Middleware):
        """添加中间件"""
        self.middlewares.append(middleware)
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理请求"""
        # 执行请求中间件
        for middleware in self.middlewares:
            request = await middleware.process_request(request)
        
        # 选择后端服务
        backend = self._select_backend()
        if not backend:
            return {
                'status_code': 503,
                'body': {'error': 'No healthy backend available'},
                'headers': {}
            }
        
        # 转发请求
        response = await self._forward_request(request, backend)
        
        # 执行响应中间件
        for middleware in reversed(self.middlewares):
            response = await middleware.process_response(response)
        
        return response
    
    def _select_backend(self) -> Optional[Backend]:
        """选择后端服务"""
        healthy_backends = [b for b in self.config.backends if b.is_healthy]
        if not healthy_backends:
            return None
        
        if self.config.load_balance_strategy == LoadBalanceStrategy.ROUND_ROBIN:
            with self._lock:
                backend = healthy_backends[self._current_backend_index % len(healthy_backends)]
                self._current_backend_index += 1
                return backend
        
        elif self.config.load_balance_strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            # 简化的加权轮询
            weights = [b.weight for b in healthy_backends]
            total_weight = sum(weights)
            if total_weight == 0:
                return healthy_backends[0]
            
            import random
            rand_num = random.uniform(0, total_weight)
            current_weight = 0
            for i, backend in enumerate(healthy_backends):
                current_weight += weights[i]
                if rand_num <= current_weight:
                    return backend
        
        elif self.config.load_balance_strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return min(healthy_backends, key=lambda b: b.active_connections)
        
        elif self.config.load_balance_strategy == LoadBalanceStrategy.RANDOM:
            import random
            return random.choice(healthy_backends)
        
        return healthy_backends[0]
    
    async def _forward_request(self, request: Dict[str, Any], backend: Backend) -> Dict[str, Any]:
        """转发请求到后端服务"""
        if not HTTP_AVAILABLE:
            return {
                'status_code': 500,
                'body': {'error': 'HTTP client not available'},
                'headers': {}
            }
        
        start_time = time.time()
        backend.active_connections += 1
        
        try:
            url = urljoin(backend.url, request.get('path', '/'))
            method = request.get('method', 'GET')
            headers = request.get('headers', {})
            data = request.get('body')
            params = request.get('query_params', {})
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=data,
                    params=params
                ) as response:
                    response_time = time.time() - start_time
                    success = response.status < 400
                    
                    backend.record_request(response_time, success)
                    
                    response_body = await response.text()
                    try:
                        response_body = json.loads(response_body)
                    except json.JSONDecodeError:
                        pass
                    
                    return {
                        'status_code': response.status,
                        'body': response_body,
                        'headers': dict(response.headers),
                        'start_time': request.get('start_time', start_time)
                    }
        
        except Exception as e:
            response_time = time.time() - start_time
            backend.record_request(response_time, False)
            
            return {
                'status_code': 500,
                'body': {'error': f'Backend request failed: {str(e)}'},
                'headers': {},
                'start_time': request.get('start_time', start_time)
            }
        
        finally:
            backend.active_connections -= 1


class APIGateway:
    """API网关"""
    
    def __init__(self, config: 'GatewayConfig'):
        self.config = config
        self.routes: List[Route] = []
        self.global_middlewares: List[Middleware] = []
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0.0
        }
        self._lock = threading.Lock()
        
        # 健康检查
        self._health_check_thread = None
        self._running = False
        
        if config.health_check_enabled:
            self.start_health_check()
    
    def add_route(self, route_config: RouteConfig) -> Route:
        """添加路由"""
        route = Route(route_config)
        
        # 添加全局中间件
        for middleware in self.global_middlewares:
            route.add_middleware(middleware)
        
        self.routes.append(route)
        return route
    
    def add_global_middleware(self, middleware: Middleware):
        """添加全局中间件"""
        self.global_middlewares.append(middleware)
        
        # 为现有路由添加中间件
        for route in self.routes:
            route.add_middleware(middleware)
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理请求"""
        start_time = time.time()
        
        with self._lock:
            self._stats['total_requests'] += 1
        
        try:
            # 查找匹配的路由
            route = self._find_route(request)
            if not route:
                return {
                    'status_code': 404,
                    'body': {'error': 'Route not found'},
                    'headers': {}
                }
            
            # 处理请求
            response = await route.handle_request(request)
            
            # 更新统计
            response_time = time.time() - start_time
            with self._lock:
                self._stats['total_response_time'] += response_time
                if response.get('status_code', 500) < 400:
                    self._stats['successful_requests'] += 1
                else:
                    self._stats['failed_requests'] += 1
            
            return response
        
        except Exception as e:
            with self._lock:
                self._stats['failed_requests'] += 1
            
            return {
                'status_code': 500,
                'body': {'error': f'Internal server error: {str(e)}'},
                'headers': {}
            }
    
    def _find_route(self, request: Dict[str, Any]) -> Optional[Route]:
        """查找匹配的路由"""
        path = request.get('path', '/')
        method = request.get('method', 'GET')
        
        for route in self.routes:
            if route.config.matches(path, method):
                return route
        
        return None
    
    def start_health_check(self):
        """启动健康检查"""
        if self._running:
            return
        
        self._running = True
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()
    
    def stop_health_check(self):
        """停止健康检查"""
        self._running = False
        if self._health_check_thread:
            self._health_check_thread.join()
    
    def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                self._perform_health_checks()
            except Exception as e:
                print(f"Health check error: {e}")
            
            time.sleep(self.config.health_check_interval)
    
    def _perform_health_checks(self):
        """执行健康检查"""
        for route in self.routes:
            for backend in route.config.backends:
                try:
                    is_healthy = self._check_backend_health(backend)
                    backend.is_healthy = is_healthy
                    backend.last_health_check = time.time()
                except Exception as e:
                    print(f"Health check failed for backend {backend.id}: {e}")
                    backend.is_healthy = False
    
    def _check_backend_health(self, backend: Backend) -> bool:
        """检查后端健康状态"""
        if not HTTP_AVAILABLE:
            return True
        
        try:
            health_url = urljoin(backend.url, backend.health_check_path)
            response = requests.get(health_url, timeout=5)
            return response.status_code < 400
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = self._stats.copy()
        
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
            stats['average_response_time'] = stats['total_response_time'] / stats['total_requests']
        else:
            stats['success_rate'] = 0.0
            stats['average_response_time'] = 0.0
        
        # 添加路由统计
        stats['routes'] = []
        for route in self.routes:
            route_stats = {
                'path': route.config.path,
                'methods': [m.value for m in route.config.methods],
                'backends': [b.to_dict() for b in route.config.backends]
            }
            stats['routes'].append(route_stats)
        
        return stats


class GatewayConfig(BaseSettings):
    """网关配置"""
    host: str = "0.0.0.0"
    port: int = 8080
    
    # 健康检查配置
    health_check_enabled: bool = True
    health_check_interval: int = 30  # seconds
    
    # 全局超时配置
    default_timeout: int = 30
    
    # 缓存配置
    cache_enabled: bool = False
    cache_backend: str = "memory"  # memory, redis
    cache_default_ttl: int = 300
    
    # 监控配置
    metrics_enabled: bool = True
    metrics_path: str = "/metrics"
    
    # 日志配置
    log_level: str = "INFO"
    access_log_enabled: bool = True
    
    # 安全配置
    cors_enabled: bool = True
    cors_origins: List[str] = ["*"]
    
    class Config:
        env_prefix = "GATEWAY_"


# 全局API网关实例
_api_gateway: Optional[APIGateway] = None


def initialize_api_gateway(config: GatewayConfig) -> APIGateway:
    """初始化API网关"""
    global _api_gateway
    _api_gateway = APIGateway(config)
    return _api_gateway


def get_api_gateway() -> Optional[APIGateway]:
    """获取全局API网关实例"""
    return _api_gateway


# 便捷函数
def add_route(route_config: RouteConfig) -> Optional[Route]:
    """添加路由的便捷函数"""
    gateway = get_api_gateway()
    return gateway.add_route(route_config) if gateway else None


def add_global_middleware(middleware: Middleware):
    """添加全局中间件的便捷函数"""
    gateway = get_api_gateway()
    if gateway:
        gateway.add_global_middleware(middleware)


async def handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """处理请求的便捷函数"""
    gateway = get_api_gateway()
    if gateway:
        return await gateway.handle_request(request)
    return {
        'status_code': 503,
        'body': {'error': 'Gateway not initialized'},
        'headers': {}
    }