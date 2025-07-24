"""负载均衡器

提供完整的负载均衡功能，包括：
- 多种负载均衡策略
- 健康检查
- 权重配置
- 连接跟踪
- 故障检测
"""

import time
import random
import hashlib
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timezone

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

try:
    import requests
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False


class LoadBalancingStrategy(str, Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"
    CONSISTENT_HASH = "consistent_hash"
    IP_HASH = "ip_hash"
    LEAST_RESPONSE_TIME = "least_response_time"
    HEALTH_CHECK = "health_check"


class ServerStatus(str, Enum):
    """服务器状态"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


@dataclass
class ServerInstance:
    """服务器实例"""
    id: str
    host: str
    port: int
    weight: int = 1
    status: ServerStatus = ServerStatus.HEALTHY
    
    # 连接统计
    active_connections: int = 0
    total_connections: int = 0
    
    # 性能统计
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    success_count: int = 0
    failure_count: int = 0
    
    # 健康检查
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    @property
    def url(self) -> str:
        """获取服务器URL"""
        return f"http://{self.host}:{self.port}"
    
    @property
    def is_healthy(self) -> bool:
        """检查服务器是否健康"""
        return self.status == ServerStatus.HEALTHY
    
    @property
    def average_response_time(self) -> float:
        """获取平均响应时间"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    @property
    def success_rate(self) -> float:
        """获取成功率"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total
    
    def record_request(self, response_time: float, success: bool):
        """记录请求"""
        self.response_times.append(response_time)
        if success:
            self.success_count += 1
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.failure_count += 1
            self.consecutive_failures += 1
            self.consecutive_successes = 0
    
    def increment_connections(self):
        """增加连接数"""
        self.active_connections += 1
        self.total_connections += 1
    
    def decrement_connections(self):
        """减少连接数"""
        if self.active_connections > 0:
            self.active_connections -= 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "weight": self.weight,
            "status": self.status.value,
            "active_connections": self.active_connections,
            "total_connections": self.total_connections,
            "average_response_time": self.average_response_time,
            "success_rate": self.success_rate,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "metadata": self.metadata,
            "created_at": self.created_at
        }


class LoadBalancer(ABC):
    """负载均衡器基类"""
    
    def __init__(self, servers: List[ServerInstance]):
        self.servers = {server.id: server for server in servers}
        self._lock = threading.Lock()
    
    @abstractmethod
    def select_server(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServerInstance]:
        """选择服务器"""
        pass
    
    def add_server(self, server: ServerInstance):
        """添加服务器"""
        with self._lock:
            self.servers[server.id] = server
    
    def remove_server(self, server_id: str):
        """移除服务器"""
        with self._lock:
            self.servers.pop(server_id, None)
    
    def get_server(self, server_id: str) -> Optional[ServerInstance]:
        """获取服务器"""
        return self.servers.get(server_id)
    
    def get_healthy_servers(self) -> List[ServerInstance]:
        """获取健康的服务器"""
        return [server for server in self.servers.values() if server.is_healthy]
    
    def get_all_servers(self) -> List[ServerInstance]:
        """获取所有服务器"""
        return list(self.servers.values())
    
    def update_server_status(self, server_id: str, status: ServerStatus):
        """更新服务器状态"""
        server = self.get_server(server_id)
        if server:
            server.status = status
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        servers = list(self.servers.values())
        healthy_servers = self.get_healthy_servers()
        
        return {
            "total_servers": len(servers),
            "healthy_servers": len(healthy_servers),
            "unhealthy_servers": len(servers) - len(healthy_servers),
            "total_connections": sum(s.active_connections for s in servers),
            "total_requests": sum(s.total_connections for s in servers),
            "average_response_time": sum(s.average_response_time for s in servers) / len(servers) if servers else 0,
            "overall_success_rate": sum(s.success_rate for s in servers) / len(servers) if servers else 0
        }


class RoundRobinBalancer(LoadBalancer):
    """轮询负载均衡器"""
    
    def __init__(self, servers: List[ServerInstance]):
        super().__init__(servers)
        self._current_index = 0
    
    def select_server(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServerInstance]:
        """轮询选择服务器"""
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None
        
        with self._lock:
            server = healthy_servers[self._current_index % len(healthy_servers)]
            self._current_index += 1
            return server


class WeightedRoundRobinBalancer(LoadBalancer):
    """加权轮询负载均衡器"""
    
    def __init__(self, servers: List[ServerInstance]):
        super().__init__(servers)
        self._current_weights = {}
        self._reset_weights()
    
    def _reset_weights(self):
        """重置权重"""
        for server in self.servers.values():
            self._current_weights[server.id] = 0
    
    def select_server(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServerInstance]:
        """加权轮询选择服务器"""
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None
        
        with self._lock:
            # 计算总权重
            total_weight = sum(server.weight for server in healthy_servers)
            if total_weight == 0:
                return random.choice(healthy_servers)
            
            # 增加当前权重
            for server in healthy_servers:
                self._current_weights[server.id] += server.weight
            
            # 选择权重最高的服务器
            selected_server = max(healthy_servers, key=lambda s: self._current_weights[s.id])
            
            # 减少选中服务器的权重
            self._current_weights[selected_server.id] -= total_weight
            
            return selected_server
    
    def add_server(self, server: ServerInstance):
        """添加服务器"""
        super().add_server(server)
        self._current_weights[server.id] = 0
    
    def remove_server(self, server_id: str):
        """移除服务器"""
        super().remove_server(server_id)
        self._current_weights.pop(server_id, None)


class LeastConnectionsBalancer(LoadBalancer):
    """最少连接负载均衡器"""
    
    def select_server(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServerInstance]:
        """选择连接数最少的服务器"""
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None
        
        return min(healthy_servers, key=lambda s: s.active_connections)


class WeightedLeastConnectionsBalancer(LoadBalancer):
    """加权最少连接负载均衡器"""
    
    def select_server(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServerInstance]:
        """选择加权连接数最少的服务器"""
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None
        
        # 计算加权连接数（连接数/权重）
        def weighted_connections(server):
            if server.weight == 0:
                return float('inf')
            return server.active_connections / server.weight
        
        return min(healthy_servers, key=weighted_connections)


class RandomBalancer(LoadBalancer):
    """随机负载均衡器"""
    
    def select_server(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServerInstance]:
        """随机选择服务器"""
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None
        
        return random.choice(healthy_servers)


class WeightedRandomBalancer(LoadBalancer):
    """加权随机负载均衡器"""
    
    def select_server(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServerInstance]:
        """加权随机选择服务器"""
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None
        
        # 计算权重
        weights = [server.weight for server in healthy_servers]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(healthy_servers)
        
        # 加权随机选择
        rand_num = random.uniform(0, total_weight)
        current_weight = 0
        
        for i, server in enumerate(healthy_servers):
            current_weight += weights[i]
            if rand_num <= current_weight:
                return server
        
        return healthy_servers[-1]  # 备用


class ConsistentHashBalancer(LoadBalancer):
    """一致性哈希负载均衡器"""
    
    def __init__(self, servers: List[ServerInstance], virtual_nodes: int = 150):
        super().__init__(servers)
        self.virtual_nodes = virtual_nodes
        self._hash_ring = {}
        self._rebuild_hash_ring()
    
    def _rebuild_hash_ring(self):
        """重建哈希环"""
        self._hash_ring = {}
        
        for server in self.servers.values():
            for i in range(self.virtual_nodes):
                virtual_key = f"{server.id}:{i}"
                hash_value = int(hashlib.md5(virtual_key.encode()).hexdigest(), 16)
                self._hash_ring[hash_value] = server
    
    def select_server(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServerInstance]:
        """基于一致性哈希选择服务器"""
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None
        
        # 获取哈希键
        hash_key = self._get_hash_key(request_context)
        if not hash_key:
            return random.choice(healthy_servers)
        
        # 计算哈希值
        hash_value = int(hashlib.md5(hash_key.encode()).hexdigest(), 16)
        
        # 在哈希环中查找
        if not self._hash_ring:
            return random.choice(healthy_servers)
        
        # 找到第一个大于等于hash_value的节点
        sorted_hashes = sorted(self._hash_ring.keys())
        for ring_hash in sorted_hashes:
            server = self._hash_ring[ring_hash]
            if ring_hash >= hash_value and server.is_healthy:
                return server
        
        # 如果没找到，返回第一个健康的节点
        for ring_hash in sorted_hashes:
            server = self._hash_ring[ring_hash]
            if server.is_healthy:
                return server
        
        return random.choice(healthy_servers)
    
    def _get_hash_key(self, request_context: Optional[Dict[str, Any]]) -> Optional[str]:
        """获取哈希键"""
        if not request_context:
            return None
        
        # 尝试从请求上下文中获取客户端IP或用户ID
        return (request_context.get('client_ip') or 
                request_context.get('user_id') or 
                request_context.get('session_id'))
    
    def add_server(self, server: ServerInstance):
        """添加服务器"""
        super().add_server(server)
        self._rebuild_hash_ring()
    
    def remove_server(self, server_id: str):
        """移除服务器"""
        super().remove_server(server_id)
        self._rebuild_hash_ring()


class IPHashBalancer(LoadBalancer):
    """IP哈希负载均衡器"""
    
    def select_server(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServerInstance]:
        """基于客户端IP哈希选择服务器"""
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None
        
        # 获取客户端IP
        client_ip = None
        if request_context:
            client_ip = request_context.get('client_ip')
        
        if not client_ip:
            return random.choice(healthy_servers)
        
        # 计算哈希值
        hash_value = hash(client_ip)
        index = hash_value % len(healthy_servers)
        
        return healthy_servers[index]


class LeastResponseTimeBalancer(LoadBalancer):
    """最少响应时间负载均衡器"""
    
    def select_server(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServerInstance]:
        """选择响应时间最短的服务器"""
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None
        
        return min(healthy_servers, key=lambda s: s.average_response_time)


class HealthCheckBalancer(LoadBalancer):
    """健康检查负载均衡器"""
    
    def __init__(self, servers: List[ServerInstance], base_balancer: LoadBalancer,
                 health_check_interval: int = 30, health_check_timeout: int = 5,
                 failure_threshold: int = 3, success_threshold: int = 2):
        super().__init__(servers)
        self.base_balancer = base_balancer
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        
        self._health_check_thread = None
        self._running = False
        
        # 启动健康检查
        self.start_health_check()
    
    def select_server(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServerInstance]:
        """选择服务器（委托给基础负载均衡器）"""
        return self.base_balancer.select_server(request_context)
    
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
            
            time.sleep(self.health_check_interval)
    
    def _perform_health_checks(self):
        """执行健康检查"""
        for server in self.servers.values():
            try:
                is_healthy = self._check_server_health(server)
                self._update_server_health(server, is_healthy)
            except Exception as e:
                print(f"Health check failed for server {server.id}: {e}")
                self._update_server_health(server, False)
    
    def _check_server_health(self, server: ServerInstance) -> bool:
        """检查单个服务器健康状态"""
        if not HTTP_AVAILABLE:
            return True  # 如果没有requests库，假设健康
        
        try:
            # 发送健康检查请求
            health_url = f"{server.url}/health"
            response = requests.get(health_url, timeout=self.health_check_timeout)
            
            # 记录响应时间
            server.record_request(response.elapsed.total_seconds(), response.status_code < 400)
            
            return response.status_code < 400
        except Exception:
            return False
    
    def _update_server_health(self, server: ServerInstance, is_healthy: bool):
        """更新服务器健康状态"""
        server.last_health_check = time.time()
        
        if is_healthy:
            server.consecutive_successes += 1
            server.consecutive_failures = 0
            
            # 如果连续成功次数达到阈值，标记为健康
            if (server.status != ServerStatus.HEALTHY and 
                server.consecutive_successes >= self.success_threshold):
                server.status = ServerStatus.HEALTHY
                print(f"Server {server.id} is now healthy")
        else:
            server.consecutive_failures += 1
            server.consecutive_successes = 0
            
            # 如果连续失败次数达到阈值，标记为不健康
            if (server.status == ServerStatus.HEALTHY and 
                server.consecutive_failures >= self.failure_threshold):
                server.status = ServerStatus.UNHEALTHY
                print(f"Server {server.id} is now unhealthy")


class LoadBalancerConfig(BaseSettings):
    """负载均衡器配置"""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    
    # 健康检查配置
    health_check_enabled: bool = True
    health_check_interval: int = 30  # seconds
    health_check_timeout: int = 5  # seconds
    failure_threshold: int = 3
    success_threshold: int = 2
    
    # 一致性哈希配置
    virtual_nodes: int = 150
    
    # 连接配置
    max_connections_per_server: int = 1000
    connection_timeout: int = 30
    
    # 统计配置
    enable_stats: bool = True
    stats_window_size: int = 100
    
    class Config:
        env_prefix = "LB_"


class LoadBalancerManager:
    """负载均衡器管理器"""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self._balancers: Dict[str, LoadBalancer] = {}
        self._lock = threading.Lock()
    
    def create_balancer(self, name: str, servers: List[ServerInstance],
                       strategy: Optional[LoadBalancingStrategy] = None) -> LoadBalancer:
        """创建负载均衡器"""
        strategy = strategy or self.config.strategy
        
        # 创建基础负载均衡器
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            base_balancer = RoundRobinBalancer(servers)
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            base_balancer = WeightedRoundRobinBalancer(servers)
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            base_balancer = LeastConnectionsBalancer(servers)
        elif strategy == LoadBalancingStrategy.WEIGHTED_LEAST_CONNECTIONS:
            base_balancer = WeightedLeastConnectionsBalancer(servers)
        elif strategy == LoadBalancingStrategy.RANDOM:
            base_balancer = RandomBalancer(servers)
        elif strategy == LoadBalancingStrategy.WEIGHTED_RANDOM:
            base_balancer = WeightedRandomBalancer(servers)
        elif strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            base_balancer = ConsistentHashBalancer(servers, self.config.virtual_nodes)
        elif strategy == LoadBalancingStrategy.IP_HASH:
            base_balancer = IPHashBalancer(servers)
        elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            base_balancer = LeastResponseTimeBalancer(servers)
        else:
            base_balancer = RoundRobinBalancer(servers)
        
        # 如果启用健康检查，包装为健康检查负载均衡器
        if self.config.health_check_enabled:
            balancer = HealthCheckBalancer(
                servers, base_balancer,
                self.config.health_check_interval,
                self.config.health_check_timeout,
                self.config.failure_threshold,
                self.config.success_threshold
            )
        else:
            balancer = base_balancer
        
        with self._lock:
            self._balancers[name] = balancer
        
        return balancer
    
    def get_balancer(self, name: str) -> Optional[LoadBalancer]:
        """获取负载均衡器"""
        return self._balancers.get(name)
    
    def remove_balancer(self, name: str):
        """移除负载均衡器"""
        with self._lock:
            balancer = self._balancers.pop(name, None)
            if isinstance(balancer, HealthCheckBalancer):
                balancer.stop_health_check()
    
    def get_all_balancers(self) -> Dict[str, LoadBalancer]:
        """获取所有负载均衡器"""
        return self._balancers.copy()
    
    def get_global_stats(self) -> Dict[str, Any]:
        """获取全局统计信息"""
        stats = {
            "total_balancers": len(self._balancers),
            "balancers": {}
        }
        
        for name, balancer in self._balancers.items():
            stats["balancers"][name] = balancer.get_stats()
        
        return stats


# 全局负载均衡器管理器
_load_balancer_manager: Optional[LoadBalancerManager] = None


def initialize_load_balancer(config: LoadBalancerConfig) -> LoadBalancerManager:
    """初始化负载均衡器"""
    global _load_balancer_manager
    _load_balancer_manager = LoadBalancerManager(config)
    return _load_balancer_manager


def get_load_balancer() -> Optional[LoadBalancerManager]:
    """获取全局负载均衡器管理器"""
    return _load_balancer_manager


# 便捷函数
def create_load_balancer(name: str, servers: List[ServerInstance],
                         strategy: Optional[LoadBalancingStrategy] = None) -> Optional[LoadBalancer]:
    """创建负载均衡器的便捷函数"""
    manager = get_load_balancer()
    return manager.create_balancer(name, servers, strategy) if manager else None


def get_balancer(name: str) -> Optional[LoadBalancer]:
    """获取负载均衡器的便捷函数"""
    manager = get_load_balancer()
    return manager.get_balancer(name) if manager else None


def select_server(balancer_name: str, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServerInstance]:
    """选择服务器的便捷函数"""
    balancer = get_balancer(balancer_name)
    return balancer.select_server(request_context) if balancer else None