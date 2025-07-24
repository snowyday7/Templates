# -*- coding: utf-8 -*-
"""
高可用性模块
提供负载均衡、故障转移、健康检查等高可用性功能
"""

import asyncio
import time
import random
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class LoadBalanceStrategy(Enum):
    """负载均衡策略枚举"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"
    IP_HASH = "ip_hash"


@dataclass
class ServiceNode:
    """服务节点"""
    id: str
    host: str
    port: int
    weight: int = 1
    max_connections: int = 100
    current_connections: int = 0
    health_status: HealthStatus = HealthStatus.UNKNOWN
    last_health_check: Optional[float] = None
    response_time: float = 0.0
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        """获取节点端点"""
        return f"{self.host}:{self.port}"

    @property
    def is_healthy(self) -> bool:
        """检查节点是否健康"""
        return self.health_status == HealthStatus.HEALTHY

    @property
    def load_factor(self) -> float:
        """计算负载因子"""
        if self.max_connections == 0:
            return 1.0
        return self.current_connections / self.max_connections


class HealthChecker:
    """健康检查器"""

    def __init__(
        self,
        check_interval: int = 30,
        timeout: int = 5,
        failure_threshold: int = 3,
        success_threshold: int = 2
    ):
        self.check_interval = check_interval
        self.timeout = timeout
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self._running = False
        self._check_tasks: Dict[str, asyncio.Task] = {}

    async def check_node_health(
        self,
        node: ServiceNode,
        health_check_func: Optional[Callable] = None
    ) -> HealthStatus:
        """检查单个节点健康状态"""
        try:
            start_time = time.time()

            if health_check_func:
                result = await asyncio.wait_for(
                    health_check_func(node),
                    timeout=self.timeout
                )
            else:
                # 默认TCP连接检查
                result = await self._tcp_health_check(node)

            node.response_time = time.time() - start_time
            node.last_health_check = time.time()

            if result:
                node.failure_count = 0
                return HealthStatus.HEALTHY
            else:
                node.failure_count += 1
                if node.failure_count >= self.failure_threshold:
                    return HealthStatus.UNHEALTHY
                return HealthStatus.DEGRADED

        except asyncio.TimeoutError:
            logger.warning(f"Health check timeout for node {node.id}")
            node.failure_count += 1
            return HealthStatus.UNHEALTHY
        except Exception as e:
            logger.error(f"Health check error for node {node.id}: {e}")
            node.failure_count += 1
            return HealthStatus.UNHEALTHY

    async def _tcp_health_check(self, node: ServiceNode) -> bool:
        """TCP连接健康检查"""
        try:
            reader, writer = await asyncio.open_connection(
                node.host, node.port
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    async def start_monitoring(
        self,
        nodes: List[ServiceNode],
        health_check_func: Optional[Callable] = None
    ):
        """开始监控节点健康状态"""
        self._running = True
        for node in nodes:
            task = asyncio.create_task(
                self._monitor_node(node, health_check_func)
            )
            self._check_tasks[node.id] = task

    async def stop_monitoring(self):
        """停止监控"""
        self._running = False
        for task in self._check_tasks.values():
            task.cancel()
        await asyncio.gather(*self._check_tasks.values(), return_exceptions=True)
        self._check_tasks.clear()

    async def _monitor_node(
        self,
        node: ServiceNode,
        health_check_func: Optional[Callable] = None
    ):
        """监控单个节点"""
        while self._running:
            try:
                status = await self.check_node_health(node, health_check_func)
                node.health_status = status
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring node {node.id}: {e}")
                await asyncio.sleep(self.check_interval)


class LoadBalancer:
    """负载均衡器"""

    def __init__(
        self,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
        health_checker: Optional[HealthChecker] = None
    ):
        self.strategy = strategy
        self.health_checker = health_checker or HealthChecker()
        self.nodes: List[ServiceNode] = []
        self._current_index = 0
        self._node_weights: Dict[str, int] = {}

    def add_node(self, node: ServiceNode):
        """添加服务节点"""
        self.nodes.append(node)
        self._node_weights[node.id] = node.weight
        logger.info(f"Added node {node.id} to load balancer")

    def remove_node(self, node_id: str):
        """移除服务节点"""
        self.nodes = [n for n in self.nodes if n.id != node_id]
        self._node_weights.pop(node_id, None)
        logger.info(f"Removed node {node_id} from load balancer")

    def get_healthy_nodes(self) -> List[ServiceNode]:
        """获取健康的节点列表"""
        return [node for node in self.nodes if node.is_healthy]

    async def select_node(
        self,
        client_ip: Optional[str] = None
    ) -> Optional[ServiceNode]:
        """选择一个节点"""
        healthy_nodes = self.get_healthy_nodes()
        if not healthy_nodes:
            logger.warning("No healthy nodes available")
            return None

        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin_select(healthy_nodes)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(healthy_nodes)
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_nodes)
        elif self.strategy == LoadBalanceStrategy.RANDOM:
            return self._random_select(healthy_nodes)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_RANDOM:
            return self._weighted_random_select(healthy_nodes)
        elif self.strategy == LoadBalanceStrategy.IP_HASH:
            return self._ip_hash_select(healthy_nodes, client_ip)
        else:
            return self._round_robin_select(healthy_nodes)

    def _round_robin_select(self, nodes: List[ServiceNode]) -> ServiceNode:
        """轮询选择"""
        node = nodes[self._current_index % len(nodes)]
        self._current_index += 1
        return node

    def _weighted_round_robin_select(
        self, nodes: List[ServiceNode]
    ) -> ServiceNode:
        """加权轮询选择"""
        total_weight = sum(node.weight for node in nodes)
        if total_weight == 0:
            return self._round_robin_select(nodes)

        target = self._current_index % total_weight
        current_weight = 0

        for node in nodes:
            current_weight += node.weight
            if current_weight > target:
                self._current_index += 1
                return node

        return nodes[0]

    def _least_connections_select(
        self, nodes: List[ServiceNode]
    ) -> ServiceNode:
        """最少连接选择"""
        return min(nodes, key=lambda n: n.current_connections)

    def _random_select(self, nodes: List[ServiceNode]) -> ServiceNode:
        """随机选择"""
        return random.choice(nodes)

    def _weighted_random_select(
        self, nodes: List[ServiceNode]
    ) -> ServiceNode:
        """加权随机选择"""
        weights = [node.weight for node in nodes]
        return random.choices(nodes, weights=weights)[0]

    def _ip_hash_select(
        self, nodes: List[ServiceNode], client_ip: Optional[str]
    ) -> ServiceNode:
        """IP哈希选择"""
        if not client_ip:
            return self._round_robin_select(nodes)

        hash_value = hash(client_ip)
        index = hash_value % len(nodes)
        return nodes[index]

    async def start(self, health_check_func: Optional[Callable] = None):
        """启动负载均衡器"""
        if self.health_checker:
            await self.health_checker.start_monitoring(
                self.nodes, health_check_func
            )
        logger.info("Load balancer started")

    async def stop(self):
        """停止负载均衡器"""
        if self.health_checker:
            await self.health_checker.stop_monitoring()
        logger.info("Load balancer stopped")

    @asynccontextmanager
    async def get_connection(self, client_ip: Optional[str] = None):
        """获取连接上下文管理器"""
        node = await self.select_node(client_ip)
        if not node:
            raise RuntimeError("No available nodes")

        node.current_connections += 1
        try:
            yield node
        finally:
            node.current_connections -= 1


class CircuitBreaker:
    """熔断器"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func: Callable, *args, **kwargs):
        """调用函数并处理熔断逻辑"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise RuntimeError("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试重置"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )

    def _on_success(self):
        """成功时的处理"""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        """失败时的处理"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class FailoverManager:
    """故障转移管理器"""

    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
        self.primary_nodes: List[str] = []
        self.backup_nodes: List[str] = []

    def set_primary_nodes(self, node_ids: List[str]):
        """设置主节点"""
        self.primary_nodes = node_ids

    def set_backup_nodes(self, node_ids: List[str]):
        """设置备用节点"""
        self.backup_nodes = node_ids

    async def get_available_node(
        self, client_ip: Optional[str] = None
    ) -> Optional[ServiceNode]:
        """获取可用节点（优先主节点）"""
        # 首先尝试主节点
        primary_nodes = [
            node for node in self.load_balancer.nodes
            if node.id in self.primary_nodes and node.is_healthy
        ]

        if primary_nodes:
            # 从主节点中选择
            temp_lb = LoadBalancer(self.load_balancer.strategy)
            temp_lb.nodes = primary_nodes
            return await temp_lb.select_node(client_ip)

        # 主节点不可用，尝试备用节点
        backup_nodes = [
            node for node in self.load_balancer.nodes
            if node.id in self.backup_nodes and node.is_healthy
        ]

        if backup_nodes:
            temp_lb = LoadBalancer(self.load_balancer.strategy)
            temp_lb.nodes = backup_nodes
            return await temp_lb.select_node(client_ip)

        # 所有指定节点都不可用，使用任何可用节点
        return await self.load_balancer.select_node(client_ip)


# 使用示例
if __name__ == "__main__":
    async def example_usage():
        # 创建服务节点
        nodes = [
            ServiceNode("node1", "127.0.0.1", 8001, weight=3),
            ServiceNode("node2", "127.0.0.1", 8002, weight=2),
            ServiceNode("node3", "127.0.0.1", 8003, weight=1),
        ]

        # 创建负载均衡器
        lb = LoadBalancer(LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN)
        for node in nodes:
            lb.add_node(node)

        # 启动负载均衡器
        await lb.start()

        try:
            # 使用负载均衡器
            for i in range(10):
                async with lb.get_connection() as node:
                    print(f"Request {i} -> {node.endpoint}")
                    await asyncio.sleep(0.1)
        finally:
            await lb.stop()

    # 运行示例
    asyncio.run(example_usage())