# -*- coding: utf-8 -*-
"""
微服务架构模块
提供服务发现、负载均衡、API网关、服务治理等功能
"""

import asyncio
import json
import time
import uuid
import hashlib
import random
from typing import (
    Dict, List, Any, Optional, Union, Tuple, Callable,
    Set, AsyncIterator, Iterator
)
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import asynccontextmanager
import weakref
import socket
import ipaddress
from urllib.parse import urlparse, urljoin
import re

# 网络和HTTP库
import aiohttp
from aiohttp import web, ClientSession, ClientTimeout
from aiohttp.web_middlewares import cors_handler
from aiohttp_cors import setup as cors_setup

# 数据存储
import redis
import aioredis
from motor.motor_asyncio import AsyncIOMotorClient
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# 消息队列
import pika
import aio_pika
from celery import Celery

# 监控和指标
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import structlog

# 配置管理
import yaml
import toml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

# 安全
from cryptography.fernet import Fernet
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta

# 其他工具
import consul
import etcd3
from kazoo.client import KazooClient
from kazoo.recipe.watchers import DataWatch
import requests
from jinja2 import Template
import schedule
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)


class ServiceStatus(Enum):
    """服务状态枚举"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"


class LoadBalanceStrategy(Enum):
    """负载均衡策略枚举"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"
    IP_HASH = "ip_hash"
    CONSISTENT_HASH = "consistent_hash"
    LEAST_RESPONSE_TIME = "least_response_time"
    HEALTH_BASED = "health_based"


class ProtocolType(Enum):
    """协议类型枚举"""
    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"
    UDP = "udp"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    MQTT = "mqtt"
    AMQP = "amqp"


class CircuitBreakerState(Enum):
    """熔断器状态枚举"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RateLimitStrategy(Enum):
    """限流策略枚举"""
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    SLIDING_LOG = "sliding_log"


@dataclass
class ServiceInstance:
    """服务实例"""
    id: str
    name: str
    host: str
    port: int
    protocol: ProtocolType = ProtocolType.HTTP
    weight: int = 100
    status: ServiceStatus = ServiceStatus.STARTING
    metadata: Dict[str, Any] = field(default_factory=dict)
    health_check_url: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    version: str = "1.0.0"
    region: Optional[str] = None
    zone: Optional[str] = None
    datacenter: Optional[str] = None
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    failure_count: int = 0
    success_count: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    current_connections: int = 0
    max_connections: int = 1000
    
    @property
    def url(self) -> str:
        """获取服务URL"""
        return f"{self.protocol.value}://{self.host}:{self.port}"
    
    @property
    def is_healthy(self) -> bool:
        """检查服务是否健康"""
        if self.status not in [ServiceStatus.RUNNING, ServiceStatus.DEGRADED]:
            return False
        
        # 检查心跳超时
        heartbeat_timeout = timedelta(seconds=30)
        if datetime.now() - self.last_heartbeat > heartbeat_timeout:
            return False
        
        # 检查失败率
        if self.total_requests > 0:
            failure_rate = self.failure_count / self.total_requests
            if failure_rate > 0.5:  # 失败率超过50%
                return False
        
        return True
    
    def update_metrics(self, success: bool, response_time: float):
        """更新指标"""
        self.total_requests += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # 更新平均响应时间
        self.avg_response_time = (
            (self.avg_response_time * (self.total_requests - 1) + response_time) / 
            self.total_requests
        )
        
        self.last_heartbeat = datetime.now()


@dataclass
class ServiceConfig:
    """服务配置"""
    name: str
    instances: List[ServiceInstance] = field(default_factory=list)
    load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    health_check_interval: int = 10  # 秒
    health_check_timeout: int = 5  # 秒
    health_check_retries: int = 3
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60  # 秒
    rate_limit_enabled: bool = False
    rate_limit_strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # 秒
    timeout: int = 30  # 秒
    retries: int = 3
    retry_backoff: float = 1.0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteConfig:
    """路由配置"""
    path: str
    service_name: str
    methods: List[str] = field(default_factory=lambda: ['GET'])
    strip_prefix: bool = False
    add_prefix: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    middleware: List[str] = field(default_factory=list)
    auth_required: bool = False
    rate_limit: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = None
    retries: Optional[int] = None
    circuit_breaker: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: int = 60  # 秒
    monitor_window: int = 60  # 秒
    min_requests: int = 10


@dataclass
class RateLimitConfig:
    """限流配置"""
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    requests: int = 100
    window: int = 60  # 秒
    burst: Optional[int] = None
    key_func: Optional[Callable] = None


class ServiceDiscovery(ABC):
    """服务发现抽象基类"""
    
    @abstractmethod
    async def register_service(self, instance: ServiceInstance) -> bool:
        """注册服务"""
        pass
    
    @abstractmethod
    async def deregister_service(self, instance_id: str) -> bool:
        """注销服务"""
        pass
    
    @abstractmethod
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """发现服务"""
        pass
    
    @abstractmethod
    async def watch_service(self, service_name: str, callback: Callable) -> None:
        """监听服务变化"""
        pass
    
    @abstractmethod
    async def health_check(self, instance: ServiceInstance) -> bool:
        """健康检查"""
        pass


class ConsulServiceDiscovery(ServiceDiscovery):
    """Consul服务发现"""
    
    def __init__(self, host: str = 'localhost', port: int = 8500):
        self.consul = consul.Consul(host=host, port=port)
        self.registered_services: Dict[str, ServiceInstance] = {}
        self.watchers: Dict[str, List[Callable]] = {}
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
    
    async def register_service(self, instance: ServiceInstance) -> bool:
        """注册服务到Consul"""
        try:
            service_def = {
                'ID': instance.id,
                'Name': instance.name,
                'Address': instance.host,
                'Port': instance.port,
                'Tags': list(instance.tags),
                'Meta': instance.metadata,
                'Check': {
                    'HTTP': instance.health_check_url or f"{instance.url}/health",
                    'Interval': '10s',
                    'Timeout': '5s',
                    'DeregisterCriticalServiceAfter': '30s'
                }
            }
            
            success = self.consul.agent.service.register(**service_def)
            
            if success:
                self.registered_services[instance.id] = instance
                instance.status = ServiceStatus.RUNNING
                
                # 启动健康检查任务
                task = asyncio.create_task(self._health_check_loop(instance))
                self.health_check_tasks[instance.id] = task
                
                logger.info(f"服务已注册到Consul: {instance.name}#{instance.id}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"服务注册失败: {e}", instance_id=instance.id)
            return False
    
    async def deregister_service(self, instance_id: str) -> bool:
        """从Consul注销服务"""
        try:
            success = self.consul.agent.service.deregister(instance_id)
            
            if success and instance_id in self.registered_services:
                instance = self.registered_services[instance_id]
                instance.status = ServiceStatus.STOPPED
                del self.registered_services[instance_id]
                
                # 停止健康检查任务
                if instance_id in self.health_check_tasks:
                    self.health_check_tasks[instance_id].cancel()
                    del self.health_check_tasks[instance_id]
                
                logger.info(f"服务已从Consul注销: {instance_id}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"服务注销失败: {e}", instance_id=instance_id)
            return False
    
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """从Consul发现服务"""
        try:
            _, services = self.consul.health.service(service_name, passing=True)
            instances = []
            
            for service in services:
                service_info = service['Service']
                instance = ServiceInstance(
                    id=service_info['ID'],
                    name=service_info['Service'],
                    host=service_info['Address'],
                    port=service_info['Port'],
                    tags=set(service_info.get('Tags', [])),
                    metadata=service_info.get('Meta', {}),
                    status=ServiceStatus.RUNNING
                )
                instances.append(instance)
            
            return instances
        
        except Exception as e:
            logger.error(f"服务发现失败: {e}", service_name=service_name)
            return []
    
    async def watch_service(self, service_name: str, callback: Callable) -> None:
        """监听服务变化"""
        if service_name not in self.watchers:
            self.watchers[service_name] = []
        
        self.watchers[service_name].append(callback)
        
        # 启动监听任务
        asyncio.create_task(self._watch_service_loop(service_name))
    
    async def _watch_service_loop(self, service_name: str):
        """服务监听循环"""
        last_index = None
        
        while True:
            try:
                index, services = self.consul.health.service(
                    service_name, index=last_index, wait='10s'
                )
                
                if index != last_index:
                    last_index = index
                    instances = await self.discover_services(service_name)
                    
                    # 通知所有回调
                    for callback in self.watchers.get(service_name, []):
                        try:
                            await callback(service_name, instances)
                        except Exception as e:
                            logger.error(f"服务监听回调失败: {e}")
            
            except Exception as e:
                logger.error(f"服务监听失败: {e}")
                await asyncio.sleep(5)
    
    async def health_check(self, instance: ServiceInstance) -> bool:
        """健康检查"""
        try:
            health_url = instance.health_check_url or f"{instance.url}/health"
            
            async with aiohttp.ClientSession(timeout=ClientTimeout(total=5)) as session:
                async with session.get(health_url) as response:
                    return response.status == 200
        
        except Exception as e:
            logger.warning(f"健康检查失败: {e}", instance_id=instance.id)
            return False
    
    async def _health_check_loop(self, instance: ServiceInstance):
        """健康检查循环"""
        while instance.id in self.registered_services:
            try:
                is_healthy = await self.health_check(instance)
                
                if is_healthy:
                    if instance.status == ServiceStatus.FAILED:
                        instance.status = ServiceStatus.RUNNING
                        logger.info(f"服务恢复健康: {instance.id}")
                else:
                    instance.failure_count += 1
                    if instance.failure_count >= 3:
                        instance.status = ServiceStatus.FAILED
                        logger.warning(f"服务不健康: {instance.id}")
                
                await asyncio.sleep(10)  # 10秒检查一次
            
            except Exception as e:
                logger.error(f"健康检查循环错误: {e}")
                await asyncio.sleep(10)


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.round_robin_counters: Dict[str, int] = {}
        self.consistent_hash_rings: Dict[str, Dict[int, ServiceInstance]] = {}
        self.metrics = {
            'requests_total': Counter('lb_requests_total'),
            'request_duration': Histogram('lb_request_duration_seconds'),
            'active_connections': Gauge('lb_active_connections')
        }
    
    async def select_instance(self, service_name: str, instances: List[ServiceInstance], 
                            client_ip: Optional[str] = None) -> Optional[ServiceInstance]:
        """选择服务实例"""
        if not instances:
            return None
        
        # 过滤健康的实例
        healthy_instances = [inst for inst in instances if inst.is_healthy]
        if not healthy_instances:
            # 如果没有健康实例，使用所有实例
            healthy_instances = instances
        
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin_select(service_name, healthy_instances)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(service_name, healthy_instances)
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_instances)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_LEAST_CONNECTIONS:
            return self._weighted_least_connections_select(healthy_instances)
        elif self.strategy == LoadBalanceStrategy.RANDOM:
            return self._random_select(healthy_instances)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_RANDOM:
            return self._weighted_random_select(healthy_instances)
        elif self.strategy == LoadBalanceStrategy.IP_HASH:
            return self._ip_hash_select(healthy_instances, client_ip)
        elif self.strategy == LoadBalanceStrategy.CONSISTENT_HASH:
            return self._consistent_hash_select(service_name, healthy_instances, client_ip)
        elif self.strategy == LoadBalanceStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_select(healthy_instances)
        elif self.strategy == LoadBalanceStrategy.HEALTH_BASED:
            return self._health_based_select(healthy_instances)
        else:
            return healthy_instances[0]
    
    def _round_robin_select(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """轮询选择"""
        if service_name not in self.round_robin_counters:
            self.round_robin_counters[service_name] = 0
        
        index = self.round_robin_counters[service_name] % len(instances)
        self.round_robin_counters[service_name] += 1
        
        return instances[index]
    
    def _weighted_round_robin_select(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """加权轮询选择"""
        total_weight = sum(inst.weight for inst in instances)
        if total_weight == 0:
            return self._round_robin_select(service_name, instances)
        
        if service_name not in self.round_robin_counters:
            self.round_robin_counters[service_name] = 0
        
        target = self.round_robin_counters[service_name] % total_weight
        self.round_robin_counters[service_name] += 1
        
        current_weight = 0
        for instance in instances:
            current_weight += instance.weight
            if current_weight > target:
                return instance
        
        return instances[0]
    
    def _least_connections_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """最少连接选择"""
        return min(instances, key=lambda inst: inst.current_connections)
    
    def _weighted_least_connections_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """加权最少连接选择"""
        def connection_ratio(inst):
            if inst.weight == 0:
                return float('inf')
            return inst.current_connections / inst.weight
        
        return min(instances, key=connection_ratio)
    
    def _random_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """随机选择"""
        return random.choice(instances)
    
    def _weighted_random_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """加权随机选择"""
        total_weight = sum(inst.weight for inst in instances)
        if total_weight == 0:
            return self._random_select(instances)
        
        target = random.randint(0, total_weight - 1)
        current_weight = 0
        
        for instance in instances:
            current_weight += instance.weight
            if current_weight > target:
                return instance
        
        return instances[0]
    
    def _ip_hash_select(self, instances: List[ServiceInstance], client_ip: Optional[str]) -> ServiceInstance:
        """IP哈希选择"""
        if not client_ip:
            return self._random_select(instances)
        
        hash_value = hash(client_ip)
        index = hash_value % len(instances)
        return instances[index]
    
    def _consistent_hash_select(self, service_name: str, instances: List[ServiceInstance], 
                              client_ip: Optional[str]) -> ServiceInstance:
        """一致性哈希选择"""
        if not client_ip:
            return self._random_select(instances)
        
        # 构建一致性哈希环
        if service_name not in self.consistent_hash_rings:
            self.consistent_hash_rings[service_name] = {}
        
        ring = self.consistent_hash_rings[service_name]
        ring.clear()
        
        # 为每个实例创建虚拟节点
        virtual_nodes = 150
        for instance in instances:
            for i in range(virtual_nodes):
                key = f"{instance.id}:{i}"
                hash_value = hash(key) % (2**32)
                ring[hash_value] = instance
        
        # 查找客户端IP对应的节点
        client_hash = hash(client_ip) % (2**32)
        
        # 找到第一个大于等于客户端哈希值的节点
        sorted_keys = sorted(ring.keys())
        for key in sorted_keys:
            if key >= client_hash:
                return ring[key]
        
        # 如果没找到，返回第一个节点
        return ring[sorted_keys[0]] if sorted_keys else instances[0]
    
    def _least_response_time_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """最少响应时间选择"""
        return min(instances, key=lambda inst: inst.avg_response_time)
    
    def _health_based_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """基于健康度选择"""
        def health_score(inst):
            if inst.total_requests == 0:
                return 1.0
            
            success_rate = inst.success_count / inst.total_requests
            response_time_factor = 1.0 / (1.0 + inst.avg_response_time)
            connection_factor = 1.0 / (1.0 + inst.current_connections / inst.max_connections)
            
            return success_rate * response_time_factor * connection_factor
        
        return max(instances, key=health_score)


class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.request_count = 0
        self.window_start = time.time()
        self.lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """通过熔断器调用函数"""
        async with self.lock:
            current_time = time.time()
            
            # 重置监控窗口
            if current_time - self.window_start >= self.config.monitor_window:
                self._reset_window()
            
            # 检查熔断器状态
            if self.state == CircuitBreakerState.OPEN:
                if current_time - self.last_failure_time >= self.config.timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    logger.info("熔断器进入半开状态")
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    logger.info("熔断器关闭")
        
        # 执行函数
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self):
        """成功回调"""
        async with self.lock:
            self.request_count += 1
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
    
    async def _on_failure(self):
        """失败回调"""
        async with self.lock:
            self.request_count += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # 检查是否需要打开熔断器
            if (self.state == CircuitBreakerState.CLOSED and 
                self.request_count >= self.config.min_requests and
                self.failure_count >= self.config.failure_threshold):
                
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"熔断器打开，失败次数: {self.failure_count}")
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning("熔断器重新打开")
    
    def _reset_window(self):
        """重置监控窗口"""
        self.window_start = time.time()
        self.request_count = 0
        self.failure_count = 0
    
    @property
    def is_open(self) -> bool:
        """检查熔断器是否打开"""
        return self.state == CircuitBreakerState.OPEN


class RateLimiter:
    """限流器"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.buckets: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, key: str) -> bool:
        """检查是否允许请求"""
        async with self.lock:
            current_time = time.time()
            
            if key not in self.buckets:
                self.buckets[key] = self._create_bucket(current_time)
            
            bucket = self.buckets[key]
            
            if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return self._token_bucket_check(bucket, current_time)
            elif self.config.strategy == RateLimitStrategy.LEAKY_BUCKET:
                return self._leaky_bucket_check(bucket, current_time)
            elif self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
                return self._fixed_window_check(bucket, current_time)
            elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return self._sliding_window_check(bucket, current_time)
            elif self.config.strategy == RateLimitStrategy.SLIDING_LOG:
                return self._sliding_log_check(bucket, current_time)
            else:
                return True
    
    def _create_bucket(self, current_time: float) -> Dict[str, Any]:
        """创建令牌桶"""
        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return {
                'tokens': self.config.requests,
                'last_refill': current_time,
                'capacity': self.config.requests
            }
        elif self.config.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return {
                'queue': [],
                'last_leak': current_time,
                'capacity': self.config.burst or self.config.requests
            }
        elif self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
            return {
                'count': 0,
                'window_start': current_time
            }
        elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return {
                'windows': {},
                'current_window': int(current_time // self.config.window)
            }
        elif self.config.strategy == RateLimitStrategy.SLIDING_LOG:
            return {
                'requests': []
            }
        else:
            return {}
    
    def _token_bucket_check(self, bucket: Dict[str, Any], current_time: float) -> bool:
        """令牌桶检查"""
        # 计算需要添加的令牌数
        time_passed = current_time - bucket['last_refill']
        tokens_to_add = time_passed * (self.config.requests / self.config.window)
        
        # 更新令牌数
        bucket['tokens'] = min(bucket['capacity'], bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = current_time
        
        # 检查是否有足够的令牌
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True
        
        return False
    
    def _leaky_bucket_check(self, bucket: Dict[str, Any], current_time: float) -> bool:
        """漏桶检查"""
        # 计算需要漏出的请求数
        time_passed = current_time - bucket['last_leak']
        requests_to_leak = time_passed * (self.config.requests / self.config.window)
        
        # 漏出请求
        for _ in range(int(requests_to_leak)):
            if bucket['queue']:
                bucket['queue'].pop(0)
        
        bucket['last_leak'] = current_time
        
        # 检查队列是否已满
        if len(bucket['queue']) < bucket['capacity']:
            bucket['queue'].append(current_time)
            return True
        
        return False
    
    def _fixed_window_check(self, bucket: Dict[str, Any], current_time: float) -> bool:
        """固定窗口检查"""
        window_start = int(current_time // self.config.window) * self.config.window
        
        # 重置窗口
        if window_start > bucket['window_start']:
            bucket['count'] = 0
            bucket['window_start'] = window_start
        
        # 检查请求数是否超限
        if bucket['count'] < self.config.requests:
            bucket['count'] += 1
            return True
        
        return False
    
    def _sliding_window_check(self, bucket: Dict[str, Any], current_time: float) -> bool:
        """滑动窗口检查"""
        current_window = int(current_time // self.config.window)
        
        # 清理过期窗口
        expired_windows = [w for w in bucket['windows'] if w < current_window - 1]
        for w in expired_windows:
            del bucket['windows'][w]
        
        # 计算当前请求数
        total_requests = sum(bucket['windows'].values())
        
        # 检查是否超限
        if total_requests < self.config.requests:
            if current_window not in bucket['windows']:
                bucket['windows'][current_window] = 0
            bucket['windows'][current_window] += 1
            return True
        
        return False
    
    def _sliding_log_check(self, bucket: Dict[str, Any], current_time: float) -> bool:
        """滑动日志检查"""
        # 清理过期请求
        cutoff_time = current_time - self.config.window
        bucket['requests'] = [t for t in bucket['requests'] if t > cutoff_time]
        
        # 检查请求数是否超限
        if len(bucket['requests']) < self.config.requests:
            bucket['requests'].append(current_time)
            return True
        
        return False


class APIGateway:
    """API网关"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.service_discovery: Optional[ServiceDiscovery] = None
        self.load_balancer = LoadBalancer()
        self.routes: Dict[str, RouteConfig] = {}
        self.services: Dict[str, ServiceConfig] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.middleware_registry: Dict[str, Callable] = {}
        self.metrics = {
            'requests_total': Counter('gateway_requests_total'),
            'request_duration': Histogram('gateway_request_duration_seconds'),
            'errors_total': Counter('gateway_errors_total')
        }
        
        # 设置中间件
        self._setup_middleware()
        
        # 设置路由
        self._setup_routes()
    
    def set_service_discovery(self, service_discovery: ServiceDiscovery):
        """设置服务发现"""
        self.service_discovery = service_discovery
    
    def add_route(self, route_config: RouteConfig):
        """添加路由"""
        self.routes[route_config.path] = route_config
        logger.info(f"路由已添加: {route_config.path} -> {route_config.service_name}")
    
    def add_service(self, service_config: ServiceConfig):
        """添加服务"""
        self.services[service_config.name] = service_config
        
        # 创建熔断器
        if service_config.circuit_breaker_enabled:
            cb_config = CircuitBreakerConfig(
                failure_threshold=service_config.circuit_breaker_threshold,
                timeout=service_config.circuit_breaker_timeout
            )
            self.circuit_breakers[service_config.name] = CircuitBreaker(cb_config)
        
        # 创建限流器
        if service_config.rate_limit_enabled:
            rl_config = RateLimitConfig(
                strategy=service_config.rate_limit_strategy,
                requests=service_config.rate_limit_requests,
                window=service_config.rate_limit_window
            )
            self.rate_limiters[service_config.name] = RateLimiter(rl_config)
        
        logger.info(f"服务已添加: {service_config.name}")
    
    def register_middleware(self, name: str, middleware: Callable):
        """注册中间件"""
        self.middleware_registry[name] = middleware
        logger.info(f"中间件已注册: {name}")
    
    def _setup_middleware(self):
        """设置中间件"""
        # CORS中间件
        cors = cors_setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # 请求日志中间件
        self.app.middlewares.append(self._logging_middleware)
        
        # 指标中间件
        self.app.middlewares.append(self._metrics_middleware)
        
        # 错误处理中间件
        self.app.middlewares.append(self._error_middleware)
    
    def _setup_routes(self):
        """设置路由"""
        # 健康检查
        self.app.router.add_get('/health', self._health_handler)
        
        # 指标端点
        self.app.router.add_get('/metrics', self._metrics_handler)
        
        # 代理路由
        self.app.router.add_route('*', '/{path:.*}', self._proxy_handler)
    
    async def _logging_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """请求日志中间件"""
        start_time = time.time()
        
        try:
            response = await handler(request)
            duration = time.time() - start_time
            
            logger.info(
                "请求处理完成",
                method=request.method,
                path=request.path,
                status=response.status,
                duration=duration,
                remote=request.remote
            )
            
            return response
        
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(
                "请求处理失败",
                method=request.method,
                path=request.path,
                duration=duration,
                error=str(e),
                remote=request.remote
            )
            
            raise
    
    async def _metrics_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """指标中间件"""
        start_time = time.time()
        
        try:
            response = await handler(request)
            duration = time.time() - start_time
            
            # 记录指标
            self.metrics['requests_total'].labels(
                method=request.method,
                path=request.path,
                status=response.status
            ).inc()
            
            self.metrics['request_duration'].labels(
                method=request.method,
                path=request.path
            ).observe(duration)
            
            return response
        
        except Exception as e:
            duration = time.time() - start_time
            
            self.metrics['errors_total'].labels(
                method=request.method,
                path=request.path,
                error=type(e).__name__
            ).inc()
            
            self.metrics['request_duration'].labels(
                method=request.method,
                path=request.path
            ).observe(duration)
            
            raise
    
    async def _error_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """错误处理中间件"""
        try:
            return await handler(request)
        
        except web.HTTPException:
            raise
        
        except Exception as e:
            logger.error(f"未处理的错误: {e}", exc_info=True)
            
            return web.json_response(
                {'error': 'Internal Server Error', 'message': str(e)},
                status=500
            )
    
    async def _health_handler(self, request: web.Request) -> web.Response:
        """健康检查处理器"""
        return web.json_response({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
    
    async def _metrics_handler(self, request: web.Request) -> web.Response:
        """指标处理器"""
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        
        return web.Response(
            text=generate_latest().decode('utf-8'),
            content_type=CONTENT_TYPE_LATEST
        )
    
    async def _proxy_handler(self, request: web.Request) -> web.Response:
        """代理处理器"""
        path = request.match_info.get('path', '')
        
        # 查找匹配的路由
        route_config = self._find_route(request.method, path)
        if not route_config:
            raise web.HTTPNotFound(text='Route not found')
        
        # 检查认证
        if route_config.auth_required:
            if not await self._check_auth(request):
                raise web.HTTPUnauthorized(text='Authentication required')
        
        # 限流检查
        if route_config.rate_limit:
            rate_limiter = self.rate_limiters.get(route_config.service_name)
            if rate_limiter:
                client_key = self._get_client_key(request, route_config.rate_limit)
                if not await rate_limiter.is_allowed(client_key):
                    raise web.HTTPTooManyRequests(text='Rate limit exceeded')
        
        # 获取服务实例
        instances = await self._get_service_instances(route_config.service_name)
        if not instances:
            raise web.HTTPServiceUnavailable(text='Service unavailable')
        
        # 负载均衡选择实例
        instance = await self.load_balancer.select_instance(
            route_config.service_name, instances, request.remote
        )
        
        if not instance:
            raise web.HTTPServiceUnavailable(text='No healthy instances')
        
        # 构建目标URL
        target_url = self._build_target_url(instance, request, route_config)
        
        # 执行代理请求
        try:
            # 使用熔断器
            circuit_breaker = self.circuit_breakers.get(route_config.service_name)
            if circuit_breaker:
                response = await circuit_breaker.call(
                    self._make_request, instance, target_url, request, route_config
                )
            else:
                response = await self._make_request(instance, target_url, request, route_config)
            
            return response
        
        except Exception as e:
            logger.error(f"代理请求失败: {e}", target_url=target_url)
            raise web.HTTPBadGateway(text='Proxy request failed')
    
    def _find_route(self, method: str, path: str) -> Optional[RouteConfig]:
        """查找匹配的路由"""
        for route_path, route_config in self.routes.items():
            if self._match_route(route_path, path) and method in route_config.methods:
                return route_config
        return None
    
    def _match_route(self, route_path: str, request_path: str) -> bool:
        """匹配路由"""
        # 简单的路径匹配，可以扩展为正则表达式匹配
        if route_path == request_path:
            return True
        
        # 支持通配符匹配
        if route_path.endswith('*'):
            prefix = route_path[:-1]
            return request_path.startswith(prefix)
        
        return False
    
    async def _check_auth(self, request: web.Request) -> bool:
        """检查认证"""
        # 简单的Bearer token认证
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return False
        
        token = auth_header[7:]
        # 这里应该验证token的有效性
        return len(token) > 0
    
    def _get_client_key(self, request: web.Request, rate_limit_config: Dict[str, Any]) -> str:
        """获取客户端标识"""
        # 可以基于IP、用户ID、API Key等生成
        return request.remote or 'unknown'
    
    async def _get_service_instances(self, service_name: str) -> List[ServiceInstance]:
        """获取服务实例"""
        if self.service_discovery:
            return await self.service_discovery.discover_services(service_name)
        
        # 从配置中获取
        service_config = self.services.get(service_name)
        if service_config:
            return service_config.instances
        
        return []
    
    def _build_target_url(self, instance: ServiceInstance, request: web.Request, 
                         route_config: RouteConfig) -> str:
        """构建目标URL"""
        path = request.path_qs
        
        # 处理路径前缀
        if route_config.strip_prefix:
            # 移除匹配的路由前缀
            route_prefix = route_config.path.rstrip('*')
            if path.startswith(route_prefix):
                path = path[len(route_prefix):]
        
        if route_config.add_prefix:
            path = route_config.add_prefix + path
        
        return f"{instance.url}{path}"
    
    async def _make_request(self, instance: ServiceInstance, target_url: str, 
                          request: web.Request, route_config: RouteConfig) -> web.Response:
        """发起代理请求"""
        start_time = time.time()
        
        try:
            # 更新连接数
            instance.current_connections += 1
            
            # 准备请求头
            headers = dict(request.headers)
            headers.update(route_config.headers)
            
            # 移除hop-by-hop头
            hop_by_hop_headers = {
                'connection', 'keep-alive', 'proxy-authenticate',
                'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade'
            }
            for header in hop_by_hop_headers:
                headers.pop(header, None)
            
            # 读取请求体
            body = await request.read() if request.can_read_body else None
            
            # 设置超时
            timeout = route_config.timeout or 30
            client_timeout = ClientTimeout(total=timeout)
            
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    data=body,
                    params=route_config.query_params
                ) as response:
                    
                    # 读取响应
                    response_body = await response.read()
                    response_headers = dict(response.headers)
                    
                    # 移除hop-by-hop头
                    for header in hop_by_hop_headers:
                        response_headers.pop(header, None)
                    
                    # 更新实例指标
                    response_time = time.time() - start_time
                    instance.update_metrics(True, response_time)
                    
                    return web.Response(
                        body=response_body,
                        status=response.status,
                        headers=response_headers
                    )
        
        except Exception as e:
            # 更新实例指标
            response_time = time.time() - start_time
            instance.update_metrics(False, response_time)
            raise e
        
        finally:
            # 减少连接数
            instance.current_connections = max(0, instance.current_connections - 1)
    
    async def start(self):
        """启动网关"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"API网关已启动: http://{self.host}:{self.port}")
    
    async def stop(self):
        """停止网关"""
        await self.app.cleanup()
        logger.info("API网关已停止")


# 示例使用
async def example_usage():
    """示例用法"""
    # 创建服务发现
    service_discovery = ConsulServiceDiscovery()
    
    # 创建API网关
    gateway = APIGateway()
    gateway.set_service_discovery(service_discovery)
    
    # 注册服务实例
    user_service = ServiceInstance(
        id="user-service-1",
        name="user-service",
        host="localhost",
        port=8001,
        health_check_url="http://localhost:8001/health"
    )
    
    await service_discovery.register_service(user_service)
    
    # 添加服务配置
    service_config = ServiceConfig(
        name="user-service",
        instances=[user_service],
        load_balance_strategy=LoadBalanceStrategy.ROUND_ROBIN,
        circuit_breaker_enabled=True,
        rate_limit_enabled=True,
        rate_limit_requests=100,
        rate_limit_window=60
    )
    
    gateway.add_service(service_config)
    
    # 添加路由
    route_config = RouteConfig(
        path="/api/users/*",
        service_name="user-service",
        methods=["GET", "POST", "PUT", "DELETE"],
        strip_prefix=True,
        auth_required=True
    )
    
    gateway.add_route(route_config)
    
    # 启动网关
    await gateway.start()
    
    print("API网关运行中...")
    
    # 保持运行
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await gateway.stop()
        await service_discovery.deregister_service(user_service.id)


if __name__ == "__main__":
    asyncio.run(example_usage())