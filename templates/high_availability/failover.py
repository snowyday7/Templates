"""故障转移系统

提供完整的故障转移功能，包括：
- 自动故障检测
- 故障转移策略
- 服务恢复
- 状态同步
- 故障通知
"""

import time
import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from collections import defaultdict, deque

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

try:
    import requests
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False


class FailoverStrategy(str, Enum):
    """故障转移策略"""
    ACTIVE_PASSIVE = "active_passive"
    ACTIVE_ACTIVE = "active_active"
    ROUND_ROBIN = "round_robin"
    PRIORITY_BASED = "priority_based"
    GEOGRAPHIC = "geographic"
    LOAD_BASED = "load_based"
    CUSTOM = "custom"


class ServiceStatus(str, Enum):
    """服务状态"""
    ACTIVE = "active"
    STANDBY = "standby"
    FAILED = "failed"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class FailoverTrigger(str, Enum):
    """故障转移触发器"""
    HEALTH_CHECK_FAILURE = "health_check_failure"
    RESPONSE_TIME_THRESHOLD = "response_time_threshold"
    ERROR_RATE_THRESHOLD = "error_rate_threshold"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MANUAL_TRIGGER = "manual_trigger"
    EXTERNAL_SIGNAL = "external_signal"


class FailoverEvent(str, Enum):
    """故障转移事件"""
    FAILOVER_STARTED = "failover_started"
    FAILOVER_COMPLETED = "failover_completed"
    FAILOVER_FAILED = "failover_failed"
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_COMPLETED = "recovery_completed"
    RECOVERY_FAILED = "recovery_failed"
    SERVICE_DEGRADED = "service_degraded"
    SERVICE_RESTORED = "service_restored"


@dataclass
class ServiceInstance:
    """服务实例"""
    id: str
    name: str
    host: str
    port: int
    priority: int = 1  # 优先级，数字越小优先级越高
    status: ServiceStatus = ServiceStatus.STANDBY
    
    # 健康检查
    health_check_url: Optional[str] = None
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    # 性能指标
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    request_count: int = 0
    
    # 资源使用
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    
    # 地理位置
    region: Optional[str] = None
    zone: Optional[str] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    @property
    def url(self) -> str:
        """获取服务URL"""
        return f"http://{self.host}:{self.port}"
    
    @property
    def is_healthy(self) -> bool:
        """检查服务是否健康"""
        return self.status in [ServiceStatus.ACTIVE, ServiceStatus.STANDBY]
    
    @property
    def is_active(self) -> bool:
        """检查服务是否活跃"""
        return self.status == ServiceStatus.ACTIVE
    
    @property
    def average_response_time(self) -> float:
        """获取平均响应时间"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    @property
    def error_rate(self) -> float:
        """获取错误率"""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count
    
    def record_request(self, response_time: float, success: bool):
        """记录请求"""
        self.response_times.append(response_time)
        self.request_count += 1
        if not success:
            self.error_count += 1
        self.last_updated = time.time()
    
    def update_resource_usage(self, cpu: float, memory: float, disk: float):
        """更新资源使用情况"""
        self.cpu_usage = cpu
        self.memory_usage = memory
        self.disk_usage = disk
        self.last_updated = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "priority": self.priority,
            "status": self.status.value,
            "url": self.url,
            "is_healthy": self.is_healthy,
            "is_active": self.is_active,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "average_response_time": self.average_response_time,
            "error_rate": self.error_rate,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "disk_usage": self.disk_usage,
            "region": self.region,
            "zone": self.zone,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }


@dataclass
class FailoverRecord:
    """故障转移记录"""
    id: str
    event: FailoverEvent
    trigger: FailoverTrigger
    from_service: Optional[str]
    to_service: Optional[str]
    timestamp: float = field(default_factory=time.time)
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "event": self.event.value,
            "trigger": self.trigger.value,
            "from_service": self.from_service,
            "to_service": self.to_service,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


class FailoverHandler(ABC):
    """故障转移处理器基类"""
    
    @abstractmethod
    async def can_handle_failover(self, from_service: ServiceInstance, 
                                 to_service: ServiceInstance) -> bool:
        """检查是否可以处理故障转移"""
        pass
    
    @abstractmethod
    async def execute_failover(self, from_service: ServiceInstance, 
                              to_service: ServiceInstance) -> bool:
        """执行故障转移"""
        pass
    
    @abstractmethod
    async def execute_recovery(self, service: ServiceInstance) -> bool:
        """执行服务恢复"""
        pass


class DefaultFailoverHandler(FailoverHandler):
    """默认故障转移处理器"""
    
    async def can_handle_failover(self, from_service: ServiceInstance, 
                                 to_service: ServiceInstance) -> bool:
        """检查是否可以处理故障转移"""
        return to_service.is_healthy and not to_service.is_active
    
    async def execute_failover(self, from_service: ServiceInstance, 
                              to_service: ServiceInstance) -> bool:
        """执行故障转移"""
        try:
            # 停用原服务
            from_service.status = ServiceStatus.FAILED
            
            # 激活新服务
            to_service.status = ServiceStatus.ACTIVE
            
            # 这里可以添加具体的故障转移逻辑
            # 例如：更新负载均衡器配置、DNS记录等
            
            return True
        except Exception as e:
            print(f"Failover execution failed: {e}")
            return False
    
    async def execute_recovery(self, service: ServiceInstance) -> bool:
        """执行服务恢复"""
        try:
            # 将服务状态设置为恢复中
            service.status = ServiceStatus.RECOVERING
            
            # 这里可以添加具体的恢复逻辑
            # 例如：重启服务、清理资源等
            
            # 恢复完成后设置为待机状态
            service.status = ServiceStatus.STANDBY
            
            return True
        except Exception as e:
            print(f"Service recovery failed: {e}")
            return False


class FailoverManager:
    """故障转移管理器"""
    
    def __init__(self, strategy: FailoverStrategy = FailoverStrategy.ACTIVE_PASSIVE,
                 health_check_interval: int = 30, failure_threshold: int = 3,
                 recovery_threshold: int = 2, response_time_threshold: float = 5.0,
                 error_rate_threshold: float = 0.1):
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.failure_threshold = failure_threshold
        self.recovery_threshold = recovery_threshold
        self.response_time_threshold = response_time_threshold
        self.error_rate_threshold = error_rate_threshold
        
        self.services: Dict[str, ServiceInstance] = {}
        self.handlers: List[FailoverHandler] = [DefaultFailoverHandler()]
        self.records: List[FailoverRecord] = []
        self.event_callbacks: Dict[FailoverEvent, List[Callable]] = defaultdict(list)
        
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread = None
    
    def add_service(self, service: ServiceInstance):
        """添加服务"""
        with self._lock:
            self.services[service.id] = service
    
    def remove_service(self, service_id: str):
        """移除服务"""
        with self._lock:
            self.services.pop(service_id, None)
    
    def get_service(self, service_id: str) -> Optional[ServiceInstance]:
        """获取服务"""
        return self.services.get(service_id)
    
    def get_active_services(self) -> List[ServiceInstance]:
        """获取活跃服务"""
        return [s for s in self.services.values() if s.is_active]
    
    def get_standby_services(self) -> List[ServiceInstance]:
        """获取待机服务"""
        return [s for s in self.services.values() if s.status == ServiceStatus.STANDBY]
    
    def get_failed_services(self) -> List[ServiceInstance]:
        """获取失败服务"""
        return [s for s in self.services.values() if s.status == ServiceStatus.FAILED]
    
    def add_handler(self, handler: FailoverHandler):
        """添加故障转移处理器"""
        self.handlers.append(handler)
    
    def add_event_callback(self, event: FailoverEvent, callback: Callable):
        """添加事件回调"""
        self.event_callbacks[event].append(callback)
    
    def start_monitoring(self):
        """启动监控"""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                asyncio.run(self._check_services())
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            time.sleep(self.health_check_interval)
    
    async def _check_services(self):
        """检查服务状态"""
        for service in list(self.services.values()):
            try:
                await self._check_service_health(service)
                await self._check_service_performance(service)
                await self._check_service_resources(service)
            except Exception as e:
                print(f"Service check failed for {service.id}: {e}")
    
    async def _check_service_health(self, service: ServiceInstance):
        """检查服务健康状态"""
        if not HTTP_AVAILABLE or not service.health_check_url:
            return
        
        try:
            start_time = time.time()
            response = requests.get(service.health_check_url, timeout=5)
            response_time = time.time() - start_time
            
            success = response.status_code < 400
            service.record_request(response_time, success)
            
            if success:
                service.consecutive_successes += 1
                service.consecutive_failures = 0
                
                # 如果服务从失败状态恢复
                if (service.status == ServiceStatus.FAILED and 
                    service.consecutive_successes >= self.recovery_threshold):
                    await self._trigger_recovery(service)
            else:
                service.consecutive_failures += 1
                service.consecutive_successes = 0
                
                # 如果连续失败次数达到阈值
                if (service.is_active and 
                    service.consecutive_failures >= self.failure_threshold):
                    await self._trigger_failover(service, FailoverTrigger.HEALTH_CHECK_FAILURE)
        
        except Exception:
            service.consecutive_failures += 1
            service.consecutive_successes = 0
            
            if (service.is_active and 
                service.consecutive_failures >= self.failure_threshold):
                await self._trigger_failover(service, FailoverTrigger.HEALTH_CHECK_FAILURE)
    
    async def _check_service_performance(self, service: ServiceInstance):
        """检查服务性能"""
        # 检查响应时间
        if (service.is_active and 
            service.average_response_time > self.response_time_threshold):
            await self._trigger_failover(service, FailoverTrigger.RESPONSE_TIME_THRESHOLD)
        
        # 检查错误率
        if (service.is_active and 
            service.error_rate > self.error_rate_threshold):
            await self._trigger_failover(service, FailoverTrigger.ERROR_RATE_THRESHOLD)
    
    async def _check_service_resources(self, service: ServiceInstance):
        """检查服务资源"""
        # 检查资源使用情况
        if (service.is_active and 
            (service.cpu_usage > 90 or service.memory_usage > 90 or service.disk_usage > 90)):
            await self._trigger_failover(service, FailoverTrigger.RESOURCE_EXHAUSTION)
    
    async def _trigger_failover(self, failed_service: ServiceInstance, trigger: FailoverTrigger):
        """触发故障转移"""
        # 记录故障转移开始事件
        record = FailoverRecord(
            id=f"failover_{int(time.time() * 1000)}",
            event=FailoverEvent.FAILOVER_STARTED,
            trigger=trigger,
            from_service=failed_service.id,
            to_service=None
        )
        self.records.append(record)
        await self._emit_event(FailoverEvent.FAILOVER_STARTED, record)
        
        try:
            # 选择目标服务
            target_service = await self._select_target_service(failed_service)
            if not target_service:
                record.success = False
                record.error_message = "No suitable target service found"
                await self._emit_event(FailoverEvent.FAILOVER_FAILED, record)
                return
            
            record.to_service = target_service.id
            
            # 执行故障转移
            success = await self._execute_failover(failed_service, target_service)
            
            if success:
                record.duration = time.time() - record.timestamp
                record.event = FailoverEvent.FAILOVER_COMPLETED
                await self._emit_event(FailoverEvent.FAILOVER_COMPLETED, record)
            else:
                record.success = False
                record.error_message = "Failover execution failed"
                await self._emit_event(FailoverEvent.FAILOVER_FAILED, record)
        
        except Exception as e:
            record.success = False
            record.error_message = str(e)
            await self._emit_event(FailoverEvent.FAILOVER_FAILED, record)
    
    async def _trigger_recovery(self, service: ServiceInstance):
        """触发服务恢复"""
        # 记录恢复开始事件
        record = FailoverRecord(
            id=f"recovery_{int(time.time() * 1000)}",
            event=FailoverEvent.RECOVERY_STARTED,
            trigger=FailoverTrigger.HEALTH_CHECK_FAILURE,
            from_service=None,
            to_service=service.id
        )
        self.records.append(record)
        await self._emit_event(FailoverEvent.RECOVERY_STARTED, record)
        
        try:
            # 执行恢复
            success = await self._execute_recovery(service)
            
            if success:
                record.duration = time.time() - record.timestamp
                record.event = FailoverEvent.RECOVERY_COMPLETED
                await self._emit_event(FailoverEvent.RECOVERY_COMPLETED, record)
            else:
                record.success = False
                record.error_message = "Recovery execution failed"
                await self._emit_event(FailoverEvent.RECOVERY_FAILED, record)
        
        except Exception as e:
            record.success = False
            record.error_message = str(e)
            await self._emit_event(FailoverEvent.RECOVERY_FAILED, record)
    
    async def _select_target_service(self, failed_service: ServiceInstance) -> Optional[ServiceInstance]:
        """选择目标服务"""
        standby_services = self.get_standby_services()
        if not standby_services:
            return None
        
        if self.strategy == FailoverStrategy.PRIORITY_BASED:
            # 按优先级选择
            return min(standby_services, key=lambda s: s.priority)
        elif self.strategy == FailoverStrategy.GEOGRAPHIC:
            # 按地理位置选择（同区域优先）
            same_region = [s for s in standby_services if s.region == failed_service.region]
            if same_region:
                return min(same_region, key=lambda s: s.priority)
            return min(standby_services, key=lambda s: s.priority)
        elif self.strategy == FailoverStrategy.LOAD_BASED:
            # 按负载选择（CPU使用率最低）
            return min(standby_services, key=lambda s: s.cpu_usage)
        else:
            # 默认按优先级选择
            return min(standby_services, key=lambda s: s.priority)
    
    async def _execute_failover(self, from_service: ServiceInstance, 
                               to_service: ServiceInstance) -> bool:
        """执行故障转移"""
        for handler in self.handlers:
            try:
                if await handler.can_handle_failover(from_service, to_service):
                    return await handler.execute_failover(from_service, to_service)
            except Exception as e:
                print(f"Failover handler error: {e}")
        
        return False
    
    async def _execute_recovery(self, service: ServiceInstance) -> bool:
        """执行服务恢复"""
        for handler in self.handlers:
            try:
                return await handler.execute_recovery(service)
            except Exception as e:
                print(f"Recovery handler error: {e}")
        
        return False
    
    async def _emit_event(self, event: FailoverEvent, record: FailoverRecord):
        """发送事件"""
        for callback in self.event_callbacks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(record)
                else:
                    callback(record)
            except Exception as e:
                print(f"Event callback error: {e}")
    
    async def manual_failover(self, from_service_id: str, to_service_id: str) -> bool:
        """手动故障转移"""
        from_service = self.get_service(from_service_id)
        to_service = self.get_service(to_service_id)
        
        if not from_service or not to_service:
            return False
        
        await self._trigger_failover(from_service, FailoverTrigger.MANUAL_TRIGGER)
        return True
    
    async def manual_recovery(self, service_id: str) -> bool:
        """手动服务恢复"""
        service = self.get_service(service_id)
        if not service:
            return False
        
        await self._trigger_recovery(service)
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        services = list(self.services.values())
        active_services = self.get_active_services()
        standby_services = self.get_standby_services()
        failed_services = self.get_failed_services()
        
        recent_records = [r for r in self.records if time.time() - r.timestamp < 3600]  # 最近1小时
        
        return {
            "total_services": len(services),
            "active_services": len(active_services),
            "standby_services": len(standby_services),
            "failed_services": len(failed_services),
            "strategy": self.strategy.value,
            "recent_failovers": len([r for r in recent_records if r.event == FailoverEvent.FAILOVER_COMPLETED]),
            "recent_recoveries": len([r for r in recent_records if r.event == FailoverEvent.RECOVERY_COMPLETED]),
            "total_records": len(self.records),
            "services": [s.to_dict() for s in services]
        }
    
    def get_records(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取故障转移记录"""
        return [r.to_dict() for r in self.records[-limit:]]


class FailoverConfig(BaseSettings):
    """故障转移配置"""
    strategy: FailoverStrategy = FailoverStrategy.ACTIVE_PASSIVE
    
    # 监控配置
    health_check_interval: int = 30  # seconds
    failure_threshold: int = 3
    recovery_threshold: int = 2
    
    # 性能阈值
    response_time_threshold: float = 5.0  # seconds
    error_rate_threshold: float = 0.1  # 10%
    
    # 资源阈值
    cpu_threshold: float = 90.0  # %
    memory_threshold: float = 90.0  # %
    disk_threshold: float = 90.0  # %
    
    # 通知配置
    enable_notifications: bool = True
    notification_channels: List[str] = []
    
    class Config:
        env_prefix = "FAILOVER_"


# 全局故障转移管理器
_failover_manager: Optional[FailoverManager] = None


def initialize_failover(config: FailoverConfig) -> FailoverManager:
    """初始化故障转移管理器"""
    global _failover_manager
    _failover_manager = FailoverManager(
        strategy=config.strategy,
        health_check_interval=config.health_check_interval,
        failure_threshold=config.failure_threshold,
        recovery_threshold=config.recovery_threshold,
        response_time_threshold=config.response_time_threshold,
        error_rate_threshold=config.error_rate_threshold
    )
    return _failover_manager


def get_failover_manager() -> Optional[FailoverManager]:
    """获取全局故障转移管理器"""
    return _failover_manager


# 便捷函数
def add_service(service: ServiceInstance):
    """添加服务的便捷函数"""
    manager = get_failover_manager()
    if manager:
        manager.add_service(service)


def remove_service(service_id: str):
    """移除服务的便捷函数"""
    manager = get_failover_manager()
    if manager:
        manager.remove_service(service_id)


async def manual_failover(from_service_id: str, to_service_id: str) -> bool:
    """手动故障转移的便捷函数"""
    manager = get_failover_manager()
    if manager:
        return await manager.manual_failover(from_service_id, to_service_id)
    return False


async def manual_recovery(service_id: str) -> bool:
    """手动服务恢复的便捷函数"""
    manager = get_failover_manager()
    if manager:
        return await manager.manual_recovery(service_id)
    return False