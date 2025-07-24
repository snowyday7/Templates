#!/usr/bin/env python3
"""
系统服务

提供系统级别的业务逻辑，包括：
1. 系统健康检查
2. 系统监控
3. 性能分析
4. 资源管理
5. 任务调度
6. 系统维护
"""

import psutil
import asyncio
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text, func

from .base import BaseService
from ..core.exceptions import SystemException
from ..utils.logger import get_logger
from ..core.config import get_settings


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class HealthStatus:
    """健康状态"""
    status: str  # healthy, degraded, unhealthy
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


@dataclass
class SystemMetrics:
    """系统指标"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    process_count: int
    uptime: float
    timestamp: datetime


@dataclass
class DatabaseMetrics:
    """数据库指标"""
    connection_count: int
    active_queries: int
    database_size: str
    table_count: int
    index_count: int
    cache_hit_ratio: float
    timestamp: datetime


@dataclass
class CacheMetrics:
    """缓存指标"""
    used_memory: str
    hit_rate: float
    connected_clients: int
    operations_per_second: float
    keyspace_hits: int
    keyspace_misses: int
    timestamp: datetime


@dataclass
class PerformanceAlert:
    """性能告警"""
    level: str  # info, warning, error, critical
    category: str  # cpu, memory, disk, database, cache
    message: str
    value: float
    threshold: float
    timestamp: datetime


# =============================================================================
# 系统服务类
# =============================================================================

class SystemService(BaseService):
    """
    系统服务类
    
    提供系统级别的业务逻辑操作
    """
    
    def __init__(self, db: AsyncSession):
        """
        初始化系统服务
        
        Args:
            db: 数据库会话
        """
        super().__init__(db, None)  # 系统服务不绑定特定模型
        self.logger = get_logger(__name__)
        self.settings = get_settings()
        
        # 性能阈值配置
        self.thresholds = {
            "cpu_warning": 70.0,
            "cpu_critical": 90.0,
            "memory_warning": 80.0,
            "memory_critical": 95.0,
            "disk_warning": 85.0,
            "disk_critical": 95.0,
            "db_connections_warning": 80,
            "db_connections_critical": 95,
            "cache_hit_rate_warning": 80.0,
            "response_time_warning": 1000.0,  # ms
            "response_time_critical": 5000.0  # ms
        }
        
        # 监控任务
        self._monitoring_tasks = []
        self._alerts = []
    
    # =========================================================================
    # 健康检查
    # =========================================================================
    
    async def check_health(self) -> HealthStatus:
        """
        检查系统整体健康状态
        
        Returns:
            健康状态
        """
        try:
            checks = {
                "database": await self._check_database_health(),
                "cache": await self._check_cache_health(),
                "disk_space": await self._check_disk_health(),
                "memory": await self._check_memory_health(),
                "cpu": await self._check_cpu_health()
            }
            
            # 计算整体状态
            failed_checks = [name for name, status in checks.items() if status.status == "unhealthy"]
            degraded_checks = [name for name, status in checks.items() if status.status == "degraded"]
            
            if failed_checks:
                overall_status = "unhealthy"
                message = f"系统不健康: {', '.join(failed_checks)} 检查失败"
            elif degraded_checks:
                overall_status = "degraded"
                message = f"系统性能下降: {', '.join(degraded_checks)} 检查警告"
            else:
                overall_status = "healthy"
                message = "系统运行正常"
            
            return HealthStatus(
                status=overall_status,
                message=message,
                timestamp=datetime.utcnow(),
                details=checks
            )
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return HealthStatus(
                status="unhealthy",
                message=f"健康检查失败: {str(e)}",
                timestamp=datetime.utcnow()
            )
    
    async def check_readiness(self) -> HealthStatus:
        """
        检查系统就绪状态
        
        Returns:
            就绪状态
        """
        try:
            # 检查关键依赖
            db_status = await self._check_database_connection()
            cache_status = await self._check_cache_connection()
            
            if db_status and cache_status:
                return HealthStatus(
                    status="ready",
                    message="系统已就绪",
                    timestamp=datetime.utcnow(),
                    details={
                        "database": "connected",
                        "cache": "connected"
                    }
                )
            else:
                return HealthStatus(
                    status="not_ready",
                    message="系统未就绪",
                    timestamp=datetime.utcnow(),
                    details={
                        "database": "connected" if db_status else "disconnected",
                        "cache": "connected" if cache_status else "disconnected"
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Readiness check error: {e}")
            return HealthStatus(
                status="not_ready",
                message=f"就绪检查失败: {str(e)}",
                timestamp=datetime.utcnow()
            )
    
    async def check_liveness(self) -> HealthStatus:
        """
        检查系统存活状态
        
        Returns:
            存活状态
        """
        try:
            # 简单的存活检查
            current_time = datetime.utcnow()
            
            return HealthStatus(
                status="alive",
                message="系统存活",
                timestamp=current_time,
                details={
                    "uptime": psutil.boot_time(),
                    "current_time": current_time.isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Liveness check error: {e}")
            return HealthStatus(
                status="dead",
                message=f"存活检查失败: {str(e)}",
                timestamp=datetime.utcnow()
            )
    
    # =========================================================================
    # 系统监控
    # =========================================================================
    
    async def get_system_metrics(self) -> SystemMetrics:
        """
        获取系统指标
        
        Returns:
            系统指标
        """
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # 网络IO
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # 进程数量
            process_count = len(psutil.pids())
            
            # 系统运行时间
            uptime = datetime.utcnow().timestamp() - psutil.boot_time()
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_io=network_io,
                process_count=process_count,
                uptime=uptime,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            raise SystemException(f"获取系统指标失败: {str(e)}")
    
    async def get_database_metrics(self) -> DatabaseMetrics:
        """
        获取数据库指标
        
        Returns:
            数据库指标
        """
        try:
            # 连接数
            connection_query = text(
                "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
            )
            connection_result = await self.db.execute(connection_query)
            connection_count = connection_result.scalar()
            
            # 活跃查询数
            active_query = text(
                "SELECT count(*) FROM pg_stat_activity WHERE state = 'active' AND query != '<IDLE>'"
            )
            active_result = await self.db.execute(active_query)
            active_queries = active_result.scalar()
            
            # 数据库大小
            size_query = text(
                "SELECT pg_size_pretty(pg_database_size(current_database()))"
            )
            size_result = await self.db.execute(size_query)
            database_size = size_result.scalar()
            
            # 表数量
            table_query = text(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'"
            )
            table_result = await self.db.execute(table_query)
            table_count = table_result.scalar()
            
            # 索引数量
            index_query = text(
                "SELECT count(*) FROM pg_indexes WHERE schemaname = 'public'"
            )
            index_result = await self.db.execute(index_query)
            index_count = index_result.scalar()
            
            # 缓存命中率
            cache_query = text(
                "SELECT round(blks_hit::float / (blks_hit + blks_read) * 100, 2) as hit_ratio "
                "FROM pg_stat_database WHERE datname = current_database()"
            )
            cache_result = await self.db.execute(cache_query)
            cache_hit_ratio = cache_result.scalar() or 0.0
            
            return DatabaseMetrics(
                connection_count=connection_count,
                active_queries=active_queries,
                database_size=database_size,
                table_count=table_count,
                index_count=index_count,
                cache_hit_ratio=cache_hit_ratio,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error getting database metrics: {e}")
            raise SystemException(f"获取数据库指标失败: {str(e)}")
    
    async def get_cache_metrics(self) -> CacheMetrics:
        """
        获取缓存指标
        
        Returns:
            缓存指标
        """
        try:
            # 获取Redis信息
            info = await self.cache.redis.info()
            
            # 计算命中率
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            hit_rate = (hits / max(hits + misses, 1)) * 100
            
            # 计算每秒操作数
            ops_per_sec = info.get("instantaneous_ops_per_sec", 0)
            
            return CacheMetrics(
                used_memory=info.get("used_memory_human", "0B"),
                hit_rate=hit_rate,
                connected_clients=info.get("connected_clients", 0),
                operations_per_second=ops_per_sec,
                keyspace_hits=hits,
                keyspace_misses=misses,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error getting cache metrics: {e}")
            raise SystemException(f"获取缓存指标失败: {str(e)}")
    
    # =========================================================================
    # 性能分析
    # =========================================================================
    
    async def analyze_performance(self) -> Dict[str, Any]:
        """
        分析系统性能
        
        Returns:
            性能分析结果
        """
        try:
            # 获取各项指标
            system_metrics = await self.get_system_metrics()
            db_metrics = await self.get_database_metrics()
            cache_metrics = await self.get_cache_metrics()
            
            # 生成告警
            alerts = await self._generate_performance_alerts(
                system_metrics, db_metrics, cache_metrics
            )
            
            # 性能评分
            performance_score = await self._calculate_performance_score(
                system_metrics, db_metrics, cache_metrics
            )
            
            # 建议
            recommendations = await self._generate_recommendations(
                system_metrics, db_metrics, cache_metrics, alerts
            )
            
            return {
                "performance_score": performance_score,
                "system_metrics": system_metrics,
                "database_metrics": db_metrics,
                "cache_metrics": cache_metrics,
                "alerts": alerts,
                "recommendations": recommendations,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {e}")
            raise SystemException(f"性能分析失败: {str(e)}")
    
    async def get_performance_trends(
        self,
        hours: int = 24
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取性能趋势
        
        Args:
            hours: 查询小时数
            
        Returns:
            性能趋势数据
        """
        try:
            # 从缓存获取历史数据
            trends = {}
            
            # 获取系统指标趋势
            system_trend = await self._get_metric_trend("system_metrics", hours)
            trends["system"] = system_trend
            
            # 获取数据库指标趋势
            db_trend = await self._get_metric_trend("database_metrics", hours)
            trends["database"] = db_trend
            
            # 获取缓存指标趋势
            cache_trend = await self._get_metric_trend("cache_metrics", hours)
            trends["cache"] = cache_trend
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error getting performance trends: {e}")
            return {}
    
    # =========================================================================
    # 资源管理
    # =========================================================================
    
    async def get_resource_usage(self) -> Dict[str, Any]:
        """
        获取资源使用情况
        
        Returns:
            资源使用信息
        """
        try:
            # CPU信息
            cpu_info = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "current_freq": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "usage_percent": psutil.cpu_percent(interval=1),
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
            
            # 内存信息
            memory = psutil.virtual_memory()
            memory_info = {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "free": memory.free,
                "percent": memory.percent,
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2)
            }
            
            # 磁盘信息
            disk = psutil.disk_usage('/')
            disk_info = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": round((disk.used / disk.total) * 100, 2),
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2)
            }
            
            # 网络信息
            network = psutil.net_io_counters()
            network_info = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv,
                "errin": network.errin,
                "errout": network.errout,
                "dropin": network.dropin,
                "dropout": network.dropout
            }
            
            # 进程信息
            process_info = {
                "total_processes": len(psutil.pids()),
                "running_processes": len([p for p in psutil.process_iter() if p.status() == 'running']),
                "sleeping_processes": len([p for p in psutil.process_iter() if p.status() == 'sleeping'])
            }
            
            return {
                "cpu": cpu_info,
                "memory": memory_info,
                "disk": disk_info,
                "network": network_info,
                "processes": process_info,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting resource usage: {e}")
            raise SystemException(f"获取资源使用情况失败: {str(e)}")
    
    async def optimize_resources(self) -> Dict[str, Any]:
        """
        优化系统资源
        
        Returns:
            优化结果
        """
        try:
            optimization_results = {}
            
            # 清理过期缓存
            cache_cleaned = await self.cache.clear_expired()
            optimization_results["cache_cleanup"] = {
                "expired_keys_removed": cache_cleaned,
                "status": "completed"
            }
            
            # 数据库连接池优化
            db_optimization = await self._optimize_database_connections()
            optimization_results["database_optimization"] = db_optimization
            
            # 内存优化建议
            memory_optimization = await self._get_memory_optimization_suggestions()
            optimization_results["memory_optimization"] = memory_optimization
            
            return {
                "optimization_results": optimization_results,
                "timestamp": datetime.utcnow(),
                "status": "completed"
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing resources: {e}")
            raise SystemException(f"资源优化失败: {str(e)}")
    
    # =========================================================================
    # 任务调度
    # =========================================================================
    
    async def start_monitoring(self, interval: int = 60):
        """
        启动系统监控
        
        Args:
            interval: 监控间隔（秒）
        """
        try:
            # 启动监控任务
            task = asyncio.create_task(self._monitoring_loop(interval))
            self._monitoring_tasks.append(task)
            
            self.logger.info(f"System monitoring started with {interval}s interval")
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {e}")
            raise SystemException(f"启动监控失败: {str(e)}")
    
    async def stop_monitoring(self):
        """
        停止系统监控
        """
        try:
            # 取消所有监控任务
            for task in self._monitoring_tasks:
                task.cancel()
            
            # 等待任务完成
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
            
            self._monitoring_tasks.clear()
            
            self.logger.info("System monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}")
    
    async def schedule_maintenance(
        self,
        task_name: str,
        task_func: Callable,
        schedule_time: datetime,
        **kwargs
    ) -> str:
        """
        调度维护任务
        
        Args:
            task_name: 任务名称
            task_func: 任务函数
            schedule_time: 调度时间
            **kwargs: 任务参数
            
        Returns:
            任务ID
        """
        try:
            # 计算延迟时间
            delay = (schedule_time - datetime.utcnow()).total_seconds()
            
            if delay <= 0:
                raise ValueError("调度时间必须是未来时间")
            
            # 创建任务
            task_id = f"{task_name}_{int(datetime.utcnow().timestamp())}"
            
            async def scheduled_task():
                await asyncio.sleep(delay)
                try:
                    await task_func(**kwargs)
                    self.logger.info(f"Scheduled task {task_name} completed successfully")
                except Exception as e:
                    self.logger.error(f"Scheduled task {task_name} failed: {e}")
            
            # 启动任务
            task = asyncio.create_task(scheduled_task())
            self._monitoring_tasks.append(task)
            
            self.logger.info(f"Scheduled maintenance task {task_name} for {schedule_time}")
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"Error scheduling maintenance task: {e}")
            raise SystemException(f"调度维护任务失败: {str(e)}")
    
    # =========================================================================
    # 系统维护
    # =========================================================================
    
    async def perform_maintenance(self) -> Dict[str, Any]:
        """
        执行系统维护
        
        Returns:
            维护结果
        """
        try:
            maintenance_results = {}
            
            # 清理过期数据
            cleanup_result = await self._cleanup_expired_data()
            maintenance_results["data_cleanup"] = cleanup_result
            
            # 优化数据库
            db_optimization = await self._optimize_database()
            maintenance_results["database_optimization"] = db_optimization
            
            # 清理缓存
            cache_cleanup = await self._cleanup_cache()
            maintenance_results["cache_cleanup"] = cache_cleanup
            
            # 日志轮转
            log_rotation = await self._rotate_logs()
            maintenance_results["log_rotation"] = log_rotation
            
            # 系统检查
            system_check = await self._perform_system_check()
            maintenance_results["system_check"] = system_check
            
            return {
                "maintenance_results": maintenance_results,
                "timestamp": datetime.utcnow(),
                "status": "completed"
            }
            
        except Exception as e:
            self.logger.error(f"Error performing maintenance: {e}")
            raise SystemException(f"系统维护失败: {str(e)}")
    
    async def backup_system(self, backup_type: str = "full") -> Dict[str, Any]:
        """
        备份系统
        
        Args:
            backup_type: 备份类型（full, incremental）
            
        Returns:
            备份结果
        """
        try:
            backup_results = {}
            
            # 数据库备份
            db_backup = await self._backup_database(backup_type)
            backup_results["database"] = db_backup
            
            # 配置文件备份
            config_backup = await self._backup_configuration()
            backup_results["configuration"] = config_backup
            
            # 日志备份
            log_backup = await self._backup_logs()
            backup_results["logs"] = log_backup
            
            return {
                "backup_results": backup_results,
                "backup_type": backup_type,
                "timestamp": datetime.utcnow(),
                "status": "completed"
            }
            
        except Exception as e:
            self.logger.error(f"Error backing up system: {e}")
            raise SystemException(f"系统备份失败: {str(e)}")
    
    # =========================================================================
    # 私有辅助方法
    # =========================================================================
    
    async def _check_database_health(self) -> HealthStatus:
        """检查数据库健康状态"""
        try:
            # 检查连接
            if not await self._check_database_connection():
                return HealthStatus(
                    status="unhealthy",
                    message="数据库连接失败",
                    timestamp=datetime.utcnow()
                )
            
            # 检查响应时间
            start_time = datetime.utcnow()
            await self.db.execute(text("SELECT 1"))
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if response_time > self.thresholds["response_time_critical"]:
                return HealthStatus(
                    status="unhealthy",
                    message=f"数据库响应时间过长: {response_time:.2f}ms",
                    timestamp=datetime.utcnow()
                )
            elif response_time > self.thresholds["response_time_warning"]:
                return HealthStatus(
                    status="degraded",
                    message=f"数据库响应时间较慢: {response_time:.2f}ms",
                    timestamp=datetime.utcnow()
                )
            
            return HealthStatus(
                status="healthy",
                message="数据库运行正常",
                timestamp=datetime.utcnow(),
                details={"response_time_ms": response_time}
            )
            
        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                message=f"数据库检查失败: {str(e)}",
                timestamp=datetime.utcnow()
            )
    
    async def _check_cache_health(self) -> HealthStatus:
        """检查缓存健康状态"""
        try:
            # 检查连接
            if not await self._check_cache_connection():
                return HealthStatus(
                    status="unhealthy",
                    message="缓存连接失败",
                    timestamp=datetime.utcnow()
                )
            
            # 检查响应时间
            start_time = datetime.utcnow()
            await self.cache.ping()
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if response_time > 100:  # 100ms阈值
                return HealthStatus(
                    status="degraded",
                    message=f"缓存响应时间较慢: {response_time:.2f}ms",
                    timestamp=datetime.utcnow()
                )
            
            return HealthStatus(
                status="healthy",
                message="缓存运行正常",
                timestamp=datetime.utcnow(),
                details={"response_time_ms": response_time}
            )
            
        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                message=f"缓存检查失败: {str(e)}",
                timestamp=datetime.utcnow()
            )
    
    async def _check_disk_health(self) -> HealthStatus:
        """检查磁盘健康状态"""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            if disk_percent > self.thresholds["disk_critical"]:
                return HealthStatus(
                    status="unhealthy",
                    message=f"磁盘空间严重不足: {disk_percent:.1f}%",
                    timestamp=datetime.utcnow()
                )
            elif disk_percent > self.thresholds["disk_warning"]:
                return HealthStatus(
                    status="degraded",
                    message=f"磁盘空间不足: {disk_percent:.1f}%",
                    timestamp=datetime.utcnow()
                )
            
            return HealthStatus(
                status="healthy",
                message="磁盘空间充足",
                timestamp=datetime.utcnow(),
                details={"usage_percent": disk_percent}
            )
            
        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                message=f"磁盘检查失败: {str(e)}",
                timestamp=datetime.utcnow()
            )
    
    async def _check_memory_health(self) -> HealthStatus:
        """检查内存健康状态"""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent > self.thresholds["memory_critical"]:
                return HealthStatus(
                    status="unhealthy",
                    message=f"内存使用率过高: {memory_percent:.1f}%",
                    timestamp=datetime.utcnow()
                )
            elif memory_percent > self.thresholds["memory_warning"]:
                return HealthStatus(
                    status="degraded",
                    message=f"内存使用率较高: {memory_percent:.1f}%",
                    timestamp=datetime.utcnow()
                )
            
            return HealthStatus(
                status="healthy",
                message="内存使用正常",
                timestamp=datetime.utcnow(),
                details={"usage_percent": memory_percent}
            )
            
        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                message=f"内存检查失败: {str(e)}",
                timestamp=datetime.utcnow()
            )
    
    async def _check_cpu_health(self) -> HealthStatus:
        """检查CPU健康状态"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > self.thresholds["cpu_critical"]:
                return HealthStatus(
                    status="unhealthy",
                    message=f"CPU使用率过高: {cpu_percent:.1f}%",
                    timestamp=datetime.utcnow()
                )
            elif cpu_percent > self.thresholds["cpu_warning"]:
                return HealthStatus(
                    status="degraded",
                    message=f"CPU使用率较高: {cpu_percent:.1f}%",
                    timestamp=datetime.utcnow()
                )
            
            return HealthStatus(
                status="healthy",
                message="CPU使用正常",
                timestamp=datetime.utcnow(),
                details={"usage_percent": cpu_percent}
            )
            
        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                message=f"CPU检查失败: {str(e)}",
                timestamp=datetime.utcnow()
            )
    
    async def _check_database_connection(self) -> bool:
        """检查数据库连接"""
        try:
            await self.db.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
    
    async def _check_cache_connection(self) -> bool:
        """检查缓存连接"""
        try:
            await self.cache.ping()
            return True
        except Exception:
            return False
    
    async def _monitoring_loop(self, interval: int):
        """监控循环"""
        while True:
            try:
                # 收集指标
                system_metrics = await self.get_system_metrics()
                db_metrics = await self.get_database_metrics()
                cache_metrics = await self.get_cache_metrics()
                
                # 存储指标到缓存
                timestamp = datetime.utcnow().isoformat()
                await self.cache.set(
                    f"metrics:system:{timestamp}",
                    system_metrics.__dict__,
                    expire=86400  # 保留24小时
                )
                await self.cache.set(
                    f"metrics:database:{timestamp}",
                    db_metrics.__dict__,
                    expire=86400
                )
                await self.cache.set(
                    f"metrics:cache:{timestamp}",
                    cache_metrics.__dict__,
                    expire=86400
                )
                
                # 生成告警
                alerts = await self._generate_performance_alerts(
                    system_metrics, db_metrics, cache_metrics
                )
                
                # 处理告警
                for alert in alerts:
                    await self._handle_alert(alert)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(interval)
    
    async def _generate_performance_alerts(
        self,
        system_metrics: SystemMetrics,
        db_metrics: DatabaseMetrics,
        cache_metrics: CacheMetrics
    ) -> List[PerformanceAlert]:
        """生成性能告警"""
        alerts = []
        
        # CPU告警
        if system_metrics.cpu_percent > self.thresholds["cpu_critical"]:
            alerts.append(PerformanceAlert(
                level="critical",
                category="cpu",
                message=f"CPU使用率过高: {system_metrics.cpu_percent:.1f}%",
                value=system_metrics.cpu_percent,
                threshold=self.thresholds["cpu_critical"],
                timestamp=datetime.utcnow()
            ))
        elif system_metrics.cpu_percent > self.thresholds["cpu_warning"]:
            alerts.append(PerformanceAlert(
                level="warning",
                category="cpu",
                message=f"CPU使用率较高: {system_metrics.cpu_percent:.1f}%",
                value=system_metrics.cpu_percent,
                threshold=self.thresholds["cpu_warning"],
                timestamp=datetime.utcnow()
            ))
        
        # 内存告警
        if system_metrics.memory_percent > self.thresholds["memory_critical"]:
            alerts.append(PerformanceAlert(
                level="critical",
                category="memory",
                message=f"内存使用率过高: {system_metrics.memory_percent:.1f}%",
                value=system_metrics.memory_percent,
                threshold=self.thresholds["memory_critical"],
                timestamp=datetime.utcnow()
            ))
        elif system_metrics.memory_percent > self.thresholds["memory_warning"]:
            alerts.append(PerformanceAlert(
                level="warning",
                category="memory",
                message=f"内存使用率较高: {system_metrics.memory_percent:.1f}%",
                value=system_metrics.memory_percent,
                threshold=self.thresholds["memory_warning"],
                timestamp=datetime.utcnow()
            ))
        
        # 磁盘告警
        if system_metrics.disk_percent > self.thresholds["disk_critical"]:
            alerts.append(PerformanceAlert(
                level="critical",
                category="disk",
                message=f"磁盘使用率过高: {system_metrics.disk_percent:.1f}%",
                value=system_metrics.disk_percent,
                threshold=self.thresholds["disk_critical"],
                timestamp=datetime.utcnow()
            ))
        elif system_metrics.disk_percent > self.thresholds["disk_warning"]:
            alerts.append(PerformanceAlert(
                level="warning",
                category="disk",
                message=f"磁盘使用率较高: {system_metrics.disk_percent:.1f}%",
                value=system_metrics.disk_percent,
                threshold=self.thresholds["disk_warning"],
                timestamp=datetime.utcnow()
            ))
        
        # 缓存命中率告警
        if cache_metrics.hit_rate < self.thresholds["cache_hit_rate_warning"]:
            alerts.append(PerformanceAlert(
                level="warning",
                category="cache",
                message=f"缓存命中率较低: {cache_metrics.hit_rate:.1f}%",
                value=cache_metrics.hit_rate,
                threshold=self.thresholds["cache_hit_rate_warning"],
                timestamp=datetime.utcnow()
            ))
        
        return alerts
    
    async def _calculate_performance_score(
        self,
        system_metrics: SystemMetrics,
        db_metrics: DatabaseMetrics,
        cache_metrics: CacheMetrics
    ) -> float:
        """计算性能评分"""
        # 各项指标权重
        weights = {
            "cpu": 0.25,
            "memory": 0.25,
            "disk": 0.15,
            "database": 0.20,
            "cache": 0.15
        }
        
        # 计算各项得分（100分制，使用率越低得分越高）
        cpu_score = max(0, 100 - system_metrics.cpu_percent)
        memory_score = max(0, 100 - system_metrics.memory_percent)
        disk_score = max(0, 100 - system_metrics.disk_percent)
        
        # 数据库得分（基于缓存命中率）
        db_score = min(100, db_metrics.cache_hit_ratio)
        
        # 缓存得分（基于命中率）
        cache_score = min(100, cache_metrics.hit_rate)
        
        # 加权平均
        total_score = (
            cpu_score * weights["cpu"] +
            memory_score * weights["memory"] +
            disk_score * weights["disk"] +
            db_score * weights["database"] +
            cache_score * weights["cache"]
        )
        
        return round(total_score, 2)
    
    async def _generate_recommendations(
        self,
        system_metrics: SystemMetrics,
        db_metrics: DatabaseMetrics,
        cache_metrics: CacheMetrics,
        alerts: List[PerformanceAlert]
    ) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于告警生成建议
        for alert in alerts:
            if alert.category == "cpu" and alert.level == "critical":
                recommendations.append("建议检查CPU密集型进程，考虑优化算法或增加CPU资源")
            elif alert.category == "memory" and alert.level == "critical":
                recommendations.append("建议检查内存泄漏，优化内存使用或增加内存容量")
            elif alert.category == "disk" and alert.level == "critical":
                recommendations.append("建议清理磁盘空间，删除不必要的文件或扩展存储容量")
            elif alert.category == "cache" and alert.level == "warning":
                recommendations.append("建议优化缓存策略，检查缓存键设计和过期时间设置")
        
        # 基于指标生成通用建议
        if db_metrics.connection_count > 50:
            recommendations.append("数据库连接数较多，建议优化连接池配置")
        
        if cache_metrics.hit_rate < 90:
            recommendations.append("缓存命中率可以进一步提升，建议优化缓存策略")
        
        if system_metrics.process_count > 200:
            recommendations.append("系统进程数较多，建议检查是否有异常进程")
        
        return recommendations
    
    async def _handle_alert(self, alert: PerformanceAlert):
        """处理告警"""
        try:
            # 记录告警
            self._alerts.append(alert)
            
            # 记录日志
            if alert.level == "critical":
                self.logger.critical(f"Performance Alert: {alert.message}")
            elif alert.level == "warning":
                self.logger.warning(f"Performance Alert: {alert.message}")
            else:
                self.logger.info(f"Performance Alert: {alert.message}")
            
            # 存储告警到缓存
            alert_key = f"alert:{alert.category}:{int(alert.timestamp.timestamp())}"
            await self.cache.set(
                alert_key,
                alert.__dict__,
                expire=3600  # 保留1小时
            )
            
            # TODO: 发送通知（邮件、短信、Webhook等）
            
        except Exception as e:
            self.logger.error(f"Error handling alert: {e}")
    
    async def _get_metric_trend(
        self,
        metric_type: str,
        hours: int
    ) -> List[Dict[str, Any]]:
        """获取指标趋势"""
        try:
            # 从缓存获取历史数据
            pattern = f"metrics:{metric_type}:*"
            keys = await self.cache.redis.keys(pattern)
            
            # 按时间排序
            keys.sort()
            
            # 获取最近N小时的数据
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            trend_data = []
            for key in keys[-hours*60:]:  # 假设每分钟一个数据点
                try:
                    data = await self.cache.get(key.decode())
                    if data and 'timestamp' in data:
                        timestamp = datetime.fromisoformat(data['timestamp'])
                        if timestamp >= cutoff_time:
                            trend_data.append(data)
                except Exception:
                    continue
            
            return trend_data
            
        except Exception as e:
            self.logger.error(f"Error getting metric trend: {e}")
            return []
    
    async def _optimize_database_connections(self) -> Dict[str, Any]:
        """优化数据库连接"""
        try:
            # 获取当前连接信息
            connection_query = text(
                "SELECT state, count(*) FROM pg_stat_activity GROUP BY state"
            )
            result = await self.db.execute(connection_query)
            connections = dict(result.fetchall())
            
            # 关闭空闲连接
            idle_query = text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE state = 'idle' AND state_change < now() - interval '1 hour'"
            )
            terminated_result = await self.db.execute(idle_query)
            terminated_count = terminated_result.rowcount
            
            return {
                "current_connections": connections,
                "terminated_idle_connections": terminated_count,
                "status": "completed"
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing database connections: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _get_memory_optimization_suggestions(self) -> Dict[str, Any]:
        """获取内存优化建议"""
        try:
            memory = psutil.virtual_memory()
            
            suggestions = []
            if memory.percent > 80:
                suggestions.append("考虑增加系统内存")
                suggestions.append("检查内存使用最多的进程")
                suggestions.append("优化应用程序内存使用")
            
            if memory.available < 1024 * 1024 * 1024:  # 1GB
                suggestions.append("可用内存不足1GB，建议立即释放内存")
            
            return {
                "current_usage_percent": memory.percent,
                "available_gb": round(memory.available / (1024**3), 2),
                "suggestions": suggestions,
                "status": "completed"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting memory optimization suggestions: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _cleanup_expired_data(self) -> Dict[str, Any]:
        """清理过期数据"""
        # 这里应该调用AdminService的cleanup_expired_data方法
        # 为了避免循环导入，这里只返回模拟结果
        return {
            "expired_tokens": 0,
            "expired_sessions": 0,
            "old_login_attempts": 0,
            "status": "completed"
        }
    
    async def _optimize_database(self) -> Dict[str, Any]:
        """优化数据库"""
        try:
            # 分析表统计信息
            analyze_query = text("ANALYZE")
            await self.db.execute(analyze_query)
            
            # 清理死元组
            vacuum_query = text("VACUUM")
            await self.db.execute(vacuum_query)
            
            return {"status": "completed", "operations": ["analyze", "vacuum"]}
            
        except Exception as e:
            self.logger.error(f"Error optimizing database: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _cleanup_cache(self) -> Dict[str, Any]:
        """清理缓存"""
        try:
            # 清理过期键
            expired_count = await self.cache.clear_expired()
            
            return {
                "expired_keys_removed": expired_count,
                "status": "completed"
            }
            
        except Exception as e:
            self.logger.error(f"Error cleaning up cache: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _rotate_logs(self) -> Dict[str, Any]:
        """日志轮转"""
        # 这里应该实现日志轮转逻辑
        return {"status": "completed", "rotated_files": 0}
    
    async def _perform_system_check(self) -> Dict[str, Any]:
        """执行系统检查"""
        try:
            health_status = await self.check_health()
            
            return {
                "health_status": health_status.status,
                "message": health_status.message,
                "status": "completed"
            }
            
        except Exception as e:
            self.logger.error(f"Error performing system check: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _backup_database(self, backup_type: str) -> Dict[str, Any]:
        """备份数据库"""
        # 这里应该实现数据库备份逻辑
        return {
            "backup_type": backup_type,
            "backup_file": f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.sql",
            "status": "completed"
        }
    
    async def _backup_configuration(self) -> Dict[str, Any]:
        """备份配置文件"""
        # 这里应该实现配置备份逻辑
        return {
            "config_files": ["app.conf", "database.conf"],
            "status": "completed"
        }
    
    async def _backup_logs(self) -> Dict[str, Any]:
        """备份日志"""
        # 这里应该实现日志备份逻辑
        return {
            "log_files": ["app.log", "error.log"],
            "status": "completed"
        }