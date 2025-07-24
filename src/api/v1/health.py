#!/usr/bin/env python3
"""
健康检查API端点

提供系统健康状态检查的API端点，包括：
1. 基础健康检查
2. 详细健康检查
3. 就绪状态检查
4. 存活状态检查
5. 依赖服务检查
"""

from typing import Dict, Any, List
from datetime import datetime
import asyncio
import psutil

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ...core.config import get_settings
from ...core.database import DatabaseManager
from ...core.cache import CacheManager
from ...core.responses import ResponseBuilder


# =============================================================================
# 响应模型
# =============================================================================

class HealthStatus(BaseModel):
    """健康状态模型"""
    status: str
    timestamp: datetime
    version: str
    uptime: float
    

class DetailedHealthStatus(BaseModel):
    """详细健康状态模型"""
    status: str
    timestamp: datetime
    version: str
    uptime: float
    system: Dict[str, Any]
    database: Dict[str, Any]
    cache: Dict[str, Any]
    dependencies: Dict[str, Any]
    

class ServiceStatus(BaseModel):
    """服务状态模型"""
    name: str
    status: str
    response_time: float
    error: str = None
    details: Dict[str, Any] = None


# =============================================================================
# 路由器
# =============================================================================

router = APIRouter()

# 应用启动时间
START_TIME = datetime.utcnow()


# =============================================================================
# 辅助函数
# =============================================================================

def get_uptime() -> float:
    """获取应用运行时间（秒）"""
    return (datetime.utcnow() - START_TIME).total_seconds()


def get_system_info() -> Dict[str, Any]:
    """获取系统信息"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_usage": cpu_percent,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
                "free": memory.free
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            },
            "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
        }
    except Exception as e:
        return {"error": str(e)}


async def check_database() -> Dict[str, Any]:
    """检查数据库连接"""
    try:
        start_time = datetime.utcnow()
        db_manager = DatabaseManager()
        
        # 执行简单查询测试连接
        result = await db_manager.execute_raw_sql("SELECT 1 as test")
        
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "status": "healthy",
            "response_time_ms": response_time,
            "connection_pool": {
                "size": db_manager.engine.pool.size(),
                "checked_in": db_manager.engine.pool.checkedin(),
                "checked_out": db_manager.engine.pool.checkedout(),
                "overflow": db_manager.engine.pool.overflow(),
            } if hasattr(db_manager.engine, 'pool') else None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_cache() -> Dict[str, Any]:
    """检查缓存连接"""
    try:
        start_time = datetime.utcnow()
        cache_manager = CacheManager()
        
        # 测试缓存连接
        test_key = "health_check_test"
        test_value = "test_value"
        
        await cache_manager.set(test_key, test_value, ttl=10)
        retrieved_value = await cache_manager.get(test_key)
        await cache_manager.delete(test_key)
        
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        if retrieved_value != test_value:
            raise Exception("Cache test failed: value mismatch")
        
        # 获取Redis信息
        info = await cache_manager.get_info()
        
        return {
            "status": "healthy",
            "response_time_ms": response_time,
            "info": {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_dependencies() -> Dict[str, Any]:
    """检查外部依赖服务"""
    dependencies = {}
    
    # 检查数据库
    dependencies["database"] = await check_database()
    
    # 检查缓存
    dependencies["cache"] = await check_cache()
    
    # 可以添加更多依赖检查，如：
    # - 外部API
    # - 消息队列
    # - 文件存储
    # - 搜索引擎等
    
    return dependencies


# =============================================================================
# API端点
# =============================================================================

@router.get(
    "/",
    response_model=HealthStatus,
    summary="基础健康检查",
    description="返回应用的基础健康状态"
)
async def health_check():
    """基础健康检查"""
    settings = get_settings()
    
    return ResponseBuilder.success(
        data=HealthStatus(
            status="healthy",
            timestamp=datetime.utcnow(),
            version=settings.app_version,
            uptime=get_uptime()
        )
    )


@router.get(
    "/detailed",
    response_model=DetailedHealthStatus,
    summary="详细健康检查",
    description="返回应用的详细健康状态，包括系统信息和依赖服务状态"
)
async def detailed_health_check():
    """详细健康检查"""
    settings = get_settings()
    
    # 并发检查各个组件
    system_info_task = asyncio.create_task(asyncio.to_thread(get_system_info))
    dependencies_task = asyncio.create_task(check_dependencies())
    
    system_info = await system_info_task
    dependencies = await dependencies_task
    
    # 判断整体状态
    overall_status = "healthy"
    for dep_name, dep_status in dependencies.items():
        if dep_status.get("status") == "unhealthy":
            overall_status = "unhealthy"
            break
    
    return ResponseBuilder.success(
        data=DetailedHealthStatus(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version=settings.app_version,
            uptime=get_uptime(),
            system=system_info,
            database=dependencies.get("database", {}),
            cache=dependencies.get("cache", {}),
            dependencies=dependencies
        )
    )


@router.get(
    "/ready",
    summary="就绪状态检查",
    description="检查应用是否准备好接收请求"
)
async def readiness_check():
    """就绪状态检查"""
    try:
        # 检查关键依赖
        dependencies = await check_dependencies()
        
        # 检查数据库连接
        db_status = dependencies.get("database", {})
        if db_status.get("status") != "healthy":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database is not ready"
            )
        
        # 检查缓存连接
        cache_status = dependencies.get("cache", {})
        if cache_status.get("status") != "healthy":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cache is not ready"
            )
        
        return ResponseBuilder.success(
            message="Application is ready",
            data={
                "status": "ready",
                "timestamp": datetime.utcnow(),
                "dependencies": dependencies
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Application is not ready: {str(e)}"
        )


@router.get(
    "/live",
    summary="存活状态检查",
    description="检查应用是否存活（用于Kubernetes liveness probe）"
)
async def liveness_check():
    """存活状态检查"""
    return ResponseBuilder.success(
        message="Application is alive",
        data={
            "status": "alive",
            "timestamp": datetime.utcnow(),
            "uptime": get_uptime()
        }
    )


@router.get(
    "/dependencies",
    summary="依赖服务检查",
    description="检查所有外部依赖服务的状态"
)
async def dependencies_check():
    """依赖服务检查"""
    dependencies = await check_dependencies()
    
    # 计算整体状态
    overall_status = "healthy"
    unhealthy_services = []
    
    for service_name, service_status in dependencies.items():
        if service_status.get("status") == "unhealthy":
            overall_status = "degraded"
            unhealthy_services.append(service_name)
    
    return ResponseBuilder.success(
        data={
            "overall_status": overall_status,
            "timestamp": datetime.utcnow(),
            "services": dependencies,
            "unhealthy_services": unhealthy_services
        }
    )


@router.get(
    "/metrics",
    summary="应用指标",
    description="返回应用的性能指标"
)
async def metrics():
    """应用指标"""
    system_info = await asyncio.to_thread(get_system_info)
    
    return ResponseBuilder.success(
        data={
            "timestamp": datetime.utcnow(),
            "uptime": get_uptime(),
            "system": system_info,
            "application": {
                "version": get_settings().app_version,
                "environment": get_settings().environment,
                "debug": get_settings().debug
            }
        }
    )


@router.get(
    "/version",
    summary="版本信息",
    description="返回应用版本信息"
)
async def version_info():
    """版本信息"""
    settings = get_settings()
    
    return ResponseBuilder.success(
        data={
            "version": settings.app_version,
            "name": settings.app_name,
            "environment": settings.environment,
            "build_time": START_TIME.isoformat(),
            "uptime": get_uptime()
        }
    )