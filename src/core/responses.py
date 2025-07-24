#!/usr/bin/env python3
"""
响应模型模块

提供统一的API响应格式，包括：
1. 标准响应模型
2. 分页响应模型
3. 错误响应模型
4. 文件响应模型
5. 批量操作响应模型
6. 统计响应模型
7. 响应构建器
8. 响应装饰器
"""

from typing import Any, Dict, List, Optional, Union, Generic, TypeVar
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field
from pydantic.generics import GenericModel
from fastapi import status
from fastapi.responses import JSONResponse


# 泛型类型变量
T = TypeVar('T')


# =============================================================================
# 响应状态枚举
# =============================================================================

class ResponseStatus(str, Enum):
    """响应状态枚举"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# =============================================================================
# 基础响应模型
# =============================================================================

class BaseResponse(GenericModel, Generic[T]):
    """基础响应模型"""
    
    success: bool = Field(description="请求是否成功")
    message: Optional[str] = Field(default=None, description="响应消息")
    data: Optional[T] = Field(default=None, description="响应数据")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {},
                "timestamp": "2023-01-01T00:00:00Z",
                "request_id": "req_123456789"
            }
        }


class SuccessResponse(BaseResponse[T]):
    """成功响应模型"""
    
    success: bool = Field(default=True, description="请求成功")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {},
                "timestamp": "2023-01-01T00:00:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """错误响应模型"""
    
    success: bool = Field(default=False, description="请求失败")
    error: Dict[str, Any] = Field(description="错误信息")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid input data",
                    "details": {}
                },
                "timestamp": "2023-01-01T00:00:00Z",
                "request_id": "req_123456789"
            }
        }


# =============================================================================
# 分页响应模型
# =============================================================================

class PaginationMeta(BaseModel):
    """分页元数据模型"""
    
    page: int = Field(description="当前页码")
    size: int = Field(description="每页大小")
    total: int = Field(description="总记录数")
    pages: int = Field(description="总页数")
    has_next: bool = Field(description="是否有下一页")
    has_prev: bool = Field(description="是否有上一页")
    
    @classmethod
    def create(cls, page: int, size: int, total: int) -> "PaginationMeta":
        """创建分页元数据"""
        pages = (total + size - 1) // size if total > 0 else 0
        return cls(
            page=page,
            size=size,
            total=total,
            pages=pages,
            has_next=page < pages,
            has_prev=page > 1
        )


class PaginatedResponse(GenericModel, Generic[T]):
    """分页响应模型"""
    
    success: bool = Field(default=True, description="请求是否成功")
    message: Optional[str] = Field(default=None, description="响应消息")
    data: List[T] = Field(description="数据列表")
    meta: PaginationMeta = Field(description="分页元数据")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Data retrieved successfully",
                "data": [],
                "meta": {
                    "page": 1,
                    "size": 20,
                    "total": 100,
                    "pages": 5,
                    "has_next": True,
                    "has_prev": False
                },
                "timestamp": "2023-01-01T00:00:00Z"
            }
        }


# =============================================================================
# 批量操作响应模型
# =============================================================================

class BatchOperationResult(BaseModel):
    """批量操作结果模型"""
    
    total: int = Field(description="总操作数")
    success: int = Field(description="成功数")
    failed: int = Field(description="失败数")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="错误列表")
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.success / self.total if self.total > 0 else 0.0


class BatchResponse(BaseModel):
    """批量操作响应模型"""
    
    success: bool = Field(description="批量操作是否成功")
    message: Optional[str] = Field(default=None, description="响应消息")
    result: BatchOperationResult = Field(description="批量操作结果")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Batch operation completed",
                "result": {
                    "total": 10,
                    "success": 8,
                    "failed": 2,
                    "errors": [
                        {
                            "index": 3,
                            "error": "Validation failed",
                            "details": {}
                        }
                    ]
                },
                "timestamp": "2023-01-01T00:00:00Z"
            }
        }


# =============================================================================
# 统计响应模型
# =============================================================================

class StatisticsData(BaseModel):
    """统计数据模型"""
    
    metrics: Dict[str, Union[int, float, str]] = Field(description="统计指标")
    charts: Optional[Dict[str, Any]] = Field(default=None, description="图表数据")
    summary: Optional[Dict[str, Any]] = Field(default=None, description="汇总信息")
    
    class Config:
        schema_extra = {
            "example": {
                "metrics": {
                    "total_users": 1000,
                    "active_users": 800,
                    "growth_rate": 15.5
                },
                "charts": {
                    "user_growth": {
                        "labels": ["Jan", "Feb", "Mar"],
                        "data": [100, 150, 200]
                    }
                },
                "summary": {
                    "period": "last_30_days",
                    "trend": "increasing"
                }
            }
        }


class StatisticsResponse(BaseModel):
    """统计响应模型"""
    
    success: bool = Field(default=True, description="请求是否成功")
    message: Optional[str] = Field(default=None, description="响应消息")
    data: StatisticsData = Field(description="统计数据")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")


# =============================================================================
# 文件响应模型
# =============================================================================

class FileInfo(BaseModel):
    """文件信息模型"""
    
    filename: str = Field(description="文件名")
    size: int = Field(description="文件大小（字节）")
    content_type: str = Field(description="文件类型")
    url: Optional[str] = Field(default=None, description="文件URL")
    checksum: Optional[str] = Field(default=None, description="文件校验和")
    upload_time: datetime = Field(default_factory=datetime.utcnow, description="上传时间")


class FileUploadResponse(BaseModel):
    """文件上传响应模型"""
    
    success: bool = Field(default=True, description="上传是否成功")
    message: Optional[str] = Field(default=None, description="响应消息")
    file: FileInfo = Field(description="文件信息")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")


class MultiFileUploadResponse(BaseModel):
    """多文件上传响应模型"""
    
    success: bool = Field(description="上传是否成功")
    message: Optional[str] = Field(default=None, description="响应消息")
    files: List[FileInfo] = Field(description="文件信息列表")
    result: BatchOperationResult = Field(description="批量上传结果")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")


# =============================================================================
# 健康检查响应模型
# =============================================================================

class HealthCheckStatus(str, Enum):
    """健康检查状态枚举"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class ServiceHealth(BaseModel):
    """服务健康状态模型"""
    
    name: str = Field(description="服务名称")
    status: HealthCheckStatus = Field(description="健康状态")
    response_time: Optional[float] = Field(default=None, description="响应时间（毫秒）")
    details: Optional[Dict[str, Any]] = Field(default=None, description="详细信息")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="最后检查时间")


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    
    status: HealthCheckStatus = Field(description="整体健康状态")
    services: List[ServiceHealth] = Field(description="服务健康状态列表")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="检查时间戳")
    uptime: Optional[float] = Field(default=None, description="运行时间（秒）")
    version: Optional[str] = Field(default=None, description="应用版本")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "services": [
                    {
                        "name": "database",
                        "status": "healthy",
                        "response_time": 5.2,
                        "last_check": "2023-01-01T00:00:00Z"
                    },
                    {
                        "name": "cache",
                        "status": "healthy",
                        "response_time": 1.1,
                        "last_check": "2023-01-01T00:00:00Z"
                    }
                ],
                "timestamp": "2023-01-01T00:00:00Z",
                "uptime": 86400.0,
                "version": "1.0.0"
            }
        }


# =============================================================================
# 响应构建器
# =============================================================================

class ResponseBuilder:
    """响应构建器"""
    
    @staticmethod
    def success(
        data: Any = None,
        message: str = "Operation completed successfully",
        request_id: Optional[str] = None,
        status_code: int = status.HTTP_200_OK
    ) -> JSONResponse:
        """构建成功响应"""
        response_data = {
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if request_id:
            response_data["request_id"] = request_id
        
        return JSONResponse(
            content=response_data,
            status_code=status_code
        )
    
    @staticmethod
    def error(
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        status_code: int = status.HTTP_400_BAD_REQUEST
    ) -> JSONResponse:
        """构建错误响应"""
        error_data = {
            "code": error_code,
            "message": message
        }
        
        if details:
            error_data["details"] = details
        
        response_data = {
            "success": False,
            "error": error_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if request_id:
            response_data["request_id"] = request_id
        
        return JSONResponse(
            content=response_data,
            status_code=status_code
        )
    
    @staticmethod
    def paginated(
        data: List[Any],
        page: int,
        size: int,
        total: int,
        message: str = "Data retrieved successfully",
        request_id: Optional[str] = None
    ) -> JSONResponse:
        """构建分页响应"""
        meta = PaginationMeta.create(page, size, total)
        
        response_data = {
            "success": True,
            "message": message,
            "data": data,
            "meta": meta.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if request_id:
            response_data["request_id"] = request_id
        
        return JSONResponse(
            content=response_data,
            status_code=status.HTTP_200_OK
        )
    
    @staticmethod
    def batch(
        total: int,
        success_count: int,
        failed_count: int,
        errors: List[Dict[str, Any]] = None,
        message: str = "Batch operation completed",
        request_id: Optional[str] = None
    ) -> JSONResponse:
        """构建批量操作响应"""
        result = BatchOperationResult(
            total=total,
            success=success_count,
            failed=failed_count,
            errors=errors or []
        )
        
        response_data = {
            "success": failed_count == 0,
            "message": message,
            "result": result.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if request_id:
            response_data["request_id"] = request_id
        
        return JSONResponse(
            content=response_data,
            status_code=status.HTTP_200_OK
        )
    
    @staticmethod
    def created(
        data: Any = None,
        message: str = "Resource created successfully",
        request_id: Optional[str] = None
    ) -> JSONResponse:
        """构建创建成功响应"""
        return ResponseBuilder.success(
            data=data,
            message=message,
            request_id=request_id,
            status_code=status.HTTP_201_CREATED
        )
    
    @staticmethod
    def updated(
        data: Any = None,
        message: str = "Resource updated successfully",
        request_id: Optional[str] = None
    ) -> JSONResponse:
        """构建更新成功响应"""
        return ResponseBuilder.success(
            data=data,
            message=message,
            request_id=request_id,
            status_code=status.HTTP_200_OK
        )
    
    @staticmethod
    def deleted(
        message: str = "Resource deleted successfully",
        request_id: Optional[str] = None
    ) -> JSONResponse:
        """构建删除成功响应"""
        return ResponseBuilder.success(
            data=None,
            message=message,
            request_id=request_id,
            status_code=status.HTTP_200_OK
        )
    
    @staticmethod
    def no_content(
        request_id: Optional[str] = None
    ) -> JSONResponse:
        """构建无内容响应"""
        response_data = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if request_id:
            response_data["request_id"] = request_id
        
        return JSONResponse(
            content=response_data,
            status_code=status.HTTP_204_NO_CONTENT
        )


# =============================================================================
# 响应装饰器
# =============================================================================

def format_response(message: str = None):
    """格式化响应装饰器"""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # 如果已经是JSONResponse，直接返回
            if isinstance(result, JSONResponse):
                return result
            
            # 获取请求ID（如果存在）
            request_id = None
            for arg in args:
                if hasattr(arg, 'state') and hasattr(arg.state, 'request_id'):
                    request_id = arg.state.request_id
                    break
            
            # 构建响应
            return ResponseBuilder.success(
                data=result,
                message=message or "Operation completed successfully",
                request_id=request_id
            )
        
        return wrapper
    return decorator


# =============================================================================
# 便捷函数
# =============================================================================

def success_response(
    data: Any = None,
    message: str = "Operation completed successfully",
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """创建成功响应数据"""
    response = {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if request_id:
        response["request_id"] = request_id
    
    return response


def error_response(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """创建错误响应数据"""
    error_data = {
        "code": error_code,
        "message": message
    }
    
    if details:
        error_data["details"] = details
    
    response = {
        "success": False,
        "error": error_data,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if request_id:
        response["request_id"] = request_id
    
    return response


def paginated_response(
    data: List[Any],
    page: int,
    size: int,
    total: int,
    message: str = "Data retrieved successfully",
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """创建分页响应数据"""
    meta = PaginationMeta.create(page, size, total)
    
    response = {
        "success": True,
        "message": message,
        "data": data,
        "meta": meta.dict(),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if request_id:
        response["request_id"] = request_id
    
    return response