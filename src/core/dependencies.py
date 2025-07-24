#!/usr/bin/env python3
"""
依赖注入模块

提供FastAPI的依赖注入功能，包括：
1. 数据库会话依赖
2. 缓存管理器依赖
3. 用户认证依赖
4. 权限检查依赖
5. 分页参数依赖
6. 查询参数依赖
7. 文件上传依赖
8. 请求上下文依赖
"""

import re
from typing import Optional, List, Dict, Any, Union, Callable
from datetime import datetime

from fastapi import Depends, HTTPException, Query, Path, Header, File, UploadFile, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, validator

from .config import get_settings
from .database import get_db_session
from .cache import CacheManager
from .security import SecurityManager
from .logging import get_logger
from .exceptions import (
    AuthenticationException,
    AuthorizationException,
    ValidationException,
    raise_validation_error,
    raise_auth_error,
    raise_permission_error,
)


# 配置日志
logger = get_logger(__name__)


# =============================================================================
# 基础依赖
# =============================================================================

def get_settings_dependency():
    """获取配置依赖"""
    return get_settings()


def get_cache_manager() -> CacheManager:
    """获取缓存管理器依赖"""
    return CacheManager()


def get_security_manager() -> SecurityManager:
    """获取安全管理器依赖"""
    return SecurityManager()


def get_request_id(request: Request) -> Optional[str]:
    """获取请求ID依赖"""
    return getattr(request.state, 'request_id', None)


def get_client_ip(request: Request) -> str:
    """获取客户端IP依赖"""
    # 检查代理头
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip
    
    return request.client.host if request.client else "unknown"


# =============================================================================
# 分页参数
# =============================================================================

class PaginationParams(BaseModel):
    """分页参数模型"""
    
    page: int = Field(default=1, ge=1, description="页码")
    size: int = Field(default=20, ge=1, le=100, description="每页大小")
    
    @property
    def offset(self) -> int:
        """计算偏移量"""
        return (self.page - 1) * self.size
    
    @property
    def limit(self) -> int:
        """获取限制数量"""
        return self.size


def get_pagination_params(
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页大小")
) -> PaginationParams:
    """获取分页参数依赖"""
    return PaginationParams(page=page, size=size)


# =============================================================================
# 排序参数
# =============================================================================

class SortParams(BaseModel):
    """排序参数模型"""
    
    sort_by: Optional[str] = Field(default=None, description="排序字段")
    sort_order: str = Field(default="asc", regex="^(asc|desc)$", description="排序方向")
    
    @validator('sort_by')
    def validate_sort_by(cls, v):
        if v and not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
            raise ValueError("Invalid sort field name")
        return v


def get_sort_params(
    sort_by: Optional[str] = Query(None, description="排序字段"),
    sort_order: str = Query("asc", regex="^(asc|desc)$", description="排序方向")
) -> SortParams:
    """获取排序参数依赖"""
    return SortParams(sort_by=sort_by, sort_order=sort_order)


# =============================================================================
# 搜索参数
# =============================================================================

class SearchParams(BaseModel):
    """搜索参数模型"""
    
    q: Optional[str] = Field(default=None, max_length=200, description="搜索关键词")
    fields: Optional[List[str]] = Field(default=None, description="搜索字段")
    
    @validator('q')
    def validate_query(cls, v):
        if v and len(v.strip()) < 2:
            raise ValueError("Search query must be at least 2 characters")
        return v.strip() if v else None
    
    @validator('fields')
    def validate_fields(cls, v):
        if v:
            for field in v:
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', field):
                    raise ValueError(f"Invalid field name: {field}")
        return v


def get_search_params(
    q: Optional[str] = Query(None, max_length=200, description="搜索关键词"),
    fields: Optional[str] = Query(None, description="搜索字段（逗号分隔）")
) -> SearchParams:
    """获取搜索参数依赖"""
    field_list = None
    if fields:
        field_list = [f.strip() for f in fields.split(",") if f.strip()]
    
    return SearchParams(q=q, fields=field_list)


# =============================================================================
# 过滤参数
# =============================================================================

class FilterParams(BaseModel):
    """过滤参数模型"""
    
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")
    
    def add_filter(self, key: str, value: Any):
        """添加过滤条件"""
        if value is not None:
            self.filters[key] = value
    
    def get_filter(self, key: str, default: Any = None) -> Any:
        """获取过滤条件"""
        return self.filters.get(key, default)


def get_filter_params(
    status: Optional[str] = Query(None, description="状态过滤"),
    created_after: Optional[datetime] = Query(None, description="创建时间起始"),
    created_before: Optional[datetime] = Query(None, description="创建时间结束"),
) -> FilterParams:
    """获取过滤参数依赖"""
    params = FilterParams()
    params.add_filter("status", status)
    params.add_filter("created_after", created_after)
    params.add_filter("created_before", created_before)
    return params


# =============================================================================
# 认证依赖
# =============================================================================

security = HTTPBearer(auto_error=False)


class CurrentUser(BaseModel):
    """当前用户模型"""
    
    id: int
    username: str
    email: str
    is_active: bool = True
    is_superuser: bool = False
    roles: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    
    def has_role(self, role: str) -> bool:
        """检查是否有指定角色"""
        return role in self.roles or self.is_superuser
    
    def has_permission(self, permission: str) -> bool:
        """检查是否有指定权限"""
        return permission in self.permissions or self.is_superuser
    
    def has_any_role(self, roles: List[str]) -> bool:
        """检查是否有任意指定角色"""
        return any(self.has_role(role) for role in roles)
    
    def has_any_permission(self, permissions: List[str]) -> bool:
        """检查是否有任意指定权限"""
        return any(self.has_permission(perm) for perm in permissions)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    security_manager: SecurityManager = Depends(get_security_manager),
    db: AsyncSession = Depends(get_db_session),
    cache: CacheManager = Depends(get_cache_manager),
) -> CurrentUser:
    """获取当前用户依赖"""
    if not credentials:
        raise_auth_error("Missing authentication token")
    
    try:
        # 验证JWT令牌
        payload = security_manager.verify_jwt_token(credentials.credentials)
        user_id = payload.get("sub")
        
        if not user_id:
            raise_auth_error("Invalid token payload")
        
        # 从缓存获取用户信息
        cache_key = f"user:{user_id}"
        user_data = await cache.get(cache_key)
        
        if not user_data:
            # 从数据库获取用户信息（这里需要实际的用户模型）
            # user = await get_user_by_id(db, user_id)
            # if not user or not user.is_active:
            #     raise_auth_error("User not found or inactive")
            
            # 临时模拟用户数据
            user_data = {
                "id": int(user_id),
                "username": f"user_{user_id}",
                "email": f"user_{user_id}@example.com",
                "is_active": True,
                "is_superuser": False,
                "roles": ["user"],
                "permissions": ["read"],
            }
            
            # 缓存用户信息
            await cache.set(cache_key, user_data, expire=3600)
        
        return CurrentUser(**user_data)
    
    except Exception as e:
        logger.warning(f"Authentication failed: {e}")
        raise_auth_error("Invalid authentication token")


async def get_current_active_user(
    current_user: CurrentUser = Depends(get_current_user)
) -> CurrentUser:
    """获取当前活跃用户依赖"""
    if not current_user.is_active:
        raise_auth_error("User account is disabled")
    return current_user


async def get_current_superuser(
    current_user: CurrentUser = Depends(get_current_active_user)
) -> CurrentUser:
    """获取当前超级用户依赖"""
    if not current_user.is_superuser:
        raise_permission_error("Superuser access required")
    return current_user


# =============================================================================
# 权限检查依赖
# =============================================================================

def require_roles(required_roles: List[str]):
    """要求指定角色的依赖工厂"""
    
    async def check_roles(
        current_user: CurrentUser = Depends(get_current_active_user)
    ) -> CurrentUser:
        if not current_user.has_any_role(required_roles):
            raise_permission_error(f"Required roles: {', '.join(required_roles)}")
        return current_user
    
    return check_roles


def require_permissions(required_permissions: List[str]):
    """要求指定权限的依赖工厂"""
    
    async def check_permissions(
        current_user: CurrentUser = Depends(get_current_active_user)
    ) -> CurrentUser:
        if not current_user.has_any_permission(required_permissions):
            raise_permission_error(f"Required permissions: {', '.join(required_permissions)}")
        return current_user
    
    return check_permissions


def require_owner_or_admin(get_resource_owner_id: Callable):
    """要求资源所有者或管理员的依赖工厂"""
    
    async def check_owner_or_admin(
        resource_id: int = Path(...),
        current_user: CurrentUser = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db_session),
    ) -> CurrentUser:
        # 超级用户或管理员可以访问任何资源
        if current_user.is_superuser or current_user.has_role("admin"):
            return current_user
        
        # 检查是否为资源所有者
        owner_id = await get_resource_owner_id(db, resource_id)
        if owner_id != current_user.id:
            raise_permission_error("Access denied: not resource owner")
        
        return current_user
    
    return check_owner_or_admin


# =============================================================================
# 文件上传依赖
# =============================================================================

class FileUploadParams(BaseModel):
    """文件上传参数模型"""
    
    max_size: int = Field(default=10 * 1024 * 1024, description="最大文件大小")
    allowed_types: List[str] = Field(default_factory=list, description="允许的文件类型")
    allowed_extensions: List[str] = Field(default_factory=list, description="允许的文件扩展名")


def validate_file_upload(
    max_size: int = 10 * 1024 * 1024,
    allowed_types: Optional[List[str]] = None,
    allowed_extensions: Optional[List[str]] = None,
):
    """文件上传验证依赖工厂"""
    
    async def validate_file(
        file: UploadFile = File(...)
    ) -> UploadFile:
        # 检查文件大小
        if file.size and file.size > max_size:
            raise_validation_error(
                f"File size exceeds maximum allowed size of {max_size} bytes"
            )
        
        # 检查文件类型
        if allowed_types and file.content_type not in allowed_types:
            raise_validation_error(
                f"File type '{file.content_type}' not allowed. Allowed types: {', '.join(allowed_types)}"
            )
        
        # 检查文件扩展名
        if allowed_extensions:
            file_extension = file.filename.split('.')[-1].lower() if file.filename else ''
            if file_extension not in allowed_extensions:
                raise_validation_error(
                    f"File extension '.{file_extension}' not allowed. Allowed extensions: {', '.join(allowed_extensions)}"
                )
        
        return file
    
    return validate_file


# =============================================================================
# 请求验证依赖
# =============================================================================

def validate_json_content_type():
    """验证JSON内容类型依赖"""
    
    async def check_content_type(
        request: Request
    ):
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            raise_validation_error("Content-Type must be application/json")
    
    return check_content_type


def validate_api_key(
    header_name: str = "X-API-Key",
    valid_keys: Optional[List[str]] = None,
):
    """API密钥验证依赖工厂"""
    
    async def check_api_key(
        api_key: Optional[str] = Header(None, alias=header_name)
    ):
        if not api_key:
            raise_auth_error(f"Missing {header_name} header")
        
        if valid_keys and api_key not in valid_keys:
            raise_auth_error("Invalid API key")
        
        return api_key
    
    return check_api_key


# =============================================================================
# 缓存依赖
# =============================================================================

def get_cache_key_prefix(prefix: str):
    """获取缓存键前缀依赖工厂"""
    
    def get_prefix() -> str:
        return prefix
    
    return get_prefix


# =============================================================================
# 请求上下文依赖
# =============================================================================

class RequestContext(BaseModel):
    """请求上下文模型"""
    
    request_id: Optional[str] = None
    user_id: Optional[int] = None
    client_ip: str
    user_agent: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        arbitrary_types_allowed = True


async def get_request_context(
    request: Request,
    request_id: Optional[str] = Depends(get_request_id),
    client_ip: str = Depends(get_client_ip),
    current_user: Optional[CurrentUser] = Depends(get_current_user),
) -> RequestContext:
    """获取请求上下文依赖"""
    return RequestContext(
        request_id=request_id,
        user_id=current_user.id if current_user else None,
        client_ip=client_ip,
        user_agent=request.headers.get("user-agent"),
    )


# =============================================================================
# 数据库事务依赖
# =============================================================================

async def get_db_transaction(
    db: AsyncSession = Depends(get_db_session)
) -> AsyncSession:
    """获取数据库事务依赖"""
    async with db.begin():
        yield db


# =============================================================================
# 便捷组合依赖
# =============================================================================

class CommonQueryParams(BaseModel):
    """通用查询参数模型"""
    
    pagination: PaginationParams
    sort: SortParams
    search: SearchParams
    filters: FilterParams


async def get_common_query_params(
    pagination: PaginationParams = Depends(get_pagination_params),
    sort: SortParams = Depends(get_sort_params),
    search: SearchParams = Depends(get_search_params),
    filters: FilterParams = Depends(get_filter_params),
) -> CommonQueryParams:
    """获取通用查询参数依赖"""
    return CommonQueryParams(
        pagination=pagination,
        sort=sort,
        search=search,
        filters=filters,
    )


class AuthenticatedQueryParams(BaseModel):
    """认证查询参数模型"""
    
    common: CommonQueryParams
    user: CurrentUser
    context: RequestContext


async def get_authenticated_query_params(
    common: CommonQueryParams = Depends(get_common_query_params),
    user: CurrentUser = Depends(get_current_active_user),
    context: RequestContext = Depends(get_request_context),
) -> AuthenticatedQueryParams:
    """获取认证查询参数依赖"""
    return AuthenticatedQueryParams(
        common=common,
        user=user,
        context=context,
    )