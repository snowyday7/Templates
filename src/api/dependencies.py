#!/usr/bin/env python3
"""
API依赖项

提供FastAPI路由中使用的依赖项，包括：
1. 认证和授权依赖
2. 分页和排序依赖
3. 数据库会话依赖
4. 缓存依赖
5. 限流依赖
6. 文件上传依赖
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, status, Query, Request, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from ..core.database import get_db_session
from ..core.security import JWTBearer, verify_token
from ..core.cache import CacheManager
from ..core.exceptions import (
    AuthenticationException,
    AuthorizationException,
    ValidationException,
    RateLimitException
)
from ..models.user import User, UserPermission, UserRole
from ..services.user import UserService
from ..services.auth import AuthService


# =============================================================================
# 认证依赖
# =============================================================================

security = HTTPBearer()
jwt_bearer = JWTBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db_session)
) -> User:
    """
    获取当前认证用户
    
    Args:
        credentials: JWT凭证
        db: 数据库会话
        
    Returns:
        User: 当前用户对象
        
    Raises:
        AuthenticationException: 认证失败
    """
    try:
        # 验证JWT令牌
        payload = verify_token(credentials.credentials)
        user_id = payload.get("sub")
        
        if not user_id:
            raise AuthenticationException("无效的令牌")
        
        # 获取用户信息
        user_service = UserService(db)
        user = await user_service.get_by_id(int(user_id))
        
        if not user:
            raise AuthenticationException("用户不存在")
        
        if not user.is_active:
            raise AuthenticationException("用户已被禁用")
        
        if user.is_locked:
            raise AuthenticationException("用户账户已被锁定")
        
        return user
        
    except Exception as e:
        if isinstance(e, AuthenticationException):
            raise e
        raise AuthenticationException("认证失败")


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    获取当前活跃用户
    
    Args:
        current_user: 当前用户
        
    Returns:
        User: 活跃用户对象
        
    Raises:
        AuthenticationException: 用户未激活
    """
    if not current_user.is_active:
        raise AuthenticationException("用户未激活")
    
    return current_user


async def get_current_superuser(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    获取当前超级用户
    
    Args:
        current_user: 当前用户
        
    Returns:
        User: 超级用户对象
        
    Raises:
        AuthorizationException: 权限不足
    """
    if not current_user.is_superuser:
        raise AuthorizationException("需要超级用户权限")
    
    return current_user


async def get_current_staff(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    获取当前员工用户
    
    Args:
        current_user: 当前用户
        
    Returns:
        User: 员工用户对象
        
    Raises:
        AuthorizationException: 权限不足
    """
    if not current_user.is_staff:
        raise AuthorizationException("需要员工权限")
    
    return current_user


def require_permissions(permissions: List[str]):
    """
    权限检查依赖工厂
    
    Args:
        permissions: 需要的权限列表
        
    Returns:
        依赖函数
    """
    async def check_permissions(
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db_session)
    ) -> User:
        """
        检查用户权限
        
        Args:
            current_user: 当前用户
            db: 数据库会话
            
        Returns:
            User: 有权限的用户对象
            
        Raises:
            AuthorizationException: 权限不足
        """
        # 超级用户拥有所有权限
        if current_user.is_superuser:
            return current_user
        
        # 检查用户权限
        user_service = UserService(db)
        user_permissions = await user_service.get_user_permissions(current_user.id)
        
        user_permission_names = {perm.name for perm in user_permissions}
        
        # 检查是否拥有所需权限
        missing_permissions = set(permissions) - user_permission_names
        if missing_permissions:
            raise AuthorizationException(
                f"缺少权限: {', '.join(missing_permissions)}"
            )
        
        return current_user
    
    return check_permissions


def require_roles(roles: List[str]):
    """
    角色检查依赖工厂
    
    Args:
        roles: 需要的角色列表
        
    Returns:
        依赖函数
    """
    async def check_roles(
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db_session)
    ) -> User:
        """
        检查用户角色
        
        Args:
            current_user: 当前用户
            db: 数据库会话
            
        Returns:
            User: 有角色的用户对象
            
        Raises:
            AuthorizationException: 角色不足
        """
        # 超级用户拥有所有角色
        if current_user.is_superuser:
            return current_user
        
        # 检查用户角色
        user_service = UserService(db)
        user_roles = await user_service.get_user_roles(current_user.id)
        
        user_role_names = {role.name for role in user_roles}
        
        # 检查是否拥有所需角色
        if not any(role in user_role_names for role in roles):
            raise AuthorizationException(
                f"需要以下角色之一: {', '.join(roles)}"
            )
        
        return current_user
    
    return check_roles


# =============================================================================
# 分页和排序依赖
# =============================================================================

class PaginationParams(BaseModel):
    """分页参数"""
    page: int = 1
    size: int = 20
    
    def to_dict(self) -> Dict[str, int]:
        return {"page": self.page, "size": self.size}


class SortingParams(BaseModel):
    """排序参数"""
    sort_by: Optional[str] = None
    sort_order: str = "desc"
    
    def to_dict(self) -> Dict[str, Optional[str]]:
        return {"sort_by": self.sort_by, "sort_order": self.sort_order}


def get_pagination_params(
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页数量")
) -> Dict[str, int]:
    """
    获取分页参数
    
    Args:
        page: 页码
        size: 每页数量
        
    Returns:
        分页参数字典
    """
    return {"page": page, "size": size}


def get_sorting_params(
    sort_by: Optional[str] = Query(None, description="排序字段"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="排序方向")
) -> Dict[str, Optional[str]]:
    """
    获取排序参数
    
    Args:
        sort_by: 排序字段
        sort_order: 排序方向
        
    Returns:
        排序参数字典
    """
    return {"sort_by": sort_by, "sort_order": sort_order}


def get_filter_params(
    search: Optional[str] = Query(None, description="搜索关键词"),
    status: Optional[str] = Query(None, description="状态筛选"),
    start_date: Optional[datetime] = Query(None, description="开始日期"),
    end_date: Optional[datetime] = Query(None, description="结束日期")
) -> Dict[str, Any]:
    """
    获取筛选参数
    
    Args:
        search: 搜索关键词
        status: 状态筛选
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        筛选参数字典
    """
    return {
        "search": search,
        "status": status,
        "start_date": start_date,
        "end_date": end_date
    }


# =============================================================================
# 缓存依赖
# =============================================================================

async def get_cache_manager() -> CacheManager:
    """
    获取缓存管理器
    
    Returns:
        CacheManager: 缓存管理器实例
    """
    return CacheManager()


def cache_key_factory(prefix: str):
    """
    缓存键工厂
    
    Args:
        prefix: 缓存键前缀
        
    Returns:
        缓存键生成函数
    """
    def generate_cache_key(
        request: Request,
        current_user: Optional[User] = None
    ) -> str:
        """
        生成缓存键
        
        Args:
            request: 请求对象
            current_user: 当前用户
            
        Returns:
            缓存键
        """
        # 基础键
        key_parts = [prefix, request.url.path]
        
        # 添加查询参数
        if request.query_params:
            query_string = str(request.query_params)
            key_parts.append(query_string)
        
        # 添加用户ID（如果有）
        if current_user:
            key_parts.append(f"user:{current_user.id}")
        
        return ":".join(key_parts)
    
    return generate_cache_key


# =============================================================================
# 限流依赖
# =============================================================================

class RateLimiter:
    """
    限流器
    """
    
    def __init__(self, max_requests: int, window_seconds: int):
        """
        初始化限流器
        
        Args:
            max_requests: 最大请求数
            window_seconds: 时间窗口（秒）
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    async def __call__(
        self,
        request: Request,
        cache: CacheManager = Depends(get_cache_manager)
    ):
        """
        执行限流检查
        
        Args:
            request: 请求对象
            cache: 缓存管理器
            
        Raises:
            RateLimitException: 超出限流
        """
        # 获取客户端IP
        client_ip = request.client.host
        
        # 生成限流键
        rate_limit_key = f"rate_limit:{client_ip}:{request.url.path}"
        
        # 获取当前请求计数
        current_requests = await cache.get(rate_limit_key)
        
        if current_requests is None:
            # 首次请求
            await cache.set(rate_limit_key, 1, expire=self.window_seconds)
        else:
            current_requests = int(current_requests)
            if current_requests >= self.max_requests:
                # 超出限流
                raise RateLimitException(
                    f"请求过于频繁，请在 {self.window_seconds} 秒后重试"
                )
            
            # 增加计数
            await cache.increment(rate_limit_key)


def rate_limit(max_requests: int, window_seconds: int):
    """
    限流装饰器工厂
    
    Args:
        max_requests: 最大请求数
        window_seconds: 时间窗口（秒）
        
    Returns:
        限流依赖
    """
    return RateLimiter(max_requests, window_seconds)


# =============================================================================
# 文件上传依赖
# =============================================================================

class FileUploadValidator:
    """
    文件上传验证器
    """
    
    def __init__(
        self,
        max_size: int = 10 * 1024 * 1024,  # 10MB
        allowed_types: Optional[List[str]] = None,
        allowed_extensions: Optional[List[str]] = None
    ):
        """
        初始化文件上传验证器
        
        Args:
            max_size: 最大文件大小（字节）
            allowed_types: 允许的MIME类型
            allowed_extensions: 允许的文件扩展名
        """
        self.max_size = max_size
        self.allowed_types = allowed_types or []
        self.allowed_extensions = allowed_extensions or []
    
    async def __call__(self, file: UploadFile) -> UploadFile:
        """
        验证上传文件
        
        Args:
            file: 上传文件
            
        Returns:
            验证通过的文件
            
        Raises:
            ValidationException: 文件验证失败
        """
        # 检查文件大小
        if file.size and file.size > self.max_size:
            raise ValidationException(
                f"文件大小超出限制，最大允许 {self.max_size // (1024*1024)}MB"
            )
        
        # 检查MIME类型
        if self.allowed_types and file.content_type not in self.allowed_types:
            raise ValidationException(
                f"不支持的文件类型: {file.content_type}"
            )
        
        # 检查文件扩展名
        if self.allowed_extensions:
            file_extension = file.filename.split('.')[-1].lower() if file.filename else ''
            if file_extension not in self.allowed_extensions:
                raise ValidationException(
                    f"不支持的文件扩展名: {file_extension}"
                )
        
        return file


def validate_image_upload(
    max_size: int = 5 * 1024 * 1024  # 5MB
):
    """
    图片上传验证依赖
    
    Args:
        max_size: 最大文件大小
        
    Returns:
        文件验证器
    """
    return FileUploadValidator(
        max_size=max_size,
        allowed_types=[
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp"
        ],
        allowed_extensions=["jpg", "jpeg", "png", "gif", "webp"]
    )


def validate_document_upload(
    max_size: int = 20 * 1024 * 1024  # 20MB
):
    """
    文档上传验证依赖
    
    Args:
        max_size: 最大文件大小
        
    Returns:
        文件验证器
    """
    return FileUploadValidator(
        max_size=max_size,
        allowed_types=[
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/plain",
            "text/csv"
        ],
        allowed_extensions=[
            "pdf", "doc", "docx", "xls", "xlsx", "txt", "csv"
        ]
    )


# =============================================================================
# 请求验证依赖
# =============================================================================

async def validate_json_content_type(request: Request):
    """
    验证JSON内容类型
    
    Args:
        request: 请求对象
        
    Raises:
        ValidationException: 内容类型无效
    """
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise ValidationException("请求必须是JSON格式")


async def validate_request_size(
    request: Request,
    max_size: int = 1024 * 1024  # 1MB
):
    """
    验证请求大小
    
    Args:
        request: 请求对象
        max_size: 最大请求大小
        
    Raises:
        ValidationException: 请求过大
    """
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        raise ValidationException(
            f"请求体过大，最大允许 {max_size // 1024}KB"
        )


# =============================================================================
# 地理位置依赖
# =============================================================================

async def get_client_ip(request: Request) -> str:
    """
    获取客户端IP地址
    
    Args:
        request: 请求对象
        
    Returns:
        客户端IP地址
    """
    # 检查代理头
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # 返回直接连接的IP
    return request.client.host


async def get_user_agent(request: Request) -> Optional[str]:
    """
    获取用户代理字符串
    
    Args:
        request: 请求对象
        
    Returns:
        用户代理字符串
    """
    return request.headers.get("User-Agent")


async def get_request_info(request: Request) -> Dict[str, Any]:
    """
    获取请求信息
    
    Args:
        request: 请求对象
        
    Returns:
        请求信息字典
    """
    return {
        "ip_address": await get_client_ip(request),
        "user_agent": await get_user_agent(request),
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "timestamp": datetime.utcnow()
    }


# =============================================================================
# 常用依赖组合
# =============================================================================

# 管理员权限依赖
admin_required = Depends(require_permissions(["admin:read", "admin:write"]))

# 用户管理权限依赖
user_management_required = Depends(require_permissions(["user:read", "user:write"]))

# 系统监控权限依赖
system_monitor_required = Depends(require_permissions(["system:read"]))

# API限流依赖（每分钟60次请求）
api_rate_limit = Depends(rate_limit(60, 60))

# 严格限流依赖（每分钟10次请求）
strict_rate_limit = Depends(rate_limit(10, 60))

# 图片上传依赖
image_upload = Depends(validate_image_upload())

# 文档上传依赖
document_upload = Depends(validate_document_upload())