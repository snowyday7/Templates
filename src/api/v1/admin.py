#!/usr/bin/env python3
"""
系统管理API端点

提供系统管理相关的API端点，包括：
1. 系统配置管理
2. 角色权限管理
3. 系统监控
4. 日志管理
5. 缓存管理
6. 任务管理
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from pydantic import BaseModel, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, or_, func, desc

from ...core.database import get_db_session
from ...core.security import JWTBearer
from ...core.cache import CacheManager
from ...core.responses import ResponseBuilder
from ...core.dependencies import (
    get_pagination_params,
    get_sorting_params,
    require_permissions,
    get_current_superuser
)
from ...core.exceptions import (
    ValidationException,
    ResourceNotFoundException,
    ConflictException,
    AuthorizationException
)
from ...models.user import User, UserRole, UserPermission, RoleType, PermissionType
from ...models.auth import LoginAttempt, AuthToken, LoginResult
from ...services.admin import AdminService
from ...services.user import UserService
from ...services.auth import AuthService
from ...services.system import SystemService


# =============================================================================
# 请求模型
# =============================================================================

class RoleCreateRequest(BaseModel):
    """角色创建请求"""
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    role_type: RoleType = RoleType.CUSTOM
    priority: int = 0
    permission_ids: List[int] = []
    
    @validator('name')
    def validate_name(cls, v):
        if len(v) < 2 or len(v) > 50:
            raise ValueError('角色名称长度必须在2-50个字符之间')
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('角色名称只能包含字母、数字、下划线和连字符')
        return v


class RoleUpdateRequest(BaseModel):
    """角色更新请求"""
    display_name: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[int] = None
    is_active: Optional[bool] = None
    permission_ids: Optional[List[int]] = None


class PermissionCreateRequest(BaseModel):
    """权限创建请求"""
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    permission_type: PermissionType = PermissionType.READ
    resource: Optional[str] = None
    action: Optional[str] = None
    
    @validator('name')
    def validate_name(cls, v):
        if len(v) < 2 or len(v) > 100:
            raise ValueError('权限名称长度必须在2-100个字符之间')
        return v


class PermissionUpdateRequest(BaseModel):
    """权限更新请求"""
    display_name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class UserRoleAssignRequest(BaseModel):
    """用户角色分配请求"""
    user_id: int
    role_ids: List[int]
    expires_at: Optional[datetime] = None


class SystemConfigUpdateRequest(BaseModel):
    """系统配置更新请求"""
    key: str
    value: Union[str, int, float, bool, dict, list]
    description: Optional[str] = None


class CacheOperationRequest(BaseModel):
    """缓存操作请求"""
    pattern: Optional[str] = None
    keys: Optional[List[str]] = None


# =============================================================================
# 响应模型
# =============================================================================

class RoleResponse(BaseModel):
    """角色响应"""
    id: int
    name: str
    display_name: Optional[str]
    description: Optional[str]
    role_type: RoleType
    is_active: bool
    priority: int
    user_count: int
    permissions: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class PermissionResponse(BaseModel):
    """权限响应"""
    id: int
    name: str
    display_name: Optional[str]
    description: Optional[str]
    permission_type: PermissionType
    resource: Optional[str]
    action: Optional[str]
    is_active: bool
    role_count: int
    created_at: datetime
    updated_at: datetime


class SystemStatsResponse(BaseModel):
    """系统统计响应"""
    users: Dict[str, int]
    sessions: Dict[str, int]
    tokens: Dict[str, int]
    login_attempts: Dict[str, int]
    system: Dict[str, Any]
    database: Dict[str, Any]
    cache: Dict[str, Any]


class LoginAttemptResponse(BaseModel):
    """登录尝试响应"""
    id: int
    username: str
    email: Optional[str]
    ip_address: str
    user_agent: Optional[str]
    result: LoginResult
    failure_reason: Optional[str]
    location: Optional[str]
    created_at: datetime


class SystemConfigResponse(BaseModel):
    """系统配置响应"""
    key: str
    value: Any
    description: Optional[str]
    updated_at: datetime
    updated_by: Optional[str]


# =============================================================================
# 路由器
# =============================================================================

router = APIRouter()
jwt_bearer = JWTBearer()


# =============================================================================
# 角色管理端点
# =============================================================================

@router.get(
    "/roles",
    response_model=List[RoleResponse],
    summary="获取角色列表",
    description="获取系统角色列表",
    dependencies=[Depends(require_permissions(["role:read"]))]
)
async def get_roles(
    pagination = Depends(get_pagination_params),
    sorting = Depends(get_sorting_params),
    is_active: Optional[bool] = Query(None, description="是否激活"),
    role_type: Optional[RoleType] = Query(None, description="角色类型"),
    db: AsyncSession = Depends(get_db_session)
):
    """获取角色列表"""
    admin_service = AdminService(db)
    
    filters = []
    if is_active is not None:
        filters.append(UserRole.is_active == is_active)
    if role_type:
        filters.append(UserRole.role_type == role_type)
    
    roles, total = await admin_service.get_roles_with_pagination(
        filters=filters,
        page=pagination['page'],
        size=pagination['size'],
        sort_by=sorting.get('sort_by', 'created_at'),
        sort_order=sorting.get('sort_order', 'desc')
    )
    
    role_responses = []
    for role in roles:
        role_data = role.to_dict()
        role_data['user_count'] = await admin_service.get_role_user_count(role.id)
        role_data['permissions'] = [perm.to_dict() for perm in role.permissions]
        role_responses.append(RoleResponse(**role_data))
    
    return ResponseBuilder.paginated_success(
        data=role_responses,
        total=total,
        page=pagination['page'],
        size=pagination['size']
    )


@router.post(
    "/roles",
    response_model=RoleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建角色",
    description="创建新的系统角色",
    dependencies=[Depends(require_permissions(["role:write"]))]
)
async def create_role(
    request: RoleCreateRequest,
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """创建角色"""
    admin_service = AdminService(db)
    
    # 检查角色名是否已存在
    existing_role = await admin_service.get_role_by_name(request.name)
    if existing_role:
        raise ConflictException("角色名已存在")
    
    # 验证权限ID
    if request.permission_ids:
        permissions = await admin_service.get_permissions_by_ids(request.permission_ids)
        if len(permissions) != len(request.permission_ids):
            raise ValidationException("部分权限ID无效")
    
    # 创建角色
    role_data = request.dict(exclude={'permission_ids'})
    role_data['created_by'] = current_user.id
    
    role = await admin_service.create_role(**role_data)
    
    # 分配权限
    if request.permission_ids:
        await admin_service.assign_permissions_to_role(role.id, request.permission_ids)
    
    # 获取完整角色信息
    role = await admin_service.get_role_by_id(role.id)
    role_data = role.to_dict()
    role_data['user_count'] = 0
    role_data['permissions'] = [perm.to_dict() for perm in role.permissions]
    
    return ResponseBuilder.success(
        message="角色创建成功",
        data=RoleResponse(**role_data)
    )


@router.get(
    "/roles/{role_id}",
    response_model=RoleResponse,
    summary="获取角色详情",
    description="获取指定角色的详细信息",
    dependencies=[Depends(require_permissions(["role:read"]))]
)
async def get_role(
    role_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """获取角色详情"""
    admin_service = AdminService(db)
    
    role = await admin_service.get_role_by_id(role_id)
    if not role:
        raise ResourceNotFoundException("角色不存在")
    
    role_data = role.to_dict()
    role_data['user_count'] = await admin_service.get_role_user_count(role.id)
    role_data['permissions'] = [perm.to_dict() for perm in role.permissions]
    
    return ResponseBuilder.success(
        data=RoleResponse(**role_data)
    )


@router.put(
    "/roles/{role_id}",
    response_model=RoleResponse,
    summary="更新角色",
    description="更新角色信息",
    dependencies=[Depends(require_permissions(["role:write"]))]
)
async def update_role(
    role_id: int,
    request: RoleUpdateRequest,
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """更新角色"""
    admin_service = AdminService(db)
    
    role = await admin_service.get_role_by_id(role_id)
    if not role:
        raise ResourceNotFoundException("角色不存在")
    
    # 系统角色不能修改某些属性
    if role.role_type == RoleType.SYSTEM:
        if request.permission_ids is not None:
            raise ValidationException("系统角色的权限不能修改")
    
    # 更新角色信息
    update_data = request.dict(exclude_unset=True, exclude={'permission_ids'})
    if update_data:
        update_data['updated_by'] = current_user.id
        await admin_service.update_role(role_id, **update_data)
    
    # 更新权限
    if request.permission_ids is not None:
        # 验证权限ID
        permissions = await admin_service.get_permissions_by_ids(request.permission_ids)
        if len(permissions) != len(request.permission_ids):
            raise ValidationException("部分权限ID无效")
        
        await admin_service.assign_permissions_to_role(role_id, request.permission_ids)
    
    # 获取更新后的角色信息
    role = await admin_service.get_role_by_id(role_id)
    role_data = role.to_dict()
    role_data['user_count'] = await admin_service.get_role_user_count(role.id)
    role_data['permissions'] = [perm.to_dict() for perm in role.permissions]
    
    return ResponseBuilder.success(
        message="角色更新成功",
        data=RoleResponse(**role_data)
    )


@router.delete(
    "/roles/{role_id}",
    summary="删除角色",
    description="删除角色",
    dependencies=[Depends(require_permissions(["role:delete"]))]
)
async def delete_role(
    role_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """删除角色"""
    admin_service = AdminService(db)
    
    role = await admin_service.get_role_by_id(role_id)
    if not role:
        raise ResourceNotFoundException("角色不存在")
    
    # 系统角色不能删除
    if role.role_type == RoleType.SYSTEM:
        raise ValidationException("系统角色不能删除")
    
    # 检查是否有用户使用此角色
    user_count = await admin_service.get_role_user_count(role_id)
    if user_count > 0:
        raise ValidationException(f"角色正在被 {user_count} 个用户使用，无法删除")
    
    await admin_service.delete_role(role_id)
    
    return ResponseBuilder.success(message="角色删除成功")


# =============================================================================
# 权限管理端点
# =============================================================================

@router.get(
    "/permissions",
    response_model=List[PermissionResponse],
    summary="获取权限列表",
    description="获取系统权限列表",
    dependencies=[Depends(require_permissions(["permission:read"]))]
)
async def get_permissions(
    pagination = Depends(get_pagination_params),
    sorting = Depends(get_sorting_params),
    is_active: Optional[bool] = Query(None, description="是否激活"),
    permission_type: Optional[PermissionType] = Query(None, description="权限类型"),
    resource: Optional[str] = Query(None, description="资源名称"),
    db: AsyncSession = Depends(get_db_session)
):
    """获取权限列表"""
    admin_service = AdminService(db)
    
    filters = []
    if is_active is not None:
        filters.append(UserPermission.is_active == is_active)
    if permission_type:
        filters.append(UserPermission.permission_type == permission_type)
    if resource:
        filters.append(UserPermission.resource.ilike(f"%{resource}%"))
    
    permissions, total = await admin_service.get_permissions_with_pagination(
        filters=filters,
        page=pagination['page'],
        size=pagination['size'],
        sort_by=sorting.get('sort_by', 'created_at'),
        sort_order=sorting.get('sort_order', 'desc')
    )
    
    permission_responses = []
    for permission in permissions:
        perm_data = permission.to_dict()
        perm_data['role_count'] = await admin_service.get_permission_role_count(permission.id)
        permission_responses.append(PermissionResponse(**perm_data))
    
    return ResponseBuilder.paginated_success(
        data=permission_responses,
        total=total,
        page=pagination['page'],
        size=pagination['size']
    )


@router.post(
    "/permissions",
    response_model=PermissionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建权限",
    description="创建新的系统权限",
    dependencies=[Depends(require_permissions(["permission:write"]))]
)
async def create_permission(
    request: PermissionCreateRequest,
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """创建权限"""
    admin_service = AdminService(db)
    
    # 检查权限名是否已存在
    existing_permission = await admin_service.get_permission_by_name(request.name)
    if existing_permission:
        raise ConflictException("权限名已存在")
    
    # 创建权限
    permission_data = request.dict()
    permission_data['created_by'] = current_user.id
    
    permission = await admin_service.create_permission(**permission_data)
    
    perm_data = permission.to_dict()
    perm_data['role_count'] = 0
    
    return ResponseBuilder.success(
        message="权限创建成功",
        data=PermissionResponse(**perm_data)
    )


@router.put(
    "/permissions/{permission_id}",
    response_model=PermissionResponse,
    summary="更新权限",
    description="更新权限信息",
    dependencies=[Depends(require_permissions(["permission:write"]))]
)
async def update_permission(
    permission_id: int,
    request: PermissionUpdateRequest,
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """更新权限"""
    admin_service = AdminService(db)
    
    permission = await admin_service.get_permission_by_id(permission_id)
    if not permission:
        raise ResourceNotFoundException("权限不存在")
    
    # 更新权限信息
    update_data = request.dict(exclude_unset=True)
    update_data['updated_by'] = current_user.id
    
    permission = await admin_service.update_permission(permission_id, **update_data)
    
    perm_data = permission.to_dict()
    perm_data['role_count'] = await admin_service.get_permission_role_count(permission.id)
    
    return ResponseBuilder.success(
        message="权限更新成功",
        data=PermissionResponse(**perm_data)
    )


@router.delete(
    "/permissions/{permission_id}",
    summary="删除权限",
    description="删除权限",
    dependencies=[Depends(require_permissions(["permission:delete"]))]
)
async def delete_permission(
    permission_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """删除权限"""
    admin_service = AdminService(db)
    
    permission = await admin_service.get_permission_by_id(permission_id)
    if not permission:
        raise ResourceNotFoundException("权限不存在")
    
    # 检查是否有角色使用此权限
    role_count = await admin_service.get_permission_role_count(permission_id)
    if role_count > 0:
        raise ValidationException(f"权限正在被 {role_count} 个角色使用，无法删除")
    
    await admin_service.delete_permission(permission_id)
    
    return ResponseBuilder.success(message="权限删除成功")


# =============================================================================
# 用户角色分配端点
# =============================================================================

@router.post(
    "/users/assign-roles",
    summary="分配用户角色",
    description="为用户分配角色",
    dependencies=[Depends(require_permissions(["user:write", "role:assign"]))]
)
async def assign_user_roles(
    request: UserRoleAssignRequest,
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """分配用户角色"""
    admin_service = AdminService(db)
    user_service = UserService(db)
    
    # 验证用户存在
    user = await user_service.get_by_id(request.user_id)
    if not user:
        raise ResourceNotFoundException("用户不存在")
    
    # 验证角色存在
    roles = await admin_service.get_roles_by_ids(request.role_ids)
    if len(roles) != len(request.role_ids):
        raise ValidationException("部分角色ID无效")
    
    # 分配角色
    await admin_service.assign_roles_to_user(
        request.user_id,
        request.role_ids,
        assigned_by=current_user.id,
        expires_at=request.expires_at
    )
    
    return ResponseBuilder.success(message="角色分配成功")


@router.delete(
    "/users/{user_id}/roles/{role_id}",
    summary="移除用户角色",
    description="移除用户的指定角色",
    dependencies=[Depends(require_permissions(["user:write", "role:assign"]))]
)
async def remove_user_role(
    user_id: int,
    role_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """移除用户角色"""
    admin_service = AdminService(db)
    user_service = UserService(db)
    
    # 验证用户存在
    user = await user_service.get_by_id(user_id)
    if not user:
        raise ResourceNotFoundException("用户不存在")
    
    # 验证角色存在
    role = await admin_service.get_role_by_id(role_id)
    if not role:
        raise ResourceNotFoundException("角色不存在")
    
    # 移除角色
    await admin_service.remove_role_from_user(user_id, role_id)
    
    return ResponseBuilder.success(message="角色移除成功")


# =============================================================================
# 系统监控端点
# =============================================================================

@router.get(
    "/stats",
    response_model=SystemStatsResponse,
    summary="获取系统统计",
    description="获取系统统计信息",
    dependencies=[Depends(get_current_superuser)]
)
async def get_system_stats(
    db: AsyncSession = Depends(get_db_session)
):
    """获取系统统计"""
    system_service = SystemService(db)
    
    stats = await system_service.get_system_statistics()
    
    return ResponseBuilder.success(
        data=SystemStatsResponse(**stats)
    )


@router.get(
    "/login-attempts",
    response_model=List[LoginAttemptResponse],
    summary="获取登录尝试记录",
    description="获取系统登录尝试记录",
    dependencies=[Depends(require_permissions(["system:read"]))]
)
async def get_login_attempts(
    pagination = Depends(get_pagination_params),
    result: Optional[LoginResult] = Query(None, description="登录结果"),
    ip_address: Optional[str] = Query(None, description="IP地址"),
    username: Optional[str] = Query(None, description="用户名"),
    start_date: Optional[datetime] = Query(None, description="开始时间"),
    end_date: Optional[datetime] = Query(None, description="结束时间"),
    db: AsyncSession = Depends(get_db_session)
):
    """获取登录尝试记录"""
    admin_service = AdminService(db)
    
    filters = []
    if result:
        filters.append(LoginAttempt.result == result)
    if ip_address:
        filters.append(LoginAttempt.ip_address == ip_address)
    if username:
        filters.append(LoginAttempt.username.ilike(f"%{username}%"))
    if start_date:
        filters.append(LoginAttempt.created_at >= start_date)
    if end_date:
        filters.append(LoginAttempt.created_at <= end_date)
    
    attempts, total = await admin_service.get_login_attempts_with_pagination(
        filters=filters,
        page=pagination['page'],
        size=pagination['size']
    )
    
    attempt_responses = [
        LoginAttemptResponse(**attempt.to_dict())
        for attempt in attempts
    ]
    
    return ResponseBuilder.paginated_success(
        data=attempt_responses,
        total=total,
        page=pagination['page'],
        size=pagination['size']
    )


# =============================================================================
# 缓存管理端点
# =============================================================================

@router.get(
    "/cache/info",
    summary="获取缓存信息",
    description="获取Redis缓存信息",
    dependencies=[Depends(get_current_superuser)]
)
async def get_cache_info():
    """获取缓存信息"""
    cache_manager = CacheManager()
    
    info = await cache_manager.get_info()
    
    return ResponseBuilder.success(
        data={
            "redis_info": info,
            "connection_status": "connected" if await cache_manager.ping() else "disconnected"
        }
    )


@router.post(
    "/cache/clear",
    summary="清理缓存",
    description="清理指定模式的缓存",
    dependencies=[Depends(get_current_superuser)]
)
async def clear_cache(
    request: CacheOperationRequest
):
    """清理缓存"""
    cache_manager = CacheManager()
    
    if request.pattern:
        # 按模式清理
        count = await cache_manager.clear_pattern(request.pattern)
        return ResponseBuilder.success(
            message=f"已清理 {count} 个缓存键",
            data={"cleared_count": count, "pattern": request.pattern}
        )
    elif request.keys:
        # 清理指定键
        count = await cache_manager.delete_many(request.keys)
        return ResponseBuilder.success(
            message=f"已清理 {count} 个缓存键",
            data={"cleared_count": count, "keys": request.keys}
        )
    else:
        raise ValidationException("必须指定清理模式或键列表")


@router.post(
    "/cache/flush",
    summary="清空所有缓存",
    description="清空Redis中的所有缓存",
    dependencies=[Depends(get_current_superuser)]
)
async def flush_cache():
    """清空所有缓存"""
    cache_manager = CacheManager()
    
    await cache_manager.clear_all()
    
    return ResponseBuilder.success(message="所有缓存已清空")


# =============================================================================
# 系统配置端点
# =============================================================================

@router.get(
    "/config",
    response_model=List[SystemConfigResponse],
    summary="获取系统配置",
    description="获取系统配置列表",
    dependencies=[Depends(get_current_superuser)]
)
async def get_system_config(
    db: AsyncSession = Depends(get_db_session)
):
    """获取系统配置"""
    system_service = SystemService(db)
    
    configs = await system_service.get_all_configs()
    
    config_responses = [
        SystemConfigResponse(**config)
        for config in configs
    ]
    
    return ResponseBuilder.success(data=config_responses)


@router.put(
    "/config",
    summary="更新系统配置",
    description="更新系统配置项",
    dependencies=[Depends(get_current_superuser)]
)
async def update_system_config(
    request: SystemConfigUpdateRequest,
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """更新系统配置"""
    system_service = SystemService(db)
    
    await system_service.update_config(
        request.key,
        request.value,
        description=request.description,
        updated_by=current_user.username
    )
    
    return ResponseBuilder.success(message="配置更新成功")


# =============================================================================
# 系统维护端点
# =============================================================================

@router.post(
    "/maintenance/cleanup",
    summary="系统清理",
    description="清理过期数据和临时文件",
    dependencies=[Depends(get_current_superuser)]
)
async def system_cleanup(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session)
):
    """系统清理"""
    system_service = SystemService(db)
    
    # 在后台执行清理任务
    background_tasks.add_task(system_service.cleanup_expired_data)
    background_tasks.add_task(system_service.cleanup_temp_files)
    background_tasks.add_task(system_service.cleanup_old_logs)
    
    return ResponseBuilder.success(message="系统清理任务已启动")


@router.post(
    "/maintenance/backup",
    summary="数据备份",
    description="创建数据库备份",
    dependencies=[Depends(get_current_superuser)]
)
async def create_backup(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session)
):
    """创建数据备份"""
    system_service = SystemService(db)
    
    # 在后台执行备份任务
    background_tasks.add_task(system_service.create_database_backup)
    
    return ResponseBuilder.success(message="数据备份任务已启动")


@router.get(
    "/maintenance/status",
    summary="维护状态",
    description="获取系统维护状态",
    dependencies=[Depends(get_current_superuser)]
)
async def get_maintenance_status(
    db: AsyncSession = Depends(get_db_session)
):
    """获取维护状态"""
    system_service = SystemService(db)
    
    status = await system_service.get_maintenance_status()
    
    return ResponseBuilder.success(data=status)