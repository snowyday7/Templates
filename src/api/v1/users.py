#!/usr/bin/env python3
"""
用户管理API端点

提供用户管理相关的API端点，包括：
1. 用户信息查询
2. 用户信息更新
3. 用户头像上传
4. 用户偏好设置
5. 用户会话管理
6. 用户权限查询
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, or_

from ...core.database import get_db_session
from ...core.security import JWTBearer
from ...core.responses import ResponseBuilder
from ...core.dependencies import (
    get_pagination_params,
    get_sorting_params,
    get_search_params,
    require_permissions
)
from ...core.exceptions import (
    ValidationException,
    ResourceNotFoundException,
    ConflictException,
    AuthorizationException
)
from ...models.user import User, UserStatus, UserGender, UserSession
from ...services.user import UserService
from ...services.file import FileService
from ...services.auth import AuthService


# =============================================================================
# 请求模型
# =============================================================================

class UserUpdateRequest(BaseModel):
    """用户信息更新请求"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: Optional[str] = None
    phone: Optional[str] = None
    bio: Optional[str] = None
    gender: Optional[UserGender] = None
    birth_date: Optional[datetime] = None
    
    @validator('phone')
    def validate_phone(cls, v):
        if v and len(v) < 10:
            raise ValueError('手机号码格式不正确')
        return v
    
    @validator('bio')
    def validate_bio(cls, v):
        if v and len(v) > 500:
            raise ValueError('个人简介不能超过500个字符')
        return v


class UserPreferencesRequest(BaseModel):
    """用户偏好设置请求"""
    timezone: Optional[str] = None
    language: Optional[str] = None
    theme: Optional[str] = None
    
    @validator('timezone')
    def validate_timezone(cls, v):
        if v:
            import pytz
            try:
                pytz.timezone(v)
            except pytz.exceptions.UnknownTimeZoneError:
                raise ValueError('无效的时区')
        return v
    
    @validator('language')
    def validate_language(cls, v):
        if v and v not in ['en', 'zh', 'zh-CN', 'zh-TW', 'ja', 'ko']:
            raise ValueError('不支持的语言')
        return v
    
    @validator('theme')
    def validate_theme(cls, v):
        if v and v not in ['light', 'dark', 'auto']:
            raise ValueError('不支持的主题')
        return v


class UserSearchRequest(BaseModel):
    """用户搜索请求"""
    keyword: Optional[str] = None
    status: Optional[UserStatus] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None


# =============================================================================
# 响应模型
# =============================================================================

class UserDetailResponse(BaseModel):
    """用户详细信息响应"""
    id: int
    username: str
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    display_name: Optional[str]
    full_name: str
    avatar_url: Optional[str]
    phone: Optional[str]
    bio: Optional[str]
    gender: Optional[UserGender]
    birth_date: Optional[datetime]
    status: UserStatus
    is_active: bool
    is_verified: bool
    is_email_verified: bool
    is_phone_verified: bool
    two_factor_enabled: bool
    timezone: str
    language: str
    theme: str
    last_login_at: Optional[datetime]
    last_activity_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    roles: List[str]
    permissions: List[str]


class UserListResponse(BaseModel):
    """用户列表响应"""
    id: int
    username: str
    email: str
    display_name: Optional[str]
    avatar_url: Optional[str]
    status: UserStatus
    is_active: bool
    is_verified: bool
    last_login_at: Optional[datetime]
    created_at: datetime


class UserSessionResponse(BaseModel):
    """用户会话响应"""
    id: int
    ip_address: Optional[str]
    user_agent: Optional[str]
    device_info: Optional[str]
    location: Optional[str]
    is_current: bool
    last_activity_at: Optional[datetime]
    created_at: datetime
    expires_at: datetime


class UserStatsResponse(BaseModel):
    """用户统计响应"""
    total_users: int
    active_users: int
    verified_users: int
    new_users_today: int
    new_users_this_week: int
    new_users_this_month: int


# =============================================================================
# 路由器
# =============================================================================

router = APIRouter()
jwt_bearer = JWTBearer()


# =============================================================================
# API端点
# =============================================================================

@router.get(
    "/me",
    response_model=UserDetailResponse,
    summary="获取当前用户信息",
    description="获取当前登录用户的详细信息"
)
async def get_current_user(
    current_user: User = Depends(jwt_bearer)
):
    """获取当前用户信息"""
    return ResponseBuilder.success(
        data=UserDetailResponse(
            **current_user.to_dict(),
            full_name=current_user.full_name,
            roles=[role.name for role in current_user.roles],
            permissions=current_user.get_all_permissions()
        )
    )


@router.put(
    "/me",
    response_model=UserDetailResponse,
    summary="更新当前用户信息",
    description="更新当前登录用户的信息"
)
async def update_current_user(
    request: UserUpdateRequest,
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """更新当前用户信息"""
    user_service = UserService(db)
    
    # 检查手机号是否已被其他用户使用
    if request.phone and request.phone != current_user.phone:
        existing_user = await user_service.get_by_phone(request.phone)
        if existing_user and existing_user.id != current_user.id:
            raise ConflictException("手机号已被使用")
    
    # 更新用户信息
    update_data = request.dict(exclude_unset=True)
    updated_user = await user_service.update_user(current_user.id, **update_data)
    
    return ResponseBuilder.success(
        message="用户信息更新成功",
        data=UserDetailResponse(
            **updated_user.to_dict(),
            full_name=updated_user.full_name,
            roles=[role.name for role in updated_user.roles],
            permissions=updated_user.get_all_permissions()
        )
    )


@router.post(
    "/me/avatar",
    summary="上传用户头像",
    description="上传并设置用户头像"
)
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """上传用户头像"""
    # 验证文件类型
    if not file.content_type.startswith('image/'):
        raise ValidationException("只能上传图片文件")
    
    # 验证文件大小（5MB）
    if file.size > 5 * 1024 * 1024:
        raise ValidationException("文件大小不能超过5MB")
    
    file_service = FileService()
    user_service = UserService(db)
    
    # 上传文件
    file_url = await file_service.upload_avatar(
        file, current_user.id
    )
    
    # 更新用户头像URL
    current_user.avatar_url = file_url
    await db.commit()
    
    return ResponseBuilder.success(
        message="头像上传成功",
        data={"avatar_url": file_url}
    )


@router.delete(
    "/me/avatar",
    summary="删除用户头像",
    description="删除用户头像"
)
async def delete_avatar(
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """删除用户头像"""
    if current_user.avatar_url:
        file_service = FileService()
        await file_service.delete_file(current_user.avatar_url)
        
        current_user.avatar_url = None
        await db.commit()
    
    return ResponseBuilder.success(message="头像删除成功")


@router.put(
    "/me/preferences",
    summary="更新用户偏好设置",
    description="更新用户的偏好设置"
)
async def update_preferences(
    request: UserPreferencesRequest,
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """更新用户偏好设置"""
    update_data = request.dict(exclude_unset=True)
    
    for key, value in update_data.items():
        setattr(current_user, key, value)
    
    await db.commit()
    
    return ResponseBuilder.success(
        message="偏好设置更新成功",
        data={
            "timezone": current_user.timezone,
            "language": current_user.language,
            "theme": current_user.theme
        }
    )


@router.get(
    "/me/sessions",
    response_model=List[UserSessionResponse],
    summary="获取用户会话列表",
    description="获取当前用户的所有活跃会话"
)
async def get_user_sessions(
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """获取用户会话列表"""
    auth_service = AuthService(db)
    
    sessions = await auth_service.get_user_sessions(current_user.id)
    
    # 获取当前会话ID（从JWT中）
    current_session_id = getattr(current_user, '_current_session_id', None)
    
    session_responses = []
    for session in sessions:
        session_responses.append(
            UserSessionResponse(
                **session.to_dict(),
                is_current=session.id == current_session_id
            )
        )
    
    return ResponseBuilder.success(data=session_responses)


@router.delete(
    "/me/sessions/{session_id}",
    summary="撤销用户会话",
    description="撤销指定的用户会话"
)
async def revoke_session(
    session_id: int,
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """撤销用户会话"""
    auth_service = AuthService(db)
    
    session = await auth_service.get_session_by_id(session_id)
    if not session or session.user_id != current_user.id:
        raise ResourceNotFoundException("会话不存在")
    
    await auth_service.revoke_session(session_id)
    
    return ResponseBuilder.success(message="会话已撤销")


@router.delete(
    "/me/sessions",
    summary="撤销所有其他会话",
    description="撤销除当前会话外的所有其他会话"
)
async def revoke_other_sessions(
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """撤销所有其他会话"""
    auth_service = AuthService(db)
    
    # 获取当前会话ID
    current_session_id = getattr(current_user, '_current_session_id', None)
    
    await auth_service.revoke_user_sessions(
        current_user.id, 
        exclude_session_id=current_session_id
    )
    
    return ResponseBuilder.success(message="其他会话已全部撤销")


@router.get(
    "/me/permissions",
    summary="获取用户权限",
    description="获取当前用户的所有权限"
)
async def get_user_permissions(
    current_user: User = Depends(jwt_bearer)
):
    """获取用户权限"""
    permissions = current_user.get_all_permissions()
    roles = [role.name for role in current_user.roles]
    
    return ResponseBuilder.success(
        data={
            "roles": roles,
            "permissions": permissions,
            "is_superuser": current_user.is_superuser
        }
    )


# =============================================================================
# 管理员端点（需要特定权限）
# =============================================================================

@router.get(
    "/",
    response_model=List[UserListResponse],
    summary="获取用户列表",
    description="获取用户列表（需要用户管理权限）",
    dependencies=[Depends(require_permissions(["user:read"]))]
)
async def get_users(
    search: UserSearchRequest = Depends(),
    pagination = Depends(get_pagination_params),
    sorting = Depends(get_sorting_params),
    db: AsyncSession = Depends(get_db_session)
):
    """获取用户列表"""
    user_service = UserService(db)
    
    # 构建查询条件
    filters = []
    if search.keyword:
        filters.append(
            or_(
                User.username.ilike(f"%{search.keyword}%"),
                User.email.ilike(f"%{search.keyword}%"),
                User.display_name.ilike(f"%{search.keyword}%")
            )
        )
    
    if search.status:
        filters.append(User.status == search.status)
    
    if search.is_active is not None:
        filters.append(User.is_active == search.is_active)
    
    if search.is_verified is not None:
        filters.append(User.is_verified == search.is_verified)
    
    if search.created_after:
        filters.append(User.created_at >= search.created_after)
    
    if search.created_before:
        filters.append(User.created_at <= search.created_before)
    
    # 查询用户
    users, total = await user_service.get_users_with_pagination(
        filters=filters,
        page=pagination['page'],
        size=pagination['size'],
        sort_by=sorting.get('sort_by', 'created_at'),
        sort_order=sorting.get('sort_order', 'desc')
    )
    
    user_responses = [
        UserListResponse(**user.to_dict())
        for user in users
    ]
    
    return ResponseBuilder.paginated_success(
        data=user_responses,
        total=total,
        page=pagination['page'],
        size=pagination['size']
    )


@router.get(
    "/{user_id}",
    response_model=UserDetailResponse,
    summary="获取用户详情",
    description="获取指定用户的详细信息（需要用户管理权限）",
    dependencies=[Depends(require_permissions(["user:read"]))]
)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """获取用户详情"""
    user_service = UserService(db)
    
    user = await user_service.get_by_id(user_id)
    if not user:
        raise ResourceNotFoundException("用户不存在")
    
    return ResponseBuilder.success(
        data=UserDetailResponse(
            **user.to_dict(),
            full_name=user.full_name,
            roles=[role.name for role in user.roles],
            permissions=user.get_all_permissions()
        )
    )


@router.put(
    "/{user_id}/status",
    summary="更新用户状态",
    description="更新用户状态（需要用户管理权限）",
    dependencies=[Depends(require_permissions(["user:write"]))]
)
async def update_user_status(
    user_id: int,
    status: UserStatus,
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """更新用户状态"""
    user_service = UserService(db)
    
    user = await user_service.get_by_id(user_id)
    if not user:
        raise ResourceNotFoundException("用户不存在")
    
    # 不能修改自己的状态
    if user.id == current_user.id:
        raise ValidationException("不能修改自己的状态")
    
    # 不能修改超级用户的状态（除非自己也是超级用户）
    if user.is_superuser and not current_user.is_superuser:
        raise AuthorizationException("无权修改超级用户状态")
    
    user.status = status
    await db.commit()
    
    return ResponseBuilder.success(
        message=f"用户状态已更新为 {status.value}",
        data=user.to_dict()
    )


@router.delete(
    "/{user_id}",
    summary="删除用户",
    description="软删除用户（需要用户管理权限）",
    dependencies=[Depends(require_permissions(["user:delete"]))]
)
async def delete_user(
    user_id: int,
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """删除用户"""
    user_service = UserService(db)
    
    user = await user_service.get_by_id(user_id)
    if not user:
        raise ResourceNotFoundException("用户不存在")
    
    # 不能删除自己
    if user.id == current_user.id:
        raise ValidationException("不能删除自己")
    
    # 不能删除超级用户（除非自己也是超级用户）
    if user.is_superuser and not current_user.is_superuser:
        raise AuthorizationException("无权删除超级用户")
    
    await user_service.soft_delete_user(user_id)
    
    return ResponseBuilder.success(message="用户已删除")


@router.get(
    "/stats/overview",
    response_model=UserStatsResponse,
    summary="获取用户统计",
    description="获取用户统计信息（需要用户管理权限）",
    dependencies=[Depends(require_permissions(["user:read"]))]
)
async def get_user_stats(
    db: AsyncSession = Depends(get_db_session)
):
    """获取用户统计"""
    user_service = UserService(db)
    
    stats = await user_service.get_user_statistics()
    
    return ResponseBuilder.success(
        data=UserStatsResponse(**stats)
    )


@router.post(
    "/{user_id}/unlock",
    summary="解锁用户账户",
    description="解锁被锁定的用户账户（需要用户管理权限）",
    dependencies=[Depends(require_permissions(["user:write"]))]
)
async def unlock_user(
    user_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """解锁用户账户"""
    user_service = UserService(db)
    
    user = await user_service.get_by_id(user_id)
    if not user:
        raise ResourceNotFoundException("用户不存在")
    
    if not user.is_locked:
        raise ValidationException("用户账户未被锁定")
    
    user.unlock_account()
    await db.commit()
    
    return ResponseBuilder.success(message="用户账户已解锁")


@router.post(
    "/{user_id}/verify-email",
    summary="手动验证用户邮箱",
    description="手动验证用户邮箱（需要用户管理权限）",
    dependencies=[Depends(require_permissions(["user:write"]))]
)
async def verify_user_email(
    user_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """手动验证用户邮箱"""
    user_service = UserService(db)
    
    user = await user_service.get_by_id(user_id)
    if not user:
        raise ResourceNotFoundException("用户不存在")
    
    if user.is_email_verified:
        raise ValidationException("用户邮箱已验证")
    
    user.verify_email()
    await db.commit()
    
    return ResponseBuilder.success(message="用户邮箱已验证")


@router.get(
    "/{user_id}/sessions",
    response_model=List[UserSessionResponse],
    summary="获取用户会话列表",
    description="获取指定用户的会话列表（需要用户管理权限）",
    dependencies=[Depends(require_permissions(["user:read"]))]
)
async def get_user_sessions_admin(
    user_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """获取用户会话列表（管理员）"""
    user_service = UserService(db)
    auth_service = AuthService(db)
    
    user = await user_service.get_by_id(user_id)
    if not user:
        raise ResourceNotFoundException("用户不存在")
    
    sessions = await auth_service.get_user_sessions(user_id)
    
    session_responses = [
        UserSessionResponse(**session.to_dict(), is_current=False)
        for session in sessions
    ]
    
    return ResponseBuilder.success(data=session_responses)


@router.delete(
    "/{user_id}/sessions",
    summary="撤销用户所有会话",
    description="撤销指定用户的所有会话（需要用户管理权限）",
    dependencies=[Depends(require_permissions(["user:write"]))]
)
async def revoke_user_sessions_admin(
    user_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """撤销用户所有会话（管理员）"""
    user_service = UserService(db)
    auth_service = AuthService(db)
    
    user = await user_service.get_by_id(user_id)
    if not user:
        raise ResourceNotFoundException("用户不存在")
    
    await auth_service.revoke_user_sessions(user_id)
    
    return ResponseBuilder.success(message="用户所有会话已撤销")