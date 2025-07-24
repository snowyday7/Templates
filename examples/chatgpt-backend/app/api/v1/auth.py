# -*- coding: utf-8 -*-
"""
认证API路由

提供用户注册、登录、令牌刷新等认证相关的API端点。
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

# 导入模板库组件
from src.core.security import (
    create_access_token,
    create_refresh_token,
    verify_token,
    get_password_hash,
    verify_password
)
from src.utils.response import create_response, create_error_response
from src.utils.exceptions import ValidationException, AuthenticationException
from src.core.logging import get_logger

# 导入应用组件
from app.core.database import get_db
from app.core.config import get_settings
from app.models.user import User, UserCreate, UserResponse, UserRole, UserStatus
from app.services.auth_service import AuthService
from app.services.user_service import UserService

settings = get_settings()
logger = get_logger(__name__)
security = HTTPBearer()

router = APIRouter()

# 认证服务实例
auth_service = AuthService()
user_service = UserService()


# Pydantic模式
class LoginRequest(BaseModel):
    """登录请求模式"""
    email: EmailStr
    password: str
    remember_me: bool = False


class LoginResponse(BaseModel):
    """登录响应模式"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    """刷新令牌请求模式"""
    refresh_token: str


class RefreshTokenResponse(BaseModel):
    """刷新令牌响应模式"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class ChangePasswordRequest(BaseModel):
    """修改密码请求模式"""
    current_password: str
    new_password: str


class ResetPasswordRequest(BaseModel):
    """重置密码请求模式"""
    email: EmailStr


class ConfirmResetPasswordRequest(BaseModel):
    """确认重置密码请求模式"""
    token: str
    new_password: str


# 依赖函数
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)) -> User:
    """获取当前用户"""
    try:
        token = credentials.credentials
        payload = verify_token(token, settings.jwt_secret_key)
        user_id = payload.get("sub")
        
        if user_id is None:
            raise AuthenticationException("无效的令牌")
        
        user = user_service.get_user_by_id(db, int(user_id))
        if user is None:
            raise AuthenticationException("用户不存在")
        
        if not user.is_active():
            raise AuthenticationException("用户账户已被禁用")
        
        return user
        
    except Exception as e:
        logger.error(f"获取当前用户失败: {e}")
        raise AuthenticationException("认证失败")


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """获取当前活跃用户"""
    if not current_user.is_active():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户账户已被禁用"
        )
    return current_user


@router.post("/register", response_model=dict, summary="用户注册")
async def register(
    user_data: UserCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    用户注册
    
    - **email**: 用户邮箱（必须唯一）
    - **username**: 用户名（必须唯一）
    - **password**: 密码（至少8位，包含大小写字母和数字）
    - **full_name**: 全名（可选）
    """
    try:
        # 检查用户是否已存在
        existing_user = user_service.get_user_by_email(db, user_data.email)
        if existing_user:
            raise ValidationException("该邮箱已被注册")
        
        existing_user = user_service.get_user_by_username(db, user_data.username)
        if existing_user:
            raise ValidationException("该用户名已被使用")
        
        # 创建用户
        user = user_service.create_user(db, user_data)
        
        # 记录注册日志
        logger.info(f"新用户注册: {user.email} (ID: {user.id})")
        
        return create_response(
            data={
                "user": UserResponse.from_orm(user),
                "message": "注册成功"
            },
            message="用户注册成功"
        )
        
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"用户注册失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="注册失败，请稍后重试"
        )


@router.post("/login", response_model=LoginResponse, summary="用户登录")
async def login(
    login_data: LoginRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    用户登录
    
    - **email**: 用户邮箱
    - **password**: 密码
    - **remember_me**: 是否记住登录状态（延长令牌有效期）
    """
    try:
        # 验证用户凭据
        user = auth_service.authenticate_user(db, login_data.email, login_data.password)
        if not user:
            raise AuthenticationException("邮箱或密码错误")
        
        if not user.is_active():
            raise AuthenticationException("账户已被禁用")
        
        # 生成令牌
        access_token_expires = timedelta(minutes=settings.jwt_access_token_expire_minutes)
        if login_data.remember_me:
            access_token_expires = timedelta(days=7)  # 记住登录状态时延长到7天
        
        access_token = create_access_token(
            data={"sub": str(user.id), "email": user.email},
            expires_delta=access_token_expires
        )
        
        refresh_token = create_refresh_token(
            data={"sub": str(user.id)},
            expires_delta=timedelta(days=settings.jwt_refresh_token_expire_days)
        )
        
        # 更新最后登录时间
        user.update_last_login()
        db.commit()
        
        # 记录登录日志
        logger.info(f"用户登录: {user.email} (ID: {user.id})")
        
        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int(access_token_expires.total_seconds()),
            user=UserResponse.from_orm(user)
        )
        
    except AuthenticationException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"用户登录失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登录失败，请稍后重试"
        )


@router.post("/refresh", response_model=RefreshTokenResponse, summary="刷新访问令牌")
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """
    刷新访问令牌
    
    - **refresh_token**: 刷新令牌
    """
    try:
        # 验证刷新令牌
        payload = verify_token(refresh_data.refresh_token, settings.jwt_secret_key)
        user_id = payload.get("sub")
        
        if user_id is None:
            raise AuthenticationException("无效的刷新令牌")
        
        user = user_service.get_user_by_id(db, int(user_id))
        if user is None or not user.is_active():
            raise AuthenticationException("用户不存在或已被禁用")
        
        # 生成新的访问令牌
        access_token_expires = timedelta(minutes=settings.jwt_access_token_expire_minutes)
        access_token = create_access_token(
            data={"sub": str(user.id), "email": user.email},
            expires_delta=access_token_expires
        )
        
        logger.info(f"令牌刷新: {user.email} (ID: {user.id})")
        
        return RefreshTokenResponse(
            access_token=access_token,
            expires_in=int(access_token_expires.total_seconds())
        )
        
    except AuthenticationException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"令牌刷新失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="令牌刷新失败，请重新登录"
        )


@router.post("/logout", summary="用户登出")
async def logout(
    current_user: User = Depends(get_current_active_user)
):
    """
    用户登出
    
    注意：由于使用JWT令牌，服务端无法直接使令牌失效。
    客户端应该删除本地存储的令牌。
    """
    logger.info(f"用户登出: {current_user.email} (ID: {current_user.id})")
    
    return create_response(
        message="登出成功",
        data={"message": "请删除客户端存储的令牌"}
    )


@router.post("/change-password", summary="修改密码")
async def change_password(
    password_data: ChangePasswordRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    修改密码
    
    - **current_password**: 当前密码
    - **new_password**: 新密码
    """
    try:
        # 验证当前密码
        if not current_user.verify_password(password_data.current_password):
            raise ValidationException("当前密码错误")
        
        # 更新密码
        current_user.set_password(password_data.new_password)
        db.commit()
        
        logger.info(f"密码修改: {current_user.email} (ID: {current_user.id})")
        
        return create_response(
            message="密码修改成功"
        )
        
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"密码修改失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="密码修改失败，请稍后重试"
        )


@router.get("/me", response_model=UserResponse, summary="获取当前用户信息")
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    获取当前用户信息
    """
    return UserResponse.from_orm(current_user)


@router.post("/verify-token", summary="验证令牌")
async def verify_user_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    验证访问令牌的有效性
    """
    try:
        token = credentials.credentials
        payload = verify_token(token, settings.jwt_secret_key)
        user_id = payload.get("sub")
        
        if user_id is None:
            raise AuthenticationException("无效的令牌")
        
        user = user_service.get_user_by_id(db, int(user_id))
        if user is None or not user.is_active():
            raise AuthenticationException("用户不存在或已被禁用")
        
        return create_response(
            data={
                "valid": True,
                "user_id": user.id,
                "email": user.email,
                "expires_at": payload.get("exp")
            },
            message="令牌有效"
        )
        
    except Exception as e:
        return create_response(
            data={"valid": False},
            message="令牌无效"
        )


# 密码重置功能（需要邮件服务支持）
@router.post("/reset-password", summary="请求重置密码")
async def request_password_reset(
    reset_data: ResetPasswordRequest,
    db: Session = Depends(get_db)
):
    """
    请求重置密码
    
    发送密码重置邮件到用户邮箱
    """
    try:
        user = user_service.get_user_by_email(db, reset_data.email)
        if user:
            # 生成重置令牌
            reset_token = create_access_token(
                data={"sub": str(user.id), "type": "password_reset"},
                expires_delta=timedelta(hours=1)  # 1小时有效期
            )
            
            # TODO: 发送重置邮件
            # await send_password_reset_email(user.email, reset_token)
            
            logger.info(f"密码重置请求: {user.email} (ID: {user.id})")
        
        # 无论用户是否存在都返回成功，避免邮箱枚举攻击
        return create_response(
            message="如果该邮箱已注册，您将收到密码重置邮件"
        )
        
    except Exception as e:
        logger.error(f"密码重置请求失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="请求失败，请稍后重试"
        )


@router.post("/confirm-reset-password", summary="确认重置密码")
async def confirm_password_reset(
    reset_data: ConfirmResetPasswordRequest,
    db: Session = Depends(get_db)
):
    """
    确认重置密码
    
    使用重置令牌设置新密码
    """
    try:
        # 验证重置令牌
        payload = verify_token(reset_data.token, settings.jwt_secret_key)
        user_id = payload.get("sub")
        token_type = payload.get("type")
        
        if user_id is None or token_type != "password_reset":
            raise ValidationException("无效的重置令牌")
        
        user = user_service.get_user_by_id(db, int(user_id))
        if user is None:
            raise ValidationException("用户不存在")
        
        # 更新密码
        user.set_password(reset_data.new_password)
        db.commit()
        
        logger.info(f"密码重置完成: {user.email} (ID: {user.id})")
        
        return create_response(
            message="密码重置成功，请使用新密码登录"
        )
        
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"密码重置确认失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="密码重置失败，请稍后重试"
        )