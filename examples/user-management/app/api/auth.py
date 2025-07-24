#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户认证API

提供用户注册、登录、密码重置、双因子认证等功能。
"""

from typing import Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, validator

from ..core.database import get_db
from ..core.security import (
    verify_password, get_password_hash, create_access_token,
    create_refresh_token, verify_token
)
from ..models.user import User, LoginLog
from ..services.auth_service import AuthService
from ..services.email_service import EmailService
from ..core.config import settings

router = APIRouter(prefix="/auth", tags=["用户认证"])


class UserRegister(BaseModel):
    """用户注册请求模型"""
    username: str
    email: EmailStr
    password: str
    confirm_password: str
    full_name: Optional[str] = None
    phone: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3 or len(v) > 20:
            raise ValueError('用户名长度必须在3-20个字符之间')
        if not v.isalnum() and '_' not in v:
            raise ValueError('用户名只能包含字母、数字和下划线')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('密码长度至少8个字符')
        if not any(c.isupper() for c in v):
            raise ValueError('密码必须包含至少一个大写字母')
        if not any(c.islower() for c in v):
            raise ValueError('密码必须包含至少一个小写字母')
        if not any(c.isdigit() for c in v):
            raise ValueError('密码必须包含至少一个数字')
        return v
    
    @validator('confirm_password')
    def validate_confirm_password(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('两次输入的密码不一致')
        return v


class UserLogin(BaseModel):
    """用户登录请求模型"""
    username: str
    password: str
    remember_me: bool = False
    captcha: Optional[str] = None


class TokenResponse(BaseModel):
    """令牌响应模型"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_info: dict


class PasswordReset(BaseModel):
    """密码重置请求模型"""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """密码重置确认模型"""
    token: str
    new_password: str
    confirm_password: str
    
    @validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('密码长度至少8个字符')
        return v
    
    @validator('confirm_password')
    def validate_confirm_password(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('两次输入的密码不一致')
        return v


class TwoFactorSetup(BaseModel):
    """双因子认证设置模型"""
    enable: bool
    code: Optional[str] = None


@router.post("/register", response_model=dict)
async def register(
    user_data: UserRegister,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    用户注册
    
    创建新用户账户并发送验证邮件。
    """
    auth_service = AuthService(db)
    
    # 检查用户名是否已存在
    if auth_service.get_user_by_username(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已存在"
        )
    
    # 检查邮箱是否已存在
    if auth_service.get_user_by_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="邮箱已被注册"
        )
    
    # 创建用户
    user = auth_service.create_user(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password,
        full_name=user_data.full_name,
        phone=user_data.phone
    )
    
    # 发送验证邮件
    email_service = EmailService()
    verification_token = auth_service.create_verification_token(user.id)
    background_tasks.add_task(
        email_service.send_verification_email,
        user.email,
        user.username,
        verification_token
    )
    
    return {
        "message": "注册成功，请检查邮箱并验证账户",
        "user_id": user.id
    }


@router.post("/login", response_model=TokenResponse)
async def login(
    user_data: UserLogin,
    db: Session = Depends(get_db)
):
    """
    用户登录
    
    验证用户凭据并返回访问令牌。
    """
    auth_service = AuthService(db)
    
    # 验证用户凭据
    user = auth_service.authenticate_user(user_data.username, user_data.password)
    if not user:
        # 记录失败的登录尝试
        auth_service.log_login_attempt(
            username=user_data.username,
            success=False,
            ip_address="127.0.0.1"  # 实际应用中应获取真实IP
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误"
        )
    
    # 检查账户状态
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="账户已被禁用"
        )
    
    if not user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请先验证邮箱"
        )
    
    # 检查是否需要双因子认证
    if user.two_factor_enabled:
        # 这里应该返回需要2FA的响应
        return {
            "message": "需要双因子认证",
            "requires_2fa": True,
            "user_id": user.id
        }
    
    # 生成令牌
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    if user_data.remember_me:
        access_token_expires = timedelta(days=30)
    
    access_token = create_access_token(
        data={"sub": str(user.id), "username": user.username},
        expires_delta=access_token_expires
    )
    
    refresh_token = create_refresh_token(
        data={"sub": str(user.id)}
    )
    
    # 记录成功的登录
    auth_service.log_login_attempt(
        username=user_data.username,
        success=True,
        ip_address="127.0.0.1",
        user_id=user.id
    )
    
    # 更新最后登录时间
    auth_service.update_last_login(user.id)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=int(access_token_expires.total_seconds()),
        user_info={
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "is_admin": user.is_admin
        }
    )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    用户登出
    
    使当前令牌失效。
    """
    auth_service = AuthService(db)
    
    # 在实际应用中，这里应该将令牌加入黑名单
    # auth_service.blacklist_token(token)
    
    return {"message": "登出成功"}


@router.post("/refresh", response_model=dict)
async def refresh_token(
    refresh_token: str,
    db: Session = Depends(get_db)
):
    """
    刷新访问令牌
    
    使用刷新令牌获取新的访问令牌。
    """
    try:
        payload = verify_token(refresh_token)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的刷新令牌"
            )
        
        auth_service = AuthService(db)
        user = auth_service.get_user_by_id(int(user_id))
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户不存在或已被禁用"
            )
        
        # 生成新的访问令牌
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user.id), "username": user.username},
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": int(access_token_expires.total_seconds())
        }
        
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的刷新令牌"
        )


@router.post("/password-reset")
async def request_password_reset(
    reset_data: PasswordReset,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    请求密码重置
    
    发送密码重置邮件。
    """
    auth_service = AuthService(db)
    user = auth_service.get_user_by_email(reset_data.email)
    
    if user:
        # 生成重置令牌
        reset_token = auth_service.create_password_reset_token(user.id)
        
        # 发送重置邮件
        email_service = EmailService()
        background_tasks.add_task(
            email_service.send_password_reset_email,
            user.email,
            user.username,
            reset_token
        )
    
    # 无论用户是否存在都返回相同消息，避免邮箱枚举
    return {"message": "如果邮箱存在，重置链接已发送"}


@router.post("/password-reset/confirm")
async def confirm_password_reset(
    reset_data: PasswordResetConfirm,
    db: Session = Depends(get_db)
):
    """
    确认密码重置
    
    使用重置令牌设置新密码。
    """
    auth_service = AuthService(db)
    
    # 验证重置令牌
    user_id = auth_service.verify_password_reset_token(reset_data.token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无效或已过期的重置令牌"
        )
    
    # 更新密码
    auth_service.update_password(user_id, reset_data.new_password)
    
    return {"message": "密码重置成功"}


@router.get("/verify-email/{token}")
async def verify_email(
    token: str,
    db: Session = Depends(get_db)
):
    """
    验证邮箱
    
    使用验证令牌激活用户账户。
    """
    auth_service = AuthService(db)
    
    user_id = auth_service.verify_email_token(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无效或已过期的验证令牌"
        )
    
    # 激活用户账户
    auth_service.activate_user(user_id)
    
    return {"message": "邮箱验证成功，账户已激活"}


@router.post("/2fa/setup")
async def setup_two_factor(
    setup_data: TwoFactorSetup,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    设置双因子认证
    
    启用或禁用双因子认证。
    """
    auth_service = AuthService(db)
    
    if setup_data.enable:
        # 启用2FA
        if not setup_data.code:
            # 生成2FA密钥并返回二维码
            secret = auth_service.generate_2fa_secret(current_user.id)
            qr_code = auth_service.generate_2fa_qr_code(current_user.username, secret)
            return {
                "message": "请扫描二维码并输入验证码",
                "qr_code": qr_code,
                "secret": secret
            }
        else:
            # 验证代码并启用2FA
            if auth_service.verify_2fa_code(current_user.id, setup_data.code):
                auth_service.enable_2fa(current_user.id)
                return {"message": "双因子认证已启用"}
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="验证码错误"
                )
    else:
        # 禁用2FA
        auth_service.disable_2fa(current_user.id)
        return {"message": "双因子认证已禁用"}


@router.get("/me")
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    获取当前用户信息
    """
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "phone": current_user.phone,
        "is_admin": current_user.is_admin,
        "email_verified": current_user.email_verified,
        "two_factor_enabled": current_user.two_factor_enabled,
        "created_at": current_user.created_at,
        "last_login": current_user.last_login
    }