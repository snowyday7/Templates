#!/usr/bin/env python3
"""
认证API端点

提供用户认证相关的API端点，包括：
1. 用户注册
2. 用户登录
3. 令牌刷新
4. 用户登出
5. 密码重置
6. 邮箱验证
7. 双因子认证
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, validator
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_db_session
from ...core.security import SecurityManager, JWTBearer
from ...core.cache import CacheManager
from ...core.responses import ResponseBuilder
from ...core.exceptions import (
    AuthenticationException,
    ValidationException,
    ResourceNotFoundException,
    ConflictException,
    RateLimitException
)
from ...models.user import User, UserStatus
from ...models.auth import (
    AuthToken, RefreshToken, LoginAttempt, PasswordReset,
    EmailVerification, TwoFactorAuth, LoginResult, TokenType,
    VerificationType
)
from ...services.auth import AuthService
from ...services.user import UserService
from ...services.email import EmailService


# =============================================================================
# 请求模型
# =============================================================================

class UserRegisterRequest(BaseModel):
    """用户注册请求"""
    username: str
    email: EmailStr
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3 or len(v) > 50:
            raise ValueError('用户名长度必须在3-50个字符之间')
        if not v.isalnum() and '_' not in v and '-' not in v:
            raise ValueError('用户名只能包含字母、数字、下划线和连字符')
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


class UserLoginRequest(BaseModel):
    """用户登录请求"""
    username: str
    password: str
    remember_me: bool = False
    two_factor_code: Optional[str] = None


class TokenRefreshRequest(BaseModel):
    """令牌刷新请求"""
    refresh_token: str


class PasswordResetRequest(BaseModel):
    """密码重置请求"""
    email: EmailStr


class PasswordResetConfirmRequest(BaseModel):
    """密码重置确认请求"""
    token: str
    new_password: str
    
    @validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('密码长度至少8个字符')
        return v


class EmailVerificationRequest(BaseModel):
    """邮箱验证请求"""
    token: str


class ResendVerificationRequest(BaseModel):
    """重发验证邮件请求"""
    email: EmailStr


class ChangePasswordRequest(BaseModel):
    """修改密码请求"""
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('密码长度至少8个字符')
        return v


class TwoFactorSetupRequest(BaseModel):
    """双因子认证设置请求"""
    password: str


class TwoFactorConfirmRequest(BaseModel):
    """双因子认证确认请求"""
    code: str
    backup_codes: bool = True


class TwoFactorDisableRequest(BaseModel):
    """禁用双因子认证请求"""
    password: str
    code: Optional[str] = None


# =============================================================================
# 响应模型
# =============================================================================

class TokenResponse(BaseModel):
    """令牌响应"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]


class UserResponse(BaseModel):
    """用户响应"""
    id: int
    username: str
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    display_name: Optional[str]
    avatar_url: Optional[str]
    is_active: bool
    is_verified: bool
    created_at: datetime


class TwoFactorSetupResponse(BaseModel):
    """双因子认证设置响应"""
    secret: str
    qr_code_url: str
    backup_codes: Optional[list] = None


# =============================================================================
# 路由器
# =============================================================================

router = APIRouter()
security_manager = SecurityManager()
jwt_bearer = JWTBearer()


# =============================================================================
# 辅助函数
# =============================================================================

def get_client_ip(request: Request) -> str:
    """获取客户端IP地址"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host


def get_user_agent(request: Request) -> str:
    """获取用户代理"""
    return request.headers.get("User-Agent", "")


async def record_login_attempt(
    db: AsyncSession,
    username: str,
    email: Optional[str],
    result: LoginResult,
    ip_address: str,
    user_agent: str,
    user_id: Optional[int] = None,
    failure_reason: Optional[str] = None
):
    """记录登录尝试"""
    attempt = LoginAttempt(
        user_id=user_id,
        username=username,
        email=email,
        ip_address=ip_address,
        user_agent=user_agent,
        result=result,
        failure_reason=failure_reason
    )
    db.add(attempt)
    await db.commit()


# =============================================================================
# API端点
# =============================================================================

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="用户注册",
    description="注册新用户账户"
)
async def register(
    request: UserRegisterRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    db: AsyncSession = Depends(get_db_session)
):
    """用户注册"""
    auth_service = AuthService(db)
    user_service = UserService(db)
    email_service = EmailService()
    
    # 检查用户名和邮箱是否已存在
    existing_user = await user_service.get_by_username_or_email(
        request.username, request.email
    )
    if existing_user:
        if existing_user.username == request.username:
            raise ConflictException("用户名已存在")
        if existing_user.email == request.email:
            raise ConflictException("邮箱已存在")
    
    # 创建用户
    user_data = request.dict()
    user = await user_service.create_user(**user_data)
    
    # 发送验证邮件
    verification_token = await auth_service.create_email_verification(
        user.id, user.email, VerificationType.ACCOUNT_ACTIVATION
    )
    
    background_tasks.add_task(
        email_service.send_verification_email,
        user.email,
        user.username,
        verification_token
    )
    
    return ResponseBuilder.success(
        message="注册成功，请检查邮箱并验证账户",
        data=UserResponse.from_orm(user)
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="用户登录",
    description="用户登录获取访问令牌"
)
async def login(
    request: UserLoginRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db_session)
):
    """用户登录"""
    auth_service = AuthService(db)
    user_service = UserService(db)
    
    ip_address = get_client_ip(http_request)
    user_agent = get_user_agent(http_request)
    
    try:
        # 检查登录限制
        await auth_service.check_login_rate_limit(ip_address)
        
        # 获取用户
        user = await user_service.get_by_username_or_email(request.username)
        if not user:
            await record_login_attempt(
                db, request.username, None, LoginResult.FAILED,
                ip_address, user_agent, failure_reason="用户不存在"
            )
            raise AuthenticationException("用户名或密码错误")
        
        # 检查账户状态
        if user.status == UserStatus.BANNED:
            await record_login_attempt(
                db, request.username, user.email, LoginResult.BANNED,
                ip_address, user_agent, user.id, "账户已被封禁"
            )
            raise AuthenticationException("账户已被封禁")
        
        if user.status == UserStatus.SUSPENDED:
            await record_login_attempt(
                db, request.username, user.email, LoginResult.SUSPENDED,
                ip_address, user_agent, user.id, "账户已被暂停"
            )
            raise AuthenticationException("账户已被暂停")
        
        if user.is_locked:
            await record_login_attempt(
                db, request.username, user.email, LoginResult.LOCKED,
                ip_address, user_agent, user.id, "账户已被锁定"
            )
            raise AuthenticationException("账户已被锁定")
        
        # 验证密码
        if not user.verify_password(request.password):
            user.increment_failed_login()
            
            # 检查是否需要锁定账户
            if user.failed_login_attempts >= 5:
                user.lock_account(30)  # 锁定30分钟
            
            await db.commit()
            
            await record_login_attempt(
                db, request.username, user.email, LoginResult.FAILED,
                ip_address, user_agent, user.id, "密码错误"
            )
            raise AuthenticationException("用户名或密码错误")
        
        # 检查双因子认证
        if user.two_factor_enabled:
            if not request.two_factor_code:
                await record_login_attempt(
                    db, request.username, user.email, LoginResult.TWO_FACTOR_REQUIRED,
                    ip_address, user_agent, user.id
                )
                raise AuthenticationException(
                    "需要双因子认证",
                    extra_data={"two_factor_required": True}
                )
            
            # 验证双因子认证码
            if not await auth_service.verify_two_factor_code(
                user.id, request.two_factor_code
            ):
                await record_login_attempt(
                    db, request.username, user.email, LoginResult.FAILED,
                    ip_address, user_agent, user.id, "双因子认证码错误"
                )
                raise AuthenticationException("双因子认证码错误")
        
        # 登录成功
        user.reset_failed_login()
        user.update_last_login()
        
        # 创建令牌
        expires_hours = 24 * 7 if request.remember_me else 24
        access_token = await auth_service.create_access_token(
            user.id, expires_hours=expires_hours
        )
        refresh_token = await auth_service.create_refresh_token(
            user.id, expires_hours=expires_hours * 2
        )
        
        await db.commit()
        
        await record_login_attempt(
            db, request.username, user.email, LoginResult.SUCCESS,
            ip_address, user_agent, user.id
        )
        
        return ResponseBuilder.success(
            message="登录成功",
            data=TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=expires_hours * 3600,
                user=user.to_dict()
            )
        )
    
    except AuthenticationException:
        raise
    except Exception as e:
        await record_login_attempt(
            db, request.username, None, LoginResult.FAILED,
            ip_address, user_agent, failure_reason=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登录过程中发生错误"
        )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="刷新令牌",
    description="使用刷新令牌获取新的访问令牌"
)
async def refresh_token(
    request: TokenRefreshRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """刷新令牌"""
    auth_service = AuthService(db)
    
    # 验证刷新令牌
    refresh_token_obj = await auth_service.verify_refresh_token(request.refresh_token)
    if not refresh_token_obj:
        raise AuthenticationException("无效的刷新令牌")
    
    user = refresh_token_obj.user
    if not user or not user.is_active:
        raise AuthenticationException("用户账户无效")
    
    # 创建新的访问令牌
    access_token = await auth_service.create_access_token(user.id)
    
    # 可选：创建新的刷新令牌（令牌轮换）
    new_refresh_token = await auth_service.create_refresh_token(user.id)
    
    # 撤销旧的刷新令牌
    refresh_token_obj.revoke()
    await db.commit()
    
    return ResponseBuilder.success(
        message="令牌刷新成功",
        data=TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=24 * 3600,
            user=user.to_dict()
        )
    )


@router.post(
    "/logout",
    summary="用户登出",
    description="用户登出并撤销令牌"
)
async def logout(
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """用户登出"""
    auth_service = AuthService(db)
    
    # 撤销用户的所有活跃令牌
    await auth_service.revoke_user_tokens(current_user.id)
    
    return ResponseBuilder.success(message="登出成功")


@router.post(
    "/password-reset",
    summary="请求密码重置",
    description="发送密码重置邮件"
)
async def request_password_reset(
    request: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session)
):
    """请求密码重置"""
    auth_service = AuthService(db)
    user_service = UserService(db)
    email_service = EmailService()
    
    user = await user_service.get_by_email(request.email)
    if not user:
        # 为了安全，即使用户不存在也返回成功
        return ResponseBuilder.success(
            message="如果邮箱存在，重置链接已发送"
        )
    
    # 创建密码重置令牌
    reset_token = await auth_service.create_password_reset_token(
        user.id, user.email
    )
    
    # 发送重置邮件
    background_tasks.add_task(
        email_service.send_password_reset_email,
        user.email,
        user.username,
        reset_token
    )
    
    return ResponseBuilder.success(
        message="如果邮箱存在，重置链接已发送"
    )


@router.post(
    "/password-reset/confirm",
    summary="确认密码重置",
    description="使用重置令牌设置新密码"
)
async def confirm_password_reset(
    request: PasswordResetConfirmRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """确认密码重置"""
    auth_service = AuthService(db)
    user_service = UserService(db)
    
    # 验证重置令牌
    reset_obj = await auth_service.verify_password_reset_token(request.token)
    if not reset_obj:
        raise ValidationException("无效或已过期的重置令牌")
    
    # 更新密码
    user = reset_obj.user
    user.set_password(request.new_password)
    
    # 标记令牌为已使用
    reset_obj.use_token()
    
    # 撤销用户的所有令牌
    await auth_service.revoke_user_tokens(user.id)
    
    await db.commit()
    
    return ResponseBuilder.success(message="密码重置成功")


@router.post(
    "/verify-email",
    summary="验证邮箱",
    description="使用验证令牌验证邮箱"
)
async def verify_email(
    request: EmailVerificationRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """验证邮箱"""
    auth_service = AuthService(db)
    
    # 验证邮箱令牌
    verification = await auth_service.verify_email_token(request.token)
    if not verification:
        raise ValidationException("无效或已过期的验证令牌")
    
    user = verification.user
    user.verify_email()
    
    # 标记验证为已完成
    verification.verify()
    
    await db.commit()
    
    return ResponseBuilder.success(
        message="邮箱验证成功",
        data=user.to_dict()
    )


@router.post(
    "/resend-verification",
    summary="重发验证邮件",
    description="重新发送邮箱验证邮件"
)
async def resend_verification(
    request: ResendVerificationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session)
):
    """重发验证邮件"""
    auth_service = AuthService(db)
    user_service = UserService(db)
    email_service = EmailService()
    
    user = await user_service.get_by_email(request.email)
    if not user:
        return ResponseBuilder.success(
            message="如果邮箱存在，验证邮件已发送"
        )
    
    if user.is_email_verified:
        return ResponseBuilder.success(message="邮箱已验证")
    
    # 创建新的验证令牌
    verification_token = await auth_service.create_email_verification(
        user.id, user.email, VerificationType.EMAIL
    )
    
    # 发送验证邮件
    background_tasks.add_task(
        email_service.send_verification_email,
        user.email,
        user.username,
        verification_token
    )
    
    return ResponseBuilder.success(
        message="如果邮箱存在，验证邮件已发送"
    )


@router.post(
    "/change-password",
    summary="修改密码",
    description="修改当前用户密码"
)
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """修改密码"""
    # 验证当前密码
    if not current_user.verify_password(request.current_password):
        raise AuthenticationException("当前密码错误")
    
    # 设置新密码
    current_user.set_password(request.new_password)
    
    # 撤销所有令牌（强制重新登录）
    auth_service = AuthService(db)
    await auth_service.revoke_user_tokens(current_user.id)
    
    await db.commit()
    
    return ResponseBuilder.success(message="密码修改成功，请重新登录")


@router.get(
    "/me",
    response_model=UserResponse,
    summary="获取当前用户信息",
    description="获取当前登录用户的信息"
)
async def get_current_user_info(
    current_user: User = Depends(jwt_bearer)
):
    """获取当前用户信息"""
    return ResponseBuilder.success(
        data=UserResponse.from_orm(current_user)
    )


# =============================================================================
# 双因子认证相关端点
# =============================================================================

@router.post(
    "/2fa/setup",
    response_model=TwoFactorSetupResponse,
    summary="设置双因子认证",
    description="开始设置双因子认证"
)
async def setup_two_factor(
    request: TwoFactorSetupRequest,
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """设置双因子认证"""
    # 验证密码
    if not current_user.verify_password(request.password):
        raise AuthenticationException("密码错误")
    
    if current_user.two_factor_enabled:
        raise ConflictException("双因子认证已启用")
    
    auth_service = AuthService(db)
    
    # 生成双因子认证密钥
    secret, qr_code_url = await auth_service.setup_two_factor(current_user.id)
    
    return ResponseBuilder.success(
        message="双因子认证设置成功，请使用认证应用扫描二维码",
        data=TwoFactorSetupResponse(
            secret=secret,
            qr_code_url=qr_code_url
        )
    )


@router.post(
    "/2fa/confirm",
    summary="确认双因子认证",
    description="确认并启用双因子认证"
)
async def confirm_two_factor(
    request: TwoFactorConfirmRequest,
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """确认双因子认证"""
    auth_service = AuthService(db)
    
    # 验证双因子认证码
    if not await auth_service.verify_two_factor_code(
        current_user.id, request.code
    ):
        raise ValidationException("验证码错误")
    
    # 启用双因子认证
    backup_codes = await auth_service.enable_two_factor(
        current_user.id, generate_backup_codes=request.backup_codes
    )
    
    await db.commit()
    
    response_data = {"message": "双因子认证已启用"}
    if backup_codes:
        response_data["backup_codes"] = backup_codes
    
    return ResponseBuilder.success(
        message="双因子认证已启用",
        data=response_data
    )


@router.post(
    "/2fa/disable",
    summary="禁用双因子认证",
    description="禁用双因子认证"
)
async def disable_two_factor(
    request: TwoFactorDisableRequest,
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """禁用双因子认证"""
    # 验证密码
    if not current_user.verify_password(request.password):
        raise AuthenticationException("密码错误")
    
    if not current_user.two_factor_enabled:
        raise ValidationException("双因子认证未启用")
    
    # 如果提供了验证码，验证它
    if request.code:
        auth_service = AuthService(db)
        if not await auth_service.verify_two_factor_code(
            current_user.id, request.code
        ):
            raise ValidationException("验证码错误")
    
    # 禁用双因子认证
    current_user.disable_two_factor()
    await db.commit()
    
    return ResponseBuilder.success(message="双因子认证已禁用")


@router.post(
    "/2fa/backup-codes",
    summary="生成新的备用代码",
    description="生成新的双因子认证备用代码"
)
async def generate_backup_codes(
    current_user: User = Depends(jwt_bearer),
    db: AsyncSession = Depends(get_db_session)
):
    """生成新的备用代码"""
    if not current_user.two_factor_enabled:
        raise ValidationException("双因子认证未启用")
    
    auth_service = AuthService(db)
    backup_codes = await auth_service.generate_backup_codes(current_user.id)
    
    await db.commit()
    
    return ResponseBuilder.success(
        message="新的备用代码已生成",
        data={"backup_codes": backup_codes}
    )