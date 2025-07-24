#!/usr/bin/env python3
"""
认证服务

提供用户认证相关的业务逻辑，包括：
1. 用户登录/登出
2. 令牌管理
3. 密码重置
4. 邮箱验证
5. 双因子认证
6. 登录尝试记录
"""

import secrets
import pyotp
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_
from pydantic import BaseModel

from .base import BaseService
from .user import UserService
from ..core.security import (
    create_access_token,
    create_refresh_token,
    verify_password,
    get_password_hash
)
from ..core.exceptions import (
    AuthenticationException,
    ValidationException,
    ResourceNotFoundException,
    ConflictException,
    RateLimitException
)
from ..models.auth import (
    AuthToken, RefreshToken, LoginAttempt, PasswordReset,
    EmailVerification, TwoFactorAuth, TokenType, LoginResult
)
from ..models.user import User, UserSession, SessionStatus
from ..utils.logger import get_logger
from ..utils.email import EmailService
from ..utils.security import generate_secure_token, get_client_info


# =============================================================================
# 认证服务类
# =============================================================================

class AuthService(BaseService[AuthToken, dict, dict]):
    """
    认证服务类
    
    提供用户认证相关的业务逻辑操作
    """
    
    def __init__(self, db: AsyncSession):
        """
        初始化认证服务
        
        Args:
            db: 数据库会话
        """
        super().__init__(db, AuthToken)
        self.logger = get_logger(__name__)
        self.user_service = UserService(db)
        self.email_service = EmailService()
        
        # 配置参数
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.token_expiry = {
            TokenType.ACCESS: timedelta(hours=1),
            TokenType.REFRESH: timedelta(days=30),
            TokenType.RESET: timedelta(hours=1),
            TokenType.VERIFICATION: timedelta(hours=24),
            TokenType.INVITATION: timedelta(days=7),
            TokenType.API: timedelta(days=365)
        }
    
    # =========================================================================
    # 用户登录/登出
    # =========================================================================
    
    async def login(
        self,
        username_or_email: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        remember_me: bool = False
    ) -> Dict[str, Any]:
        """
        用户登录
        
        Args:
            username_or_email: 用户名或邮箱
            password: 密码
            ip_address: IP地址
            user_agent: 用户代理
            remember_me: 是否记住登录
            
        Returns:
            登录结果，包含令牌信息
        """
        try:
            # 记录登录尝试开始时间
            attempt_start = datetime.utcnow()
            
            # 查找用户
            user = await self._find_user_by_login(username_or_email)
            
            # 检查账户状态
            if user:
                await self._check_account_status(user)
            
            # 验证密码
            if not user or not verify_password(password, user.password_hash):
                await self._record_failed_login(
                    username_or_email, ip_address, user_agent,
                    "invalid_credentials", user.id if user else None
                )
                
                # 如果用户存在，增加失败次数
                if user:
                    await self._handle_failed_login(user)
                
                raise AuthenticationException("用户名或密码错误")
            
            # 检查双因子认证
            two_factor = await self._get_user_two_factor(user.id)
            if two_factor and two_factor.is_enabled:
                # 返回需要双因子认证的标识
                return {
                    "requires_2fa": True,
                    "user_id": user.id,
                    "backup_codes_count": len(two_factor.backup_codes or [])
                }
            
            # 生成令牌
            tokens = await self._create_user_tokens(
                user, ip_address, user_agent, remember_me
            )
            
            # 创建用户会话
            session = await self._create_user_session(
                user, tokens["access_token"], ip_address, user_agent
            )
            
            # 更新用户登录信息
            await self._update_user_login_info(user)
            
            # 记录成功登录
            await self._record_successful_login(
                user, ip_address, user_agent
            )
            
            # 清理过期令牌
            await self._cleanup_expired_tokens(user.id)
            
            return {
                "access_token": tokens["access_token"],
                "refresh_token": tokens["refresh_token"],
                "token_type": "bearer",
                "expires_in": int(self.token_expiry[TokenType.ACCESS].total_seconds()),
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "display_name": user.display_name,
                    "avatar_url": user.avatar_url,
                    "is_superuser": user.is_superuser
                },
                "session_id": session.id
            }
            
        except Exception as e:
            # 记录失败登录（如果不是认证异常）
            if not isinstance(e, AuthenticationException):
                await self._record_failed_login(
                    username_or_email, ip_address, user_agent, "system_error"
                )
            
            self.logger.error(f"Login error for {username_or_email}: {e}")
            raise
    
    async def verify_2fa_and_complete_login(
        self,
        user_id: int,
        code: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        remember_me: bool = False
    ) -> Dict[str, Any]:
        """
        验证双因子认证并完成登录
        
        Args:
            user_id: 用户ID
            code: 验证码
            ip_address: IP地址
            user_agent: 用户代理
            remember_me: 是否记住登录
            
        Returns:
            登录结果
        """
        try:
            # 获取用户
            user = await self.user_service.get_by_id(user_id)
            if not user:
                raise ResourceNotFoundException("用户不存在")
            
            # 获取双因子认证配置
            two_factor = await self._get_user_two_factor(user_id)
            if not two_factor or not two_factor.is_enabled:
                raise ValidationException("双因子认证未启用")
            
            # 验证代码
            is_valid = await self._verify_2fa_code(two_factor, code)
            if not is_valid:
                await self._record_failed_login(
                    user.username, ip_address, user_agent, "invalid_2fa", user_id
                )
                raise AuthenticationException("验证码错误")
            
            # 更新双因子认证使用时间
            await self._update_2fa_last_used(two_factor.id)
            
            # 生成令牌
            tokens = await self._create_user_tokens(
                user, ip_address, user_agent, remember_me
            )
            
            # 创建用户会话
            session = await self._create_user_session(
                user, tokens["access_token"], ip_address, user_agent
            )
            
            # 更新用户登录信息
            await self._update_user_login_info(user)
            
            # 记录成功登录
            await self._record_successful_login(
                user, ip_address, user_agent
            )
            
            return {
                "access_token": tokens["access_token"],
                "refresh_token": tokens["refresh_token"],
                "token_type": "bearer",
                "expires_in": int(self.token_expiry[TokenType.ACCESS].total_seconds()),
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "display_name": user.display_name,
                    "avatar_url": user.avatar_url,
                    "is_superuser": user.is_superuser
                },
                "session_id": session.id
            }
            
        except Exception as e:
            self.logger.error(f"2FA verification error for user {user_id}: {e}")
            raise
    
    async def logout(
        self,
        access_token: str,
        user_id: Optional[int] = None
    ) -> bool:
        """
        用户登出
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID
            
        Returns:
            是否登出成功
        """
        try:
            # 撤销访问令牌
            await self._revoke_token(access_token, TokenType.ACCESS)
            
            # 撤销相关会话
            if user_id:
                await self._revoke_user_session_by_token(user_id, access_token)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Logout error: {e}")
            return False
    
    async def logout_all(
        self,
        user_id: int,
        exclude_current: bool = False,
        current_token: Optional[str] = None
    ) -> int:
        """
        登出所有设备
        
        Args:
            user_id: 用户ID
            exclude_current: 是否排除当前设备
            current_token: 当前令牌
            
        Returns:
            撤销的令牌数量
        """
        try:
            # 撤销所有访问令牌
            revoked_count = await self._revoke_user_tokens(
                user_id, exclude_token=current_token if exclude_current else None
            )
            
            # 撤销所有会话
            await self.user_service.revoke_all_sessions(
                user_id,
                exclude_session_id=None,
                revoked_by=user_id
            )
            
            return revoked_count
            
        except Exception as e:
            self.logger.error(f"Logout all error for user {user_id}: {e}")
            raise
    
    # =========================================================================
    # 令牌管理
    # =========================================================================
    
    async def refresh_token(
        self,
        refresh_token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        刷新访问令牌
        
        Args:
            refresh_token: 刷新令牌
            ip_address: IP地址
            user_agent: 用户代理
            
        Returns:
            新的令牌信息
        """
        try:
            # 验证刷新令牌
            token_record = await self._get_valid_token(refresh_token, TokenType.REFRESH)
            if not token_record:
                raise AuthenticationException("刷新令牌无效或已过期")
            
            # 获取用户
            user = await self.user_service.get_by_id(token_record.user_id)
            if not user:
                raise ResourceNotFoundException("用户不存在")
            
            # 检查账户状态
            await self._check_account_status(user)
            
            # 撤销旧的刷新令牌
            await self._revoke_token(refresh_token, TokenType.REFRESH)
            
            # 生成新令牌
            tokens = await self._create_user_tokens(
                user, ip_address, user_agent, remember_me=True
            )
            
            # 更新用户活动时间
            await self.user_service.update_user(user.id, {
                "last_activity_at": datetime.utcnow()
            })
            
            return {
                "access_token": tokens["access_token"],
                "refresh_token": tokens["refresh_token"],
                "token_type": "bearer",
                "expires_in": int(self.token_expiry[TokenType.ACCESS].total_seconds())
            }
            
        except Exception as e:
            self.logger.error(f"Token refresh error: {e}")
            raise
    
    async def validate_token(
        self,
        token: str,
        token_type: TokenType = TokenType.ACCESS
    ) -> Optional[AuthToken]:
        """
        验证令牌
        
        Args:
            token: 令牌
            token_type: 令牌类型
            
        Returns:
            令牌记录或None
        """
        return await self._get_valid_token(token, token_type)
    
    # =========================================================================
    # 密码重置
    # =========================================================================
    
    async def request_password_reset(
        self,
        email: str,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        请求密码重置
        
        Args:
            email: 邮箱地址
            ip_address: IP地址
            
        Returns:
            是否发送成功
        """
        try:
            # 查找用户
            user = await self.user_service.get_by_email(email)
            if not user:
                # 为了安全，即使用户不存在也返回成功
                return True
            
            # 检查重置频率限制
            await self._check_reset_rate_limit(user.id, ip_address)
            
            # 生成重置令牌
            reset_token = generate_secure_token(32)
            
            # 保存重置记录
            reset_record = PasswordReset(
                user_id=user.id,
                token=reset_token,
                email=email,
                expires_at=datetime.utcnow() + self.token_expiry[TokenType.RESET],
                ip_address=ip_address,
                created_at=datetime.utcnow()
            )
            
            self.db.add(reset_record)
            await self.db.commit()
            
            # 发送重置邮件
            await self.email_service.send_password_reset_email(
                email, user.display_name or user.username, reset_token
            )
            
            # 记录操作日志
            self.log_operation("request_password_reset", user.id, {
                "email": email,
                "ip_address": ip_address
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Password reset request error for {email}: {e}")
            # 为了安全，不暴露具体错误
            return True
    
    async def reset_password(
        self,
        token: str,
        new_password: str,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        重置密码
        
        Args:
            token: 重置令牌
            new_password: 新密码
            ip_address: IP地址
            
        Returns:
            是否重置成功
        """
        try:
            # 验证重置令牌
            reset_record = await self._get_valid_password_reset(token)
            if not reset_record:
                raise ValidationException("重置令牌无效或已过期")
            
            # 获取用户
            user = await self.user_service.get_by_id(reset_record.user_id)
            if not user:
                raise ResourceNotFoundException("用户不存在")
            
            # 更新密码
            await self.user_service.update_user(user.id, {
                "password": new_password,
                "failed_login_attempts": 0,
                "locked_until": None
            })
            
            # 标记重置令牌为已使用
            await self._mark_password_reset_used(reset_record.id, ip_address)
            
            # 撤销所有用户令牌和会话
            await self.logout_all(user.id)
            
            # 记录操作日志
            self.log_operation("reset_password", user.id, {
                "ip_address": ip_address
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Password reset error: {e}")
            raise
    
    # =========================================================================
    # 邮箱验证
    # =========================================================================
    
    async def send_email_verification(
        self,
        user_id: int,
        email: Optional[str] = None
    ) -> bool:
        """
        发送邮箱验证
        
        Args:
            user_id: 用户ID
            email: 邮箱地址（可选，默认使用用户邮箱）
            
        Returns:
            是否发送成功
        """
        try:
            # 获取用户
            user = await self.user_service.get_by_id(user_id)
            if not user:
                raise ResourceNotFoundException("用户不存在")
            
            target_email = email or user.email
            
            # 检查是否已验证
            if not email and user.email_verified_at:
                raise ConflictException("邮箱已验证")
            
            # 生成验证令牌
            verification_token = generate_secure_token(32)
            
            # 保存验证记录
            verification_record = EmailVerification(
                user_id=user_id,
                email=target_email,
                token=verification_token,
                verification_type="registration" if not email else "change",
                expires_at=datetime.utcnow() + self.token_expiry[TokenType.VERIFICATION],
                created_at=datetime.utcnow()
            )
            
            self.db.add(verification_record)
            await self.db.commit()
            
            # 发送验证邮件
            await self.email_service.send_email_verification(
                target_email, user.display_name or user.username, verification_token
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Email verification send error for user {user_id}: {e}")
            raise
    
    async def verify_email(
        self,
        token: str,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        验证邮箱
        
        Args:
            token: 验证令牌
            ip_address: IP地址
            
        Returns:
            是否验证成功
        """
        try:
            # 验证令牌
            verification_record = await self._get_valid_email_verification(token)
            if not verification_record:
                raise ValidationException("验证令牌无效或已过期")
            
            # 获取用户
            user = await self.user_service.get_by_id(verification_record.user_id)
            if not user:
                raise ResourceNotFoundException("用户不存在")
            
            # 更新用户邮箱验证状态
            update_data = {
                "email_verified_at": datetime.utcnow()
            }
            
            # 如果是邮箱变更验证
            if verification_record.verification_type == "change":
                update_data["email"] = verification_record.email
            
            await self.user_service.update_user(user.id, update_data)
            
            # 标记验证记录为已使用
            await self._mark_email_verification_used(verification_record.id, ip_address)
            
            # 记录操作日志
            self.log_operation("verify_email", user.id, {
                "email": verification_record.email,
                "verification_type": verification_record.verification_type,
                "ip_address": ip_address
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Email verification error: {e}")
            raise
    
    # =========================================================================
    # 双因子认证
    # =========================================================================
    
    async def setup_2fa(self, user_id: int) -> Dict[str, Any]:
        """
        设置双因子认证
        
        Args:
            user_id: 用户ID
            
        Returns:
            设置信息，包含二维码和备用代码
        """
        try:
            # 获取用户
            user = await self.user_service.get_by_id(user_id)
            if not user:
                raise ResourceNotFoundException("用户不存在")
            
            # 检查是否已启用
            existing_2fa = await self._get_user_two_factor(user_id)
            if existing_2fa and existing_2fa.is_enabled:
                raise ConflictException("双因子认证已启用")
            
            # 生成密钥
            secret = pyotp.random_base32()
            
            # 生成备用代码
            backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
            
            # 保存或更新双因子认证配置
            if existing_2fa:
                existing_2fa.secret = secret
                existing_2fa.backup_codes = backup_codes
                existing_2fa.created_at = datetime.utcnow()
            else:
                two_factor = TwoFactorAuth(
                    user_id=user_id,
                    secret=secret,
                    backup_codes=backup_codes,
                    is_enabled=False,
                    created_at=datetime.utcnow()
                )
                self.db.add(two_factor)
            
            await self.db.commit()
            
            # 生成二维码URL
            totp = pyotp.TOTP(secret)
            qr_url = totp.provisioning_uri(
                name=user.email,
                issuer_name="Your App Name"
            )
            
            return {
                "secret": secret,
                "qr_url": qr_url,
                "backup_codes": backup_codes
            }
            
        except Exception as e:
            self.logger.error(f"2FA setup error for user {user_id}: {e}")
            raise
    
    async def enable_2fa(
        self,
        user_id: int,
        verification_code: str
    ) -> bool:
        """
        启用双因子认证
        
        Args:
            user_id: 用户ID
            verification_code: 验证码
            
        Returns:
            是否启用成功
        """
        try:
            # 获取双因子认证配置
            two_factor = await self._get_user_two_factor(user_id)
            if not two_factor:
                raise ResourceNotFoundException("双因子认证配置不存在")
            
            if two_factor.is_enabled:
                raise ConflictException("双因子认证已启用")
            
            # 验证代码
            is_valid = await self._verify_2fa_code(two_factor, verification_code)
            if not is_valid:
                raise ValidationException("验证码错误")
            
            # 启用双因子认证
            two_factor.is_enabled = True
            two_factor.enabled_at = datetime.utcnow()
            
            await self.db.commit()
            
            # 记录操作日志
            self.log_operation("enable_2fa", user_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"2FA enable error for user {user_id}: {e}")
            raise
    
    async def disable_2fa(
        self,
        user_id: int,
        password: str
    ) -> bool:
        """
        禁用双因子认证
        
        Args:
            user_id: 用户ID
            password: 用户密码
            
        Returns:
            是否禁用成功
        """
        try:
            # 获取用户
            user = await self.user_service.get_by_id(user_id)
            if not user:
                raise ResourceNotFoundException("用户不存在")
            
            # 验证密码
            if not verify_password(password, user.password_hash):
                raise AuthenticationException("密码错误")
            
            # 获取双因子认证配置
            two_factor = await self._get_user_two_factor(user_id)
            if not two_factor or not two_factor.is_enabled:
                raise ResourceNotFoundException("双因子认证未启用")
            
            # 禁用双因子认证
            two_factor.is_enabled = False
            two_factor.disabled_at = datetime.utcnow()
            
            await self.db.commit()
            
            # 记录操作日志
            self.log_operation("disable_2fa", user_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"2FA disable error for user {user_id}: {e}")
            raise
    
    # =========================================================================
    # 私有辅助方法
    # =========================================================================
    
    async def _find_user_by_login(self, username_or_email: str) -> Optional[User]:
        """
        根据用户名或邮箱查找用户
        """
        if "@" in username_or_email:
            return await self.user_service.get_by_email(username_or_email)
        else:
            return await self.user_service.get_by_username(username_or_email)
    
    async def _check_account_status(self, user: User):
        """
        检查账户状态
        """
        if not user.is_active:
            raise AuthenticationException("账户已被禁用")
        
        if user.locked_until and user.locked_until > datetime.utcnow():
            raise AuthenticationException(f"账户已被锁定至 {user.locked_until}")
    
    async def _handle_failed_login(self, user: User):
        """
        处理登录失败
        """
        attempts = user.failed_login_attempts + 1
        update_data = {"failed_login_attempts": attempts}
        
        # 检查是否需要锁定账户
        if attempts >= self.max_login_attempts:
            update_data["locked_until"] = datetime.utcnow() + self.lockout_duration
        
        await self.user_service.update_user(user.id, update_data)
    
    async def _create_user_tokens(
        self,
        user: User,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        remember_me: bool = False
    ) -> Dict[str, str]:
        """
        创建用户令牌
        """
        # 生成访问令牌
        access_token = create_access_token(
            data={"sub": str(user.id), "username": user.username},
            expires_delta=self.token_expiry[TokenType.ACCESS]
        )
        
        # 生成刷新令牌
        refresh_expires = self.token_expiry[TokenType.REFRESH]
        if remember_me:
            refresh_expires = timedelta(days=90)  # 记住登录延长到90天
        
        refresh_token = create_refresh_token(
            data={"sub": str(user.id)},
            expires_delta=refresh_expires
        )
        
        # 保存令牌记录
        now = datetime.utcnow()
        
        access_token_record = AuthToken(
            user_id=user.id,
            token=access_token,
            token_type=TokenType.ACCESS,
            expires_at=now + self.token_expiry[TokenType.ACCESS],
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=now
        )
        
        refresh_token_record = AuthToken(
            user_id=user.id,
            token=refresh_token,
            token_type=TokenType.REFRESH,
            expires_at=now + refresh_expires,
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=now
        )
        
        self.db.add_all([access_token_record, refresh_token_record])
        await self.db.commit()
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token
        }
    
    async def _create_user_session(
        self,
        user: User,
        access_token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> UserSession:
        """
        创建用户会话
        """
        client_info = get_client_info(user_agent)
        
        session = UserSession(
            user_id=user.id,
            session_token=access_token,
            ip_address=ip_address,
            user_agent=user_agent,
            device_type=client_info.get("device_type"),
            browser=client_info.get("browser"),
            os=client_info.get("os"),
            location=None,  # TODO: 根据IP获取地理位置
            status=SessionStatus.ACTIVE,
            expires_at=datetime.utcnow() + self.token_expiry[TokenType.ACCESS],
            created_at=datetime.utcnow(),
            last_activity_at=datetime.utcnow()
        )
        
        self.db.add(session)
        await self.db.commit()
        
        return session
    
    async def _update_user_login_info(self, user: User):
        """
        更新用户登录信息
        """
        await self.user_service.update_user(user.id, {
            "last_login_at": datetime.utcnow(),
            "last_activity_at": datetime.utcnow(),
            "failed_login_attempts": 0,
            "locked_until": None
        })
    
    async def _record_successful_login(
        self,
        user: User,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        记录成功登录
        """
        login_attempt = LoginAttempt(
            user_id=user.id,
            username=user.username,
            email=user.email,
            ip_address=ip_address,
            user_agent=user_agent,
            result=LoginResult.SUCCESS,
            created_at=datetime.utcnow()
        )
        
        self.db.add(login_attempt)
        await self.db.commit()
    
    async def _record_failed_login(
        self,
        username_or_email: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        failure_reason: str = "invalid_credentials",
        user_id: Optional[int] = None
    ):
        """
        记录失败登录
        """
        login_attempt = LoginAttempt(
            user_id=user_id,
            username=username_or_email if "@" not in username_or_email else None,
            email=username_or_email if "@" in username_or_email else None,
            ip_address=ip_address,
            user_agent=user_agent,
            result=LoginResult.FAILED,
            failure_reason=failure_reason,
            created_at=datetime.utcnow()
        )
        
        self.db.add(login_attempt)
        await self.db.commit()
    
    async def _get_valid_token(
        self,
        token: str,
        token_type: TokenType
    ) -> Optional[AuthToken]:
        """
        获取有效令牌
        """
        try:
            query = (
                select(AuthToken)
                .where(AuthToken.token == token)
                .where(AuthToken.token_type == token_type)
                .where(AuthToken.is_revoked == False)
                .where(AuthToken.expires_at > datetime.utcnow())
            )
            
            result = await self.db.execute(query)
            return result.scalar_one_or_none()
            
        except Exception as e:
            self.logger.error(f"Error getting valid token: {e}")
            return None
    
    async def _revoke_token(self, token: str, token_type: TokenType) -> bool:
        """
        撤销令牌
        """
        try:
            update_stmt = (
                update(AuthToken)
                .where(AuthToken.token == token)
                .where(AuthToken.token_type == token_type)
                .values(
                    is_revoked=True,
                    revoked_at=datetime.utcnow()
                )
            )
            
            result = await self.db.execute(update_stmt)
            await self.db.commit()
            
            return result.rowcount > 0
            
        except Exception as e:
            self.logger.error(f"Error revoking token: {e}")
            return False
    
    async def _revoke_user_tokens(
        self,
        user_id: int,
        exclude_token: Optional[str] = None
    ) -> int:
        """
        撤销用户所有令牌
        """
        try:
            query = (
                update(AuthToken)
                .where(AuthToken.user_id == user_id)
                .where(AuthToken.is_revoked == False)
            )
            
            if exclude_token:
                query = query.where(AuthToken.token != exclude_token)
            
            query = query.values(
                is_revoked=True,
                revoked_at=datetime.utcnow()
            )
            
            result = await self.db.execute(query)
            await self.db.commit()
            
            return result.rowcount
            
        except Exception as e:
            self.logger.error(f"Error revoking user tokens: {e}")
            return 0
    
    async def _cleanup_expired_tokens(self, user_id: int):
        """
        清理过期令牌
        """
        try:
            delete_stmt = (
                delete(AuthToken)
                .where(AuthToken.user_id == user_id)
                .where(AuthToken.expires_at < datetime.utcnow())
            )
            
            await self.db.execute(delete_stmt)
            await self.db.commit()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired tokens: {e}")
    
    async def _get_user_two_factor(self, user_id: int) -> Optional[TwoFactorAuth]:
        """
        获取用户双因子认证配置
        """
        try:
            query = select(TwoFactorAuth).where(TwoFactorAuth.user_id == user_id)
            result = await self.db.execute(query)
            return result.scalar_one_or_none()
            
        except Exception as e:
            self.logger.error(f"Error getting user 2FA: {e}")
            return None
    
    async def _verify_2fa_code(
        self,
        two_factor: TwoFactorAuth,
        code: str
    ) -> bool:
        """
        验证双因子认证代码
        """
        try:
            # 验证TOTP代码
            totp = pyotp.TOTP(two_factor.secret)
            if totp.verify(code, valid_window=1):
                return True
            
            # 验证备用代码
            if two_factor.backup_codes and code.upper() in two_factor.backup_codes:
                # 移除已使用的备用代码
                two_factor.backup_codes.remove(code.upper())
                await self.db.commit()
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error verifying 2FA code: {e}")
            return False
    
    async def _update_2fa_last_used(self, two_factor_id: int):
        """
        更新双因子认证最后使用时间
        """
        try:
            update_stmt = (
                update(TwoFactorAuth)
                .where(TwoFactorAuth.id == two_factor_id)
                .values(last_used_at=datetime.utcnow())
            )
            
            await self.db.execute(update_stmt)
            await self.db.commit()
            
        except Exception as e:
            self.logger.error(f"Error updating 2FA last used: {e}")
    
    async def _check_reset_rate_limit(
        self,
        user_id: int,
        ip_address: Optional[str] = None
    ):
        """
        检查密码重置频率限制
        """
        try:
            # 检查用户重置频率（1小时内最多3次）
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            
            user_count_query = (
                select(func.count(PasswordReset.id))
                .where(PasswordReset.user_id == user_id)
                .where(PasswordReset.created_at >= one_hour_ago)
            )
            
            result = await self.db.execute(user_count_query)
            user_count = result.scalar()
            
            if user_count >= 3:
                raise RateLimitException("密码重置请求过于频繁，请稍后再试")
            
            # 检查IP重置频率（1小时内最多10次）
            if ip_address:
                ip_count_query = (
                    select(func.count(PasswordReset.id))
                    .where(PasswordReset.ip_address == ip_address)
                    .where(PasswordReset.created_at >= one_hour_ago)
                )
                
                result = await self.db.execute(ip_count_query)
                ip_count = result.scalar()
                
                if ip_count >= 10:
                    raise RateLimitException("该IP地址重置请求过于频繁，请稍后再试")
            
        except Exception as e:
            if not isinstance(e, RateLimitException):
                self.logger.error(f"Error checking reset rate limit: {e}")
            raise
    
    async def _get_valid_password_reset(self, token: str) -> Optional[PasswordReset]:
        """
        获取有效的密码重置记录
        """
        try:
            query = (
                select(PasswordReset)
                .where(PasswordReset.token == token)
                .where(PasswordReset.used_at.is_(None))
                .where(PasswordReset.expires_at > datetime.utcnow())
            )
            
            result = await self.db.execute(query)
            return result.scalar_one_or_none()
            
        except Exception as e:
            self.logger.error(f"Error getting valid password reset: {e}")
            return None
    
    async def _mark_password_reset_used(
        self,
        reset_id: int,
        ip_address: Optional[str] = None
    ):
        """
        标记密码重置为已使用
        """
        try:
            update_stmt = (
                update(PasswordReset)
                .where(PasswordReset.id == reset_id)
                .values(
                    used_at=datetime.utcnow(),
                    used_ip=ip_address
                )
            )
            
            await self.db.execute(update_stmt)
            await self.db.commit()
            
        except Exception as e:
            self.logger.error(f"Error marking password reset as used: {e}")
    
    async def _get_valid_email_verification(self, token: str) -> Optional[EmailVerification]:
        """
        获取有效的邮箱验证记录
        """
        try:
            query = (
                select(EmailVerification)
                .where(EmailVerification.token == token)
                .where(EmailVerification.verified_at.is_(None))
                .where(EmailVerification.expires_at > datetime.utcnow())
            )
            
            result = await self.db.execute(query)
            return result.scalar_one_or_none()
            
        except Exception as e:
            self.logger.error(f"Error getting valid email verification: {e}")
            return None
    
    async def _mark_email_verification_used(
        self,
        verification_id: int,
        ip_address: Optional[str] = None
    ):
        """
        标记邮箱验证为已使用
        """
        try:
            update_stmt = (
                update(EmailVerification)
                .where(EmailVerification.id == verification_id)
                .values(
                    verified_at=datetime.utcnow(),
                    verified_ip=ip_address
                )
            )
            
            await self.db.execute(update_stmt)
            await self.db.commit()
            
        except Exception as e:
            self.logger.error(f"Error marking email verification as used: {e}")
    
    async def _revoke_user_session_by_token(
        self,
        user_id: int,
        access_token: str
    ):
        """
        根据令牌撤销用户会话
        """
        try:
            update_stmt = (
                update(UserSession)
                .where(UserSession.user_id == user_id)
                .where(UserSession.session_token == access_token)
                .values(
                    status=SessionStatus.REVOKED,
                    revoked_at=datetime.utcnow()
                )
            )
            
            await self.db.execute(update_stmt)
            await self.db.commit()
            
        except Exception as e:
            self.logger.error(f"Error revoking user session by token: {e}")