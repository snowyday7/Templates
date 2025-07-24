#!/usr/bin/env python3
"""
认证模型模块

定义认证相关的数据模型，包括：
1. 认证令牌模型
2. 刷新令牌模型
3. 登录尝试模型
4. 密码重置模型
5. 邮箱验证模型
6. 双因子认证模型
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum

from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Text, ForeignKey,
    Index, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func

from .base import BaseModel, TimestampMixin
from ..core.security import SecurityManager


# =============================================================================
# 枚举定义
# =============================================================================

class TokenType(str, Enum):
    """令牌类型枚举"""
    ACCESS = "access"          # 访问令牌
    REFRESH = "refresh"        # 刷新令牌
    RESET = "reset"            # 重置令牌
    VERIFY = "verify"          # 验证令牌
    INVITE = "invite"          # 邀请令牌
    API = "api"                # API令牌


class TokenStatus(str, Enum):
    """令牌状态枚举"""
    ACTIVE = "active"          # 活跃
    EXPIRED = "expired"        # 过期
    REVOKED = "revoked"        # 撤销
    USED = "used"              # 已使用


class LoginResult(str, Enum):
    """登录结果枚举"""
    SUCCESS = "success"        # 成功
    FAILED = "failed"          # 失败
    BLOCKED = "blocked"        # 被阻止
    LOCKED = "locked"          # 账户锁定
    SUSPENDED = "suspended"    # 账户暂停
    BANNED = "banned"          # 账户封禁
    TWO_FACTOR_REQUIRED = "two_factor_required"  # 需要双因子认证


class VerificationType(str, Enum):
    """验证类型枚举"""
    EMAIL = "email"            # 邮箱验证
    PHONE = "phone"            # 手机验证
    PASSWORD_RESET = "password_reset"  # 密码重置
    ACCOUNT_ACTIVATION = "account_activation"  # 账户激活
    TWO_FACTOR_SETUP = "two_factor_setup"  # 双因子认证设置


# =============================================================================
# 认证令牌模型
# =============================================================================

class AuthToken(BaseModel, TimestampMixin):
    """认证令牌模型"""
    
    __tablename__ = "auth_tokens"
    
    user_id = Column(
        Integer,
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        comment="用户ID"
    )
    
    token = Column(
        String(500),
        unique=True,
        nullable=False,
        index=True,
        comment="令牌值"
    )
    
    token_type = Column(
        SQLEnum(TokenType),
        nullable=False,
        index=True,
        comment="令牌类型"
    )
    
    status = Column(
        SQLEnum(TokenStatus),
        default=TokenStatus.ACTIVE,
        nullable=False,
        index=True,
        comment="令牌状态"
    )
    
    name = Column(
        String(100),
        nullable=True,
        comment="令牌名称（用于API令牌）"
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="令牌描述"
    )
    
    scopes = Column(
        Text,
        nullable=True,
        comment="权限范围（JSON格式）"
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="过期时间"
    )
    
    last_used_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="最后使用时间"
    )
    
    used_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="使用次数"
    )
    
    max_uses = Column(
        Integer,
        nullable=True,
        comment="最大使用次数"
    )
    
    ip_address = Column(
        String(45),
        nullable=True,
        comment="创建时的IP地址"
    )
    
    user_agent = Column(
        Text,
        nullable=True,
        comment="创建时的用户代理"
    )
    
    revoked_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="撤销时间"
    )
    
    revoked_by = Column(
        Integer,
        ForeignKey('users.id'),
        nullable=True,
        comment="撤销者ID"
    )
    
    revoke_reason = Column(
        String(200),
        nullable=True,
        comment="撤销原因"
    )
    
    # 关系定义
    user = relationship(
        "User",
        foreign_keys=[user_id],
        backref="auth_tokens"
    )
    
    revoker = relationship(
        "User",
        foreign_keys=[revoked_by]
    )
    
    # 索引定义
    __table_args__ = (
        Index('idx_auth_tokens_user_type', 'user_id', 'token_type'),
        Index('idx_auth_tokens_status_expires', 'status', 'expires_at'),
        Index('idx_auth_tokens_last_used', 'last_used_at'),
    )
    
    @hybrid_property
    def is_expired(self) -> bool:
        """是否已过期"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    @hybrid_property
    def is_valid(self) -> bool:
        """是否有效"""
        if self.status != TokenStatus.ACTIVE:
            return False
        
        if self.is_expired:
            return False
        
        if self.max_uses and self.used_count >= self.max_uses:
            return False
        
        return True
    
    @hybrid_property
    def remaining_uses(self) -> Optional[int]:
        """剩余使用次数"""
        if not self.max_uses:
            return None
        return max(0, self.max_uses - self.used_count)
    
    def use_token(self) -> bool:
        """
        使用令牌
        
        Returns:
            bool: 是否成功使用
        """
        if not self.is_valid:
            return False
        
        self.used_count += 1
        self.last_used_at = datetime.utcnow()
        
        # 检查是否达到最大使用次数
        if self.max_uses and self.used_count >= self.max_uses:
            self.status = TokenStatus.USED
        
        return True
    
    def revoke(self, revoked_by_user_id: Optional[int] = None, reason: Optional[str] = None) -> None:
        """
        撤销令牌
        
        Args:
            revoked_by_user_id: 撤销者用户ID
            reason: 撤销原因
        """
        self.status = TokenStatus.REVOKED
        self.revoked_at = datetime.utcnow()
        if revoked_by_user_id:
            self.revoked_by = revoked_by_user_id
        if reason:
            self.revoke_reason = reason
    
    def extend_expiry(self, hours: int = 24) -> None:
        """
        延长过期时间
        
        Args:
            hours: 延长的小时数
        """
        if self.expires_at:
            self.expires_at = max(self.expires_at, datetime.utcnow()) + timedelta(hours=hours)
        else:
            self.expires_at = datetime.utcnow() + timedelta(hours=hours)
    
    def to_dict(self, include_token: bool = False) -> Dict[str, Any]:
        """
        转换为字典
        
        Args:
            include_token: 是否包含令牌值
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        exclude = [] if include_token else ['token']
        data = super().to_dict(exclude=exclude)
        
        data.update({
            'is_expired': self.is_expired,
            'is_valid': self.is_valid,
            'remaining_uses': self.remaining_uses,
        })
        
        return data


# =============================================================================
# 刷新令牌模型
# =============================================================================

class RefreshToken(BaseModel, TimestampMixin):
    """刷新令牌模型"""
    
    __tablename__ = "refresh_tokens"
    
    user_id = Column(
        Integer,
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        comment="用户ID"
    )
    
    token = Column(
        String(500),
        unique=True,
        nullable=False,
        index=True,
        comment="刷新令牌"
    )
    
    access_token_id = Column(
        Integer,
        ForeignKey('auth_tokens.id', ondelete='CASCADE'),
        nullable=True,
        comment="关联的访问令牌ID"
    )
    
    status = Column(
        SQLEnum(TokenStatus),
        default=TokenStatus.ACTIVE,
        nullable=False,
        index=True,
        comment="令牌状态"
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="过期时间"
    )
    
    last_used_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="最后使用时间"
    )
    
    used_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="使用次数"
    )
    
    ip_address = Column(
        String(45),
        nullable=True,
        comment="创建时的IP地址"
    )
    
    user_agent = Column(
        Text,
        nullable=True,
        comment="创建时的用户代理"
    )
    
    revoked_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="撤销时间"
    )
    
    # 关系定义
    user = relationship(
        "User",
        backref="refresh_tokens"
    )
    
    access_token = relationship(
        "AuthToken",
        backref="refresh_token"
    )
    
    # 索引定义
    __table_args__ = (
        Index('idx_refresh_tokens_user_status', 'user_id', 'status'),
        Index('idx_refresh_tokens_expires_at', 'expires_at'),
    )
    
    @hybrid_property
    def is_expired(self) -> bool:
        """是否已过期"""
        return datetime.utcnow() > self.expires_at
    
    @hybrid_property
    def is_valid(self) -> bool:
        """是否有效"""
        return self.status == TokenStatus.ACTIVE and not self.is_expired
    
    def use_token(self) -> bool:
        """
        使用刷新令牌
        
        Returns:
            bool: 是否成功使用
        """
        if not self.is_valid:
            return False
        
        self.used_count += 1
        self.last_used_at = datetime.utcnow()
        return True
    
    def revoke(self) -> None:
        """撤销刷新令牌"""
        self.status = TokenStatus.REVOKED
        self.revoked_at = datetime.utcnow()
    
    def to_dict(self, include_token: bool = False) -> Dict[str, Any]:
        """
        转换为字典
        
        Args:
            include_token: 是否包含令牌值
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        exclude = [] if include_token else ['token']
        data = super().to_dict(exclude=exclude)
        
        data.update({
            'is_expired': self.is_expired,
            'is_valid': self.is_valid,
        })
        
        return data


# =============================================================================
# 登录尝试模型
# =============================================================================

class LoginAttempt(BaseModel, TimestampMixin):
    """登录尝试模型"""
    
    __tablename__ = "login_attempts"
    
    user_id = Column(
        Integer,
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=True,
        index=True,
        comment="用户ID（如果用户存在）"
    )
    
    username = Column(
        String(255),
        nullable=False,
        index=True,
        comment="尝试登录的用户名"
    )
    
    email = Column(
        String(255),
        nullable=True,
        index=True,
        comment="尝试登录的邮箱"
    )
    
    ip_address = Column(
        String(45),
        nullable=False,
        index=True,
        comment="IP地址"
    )
    
    user_agent = Column(
        Text,
        nullable=True,
        comment="用户代理"
    )
    
    result = Column(
        SQLEnum(LoginResult),
        nullable=False,
        index=True,
        comment="登录结果"
    )
    
    failure_reason = Column(
        String(200),
        nullable=True,
        comment="失败原因"
    )
    
    location = Column(
        String(200),
        nullable=True,
        comment="地理位置"
    )
    
    device_fingerprint = Column(
        String(255),
        nullable=True,
        comment="设备指纹"
    )
    
    session_id = Column(
        String(255),
        nullable=True,
        comment="会话ID（成功登录时）"
    )
    
    two_factor_used = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="是否使用了双因子认证"
    )
    
    # 关系定义
    user = relationship(
        "User",
        backref="login_attempts"
    )
    
    # 索引定义
    __table_args__ = (
        Index('idx_login_attempts_ip_time', 'ip_address', 'created_at'),
        Index('idx_login_attempts_user_time', 'user_id', 'created_at'),
        Index('idx_login_attempts_result_time', 'result', 'created_at'),
        Index('idx_login_attempts_username_time', 'username', 'created_at'),
    )
    
    @property
    def is_successful(self) -> bool:
        """是否成功登录"""
        return self.result == LoginResult.SUCCESS
    
    @property
    def is_suspicious(self) -> bool:
        """是否可疑"""
        # 可以根据业务需求定义可疑行为
        suspicious_results = [
            LoginResult.BLOCKED,
            LoginResult.FAILED
        ]
        return self.result in suspicious_results
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        data = super().to_dict()
        data.update({
            'is_successful': self.is_successful,
            'is_suspicious': self.is_suspicious,
        })
        return data


# =============================================================================
# 密码重置模型
# =============================================================================

class PasswordReset(BaseModel, TimestampMixin):
    """密码重置模型"""
    
    __tablename__ = "password_resets"
    
    user_id = Column(
        Integer,
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        comment="用户ID"
    )
    
    token = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="重置令牌"
    )
    
    email = Column(
        String(255),
        nullable=False,
        comment="重置邮箱"
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="过期时间"
    )
    
    used_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="使用时间"
    )
    
    ip_address = Column(
        String(45),
        nullable=True,
        comment="请求IP地址"
    )
    
    user_agent = Column(
        Text,
        nullable=True,
        comment="用户代理"
    )
    
    # 关系定义
    user = relationship(
        "User",
        backref="password_resets"
    )
    
    # 索引定义
    __table_args__ = (
        Index('idx_password_resets_user_expires', 'user_id', 'expires_at'),
        Index('idx_password_resets_email_expires', 'email', 'expires_at'),
    )
    
    @hybrid_property
    def is_expired(self) -> bool:
        """是否已过期"""
        return datetime.utcnow() > self.expires_at
    
    @hybrid_property
    def is_used(self) -> bool:
        """是否已使用"""
        return self.used_at is not None
    
    @hybrid_property
    def is_valid(self) -> bool:
        """是否有效"""
        return not self.is_expired and not self.is_used
    
    def use_token(self) -> bool:
        """
        使用重置令牌
        
        Returns:
            bool: 是否成功使用
        """
        if not self.is_valid:
            return False
        
        self.used_at = datetime.utcnow()
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        data = super().to_dict(exclude=['token'])
        data.update({
            'is_expired': self.is_expired,
            'is_used': self.is_used,
            'is_valid': self.is_valid,
        })
        return data


# =============================================================================
# 邮箱验证模型
# =============================================================================

class EmailVerification(BaseModel, TimestampMixin):
    """邮箱验证模型"""
    
    __tablename__ = "email_verifications"
    
    user_id = Column(
        Integer,
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        comment="用户ID"
    )
    
    email = Column(
        String(255),
        nullable=False,
        index=True,
        comment="待验证邮箱"
    )
    
    token = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="验证令牌"
    )
    
    verification_type = Column(
        SQLEnum(VerificationType),
        nullable=False,
        comment="验证类型"
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="过期时间"
    )
    
    verified_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="验证时间"
    )
    
    attempts = Column(
        Integer,
        default=0,
        nullable=False,
        comment="尝试次数"
    )
    
    max_attempts = Column(
        Integer,
        default=3,
        nullable=False,
        comment="最大尝试次数"
    )
    
    ip_address = Column(
        String(45),
        nullable=True,
        comment="请求IP地址"
    )
    
    # 关系定义
    user = relationship(
        "User",
        backref="email_verifications"
    )
    
    # 索引定义
    __table_args__ = (
        Index('idx_email_verifications_user_type', 'user_id', 'verification_type'),
        Index('idx_email_verifications_email_expires', 'email', 'expires_at'),
    )
    
    @hybrid_property
    def is_expired(self) -> bool:
        """是否已过期"""
        return datetime.utcnow() > self.expires_at
    
    @hybrid_property
    def is_verified(self) -> bool:
        """是否已验证"""
        return self.verified_at is not None
    
    @hybrid_property
    def is_valid(self) -> bool:
        """是否有效"""
        return (
            not self.is_expired and
            not self.is_verified and
            self.attempts < self.max_attempts
        )
    
    def verify(self) -> bool:
        """
        验证邮箱
        
        Returns:
            bool: 是否成功验证
        """
        if not self.is_valid:
            return False
        
        self.verified_at = datetime.utcnow()
        return True
    
    def increment_attempts(self) -> None:
        """增加尝试次数"""
        self.attempts += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        data = super().to_dict(exclude=['token'])
        data.update({
            'is_expired': self.is_expired,
            'is_verified': self.is_verified,
            'is_valid': self.is_valid,
            'remaining_attempts': max(0, self.max_attempts - self.attempts),
        })
        return data


# =============================================================================
# 双因子认证模型
# =============================================================================

class TwoFactorAuth(BaseModel, TimestampMixin):
    """双因子认证模型"""
    
    __tablename__ = "two_factor_auth"
    
    user_id = Column(
        Integer,
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        comment="用户ID"
    )
    
    secret = Column(
        String(255),
        nullable=False,
        comment="双因子认证密钥"
    )
    
    backup_codes = Column(
        Text,
        nullable=True,
        comment="备用代码（JSON格式）"
    )
    
    is_enabled = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="是否启用"
    )
    
    enabled_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="启用时间"
    )
    
    last_used_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="最后使用时间"
    )
    
    recovery_codes_generated_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="恢复代码生成时间"
    )
    
    # 关系定义
    user = relationship(
        "User",
        backref="two_factor_auth"
    )
    
    # 索引定义
    __table_args__ = (
        Index('idx_two_factor_auth_user_enabled', 'user_id', 'is_enabled'),
    )
    
    def enable(self) -> None:
        """启用双因子认证"""
        self.is_enabled = True
        self.enabled_at = datetime.utcnow()
    
    def disable(self) -> None:
        """禁用双因子认证"""
        self.is_enabled = False
        self.enabled_at = None
    
    def update_last_used(self) -> None:
        """更新最后使用时间"""
        self.last_used_at = datetime.utcnow()
    
    def generate_backup_codes(self) -> List[str]:
        """
        生成备用代码
        
        Returns:
            List[str]: 备用代码列表
        """
        import json
        import secrets
        
        codes = [secrets.token_hex(4).upper() for _ in range(10)]
        self.backup_codes = json.dumps(codes)
        self.recovery_codes_generated_at = datetime.utcnow()
        return codes
    
    def verify_backup_code(self, code: str) -> bool:
        """
        验证备用代码
        
        Args:
            code: 备用代码
        
        Returns:
            bool: 是否验证成功
        """
        if not self.backup_codes:
            return False
        
        import json
        
        try:
            codes = json.loads(self.backup_codes)
            if code.upper() in codes:
                # 移除已使用的代码
                codes.remove(code.upper())
                self.backup_codes = json.dumps(codes)
                self.update_last_used()
                return True
        except (json.JSONDecodeError, ValueError):
            pass
        
        return False
    
    def to_dict(self, include_secret: bool = False) -> Dict[str, Any]:
        """
        转换为字典
        
        Args:
            include_secret: 是否包含密钥
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        exclude = [] if include_secret else ['secret', 'backup_codes']
        data = super().to_dict(exclude=exclude)
        
        if not include_secret and self.backup_codes:
            import json
            try:
                codes = json.loads(self.backup_codes)
                data['backup_codes_count'] = len(codes)
            except (json.JSONDecodeError, ValueError):
                data['backup_codes_count'] = 0
        
        return data