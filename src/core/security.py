#!/usr/bin/env python3
"""
安全模块

提供统一的安全功能，支持：
1. JWT认证
2. 密码哈希和验证
3. 权限控制
4. 数据加密
5. 安全头设置
6. CSRF保护
7. 限流保护
"""

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import bcrypt
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from passlib.context import CryptContext
from passlib.hash import argon2
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.requests import Request
from starlette.responses import Response

from .config import get_settings
from .logging import get_logger


# 配置日志
logger = get_logger(__name__)

# 密码上下文
pwd_context = CryptContext(
    schemes=["argon2", "bcrypt"],
    deprecated="auto",
    argon2__memory_cost=65536,
    argon2__time_cost=3,
    argon2__parallelism=1,
)


class SecurityManager:
    """
    安全管理器
    
    负责管理应用的安全功能
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._fernet = None
        self._rsa_private_key = None
        self._rsa_public_key = None
        self._initialized = False
    
    def initialize(self) -> None:
        """
        初始化安全管理器
        """
        if self._initialized:
            return
        
        # 初始化对称加密
        self._init_symmetric_encryption()
        
        # 初始化非对称加密
        self._init_asymmetric_encryption()
        
        self._initialized = True
        logger.info("Security manager initialized")
    
    def _init_symmetric_encryption(self) -> None:
        """
        初始化对称加密
        """
        # 使用配置的加密密钥或生成新的
        key = self.settings.encryption_key.encode()
        
        # 使用PBKDF2派生密钥
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'stable_salt',  # 在生产环境中应该使用随机盐
            iterations=100000,
        )
        derived_key = kdf.derive(key)
        
        # 创建Fernet实例
        import base64
        fernet_key = base64.urlsafe_b64encode(derived_key)
        self._fernet = Fernet(fernet_key)
    
    def _init_asymmetric_encryption(self) -> None:
        """
        初始化非对称加密
        """
        # 生成RSA密钥对
        self._rsa_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self._rsa_public_key = self._rsa_private_key.public_key()
    
    # =============================================================================
    # 密码相关功能
    # =============================================================================
    
    def hash_password(self, password: str) -> str:
        """
        哈希密码
        
        Args:
            password: 明文密码
        
        Returns:
            str: 哈希后的密码
        """
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        验证密码
        
        Args:
            plain_password: 明文密码
            hashed_password: 哈希密码
        
        Returns:
            bool: 是否匹配
        """
        return pwd_context.verify(plain_password, hashed_password)
    
    def generate_password(self, length: int = 12) -> str:
        """
        生成随机密码
        
        Args:
            length: 密码长度
        
        Returns:
            str: 随机密码
        """
        import string
        
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        
        # 确保包含各种字符类型
        if not any(c.islower() for c in password):
            password = password[:-1] + secrets.choice(string.ascii_lowercase)
        if not any(c.isupper() for c in password):
            password = password[:-1] + secrets.choice(string.ascii_uppercase)
        if not any(c.isdigit() for c in password):
            password = password[:-1] + secrets.choice(string.digits)
        if not any(c in "!@#$%^&*" for c in password):
            password = password[:-1] + secrets.choice("!@#$%^&*")
        
        return password
    
    def check_password_strength(self, password: str) -> Dict[str, Any]:
        """
        检查密码强度
        
        Args:
            password: 密码
        
        Returns:
            Dict[str, Any]: 密码强度信息
        """
        import re
        
        score = 0
        feedback = []
        
        # 长度检查
        if len(password) >= 8:
            score += 1
        else:
            feedback.append("密码长度至少8位")
        
        if len(password) >= 12:
            score += 1
        
        # 字符类型检查
        if re.search(r'[a-z]', password):
            score += 1
        else:
            feedback.append("需要包含小写字母")
        
        if re.search(r'[A-Z]', password):
            score += 1
        else:
            feedback.append("需要包含大写字母")
        
        if re.search(r'\d', password):
            score += 1
        else:
            feedback.append("需要包含数字")
        
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 1
        else:
            feedback.append("需要包含特殊字符")
        
        # 常见密码检查
        common_passwords = [
            "password", "123456", "123456789", "qwerty", "abc123",
            "password123", "admin", "letmein", "welcome", "monkey"
        ]
        
        if password.lower() in common_passwords:
            score = 0
            feedback.append("不能使用常见密码")
        
        # 评级
        if score >= 5:
            strength = "强"
        elif score >= 3:
            strength = "中"
        else:
            strength = "弱"
        
        return {
            "score": score,
            "max_score": 6,
            "strength": strength,
            "feedback": feedback,
            "is_strong": score >= 4,
        }
    
    # =============================================================================
    # JWT相关功能
    # =============================================================================
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        创建访问令牌
        
        Args:
            data: 令牌数据
            expires_delta: 过期时间
        
        Returns:
            str: JWT令牌
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.settings.jwt_access_token_expire_minutes
            )
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
        })
        
        return jwt.encode(
            to_encode,
            self.settings.jwt_secret_key,
            algorithm=self.settings.jwt_algorithm
        )
    
    def create_refresh_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        创建刷新令牌
        
        Args:
            data: 令牌数据
            expires_delta: 过期时间
        
        Returns:
            str: JWT令牌
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                days=self.settings.jwt_refresh_token_expire_days
            )
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
        })
        
        return jwt.encode(
            to_encode,
            self.settings.jwt_secret_key,
            algorithm=self.settings.jwt_algorithm
        )
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """
        验证JWT令牌
        
        Args:
            token: JWT令牌
            token_type: 令牌类型
        
        Returns:
            Dict[str, Any]: 令牌数据
        
        Raises:
            HTTPException: 令牌无效
        """
        try:
            payload = jwt.decode(
                token,
                self.settings.jwt_secret_key,
                algorithms=[self.settings.jwt_algorithm]
            )
            
            # 检查令牌类型
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type",
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
    
    # =============================================================================
    # 数据加密功能
    # =============================================================================
    
    def encrypt_data(self, data: str) -> str:
        """
        加密数据（对称加密）
        
        Args:
            data: 明文数据
        
        Returns:
            str: 加密后的数据
        """
        if not self._initialized:
            self.initialize()
        
        encrypted = self._fernet.encrypt(data.encode())
        return encrypted.decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """
        解密数据（对称加密）
        
        Args:
            encrypted_data: 加密数据
        
        Returns:
            str: 明文数据
        """
        if not self._initialized:
            self.initialize()
        
        decrypted = self._fernet.decrypt(encrypted_data.encode())
        return decrypted.decode()
    
    def encrypt_data_rsa(self, data: str) -> bytes:
        """
        加密数据（非对称加密）
        
        Args:
            data: 明文数据
        
        Returns:
            bytes: 加密后的数据
        """
        if not self._initialized:
            self.initialize()
        
        encrypted = self._rsa_public_key.encrypt(
            data.encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted
    
    def decrypt_data_rsa(self, encrypted_data: bytes) -> str:
        """
        解密数据（非对称加密）
        
        Args:
            encrypted_data: 加密数据
        
        Returns:
            str: 明文数据
        """
        if not self._initialized:
            self.initialize()
        
        decrypted = self._rsa_private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted.decode()
    
    # =============================================================================
    # 签名和验证功能
    # =============================================================================
    
    def create_signature(self, data: str, secret: Optional[str] = None) -> str:
        """
        创建HMAC签名
        
        Args:
            data: 要签名的数据
            secret: 签名密钥
        
        Returns:
            str: 签名
        """
        if secret is None:
            secret = self.settings.secret_key
        
        signature = hmac.new(
            secret.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_signature(self, data: str, signature: str, secret: Optional[str] = None) -> bool:
        """
        验证HMAC签名
        
        Args:
            data: 原始数据
            signature: 签名
            secret: 签名密钥
        
        Returns:
            bool: 是否验证通过
        """
        if secret is None:
            secret = self.settings.secret_key
        
        expected_signature = self.create_signature(data, secret)
        return hmac.compare_digest(signature, expected_signature)
    
    # =============================================================================
    # CSRF保护
    # =============================================================================
    
    def generate_csrf_token(self, session_id: str) -> str:
        """
        生成CSRF令牌
        
        Args:
            session_id: 会话ID
        
        Returns:
            str: CSRF令牌
        """
        timestamp = str(int(time.time()))
        data = f"{session_id}:{timestamp}"
        signature = self.create_signature(data, self.settings.csrf_secret)
        
        return f"{data}:{signature}"
    
    def verify_csrf_token(self, token: str, session_id: str, max_age: int = 3600) -> bool:
        """
        验证CSRF令牌
        
        Args:
            token: CSRF令牌
            session_id: 会话ID
            max_age: 最大有效期（秒）
        
        Returns:
            bool: 是否验证通过
        """
        try:
            parts = token.split(':')
            if len(parts) != 3:
                return False
            
            token_session_id, timestamp, signature = parts
            
            # 验证会话ID
            if token_session_id != session_id:
                return False
            
            # 验证时间戳
            token_time = int(timestamp)
            current_time = int(time.time())
            if current_time - token_time > max_age:
                return False
            
            # 验证签名
            data = f"{token_session_id}:{timestamp}"
            return self.verify_signature(data, signature, self.settings.csrf_secret)
            
        except (ValueError, IndexError):
            return False
    
    # =============================================================================
    # 安全工具函数
    # =============================================================================
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        生成安全令牌
        
        Args:
            length: 令牌长度
        
        Returns:
            str: 安全令牌
        """
        return secrets.token_urlsafe(length)
    
    def generate_otp(self, length: int = 6) -> str:
        """
        生成一次性密码
        
        Args:
            length: 密码长度
        
        Returns:
            str: 一次性密码
        """
        return ''.join(secrets.choice('0123456789') for _ in range(length))
    
    def hash_data(self, data: str, algorithm: str = "sha256") -> str:
        """
        哈希数据
        
        Args:
            data: 要哈希的数据
            algorithm: 哈希算法
        
        Returns:
            str: 哈希值
        """
        hash_func = getattr(hashlib, algorithm)
        return hash_func(data.encode()).hexdigest()
    
    def constant_time_compare(self, a: str, b: str) -> bool:
        """
        常量时间比较（防止时序攻击）
        
        Args:
            a: 字符串A
            b: 字符串B
        
        Returns:
            bool: 是否相等
        """
        return hmac.compare_digest(a, b)


class SecurityHeaders:
    """
    安全头管理器
    
    负责设置HTTP安全头
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    def add_security_headers(self, response: Response) -> Response:
        """
        添加安全头
        
        Args:
            response: HTTP响应
        
        Returns:
            Response: 添加安全头后的响应
        """
        if not self.settings.security_headers_enabled:
            return response
        
        # HSTS
        response.headers["Strict-Transport-Security"] = f"max-age={self.settings.hsts_max_age}; includeSubDomains"
        
        # CSP
        response.headers["Content-Security-Policy"] = self.settings.csp_policy
        
        # X-Frame-Options
        response.headers["X-Frame-Options"] = "DENY"
        
        # X-Content-Type-Options
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # X-XSS-Protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer-Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions-Policy
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


class RateLimiter:
    """
    限流器
    
    基于令牌桶算法的限流实现
    """
    
    def __init__(self, redis_client=None):
        self.settings = get_settings()
        self.redis_client = redis_client
    
    async def is_allowed(
        self,
        key: str,
        limit: int,
        window: int,
        identifier: str = "default"
    ) -> bool:
        """
        检查是否允许请求
        
        Args:
            key: 限流键
            limit: 限制次数
            window: 时间窗口（秒）
            identifier: 标识符
        
        Returns:
            bool: 是否允许
        """
        if not self.settings.rate_limit_enabled:
            return True
        
        if not self.redis_client:
            # 如果没有Redis，使用内存限流（仅适用于单实例）
            return await self._memory_rate_limit(key, limit, window)
        
        return await self._redis_rate_limit(key, limit, window)
    
    async def _redis_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """
        Redis限流
        
        Args:
            key: 限流键
            limit: 限制次数
            window: 时间窗口
        
        Returns:
            bool: 是否允许
        """
        current_time = int(time.time())
        pipeline = self.redis_client.pipeline()
        
        # 使用滑动窗口算法
        pipeline.zremrangebyscore(key, 0, current_time - window)
        pipeline.zcard(key)
        pipeline.zadd(key, {str(current_time): current_time})
        pipeline.expire(key, window)
        
        results = await pipeline.execute()
        current_requests = results[1]
        
        return current_requests < limit
    
    async def _memory_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """
        内存限流（简单实现）
        
        Args:
            key: 限流键
            limit: 限制次数
            window: 时间窗口
        
        Returns:
            bool: 是否允许
        """
        # 这里应该使用更复杂的内存存储，这只是示例
        import asyncio
        
        if not hasattr(self, '_memory_store'):
            self._memory_store = {}
        
        current_time = time.time()
        
        if key not in self._memory_store:
            self._memory_store[key] = []
        
        # 清理过期记录
        self._memory_store[key] = [
            timestamp for timestamp in self._memory_store[key]
            if current_time - timestamp < window
        ]
        
        # 检查是否超过限制
        if len(self._memory_store[key]) >= limit:
            return False
        
        # 添加当前请求
        self._memory_store[key].append(current_time)
        return True


# 全局安全管理器实例
security_manager = SecurityManager()
security_headers = SecurityHeaders()


# HTTP Bearer认证
class JWTBearer(HTTPBearer):
    """
    JWT Bearer认证
    """
    
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
    
    async def __call__(self, request: Request) -> Optional[str]:
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid authentication scheme."
                )
            
            if not self.verify_jwt(credentials.credentials):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid token or expired token."
                )
            
            return credentials.credentials
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid authorization code."
            )
    
    def verify_jwt(self, token: str) -> bool:
        """
        验证JWT令牌
        
        Args:
            token: JWT令牌
        
        Returns:
            bool: 是否有效
        """
        try:
            security_manager.verify_token(token)
            return True
        except HTTPException:
            return False


# 便捷函数
def hash_password(password: str) -> str:
    """
    哈希密码
    
    Args:
        password: 明文密码
    
    Returns:
        str: 哈希后的密码
    """
    return security_manager.hash_password(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码
    
    Args:
        plain_password: 明文密码
        hashed_password: 哈希密码
    
    Returns:
        bool: 是否匹配
    """
    return security_manager.verify_password(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    创建访问令牌
    
    Args:
        data: 令牌数据
        expires_delta: 过期时间
    
    Returns:
        str: JWT令牌
    """
    return security_manager.create_access_token(data, expires_delta)


def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """
    验证JWT令牌
    
    Args:
        token: JWT令牌
        token_type: 令牌类型
    
    Returns:
        Dict[str, Any]: 令牌数据
    """
    return security_manager.verify_token(token, token_type)


# 初始化函数
def init_security() -> None:
    """
    初始化安全管理器
    """
    security_manager.initialize()