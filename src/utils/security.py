#!/usr/bin/env python3
"""
安全工具模块

提供密码哈希、令牌生成、数据加密等安全功能
"""

import hashlib
import hmac
import secrets
import base64
from typing import Optional, Dict, Any, Tuple

try:
    import pyotp
    HAS_PYOTP = True
except ImportError:
    HAS_PYOTP = False
    pyotp = None
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from passlib.context import CryptContext
import jwt

import os
from .logger import get_logger


# =============================================================================
# 配置和初始化
# =============================================================================

logger = get_logger(__name__)

# 密码上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT配置
JWT_ALGORITHM = "HS256"
JWT_SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7

# 加密密钥
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode()).encode()
cipher_suite = Fernet(ENCRYPTION_KEY)


# =============================================================================
# 密码相关
# =============================================================================

def hash_password(password: str) -> str:
    """
    哈希密码

    Args:
        password: 明文密码

    Returns:
        哈希后的密码
    """
    try:
        return pwd_context.hash(password)
    except Exception as e:
        logger.error(f"Error hashing password: {e}")
        raise


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码

    Args:
        plain_password: 明文密码
        hashed_password: 哈希密码

    Returns:
        是否匹配
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False


def generate_password_reset_token(user_id: int, expires_delta: Optional[timedelta] = None) -> str:
    """
    生成密码重置令牌

    Args:
        user_id: 用户ID
        expires_delta: 过期时间

    Returns:
        重置令牌
    """
    try:
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=30)

        to_encode = {
            "user_id": user_id,
            "exp": expire,
            "type": "password_reset"
        }

        encoded_jwt = jwt.encode(
            to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM
        )
        return encoded_jwt

    except Exception as e:
        logger.error(f"Error generating password reset token: {e}")
        raise


def verify_password_reset_token(token: str) -> Optional[int]:
    """
    验证密码重置令牌

    Args:
        token: 重置令牌

    Returns:
        用户ID，如果无效则返回None
    """
    try:
        payload = jwt.decode(
            token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM]
        )

        if payload.get("type") != "password_reset":
            return None

        user_id: int = payload.get("user_id")
        return user_id

    except jwt.ExpiredSignatureError:
        logger.warning("Password reset token expired")
        return None
    except jwt.JWTError as e:
        logger.warning(f"Invalid password reset token: {e}")
        return None


# =============================================================================
# JWT令牌
# =============================================================================

def generate_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
    token_type: str = "access"
) -> str:
    """
    生成JWT令牌

    Args:
        data: 要编码的数据
        expires_delta: 过期时间
        token_type: 令牌类型

    Returns:
        JWT令牌
    """
    try:
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            if token_type == "refresh":
                expire = datetime.utcnow() + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
            else:
                expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": token_type
        })

        encoded_jwt = jwt.encode(
            to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM
        )
        return encoded_jwt

    except Exception as e:
        logger.error(f"Error generating token: {e}")
        raise


def verify_token(token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
    """
    验证JWT令牌

    Args:
        token: JWT令牌
        token_type: 令牌类型

    Returns:
        解码后的数据，如果无效则返回None
    """
    try:
        payload = jwt.decode(
            token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM]
        )

        if payload.get("type") != token_type:
            return None

        return payload

    except jwt.ExpiredSignatureError:
        logger.warning(f"{token_type} token expired")
        return None
    except jwt.JWTError as e:
        logger.warning(f"Invalid {token_type} token: {e}")
        return None


def generate_access_token(user_id: int, permissions: Optional[list] = None) -> str:
    """
    生成访问令牌

    Args:
        user_id: 用户ID
        permissions: 用户权限

    Returns:
        访问令牌
    """
    data = {
        "user_id": user_id,
        "permissions": permissions or []
    }
    return generate_token(data, token_type="access")


def generate_refresh_token(user_id: int) -> str:
    """
    生成刷新令牌

    Args:
        user_id: 用户ID

    Returns:
        刷新令牌
    """
    data = {"user_id": user_id}
    return generate_token(data, token_type="refresh")


# =============================================================================
# OTP（一次性密码）
# =============================================================================

def generate_otp_secret() -> str:
    """
    生成OTP密钥

    Returns:
        Base32编码的密钥
    """
    if not HAS_PYOTP:
        raise ImportError("pyotp is required for OTP functionality. Install with: pip install pyotp")
    return pyotp.random_base32()


def generate_otp(secret: str) -> str:
    """
    生成OTP代码

    Args:
        secret: OTP密钥

    Returns:
        6位OTP代码
    """
    if not HAS_PYOTP:
        raise ImportError("pyotp is required for OTP functionality. Install with: pip install pyotp")
    try:
        totp = pyotp.TOTP(secret)
        return totp.now()
    except Exception as e:
        logger.error(f"Error generating OTP: {e}")
        raise


def verify_otp(secret: str, token: str, window: int = 1) -> bool:
    """
    验证OTP代码

    Args:
        secret: OTP密钥
        token: OTP代码
        window: 时间窗口（允许的时间偏差）

    Returns:
        是否有效
    """
    if not HAS_PYOTP:
        logger.error("pyotp is required for OTP functionality")
        return False
    try:
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=window)
    except Exception as e:
        logger.error(f"Error verifying OTP: {e}")
        return False


def generate_backup_codes(count: int = 10) -> list:
    """
    生成备用代码

    Args:
        count: 生成数量

    Returns:
        备用代码列表
    """
    codes = []
    for _ in range(count):
        code = secrets.token_hex(4).upper()  # 8位十六进制代码
        codes.append(code)
    return codes


# =============================================================================
# 数据加密
# =============================================================================

def encrypt_data(data: str) -> str:
    """
    加密数据

    Args:
        data: 要加密的数据

    Returns:
        加密后的数据（Base64编码）
    """
    try:
        encrypted_data = cipher_suite.encrypt(data.encode())
        return base64.b64encode(encrypted_data).decode()
    except Exception as e:
        logger.error(f"Error encrypting data: {e}")
        raise


def decrypt_data(encrypted_data: str) -> str:
    """
    解密数据

    Args:
        encrypted_data: 加密的数据（Base64编码）

    Returns:
        解密后的数据
    """
    try:
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted_data = cipher_suite.decrypt(encrypted_bytes)
        return decrypted_data.decode()
    except Exception as e:
        logger.error(f"Error decrypting data: {e}")
        raise


def generate_encryption_key() -> bytes:
    """
    生成加密密钥

    Returns:
        加密密钥
    """
    return Fernet.generate_key()


def derive_key_from_password(password: str, salt: bytes) -> bytes:
    """
    从密码派生密钥

    Args:
        password: 密码
        salt: 盐值

    Returns:
        派生的密钥
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


# =============================================================================
# 随机数生成
# =============================================================================

def generate_random_string(length: int = 32, use_digits: bool = True, use_letters: bool = True) -> str:
    """
    生成随机字符串

    Args:
        length: 字符串长度
        use_digits: 是否包含数字
        use_letters: 是否包含字母

    Returns:
        随机字符串
    """
    alphabet = ""
    if use_letters:
        alphabet += "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if use_digits:
        alphabet += "0123456789"

    if not alphabet:
        raise ValueError("At least one character type must be enabled")

    return ''.join(
        secrets.choice(alphabet) for _ in range(length)
    )


def generate_secure_token(length: int = 32) -> str:
    """
    生成安全令牌

    Args:
        length: 令牌长度

    Returns:
        安全令牌（十六进制）
    """
    return secrets.token_hex(length)


def generate_url_safe_token(length: int = 32) -> str:
    """
    生成URL安全令牌

    Args:
        length: 令牌长度

    Returns:
        URL安全令牌
    """
    return secrets.token_urlsafe(length)


# =============================================================================
# 哈希和签名
# =============================================================================

def generate_hash(data: str, algorithm: str = "sha256") -> str:
    """
    生成哈希值

    Args:
        data: 要哈希的数据
        algorithm: 哈希算法

    Returns:
        哈希值（十六进制）
    """
    try:
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(data.encode())
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"Error generating hash: {e}")
        raise


def generate_hmac(data: str, key: str, algorithm: str = "sha256") -> str:
    """
    生成HMAC签名

    Args:
        data: 要签名的数据
        key: 签名密钥
        algorithm: 哈希算法

    Returns:
        HMAC签名（十六进制）
    """
    try:
        return hmac.new(
            key.encode(),
            data.encode(),
            getattr(hashlib, algorithm)
        ).hexdigest()
    except Exception as e:
        logger.error(f"Error generating HMAC: {e}")
        raise


def verify_hmac(data: str, signature: str, key: str, algorithm: str = "sha256") -> bool:
    """
    验证HMAC签名

    Args:
        data: 原始数据
        signature: HMAC签名
        key: 签名密钥
        algorithm: 哈希算法

    Returns:
        是否有效
    """
    try:
        expected_signature = generate_hmac(data, key, algorithm)
        return hmac.compare_digest(signature, expected_signature)
    except Exception as e:
        logger.error(f"Error verifying HMAC: {e}")
        return False


# =============================================================================
# 安全检查
# =============================================================================

def is_password_strong(password: str) -> Tuple[bool, list]:
    """
    检查密码强度

    Args:
        password: 密码

    Returns:
        (是否强密码, 不满足的条件列表)
    """
    issues = []

    if len(password) < 8:
        issues.append("密码长度至少8位")

    if not any(c.islower() for c in password):
        issues.append("至少包含一个小写字母")

    if not any(c.isupper() for c in password):
        issues.append("至少包含一个大写字母")

    if not any(c.isdigit() for c in password):
        issues.append("至少包含一个数字")

    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    if not any(c in special_chars for c in password):
        issues.append("至少包含一个特殊字符")

    return len(issues) == 0, issues


def check_common_passwords(password: str) -> bool:
    """
    检查是否为常见密码

    Args:
        password: 密码

    Returns:
        是否为常见密码
    """
    common_passwords = {
        "123456", "password", "123456789", "12345678", "12345",
        "1234567", "1234567890", "qwerty", "abc123", "111111",
        "123123", "admin", "letmein", "welcome", "monkey",
        "password123", "123qwe", "qwerty123", "iloveyou", "princess"
    }

    return password.lower() in common_passwords


def generate_csrf_token() -> str:
    """
    生成CSRF令牌

    Returns:
        CSRF令牌
    """
    return generate_secure_token(32)


def verify_csrf_token(token: str, expected_token: str) -> bool:
    """
    验证CSRF令牌

    Args:
        token: 提交的令牌
        expected_token: 期望的令牌

    Returns:
        是否有效
    """
    return hmac.compare_digest(token, expected_token)


# =============================================================================
# 安全头部
# =============================================================================

def get_security_headers() -> Dict[str, str]:
    """
    获取安全头部

    Returns:
        安全头部字典
    """
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": (
            "max-age=31536000; includeSubDomains"
        ),
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": (
            "geolocation=(), microphone=(), camera=()"
        )
    }


# =============================================================================
# 速率限制相关
# =============================================================================

def generate_rate_limit_key(identifier: str, action: str) -> str:
    """
    生成速率限制键

    Args:
        identifier: 标识符（如IP地址、用户ID）
        action: 操作类型

    Returns:
        速率限制键
    """
    return f"rate_limit:{action}:{identifier}"


def hash_ip_address(ip_address: str) -> str:
    """
    哈希IP地址（用于隐私保护）

    Args:
        ip_address: IP地址

    Returns:
        哈希后的IP地址
    """
    return generate_hash(f"{ip_address}:{JWT_SECRET_KEY}")