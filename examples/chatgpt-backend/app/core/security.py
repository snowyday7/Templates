# -*- coding: utf-8 -*-
"""
安全模块

提供密码哈希、JWT令牌生成和验证等安全功能。
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime, timedelta

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from passlib.context import CryptContext
from jose import JWTError, jwt
from functools import lru_cache

# 导入模板库组件
from src.core.security import BaseSecurity
from src.core.logging import get_logger
from src.utils.exceptions import AuthenticationException, AuthorizationException

# 导入应用组件
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class Security(BaseSecurity):
    """
    安全管理类
    """
    
    def __init__(self):
        super().__init__()
        # 密码上下文
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12
        )
        
        # JWT算法
        self.algorithm = "HS256"
    
    def get_password_hash(self, password: str) -> str:
        """
        生成密码哈希
        """
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        验证密码
        """
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        创建访问令牌
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=self.algorithm
        )
        
        return encoded_jwt
    
    def create_refresh_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        创建刷新令牌
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=self.algorithm
        )
        
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """
        验证令牌
        """
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[self.algorithm]
            )
            
            # 检查令牌类型
            if payload.get("type") != token_type:
                raise AuthenticationException(
                    message=f"Invalid token type, expected {token_type}",
                    error_code="INVALID_TOKEN_TYPE"
                )
            
            # 检查令牌是否过期
            exp = payload.get("exp")
            if exp and datetime.utcnow().timestamp() > exp:
                raise AuthenticationException(
                    message="Token has expired",
                    error_code="TOKEN_EXPIRED"
                )
            
            return payload
            
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            raise AuthenticationException(
                message="Invalid token",
                error_code="INVALID_TOKEN"
            )
    
    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        解码令牌（不验证）
        """
        try:
            payload = jwt.decode(
                token,
                options={"verify_signature": False, "verify_exp": False}
            )
            return payload
        except JWTError:
            return None
    
    def create_password_reset_token(self, email: str) -> str:
        """
        创建密码重置令牌
        """
        data = {
            "email": email,
            "type": "password_reset"
        }
        
        expire = datetime.utcnow() + timedelta(hours=1)  # 1小时有效期
        data["exp"] = expire
        data["iat"] = datetime.utcnow()
        
        encoded_jwt = jwt.encode(
            data,
            settings.SECRET_KEY,
            algorithm=self.algorithm
        )
        
        return encoded_jwt
    
    def verify_password_reset_token(self, token: str) -> Optional[str]:
        """
        验证密码重置令牌
        """
        try:
            payload = self.verify_token(token, "password_reset")
            return payload.get("email")
        except AuthenticationException:
            return None
    
    def create_email_verification_token(self, email: str) -> str:
        """
        创建邮箱验证令牌
        """
        data = {
            "email": email,
            "type": "email_verification"
        }
        
        expire = datetime.utcnow() + timedelta(days=1)  # 1天有效期
        data["exp"] = expire
        data["iat"] = datetime.utcnow()
        
        encoded_jwt = jwt.encode(
            data,
            settings.SECRET_KEY,
            algorithm=self.algorithm
        )
        
        return encoded_jwt
    
    def verify_email_verification_token(self, token: str) -> Optional[str]:
        """
        验证邮箱验证令牌
        """
        try:
            payload = self.verify_token(token, "email_verification")
            return payload.get("email")
        except AuthenticationException:
            return None
    
    def create_api_key(self, user_id: int, name: str) -> str:
        """
        创建API密钥
        """
        data = {
            "user_id": user_id,
            "name": name,
            "type": "api_key"
        }
        
        # API密钥不设置过期时间
        data["iat"] = datetime.utcnow()
        
        encoded_jwt = jwt.encode(
            data,
            settings.SECRET_KEY,
            algorithm=self.algorithm
        )
        
        return encoded_jwt
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        验证API密钥
        """
        try:
            payload = jwt.decode(
                api_key,
                settings.SECRET_KEY,
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # API密钥不验证过期时间
            )
            
            if payload.get("type") != "api_key":
                return None
            
            return payload
            
        except JWTError:
            return None
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        生成安全随机令牌
        """
        import secrets
        import string
        
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def hash_data(self, data: str) -> str:
        """
        哈希数据
        """
        import hashlib
        return hashlib.sha256(data.encode()).hexdigest()
    
    def verify_hash(self, data: str, hash_value: str) -> bool:
        """
        验证哈希
        """
        return self.hash_data(data) == hash_value
    
    def encrypt_data(self, data: str, key: Optional[str] = None) -> str:
        """
        加密数据
        """
        from cryptography.fernet import Fernet
        import base64
        
        if key is None:
            key = settings.SECRET_KEY
        
        # 生成Fernet密钥
        key_bytes = key.encode()[:32].ljust(32, b'0')
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        fernet = Fernet(fernet_key)
        
        encrypted_data = fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_data(self, encrypted_data: str, key: Optional[str] = None) -> str:
        """
        解密数据
        """
        from cryptography.fernet import Fernet
        import base64
        
        if key is None:
            key = settings.SECRET_KEY
        
        try:
            # 生成Fernet密钥
            key_bytes = key.encode()[:32].ljust(32, b'0')
            fernet_key = base64.urlsafe_b64encode(key_bytes)
            fernet = Fernet(fernet_key)
            
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode()
            
        except Exception as e:
            logger.error(f"Data decryption failed: {e}")
            raise AuthenticationException(
                message="Failed to decrypt data",
                error_code="DECRYPTION_FAILED"
            )


# 全局安全实例
_security: Optional[Security] = None


@lru_cache()
def get_security() -> Security:
    """
    获取安全实例（单例模式）
    """
    global _security
    if _security is None:
        _security = Security()
    return _security


# 便捷函数
def get_password_hash(password: str) -> str:
    """
    生成密码哈希
    """
    return get_security().get_password_hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码
    """
    return get_security().verify_password(plain_password, hashed_password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    创建访问令牌
    """
    return get_security().create_access_token(data, expires_delta)


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    创建刷新令牌
    """
    return get_security().create_refresh_token(data, expires_delta)


def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """
    验证令牌
    """
    return get_security().verify_token(token, token_type)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    解码令牌（不验证）
    """
    return get_security().decode_token(token)


def create_password_reset_token(email: str) -> str:
    """
    创建密码重置令牌
    """
    return get_security().create_password_reset_token(email)


def verify_password_reset_token(token: str) -> Optional[str]:
    """
    验证密码重置令牌
    """
    return get_security().verify_password_reset_token(token)


def create_email_verification_token(email: str) -> str:
    """
    创建邮箱验证令牌
    """
    return get_security().create_email_verification_token(email)


def verify_email_verification_token(token: str) -> Optional[str]:
    """
    验证邮箱验证令牌
    """
    return get_security().verify_email_verification_token(token)


def create_api_key(user_id: int, name: str) -> str:
    """
    创建API密钥
    """
    return get_security().create_api_key(user_id, name)


def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """
    验证API密钥
    """
    return get_security().verify_api_key(api_key)


def generate_secure_token(length: int = 32) -> str:
    """
    生成安全随机令牌
    """
    return get_security().generate_secure_token(length)


def hash_data(data: str) -> str:
    """
    哈希数据
    """
    return get_security().hash_data(data)


def verify_hash(data: str, hash_value: str) -> bool:
    """
    验证哈希
    """
    return get_security().verify_hash(data, hash_value)


def encrypt_data(data: str, key: Optional[str] = None) -> str:
    """
    加密数据
    """
    return get_security().encrypt_data(data, key)


def decrypt_data(encrypted_data: str, key: Optional[str] = None) -> str:
    """
    解密数据
    """
    return get_security().decrypt_data(encrypted_data, key)