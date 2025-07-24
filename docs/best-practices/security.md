# 安全开发指南

本文档提供了Python后端开发中的安全最佳实践，帮助开发者构建安全可靠的应用程序。

## 📋 目录

- [安全原则](#安全原则)
- [认证与授权](#认证与授权)
- [数据保护](#数据保护)
- [输入验证](#输入验证)
- [API安全](#api安全)
- [数据库安全](#数据库安全)
- [部署安全](#部署安全)
- [监控与审计](#监控与审计)
- [安全测试](#安全测试)
- [常见漏洞防护](#常见漏洞防护)

## 🛡️ 安全原则

### 1. 最小权限原则

```python
# 用户权限管理
from enum import Enum
from typing import List, Set

class Permission(Enum):
    READ_USER = "read:user"
    WRITE_USER = "write:user"
    DELETE_USER = "delete:user"
    READ_ADMIN = "read:admin"
    WRITE_ADMIN = "write:admin"
    SYSTEM_CONFIG = "system:config"

class Role:
    def __init__(self, name: str, permissions: Set[Permission]):
        self.name = name
        self.permissions = permissions
    
    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions

# 预定义角色
ROLES = {
    "user": Role("user", {Permission.READ_USER}),
    "moderator": Role("moderator", {
        Permission.READ_USER, 
        Permission.WRITE_USER
    }),
    "admin": Role("admin", {
        Permission.READ_USER, 
        Permission.WRITE_USER, 
        Permission.DELETE_USER,
        Permission.READ_ADMIN,
        Permission.WRITE_ADMIN
    }),
    "superuser": Role("superuser", set(Permission))
}

class User:
    def __init__(self, username: str, roles: List[str]):
        self.username = username
        self.roles = [ROLES[role] for role in roles if role in ROLES]
    
    def has_permission(self, permission: Permission) -> bool:
        return any(role.has_permission(permission) for role in self.roles)

# 权限装饰器
from functools import wraps
from fastapi import HTTPException, Depends

def require_permission(permission: Permission):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user: User = Depends(get_current_user), **kwargs):
            if not current_user.has_permission(permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission {permission.value} required"
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator

# 使用示例
@app.delete("/users/{user_id}")
@require_permission(Permission.DELETE_USER)
async def delete_user(user_id: int, current_user: User = Depends(get_current_user)):
    # 只有具有删除用户权限的用户才能访问
    pass
```

### 2. 深度防御

```python
# 多层安全验证
class SecurityLayer:
    """安全层基类"""
    
    async def validate(self, request, context) -> bool:
        raise NotImplementedError

class RateLimitLayer(SecurityLayer):
    """速率限制层"""
    
    def __init__(self, redis_client, max_requests: int = 100, window: int = 3600):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window = window
    
    async def validate(self, request, context) -> bool:
        client_ip = request.client.host
        key = f"rate_limit:{client_ip}"
        
        current = await self.redis.get(key)
        if current is None:
            await self.redis.setex(key, self.window, 1)
            return True
        
        if int(current) >= self.max_requests:
            return False
        
        await self.redis.incr(key)
        return True

class IPWhitelistLayer(SecurityLayer):
    """IP白名单层"""
    
    def __init__(self, allowed_ips: Set[str]):
        self.allowed_ips = allowed_ips
    
    async def validate(self, request, context) -> bool:
        client_ip = request.client.host
        return client_ip in self.allowed_ips

class TokenValidationLayer(SecurityLayer):
    """令牌验证层"""
    
    def __init__(self, token_service):
        self.token_service = token_service
    
    async def validate(self, request, context) -> bool:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return False
        
        token = auth_header.split(" ")[1]
        return await self.token_service.validate_token(token)

class SecurityMiddleware:
    """安全中间件"""
    
    def __init__(self, layers: List[SecurityLayer]):
        self.layers = layers
    
    async def __call__(self, request, call_next):
        context = {"request": request}
        
        # 逐层验证
        for layer in self.layers:
            if not await layer.validate(request, context):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Access denied"}
                )
        
        response = await call_next(request)
        return response

# 配置安全层
security_layers = [
    RateLimitLayer(redis_client, max_requests=1000, window=3600),
    TokenValidationLayer(token_service),
    # IPWhitelistLayer({"192.168.1.0/24"})  # 可选的IP白名单
]

app.add_middleware(SecurityMiddleware, layers=security_layers)
```

## 🔐 认证与授权

### 1. JWT令牌安全

```python
import jwt
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class SecureJWTManager:
    """安全的JWT管理器"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_blacklist = set()  # 在生产环境中应使用Redis
    
    def create_access_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建访问令牌"""
        to_encode = data.copy()
        
        # 设置过期时间
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        # 添加标准声明
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(32),  # JWT ID，用于撤销
            "type": "access"
        })
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """创建刷新令牌"""
        to_encode = {
            "sub": user_id,
            "exp": datetime.utcnow() + timedelta(days=7),
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(32),
            "type": "refresh"
        }
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证令牌"""
        try:
            # 检查黑名单
            if token in self.token_blacklist:
                return None
            
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            # 验证令牌类型
            if payload.get("type") != "access":
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.JWTError:
            return None
    
    def revoke_token(self, token: str):
        """撤销令牌"""
        self.token_blacklist.add(token)
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """使用刷新令牌获取新的访问令牌"""
        try:
            payload = jwt.decode(
                refresh_token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            if payload.get("type") != "refresh":
                return None
            
            # 创建新的访问令牌
            new_token_data = {
                "sub": payload["sub"],
                "user_id": payload.get("user_id")
            }
            
            return self.create_access_token(new_token_data)
            
        except jwt.JWTError:
            return None

# 密码安全处理
from passlib.context import CryptContext
from passlib.hash import bcrypt

class PasswordManager:
    """密码管理器"""
    
    def __init__(self):
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12  # 增加加密轮数
        )
    
    def hash_password(self, password: str) -> str:
        """哈希密码"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def is_password_strong(self, password: str) -> bool:
        """检查密码强度"""
        if len(password) < 8:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return all([has_upper, has_lower, has_digit, has_special])
    
    def generate_secure_password(self, length: int = 16) -> str:
        """生成安全密码"""
        import string
        import random
        
        characters = (
            string.ascii_lowercase + 
            string.ascii_uppercase + 
            string.digits + 
            "!@#$%^&*"
        )
        
        password = ''.join(random.choice(characters) for _ in range(length))
        
        # 确保包含所有字符类型
        if not self.is_password_strong(password):
            return self.generate_secure_password(length)
        
        return password

# 多因素认证
class MFAManager:
    """多因素认证管理器"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def generate_totp_secret(self) -> str:
        """生成TOTP密钥"""
        return secrets.token_urlsafe(32)
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """生成备用代码"""
        return [secrets.token_urlsafe(8) for _ in range(count)]
    
    async def send_sms_code(self, phone: str) -> str:
        """发送短信验证码"""
        code = f"{random.randint(100000, 999999)}"
        
        # 存储验证码（5分钟过期）
        await self.redis.setex(f"sms_code:{phone}", 300, code)
        
        # 发送短信（这里需要集成短信服务）
        await self._send_sms(phone, f"Your verification code is: {code}")
        
        return code
    
    async def verify_sms_code(self, phone: str, code: str) -> bool:
        """验证短信验证码"""
        stored_code = await self.redis.get(f"sms_code:{phone}")
        if stored_code and stored_code.decode() == code:
            await self.redis.delete(f"sms_code:{phone}")
            return True
        return False
    
    def verify_totp_code(self, secret: str, code: str) -> bool:
        """验证TOTP代码"""
        import pyotp
        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=1)
```

### 2. 会话管理

```python
import redis
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class SecureSessionManager:
    """安全会话管理器"""
    
    def __init__(self, redis_client: redis.Redis, session_timeout: int = 3600):
        self.redis = redis_client
        self.session_timeout = session_timeout
    
    async def create_session(self, user_id: str, user_data: Dict[str, Any]) -> str:
        """创建会话"""
        session_id = secrets.token_urlsafe(32)
        
        session_data = {
            "user_id": user_id,
            "user_data": user_data,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "ip_address": None,  # 在中间件中设置
            "user_agent": None   # 在中间件中设置
        }
        
        await self.redis.setex(
            f"session:{session_id}",
            self.session_timeout,
            json.dumps(session_data)
        )
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话"""
        session_data = await self.redis.get(f"session:{session_id}")
        if session_data:
            return json.loads(session_data)
        return None
    
    async def update_session(self, session_id: str, data: Dict[str, Any]):
        """更新会话"""
        session_data = await self.get_session(session_id)
        if session_data:
            session_data.update(data)
            session_data["last_activity"] = datetime.utcnow().isoformat()
            
            await self.redis.setex(
                f"session:{session_id}",
                self.session_timeout,
                json.dumps(session_data)
            )
    
    async def delete_session(self, session_id: str):
        """删除会话"""
        await self.redis.delete(f"session:{session_id}")
    
    async def cleanup_expired_sessions(self):
        """清理过期会话"""
        # Redis会自动处理过期，这里可以添加额外的清理逻辑
        pass

# 会话中间件
class SessionMiddleware:
    def __init__(self, session_manager: SecureSessionManager):
        self.session_manager = session_manager
    
    async def __call__(self, request, call_next):
        # 从cookie或header中获取session_id
        session_id = request.cookies.get("session_id") or \
                    request.headers.get("X-Session-ID")
        
        if session_id:
            session_data = await self.session_manager.get_session(session_id)
            if session_data:
                # 更新会话活动时间
                await self.session_manager.update_session(session_id, {
                    "ip_address": request.client.host,
                    "user_agent": request.headers.get("User-Agent")
                })
                
                request.state.session = session_data
                request.state.session_id = session_id
        
        response = await call_next(request)
        return response
```

## 🔒 数据保护

### 1. 数据加密

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class DataEncryption:
    """数据加密工具"""
    
    def __init__(self, password: str):
        self.password = password.encode()
        self.salt = os.urandom(16)
        self.key = self._derive_key()
        self.fernet = Fernet(self.key)
    
    def _derive_key(self) -> bytes:
        """从密码派生密钥"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password))
        return key
    
    def encrypt(self, data: str) -> str:
        """加密数据"""
        encrypted_data = self.fernet.encrypt(data.encode())
        # 将salt和加密数据一起存储
        return base64.urlsafe_b64encode(self.salt + encrypted_data).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """解密数据"""
        data = base64.urlsafe_b64decode(encrypted_data.encode())
        salt = data[:16]
        encrypted = data[16:]
        
        # 使用存储的salt重新派生密钥
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password))
        fernet = Fernet(key)
        
        decrypted_data = fernet.decrypt(encrypted)
        return decrypted_data.decode()

# 敏感数据字段加密
from sqlalchemy import TypeDecorator, String
from sqlalchemy.ext.declarative import declarative_base

class EncryptedType(TypeDecorator):
    """加密字段类型"""
    
    impl = String
    cache_ok = True
    
    def __init__(self, encryption_key: str, *args, **kwargs):
        self.encryption = DataEncryption(encryption_key)
        super().__init__(*args, **kwargs)
    
    def process_bind_param(self, value, dialect):
        """存储时加密"""
        if value is not None:
            return self.encryption.encrypt(value)
        return value
    
    def process_result_value(self, value, dialect):
        """读取时解密"""
        if value is not None:
            return self.encryption.decrypt(value)
        return value

# 使用示例
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), nullable=False)
    
    # 加密敏感字段
    phone = Column(EncryptedType(os.getenv("ENCRYPTION_KEY")), nullable=True)
    ssn = Column(EncryptedType(os.getenv("ENCRYPTION_KEY")), nullable=True)
    credit_card = Column(EncryptedType(os.getenv("ENCRYPTION_KEY")), nullable=True)
```

### 2. 数据脱敏

```python
import re
from typing import Any, Dict

class DataMasking:
    """数据脱敏工具"""
    
    @staticmethod
    def mask_email(email: str) -> str:
        """邮箱脱敏"""
        if not email or '@' not in email:
            return email
        
        local, domain = email.split('@', 1)
        if len(local) <= 2:
            masked_local = '*' * len(local)
        else:
            masked_local = local[0] + '*' * (len(local) - 2) + local[-1]
        
        return f"{masked_local}@{domain}"
    
    @staticmethod
    def mask_phone(phone: str) -> str:
        """手机号脱敏"""
        if not phone:
            return phone
        
        # 移除非数字字符
        digits = re.sub(r'\D', '', phone)
        if len(digits) < 7:
            return '*' * len(phone)
        
        # 保留前3位和后4位
        return digits[:3] + '*' * (len(digits) - 7) + digits[-4:]
    
    @staticmethod
    def mask_credit_card(card_number: str) -> str:
        """信用卡号脱敏"""
        if not card_number:
            return card_number
        
        digits = re.sub(r'\D', '', card_number)
        if len(digits) < 8:
            return '*' * len(card_number)
        
        return '*' * (len(digits) - 4) + digits[-4:]
    
    @staticmethod
    def mask_ssn(ssn: str) -> str:
        """社会保障号脱敏"""
        if not ssn:
            return ssn
        
        digits = re.sub(r'\D', '', ssn)
        if len(digits) != 9:
            return '*' * len(ssn)
        
        return f"***-**-{digits[-4:]}"
    
    @classmethod
    def mask_sensitive_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """批量脱敏敏感数据"""
        masked_data = data.copy()
        
        sensitive_fields = {
            'email': cls.mask_email,
            'phone': cls.mask_phone,
            'mobile': cls.mask_phone,
            'credit_card': cls.mask_credit_card,
            'card_number': cls.mask_credit_card,
            'ssn': cls.mask_ssn,
            'social_security': cls.mask_ssn
        }
        
        for field, mask_func in sensitive_fields.items():
            if field in masked_data and isinstance(masked_data[field], str):
                masked_data[field] = mask_func(masked_data[field])
        
        return masked_data

# 响应数据脱敏中间件
class DataMaskingMiddleware:
    def __init__(self, mask_in_production: bool = True):
        self.mask_in_production = mask_in_production
    
    async def __call__(self, request, call_next):
        response = await call_next(request)
        
        # 只在生产环境或明确要求时进行脱敏
        if self.mask_in_production and os.getenv("ENVIRONMENT") == "production":
            if hasattr(response, 'body'):
                try:
                    import json
                    data = json.loads(response.body)
                    
                    if isinstance(data, dict):
                        masked_data = DataMasking.mask_sensitive_data(data)
                        response.body = json.dumps(masked_data).encode()
                    elif isinstance(data, list):
                        masked_data = [
                            DataMasking.mask_sensitive_data(item) 
                            if isinstance(item, dict) else item 
                            for item in data
                        ]
                        response.body = json.dumps(masked_data).encode()
                        
                except (json.JSONDecodeError, AttributeError):
                    # 如果不是JSON数据，跳过脱敏
                    pass
        
        return response
```

## ✅ 输入验证

### 1. 数据验证

```python
from pydantic import BaseModel, validator, Field
from typing import Optional, List
import re
from datetime import datetime

class SecureUserInput(BaseModel):
    """安全的用户输入验证"""
    
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    password: str = Field(..., min_length=8, max_length=128)
    phone: Optional[str] = Field(None, regex=r'^\+?[1-9]\d{1,14}$')
    age: Optional[int] = Field(None, ge=13, le=120)
    website: Optional[str] = Field(None, regex=r'^https?://[^\s/$.?#].[^\s]*$')
    
    @validator('username')
    def validate_username(cls, v):
        """用户名验证"""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username can only contain letters, numbers, underscore and hyphen')
        
        # 检查是否包含敏感词
        forbidden_words = ['admin', 'root', 'system', 'null', 'undefined']
        if v.lower() in forbidden_words:
            raise ValueError('Username contains forbidden words')
        
        return v
    
    @validator('password')
    def validate_password(cls, v):
        """密码强度验证"""
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', v):
            raise ValueError('Password must contain at least one special character')
        
        # 检查常见弱密码
        weak_passwords = [
            'password', '12345678', 'qwerty123', 'admin123',
            'password123', '123456789', 'welcome123'
        ]
        if v.lower() in weak_passwords:
            raise ValueError('Password is too weak')
        
        return v
    
    @validator('email')
    def validate_email(cls, v):
        """邮箱验证"""
        # 检查邮箱长度
        if len(v) > 254:
            raise ValueError('Email address is too long')
        
        # 检查本地部分长度
        local_part = v.split('@')[0]
        if len(local_part) > 64:
            raise ValueError('Email local part is too long')
        
        # 检查是否为一次性邮箱
        disposable_domains = [
            '10minutemail.com', 'tempmail.org', 'guerrillamail.com'
        ]
        domain = v.split('@')[1].lower()
        if domain in disposable_domains:
            raise ValueError('Disposable email addresses are not allowed')
        
        return v.lower()

# SQL注入防护
class SQLInjectionProtection:
    """SQL注入防护"""
    
    # 危险的SQL关键词
    DANGEROUS_KEYWORDS = [
        'union', 'select', 'insert', 'update', 'delete', 'drop', 'create',
        'alter', 'exec', 'execute', 'script', 'javascript', 'vbscript',
        'onload', 'onerror', 'onclick', '<script', '</script>'
    ]
    
    @classmethod
    def is_safe_input(cls, input_string: str) -> bool:
        """检查输入是否安全"""
        if not input_string:
            return True
        
        input_lower = input_string.lower()
        
        # 检查危险关键词
        for keyword in cls.DANGEROUS_KEYWORDS:
            if keyword in input_lower:
                return False
        
        # 检查SQL注入模式
        sql_patterns = [
            r"'\s*;\s*--",  # '; --
            r"'\s*or\s+'.*'\s*=\s*'",  # ' or '1'='1
            r"\bunion\s+select\b",  # union select
            r"\bdrop\s+table\b",  # drop table
            r"\bexec\s*\(",  # exec(
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, input_lower, re.IGNORECASE):
                return False
        
        return True
    
    @classmethod
    def sanitize_input(cls, input_string: str) -> str:
        """清理输入"""
        if not input_string:
            return input_string
        
        # 移除危险字符
        sanitized = re.sub(r'[<>"\'\\/]', '', input_string)
        
        # 转义特殊字符
        sanitized = sanitized.replace('&', '&amp;')
        sanitized = sanitized.replace('<', '&lt;')
        sanitized = sanitized.replace('>', '&gt;')
        
        return sanitized

# XSS防护
class XSSProtection:
    """XSS攻击防护"""
    
    @staticmethod
    def escape_html(text: str) -> str:
        """HTML转义"""
        if not text:
            return text
        
        escape_chars = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '/': '&#x2F;'
        }
        
        for char, escape in escape_chars.items():
            text = text.replace(char, escape)
        
        return text
    
    @staticmethod
    def remove_script_tags(text: str) -> str:
        """移除脚本标签"""
        if not text:
            return text
        
        # 移除script标签
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # 移除事件处理器
        text = re.sub(r'\s*on\w+\s*=\s*["\'][^"\'>]*["\']', '', text, flags=re.IGNORECASE)
        
        # 移除javascript:协议
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        return text
    
    @classmethod
    def sanitize_html(cls, html: str) -> str:
        """清理HTML内容"""
        if not html:
            return html
        
        # 移除脚本
        html = cls.remove_script_tags(html)
        
        # HTML转义
        html = cls.escape_html(html)
        
        return html

# 输入验证中间件
class InputValidationMiddleware:
    def __init__(self):
        self.sql_protection = SQLInjectionProtection()
        self.xss_protection = XSSProtection()
    
    async def __call__(self, request, call_next):
        # 验证查询参数
        for key, value in request.query_params.items():
            if not self.sql_protection.is_safe_input(value):
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"Invalid input in parameter: {key}"}
                )
        
        # 验证表单数据
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                if request.headers.get("content-type", "").startswith("application/json"):
                    body = await request.body()
                    if body:
                        import json
                        data = json.loads(body)
                        if not self._validate_json_data(data):
                            return JSONResponse(
                                status_code=400,
                                content={"detail": "Invalid input data"}
                            )
            except Exception:
                pass
        
        response = await call_next(request)
        return response
    
    def _validate_json_data(self, data) -> bool:
        """验证JSON数据"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    if not self.sql_protection.is_safe_input(value):
                        return False
                elif isinstance(value, (dict, list)):
                    if not self._validate_json_data(value):
                        return False
        elif isinstance(data, list):
            for item in data:
                if not self._validate_json_data(item):
                    return False
        
        return True
```

## 🌐 API安全

### 1. API速率限制

```python
import time
import asyncio
from typing import Dict, Optional
from fastapi import HTTPException, Request
from collections import defaultdict, deque

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.local_cache = defaultdict(deque)  # 本地缓存，用于无Redis时
    
    async def is_allowed(
        self, 
        key: str, 
        limit: int, 
        window: int,
        identifier: str = None
    ) -> bool:
        """检查是否允许请求"""
        if self.redis:
            return await self._redis_rate_limit(key, limit, window)
        else:
            return self._memory_rate_limit(key, limit, window)
    
    async def _redis_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """基于Redis的速率限制"""
        current_time = int(time.time())
        pipeline = self.redis.pipeline()
        
        # 使用滑动窗口算法
        pipeline.zremrangebyscore(key, 0, current_time - window)
        pipeline.zcard(key)
        pipeline.zadd(key, {str(current_time): current_time})
        pipeline.expire(key, window)
        
        results = await pipeline.execute()
        current_requests = results[1]
        
        return current_requests < limit
    
    def _memory_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """基于内存的速率限制"""
        current_time = time.time()
        requests = self.local_cache[key]
        
        # 清理过期请求
        while requests and requests[0] <= current_time - window:
            requests.popleft()
        
        if len(requests) >= limit:
            return False
        
        requests.append(current_time)
        return True

# 速率限制装饰器
def rate_limit(requests: int, window: int, per: str = "ip"):
    """速率限制装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, request: Request, **kwargs):
            rate_limiter = RateLimiter()
            
            # 确定限制键
            if per == "ip":
                key = f"rate_limit:ip:{request.client.host}"
            elif per == "user":
                user = getattr(request.state, 'user', None)
                if user:
                    key = f"rate_limit:user:{user.id}"
                else:
                    key = f"rate_limit:ip:{request.client.host}"
            else:
                key = f"rate_limit:global"
            
            if not await rate_limiter.is_allowed(key, requests, window):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(window)}
                )
            
            return await func(*args, request=request, **kwargs)
        return wrapper
    return decorator

# 使用示例
@app.post("/api/login")
@rate_limit(requests=5, window=300, per="ip")  # 每5分钟最多5次登录尝试
async def login(request: Request, credentials: LoginCredentials):
    # 登录逻辑
    pass

@app.get("/api/data")
@rate_limit(requests=100, window=3600, per="user")  # 每小时最多100次请求
async def get_data(request: Request, current_user: User = Depends(get_current_user)):
    # 获取数据逻辑
    pass
```

### 2. API版本控制和弃用

```python
from fastapi import APIRouter, Header, HTTPException
from typing import Optional
from datetime import datetime, timedelta

class APIVersionManager:
    """API版本管理器"""
    
    def __init__(self):
        self.versions = {
            "v1": {
                "status": "deprecated",
                "sunset_date": datetime(2024, 12, 31),
                "supported_until": datetime(2024, 6, 30)
            },
            "v2": {
                "status": "current",
                "sunset_date": None,
                "supported_until": None
            },
            "v3": {
                "status": "beta",
                "sunset_date": None,
                "supported_until": None
            }
        }
    
    def get_version_info(self, version: str) -> dict:
        """获取版本信息"""
        return self.versions.get(version, {})
    
    def is_version_supported(self, version: str) -> bool:
        """检查版本是否支持"""
        version_info = self.get_version_info(version)
        if not version_info:
            return False
        
        if version_info["status"] == "deprecated":
            supported_until = version_info.get("supported_until")
            if supported_until and datetime.now() > supported_until:
                return False
        
        return True

# 版本控制中间件
class APIVersionMiddleware:
    def __init__(self, version_manager: APIVersionManager):
        self.version_manager = version_manager
    
    async def __call__(self, request, call_next):
        # 从URL路径或Header中获取版本
        version = self._extract_version(request)
        
        if version:
            if not self.version_manager.is_version_supported(version):
                return JSONResponse(
                    status_code=410,
                    content={
                        "error": "API version no longer supported",
                        "version": version
                    }
                )
            
            version_info = self.version_manager.get_version_info(version)
            request.state.api_version = version
            request.state.version_info = version_info
        
        response = await call_next(request)
        
        # 添加版本相关的响应头
        if version:
            response.headers["API-Version"] = version
            
            if version_info.get("status") == "deprecated":
                response.headers["Deprecation"] = "true"
                sunset_date = version_info.get("sunset_date")
                if sunset_date:
                    response.headers["Sunset"] = sunset_date.isoformat()
        
        return response
    
    def _extract_version(self, request) -> Optional[str]:
        """从请求中提取版本号"""
        # 从URL路径提取
        path_parts = request.url.path.split('/')
        for part in path_parts:
            if part.startswith('v') and part[1:].isdigit():
                return part
        
        # 从Header提取
        return request.headers.get("API-Version")
```

## 🗄️ 数据库安全

### 1. 连接安全

```python
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import ssl

class SecureDatabaseManager:
    """安全数据库管理器"""
    
    def __init__(self, database_url: str, ssl_config: dict = None):
        self.database_url = database_url
        self.ssl_config = ssl_config or {}
        self.engine = self._create_secure_engine()
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def _create_secure_engine(self):
        """创建安全的数据库引擎"""
        connect_args = {}
        
        # SSL配置
        if self.ssl_config:
            if 'postgresql' in self.database_url:
                connect_args.update({
                    'sslmode': self.ssl_config.get('sslmode', 'require'),
                    'sslcert': self.ssl_config.get('sslcert'),
                    'sslkey': self.ssl_config.get('sslkey'),
                    'sslrootcert': self.ssl_config.get('sslrootcert')
                })
            elif 'mysql' in self.database_url:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_REQUIRED
                connect_args['ssl'] = ssl_context
        
        engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True,
            connect_args=connect_args,
            echo=False  # 生产环境不输出SQL
        )
        
        # 添加连接事件监听器
        self._setup_connection_events(engine)
        
        return engine
    
    def _setup_connection_events(self, engine):
        """设置连接事件"""
        
        @event.listens_for(engine, "connect")
        def set_connection_security(dbapi_connection, connection_record):
            """设置连接安全参数"""
            if 'postgresql' in self.database_url:
                # PostgreSQL安全设置
                with dbapi_connection.cursor() as cursor:
                    cursor.execute("SET statement_timeout = '30s'")
                    cursor.execute("SET lock_timeout = '10s'")
                    cursor.execute("SET idle_in_transaction_session_timeout = '60s'")
            
            elif 'mysql' in self.database_url:
                # MySQL安全设置
                with dbapi_connection.cursor() as cursor:
                    cursor.execute("SET SESSION sql_mode = 'STRICT_TRANS_TABLES'")
                    cursor.execute("SET SESSION max_execution_time = 30000")
        
        @event.listens_for(engine, "before_cursor_execute")
        def log_sql_queries(conn, cursor, statement, parameters, context, executemany):
            """记录SQL查询（仅在开发环境）"""
            if os.getenv("ENVIRONMENT") == "development":
                logger.debug(f"SQL: {statement}")
                logger.debug(f"Parameters: {parameters}")

# 查询安全
class SecureQueryBuilder:
    """安全查询构建器"""
    
    def __init__(self, session):
        self.session = session
    
    def safe_query(self, model, filters: dict = None, limit: int = 100):
        """安全查询"""
        query = self.session.query(model)
        
        # 限制查询结果数量
        if limit > 1000:
            limit = 1000
        
        if filters:
            for key, value in filters.items():
                if hasattr(model, key):
                    # 使用参数化查询防止SQL注入
                    query = query.filter(getattr(model, key) == value)
        
        return query.limit(limit)
    
    def safe_raw_query(self, sql: str, params: dict = None):
        """安全原生查询"""
        # 检查SQL语句
        if not self._is_safe_sql(sql):
            raise ValueError("Unsafe SQL query detected")
        
        # 使用参数化查询
        return self.session.execute(sql, params or {})
    
    def _is_safe_sql(self, sql: str) -> bool:
        """检查SQL是否安全"""
        sql_lower = sql.lower().strip()
        
        # 只允许SELECT语句
        if not sql_lower.startswith('select'):
            return False
        
        # 禁止的关键词
        forbidden_keywords = [
            'drop', 'delete', 'update', 'insert', 'create', 'alter',
            'exec', 'execute', 'sp_', 'xp_', '--', '/*', '*/', ';'
        ]
        
        for keyword in forbidden_keywords:
            if keyword in sql_lower:
                return False
        
        return True
```

## 🚀 部署安全

### 1. 容器安全

```dockerfile
# 安全的Dockerfile
FROM python:3.11-slim as base

# 安全更新
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 创建非特权用户
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY --chown=appuser:appuser . .

# 移除不必要的文件
RUN rm -rf /app/tests /app/docs /app/.git

# 设置文件权限
RUN chmod -R 755 /app && \
    chmod -R 644 /app/*.py

# 切换到非特权用户
USER appuser

# 设置安全的环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH=/home/appuser/.local/bin:$PATH

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Kubernetes安全配置

```yaml
# 安全的Kubernetes部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secure-app
  labels:
    app: secure-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: secure-app
  template:
    metadata:
      labels:
        app: secure-app
    spec:
      serviceAccountName: secure-app-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
      containers:
      - name: app
        image: secure-app:latest
        ports:
        - containerPort: 8000
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1000
          capabilities:
            drop:
            - ALL
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: secret-key
        volumeMounts:
        - name: tmp-volume
          mountPath: /tmp
        - name: cache-volume
          mountPath: /app/cache
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: tmp-volume
        emptyDir: {}
      - name: cache-volume
        emptyDir: {}
      imagePullSecrets:
      - name: registry-secret

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: secure-app-sa
automountServiceAccountToken: false

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: secure-app-netpol
spec:
  podSelector:
    matchLabels:
      app: secure-app
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

## 📊 监控与审计

### 1. 安全事件监控

```python
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

class SecurityEventType(Enum):
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    PERMISSION_DENIED = "permission_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "config_change"
    API_ABUSE = "api_abuse"
    INJECTION_ATTEMPT = "injection_attempt"

class SecurityAuditLogger:
    """安全审计日志记录器"""
    
    def __init__(self, logger_name: str = "security_audit"):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # 配置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler('/var/log/security_audit.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 控制台处理器（开发环境）
        if os.getenv("ENVIRONMENT") == "development":
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def log_security_event(
        self,
        event_type: SecurityEventType,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "INFO"
    ):
        """记录安全事件"""
        event_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type.value,
            "user_id": user_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "details": details or {},
            "severity": severity
        }
        
        log_message = f"Security Event: {event_type.value} | {event_data}"
        
        if severity == "CRITICAL":
            self.logger.critical(log_message)
        elif severity == "WARNING":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # 发送到外部监控系统（如Sentry、ELK等）
        self._send_to_monitoring_system(event_data)
    
    def _send_to_monitoring_system(self, event_data: Dict[str, Any]):
        """发送到监控系统"""
        # 这里可以集成Sentry、ELK、Prometheus等监控系统
        pass

# 安全监控中间件
class SecurityMonitoringMiddleware:
    def __init__(self, audit_logger: SecurityAuditLogger):
        self.audit_logger = audit_logger
        self.suspicious_patterns = {
            'rapid_requests': 100,  # 1分钟内超过100个请求
            'failed_logins': 5,     # 5分钟内超过5次失败登录
            'unusual_endpoints': [  # 异常端点访问
                '/admin', '/.env', '/config', '/backup'
            ]
        }
    
    async def __call__(self, request, call_next):
        start_time = time.time()
        
        # 检测可疑活动
        await self._detect_suspicious_activity(request)
        
        response = await call_next(request)
        
        # 记录访问日志
        processing_time = time.time() - start_time
        await self._log_access(request, response, processing_time)
        
        return response
    
    async def _detect_suspicious_activity(self, request):
        """检测可疑活动"""
        client_ip = request.client.host
        
        # 检查异常端点访问
        for endpoint in self.suspicious_patterns['unusual_endpoints']:
            if endpoint in request.url.path:
                self.audit_logger.log_security_event(
                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                    ip_address=client_ip,
                    details={"endpoint": request.url.path, "method": request.method},
                    severity="WARNING"
                )
    
    async def _log_access(self, request, response, processing_time):
        """记录访问日志"""
        access_data = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "processing_time": processing_time,
            "ip_address": request.client.host,
            "user_agent": request.headers.get("User-Agent")
        }
        
        if response.status_code >= 400:
            self.audit_logger.log_security_event(
                SecurityEventType.SUSPICIOUS_ACTIVITY,
                ip_address=request.client.host,
                details=access_data,
                severity="WARNING" if response.status_code < 500 else "CRITICAL"
            )
```

### 2. 实时威胁检测

```python
class ThreatDetector:
    """威胁检测器"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.threat_patterns = self._load_threat_patterns()
    
    def _load_threat_patterns(self) -> Dict[str, Any]:
        """加载威胁模式"""
        return {
            'sql_injection': [
                r"'\s*or\s+'.*'\s*=\s*'",
                r"\bunion\s+select\b",
                r"\bdrop\s+table\b",
                r"\bexec\s*\("
            ],
            'xss_patterns': [
                r"<script[^>]*>",
                r"javascript:",
                r"on\w+\s*="
            ],
            'path_traversal': [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c"
            ]
        }
    
    async def analyze_request(self, request) -> Dict[str, Any]:
        """分析请求威胁"""
        threats = []
        
        # 分析URL
        url_threats = self._analyze_url(str(request.url))
        threats.extend(url_threats)
        
        # 分析请求体
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    body_threats = self._analyze_body(body.decode())
                    threats.extend(body_threats)
            except Exception:
                pass
        
        # 分析Headers
        header_threats = self._analyze_headers(request.headers)
        threats.extend(header_threats)
        
        return {
            "threats_detected": len(threats) > 0,
            "threat_count": len(threats),
            "threats": threats
        }
    
    def _analyze_url(self, url: str) -> List[Dict[str, str]]:
        """分析URL威胁"""
        threats = []
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    threats.append({
                        "type": threat_type,
                        "pattern": pattern,
                        "location": "url"
                    })
        
        return threats
    
    def _analyze_body(self, body: str) -> List[Dict[str, str]]:
        """分析请求体威胁"""
        threats = []
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, body, re.IGNORECASE):
                    threats.append({
                        "type": threat_type,
                        "pattern": pattern,
                        "location": "body"
                    })
        
        return threats
    
    def _analyze_headers(self, headers) -> List[Dict[str, str]]:
        """分析请求头威胁"""
        threats = []
        
        for header_name, header_value in headers.items():
            for threat_type, patterns in self.threat_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, header_value, re.IGNORECASE):
                        threats.append({
                            "type": threat_type,
                            "pattern": pattern,
                            "location": f"header:{header_name}"
                        })
        
        return threats
```

## 🧪 安全测试

### 1. 自动化安全测试

```python
import pytest
import requests
from typing import List, Dict

class SecurityTestSuite:
    """安全测试套件"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_sql_injection(self) -> List[Dict[str, Any]]:
        """SQL注入测试"""
        results = []
        
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
            "1' AND 1=1 --",
            "admin'--"
        ]
        
        test_endpoints = [
            "/api/users",
            "/api/login",
            "/api/search"
        ]
        
        for endpoint in test_endpoints:
            for payload in sql_payloads:
                # GET参数测试
                response = self.session.get(
                    f"{self.base_url}{endpoint}",
                    params={"q": payload}
                )
                
                results.append({
                    "endpoint": endpoint,
                    "method": "GET",
                    "payload": payload,
                    "status_code": response.status_code,
                    "vulnerable": self._check_sql_injection_response(response)
                })
                
                # POST数据测试
                response = self.session.post(
                    f"{self.base_url}{endpoint}",
                    json={"username": payload, "password": "test"}
                )
                
                results.append({
                    "endpoint": endpoint,
                    "method": "POST",
                    "payload": payload,
                    "status_code": response.status_code,
                    "vulnerable": self._check_sql_injection_response(response)
                })
        
        return results
    
    def _check_sql_injection_response(self, response) -> bool:
        """检查SQL注入响应"""
        error_indicators = [
            "sql syntax",
            "mysql_fetch",
            "postgresql",
            "ora-",
            "microsoft jet database",
            "sqlite_"
        ]
        
        response_text = response.text.lower()
        return any(indicator in response_text for indicator in error_indicators)
    
    def test_xss_vulnerabilities(self) -> List[Dict[str, Any]]:
        """XSS漏洞测试"""
        results = []
        
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "'><script>alert('XSS')</script>"
        ]
        
        test_endpoints = [
            "/api/comments",
            "/api/profile",
            "/api/search"
        ]
        
        for endpoint in test_endpoints:
            for payload in xss_payloads:
                response = self.session.post(
                    f"{self.base_url}{endpoint}",
                    json={"content": payload}
                )
                
                results.append({
                    "endpoint": endpoint,
                    "payload": payload,
                    "status_code": response.status_code,
                    "vulnerable": payload in response.text
                })
        
        return results
    
    def test_authentication_bypass(self) -> List[Dict[str, Any]]:
        """认证绕过测试"""
        results = []
        
        protected_endpoints = [
            "/api/admin",
            "/api/users/profile",
            "/api/settings"
        ]
        
        for endpoint in protected_endpoints:
            # 无认证访问
            response = self.session.get(f"{self.base_url}{endpoint}")
            
            results.append({
                "endpoint": endpoint,
                "test": "no_auth",
                "status_code": response.status_code,
                "vulnerable": response.status_code == 200
            })
            
            # 无效token
            headers = {"Authorization": "Bearer invalid_token"}
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                headers=headers
            )
            
            results.append({
                "endpoint": endpoint,
                "test": "invalid_token",
                "status_code": response.status_code,
                "vulnerable": response.status_code == 200
            })
        
        return results
    
    def test_rate_limiting(self) -> Dict[str, Any]:
        """速率限制测试"""
        endpoint = "/api/login"
        
        # 快速发送多个请求
        responses = []
        for i in range(20):
            response = self.session.post(
                f"{self.base_url}{endpoint}",
                json={"username": f"test{i}", "password": "test"}
            )
            responses.append(response.status_code)
        
        # 检查是否有429状态码（Too Many Requests）
        rate_limited = 429 in responses
        
        return {
            "endpoint": endpoint,
            "total_requests": len(responses),
            "rate_limited": rate_limited,
            "status_codes": responses
        }

# pytest测试用例
class TestSecurity:
    """安全测试类"""
    
    @pytest.fixture
    def security_tester(self):
        return SecurityTestSuite("http://localhost:8000")
    
    def test_sql_injection_protection(self, security_tester):
        """测试SQL注入防护"""
        results = security_tester.test_sql_injection()
        
        vulnerable_tests = [r for r in results if r["vulnerable"]]
        
        assert len(vulnerable_tests) == 0, f"SQL injection vulnerabilities found: {vulnerable_tests}"
    
    def test_xss_protection(self, security_tester):
        """测试XSS防护"""
        results = security_tester.test_xss_vulnerabilities()
        
        vulnerable_tests = [r for r in results if r["vulnerable"]]
        
        assert len(vulnerable_tests) == 0, f"XSS vulnerabilities found: {vulnerable_tests}"
    
    def test_authentication_required(self, security_tester):
        """测试认证要求"""
        results = security_tester.test_authentication_bypass()
        
        vulnerable_tests = [r for r in results if r["vulnerable"]]
        
        assert len(vulnerable_tests) == 0, f"Authentication bypass found: {vulnerable_tests}"
    
    def test_rate_limiting_enabled(self, security_tester):
        """测试速率限制"""
        result = security_tester.test_rate_limiting()
        
        assert result["rate_limited"], "Rate limiting is not properly configured"
```

## 🛡️ 常见漏洞防护

### 1. OWASP Top 10防护

```python
class OWASPProtection:
    """OWASP Top 10防护"""
    
    @staticmethod
    def prevent_injection(input_data: str) -> str:
        """防止注入攻击"""
        # 使用参数化查询，这里只是示例
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '*']
        
        for char in dangerous_chars:
            input_data = input_data.replace(char, '')
        
        return input_data
    
    @staticmethod
    def secure_authentication():
        """安全认证实现"""
        return {
            "use_strong_passwords": True,
            "implement_mfa": True,
            "secure_session_management": True,
            "account_lockout": True
        }
    
    @staticmethod
    def data_exposure_prevention():
        """敏感数据暴露防护"""
        return {
            "encrypt_data_at_rest": True,
            "encrypt_data_in_transit": True,
            "minimize_data_collection": True,
            "secure_key_management": True
        }
    
    @staticmethod
    def xml_external_entities_prevention():
        """XXE防护"""
        return {
            "disable_external_entities": True,
            "use_safe_xml_parsers": True,
            "validate_xml_input": True
        }
    
    @staticmethod
    def broken_access_control_prevention():
        """访问控制防护"""
        return {
            "implement_rbac": True,
            "principle_of_least_privilege": True,
            "secure_direct_object_references": True,
            "cors_configuration": True
        }
    
    @staticmethod
    def security_misconfiguration_prevention():
        """安全配置错误防护"""
        return {
            "remove_default_accounts": True,
            "disable_unnecessary_features": True,
            "keep_software_updated": True,
            "secure_headers": True
        }
    
    @staticmethod
    def vulnerable_components_management():
        """已知漏洞组件管理"""
        return {
            "dependency_scanning": True,
            "regular_updates": True,
            "vulnerability_monitoring": True,
            "secure_development_lifecycle": True
        }
    
    @staticmethod
    def insufficient_logging_prevention():
        """日志记录不足防护"""
        return {
            "comprehensive_logging": True,
            "log_integrity": True,
            "real_time_monitoring": True,
            "incident_response": True
        }
```

## 📋 安全检查清单

### 开发阶段
- [ ] 实施输入验证和输出编码
- [ ] 使用参数化查询防止SQL注入
- [ ] 实施适当的认证和授权机制
- [ ] 加密敏感数据
- [ ] 实施安全的会话管理
- [ ] 配置安全的HTTP头
- [ ] 实施速率限制
- [ ] 进行代码安全审查

### 测试阶段
- [ ] 进行渗透测试
- [ ] 执行自动化安全扫描
- [ ] 测试认证和授权机制
- [ ] 验证输入验证机制
- [ ] 测试错误处理
- [ ] 检查日志记录

### 部署阶段
- [ ] 使用HTTPS
- [ ] 配置防火墙
- [ ] 实施入侵检测系统
- [ ] 定期更新系统和依赖
- [ ] 配置安全监控
- [ ] 实施备份和恢复策略

### 运维阶段
- [ ] 定期安全审计
- [ ] 监控安全事件
- [ ] 更新安全策略
- [ ] 进行安全培训
- [ ] 制定事件响应计划
- [ ] 定期漏洞评估

## 📚 相关资源

### 安全标准和框架
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [ISO 27001](https://www.iso.org/isoiec-27001-information-security.html)

### 安全工具
- [Bandit](https://bandit.readthedocs.io/) - Python安全扫描
- [Safety](https://pyup.io/safety/) - 依赖漏洞检查
- [OWASP ZAP](https://www.zaproxy.org/) - Web应用安全测试
- [SonarQube](https://www.sonarqube.org/) - 代码质量和安全分析

### 学习资源
- [OWASP WebGoat](https://owasp.org/www-project-webgoat/) - 安全学习平台
- [PortSwigger Web Security Academy](https://portswigger.net/web-security) - Web安全学习
- [Cybrary](https://www.cybrary.it/) - 网络安全培训

---

> **注意**: 安全是一个持续的过程，需要定期更新和改进。请根据最新的威胁情报和最佳实践来调整您的安全策略。