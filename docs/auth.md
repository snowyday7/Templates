# 认证授权模块使用指南

认证授权模块提供了完整的用户认证和权限管理功能，包括JWT令牌、OAuth2、RBAC权限控制等。

## 📋 目录

- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [基础使用](#基础使用)
- [高级功能](#高级功能)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

## 🚀 快速开始

### 安装依赖

```bash
pip install python-jose[cryptography] passlib[bcrypt] python-multipart
```

### 基础配置

```python
from templates.auth import AuthConfig, JWTManager

# 创建认证配置
auth_config = AuthConfig(
    SECRET_KEY="your-secret-key-here",
    ALGORITHM="HS256",
    ACCESS_TOKEN_EXPIRE_MINUTES=30,
    REFRESH_TOKEN_EXPIRE_DAYS=7
)

# 创建JWT管理器
jwt_manager = JWTManager(auth_config)
```

## ⚙️ 配置说明

### AuthConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `SECRET_KEY` | str | 必填 | JWT签名密钥 |
| `ALGORITHM` | str | "HS256" | JWT算法 |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | int | 30 | 访问令牌过期时间(分钟) |
| `REFRESH_TOKEN_EXPIRE_DAYS` | int | 7 | 刷新令牌过期时间(天) |
| `PASSWORD_MIN_LENGTH` | int | 8 | 密码最小长度 |
| `PASSWORD_REQUIRE_UPPERCASE` | bool | True | 密码需要大写字母 |
| `PASSWORD_REQUIRE_LOWERCASE` | bool | True | 密码需要小写字母 |
| `PASSWORD_REQUIRE_DIGITS` | bool | True | 密码需要数字 |
| `PASSWORD_REQUIRE_SPECIAL` | bool | False | 密码需要特殊字符 |
| `MAX_LOGIN_ATTEMPTS` | int | 5 | 最大登录尝试次数 |
| `LOCKOUT_DURATION_MINUTES` | int | 15 | 账户锁定时间(分钟) |
| `ENABLE_2FA` | bool | False | 启用双因子认证 |

### 环境变量配置

创建 `.env` 文件：

```env
SECRET_KEY=your-super-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
PASSWORD_MIN_LENGTH=8
MAX_LOGIN_ATTEMPTS=5
LOCKOUT_DURATION_MINUTES=15
ENABLE_2FA=false
```

## 💻 基础使用

### 1. 用户注册和登录

```python
from templates.auth import PasswordManager, UserManager
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["authentication"])

# 数据模型
class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    full_name: str = None

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

# 密码管理器
password_manager = PasswordManager()
user_manager = UserManager()

@router.post("/register", response_model=dict)
async def register(user_data: UserRegister):
    # 检查用户是否已存在
    if await user_manager.get_by_username(user_data.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    
    if await user_manager.get_by_email(user_data.email):
        raise HTTPException(status_code=400, detail="Email already exists")
    
    # 验证密码强度
    password_manager.validate_password(user_data.password)
    
    # 创建用户
    hashed_password = password_manager.hash_password(user_data.password)
    user = await user_manager.create_user(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name
    )
    
    return {"message": "User created successfully", "user_id": user.id}

@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin):
    # 验证用户凭据
    user = await user_manager.authenticate_user(
        user_credentials.username,
        user_credentials.password
    )
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )
    
    # 生成令牌
    access_token = jwt_manager.create_access_token(
        data={"sub": user.username, "user_id": user.id}
    )
    refresh_token = jwt_manager.create_refresh_token(
        data={"sub": user.username, "user_id": user.id}
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }
```

### 2. 令牌验证和用户获取

```python
from templates.auth import get_current_user, get_current_active_user
from fastapi import Depends

# 获取当前用户
@router.get("/me")
async def get_current_user_info(
    current_user = Depends(get_current_user)
):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "is_active": current_user.is_active
    }

# 需要活跃用户的端点
@router.get("/protected")
async def protected_route(
    current_user = Depends(get_current_active_user)
):
    return {"message": f"Hello {current_user.username}, this is a protected route"}
```

### 3. 令牌刷新

```python
@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    try:
        # 验证刷新令牌
        payload = jwt_manager.verify_token(refresh_token)
        username = payload.get("sub")
        user_id = payload.get("user_id")
        
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # 生成新的访问令牌
        new_access_token = jwt_manager.create_access_token(
            data={"sub": username, "user_id": user_id}
        )
        
        return {
            "access_token": new_access_token,
            "refresh_token": refresh_token,  # 保持原刷新令牌
            "token_type": "bearer"
        }
        
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
```

## 🔧 高级功能

### 1. 基于角色的访问控制 (RBAC)

```python
from templates.auth import RoleManager, PermissionManager, require_permission
from enum import Enum

# 定义权限
class Permission(str, Enum):
    READ_USERS = "read:users"
    WRITE_USERS = "write:users"
    DELETE_USERS = "delete:users"
    ADMIN_ACCESS = "admin:access"

# 定义角色
class Role(str, Enum):
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

# 角色权限映射
ROLE_PERMISSIONS = {
    Role.USER: [Permission.READ_USERS],
    Role.MODERATOR: [Permission.READ_USERS, Permission.WRITE_USERS],
    Role.ADMIN: [Permission.READ_USERS, Permission.WRITE_USERS, Permission.DELETE_USERS],
    Role.SUPER_ADMIN: [perm for perm in Permission]
}

# 权限检查装饰器
@router.get("/admin/users")
@require_permission(Permission.READ_USERS)
async def get_all_users(
    current_user = Depends(get_current_active_user)
):
    # 只有有读取用户权限的用户才能访问
    return await user_manager.get_all_users()

@router.delete("/admin/users/{user_id}")
@require_permission(Permission.DELETE_USERS)
async def delete_user(
    user_id: int,
    current_user = Depends(get_current_active_user)
):
    # 只有有删除用户权限的用户才能访问
    return await user_manager.delete_user(user_id)
```

### 2. OAuth2 集成

```python
from templates.auth import OAuth2Manager
from fastapi import Request

# OAuth2 配置
oauth2_config = {
    "google": {
        "client_id": "your-google-client-id",
        "client_secret": "your-google-client-secret",
        "redirect_uri": "http://localhost:8000/auth/google/callback"
    },
    "github": {
        "client_id": "your-github-client-id",
        "client_secret": "your-github-client-secret",
        "redirect_uri": "http://localhost:8000/auth/github/callback"
    }
}

oauth2_manager = OAuth2Manager(oauth2_config)

@router.get("/oauth/{provider}")
async def oauth_login(provider: str):
    if provider not in oauth2_config:
        raise HTTPException(status_code=400, detail="Unsupported provider")
    
    auth_url = oauth2_manager.get_authorization_url(provider)
    return {"auth_url": auth_url}

@router.get("/oauth/{provider}/callback")
async def oauth_callback(provider: str, code: str, request: Request):
    try:
        # 获取访问令牌
        token_data = await oauth2_manager.exchange_code_for_token(provider, code)
        
        # 获取用户信息
        user_info = await oauth2_manager.get_user_info(provider, token_data["access_token"])
        
        # 创建或获取用户
        user = await user_manager.get_or_create_oauth_user(
            provider=provider,
            provider_id=user_info["id"],
            email=user_info["email"],
            username=user_info.get("login", user_info["email"]),
            full_name=user_info.get("name")
        )
        
        # 生成JWT令牌
        access_token = jwt_manager.create_access_token(
            data={"sub": user.username, "user_id": user.id}
        )
        
        return {"access_token": access_token, "token_type": "bearer"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OAuth authentication failed: {str(e)}")
```

### 3. 双因子认证 (2FA)

```python
from templates.auth import TwoFactorAuth
import qrcode
from io import BytesIO
import base64

two_factor = TwoFactorAuth()

@router.post("/2fa/setup")
async def setup_2fa(
    current_user = Depends(get_current_active_user)
):
    # 生成密钥
    secret = two_factor.generate_secret()
    
    # 生成QR码
    qr_code_url = two_factor.generate_qr_code_url(
        secret=secret,
        username=current_user.username,
        issuer="My App"
    )
    
    # 保存密钥到用户记录
    await user_manager.update_user(
        current_user.id,
        {"two_factor_secret": secret}
    )
    
    return {
        "secret": secret,
        "qr_code_url": qr_code_url,
        "message": "Scan the QR code with your authenticator app"
    }

@router.post("/2fa/verify")
async def verify_2fa(
    token: str,
    current_user = Depends(get_current_active_user)
):
    if not current_user.two_factor_secret:
        raise HTTPException(status_code=400, detail="2FA not set up")
    
    is_valid = two_factor.verify_token(
        token=token,
        secret=current_user.two_factor_secret
    )
    
    if is_valid:
        # 启用2FA
        await user_manager.update_user(
            current_user.id,
            {"two_factor_enabled": True}
        )
        return {"message": "2FA enabled successfully"}
    else:
        raise HTTPException(status_code=400, detail="Invalid 2FA token")

@router.post("/login-2fa")
async def login_with_2fa(user_credentials: UserLogin, two_factor_token: str):
    # 首先验证用户名和密码
    user = await user_manager.authenticate_user(
        user_credentials.username,
        user_credentials.password
    )
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # 如果启用了2FA，验证2FA令牌
    if user.two_factor_enabled:
        is_valid = two_factor.verify_token(
            token=two_factor_token,
            secret=user.two_factor_secret
        )
        
        if not is_valid:
            raise HTTPException(status_code=401, detail="Invalid 2FA token")
    
    # 生成JWT令牌
    access_token = jwt_manager.create_access_token(
        data={"sub": user.username, "user_id": user.id}
    )
    
    return {"access_token": access_token, "token_type": "bearer"}
```

### 4. 会话管理

```python
from templates.auth import SessionManager
from fastapi import Request, Response

session_manager = SessionManager()

@router.post("/login-session")
async def login_with_session(
    user_credentials: UserLogin,
    request: Request,
    response: Response
):
    # 验证用户
    user = await user_manager.authenticate_user(
        user_credentials.username,
        user_credentials.password
    )
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # 创建会话
    session_id = await session_manager.create_session(
        user_id=user.id,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent")
    )
    
    # 设置会话cookie
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=3600  # 1小时
    )
    
    return {"message": "Login successful"}

@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    current_user = Depends(get_current_user)
):
    session_id = request.cookies.get("session_id")
    
    if session_id:
        await session_manager.invalidate_session(session_id)
    
    response.delete_cookie("session_id")
    
    return {"message": "Logout successful"}
```

## 📝 最佳实践

### 1. 密码安全

```python
# 强密码策略
class PasswordPolicy:
    MIN_LENGTH = 12
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGITS = True
    REQUIRE_SPECIAL_CHARS = True
    FORBIDDEN_PATTERNS = [
        "password", "123456", "qwerty", "admin"
    ]
    
    @classmethod
    def validate(cls, password: str) -> bool:
        if len(password) < cls.MIN_LENGTH:
            raise ValueError(f"Password must be at least {cls.MIN_LENGTH} characters")
        
        if cls.REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
            raise ValueError("Password must contain uppercase letter")
        
        if cls.REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
            raise ValueError("Password must contain lowercase letter")
        
        if cls.REQUIRE_DIGITS and not re.search(r'\d', password):
            raise ValueError("Password must contain digit")
        
        if cls.REQUIRE_SPECIAL_CHARS and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            raise ValueError("Password must contain special character")
        
        for pattern in cls.FORBIDDEN_PATTERNS:
            if pattern.lower() in password.lower():
                raise ValueError(f"Password cannot contain '{pattern}'")
        
        return True

# 密码历史检查
async def check_password_history(user_id: int, new_password: str) -> bool:
    """检查新密码是否与历史密码重复"""
    password_history = await user_manager.get_password_history(user_id, limit=5)
    
    for old_password_hash in password_history:
        if password_manager.verify_password(new_password, old_password_hash):
            return False
    
    return True
```

### 2. 令牌安全

```python
# 令牌黑名单
class TokenBlacklist:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def add_token(self, token: str, expire_time: int):
        """将令牌添加到黑名单"""
        await self.redis.setex(f"blacklist:{token}", expire_time, "1")
    
    async def is_blacklisted(self, token: str) -> bool:
        """检查令牌是否在黑名单中"""
        result = await self.redis.get(f"blacklist:{token}")
        return result is not None

# 令牌轮换
@router.post("/logout")
async def logout(
    current_user = Depends(get_current_user),
    token: str = Depends(get_token_from_header)
):
    # 将当前令牌加入黑名单
    payload = jwt_manager.verify_token(token)
    expire_time = payload.get("exp") - int(time.time())
    
    await token_blacklist.add_token(token, expire_time)
    
    return {"message": "Logout successful"}
```

### 3. 安全中间件

```python
from templates.auth import SecurityMiddleware

# 安全头部中间件
class SecurityHeadersMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            response = await self.app(scope, receive, send)
            
            # 添加安全头部
            headers = [
                (b"x-content-type-options", b"nosniff"),
                (b"x-frame-options", b"DENY"),
                (b"x-xss-protection", b"1; mode=block"),
                (b"strict-transport-security", b"max-age=31536000; includeSubDomains"),
                (b"content-security-policy", b"default-src 'self'"),
            ]
            
            return response
        
        return await self.app(scope, receive, send)

# 速率限制中间件
class RateLimitMiddleware:
    def __init__(self, app, redis_client, max_requests=100, window_seconds=3600):
        self.app = app
        self.redis = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            client_ip = scope["client"][0]
            key = f"rate_limit:{client_ip}"
            
            current_requests = await self.redis.get(key)
            
            if current_requests and int(current_requests) >= self.max_requests:
                # 返回429状态码
                response = Response(
                    content="Rate limit exceeded",
                    status_code=429
                )
                await response(scope, receive, send)
                return
            
            # 增加请求计数
            await self.redis.incr(key)
            await self.redis.expire(key, self.window_seconds)
        
        await self.app(scope, receive, send)
```

## ❓ 常见问题

### Q: 如何处理令牌过期？

A: 实现自动刷新机制：

```python
# 前端自动刷新令牌
async function apiCall(url, options = {}) {
    let response = await fetch(url, {
        ...options,
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
            ...options.headers
        }
    });
    
    if (response.status === 401) {
        // 尝试刷新令牌
        const refreshResponse = await fetch('/auth/refresh', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                refresh_token: localStorage.getItem('refresh_token')
            })
        });
        
        if (refreshResponse.ok) {
            const tokens = await refreshResponse.json();
            localStorage.setItem('access_token', tokens.access_token);
            
            // 重试原请求
            response = await fetch(url, {
                ...options,
                headers: {
                    'Authorization': `Bearer ${tokens.access_token}`,
                    ...options.headers
                }
            });
        }
    }
    
    return response;
}
```

### Q: 如何实现单点登录 (SSO)？

A: 使用SAML或OpenID Connect：

```python
from templates.auth import SAMLManager

saml_manager = SAMLManager(
    entity_id="your-app",
    sso_url="https://idp.example.com/sso",
    x509_cert="your-cert"
)

@router.get("/saml/login")
async def saml_login():
    auth_url = saml_manager.get_auth_url()
    return {"auth_url": auth_url}

@router.post("/saml/acs")
async def saml_callback(saml_response: str):
    user_data = saml_manager.parse_response(saml_response)
    # 创建或获取用户
    user = await user_manager.get_or_create_saml_user(user_data)
    # 生成JWT令牌
    token = jwt_manager.create_access_token({"sub": user.username})
    return {"access_token": token}
```

### Q: 如何实现账户锁定？

A: 跟踪登录失败次数：

```python
class AccountLockManager:
    def __init__(self, redis_client, max_attempts=5, lockout_duration=900):
        self.redis = redis_client
        self.max_attempts = max_attempts
        self.lockout_duration = lockout_duration
    
    async def record_failed_attempt(self, username: str):
        key = f"failed_attempts:{username}"
        attempts = await self.redis.incr(key)
        await self.redis.expire(key, self.lockout_duration)
        
        if attempts >= self.max_attempts:
            await self.lock_account(username)
    
    async def lock_account(self, username: str):
        key = f"locked_account:{username}"
        await self.redis.setex(key, self.lockout_duration, "1")
    
    async def is_locked(self, username: str) -> bool:
        key = f"locked_account:{username}"
        return await self.redis.exists(key)
    
    async def clear_failed_attempts(self, username: str):
        key = f"failed_attempts:{username}"
        await self.redis.delete(key)
```

## 📚 相关文档

- [API开发模块使用指南](api.md)
- [数据库模块使用指南](database.md)
- [缓存消息模块使用指南](cache.md)
- [安全开发指南](best-practices/security.md)
- [测试策略指南](best-practices/testing.md)

---

如有其他问题，请查看 [GitHub Issues](https://github.com/your-username/python-backend-templates/issues) 或提交新的问题。