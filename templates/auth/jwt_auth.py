"""JWT认证模板

提供完整的JWT认证功能，包括：
- JWT token生成和验证
- 用户认证和授权
- Token刷新机制
- 安全配置
"""

import jwt
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass
from enum import Enum

from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class TokenType(str, Enum):
    """Token类型枚举"""

    ACCESS = "access"
    REFRESH = "refresh"
    RESET_PASSWORD = "reset_password"
    EMAIL_VERIFICATION = "email_verification"


class AuthSettings(BaseSettings):
    """认证配置类"""

    # JWT配置
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # 密码配置
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_NUMBERS: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True

    # 安全配置
    MAX_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_DURATION_MINUTES: int = 15

    # Token黑名单配置
    ENABLE_TOKEN_BLACKLIST: bool = True

    class Config:
        env_file = ".env"


@dataclass
class TokenPayload:
    """Token载荷数据类"""

    user_id: int
    username: str
    email: str
    token_type: TokenType
    permissions: List[str] = None
    roles: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "token_type": self.token_type.value,
            "permissions": self.permissions or [],
            "roles": self.roles or [],
        }


class UserLogin(BaseModel):
    """用户登录模型"""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    remember_me: bool = False


class UserRegister(BaseModel):
    """用户注册模型"""

    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    password: str = Field(..., min_length=8)
    confirm_password: str
    full_name: Optional[str] = None

    @validator("confirm_password")
    def passwords_match(cls, v, values):
        if "password" in values and v != values["password"]:
            raise ValueError("Passwords do not match")
        return v


class TokenResponse(BaseModel):
    """Token响应模型"""

    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    user_info: Dict[str, Any]


class RefreshTokenRequest(BaseModel):
    """刷新Token请求模型"""

    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """密码修改请求模型"""

    current_password: str
    new_password: str = Field(..., min_length=8)
    confirm_password: str

    @validator("confirm_password")
    def passwords_match(cls, v, values):
        if "new_password" in values and v != values["new_password"]:
            raise ValueError("Passwords do not match")
        return v


class JWTManager:
    """JWT管理器"""

    def __init__(self, settings: Optional[AuthSettings] = None):
        self.settings = settings or AuthSettings()
        self.security = HTTPBearer()
        self.blacklisted_tokens = set()  # 在生产环境中应使用Redis等持久化存储

    def create_access_token(
        self, payload: TokenPayload, expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建访问Token"""
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES
            )

        to_encode = payload.to_dict()
        to_encode.update(
            {
                "exp": expire,
                "iat": datetime.now(timezone.utc),
                "type": TokenType.ACCESS.value,
            }
        )

        encoded_jwt = jwt.encode(
            to_encode, self.settings.SECRET_KEY, algorithm=self.settings.ALGORITHM
        )
        return encoded_jwt

    def create_refresh_token(
        self, payload: TokenPayload, expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建刷新Token"""
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                days=self.settings.REFRESH_TOKEN_EXPIRE_DAYS
            )

        to_encode = {
            "user_id": payload.user_id,
            "username": payload.username,
            "email": payload.email,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": TokenType.REFRESH.value,
        }

        encoded_jwt = jwt.encode(
            to_encode, self.settings.SECRET_KEY, algorithm=self.settings.ALGORITHM
        )
        return encoded_jwt

    def verify_token(
        self, token: str, expected_type: TokenType = TokenType.ACCESS
    ) -> Dict[str, Any]:
        """验证Token"""
        try:
            # 检查Token是否在黑名单中
            if (
                self.settings.ENABLE_TOKEN_BLACKLIST
                and token in self.blacklisted_tokens
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                )

            # 解码Token
            payload = jwt.decode(
                token, self.settings.SECRET_KEY, algorithms=[self.settings.ALGORITHM]
            )

            # 验证Token类型
            token_type = payload.get("type")
            if token_type != expected_type.value:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token type. Expected {expected_type.value}",
                )

            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
            )

    def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """刷新访问Token"""
        # 验证刷新Token
        payload = self.verify_token(refresh_token, TokenType.REFRESH)

        # 创建新的Token载荷
        token_payload = TokenPayload(
            user_id=payload["user_id"],
            username=payload["username"],
            email=payload["email"],
            token_type=TokenType.ACCESS,
        )

        # 生成新的访问Token
        access_token = self.create_access_token(token_payload)

        return TokenResponse(
            access_token=access_token,
            expires_in=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_info={
                "user_id": payload["user_id"],
                "username": payload["username"],
                "email": payload["email"],
            },
        )

    def revoke_token(self, token: str):
        """撤销Token（加入黑名单）"""
        if self.settings.ENABLE_TOKEN_BLACKLIST:
            self.blacklisted_tokens.add(token)

    def get_current_user_dependency(self):
        """获取当前用户的依赖函数"""

        async def get_current_user(
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ) -> Dict[str, Any]:
            token = credentials.credentials
            payload = self.verify_token(token)

            # 这里应该从数据库获取完整的用户信息
            # 示例实现
            user_info = {
                "user_id": payload["user_id"],
                "username": payload["username"],
                "email": payload["email"],
                "permissions": payload.get("permissions", []),
                "roles": payload.get("roles", []),
            }

            return user_info

        return get_current_user


class PasswordManager:
    """密码管理器"""

    @staticmethod
    def hash_password(password: str) -> str:
        """哈希密码"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """验证密码"""
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))

    @staticmethod
    def validate_password_strength(password: str, settings: AuthSettings) -> List[str]:
        """验证密码强度"""
        errors = []

        if len(password) < settings.PASSWORD_MIN_LENGTH:
            errors.append(
                f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters long"
            )

        if settings.PASSWORD_REQUIRE_UPPERCASE and not any(
            c.isupper() for c in password
        ):
            errors.append("Password must contain at least one uppercase letter")

        if settings.PASSWORD_REQUIRE_LOWERCASE and not any(
            c.islower() for c in password
        ):
            errors.append("Password must contain at least one lowercase letter")

        if settings.PASSWORD_REQUIRE_NUMBERS and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")

        if settings.PASSWORD_REQUIRE_SPECIAL:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                errors.append("Password must contain at least one special character")

        return errors


class AuthService:
    """认证服务"""

    def __init__(self, settings: Optional[AuthSettings] = None):
        self.settings = settings or AuthSettings()
        self.jwt_manager = JWTManager(settings)
        self.password_manager = PasswordManager()
        self.login_attempts = {}  # 在生产环境中应使用Redis等持久化存储

    async def register_user(self, user_data: UserRegister) -> Dict[str, Any]:
        """用户注册"""
        # 验证密码强度
        password_errors = self.password_manager.validate_password_strength(
            user_data.password, self.settings
        )
        if password_errors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"password_errors": password_errors},
            )

        # 检查用户名和邮箱是否已存在（这里需要数据库查询）
        # 示例实现

        # 哈希密码
        hashed_password = self.password_manager.hash_password(user_data.password)

        # 创建用户（这里需要数据库操作）
        user = {
            "id": 1,  # 应该是数据库生成的ID
            "username": user_data.username,
            "email": user_data.email,
            "full_name": user_data.full_name,
            "hashed_password": hashed_password,
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
        }

        return user

    async def authenticate_user(self, login_data: UserLogin) -> TokenResponse:
        """用户认证"""
        # 检查登录尝试次数
        if self._is_account_locked(login_data.username):
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account is temporarily locked due to too many failed login attempts",
            )

        # 验证用户凭据（这里需要数据库查询）
        user = await self._get_user_by_username(login_data.username)
        if not user or not self.password_manager.verify_password(
            login_data.password, user["hashed_password"]
        ):
            self._record_failed_login(login_data.username)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
            )

        # 清除失败登录记录
        self._clear_failed_logins(login_data.username)

        # 创建Token载荷
        token_payload = TokenPayload(
            user_id=user["id"],
            username=user["username"],
            email=user["email"],
            token_type=TokenType.ACCESS,
            permissions=user.get("permissions", []),
            roles=user.get("roles", []),
        )

        # 生成Token
        access_token = self.jwt_manager.create_access_token(token_payload)
        refresh_token = None

        if login_data.remember_me:
            refresh_token = self.jwt_manager.create_refresh_token(token_payload)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_info={
                "user_id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "full_name": user.get("full_name"),
            },
        )

    async def change_password(
        self, user_id: int, password_data: PasswordChangeRequest
    ) -> bool:
        """修改密码"""
        # 获取用户信息
        user = await self._get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        # 验证当前密码
        if not self.password_manager.verify_password(
            password_data.current_password, user["hashed_password"]
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect",
            )

        # 验证新密码强度
        password_errors = self.password_manager.validate_password_strength(
            password_data.new_password, self.settings
        )
        if password_errors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"password_errors": password_errors},
            )

        # 更新密码（这里需要数据库操作）
        new_hashed_password = self.password_manager.hash_password(
            password_data.new_password
        )

        # 更新数据库中的密码
        # await update_user_password(user_id, new_hashed_password)

        return True

    def _is_account_locked(self, username: str) -> bool:
        """检查账户是否被锁定"""
        if username not in self.login_attempts:
            return False

        attempts = self.login_attempts[username]
        if attempts["count"] >= self.settings.MAX_LOGIN_ATTEMPTS:
            lockout_time = timedelta(minutes=self.settings.LOCKOUT_DURATION_MINUTES)
            if datetime.now(timezone.utc) - attempts["last_attempt"] < lockout_time:
                return True
            else:
                # 锁定时间已过，清除记录
                del self.login_attempts[username]

        return False

    def _record_failed_login(self, username: str):
        """记录失败登录"""
        now = datetime.now(timezone.utc)
        if username in self.login_attempts:
            self.login_attempts[username]["count"] += 1
            self.login_attempts[username]["last_attempt"] = now
        else:
            self.login_attempts[username] = {"count": 1, "last_attempt": now}

    def _clear_failed_logins(self, username: str):
        """清除失败登录记录"""
        if username in self.login_attempts:
            del self.login_attempts[username]

    async def _get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """根据用户名获取用户（示例实现）"""
        # 这里应该是数据库查询
        # 示例数据
        if username == "testuser":
            return {
                "id": 1,
                "username": "testuser",
                "email": "test@example.com",
                "full_name": "Test User",
                "hashed_password": self.password_manager.hash_password("testpassword"),
                "is_active": True,
                "permissions": ["read", "write"],
                "roles": ["user"],
            }
        return None

    async def _get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """根据用户ID获取用户（示例实现）"""
        # 这里应该是数据库查询
        if user_id == 1:
            return {
                "id": 1,
                "username": "testuser",
                "email": "test@example.com",
                "full_name": "Test User",
                "hashed_password": self.password_manager.hash_password("testpassword"),
                "is_active": True,
            }
        return None


# 全局实例
auth_settings = AuthSettings()
jwt_manager = JWTManager(auth_settings)
auth_service = AuthService(auth_settings)

# 便捷函数
create_access_token = jwt_manager.create_access_token
verify_token = jwt_manager.verify_token
get_current_user = jwt_manager.get_current_user_dependency()
hash_password = PasswordManager.hash_password
verify_password = PasswordManager.verify_password


# 使用示例
if __name__ == "__main__":
    import asyncio

    async def test_auth():
        # 创建认证服务
        auth = AuthService()

        # 测试用户注册
        register_data = UserRegister(
            username="testuser",
            email="test@example.com",
            password="TestPassword123!",
            confirm_password="TestPassword123!",
            full_name="Test User",
        )

        try:
            user = await auth.register_user(register_data)
            print(f"User registered: {user}")
        except HTTPException as e:
            print(f"Registration failed: {e.detail}")

        # 测试用户登录
        login_data = UserLogin(
            username="testuser", password="testpassword", remember_me=True
        )

        try:
            token_response = await auth.authenticate_user(login_data)
            print(f"Login successful: {token_response}")

            # 测试Token验证
            payload = jwt_manager.verify_token(token_response.access_token)
            print(f"Token verified: {payload}")

        except HTTPException as e:
            print(f"Login failed: {e.detail}")

    # 运行测试
    asyncio.run(test_auth())
