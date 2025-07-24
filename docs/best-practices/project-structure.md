# 项目结构最佳实践

本文档提供了Python后端项目的结构设计最佳实践，帮助开发者构建可维护、可扩展的项目架构。

## 📋 目录

- [项目结构概览](#项目结构概览)
- [目录组织原则](#目录组织原则)
- [模块化设计](#模块化设计)
- [配置管理](#配置管理)
- [依赖管理](#依赖管理)
- [文档组织](#文档组织)
- [测试结构](#测试结构)
- [部署结构](#部署结构)

## 🏗️ 项目结构概览

### 推荐的项目结构

```
project-name/
├── README.md                 # 项目说明文档
├── requirements.txt          # 生产依赖
├── requirements-dev.txt      # 开发依赖
├── pyproject.toml           # 项目配置文件
├── .env.example             # 环境变量示例
├── .gitignore               # Git忽略文件
├── Dockerfile               # Docker构建文件
├── docker-compose.yml       # Docker Compose配置
├── Makefile                 # 常用命令脚本
│
├── app/                     # 应用主目录
│   ├── __init__.py
│   ├── main.py              # 应用入口点
│   ├── config.py            # 配置管理
│   ├── dependencies.py      # 依赖注入
│   │
│   ├── api/                 # API层
│   │   ├── __init__.py
│   │   ├── deps.py          # API依赖
│   │   ├── errors.py        # 错误处理
│   │   └── v1/              # API版本
│   │       ├── __init__.py
│   │       ├── endpoints/   # API端点
│   │       │   ├── __init__.py
│   │       │   ├── auth.py
│   │       │   ├── users.py
│   │       │   └── items.py
│   │       └── api.py       # API路由聚合
│   │
│   ├── core/                # 核心功能
│   │   ├── __init__.py
│   │   ├── auth.py          # 认证逻辑
│   │   ├── security.py      # 安全工具
│   │   ├── logging.py       # 日志配置
│   │   └── exceptions.py    # 自定义异常
│   │
│   ├── models/              # 数据模型
│   │   ├── __init__.py
│   │   ├── base.py          # 基础模型
│   │   ├── user.py          # 用户模型
│   │   └── item.py          # 业务模型
│   │
│   ├── schemas/             # Pydantic模式
│   │   ├── __init__.py
│   │   ├── user.py          # 用户模式
│   │   └── item.py          # 业务模式
│   │
│   ├── services/            # 业务逻辑层
│   │   ├── __init__.py
│   │   ├── user_service.py  # 用户服务
│   │   └── item_service.py  # 业务服务
│   │
│   ├── repositories/        # 数据访问层
│   │   ├── __init__.py
│   │   ├── base.py          # 基础仓库
│   │   ├── user_repo.py     # 用户仓库
│   │   └── item_repo.py     # 业务仓库
│   │
│   ├── utils/               # 工具函数
│   │   ├── __init__.py
│   │   ├── datetime.py      # 时间工具
│   │   ├── validators.py    # 验证器
│   │   └── helpers.py       # 辅助函数
│   │
│   └── db/                  # 数据库相关
│       ├── __init__.py
│       ├── database.py      # 数据库连接
│       ├── session.py       # 会话管理
│       └── migrations/      # 数据库迁移
│           └── versions/
│
├── tests/                   # 测试目录
│   ├── __init__.py
│   ├── conftest.py          # 测试配置
│   ├── test_main.py         # 主要测试
│   ├── api/                 # API测试
│   │   ├── __init__.py
│   │   └── test_endpoints/
│   ├── services/            # 服务测试
│   ├── repositories/        # 仓库测试
│   └── utils/               # 工具测试
│
├── scripts/                 # 脚本目录
│   ├── init_db.py          # 数据库初始化
│   ├── seed_data.py        # 种子数据
│   └── backup.py           # 备份脚本
│
├── docs/                    # 文档目录
│   ├── api.md              # API文档
│   ├── deployment.md       # 部署文档
│   └── development.md      # 开发文档
│
├── k8s/                     # Kubernetes配置
│   ├── base/               # 基础配置
│   ├── overlays/           # 环境特定配置
│   │   ├── development/
│   │   ├── staging/
│   │   └── production/
│   └── secrets/            # 密钥配置
│
└── .github/                # GitHub配置
    ├── workflows/          # GitHub Actions
    │   ├── ci.yml
    │   └── deploy.yml
    └── ISSUE_TEMPLATE/     # Issue模板
```

## 📁 目录组织原则

### 1. 分层架构原则

```python
# 清晰的分层结构
class LayeredArchitecture:
    """
    分层架构示例：
    
    Presentation Layer (API) -> Business Layer (Services) -> Data Layer (Repositories)
    """
    
    def __init__(self):
        # API层：处理HTTP请求和响应
        self.api_layer = "app/api/"
        
        # 业务层：处理业务逻辑
        self.business_layer = "app/services/"
        
        # 数据层：处理数据访问
        self.data_layer = "app/repositories/"
        
        # 模型层：定义数据结构
        self.model_layer = "app/models/"
        
        # 核心层：提供基础功能
        self.core_layer = "app/core/"

# API层示例
# app/api/v1/endpoints/users.py
from fastapi import APIRouter, Depends
from app.services.user_service import UserService
from app.schemas.user import UserCreate, UserResponse

router = APIRouter()

@router.post("/users/", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    user_service: UserService = Depends()
):
    """创建用户 - API层只处理HTTP相关逻辑"""
    return await user_service.create_user(user_data)

# 业务层示例
# app/services/user_service.py
from app.repositories.user_repo import UserRepository
from app.schemas.user import UserCreate
from app.core.security import get_password_hash

class UserService:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo
    
    async def create_user(self, user_data: UserCreate):
        """创建用户 - 业务层处理业务逻辑"""
        # 业务逻辑：密码加密
        hashed_password = get_password_hash(user_data.password)
        
        # 业务逻辑：数据验证
        if await self.user_repo.get_by_email(user_data.email):
            raise ValueError("Email already registered")
        
        # 委托给数据层
        return await self.user_repo.create({
            **user_data.dict(),
            "hashed_password": hashed_password
        })

# 数据层示例
# app/repositories/user_repo.py
from app.repositories.base import BaseRepository
from app.models.user import User

class UserRepository(BaseRepository[User]):
    """用户数据访问层 - 只处理数据操作"""
    
    async def get_by_email(self, email: str) -> User | None:
        """根据邮箱查询用户"""
        return await self.db.query(User).filter(User.email == email).first()
    
    async def create(self, user_data: dict) -> User:
        """创建用户"""
        user = User(**user_data)
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user
```

### 2. 单一职责原则

```python
# 每个模块都有明确的职责

# app/core/auth.py - 只处理认证相关功能
class AuthManager:
    """认证管理器 - 单一职责：处理认证"""
    
    def create_access_token(self, data: dict) -> str:
        """创建访问令牌"""
        pass
    
    def verify_token(self, token: str) -> dict:
        """验证令牌"""
        pass

# app/core/security.py - 只处理安全相关功能
class SecurityManager:
    """安全管理器 - 单一职责：处理安全"""
    
    def hash_password(self, password: str) -> str:
        """密码哈希"""
        pass
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """密码验证"""
        pass

# app/core/logging.py - 只处理日志相关功能
class LoggingManager:
    """日志管理器 - 单一职责：处理日志"""
    
    def setup_logging(self, config: dict):
        """设置日志配置"""
        pass
    
    def get_logger(self, name: str):
        """获取日志器"""
        pass
```

### 3. 依赖方向原则

```python
# 依赖应该从外层指向内层
# API -> Services -> Repositories -> Models

# ✅ 正确的依赖方向
# app/api/v1/endpoints/users.py
from app.services.user_service import UserService  # API依赖Service

# app/services/user_service.py
from app.repositories.user_repo import UserRepository  # Service依赖Repository

# app/repositories/user_repo.py
from app.models.user import User  # Repository依赖Model

# ❌ 错误的依赖方向
# app/models/user.py
# from app.services.user_service import UserService  # Model不应该依赖Service

# 使用依赖注入解决循环依赖
# app/dependencies.py
from fastapi import Depends
from app.db.session import get_db
from app.repositories.user_repo import UserRepository
from app.services.user_service import UserService

def get_user_repository(db = Depends(get_db)) -> UserRepository:
    return UserRepository(db)

def get_user_service(
    user_repo: UserRepository = Depends(get_user_repository)
) -> UserService:
    return UserService(user_repo)
```

## 🧩 模块化设计

### 1. 功能模块划分

```python
# 按功能领域划分模块
project/
├── app/
│   ├── auth/              # 认证模块
│   │   ├── __init__.py
│   │   ├── models.py      # 认证相关模型
│   │   ├── schemas.py     # 认证相关模式
│   │   ├── services.py    # 认证服务
│   │   ├── repositories.py # 认证数据访问
│   │   └── api.py         # 认证API
│   │
│   ├── users/             # 用户模块
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── services.py
│   │   ├── repositories.py
│   │   └── api.py
│   │
│   ├── orders/            # 订单模块
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── services.py
│   │   ├── repositories.py
│   │   └── api.py
│   │
│   └── shared/            # 共享模块
│       ├── __init__.py
│       ├── exceptions.py  # 共享异常
│       ├── validators.py  # 共享验证器
│       └── utils.py       # 共享工具

# 模块接口定义
# app/users/__init__.py
from .api import router as users_router
from .services import UserService
from .models import User

__all__ = ["users_router", "UserService", "User"]

# 主应用中注册模块
# app/main.py
from fastapi import FastAPI
from app.users import users_router
from app.auth import auth_router
from app.orders import orders_router

app = FastAPI()

# 注册模块路由
app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(users_router, prefix="/api/v1/users", tags=["users"])
app.include_router(orders_router, prefix="/api/v1/orders", tags=["orders"])
```

### 2. 插件化架构

```python
# 插件系统设计
# app/core/plugins.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any

class Plugin(ABC):
    """插件基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """插件名称"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """插件版本"""
        pass
    
    @abstractmethod
    async def initialize(self, app: Any) -> None:
        """初始化插件"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """清理插件"""
        pass

class PluginManager:
    """插件管理器"""
    
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.hooks: Dict[str, List[callable]] = {}
    
    def register_plugin(self, plugin: Plugin):
        """注册插件"""
        self.plugins[plugin.name] = plugin
    
    def register_hook(self, event: str, callback: callable):
        """注册钩子"""
        if event not in self.hooks:
            self.hooks[event] = []
        self.hooks[event].append(callback)
    
    async def trigger_hook(self, event: str, *args, **kwargs):
        """触发钩子"""
        if event in self.hooks:
            for callback in self.hooks[event]:
                await callback(*args, **kwargs)
    
    async def initialize_all(self, app: Any):
        """初始化所有插件"""
        for plugin in self.plugins.values():
            await plugin.initialize(app)

# 缓存插件示例
# app/plugins/cache_plugin.py
from app.core.plugins import Plugin
from app.core.cache import CacheManager

class CachePlugin(Plugin):
    """缓存插件"""
    
    @property
    def name(self) -> str:
        return "cache"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, app: Any) -> None:
        """初始化缓存"""
        cache_manager = CacheManager()
        await cache_manager.connect()
        app.state.cache = cache_manager
    
    async def cleanup(self) -> None:
        """清理缓存连接"""
        if hasattr(app.state, 'cache'):
            await app.state.cache.disconnect()

# 使用插件系统
# app/main.py
from app.core.plugins import PluginManager
from app.plugins.cache_plugin import CachePlugin
from app.plugins.monitoring_plugin import MonitoringPlugin

app = FastAPI()
plugin_manager = PluginManager()

# 注册插件
plugin_manager.register_plugin(CachePlugin())
plugin_manager.register_plugin(MonitoringPlugin())

@app.on_event("startup")
async def startup_event():
    await plugin_manager.initialize_all(app)
    await plugin_manager.trigger_hook("app_started", app)

@app.on_event("shutdown")
async def shutdown_event():
    await plugin_manager.trigger_hook("app_stopping", app)
```

## ⚙️ 配置管理

### 1. 分层配置系统

```python
# app/config.py
from pydantic import BaseSettings, Field
from typing import Optional, List
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class DatabaseSettings(BaseSettings):
    """数据库配置"""
    url: str = Field(..., env="DATABASE_URL")
    echo: bool = Field(False, env="DATABASE_ECHO")
    pool_size: int = Field(10, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(20, env="DATABASE_MAX_OVERFLOW")
    pool_timeout: int = Field(30, env="DATABASE_POOL_TIMEOUT")
    pool_recycle: int = Field(3600, env="DATABASE_POOL_RECYCLE")

class RedisSettings(BaseSettings):
    """Redis配置"""
    url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    max_connections: int = Field(10, env="REDIS_MAX_CONNECTIONS")
    retry_on_timeout: bool = Field(True, env="REDIS_RETRY_ON_TIMEOUT")
    socket_timeout: int = Field(5, env="REDIS_SOCKET_TIMEOUT")

class SecuritySettings(BaseSettings):
    """安全配置"""
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    password_min_length: int = Field(8, env="PASSWORD_MIN_LENGTH")
    allowed_hosts: List[str] = Field(["*"], env="ALLOWED_HOSTS")

class LoggingSettings(BaseSettings):
    """日志配置"""
    level: str = Field("INFO", env="LOG_LEVEL")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    file_path: Optional[str] = Field(None, env="LOG_FILE_PATH")
    max_file_size: int = Field(10 * 1024 * 1024, env="LOG_MAX_FILE_SIZE")  # 10MB
    backup_count: int = Field(5, env="LOG_BACKUP_COUNT")

class APISettings(BaseSettings):
    """API配置"""
    title: str = Field("My API", env="API_TITLE")
    description: str = Field("My API Description", env="API_DESCRIPTION")
    version: str = Field("1.0.0", env="API_VERSION")
    docs_url: Optional[str] = Field("/docs", env="API_DOCS_URL")
    redoc_url: Optional[str] = Field("/redoc", env="API_REDOC_URL")
    openapi_url: Optional[str] = Field("/openapi.json", env="API_OPENAPI_URL")
    cors_origins: List[str] = Field(["*"], env="CORS_ORIGINS")
    cors_methods: List[str] = Field(["*"], env="CORS_METHODS")
    cors_headers: List[str] = Field(["*"], env="CORS_HEADERS")

class Settings(BaseSettings):
    """主配置类"""
    environment: Environment = Field(Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    
    # 子配置
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    security: SecuritySettings = SecuritySettings()
    logging: LoggingSettings = LoggingSettings()
    api: APISettings = APISettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # 环境特定配置文件
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )

# 配置实例
settings = Settings()

# 环境特定配置
# config/development.py
class DevelopmentSettings(Settings):
    debug: bool = True
    database: DatabaseSettings = DatabaseSettings(
        url="postgresql://dev_user:dev_pass@localhost:5432/dev_db",
        echo=True
    )
    logging: LoggingSettings = LoggingSettings(level="DEBUG")

# config/production.py
class ProductionSettings(Settings):
    debug: bool = False
    api: APISettings = APISettings(
        docs_url=None,  # 生产环境禁用文档
        redoc_url=None,
        openapi_url=None
    )
    logging: LoggingSettings = LoggingSettings(
        level="WARNING",
        file_path="/var/log/app.log"
    )

# 配置工厂
def get_settings() -> Settings:
    """获取配置实例"""
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "development":
        return DevelopmentSettings()
    elif env == "production":
        return ProductionSettings()
    else:
        return Settings()
```

### 2. 配置验证和类型安全

```python
# app/core/config_validator.py
from pydantic import validator, root_validator
from typing import Any, Dict
import re

class ValidatedSettings(BaseSettings):
    """带验证的配置类"""
    
    @validator('database_url')
    def validate_database_url(cls, v):
        """验证数据库URL格式"""
        if not v.startswith(('postgresql://', 'mysql://', 'sqlite://')):
            raise ValueError('Database URL must start with postgresql://, mysql://, or sqlite://')
        return v
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        """验证密钥强度"""
        if len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters long')
        return v
    
    @validator('allowed_hosts')
    def validate_allowed_hosts(cls, v):
        """验证允许的主机"""
        for host in v:
            if host != '*' and not re.match(r'^[a-zA-Z0-9.-]+$', host):
                raise ValueError(f'Invalid host format: {host}')
        return v
    
    @root_validator
    def validate_environment_consistency(cls, values):
        """验证环境配置一致性"""
        env = values.get('environment')
        debug = values.get('debug')
        
        if env == Environment.PRODUCTION and debug:
            raise ValueError('Debug mode should not be enabled in production')
        
        if env == Environment.DEVELOPMENT and not debug:
            values['debug'] = True  # 开发环境自动启用调试
        
        return values

# 配置加载器
class ConfigLoader:
    """配置加载器"""
    
    def __init__(self):
        self._settings = None
        self._config_files = {
            Environment.DEVELOPMENT: "config/development.env",
            Environment.STAGING: "config/staging.env",
            Environment.PRODUCTION: "config/production.env",
            Environment.TESTING: "config/testing.env"
        }
    
    def load_config(self, env: Environment = None) -> Settings:
        """加载配置"""
        if self._settings is None:
            if env is None:
                env = Environment(os.getenv("ENVIRONMENT", "development"))
            
            # 加载基础配置
            config_file = self._config_files.get(env, ".env")
            
            if os.path.exists(config_file):
                self._settings = ValidatedSettings(_env_file=config_file)
            else:
                self._settings = ValidatedSettings()
            
            # 验证配置
            self._validate_config()
        
        return self._settings
    
    def _validate_config(self):
        """验证配置完整性"""
        required_settings = [
            'database.url',
            'security.secret_key'
        ]
        
        for setting_path in required_settings:
            value = self._get_nested_value(self._settings, setting_path)
            if not value:
                raise ValueError(f"Required setting '{setting_path}' is missing")
    
    def _get_nested_value(self, obj: Any, path: str) -> Any:
        """获取嵌套配置值"""
        keys = path.split('.')
        current = obj
        
        for key in keys:
            if hasattr(current, key):
                current = getattr(current, key)
            else:
                return None
        
        return current
    
    def reload_config(self):
        """重新加载配置"""
        self._settings = None
        return self.load_config()

# 全局配置实例
config_loader = ConfigLoader()
settings = config_loader.load_config()
```

## 📦 依赖管理

### 1. 依赖文件组织

```bash
# requirements/
├── base.txt              # 基础依赖
├── development.txt       # 开发依赖
├── production.txt        # 生产依赖
├── testing.txt          # 测试依赖
└── optional.txt         # 可选依赖

# requirements/base.txt
fastapi>=0.104.0,<0.105.0
uvicorn[standard]>=0.24.0,<0.25.0
pydantic>=2.5.0,<3.0.0
sqlalchemy>=2.0.0,<2.1.0
alembic>=1.13.0,<1.14.0
psycopg2-binary>=2.9.0,<3.0.0
redis>=5.0.0,<6.0.0
celery>=5.3.0,<5.4.0
python-jose[cryptography]>=3.3.0,<4.0.0
passlib[bcrypt]>=1.7.0,<2.0.0
python-multipart>=0.0.6,<0.1.0
email-validator>=2.1.0,<3.0.0

# requirements/development.txt
-r base.txt
pytest>=7.4.0,<8.0.0
pytest-asyncio>=0.21.0,<0.22.0
pytest-cov>=4.1.0,<5.0.0
black>=23.11.0,<24.0.0
isort>=5.12.0,<6.0.0
flake8>=6.1.0,<7.0.0
mypy>=1.7.0,<2.0.0
pre-commit>=3.6.0,<4.0.0
httpx>=0.25.0,<0.26.0  # for testing
factory-boy>=3.3.0,<4.0.0
faker>=20.1.0,<21.0.0

# requirements/production.txt
-r base.txt
gunicorn>=21.2.0,<22.0.0
sentry-sdk[fastapi]>=1.38.0,<2.0.0
prometheus-client>=0.19.0,<0.20.0

# requirements/testing.txt
-r base.txt
pytest>=7.4.0,<8.0.0
pytest-asyncio>=0.21.0,<0.22.0
pytest-cov>=4.1.0,<5.0.0
pytest-mock>=3.12.0,<4.0.0
httpx>=0.25.0,<0.26.0
testcontainers>=3.7.0,<4.0.0
```

### 2. pyproject.toml配置

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-backend-project"
version = "1.0.0"
description = "My Backend Project"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "fastapi>=0.104.0,<0.105.0",
    "uvicorn[standard]>=0.24.0,<0.25.0",
    "pydantic>=2.5.0,<3.0.0",
    "sqlalchemy>=2.0.0,<2.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0,<8.0.0",
    "pytest-asyncio>=0.21.0,<0.22.0",
    "pytest-cov>=4.1.0,<5.0.0",
    "black>=23.11.0,<24.0.0",
    "isort>=5.12.0,<6.0.0",
    "flake8>=6.1.0,<7.0.0",
    "mypy>=1.7.0,<2.0.0",
]
test = [
    "pytest>=7.4.0,<8.0.0",
    "pytest-asyncio>=0.21.0,<0.22.0",
    "pytest-cov>=4.1.0,<5.0.0",
    "httpx>=0.25.0,<0.26.0",
]
prod = [
    "gunicorn>=21.2.0,<22.0.0",
    "sentry-sdk[fastapi]>=1.38.0,<2.0.0",
]

[project.scripts]
dev = "app.main:run_dev"
start = "app.main:run_prod"
migrate = "app.db.migrations:run_migrations"

[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
(
  /(
      \.eggs
    | \.git
    | \.hg
    | \.mypy_cache
    | \.tox
    | \.venv
    | _build
    | buck-out
    | build
    | dist
    | migrations
  )/
)
'''

[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers --strict-config"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow (deselect with '-m "not slow"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
filterwarnings = [
    "error",
    "ignore::UserWarning",
    "ignore::DeprecationWarning",
]

[tool.coverage.run]
source = ["app"]
omit = [
    "*/tests/*",
    "*/migrations/*",
    "*/venv/*",
    "*/env/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\bProtocol\):",
    "@(abc\.)?abstractmethod",
]
```

## 📚 文档组织

### 1. 文档结构

```
docs/
├── README.md                 # 项目概述
├── CONTRIBUTING.md           # 贡献指南
├── CHANGELOG.md             # 变更日志
├── LICENSE                  # 许可证
│
├── api/                     # API文档
│   ├── overview.md          # API概述
│   ├── authentication.md   # 认证文档
│   ├── endpoints/           # 端点文档
│   │   ├── users.md
│   │   ├── auth.md
│   │   └── orders.md
│   └── examples/            # API示例
│       ├── curl.md
│       ├── python.md
│       └── javascript.md
│
├── development/             # 开发文档
│   ├── setup.md            # 环境搭建
│   ├── coding-standards.md # 编码规范
│   ├── testing.md          # 测试指南
│   ├── debugging.md        # 调试指南
│   └── tools.md            # 开发工具
│
├── deployment/              # 部署文档
│   ├── docker.md           # Docker部署
│   ├── kubernetes.md       # Kubernetes部署
│   ├── monitoring.md       # 监控配置
│   └── troubleshooting.md  # 故障排除
│
├── architecture/            # 架构文档
│   ├── overview.md         # 架构概述
│   ├── database.md         # 数据库设计
│   ├── security.md         # 安全架构
│   └── performance.md      # 性能考虑
│
└── guides/                  # 使用指南
    ├── quick-start.md      # 快速开始
    ├── user-guide.md       # 用户指南
    ├── admin-guide.md      # 管理员指南
    └── migration-guide.md  # 迁移指南
```

### 2. 自动化文档生成

```python
# scripts/generate_docs.py
import os
import inspect
from typing import get_type_hints
from fastapi import FastAPI
from app.main import app

class DocumentationGenerator:
    """文档生成器"""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.output_dir = "docs/api/generated"
    
    def generate_api_docs(self):
        """生成API文档"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 生成端点文档
        for route in self.app.routes:
            if hasattr(route, 'endpoint'):
                self._generate_endpoint_doc(route)
        
        # 生成模型文档
        self._generate_models_doc()
        
        # 生成OpenAPI文档
        self._generate_openapi_doc()
    
    def _generate_endpoint_doc(self, route):
        """生成端点文档"""
        endpoint = route.endpoint
        doc_content = f"# {route.path}\n\n"
        
        # 添加方法信息
        methods = getattr(route, 'methods', [])
        doc_content += f"**Methods:** {', '.join(methods)}\n\n"
        
        # 添加函数文档
        if endpoint.__doc__:
            doc_content += f"## Description\n\n{endpoint.__doc__}\n\n"
        
        # 添加参数信息
        sig = inspect.signature(endpoint)
        type_hints = get_type_hints(endpoint)
        
        if sig.parameters:
            doc_content += "## Parameters\n\n"
            for param_name, param in sig.parameters.items():
                param_type = type_hints.get(param_name, 'Any')
                doc_content += f"- **{param_name}** ({param_type}): {param.annotation}\n"
        
        # 保存文档
        filename = f"{route.path.replace('/', '_').strip('_')}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(doc_content)
    
    def _generate_models_doc(self):
        """生成模型文档"""
        from app import models
        
        doc_content = "# Data Models\n\n"
        
        for name in dir(models):
            obj = getattr(models, name)
            if inspect.isclass(obj) and hasattr(obj, '__annotations__'):
                doc_content += f"## {name}\n\n"
                
                if obj.__doc__:
                    doc_content += f"{obj.__doc__}\n\n"
                
                # 添加字段信息
                doc_content += "### Fields\n\n"
                for field_name, field_type in obj.__annotations__.items():
                    doc_content += f"- **{field_name}** ({field_type})\n"
                
                doc_content += "\n"
        
        filepath = os.path.join(self.output_dir, "models.md")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(doc_content)
    
    def _generate_openapi_doc(self):
        """生成OpenAPI文档"""
        import json
        
        openapi_schema = self.app.openapi()
        
        filepath = os.path.join(self.output_dir, "openapi.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(openapi_schema, f, indent=2, ensure_ascii=False)

# 使用文档生成器
if __name__ == "__main__":
    generator = DocumentationGenerator(app)
    generator.generate_api_docs()
    print("API documentation generated successfully!")
```

## 🧪 测试结构

### 1. 测试目录组织

```
tests/
├── conftest.py              # 测试配置和夹具
├── test_main.py             # 主应用测试
│
├── unit/                    # 单元测试
│   ├── __init__.py
│   ├── test_models/         # 模型测试
│   │   ├── __init__.py
│   │   ├── test_user.py
│   │   └── test_order.py
│   ├── test_services/       # 服务测试
│   │   ├── __init__.py
│   │   ├── test_user_service.py
│   │   └── test_order_service.py
│   ├── test_repositories/   # 仓库测试
│   │   ├── __init__.py
│   │   ├── test_user_repo.py
│   │   └── test_order_repo.py
│   └── test_utils/          # 工具测试
│       ├── __init__.py
│       ├── test_validators.py
│       └── test_helpers.py
│
├── integration/             # 集成测试
│   ├── __init__.py
│   ├── test_api/           # API集成测试
│   │   ├── __init__.py
│   │   ├── test_auth_api.py
│   │   ├── test_user_api.py
│   │   └── test_order_api.py
│   ├── test_database/      # 数据库集成测试
│   │   ├── __init__.py
│   │   └── test_migrations.py
│   └── test_external/      # 外部服务测试
│       ├── __init__.py
│       ├── test_redis.py
│       └── test_email.py
│
├── e2e/                    # 端到端测试
│   ├── __init__.py
│   ├── test_user_journey.py
│   └── test_order_flow.py
│
├── performance/            # 性能测试
│   ├── __init__.py
│   ├── test_load.py
│   └── test_stress.py
│
├── fixtures/               # 测试数据
│   ├── __init__.py
│   ├── users.json
│   ├── orders.json
│   └── test_data.py
│
└── utils/                  # 测试工具
    ├── __init__.py
    ├── factories.py        # 数据工厂
    ├── helpers.py          # 测试辅助函数
    └── mocks.py            # 模拟对象
```

### 2. 测试配置和夹具

```python
# tests/conftest.py
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.database import get_db, Base
from app.core.config import get_settings
from app.core.security import create_access_token
from tests.utils.factories import UserFactory, OrderFactory

# 测试数据库配置
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def db_session():
    """创建数据库会话"""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def override_get_db(db_session):
    """覆盖数据库依赖"""
    def _override_get_db():
        try:
            yield db_session
        finally:
            db_session.close()
    
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def client(override_get_db) -> Generator[TestClient, None, None]:
    """创建测试客户端"""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture(scope="function")
async def async_client(override_get_db) -> AsyncGenerator[AsyncClient, None]:
    """创建异步测试客户端"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def test_user(db_session):
    """创建测试用户"""
    user = UserFactory.create()
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user

@pytest.fixture
def test_admin_user(db_session):
    """创建测试管理员用户"""
    user = UserFactory.create(is_admin=True)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user

@pytest.fixture
def auth_headers(test_user):
    """创建认证头"""
    access_token = create_access_token(data={"sub": str(test_user.id)})
    return {"Authorization": f"Bearer {access_token}"}

@pytest.fixture
def admin_auth_headers(test_admin_user):
    """创建管理员认证头"""
    access_token = create_access_token(data={"sub": str(test_admin_user.id)})
    return {"Authorization": f"Bearer {access_token}"}

@pytest.fixture
def test_orders(db_session, test_user):
    """创建测试订单"""
    orders = OrderFactory.create_batch(3, user_id=test_user.id)
    for order in orders:
        db_session.add(order)
    db_session.commit()
    return orders

# 测试配置
@pytest.fixture
def test_settings():
    """测试配置"""
    from app.core.config import Settings
    return Settings(
        environment="testing",
        database_url="sqlite:///./test.db",
        secret_key="test-secret-key-for-testing-only",
        access_token_expire_minutes=30
    )

# 模拟外部服务
@pytest.fixture
def mock_redis(monkeypatch):
    """模拟Redis"""
    class MockRedis:
        def __init__(self):
            self.data = {}
        
        async def get(self, key):
            return self.data.get(key)
        
        async def set(self, key, value, ex=None):
            self.data[key] = value
        
        async def delete(self, key):
            self.data.pop(key, None)
    
    mock_redis_instance = MockRedis()
    monkeypatch.setattr("app.core.cache.redis_client", mock_redis_instance)
    return mock_redis_instance

@pytest.fixture
def mock_email_service(monkeypatch):
    """模拟邮件服务"""
    sent_emails = []
    
    async def mock_send_email(to: str, subject: str, body: str):
        sent_emails.append({
            "to": to,
            "subject": subject,
            "body": body
        })
    
    monkeypatch.setattr("app.services.email_service.send_email", mock_send_email)
    return sent_emails
```

## 🚀 部署结构

### 1. 容器化配置

```dockerfile
# Dockerfile
FROM python:3.11-slim as base

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 创建应用用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements/production.txt requirements.txt

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置文件权限
RUN chown -R appuser:appuser /app

# 切换到应用用户
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# 多阶段构建 - 开发环境
FROM base as development

USER root

# 安装开发依赖
COPY requirements/development.txt dev-requirements.txt
RUN pip install --no-cache-dir -r dev-requirements.txt

USER appuser

# 开发环境启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### 2. Kubernetes配置

```yaml
# k8s/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-app
  labels:
    app: backend-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend-app
  template:
    metadata:
      labels:
        app: backend-app
    spec:
      containers:
      - name: backend-app
        image: backend-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: backend-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: backend-secrets
              key: secret-key
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
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
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: config-volume
        configMap:
          name: backend-config
      imagePullSecrets:
      - name: registry-secret

---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  selector:
    app: backend-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: backend-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: backend-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-service
            port:
              number: 80
```

### 3. 环境特定配置

```yaml
# k8s/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
- ../../base

patchesStrategicMerge:
- deployment-patch.yaml
- service-patch.yaml

configMapGenerator:
- name: backend-config
  files:
  - config/production.env

secretGenerator:
- name: backend-secrets
  literals:
  - database-url=postgresql://prod_user:prod_pass@prod-db:5432/prod_db
  - secret-key=super-secret-production-key

images:
- name: backend-app
  newTag: v1.0.0

replicas:
- name: backend-app
  count: 5

---
# k8s/overlays/production/deployment-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-app
spec:
  template:
    spec:
      containers:
      - name: backend-app
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        env:
        - name: LOG_LEVEL
          value: "WARNING"
        - name: DEBUG
          value: "false"
```

## 📋 项目结构检查清单

### ✅ 目录结构检查

- [ ] 项目根目录包含必要的配置文件（README.md, requirements.txt, pyproject.toml）
- [ ] 应用代码按功能模块清晰组织
- [ ] 测试目录结构与应用代码结构对应
- [ ] 文档目录包含完整的项目文档
- [ ] 部署配置文件组织良好

### ✅ 代码组织检查

- [ ] 每个模块都有明确的职责
- [ ] 依赖关系清晰，避免循环依赖
- [ ] 使用依赖注入管理组件依赖
- [ ] 配置管理统一且类型安全
- [ ] 错误处理统一且完善

### ✅ 文档完整性检查

- [ ] API文档完整且最新
- [ ] 部署文档详细且可操作
- [ ] 开发文档包含环境搭建和编码规范
- [ ] 架构文档描述系统设计
- [ ] 用户指南易于理解

### ✅ 测试覆盖检查

- [ ] 单元测试覆盖核心业务逻辑
- [ ] 集成测试覆盖关键流程
- [ ] API测试覆盖所有端点
- [ ] 测试数据和夹具管理良好
- [ ] 测试环境隔离且可重复

### ✅ 部署就绪检查

- [ ] Docker镜像构建优化
- [ ] Kubernetes配置完整
- [ ] 环境配置分离
- [ ] 健康检查配置正确
- [ ] 监控和日志配置完善

## 🔧 常用工具和脚本

### 1. 项目初始化脚本

```bash
#!/bin/bash
# scripts/init_project.sh

set -e

echo "Initializing project structure..."

# 创建目录结构
mkdir -p app/{api/v1/endpoints,core,models,schemas,services,repositories,utils,db}
mkdir -p tests/{unit,integration,e2e,fixtures,utils}
mkdir -p docs/{api,development,deployment,architecture,guides}
mkdir -p scripts
mkdir -p k8s/{base,overlays/{development,staging,production}}
mkdir -p requirements

# 创建__init__.py文件
find app tests -type d -exec touch {}/__init__.py \;

# 创建基础配置文件
touch .env.example
touch .gitignore
touch README.md
touch requirements/base.txt
touch requirements/development.txt
touch requirements/production.txt

echo "Project structure initialized successfully!"
```

### 2. 代码质量检查脚本

```bash
#!/bin/bash
# scripts/check_quality.sh

set -e

echo "Running code quality checks..."

# 代码格式化
echo "Formatting code with black..."
black app tests

# 导入排序
echo "Sorting imports with isort..."
isort app tests

# 代码检查
echo "Running flake8..."
flake8 app tests

# 类型检查
echo "Running mypy..."
mypy app

# 安全检查
echo "Running bandit..."
bandit -r app

echo "Code quality checks completed!"
```

### 3. 测试运行脚本

```bash
#!/bin/bash
# scripts/run_tests.sh

set -e

echo "Running tests..."

# 单元测试
echo "Running unit tests..."
pytest tests/unit -v --cov=app --cov-report=term-missing

# 集成测试
echo "Running integration tests..."
pytest tests/integration -v

# 端到端测试
echo "Running e2e tests..."
pytest tests/e2e -v

echo "All tests completed!"
```

### 4. 部署脚本

```bash
#!/bin/bash
# scripts/deploy.sh

set -e

ENVIRONMENT=${1:-development}
IMAGE_TAG=${2:-latest}

echo "Deploying to $ENVIRONMENT environment..."

# 构建Docker镜像
echo "Building Docker image..."
docker build -t backend-app:$IMAGE_TAG .

# 推送镜像到仓库
if [ "$ENVIRONMENT" != "development" ]; then
    echo "Pushing image to registry..."
    docker tag backend-app:$IMAGE_TAG registry.example.com/backend-app:$IMAGE_TAG
    docker push registry.example.com/backend-app:$IMAGE_TAG
fi

# 部署到Kubernetes
echo "Deploying to Kubernetes..."
kubectl apply -k k8s/overlays/$ENVIRONMENT

# 等待部署完成
echo "Waiting for deployment to complete..."
kubectl rollout status deployment/backend-app -n $ENVIRONMENT

echo "Deployment to $ENVIRONMENT completed successfully!"
```

## 📚 相关资源

### 推荐阅读

- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Domain Driven Design](https://martinfowler.com/bliki/DomainDrivenDesign.html)
- [Microservices Patterns](https://microservices.io/patterns/)
- [12-Factor App](https://12factor.net/)

### 工具推荐

- **代码质量**: black, isort, flake8, mypy, bandit
- **测试**: pytest, pytest-asyncio, pytest-cov, factory-boy
- **文档**: mkdocs, sphinx, swagger-ui
- **部署**: docker, kubernetes, helm, kustomize
- **监控**: prometheus, grafana, jaeger, sentry

### 模板和示例

- [FastAPI项目模板](https://github.com/tiangolo/full-stack-fastapi-postgresql)
- [Django项目模板](https://github.com/cookiecutter/cookiecutter-django)
- [Flask项目模板](https://github.com/cookiecutter/cookiecutter-flask)

---

遵循这些最佳实践将帮助您构建可维护、可扩展、高质量的Python后端项目。记住，项目结构应该随着项目的发展而演进，但始终保持清晰和一致性。