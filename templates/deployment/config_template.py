"""配置管理模板

提供完整的配置管理功能，包括：
- 环境配置管理
- 配置验证
- 配置加载和保存
- 多环境支持
- 敏感信息处理
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type

from pydantic import BaseModel, Field, validator, SecretStr
import yaml


class Environment(str, Enum):
    """环境枚举"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseConfig(BaseModel):
    """数据库配置"""

    host: str = "localhost"
    port: int = 5432
    database: str = "app_db"
    username: str = "postgres"
    password: SecretStr = SecretStr("password")
    driver: str = "postgresql"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False

    @validator("port")
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    def get_url(self, async_driver: bool = False) -> str:
        """获取数据库连接URL"""
        driver = self.driver
        if async_driver and driver == "postgresql":
            driver = "postgresql+asyncpg"
        elif async_driver and driver == "mysql":
            driver = "mysql+aiomysql"

        return f"{driver}://{self.username}:{self.password.get_secret_value()}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseModel):
    """Redis配置"""

    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[SecretStr] = None
    max_connections: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    decode_responses: bool = True

    @validator("port")
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    @validator("database")
    def validate_database(cls, v):
        if not 0 <= v <= 15:
            raise ValueError("Redis database must be between 0 and 15")
        return v

    def get_url(self) -> str:
        """获取Redis连接URL"""
        if self.password:
            return f"redis://:{self.password.get_secret_value()}@{self.host}:{self.port}/{self.database}"
        else:
            return f"redis://{self.host}:{self.port}/{self.database}"


class LoggingConfig(BaseModel):
    """日志配置"""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # 文件日志
    file_enabled: bool = True
    file_path: str = "logs/app.log"
    file_max_size: int = 10 * 1024 * 1024  # 10MB
    file_backup_count: int = 5

    # 控制台日志
    console_enabled: bool = True

    # 结构化日志
    structured: bool = True

    @validator("level")
    def validate_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class SecurityConfig(BaseModel):
    """安全配置"""

    secret_key: SecretStr
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # CORS配置
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    cors_methods: List[str] = Field(default_factory=lambda: ["*"])
    cors_headers: List[str] = Field(default_factory=lambda: ["*"])

    # 密码策略
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special: bool = True

    # 限流配置
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # 秒

    @validator("access_token_expire_minutes")
    def validate_access_token_expire(cls, v):
        if v <= 0:
            raise ValueError("Access token expire minutes must be positive")
        return v

    @validator("refresh_token_expire_days")
    def validate_refresh_token_expire(cls, v):
        if v <= 0:
            raise ValueError("Refresh token expire days must be positive")
        return v


class CacheConfig(BaseModel):
    """缓存配置"""

    enabled: bool = True
    default_ttl: int = 3600  # 秒
    key_prefix: str = "app:"
    serialization: str = "json"  # json, pickle

    # 缓存策略
    strategy: str = "lru"  # lru, lfu, fifo
    max_size: int = 1000

    @validator("serialization")
    def validate_serialization(cls, v):
        valid_types = ["json", "pickle"]
        if v not in valid_types:
            raise ValueError(f"Serialization must be one of {valid_types}")
        return v


class MonitoringConfig(BaseModel):
    """监控配置"""

    enabled: bool = True

    # 指标收集
    metrics_enabled: bool = True
    metrics_port: int = 8001

    # 健康检查
    health_check_enabled: bool = True
    health_check_path: str = "/health"

    # 错误追踪
    error_tracking_enabled: bool = True
    sentry_dsn: Optional[SecretStr] = None

    # 性能监控
    performance_monitoring: bool = True
    slow_query_threshold: float = 1.0  # 秒

    @validator("metrics_port")
    def validate_metrics_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Metrics port must be between 1 and 65535")
        return v


class EnvironmentConfig(BaseModel):
    """环境配置"""

    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    testing: bool = False

    # 应用配置
    app_name: str = "Python Backend App"
    app_version: str = "1.0.0"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # 数据库配置
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    # Redis配置
    redis: RedisConfig = Field(default_factory=RedisConfig)

    # 日志配置
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # 安全配置
    security: SecurityConfig

    # 缓存配置
    cache: CacheConfig = Field(default_factory=CacheConfig)

    # 监控配置
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # 自定义配置
    custom: Dict[str, Any] = Field(default_factory=dict)

    @validator("app_port")
    def validate_app_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("App port must be between 1 and 65535")
        return v

    @validator("environment")
    def set_environment_defaults(cls, v, values):
        """根据环境设置默认值"""
        if v == Environment.PRODUCTION:
            values["debug"] = False
            values["testing"] = False
        elif v == Environment.TESTING:
            values["debug"] = True
            values["testing"] = True
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._config: Optional[EnvironmentConfig] = None

    def load_config(
        self,
        environment: Optional[Environment] = None,
        config_file: Optional[str] = None,
    ) -> EnvironmentConfig:
        """加载配置"""
        if config_file:
            config_path = Path(config_file)
        else:
            env = environment or self._detect_environment()
            config_path = self.config_dir / f"{env.value}.yml"

        if config_path.exists():
            config_data = self._load_from_file(config_path)
        else:
            config_data = {}

        # 从环境变量加载
        env_config = self._load_from_env()
        config_data.update(env_config)

        # 创建配置对象
        self._config = EnvironmentConfig(**config_data)
        return self._config

    def save_config(
        self, config: EnvironmentConfig, config_file: Optional[str] = None
    ) -> None:
        """保存配置"""
        if config_file:
            config_path = Path(config_file)
        else:
            config_path = self.config_dir / f"{config.environment.value}.yml"

        config_data = self._config_to_dict(config)
        self._save_to_file(config_data, config_path)

    def get_config(self) -> Optional[EnvironmentConfig]:
        """获取当前配置"""
        return self._config

    def validate_config(self, config: EnvironmentConfig) -> List[str]:
        """验证配置"""
        errors = []

        try:
            # 验证数据库连接
            config.database.get_url()
        except Exception as e:
            errors.append(f"Database config error: {e}")

        try:
            # 验证Redis连接
            config.redis.get_url()
        except Exception as e:
            errors.append(f"Redis config error: {e}")

        # 验证日志目录
        log_dir = Path(config.logging.file_path).parent
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create log directory: {e}")

        # 验证安全配置
        if len(config.security.secret_key.get_secret_value()) < 32:
            errors.append("Secret key should be at least 32 characters long")

        return errors

    def _detect_environment(self) -> Environment:
        """检测当前环境"""
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        try:
            return Environment(env_name)
        except ValueError:
            return Environment.DEVELOPMENT

    def _load_from_file(self, config_path: Path) -> Dict[str, Any]:
        """从文件加载配置"""
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yml", ".yaml"]:
                return yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == ".json":
                return json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}"
                )

    def _load_from_env(self) -> Dict[str, Any]:
        """从环境变量加载配置"""
        env_config = {}

        # 应用配置
        if os.getenv("APP_NAME"):
            env_config["app_name"] = os.getenv("APP_NAME")
        if os.getenv("APP_VERSION"):
            env_config["app_version"] = os.getenv("APP_VERSION")
        if os.getenv("APP_HOST"):
            env_config["app_host"] = os.getenv("APP_HOST")
        if os.getenv("APP_PORT"):
            env_config["app_port"] = int(os.getenv("APP_PORT"))

        # 数据库配置
        database_config = {}
        if os.getenv("DATABASE_URL"):
            # 解析数据库URL
            db_url = os.getenv("DATABASE_URL")
            # 这里可以添加URL解析逻辑
        else:
            if os.getenv("DB_HOST"):
                database_config["host"] = os.getenv("DB_HOST")
            if os.getenv("DB_PORT"):
                database_config["port"] = int(os.getenv("DB_PORT"))
            if os.getenv("DB_NAME"):
                database_config["database"] = os.getenv("DB_NAME")
            if os.getenv("DB_USER"):
                database_config["username"] = os.getenv("DB_USER")
            if os.getenv("DB_PASSWORD"):
                database_config["password"] = os.getenv("DB_PASSWORD")

        if database_config:
            env_config["database"] = database_config

        # Redis配置
        redis_config = {}
        if os.getenv("REDIS_URL"):
            # 解析Redis URL
            redis_url = os.getenv("REDIS_URL")
            # 这里可以添加URL解析逻辑
        else:
            if os.getenv("REDIS_HOST"):
                redis_config["host"] = os.getenv("REDIS_HOST")
            if os.getenv("REDIS_PORT"):
                redis_config["port"] = int(os.getenv("REDIS_PORT"))
            if os.getenv("REDIS_DB"):
                redis_config["database"] = int(os.getenv("REDIS_DB"))
            if os.getenv("REDIS_PASSWORD"):
                redis_config["password"] = os.getenv("REDIS_PASSWORD")

        if redis_config:
            env_config["redis"] = redis_config

        # 安全配置
        security_config = {}
        if os.getenv("SECRET_KEY"):
            security_config["secret_key"] = os.getenv("SECRET_KEY")
        if os.getenv("SENTRY_DSN"):
            if "monitoring" not in env_config:
                env_config["monitoring"] = {}
            env_config["monitoring"]["sentry_dsn"] = os.getenv("SENTRY_DSN")

        if security_config:
            env_config["security"] = security_config

        return env_config

    def _save_to_file(self, config_data: Dict[str, Any], config_path: Path) -> None:
        """保存配置到文件"""
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yml", ".yaml"]:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            elif config_path.suffix.lower() == ".json":
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}"
                )

    def _config_to_dict(self, config: EnvironmentConfig) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        config_dict = config.dict()

        # 处理敏感信息
        if "security" in config_dict and "secret_key" in config_dict["security"]:
            config_dict["security"]["secret_key"] = "[HIDDEN]"

        if "database" in config_dict and "password" in config_dict["database"]:
            config_dict["database"]["password"] = "[HIDDEN]"

        if "redis" in config_dict and "password" in config_dict["redis"]:
            config_dict["redis"]["password"] = "[HIDDEN]"

        if "monitoring" in config_dict and "sentry_dsn" in config_dict["monitoring"]:
            config_dict["monitoring"]["sentry_dsn"] = "[HIDDEN]"

        return config_dict

    def generate_env_template(self, output_file: str = ".env.template") -> None:
        """生成环境变量模板文件"""
        template_content = """# 应用配置
APP_NAME=Python Backend App
APP_VERSION=1.0.0
APP_HOST=0.0.0.0
APP_PORT=8000
ENVIRONMENT=development

# 数据库配置
DB_HOST=localhost
DB_PORT=5432
DB_NAME=app_db
DB_USER=postgres
DB_PASSWORD=password

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# 安全配置
SECRET_KEY=your-secret-key-here-at-least-32-characters-long

# 监控配置
SENTRY_DSN=

# 自定义配置
# CUSTOM_VAR=value
"""

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(template_content)


# 全局配置管理器
config_manager = ConfigManager()
current_config: Optional[EnvironmentConfig] = None


# 便捷函数
def load_config(
    environment: Optional[Environment] = None, config_file: Optional[str] = None
) -> EnvironmentConfig:
    """加载配置"""
    global current_config
    current_config = config_manager.load_config(environment, config_file)
    return current_config


def get_config() -> Optional[EnvironmentConfig]:
    """获取当前配置"""
    return current_config or config_manager.get_config()


def set_config(config: EnvironmentConfig) -> None:
    """设置当前配置"""
    global current_config
    current_config = config


def validate_config(config: Optional[EnvironmentConfig] = None) -> List[str]:
    """验证配置"""
    config = config or get_config()
    if not config:
        return ["No configuration loaded"]
    return config_manager.validate_config(config)


# 使用示例
if __name__ == "__main__":
    # 创建配置管理器
    manager = ConfigManager()

    # 生成环境变量模板
    manager.generate_env_template()

    # 创建示例配置
    config = EnvironmentConfig(
        environment=Environment.DEVELOPMENT,
        app_name="My Python App",
        security=SecurityConfig(
            secret_key="your-secret-key-here-at-least-32-characters-long"
        ),
    )

    # 保存配置
    manager.save_config(config)

    # 加载配置
    loaded_config = manager.load_config(Environment.DEVELOPMENT)

    # 验证配置
    errors = manager.validate_config(loaded_config)
    if errors:
        print(f"Configuration errors: {errors}")
    else:
        print("Configuration is valid")

    # 打印配置信息
    print(f"App: {loaded_config.app_name}")
    print(f"Environment: {loaded_config.environment}")
    print(f"Database URL: {loaded_config.database.get_url()}")
    print(f"Redis URL: {loaded_config.redis.get_url()}")
