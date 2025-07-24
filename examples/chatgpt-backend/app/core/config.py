# -*- coding: utf-8 -*-
"""
应用配置

定义应用的配置设置。
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import lru_cache

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from pydantic import BaseSettings, validator, Field
from pydantic.networks import AnyHttpUrl

# 导入模板库配置
from src.core.config import BaseConfig


class Settings(BaseConfig):
    """
    应用配置类
    """
    
    # 应用基本信息
    APP_NAME: str = "ChatGPT Backend"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "ChatGPT客户端后端API服务"
    
    # API配置
    API_V1_STR: str = "/api/v1"
    
    # OpenAI配置
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_API_BASE: Optional[str] = Field(None, env="OPENAI_API_BASE")
    OPENAI_ORGANIZATION: Optional[str] = Field(None, env="OPENAI_ORGANIZATION")
    OPENAI_DEFAULT_MODEL: str = "gpt-3.5-turbo"
    OPENAI_MAX_TOKENS: int = 4096
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_TIMEOUT: int = 60
    
    # 支持的模型列表
    SUPPORTED_MODELS: List[str] = [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-4",
        "gpt-4-32k",
        "gpt-4-turbo-preview",
        "gpt-4-vision-preview"
    ]
    
    # 模型配置
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "gpt-3.5-turbo": {
            "max_tokens": 4096,
            "context_window": 4096,
            "cost_per_1k_tokens": 0.002
        },
        "gpt-3.5-turbo-16k": {
            "max_tokens": 16384,
            "context_window": 16384,
            "cost_per_1k_tokens": 0.004
        },
        "gpt-4": {
            "max_tokens": 8192,
            "context_window": 8192,
            "cost_per_1k_tokens": 0.03
        },
        "gpt-4-32k": {
            "max_tokens": 32768,
            "context_window": 32768,
            "cost_per_1k_tokens": 0.06
        },
        "gpt-4-turbo-preview": {
            "max_tokens": 4096,
            "context_window": 128000,
            "cost_per_1k_tokens": 0.01
        },
        "gpt-4-vision-preview": {
            "max_tokens": 4096,
            "context_window": 128000,
            "cost_per_1k_tokens": 0.01
        }
    }
    
    # 用户配额配置
    DEFAULT_USER_QUOTA: Dict[str, int] = {
        "daily_requests": 100,
        "monthly_requests": 3000,
        "daily_tokens": 100000,
        "monthly_tokens": 3000000
    }
    
    # VIP用户配额配置
    VIP_USER_QUOTA: Dict[str, int] = {
        "daily_requests": 1000,
        "monthly_requests": 30000,
        "daily_tokens": 1000000,
        "monthly_tokens": 30000000
    }
    
    # 对话配置
    MAX_CONVERSATION_HISTORY: int = 50
    MAX_MESSAGE_LENGTH: int = 8000
    MAX_CONVERSATIONS_PER_USER: int = 100
    CONVERSATION_TITLE_MAX_LENGTH: int = 100
    
    # 文件上传配置
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = [
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "text/plain",
        "application/pdf"
    ]
    UPLOAD_DIR: str = "uploads"
    
    # WebSocket配置
    WEBSOCKET_HEARTBEAT_INTERVAL: int = 30
    WEBSOCKET_MAX_CONNECTIONS: int = 1000
    WEBSOCKET_MESSAGE_MAX_SIZE: int = 1024 * 1024  # 1MB
    
    # 缓存配置
    CACHE_CONVERSATION_TTL: int = 3600  # 1小时
    CACHE_USER_TTL: int = 1800  # 30分钟
    CACHE_MODEL_TTL: int = 86400  # 24小时
    
    # 限流配置
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_REQUESTS_PER_HOUR: int = 1000
    RATE_LIMIT_REQUESTS_PER_DAY: int = 10000
    
    # 管理员用户配置
    ADMIN_USERS: List[str] = Field(default_factory=list, env="ADMIN_USERS")
    
    # 邮件配置（可选）
    SMTP_HOST: Optional[str] = Field(None, env="SMTP_HOST")
    SMTP_PORT: Optional[int] = Field(587, env="SMTP_PORT")
    SMTP_USERNAME: Optional[str] = Field(None, env="SMTP_USERNAME")
    SMTP_PASSWORD: Optional[str] = Field(None, env="SMTP_PASSWORD")
    SMTP_USE_TLS: bool = Field(True, env="SMTP_USE_TLS")
    EMAIL_FROM: Optional[str] = Field(None, env="EMAIL_FROM")
    
    # 监控配置
    ENABLE_METRICS: bool = Field(True, env="ENABLE_METRICS")
    METRICS_PATH: str = "/metrics"
    HEALTH_CHECK_PATH: str = "/health"
    
    # 日志配置
    LOG_REQUESTS: bool = Field(True, env="LOG_REQUESTS")
    LOG_RESPONSES: bool = Field(False, env="LOG_RESPONSES")
    LOG_SLOW_REQUESTS: bool = Field(True, env="LOG_SLOW_REQUESTS")
    SLOW_REQUEST_THRESHOLD: float = 1.0  # 秒
    
    # 安全配置
    ENABLE_CORS: bool = Field(True, env="ENABLE_CORS")
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    CORS_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_HEADERS: List[str] = ["*"]
    
    # 备份配置
    ENABLE_AUTO_BACKUP: bool = Field(False, env="ENABLE_AUTO_BACKUP")
    BACKUP_INTERVAL_HOURS: int = Field(24, env="BACKUP_INTERVAL_HOURS")
    BACKUP_RETENTION_DAYS: int = Field(30, env="BACKUP_RETENTION_DAYS")
    BACKUP_DIR: str = "backups"
    
    @validator("OPENAI_API_KEY")
    def validate_openai_api_key(cls, v):
        if not v or not v.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")
        return v
    
    @validator("SUPPORTED_MODELS")
    def validate_supported_models(cls, v):
        if not v:
            raise ValueError("At least one model must be supported")
        return v
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("ADMIN_USERS", pre=True)
    def parse_admin_users(cls, v):
        if isinstance(v, str):
            return [user.strip() for user in v.split(",") if user.strip()]
        return v
    
    def get_model_config(self, model: str) -> Dict[str, Any]:
        """
        获取模型配置
        """
        return self.MODEL_CONFIGS.get(model, self.MODEL_CONFIGS[self.OPENAI_DEFAULT_MODEL])
    
    def is_model_supported(self, model: str) -> bool:
        """
        检查模型是否支持
        """
        return model in self.SUPPORTED_MODELS
    
    def is_admin_user(self, username: str) -> bool:
        """
        检查是否为管理员用户
        """
        return username in self.ADMIN_USERS
    
    def get_user_quota(self, is_vip: bool = False) -> Dict[str, int]:
        """
        获取用户配额
        """
        return self.VIP_USER_QUOTA if is_vip else self.DEFAULT_USER_QUOTA
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    获取应用配置实例（单例模式）
    """
    return Settings()


# 导出配置实例
settings = get_settings()