#!/usr/bin/env python3
"""
应用配置管理模块

提供统一的配置管理，支持：
1. 环境变量配置
2. 配置文件加载
3. 配置验证
4. 多环境支持
5. 配置缓存
"""

import os
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    BaseSettings,
    Field,
    validator,
    root_validator,
    AnyHttpUrl,
    PostgresDsn,
    RedisDsn,
)
from pydantic.env_settings import SettingsSourceCallable


class Settings(BaseSettings):
    """
    应用配置类
    
    使用Pydantic进行配置验证和类型转换
    """
    
    # =============================================================================
    # 应用基础配置
    # =============================================================================
    app_name: str = Field(default="Enterprise Application", description="应用名称")
    app_version: str = Field(default="1.0.0", description="应用版本")
    app_description: str = Field(default="Production-ready enterprise application", description="应用描述")
    app_environment: str = Field(default="development", description="运行环境")
    app_debug: bool = Field(default=False, description="调试模式")
    app_host: str = Field(default="0.0.0.0", description="监听主机")
    app_port: int = Field(default=8000, description="监听端口")
    app_workers: int = Field(default=4, description="工作进程数")
    app_reload: bool = Field(default=False, description="自动重载")
    app_log_level: str = Field(default="INFO", description="日志级别")
    app_timezone: str = Field(default="UTC", description="时区")
    app_locale: str = Field(default="en_US", description="语言环境")
    
    # =============================================================================
    # 安全配置
    # =============================================================================
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32), description="应用密钥")
    jwt_secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32), description="JWT密钥")
    jwt_algorithm: str = Field(default="HS256", description="JWT算法")
    jwt_access_token_expire_minutes: int = Field(default=30, description="访问令牌过期时间（分钟）")
    jwt_refresh_token_expire_days: int = Field(default=7, description="刷新令牌过期时间（天）")
    api_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32), description="API密钥")
    encryption_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32), description="加密密钥")
    session_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32), description="会话密钥")
    csrf_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32), description="CSRF密钥")
    password_reset_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32), description="密码重置密钥")
    email_verification_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32), description="邮箱验证密钥")
    totp_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32), description="TOTP密钥")
    
    # =============================================================================
    # 数据库配置
    # =============================================================================
    database_url: PostgresDsn = Field(default="postgresql://postgres:postgres@localhost:5432/enterprise_db", description="数据库连接URL")
    database_host: str = Field(default="localhost", description="数据库主机")
    database_port: int = Field(default=5432, description="数据库端口")
    database_name: str = Field(default="enterprise_db", description="数据库名称")
    database_user: str = Field(default="postgres", description="数据库用户")
    database_password: str = Field(default="postgres", description="数据库密码")
    database_pool_size: int = Field(default=10, description="连接池大小")
    database_max_overflow: int = Field(default=20, description="连接池最大溢出")
    database_pool_timeout: int = Field(default=30, description="连接池超时")
    database_pool_recycle: int = Field(default=3600, description="连接池回收时间")
    database_echo: bool = Field(default=False, description="SQL回显")
    
    # 只读数据库（可选）
    read_database_url: Optional[PostgresDsn] = Field(default=None, description="只读数据库连接URL")
    
    # 测试数据库
    test_database_url: PostgresDsn = Field(default="postgresql://test:test@localhost:5434/test_db", description="测试数据库连接URL")
    
    # =============================================================================
    # Redis配置
    # =============================================================================
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0", description="Redis连接URL")
    redis_host: str = Field(default="localhost", description="Redis主机")
    redis_port: int = Field(default=6379, description="Redis端口")
    redis_db: int = Field(default=0, description="Redis数据库")
    redis_password: Optional[str] = Field(default=None, description="Redis密码")
    redis_max_connections: int = Field(default=10, description="Redis最大连接数")
    redis_socket_timeout: int = Field(default=5, description="Redis套接字超时")
    redis_socket_connect_timeout: int = Field(default=5, description="Redis连接超时")
    redis_health_check_interval: int = Field(default=30, description="Redis健康检查间隔")
    
    # Redis Sentinel配置（高可用）
    redis_sentinel_hosts: Optional[str] = Field(default=None, description="Redis Sentinel主机列表")
    redis_sentinel_service: str = Field(default="mymaster", description="Redis Sentinel服务名")
    redis_sentinel_password: Optional[str] = Field(default=None, description="Redis Sentinel密码")
    
    # =============================================================================
    # 消息队列配置
    # =============================================================================
    rabbitmq_url: str = Field(default="amqp://guest:guest@localhost:5672/", description="RabbitMQ连接URL")
    rabbitmq_host: str = Field(default="localhost", description="RabbitMQ主机")
    rabbitmq_port: int = Field(default=5672, description="RabbitMQ端口")
    rabbitmq_user: str = Field(default="guest", description="RabbitMQ用户")
    rabbitmq_password: str = Field(default="guest", description="RabbitMQ密码")
    rabbitmq_vhost: str = Field(default="/", description="RabbitMQ虚拟主机")
    rabbitmq_management_url: str = Field(default="http://localhost:15672", description="RabbitMQ管理界面URL")
    rabbitmq_erlang_cookie: str = Field(default="your-erlang-cookie", description="RabbitMQ Erlang Cookie")
    
    # Celery配置
    celery_broker_url: str = Field(default="redis://localhost:6379/1", description="Celery代理URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/2", description="Celery结果后端")
    celery_task_serializer: str = Field(default="json", description="Celery任务序列化器")
    celery_result_serializer: str = Field(default="json", description="Celery结果序列化器")
    celery_accept_content: List[str] = Field(default=["json"], description="Celery接受内容类型")
    celery_timezone: str = Field(default="UTC", description="Celery时区")
    celery_enable_utc: bool = Field(default=True, description="Celery启用UTC")
    
    # =============================================================================
    # 邮件配置
    # =============================================================================
    smtp_host: str = Field(default="smtp.gmail.com", description="SMTP主机")
    smtp_port: int = Field(default=587, description="SMTP端口")
    smtp_user: str = Field(default="your-email@gmail.com", description="SMTP用户")
    smtp_password: str = Field(default="your-app-password", description="SMTP密码")
    smtp_tls: bool = Field(default=True, description="SMTP TLS")
    smtp_ssl: bool = Field(default=False, description="SMTP SSL")
    smtp_from_email: str = Field(default="noreply@yourcompany.com", description="发件人邮箱")
    smtp_from_name: str = Field(default="Your Company", description="发件人名称")
    
    # =============================================================================
    # 文件存储配置
    # =============================================================================
    upload_dir: str = Field(default="./uploads", description="上传目录")
    static_dir: str = Field(default="./static", description="静态文件目录")
    max_upload_size: int = Field(default=10485760, description="最大上传大小（字节）")
    allowed_extensions: List[str] = Field(
        default=["jpg", "jpeg", "png", "gif", "pdf", "doc", "docx", "xls", "xlsx", "txt"],
        description="允许的文件扩展名"
    )
    
    # AWS S3配置
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS访问密钥ID")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS秘密访问密钥")
    aws_region: str = Field(default="us-east-1", description="AWS区域")
    aws_s3_bucket: Optional[str] = Field(default=None, description="AWS S3存储桶")
    aws_s3_endpoint_url: Optional[str] = Field(default=None, description="AWS S3端点URL")
    
    # MinIO配置
    minio_endpoint: Optional[str] = Field(default=None, description="MinIO端点")
    minio_access_key: Optional[str] = Field(default=None, description="MinIO访问密钥")
    minio_secret_key: Optional[str] = Field(default=None, description="MinIO秘密密钥")
    minio_bucket: Optional[str] = Field(default=None, description="MinIO存储桶")
    minio_secure: bool = Field(default=False, description="MinIO安全连接")
    
    # =============================================================================
    # 缓存配置
    # =============================================================================
    cache_type: str = Field(default="redis", description="缓存类型")
    cache_default_timeout: int = Field(default=300, description="缓存默认超时")
    cache_key_prefix: str = Field(default="enterprise:", description="缓存键前缀")
    cache_version: str = Field(default="1", description="缓存版本")
    
    # =============================================================================
    # 搜索引擎配置
    # =============================================================================
    elasticsearch_url: str = Field(default="http://localhost:9200", description="Elasticsearch URL")
    elasticsearch_index: str = Field(default="enterprise", description="Elasticsearch索引")
    elasticsearch_username: Optional[str] = Field(default=None, description="Elasticsearch用户名")
    elasticsearch_password: Optional[str] = Field(default=None, description="Elasticsearch密码")
    elasticsearch_timeout: int = Field(default=30, description="Elasticsearch超时")
    elasticsearch_max_retries: int = Field(default=3, description="Elasticsearch最大重试次数")
    
    # =============================================================================
    # 监控和可观测性配置
    # =============================================================================
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN")
    sentry_environment: str = Field(default="development", description="Sentry环境")
    sentry_traces_sample_rate: float = Field(default=0.1, description="Sentry追踪采样率")
    
    jaeger_agent_host: str = Field(default="localhost", description="Jaeger代理主机")
    jaeger_agent_port: int = Field(default=6831, description="Jaeger代理端口")
    jaeger_collector_endpoint: str = Field(default="http://localhost:14268/api/traces", description="Jaeger收集器端点")
    jaeger_service_name: str = Field(default="enterprise-app", description="Jaeger服务名")
    jaeger_sampler_type: str = Field(default="const", description="Jaeger采样器类型")
    jaeger_sampler_param: float = Field(default=1.0, description="Jaeger采样器参数")
    
    prometheus_pushgateway_url: str = Field(default="http://localhost:9091", description="Prometheus推送网关URL")
    prometheus_job_name: str = Field(default="enterprise-app", description="Prometheus作业名")
    prometheus_instance: str = Field(default="localhost:8000", description="Prometheus实例")
    
    datadog_api_key: Optional[str] = Field(default=None, description="DataDog API密钥")
    datadog_app_key: Optional[str] = Field(default=None, description="DataDog应用密钥")
    datadog_site: str = Field(default="datadoghq.com", description="DataDog站点")
    
    new_relic_license_key: Optional[str] = Field(default=None, description="New Relic许可证密钥")
    new_relic_app_name: str = Field(default="Enterprise Application", description="New Relic应用名")
    
    # =============================================================================
    # CORS配置
    # =============================================================================
    cors_origins: List[str] = Field(default=["http://localhost:3000", "http://localhost:8080"], description="CORS允许的源")
    cors_allow_credentials: bool = Field(default=True, description="CORS允许凭据")
    cors_allow_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"], description="CORS允许的方法")
    cors_allow_headers: List[str] = Field(default=["*"], description="CORS允许的头")
    
    # =============================================================================
    # 安全配置
    # =============================================================================
    security_headers_enabled: bool = Field(default=True, description="启用安全头")
    hsts_max_age: int = Field(default=31536000, description="HSTS最大年龄")
    csp_policy: str = Field(default="default-src 'self'", description="CSP策略")
    
    # =============================================================================
    # 限流配置
    # =============================================================================
    rate_limit_enabled: bool = Field(default=True, description="启用限流")
    rate_limit_per_minute: int = Field(default=60, description="每分钟限流")
    rate_limit_per_hour: int = Field(default=1000, description="每小时限流")
    rate_limit_per_day: int = Field(default=10000, description="每天限流")
    rate_limit_storage: str = Field(default="redis", description="限流存储")
    
    # =============================================================================
    # 功能开关
    # =============================================================================
    feature_registration_enabled: bool = Field(default=True, description="启用注册功能")
    feature_email_verification_required: bool = Field(default=True, description="需要邮箱验证")
    feature_two_factor_auth_enabled: bool = Field(default=False, description="启用双因子认证")
    feature_social_login_enabled: bool = Field(default=True, description="启用社交登录")
    feature_ldap_auth_enabled: bool = Field(default=False, description="启用LDAP认证")
    feature_saml_auth_enabled: bool = Field(default=False, description="启用SAML认证")
    feature_api_documentation_enabled: bool = Field(default=True, description="启用API文档")
    feature_metrics_enabled: bool = Field(default=True, description="启用指标")
    feature_tracing_enabled: bool = Field(default=True, description="启用追踪")
    feature_caching_enabled: bool = Field(default=True, description="启用缓存")
    feature_rate_limiting_enabled: bool = Field(default=True, description="启用限流")
    feature_file_upload_enabled: bool = Field(default=True, description="启用文件上传")
    feature_search_enabled: bool = Field(default=True, description="启用搜索")
    feature_notifications_enabled: bool = Field(default=True, description="启用通知")
    feature_payments_enabled: bool = Field(default=False, description="启用支付")
    feature_analytics_enabled: bool = Field(default=True, description="启用分析")
    feature_a_b_testing_enabled: bool = Field(default=False, description="启用A/B测试")
    feature_maintenance_mode: bool = Field(default=False, description="维护模式")
    
    # =============================================================================
    # 性能配置
    # =============================================================================
    pagination_default_size: int = Field(default=20, description="分页默认大小")
    pagination_max_size: int = Field(default=100, description="分页最大大小")
    query_timeout: int = Field(default=30, description="查询超时")
    request_timeout: int = Field(default=60, description="请求超时")
    upload_timeout: int = Field(default=300, description="上传超时")
    cache_timeout: int = Field(default=3600, description="缓存超时")
    session_timeout: int = Field(default=1800, description="会话超时")
    
    # =============================================================================
    # 国际化配置
    # =============================================================================
    default_language: str = Field(default="en", description="默认语言")
    supported_languages: List[str] = Field(default=["en", "zh", "es", "fr", "de", "ja"], description="支持的语言")
    timezone_detection_enabled: bool = Field(default=True, description="启用时区检测")
    localization_enabled: bool = Field(default=True, description="启用本地化")
    
    # =============================================================================
    # 测试配置
    # =============================================================================
    testing: bool = Field(default=False, description="测试模式")
    test_redis_url: str = Field(default="redis://localhost:6379/15", description="测试Redis URL")
    test_email_backend: str = Field(default="console", description="测试邮件后端")
    test_celery_always_eager: bool = Field(default=True, description="测试Celery立即执行")
    
    # =============================================================================
    # 验证器
    # =============================================================================
    
    @validator("app_environment")
    def validate_environment(cls, v: str) -> str:
        """验证环境配置"""
        allowed_environments = ["development", "testing", "staging", "production"]
        if v not in allowed_environments:
            raise ValueError(f"Environment must be one of: {allowed_environments}")
        return v
    
    @validator("app_log_level")
    def validate_log_level(cls, v: str) -> str:
        """验证日志级别"""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """解析CORS源"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("cors_allow_methods", pre=True)
    def parse_cors_methods(cls, v: Union[str, List[str]]) -> List[str]:
        """解析CORS方法"""
        if isinstance(v, str):
            return [method.strip() for method in v.split(",")]
        return v
    
    @validator("cors_allow_headers", pre=True)
    def parse_cors_headers(cls, v: Union[str, List[str]]) -> List[str]:
        """解析CORS头"""
        if isinstance(v, str):
            return [header.strip() for header in v.split(",")]
        return v
    
    @validator("allowed_extensions", pre=True)
    def parse_allowed_extensions(cls, v: Union[str, List[str]]) -> List[str]:
        """解析允许的文件扩展名"""
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",")]
        return [ext.lower() for ext in v]
    
    @validator("supported_languages", pre=True)
    def parse_supported_languages(cls, v: Union[str, List[str]]) -> List[str]:
        """解析支持的语言"""
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",")]
        return v
    
    @root_validator
    def validate_settings(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """根级验证器"""
        # 在生产环境中确保安全配置
        if values.get("app_environment") == "production":
            if values.get("app_debug"):
                raise ValueError("Debug mode must be disabled in production")
            if values.get("secret_key") == "your-super-secret-key-change-this-in-production":
                raise ValueError("Secret key must be changed in production")
        
        # 确保上传目录存在
        upload_dir = Path(values.get("upload_dir", "./uploads"))
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # 确保静态文件目录存在
        static_dir = Path(values.get("static_dir", "./static"))
        static_dir.mkdir(parents=True, exist_ok=True)
        
        return values
    
    class Config:
        """Pydantic配置"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True
        
        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            """自定义配置源优先级"""
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )


@lru_cache()
def get_settings() -> Settings:
    """
    获取应用配置实例（带缓存）
    
    Returns:
        Settings: 配置实例
    """
    return Settings()


def get_database_url(settings: Optional[Settings] = None) -> str:
    """
    获取数据库连接URL
    
    Args:
        settings: 配置实例
    
    Returns:
        str: 数据库连接URL
    """
    if settings is None:
        settings = get_settings()
    
    if settings.testing:
        return str(settings.test_database_url)
    
    return str(settings.database_url)


def get_redis_url(settings: Optional[Settings] = None) -> str:
    """
    获取Redis连接URL
    
    Args:
        settings: 配置实例
    
    Returns:
        str: Redis连接URL
    """
    if settings is None:
        settings = get_settings()
    
    if settings.testing:
        return settings.test_redis_url
    
    return str(settings.redis_url)


def is_development() -> bool:
    """
    检查是否为开发环境
    
    Returns:
        bool: 是否为开发环境
    """
    return get_settings().app_environment == "development"


def is_production() -> bool:
    """
    检查是否为生产环境
    
    Returns:
        bool: 是否为生产环境
    """
    return get_settings().app_environment == "production"


def is_testing() -> bool:
    """
    检查是否为测试环境
    
    Returns:
        bool: 是否为测试环境
    """
    settings = get_settings()
    return settings.app_environment == "testing" or settings.testing