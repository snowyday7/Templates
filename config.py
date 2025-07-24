#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
企业级应用配置管理

提供统一的配置管理，支持环境变量、配置文件和默认值的层次化配置。
包含所有企业级功能模块的配置选项。
"""

import os
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator
from enum import Enum


class Environment(str, Enum):
    """环境类型"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseConfig(BaseSettings):
    """数据库配置"""
    
    # 主数据库
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/myapp",
        description="主数据库连接URL"
    )
    
    # 读库配置（读写分离）
    read_database_url: Optional[str] = Field(
        default=None,
        description="只读数据库连接URL"
    )
    
    # 连接池配置
    pool_size: int = Field(default=20, description="连接池大小")
    max_overflow: int = Field(default=30, description="连接池最大溢出")
    pool_timeout: int = Field(default=30, description="连接池超时时间（秒）")
    pool_recycle: int = Field(default=3600, description="连接回收时间（秒）")
    
    # 查询配置
    query_timeout: int = Field(default=30, description="查询超时时间（秒）")
    slow_query_threshold: float = Field(default=1.0, description="慢查询阈值（秒）")
    
    class Config:
        env_prefix = "DB_"


class RedisConfig(BaseSettings):
    """Redis配置"""
    
    # 连接配置
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis连接URL"
    )
    
    # 连接池配置
    max_connections: int = Field(default=50, description="最大连接数")
    retry_on_timeout: bool = Field(default=True, description="超时重试")
    health_check_interval: int = Field(default=30, description="健康检查间隔（秒）")
    
    # 缓存配置
    default_ttl: int = Field(default=3600, description="默认TTL（秒）")
    key_prefix: str = Field(default="myapp:", description="键前缀")
    
    class Config:
        env_prefix = "REDIS_"


class SecurityConfig(BaseSettings):
    """安全配置"""
    
    # JWT配置
    jwt_secret_key: str = Field(
        default="your-super-secret-jwt-key-change-in-production",
        description="JWT密钥"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT算法")
    jwt_expiration_hours: int = Field(default=24, description="JWT过期时间（小时）")
    jwt_refresh_expiration_days: int = Field(default=7, description="刷新令牌过期时间（天）")
    
    # API密钥配置
    api_key_secret: str = Field(
        default="your-api-key-secret-change-in-production",
        description="API密钥加密密钥"
    )
    api_key_length: int = Field(default=32, description="API密钥长度")
    api_key_default_expiry_days: int = Field(default=30, description="API密钥默认过期天数")
    max_api_keys_per_user: int = Field(default=10, description="每用户最大API密钥数")
    
    # RBAC配置
    enable_rbac: bool = Field(default=True, description="启用RBAC")
    enable_permission_inheritance: bool = Field(default=True, description="启用权限继承")
    enable_dynamic_permissions: bool = Field(default=True, description="启用动态权限")
    
    # 密码策略
    password_min_length: int = Field(default=8, description="密码最小长度")
    password_require_uppercase: bool = Field(default=True, description="密码需要大写字母")
    password_require_lowercase: bool = Field(default=True, description="密码需要小写字母")
    password_require_numbers: bool = Field(default=True, description="密码需要数字")
    password_require_special: bool = Field(default=True, description="密码需要特殊字符")
    
    # 会话配置
    session_timeout_minutes: int = Field(default=30, description="会话超时时间（分钟）")
    max_login_attempts: int = Field(default=5, description="最大登录尝试次数")
    lockout_duration_minutes: int = Field(default=15, description="锁定持续时间（分钟）")
    
    class Config:
        env_prefix = "SECURITY_"


class ObservabilityConfig(BaseSettings):
    """可观测性配置"""
    
    # 追踪配置
    enable_tracing: bool = Field(default=True, description="启用分布式追踪")
    tracing_service_name: str = Field(default="enterprise-app", description="服务名称")
    tracing_service_version: str = Field(default="1.0.0", description="服务版本")
    tracing_sampling_rate: float = Field(default=1.0, description="追踪采样率")
    
    # Jaeger配置
    jaeger_endpoint: Optional[str] = Field(
        default=None,
        description="Jaeger端点URL"
    )
    jaeger_agent_host: str = Field(default="localhost", description="Jaeger代理主机")
    jaeger_agent_port: int = Field(default=6831, description="Jaeger代理端口")
    
    # 指标配置
    enable_metrics: bool = Field(default=True, description="启用指标收集")
    metrics_namespace: str = Field(default="enterprise_app", description="指标命名空间")
    metrics_port: int = Field(default=9090, description="指标暴露端口")
    
    # Prometheus配置
    prometheus_gateway_url: Optional[str] = Field(
        default=None,
        description="Prometheus推送网关URL"
    )
    
    # 告警配置
    enable_alerting: bool = Field(default=True, description="启用告警")
    alert_evaluation_interval: int = Field(default=60, description="告警评估间隔（秒）")
    alert_notification_timeout: int = Field(default=30, description="告警通知超时（秒）")
    
    # 邮件告警配置
    smtp_host: Optional[str] = Field(default=None, description="SMTP主机")
    smtp_port: int = Field(default=587, description="SMTP端口")
    smtp_username: Optional[str] = Field(default=None, description="SMTP用户名")
    smtp_password: Optional[str] = Field(default=None, description="SMTP密码")
    smtp_use_tls: bool = Field(default=True, description="SMTP使用TLS")
    
    # Slack告警配置
    slack_webhook_url: Optional[str] = Field(default=None, description="Slack Webhook URL")
    slack_channel: str = Field(default="#alerts", description="Slack频道")
    
    class Config:
        env_prefix = "OBSERVABILITY_"


class PerformanceConfig(BaseSettings):
    """性能配置"""
    
    # 连接池配置
    enable_connection_pooling: bool = Field(default=True, description="启用连接池")
    pool_max_connections: int = Field(default=100, description="连接池最大连接数")
    pool_min_connections: int = Field(default=10, description="连接池最小连接数")
    pool_connection_timeout: int = Field(default=30, description="连接超时时间（秒）")
    pool_idle_timeout: int = Field(default=300, description="空闲超时时间（秒）")
    
    # 缓存配置
    enable_caching: bool = Field(default=True, description="启用缓存")
    cache_default_ttl: int = Field(default=3600, description="缓存默认TTL（秒）")
    cache_max_size: int = Field(default=1000, description="缓存最大大小")
    
    # 异步配置
    enable_async_processing: bool = Field(default=True, description="启用异步处理")
    async_worker_count: int = Field(default=4, description="异步工作线程数")
    async_queue_size: int = Field(default=1000, description="异步队列大小")
    
    # 限流配置
    enable_rate_limiting: bool = Field(default=True, description="启用限流")
    rate_limit_requests_per_minute: int = Field(default=100, description="每分钟请求限制")
    rate_limit_burst_size: int = Field(default=20, description="突发请求大小")
    
    class Config:
        env_prefix = "PERFORMANCE_"


class HighAvailabilityConfig(BaseSettings):
    """高可用性配置"""
    
    # 负载均衡配置
    enable_load_balancing: bool = Field(default=True, description="启用负载均衡")
    load_balance_strategy: str = Field(default="round_robin", description="负载均衡策略")
    health_check_enabled: bool = Field(default=True, description="启用健康检查")
    health_check_interval: int = Field(default=30, description="健康检查间隔（秒）")
    health_check_timeout: int = Field(default=5, description="健康检查超时（秒）")
    health_check_retries: int = Field(default=3, description="健康检查重试次数")
    
    # 故障转移配置
    enable_failover: bool = Field(default=True, description="启用故障转移")
    failover_strategy: str = Field(default="priority_based", description="故障转移策略")
    failure_threshold: int = Field(default=3, description="故障阈值")
    recovery_threshold: int = Field(default=2, description="恢复阈值")
    failover_timeout: int = Field(default=30, description="故障转移超时（秒）")
    
    # 熔断器配置
    enable_circuit_breaker: bool = Field(default=True, description="启用熔断器")
    circuit_breaker_failure_threshold: int = Field(default=5, description="熔断器故障阈值")
    circuit_breaker_timeout: int = Field(default=60, description="熔断器超时（秒）")
    circuit_breaker_half_open_max_calls: int = Field(default=3, description="半开状态最大调用数")
    
    class Config:
        env_prefix = "HA_"


class DataGovernanceConfig(BaseSettings):
    """数据治理配置"""
    
    # 数据质量配置
    enable_data_quality: bool = Field(default=True, description="启用数据质量检查")
    data_quality_check_interval: int = Field(default=3600, description="数据质量检查间隔（秒）")
    data_quality_threshold: float = Field(default=0.95, description="数据质量阈值")
    enable_auto_profiling: bool = Field(default=True, description="启用自动数据分析")
    enable_anomaly_detection: bool = Field(default=True, description="启用异常检测")
    
    # 数据血缘配置
    enable_data_lineage: bool = Field(default=True, description="启用数据血缘")
    lineage_tracking_depth: int = Field(default=10, description="血缘追踪深度")
    
    # 数据合规配置
    enable_data_compliance: bool = Field(default=True, description="启用数据合规")
    compliance_rules_file: str = Field(default="compliance_rules.json", description="合规规则文件")
    
    # 数据分类配置
    enable_data_classification: bool = Field(default=True, description="启用数据分类")
    classification_rules_file: str = Field(default="classification_rules.json", description="分类规则文件")
    
    class Config:
        env_prefix = "DATA_GOVERNANCE_"


class EnterpriseIntegrationConfig(BaseSettings):
    """企业集成配置"""
    
    # API网关配置
    enable_api_gateway: bool = Field(default=True, description="启用API网关")
    gateway_host: str = Field(default="0.0.0.0", description="网关主机")
    gateway_port: int = Field(default=8080, description="网关端口")
    gateway_timeout: int = Field(default=30, description="网关超时（秒）")
    
    # 消息队列配置
    enable_message_queue: bool = Field(default=True, description="启用消息队列")
    rabbitmq_url: Optional[str] = Field(
        default="amqp://guest:guest@localhost:5672/",
        description="RabbitMQ连接URL"
    )
    kafka_bootstrap_servers: Optional[str] = Field(
        default="localhost:9092",
        description="Kafka引导服务器"
    )
    
    # LDAP配置
    enable_ldap: bool = Field(default=False, description="启用LDAP集成")
    ldap_server: Optional[str] = Field(default=None, description="LDAP服务器")
    ldap_port: int = Field(default=389, description="LDAP端口")
    ldap_base_dn: Optional[str] = Field(default=None, description="LDAP基础DN")
    ldap_bind_dn: Optional[str] = Field(default=None, description="LDAP绑定DN")
    ldap_bind_password: Optional[str] = Field(default=None, description="LDAP绑定密码")
    
    # SAML配置
    enable_saml: bool = Field(default=False, description="启用SAML集成")
    saml_metadata_url: Optional[str] = Field(default=None, description="SAML元数据URL")
    saml_entity_id: Optional[str] = Field(default=None, description="SAML实体ID")
    
    class Config:
        env_prefix = "ENTERPRISE_"


class ApplicationConfig(BaseSettings):
    """应用程序主配置"""
    
    # 基础配置
    app_name: str = Field(default="Enterprise Application", description="应用名称")
    app_version: str = Field(default="1.0.0", description="应用版本")
    app_description: str = Field(default="企业级Python应用", description="应用描述")
    
    # 环境配置
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="运行环境")
    debug: bool = Field(default=True, description="调试模式")
    
    # 服务器配置
    host: str = Field(default="0.0.0.0", description="服务器主机")
    port: int = Field(default=8000, description="服务器端口")
    workers: int = Field(default=1, description="工作进程数")
    
    # 日志配置
    log_level: LogLevel = Field(default=LogLevel.INFO, description="日志级别")
    log_format: str = Field(default="json", description="日志格式")
    log_file: Optional[str] = Field(default=None, description="日志文件路径")
    
    # CORS配置
    cors_origins: List[str] = Field(default=["*"], description="CORS允许的源")
    cors_methods: List[str] = Field(default=["*"], description="CORS允许的方法")
    cors_headers: List[str] = Field(default=["*"], description="CORS允许的头部")
    
    # 中间件配置
    enable_request_logging: bool = Field(default=True, description="启用请求日志")
    enable_response_compression: bool = Field(default=True, description="启用响应压缩")
    enable_request_id: bool = Field(default=True, description="启用请求ID")
    
    # 文件上传配置
    max_upload_size: int = Field(default=10 * 1024 * 1024, description="最大上传大小（字节）")
    allowed_file_types: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".gif", ".pdf", ".doc", ".docx"],
        description="允许的文件类型"
    )
    
    # 任务队列配置
    enable_task_queue: bool = Field(default=True, description="启用任务队列")
    task_queue_broker: str = Field(default="redis://localhost:6379/1", description="任务队列代理")
    task_queue_backend: str = Field(default="redis://localhost:6379/2", description="任务队列后端")
    
    @validator('environment', pre=True)
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator('log_level', pre=True)
    def validate_log_level(cls, v):
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v
    
    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_testing(self) -> bool:
        return self.environment == Environment.TESTING
    
    class Config:
        env_prefix = "APP_"
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"


class Settings(BaseSettings):
    """全局设置"""
    
    # 各模块配置
    app: ApplicationConfig = ApplicationConfig()
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    security: SecurityConfig = SecurityConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    performance: PerformanceConfig = PerformanceConfig()
    high_availability: HighAvailabilityConfig = HighAvailabilityConfig()
    data_governance: DataGovernanceConfig = DataGovernanceConfig()
    enterprise_integration: EnterpriseIntegrationConfig = EnterpriseIntegrationConfig()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_configuration()
    
    def _validate_configuration(self):
        """验证配置的一致性"""
        # 生产环境安全检查
        if self.app.is_production:
            if self.security.jwt_secret_key == "your-super-secret-jwt-key-change-in-production":
                raise ValueError("生产环境必须更改JWT密钥")
            
            if self.security.api_key_secret == "your-api-key-secret-change-in-production":
                raise ValueError("生产环境必须更改API密钥加密密钥")
            
            if self.app.debug:
                raise ValueError("生产环境不能启用调试模式")
        
        # 数据库配置检查
        if self.database.pool_size <= 0:
            raise ValueError("数据库连接池大小必须大于0")
        
        # Redis配置检查
        if self.redis.max_connections <= 0:
            raise ValueError("Redis最大连接数必须大于0")
        
        # 性能配置检查
        if self.performance.pool_max_connections < self.performance.pool_min_connections:
            raise ValueError("连接池最大连接数不能小于最小连接数")
    
    def get_database_url(self, read_only: bool = False) -> str:
        """获取数据库连接URL"""
        if read_only and self.database.read_database_url:
            return self.database.read_database_url
        return self.database.database_url
    
    def get_redis_url(self) -> str:
        """获取Redis连接URL"""
        return self.redis.redis_url
    
    def is_feature_enabled(self, feature: str) -> bool:
        """检查功能是否启用"""
        feature_map = {
            "tracing": self.observability.enable_tracing,
            "metrics": self.observability.enable_metrics,
            "alerting": self.observability.enable_alerting,
            "caching": self.performance.enable_caching,
            "rate_limiting": self.performance.enable_rate_limiting,
            "load_balancing": self.high_availability.enable_load_balancing,
            "failover": self.high_availability.enable_failover,
            "circuit_breaker": self.high_availability.enable_circuit_breaker,
            "data_quality": self.data_governance.enable_data_quality,
            "api_gateway": self.enterprise_integration.enable_api_gateway,
            "rbac": self.security.enable_rbac,
        }
        return feature_map.get(feature, False)
    
    def export_config(self) -> Dict[str, Any]:
        """导出配置（隐藏敏感信息）"""
        config = self.dict()
        
        # 隐藏敏感信息
        sensitive_fields = [
            "jwt_secret_key", "api_key_secret", "smtp_password",
            "ldap_bind_password", "database_url", "redis_url"
        ]
        
        def hide_sensitive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if any(field in key.lower() for field in ["password", "secret", "key", "token"]):
                        obj[key] = "***HIDDEN***"
                    elif "url" in key.lower() and any(sensitive in str(value) for sensitive in ["://"]):
                        # 隐藏URL中的密码部分
                        if isinstance(value, str) and "://" in value:
                            parts = value.split("://")
                            if len(parts) == 2 and "@" in parts[1]:
                                protocol = parts[0]
                                rest = parts[1]
                                if ":" in rest.split("@")[0]:
                                    user_pass, host_part = rest.split("@", 1)
                                    user = user_pass.split(":")[0]
                                    obj[key] = f"{protocol}://{user}:***@{host_part}"
                    else:
                        hide_sensitive(value, current_path)
            elif isinstance(obj, list):
                for item in obj:
                    hide_sensitive(item, path)
        
        hide_sensitive(config)
        return config
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 全局设置实例
settings = Settings()


def get_settings() -> Settings:
    """获取全局设置实例"""
    return settings


def reload_settings() -> Settings:
    """重新加载设置"""
    global settings
    settings = Settings()
    return settings


if __name__ == "__main__":
    # 配置验证和导出示例
    import json
    
    print("=== 企业级应用配置 ===")
    print(f"应用名称: {settings.app.app_name}")
    print(f"应用版本: {settings.app.app_version}")
    print(f"运行环境: {settings.app.environment}")
    print(f"调试模式: {settings.app.debug}")
    
    print("\n=== 功能启用状态 ===")
    features = [
        "tracing", "metrics", "alerting", "caching", "rate_limiting",
        "load_balancing", "failover", "circuit_breaker", "data_quality",
        "api_gateway", "rbac"
    ]
    
    for feature in features:
        status = "✓" if settings.is_feature_enabled(feature) else "✗"
        print(f"{status} {feature}")
    
    print("\n=== 配置导出（隐藏敏感信息）===")
    config_export = settings.export_config()
    print(json.dumps(config_export, indent=2, ensure_ascii=False))