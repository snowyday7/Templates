"""部署与配置模块

提供完整的部署和配置功能，包括：
- Docker容器化
- Kubernetes部署
- 环境配置管理
- CI/CD配置
- 服务器配置
- 监控和日志配置
"""

# Docker相关
try:
    from .docker_template import (
        DockerConfig,
        DockerfileGenerator,
        DockerComposeGenerator,
        ContainerManager,
        ImageBuilder,
        generate_dockerfile,
        generate_docker_compose,
        build_image,
        run_container,
        stop_container,
    )
except ImportError:
    # 如果模块不存在，定义空的占位符
    DockerConfig = None
    DockerfileGenerator = None
    DockerComposeGenerator = None
    ContainerManager = None
    ImageBuilder = None
    generate_dockerfile = None
    generate_docker_compose = None
    build_image = None
    run_container = None
    stop_container = None

# Kubernetes相关
try:
    from .kubernetes_template import (
        DeploymentConfig,
        ServiceConfig,
        IngressConfig,
        HPAConfig,
        KubernetesGenerator,
        generate_deployment,
        generate_service,
        generate_ingress,
        generate_hpa,
        generate_complete_app,
    )
except ImportError:
    DeploymentConfig = None
    ServiceConfig = None
    IngressConfig = None
    HPAConfig = None
    KubernetesGenerator = None
    generate_deployment = None
    generate_service = None
    generate_ingress = None
    generate_hpa = None
    generate_complete_app = None

# 环境配置相关
from .config_template import (
    Environment,
    ConfigManager,
    EnvironmentConfig,
    DatabaseConfig,
    RedisConfig,
    LoggingConfig,
    SecurityConfig,
    load_config,
    get_config,
    set_config,
    validate_config,
)

# CI/CD相关
try:
    from .cicd_template import (
        CICDGenerator,
        PipelineConfig,
        generate_github_actions,
        generate_gitlab_ci,
        generate_build_pipeline,
    )
except ImportError:
    CICDGenerator = None
    PipelineConfig = None
    generate_github_actions = None
    generate_gitlab_ci = None
    generate_build_pipeline = None

# 服务器配置相关
try:
    from .server_template import (
        ServerConfig,
        ServerGenerator,
        generate_nginx_config,
        generate_apache_config,
        create_reverse_proxy_config,
    )
except ImportError:
    ServerConfig = None
    ServerGenerator = None
    generate_nginx_config = None
    generate_apache_config = None
    create_reverse_proxy_config = None

__all__ = [
    # Docker
    "DockerConfig",
    "DockerfileGenerator",
    "DockerComposeGenerator",
    "ContainerManager",
    "ImageBuilder",
    "generate_dockerfile",
    "generate_docker_compose",
    "build_image",
    "run_container",
    "stop_container",
    # Kubernetes
    "DeploymentConfig",
    "ServiceConfig",
    "IngressConfig",
    "HPAConfig",
    "KubernetesGenerator",
    "generate_deployment",
    "generate_service",
    "generate_ingress",
    "generate_hpa",
    "generate_complete_app",
    # 配置管理
    "Environment",
    "ConfigManager",
    "EnvironmentConfig",
    "DatabaseConfig",
    "RedisConfig",
    "LoggingConfig",
    "SecurityConfig",
    "load_config",
    "get_config",
    "set_config",
    "validate_config",
    # CI/CD
    "CICDGenerator",
    "PipelineConfig",
    "generate_github_actions",
    "generate_gitlab_ci",
    "generate_build_pipeline",
    # 服务器配置
    "ServerConfig",
    "ServerGenerator",
    "generate_nginx_config",
    "generate_apache_config",
    "create_reverse_proxy_config",
]
