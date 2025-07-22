"""Python后端开发功能组件模板库

这是一个全面、强大的Python后端开发功能组件模板库，为开发者提供开箱即用的高质量代码模板。

主要功能模块：
- 数据库操作模板 (database)
- API开发模板 (api)
- 认证授权模板 (auth)
- 缓存与消息队列模板 (cache)
- 监控与日志模板 (monitoring)
- 部署与配置模板 (deployment)

使用示例：
    from templates.database import DatabaseManager
from templates.api import FastAPIApp
from templates.auth import JWTManager
from templates.cache import RedisManager
from templates.monitoring import StructuredLogger
from templates.deployment import DockerfileGenerator
"""

from typing import List

__version__ = "1.0.0"
__author__ = "Python Backend Template Library"
__description__ = "全面的Python后端开发功能组件模板库"

# 可用模块信息
AVAILABLE_MODULES = {
    "database": {
        "description": "数据库ORM和连接管理",
        "features": ["SQLAlchemy ORM", "连接池管理", "数据迁移"],
        "dependencies": ["sqlalchemy", "alembic", "psycopg2-binary"],
    },
    "api": {
        "description": "RESTful API开发框架",
        "features": ["FastAPI框架", "自动文档", "中间件支持"],
        "dependencies": ["fastapi", "uvicorn", "python-multipart"],
    },
    "auth": {
        "description": "认证和授权系统",
        "features": ["JWT认证", "密码管理", "权限控制"],
        "dependencies": ["python-jose", "passlib", "bcrypt"],
    },
    "cache": {
        "description": "缓存和异步任务处理",
        "features": ["Redis缓存", "Celery任务", "WebSocket通信"],
        "dependencies": ["redis", "celery", "websockets"],
    },
    "monitoring": {
        "description": "监控、日志和错误追踪",
        "features": ["结构化日志", "性能监控", "错误追踪"],
        "dependencies": ["structlog", "sentry-sdk", "prometheus-client"],
    },
    "deployment": {
        "description": "部署和配置管理",
        "features": ["Docker配置", "Kubernetes部署", "CI/CD流程"],
        "dependencies": ["docker", "kubernetes", "pyyaml"],
    },
}

# 数据库模块
from .database import (
    DatabaseConfig,
    DatabaseManager,
    Base,
    BaseModel,
    ConnectionPoolManager,
    MigrationManager,
)

# API开发模块
from .api import (
    FastAPIApp,
    APISettings,
    create_fastapi_app,
    setup_cors,
    setup_middleware,
)

# 认证授权模块
from .auth import (
    JWTManager,
    AuthService,
    PasswordManager,
    TokenType,
    AuthSettings,
)

# 缓存与消息队列模块
from .cache import (
    RedisManager,
    CacheManager,
    CeleryManager,
    MessageQueue,
    WebSocketManager,
    DistributedLock,
    RateLimiter,
)

# 监控与日志模块
from .monitoring import (
    StructuredLogger,
    ApplicationMonitor,
    HealthCheckManager,
    ErrorTracker,
    setup_logging,
    get_logger,
)

# 部署与配置模块
from .deployment import (
    DockerfileGenerator,
    KubernetesGenerator,
    CICDGenerator,
    ServerGenerator,
    ConfigManager,
    EnvironmentConfig,
    load_config,
)

# 导出所有主要类和函数
__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    "__description__",
    "AVAILABLE_MODULES",
    # 数据库模块
    "DatabaseConfig",
    "DatabaseManager",
    "Base",
    "BaseModel",
    "ConnectionPoolManager",
    "MigrationManager",
    # API开发模块
    "FastAPIApp",
    "APISettings",
    "create_fastapi_app",
    "setup_cors",
    "setup_middleware",
    # 认证授权模块
    "JWTManager",
    "AuthService",
    "PasswordManager",
    "TokenType",
    "AuthSettings",
    # "get_current_user_dependency",  # TODO: 实现用户依赖函数
    # 缓存与消息队列模块
    "RedisManager",
    "CacheManager",
    "CeleryManager",
    "MessageQueue",
    "WebSocketManager",
    "DistributedLock",
    "RateLimiter",
    # 监控与日志模块
    "StructuredLogger",
    "ApplicationMonitor",
    "HealthCheckManager",
    "ErrorTracker",
    "setup_logging",
    "get_logger",
    # 部署与配置模块
    "DockerfileGenerator",
    "KubernetesGenerator",
    "CICDGenerator",
    "ServerGenerator",
    "ConfigManager",
    "EnvironmentConfig",
    "load_config",
]


# 便捷函数
def get_template_info():
    """获取模板库信息"""
    return {
        "name": "Python Backend Template Library",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "modules": {
            "database": "数据库操作模板",
            "api": "API开发模板",
            "auth": "认证授权模板",
            "cache": "缓存与消息队列模板",
            "monitoring": "监控与日志模板",
            "deployment": "部署与配置模板",
        },
    }


def create_project_template(
    project_name: str, modules: List[str] = None, output_dir: str = "."
) -> None:
    """创建项目模板

    Args:
        project_name: 项目名称
        modules: 需要的模块列表，默认包含所有模块
        output_dir: 输出目录
    """
    import os
    from pathlib import Path

    if modules is None:
        modules = ["database", "api", "auth", "cache", "monitoring", "deployment"]

    project_dir = Path(output_dir) / project_name
    project_dir.mkdir(exist_ok=True)

    # 创建项目结构
    (project_dir / "src").mkdir(exist_ok=True)
    (project_dir / "tests").mkdir(exist_ok=True)
    (project_dir / "docs").mkdir(exist_ok=True)
    (project_dir / "config").mkdir(exist_ok=True)

    # 创建主要文件
    main_py = project_dir / "main.py"
    with open(main_py, "w", encoding="utf-8") as f:
        f.write(
            f'"""\n{project_name}\n\n基于Python Backend Template Library构建的项目\n"""\n\n'
        )

        if "api" in modules:
            f.write("from templates.api import create_fastapi_app\n")
    if "database" in modules:
        f.write("from templates.database import DatabaseManager, DatabaseConfig\n")
    if "auth" in modules:
        f.write("from templates.auth import JWTManager\n")
    if "cache" in modules:
        f.write("from templates.cache import RedisManager\n")
    if "monitoring" in modules:
        f.write("from templates.monitoring import setup_logging\n")
    if "deployment" in modules:
        f.write("from templates.deployment import DockerfileGenerator\n")

        f.write("\n\nif __name__ == '__main__':\n")
        f.write("    print(f'Starting {project_name}...')\n")
        f.write("    # 在这里添加你的应用启动代码\n")

    # 创建requirements.txt
    requirements = project_dir / "requirements.txt"
    with open(requirements, "w", encoding="utf-8") as f:
        base_requirements = ["pydantic>=2.0.0", "python-dotenv>=1.0.0", "pyyaml>=6.0"]

        if "api" in modules:
            base_requirements.extend(["fastapi>=0.100.0", "uvicorn[standard]>=0.20.0"])

        if "database" in modules:
            base_requirements.extend(["sqlalchemy>=2.0.0", "alembic>=1.10.0"])

        if "auth" in modules:
            base_requirements.extend(
                ["python-jose[cryptography]>=3.3.0", "passlib[bcrypt]>=1.7.4"]
            )

        if "cache" in modules:
            base_requirements.extend(["redis>=4.5.0", "celery>=5.2.0"])

        if "monitoring" in modules:
            base_requirements.extend(["structlog>=23.0.0", "sentry-sdk>=1.20.0"])

        for req in base_requirements:
            f.write(f"{req}\n")

    # 创建README.md
    readme = project_dir / "README.md"
    with open(readme, "w", encoding="utf-8") as f:
        f.write(f"# {project_name}\n\n")
        f.write("基于Python Backend Template Library构建的项目\n\n")
        f.write("## 安装依赖\n\n")
        f.write("```bash\n")
        f.write("pip install -r requirements.txt\n")
        f.write("```\n\n")
        f.write("## 运行项目\n\n")
        f.write("```bash\n")
        f.write("python main.py\n")
        f.write("```\n\n")
        f.write("## 包含的模块\n\n")
        for module in modules:
            module_descriptions = {
                "database": "数据库操作",
                "api": "API开发",
                "auth": "认证授权",
                "cache": "缓存与消息队列",
                "monitoring": "监控与日志",
                "deployment": "部署与配置",
            }
            f.write(f"- {module}: {module_descriptions.get(module, module)}\n")

    print(f"项目模板 '{project_name}' 已创建在 {project_dir}")


# 模块快速导入
class QuickImport:
    """快速导入类，提供便捷的模块访问"""

    @property
    def database(self):
        """数据库模块"""
        from . import database

        return database

    @property
    def api(self):
        """API模块"""
        from . import api

        return api

    @property
    def auth(self):
        """认证模块"""
        from . import auth

        return auth

    @property
    def cache(self):
        """缓存模块"""
        from . import cache

        return cache

    @property
    def monitoring(self):
        """监控模块"""
        from . import monitoring

        return monitoring

    @property
    def deployment(self):
        """部署模块"""
        from . import deployment

        return deployment


# 创建快速导入实例
quick = QuickImport()


# 使用示例
if __name__ == "__main__":
    # 打印模板库信息
    info = get_template_info()
    print(f"模板库: {info['name']} v{info['version']}")
    print(f"描述: {info['description']}")
    print("\n可用模块:")
    for module, desc in info["modules"].items():
        print(f"  - {module}: {desc}")

    # 创建示例项目
    # create_project_template("my_backend_project", ["api", "database", "auth"])