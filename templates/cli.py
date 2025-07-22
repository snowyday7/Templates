#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python后端开发功能组件模板库 - 命令行工具

提供命令行接口来创建项目、初始化模板和管理配置。
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

import click
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax

from templates import (
    __version__,
    get_template_info,
    create_project_template,
    AVAILABLE_MODULES,
)

# 初始化Rich控制台
console = Console()
app = typer.Typer(
    name="python-templates",
    help="Python后端开发功能组件模板库命令行工具",
    add_completion=False,
)


def print_banner():
    """打印欢迎横幅。"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                Python Backend Templates                       ║
    ║           全面、强大的Python后端开发功能组件模板库                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


def print_version():
    """打印版本信息。"""
    console.print(f"版本: {__version__}", style="bold green")


def validate_project_name(name: str) -> bool:
    """验证项目名称。

    Args:
        name: 项目名称

    Returns:
        bool: 是否有效
    """
    if not name:
        return False
    if not name.replace("_", "").replace("-", "").isalnum():
        return False
    if name.startswith(("-", "_")) or name.endswith(("-", "_")):
        return False
    return True


def get_available_modules() -> List[str]:
    """获取可用模块列表。

    Returns:
        List[str]: 可用模块列表
    """
    return list(AVAILABLE_MODULES.keys())


def display_modules_table():
    """显示可用模块表格。"""
    table = Table(title="可用模块")
    table.add_column("模块名称", style="cyan", no_wrap=True)
    table.add_column("描述", style="magenta")
    table.add_column("主要功能", style="green")

    for module_name, module_info in AVAILABLE_MODULES.items():
        table.add_row(
            module_name,
            module_info.get("description", ""),
            ", ".join(module_info.get("features", [])),
        )

    console.print(table)


def create_project_structure(project_path: Path, modules: List[str]):
    """创建项目结构。

    Args:
        project_path: 项目路径
        modules: 选择的模块列表
    """
    # 创建基础目录结构
    directories = [
        "app",
        "app/api",
        "app/core",
        "app/models",
        "app/services",
        "app/utils",
        "tests",
        "docs",
        "scripts",
        "config",
        "logs",
        "data",
    ]

    for directory in directories:
        (project_path / directory).mkdir(parents=True, exist_ok=True)
        # 创建__init__.py文件
        if directory.startswith("app"):
            (project_path / directory / "__init__.py").touch()


def generate_main_file(project_path: Path, project_name: str, modules: List[str]):
    """生成主应用文件。

    Args:
        project_path: 项目路径
        project_name: 项目名称
        modules: 选择的模块列表
    """
    imports = []
    app_setup = []

    # 根据选择的模块生成相应的导入和设置代码
    if "api" in modules:
        imports.append("from templates.api import FastAPIApp, create_fastapi_app")
        app_setup.append("    # 创建FastAPI应用")
        app_setup.append("    app = create_fastapi_app()")

    if "database" in modules:
        imports.append("from templates.database import DatabaseManager, DatabaseConfig")
        app_setup.append("    # 设置数据库配置（需要时再初始化管理器）")
        app_setup.append("    # db_config = DatabaseConfig()")
        app_setup.append("    # db_manager = DatabaseManager(db_config)")
        app_setup.append("    # 使用时: with db_manager.get_session() as session: ...")

    if "auth" in modules:
        imports.append("from templates.auth import JWTManager, AuthSettings")
        app_setup.append("    # 设置JWT认证")
        app_setup.append("    auth_settings = AuthSettings(SECRET_KEY=SECRET_KEY)")
        app_setup.append("    jwt_manager = JWTManager(auth_settings)")

    if "cache" in modules:
        imports.append("from templates.cache import RedisManager")
        app_setup.append("    # 设置Redis缓存")
        app_setup.append(
            "    redis_manager = RedisManager(host=REDIS_HOST, port=REDIS_PORT)"
        )

    if "monitoring" in modules:
        imports.append("from templates.monitoring import setup_logging")
        app_setup.append("    # 设置日志")
        app_setup.append("    logger = setup_logging()")

    main_content = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
{project_name} - 基于Python后端模板库构建的项目
"""

import os
from pathlib import Path

{chr(10).join(imports)}

# 配置
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")


def create_app():
    """创建应用实例。
    
    Returns:
        应用实例
    """
{chr(10).join(app_setup)}
    
    return app


if __name__ == "__main__":
    import uvicorn
    
    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
'''

    with open(project_path / "main.py", "w", encoding="utf-8") as f:
        f.write(main_content)


def generate_requirements_file(project_path: Path, modules: List[str]):
    """生成requirements.txt文件。

    Args:
        project_path: 项目路径
        modules: 选择的模块列表
    """
    base_requirements = [
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
    ]

    module_requirements = {
        "api": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "python-multipart>=0.0.6",
        ],
        "database": ["sqlalchemy>=2.0.0", "alembic>=1.12.0", "psycopg2-binary>=2.9.0"],
        "auth": ["python-jose[cryptography]>=3.3.0", "passlib[bcrypt]>=1.7.4"],
        "cache": ["redis>=5.0.0", "celery>=5.3.0", "aioredis>=2.0.0"],
        "monitoring": [
            "structlog>=23.2.0",
            "sentry-sdk[fastapi]>=1.38.0",
            "prometheus-client>=0.19.0",
        ],
        "deployment": ["docker>=6.1.0", "pyyaml>=6.0.1"],
    }

    requirements = base_requirements.copy()
    for module in modules:
        if module in module_requirements:
            requirements.extend(module_requirements[module])

    # 去重并排序
    requirements = sorted(list(set(requirements)))

    with open(project_path / "requirements.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(requirements))


def generate_readme_file(project_path: Path, project_name: str, modules: List[str]):
    """生成README.md文件。

    Args:
        project_path: 项目路径
        project_name: 项目名称
        modules: 选择的模块列表
    """
    readme_content = f"""# {project_name}

基于Python后端开发功能组件模板库构建的项目。

## 功能模块

本项目使用了以下模块：

{chr(10).join([f"- **{module}**: {AVAILABLE_MODULES.get(module, {}).get('description', '')}" for module in modules])}

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 环境配置

复制环境变量模板文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置相应的环境变量。

### 运行应用

```bash
python main.py
```

应用将在 http://localhost:8000 启动。

### API文档

启动应用后，访问以下地址查看API文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 项目结构

```
{project_name}/
├── app/                    # 应用代码
│   ├── api/               # API路由
│   ├── core/              # 核心配置
│   ├── models/            # 数据模型
│   ├── services/          # 业务逻辑
│   └── utils/             # 工具函数
├── tests/                 # 测试代码
├── docs/                  # 文档
├── scripts/               # 脚本文件
├── config/                # 配置文件
├── logs/                  # 日志文件
├── data/                  # 数据文件
├── main.py                # 应用入口
├── requirements.txt       # 依赖列表
└── README.md             # 项目说明
```

## 开发指南

### 添加新的API端点

在 `app/api/` 目录下创建新的路由文件，然后在主应用中注册。

### 数据库迁移

```bash
# 创建迁移文件
alembic revision --autogenerate -m "描述"

# 执行迁移
alembic upgrade head
```

### 运行测试

```bash
pytest tests/
```

## 部署

### Docker部署

```bash
# 构建镜像
docker build -t {project_name.lower()} .

# 运行容器
docker run -p 8000:8000 {project_name.lower()}
```

### 生产环境

建议使用以下配置进行生产部署：

- 使用Gunicorn或uWSGI作为WSGI服务器
- 配置Nginx作为反向代理
- 使用PostgreSQL或MySQL作为数据库
- 配置Redis用于缓存和会话存储
- 设置监控和日志收集

## 许可证

MIT License
"""

    with open(project_path / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)


def generate_env_example(project_path: Path, modules: List[str]):
    """生成.env.example文件。

    Args:
        project_path: 项目路径
        modules: 选择的模块列表
    """
    env_vars = [
        "# 应用配置",
        "APP_NAME=MyApp",
        "APP_VERSION=1.0.0",
        "DEBUG=False",
        "SECRET_KEY=your-secret-key-change-in-production",
        "",
    ]

    if "database" in modules:
        env_vars.extend(
            [
                "# 数据库配置",
                "DATABASE_URL=postgresql://user:password@localhost:5432/dbname",
                "# DATABASE_URL=mysql://user:password@localhost:3306/dbname",
                "# DATABASE_URL=sqlite:///./app.db",
                "",
            ]
        )

    if "cache" in modules:
        env_vars.extend(
            [
                "# Redis配置",
                "REDIS_HOST=localhost",
                "REDIS_PORT=6379",
                "REDIS_PASSWORD=",
                "REDIS_DB=0",
                "",
            ]
        )

    if "monitoring" in modules:
        env_vars.extend(["# 监控配置", "SENTRY_DSN=", "LOG_LEVEL=INFO", ""])

    with open(project_path / ".env.example", "w", encoding="utf-8") as f:
        f.write("\n".join(env_vars))


@app.command()
def version():
    """显示版本信息。"""
    print_banner()
    print_version()

    # 显示模板信息
    info = get_template_info()
    console.print("\n模板库信息:", style="bold yellow")
    for key, value in info.items():
        console.print(f"  {key}: {value}")


@app.command()
def list_modules():
    """列出所有可用模块。"""
    console.print("\n📦 可用模块列表", style="bold blue")
    display_modules_table()


@app.command()
def create(
    project_name: str = typer.Argument(..., help="项目名称"),
    modules: Optional[List[str]] = typer.Option(
        None, "--module", "-m", help="要包含的模块（可多次指定）"
    ),
    output_dir: str = typer.Option(".", "--output", "-o", help="输出目录"),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", "-i/-ni", help="是否使用交互模式"
    ),
):
    """创建新项目。"""
    print_banner()

    # 验证项目名称
    if not validate_project_name(project_name):
        console.print(
            "❌ 项目名称无效！请使用字母、数字、下划线和连字符。", style="bold red"
        )
        raise typer.Exit(1)

    # 检查输出目录
    output_path = Path(output_dir).resolve()
    project_path = output_path / project_name

    if project_path.exists():
        if not Confirm.ask(f"目录 {project_path} 已存在，是否覆盖？"):
            console.print("操作已取消。", style="yellow")
            raise typer.Exit(0)
        shutil.rmtree(project_path)

    # 选择模块
    if interactive and not modules:
        console.print("\n📦 选择要包含的模块:", style="bold blue")
        display_modules_table()

        available_modules = get_available_modules()
        selected_modules = []

        for module in available_modules:
            if Confirm.ask(f"包含 {module} 模块？"):
                selected_modules.append(module)

        modules = selected_modules

    if not modules:
        modules = ["api", "database"]  # 默认模块

    # 验证模块
    available_modules = get_available_modules()
    invalid_modules = [m for m in modules if m not in available_modules]
    if invalid_modules:
        console.print(f"❌ 无效的模块: {', '.join(invalid_modules)}", style="bold red")
        console.print(f"可用模块: {', '.join(available_modules)}", style="yellow")
        raise typer.Exit(1)

    # 创建项目
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("创建项目中...", total=None)

        try:
            # 创建项目目录
            project_path.mkdir(parents=True, exist_ok=True)

            # 创建项目结构
            progress.update(task, description="创建项目结构...")
            create_project_structure(project_path, modules)

            # 生成文件
            progress.update(task, description="生成主应用文件...")
            generate_main_file(project_path, project_name, modules)

            progress.update(task, description="生成依赖文件...")
            generate_requirements_file(project_path, modules)

            progress.update(task, description="生成README文件...")
            generate_readme_file(project_path, project_name, modules)

            progress.update(task, description="生成环境配置文件...")
            generate_env_example(project_path, modules)

            progress.update(task, description="完成！")

        except Exception as e:
            console.print(f"❌ 创建项目失败: {e}", style="bold red")
            raise typer.Exit(1)

    # 显示成功信息
    console.print("\n✅ 项目创建成功！", style="bold green")
    console.print(f"📁 项目路径: {project_path}", style="blue")
    console.print(f"📦 包含模块: {', '.join(modules)}", style="blue")

    # 显示下一步操作
    panel = Panel(
        f"""1. cd {project_name}
2. pip install -r requirements.txt
3. cp .env.example .env
4. 编辑 .env 文件配置环境变量
5. python main.py""",
        title="下一步操作",
        border_style="green",
    )
    console.print(panel)


@app.command()
def init(
    module: str = typer.Argument(..., help="要初始化的模块名称"),
    output_dir: str = typer.Option(".", "--output", "-o", help="输出目录"),
):
    """初始化单个模块模板。"""
    available_modules = get_available_modules()

    if module not in available_modules:
        console.print(f"❌ 模块 '{module}' 不存在！", style="bold red")
        console.print(f"可用模块: {', '.join(available_modules)}", style="yellow")
        raise typer.Exit(1)

    output_path = Path(output_dir).resolve()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"初始化 {module} 模块...", total=None)

        try:
            # 这里可以添加具体的模块初始化逻辑
            # 例如复制模板文件、生成配置等

            console.print(f"✅ 模块 '{module}' 初始化成功！", style="bold green")

        except Exception as e:
            console.print(f"❌ 初始化失败: {e}", style="bold red")
            raise typer.Exit(1)


@app.command()
def info(module: Optional[str] = typer.Argument(None, help="模块名称（可选）")):
    """显示模块信息。"""
    if module:
        available_modules = get_available_modules()
        if module not in available_modules:
            console.print(f"❌ 模块 '{module}' 不存在！", style="bold red")
            raise typer.Exit(1)

        module_info = AVAILABLE_MODULES[module]
        console.print(f"\n📦 模块: {module}", style="bold blue")
        console.print(f"描述: {module_info.get('description', '')}", style="green")
        console.print(
            f"功能: {', '.join(module_info.get('features', []))}", style="yellow"
        )

        if "dependencies" in module_info:
            console.print(
                f"依赖: {', '.join(module_info['dependencies'])}", style="cyan"
            )
    else:
        # 显示所有模块信息
        console.print("\n📚 模板库信息", style="bold blue")
        display_modules_table()


def main():
    """主入口函数。"""
    app()


def create_project():
    """创建项目的快捷入口。"""
    # 这个函数可以作为setup.py中的console_scripts入口点
    if len(sys.argv) < 2:
        console.print("请提供项目名称", style="bold red")
        sys.exit(1)

    project_name = sys.argv[1]
    modules = sys.argv[2:] if len(sys.argv) > 2 else ["api", "database"]

    # 调用create命令
    create(project_name, modules, ".", False)


def init_template():
    """初始化模板的快捷入口。"""
    if len(sys.argv) < 2:
        console.print("请提供模块名称", style="bold red")
        sys.exit(1)

    module = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    # 调用init命令
    init(module, output_dir)


if __name__ == "__main__":
    main()
