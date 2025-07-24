#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用启动脚本

提供便捷的应用启动方式，支持不同的运行模式。
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加模板库路径
template_root = project_root.parent.parent
sys.path.insert(0, str(template_root))

import click
import uvicorn

# 导入模板库组件
from src.core.logging import get_logger, setup_logging

# 导入应用组件
from app.core.config import get_settings
from app.core.database import get_database
from app.core.cache import get_cache

# 设置日志
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


def check_dependencies() -> bool:
    """
    检查依赖是否安装
    """
    required_packages = [
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "pydantic",
        "passlib",
        "python-jose",
        "openai"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install them with: pip install -r requirements.txt")
        return False
    
    return True


def check_environment() -> bool:
    """
    检查环境配置
    """
    issues = []
    
    # 检查必需的环境变量
    if not settings.SECRET_KEY or settings.SECRET_KEY == "your-super-secret-key-change-this-in-production":
        issues.append("SECRET_KEY not properly configured")
    
    if not settings.OPENAI_API_KEY or not settings.OPENAI_API_KEY.startswith("sk-"):
        issues.append("OPENAI_API_KEY not properly configured")
    
    # 检查数据库连接
    try:
        database = get_database()
        if not database.health_check():
            issues.append("Database connection failed")
    except Exception as e:
        issues.append(f"Database error: {e}")
    
    # 检查缓存连接（非必需）
    try:
        cache = get_cache()
        cache_info = cache.get_info()
        if not cache_info.get("available"):
            logger.warning("Cache not available, using memory cache")
    except Exception as e:
        logger.warning(f"Cache warning: {e}")
    
    if issues:
        logger.error("Environment check failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    return True


def run_development_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = True,
    log_level: str = "info"
) -> None:
    """
    运行开发服务器
    """
    logger.info("Starting development server...")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level.lower(),
        access_log=settings.LOG_REQUESTS,
        reload_dirs=[str(project_root / "app")],
        reload_excludes=["*.pyc", "__pycache__", "*.log"]
    )


def run_production_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 4,
    log_level: str = "info"
) -> None:
    """
    运行生产服务器
    """
    logger.info("Starting production server...")
    
    # 使用Gunicorn运行
    cmd = [
        "gunicorn",
        "app.main:app",
        "-w", str(workers),
        "-k", "uvicorn.workers.UvicornWorker",
        "--bind", f"{host}:{port}",
        "--log-level", log_level.lower(),
        "--access-logfile", "-" if settings.LOG_REQUESTS else "/dev/null",
        "--error-logfile", "-",
        "--preload",
        "--max-requests", "1000",
        "--max-requests-jitter", "100",
        "--timeout", "30",
        "--keep-alive", "5"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        logger.error("Gunicorn not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gunicorn"], check=True)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start production server: {e}")
        sys.exit(1)


@click.group()
def cli():
    """
    ChatGPT Backend 启动工具
    """
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8000, help="Port to bind")
@click.option("--no-reload", is_flag=True, help="Disable auto-reload")
@click.option("--log-level", default="info", help="Log level")
@click.option("--skip-checks", is_flag=True, help="Skip environment checks")
def dev(host: str, port: int, no_reload: bool, log_level: str, skip_checks: bool):
    """
    启动开发服务器
    """
    if not skip_checks:
        logger.info("Checking dependencies and environment...")
        
        if not check_dependencies():
            sys.exit(1)
        
        if not check_environment():
            sys.exit(1)
        
        logger.info("Environment check passed")
    
    run_development_server(
        host=host,
        port=port,
        reload=not no_reload,
        log_level=log_level
    )


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8000, help="Port to bind")
@click.option("--workers", default=4, help="Number of worker processes")
@click.option("--log-level", default="info", help="Log level")
@click.option("--skip-checks", is_flag=True, help="Skip environment checks")
def prod(host: str, port: int, workers: int, log_level: str, skip_checks: bool):
    """
    启动生产服务器
    """
    if not skip_checks:
        logger.info("Checking dependencies and environment...")
        
        if not check_dependencies():
            sys.exit(1)
        
        if not check_environment():
            sys.exit(1)
        
        logger.info("Environment check passed")
    
    run_production_server(
        host=host,
        port=port,
        workers=workers,
        log_level=log_level
    )


@cli.command()
def check():
    """
    检查环境和依赖
    """
    logger.info("Checking dependencies...")
    deps_ok = check_dependencies()
    
    logger.info("Checking environment...")
    env_ok = check_environment()
    
    if deps_ok and env_ok:
        logger.info("✅ All checks passed!")
        
        # 显示配置信息
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"Database: {settings.DATABASE_URL}")
        logger.info(f"Cache: {get_cache().get_info()['type']}")
        logger.info(f"OpenAI API: {'✅ Configured' if settings.OPENAI_API_KEY.startswith('sk-') else '❌ Not configured'}")
        
        sys.exit(0)
    else:
        logger.error("❌ Some checks failed!")
        sys.exit(1)


@cli.command()
@click.option("--reset", is_flag=True, help="Reset database")
@click.option("--no-admin", is_flag=True, help="Skip admin user creation")
@click.option("--no-sample", is_flag=True, help="Skip sample data creation")
def init_db(reset: bool, no_admin: bool, no_sample: bool):
    """
    初始化数据库
    """
    cmd = [sys.executable, str(project_root / "scripts" / "init_db.py")]
    
    if reset:
        cmd.append("--reset")
    if no_admin:
        cmd.append("--no-admin")
    if no_sample:
        cmd.append("--no-sample")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


@cli.command()
@click.option("--format", "output_format", default="table", help="Output format (table, json)")
def status(output_format: str):
    """
    显示应用状态
    """
    try:
        # 检查数据库
        database = get_database()
        db_info = database.get_connection_info()
        db_healthy = database.health_check()
        
        # 检查缓存
        cache = get_cache()
        cache_info = cache.get_info()
        cache_healthy = cache.health_check()
        
        # 统计信息
        with database.get_session_context() as session:
            from app.models import User, Conversation, Message
            
            user_count = session.query(User).count()
            conversation_count = session.query(Conversation).count()
            message_count = session.query(Message).count()
        
        if output_format == "json":
            import json
            status_data = {
                "application": {
                    "name": settings.APP_NAME,
                    "version": settings.APP_VERSION,
                    "environment": settings.ENVIRONMENT
                },
                "database": {
                    "healthy": db_healthy,
                    "info": db_info
                },
                "cache": {
                    "healthy": cache_healthy,
                    "info": cache_info
                },
                "statistics": {
                    "users": user_count,
                    "conversations": conversation_count,
                    "messages": message_count
                }
            }
            print(json.dumps(status_data, indent=2))
        else:
            # 表格格式
            print(f"\n{'='*50}")
            print(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
            print(f"  Environment: {settings.ENVIRONMENT}")
            print(f"{'='*50}")
            
            print(f"\n📊 Statistics:")
            print(f"  Users: {user_count}")
            print(f"  Conversations: {conversation_count}")
            print(f"  Messages: {message_count}")
            
            print(f"\n🗄️  Database:")
            print(f"  Status: {'✅ Healthy' if db_healthy else '❌ Unhealthy'}")
            print(f"  Type: {db_info.get('url', 'Unknown').split('://', 1)[0]}")
            
            print(f"\n💾 Cache:")
            print(f"  Status: {'✅ Healthy' if cache_healthy else '❌ Unhealthy'}")
            print(f"  Type: {cache_info.get('type', 'Unknown')}")
            
            print(f"\n🔧 Configuration:")
            print(f"  CORS: {'✅ Enabled' if settings.ENABLE_CORS else '❌ Disabled'}")
            print(f"  Metrics: {'✅ Enabled' if settings.ENABLE_METRICS else '❌ Disabled'}")
            print(f"  Auto Backup: {'✅ Enabled' if settings.ENABLE_AUTO_BACKUP else '❌ Disabled'}")
            print()
    
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        sys.exit(1)


@cli.command()
def install():
    """
    安装依赖
    """
    logger.info("Installing dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", 
            str(project_root / "requirements.txt")
        ], check=True)
        
        logger.info("Dependencies installed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()