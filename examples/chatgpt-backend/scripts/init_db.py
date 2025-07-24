#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库初始化脚本

用于初始化数据库表和创建初始数据。
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加模板库路径
template_root = project_root.parent.parent
sys.path.insert(0, str(template_root))

import asyncio
from typing import Optional

# 导入模板库组件
from src.core.logging import get_logger, setup_logging

# 导入应用组件
from app.core.config import get_settings
from app.core.database import get_database, init_database, reset_database
from app.core.security import get_password_hash
from app.models import User, UserQuota

# 设置日志
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


def create_admin_user() -> None:
    """
    创建管理员用户
    """
    if not settings.ADMIN_USERS:
        logger.info("No admin users configured")
        return
    
    database = get_database()
    
    with database.get_session_context() as session:
        for admin_username in settings.ADMIN_USERS:
            # 检查用户是否已存在
            existing_user = session.query(User).filter(
                User.username == admin_username
            ).first()
            
            if existing_user:
                logger.info(f"Admin user '{admin_username}' already exists")
                continue
            
            # 创建管理员用户
            admin_user = User(
                username=admin_username,
                email=f"{admin_username}@admin.local",
                password_hash=get_password_hash("admin123"),  # 默认密码
                is_active=True,
                is_admin=True,
                is_vip=True,
                is_verified=True
            )
            
            session.add(admin_user)
            session.flush()  # 获取用户ID
            
            # 创建管理员配额
            admin_quota = UserQuota(
                user_id=admin_user.id,
                daily_requests=999999,
                monthly_requests=999999,
                daily_tokens=999999999,
                monthly_tokens=999999999
            )
            
            session.add(admin_quota)
            
            logger.info(f"Created admin user: {admin_username} (password: admin123)")
    
    logger.info(f"Admin users setup completed")


def create_sample_data() -> None:
    """
    创建示例数据（仅在开发环境）
    """
    if settings.ENVIRONMENT != "development":
        logger.info("Skipping sample data creation (not in development environment)")
        return
    
    database = get_database()
    
    with database.get_session_context() as session:
        # 检查是否已有示例用户
        existing_user = session.query(User).filter(
            User.username == "demo_user"
        ).first()
        
        if existing_user:
            logger.info("Sample data already exists")
            return
        
        # 创建示例用户
        demo_user = User(
            username="demo_user",
            email="demo@example.com",
            password_hash=get_password_hash("demo123"),
            is_active=True,
            is_verified=True
        )
        
        session.add(demo_user)
        session.flush()
        
        # 创建示例用户配额
        demo_quota = UserQuota(
            user_id=demo_user.id,
            **settings.get_user_quota(is_vip=False)
        )
        
        session.add(demo_quota)
        
        logger.info("Created demo user: demo_user (password: demo123)")
    
    logger.info("Sample data creation completed")


def verify_database() -> bool:
    """
    验证数据库连接和表结构
    """
    try:
        database = get_database()
        
        # 健康检查
        if not database.health_check():
            logger.error("Database health check failed")
            return False
        
        # 检查表是否存在
        with database.get_session_context() as session:
            # 尝试查询用户表
            user_count = session.query(User).count()
            logger.info(f"Database verification passed. User count: {user_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize ChatGPT Backend database")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset database (drop all tables and recreate)"
    )
    parser.add_argument(
        "--no-admin",
        action="store_true",
        help="Skip admin user creation"
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Skip sample data creation"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify database connection"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting database initialization...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Database URL: {settings.DATABASE_URL}")
    
    try:
        if args.verify_only:
            # 仅验证数据库
            if verify_database():
                logger.info("Database verification successful")
                sys.exit(0)
            else:
                logger.error("Database verification failed")
                sys.exit(1)
        
        if args.reset:
            # 重置数据库
            logger.warning("Resetting database (all data will be lost)...")
            if settings.ENVIRONMENT == "production":
                confirm = input("Are you sure you want to reset the production database? (yes/no): ")
                if confirm.lower() != "yes":
                    logger.info("Database reset cancelled")
                    sys.exit(0)
            
            reset_database()
            logger.info("Database reset completed")
        else:
            # 初始化数据库
            init_database()
            logger.info("Database initialization completed")
        
        # 验证数据库
        if not verify_database():
            logger.error("Database verification failed after initialization")
            sys.exit(1)
        
        # 创建管理员用户
        if not args.no_admin:
            create_admin_user()
        
        # 创建示例数据
        if not args.no_sample:
            create_sample_data()
        
        logger.info("Database setup completed successfully!")
        
        # 显示连接信息
        database = get_database()
        connection_info = database.get_connection_info()
        logger.info(f"Database connection info: {connection_info}")
        
        # 显示用户信息
        with database.get_session_context() as session:
            user_count = session.query(User).count()
            admin_count = session.query(User).filter(User.is_admin == True).count()
            
            logger.info(f"Total users: {user_count}")
            logger.info(f"Admin users: {admin_count}")
            
            if settings.ENVIRONMENT == "development":
                logger.info("\n=== Development Environment Login Info ===")
                if settings.ADMIN_USERS:
                    for admin_user in settings.ADMIN_USERS:
                        logger.info(f"Admin: {admin_user} / admin123")
                logger.info("Demo: demo_user / demo123")
                logger.info("=========================================\n")
    
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()