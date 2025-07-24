# -*- coding: utf-8 -*-
"""
用户管理API路由

提供用户信息管理、配额查看、使用统计等功能。
"""

import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

# 导入模板库组件
from src.utils.response import create_response, create_error_response
from src.utils.pagination import paginate
from src.utils.exceptions import ValidationException, AuthorizationException
from src.core.logging import get_logger

# 导入应用组件
from app.core.database import get_db
from app.models.user import (
    User, UserUpdate, UserPasswordUpdate, UserResponse, 
    UserProfile, UserStats, UserRole
)
from app.models.usage import UserQuotaResponse, UsageStats, UsageReport
from app.api.v1.auth import get_current_active_user, get_current_admin_user
from app.services.user_service import UserService
from app.services.usage_service import UsageService

logger = get_logger(__name__)
router = APIRouter()

# 服务实例
user_service = UserService()
usage_service = UsageService()


class UserFilter(BaseModel):
    """用户过滤器（管理员用）"""
    role: Optional[UserRole] = None
    status: Optional[str] = None
    search: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


@router.get("/me", response_model=UserResponse, summary="获取当前用户信息")
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    获取当前登录用户的详细信息
    """
    return UserResponse.from_orm(current_user)


@router.put("/me", response_model=UserResponse, summary="更新当前用户信息")
async def update_current_user(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    更新当前用户信息
    
    - **username**: 用户名
    - **email**: 邮箱地址
    - **full_name**: 全名
    - **avatar_url**: 头像URL
    - **bio**: 个人简介
    - **preferences**: 个人设置
    """
    try:
        updated_user = user_service.update_user(
            db=db,
            user_id=current_user.id,
            user_data=user_data
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="更新用户信息失败"
            )
        
        logger.info(f"用户更新信息: {updated_user.username} (ID: {updated_user.id})")
        
        return UserResponse.from_orm(updated_user)
        
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"更新用户信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新用户信息失败"
        )


@router.put("/me/password", summary="修改密码")
async def change_password(
    password_data: UserPasswordUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    修改当前用户密码
    
    - **current_password**: 当前密码
    - **new_password**: 新密码
    - **confirm_password**: 确认新密码
    """
    try:
        success = user_service.change_password(
            db=db,
            user_id=current_user.id,
            password_data=password_data
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="密码修改失败，请检查当前密码是否正确"
            )
        
        logger.info(f"用户修改密码: {current_user.username} (ID: {current_user.id})")
        
        return create_response(message="密码修改成功")
        
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"修改密码失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="修改密码失败"
        )


@router.get("/me/profile", response_model=UserProfile, summary="获取用户档案")
async def get_user_profile(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取用户详细档案信息
    """
    try:
        profile = user_service.get_user_profile(
            db=db,
            user_id=current_user.id
        )
        
        return profile
        
    except Exception as e:
        logger.error(f"获取用户档案失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户档案失败"
        )


@router.get("/me/stats", response_model=UserStats, summary="获取用户统计")
async def get_user_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取用户使用统计信息
    """
    try:
        stats = user_service.get_user_stats(
            db=db,
            user_id=current_user.id
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"获取用户统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户统计失败"
        )


@router.get("/me/quota", response_model=UserQuotaResponse, summary="获取用户配额")
async def get_user_quota(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取用户配额信息
    """
    try:
        quota = usage_service.get_user_quota(
            db=db,
            user_id=current_user.id
        )
        
        return quota
        
    except Exception as e:
        logger.error(f"获取用户配额失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户配额失败"
        )


@router.get("/me/usage", response_model=UsageStats, summary="获取使用统计")
async def get_usage_stats(
    days: int = Query(30, ge=1, le=365, description="统计天数"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取用户使用统计
    
    - **days**: 统计天数（1-365）
    """
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        stats = usage_service.get_usage_stats(
            db=db,
            user_id=current_user.id,
            start_date=start_date,
            end_date=end_date
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"获取使用统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取使用统计失败"
        )


@router.get("/me/usage/report", response_model=UsageReport, summary="获取使用报告")
async def get_usage_report(
    start_date: Optional[datetime] = Query(None, description="开始日期"),
    end_date: Optional[datetime] = Query(None, description="结束日期"),
    group_by: str = Query("day", description="分组方式: day, week, month"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取详细使用报告
    
    - **start_date**: 开始日期（默认30天前）
    - **end_date**: 结束日期（默认今天）
    - **group_by**: 分组方式（day, week, month）
    """
    try:
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        report = usage_service.get_usage_report(
            db=db,
            user_id=current_user.id,
            start_date=start_date,
            end_date=end_date,
            group_by=group_by
        )
        
        return report
        
    except Exception as e:
        logger.error(f"获取使用报告失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取使用报告失败"
        )


@router.delete("/me", summary="删除账户")
async def delete_account(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    删除当前用户账户（软删除）
    """
    try:
        success = user_service.delete_user(
            db=db,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="删除账户失败"
            )
        
        logger.info(f"用户删除账户: {current_user.username} (ID: {current_user.id})")
        
        return create_response(message="账户删除成功")
        
    except Exception as e:
        logger.error(f"删除账户失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除账户失败"
        )


# 管理员专用路由
@router.get("/", response_model=List[UserResponse], summary="获取用户列表（管理员）")
async def get_users(
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页数量"),
    role: Optional[UserRole] = Query(None, description="角色过滤"),
    status: Optional[str] = Query(None, description="状态过滤"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    获取用户列表（仅管理员）
    
    支持分页、过滤和搜索功能：
    - **page**: 页码（从1开始）
    - **size**: 每页数量（1-100）
    - **role**: 用户角色过滤
    - **status**: 用户状态过滤
    - **search**: 搜索用户名、邮箱、全名
    """
    try:
        filter_params = UserFilter(
            role=role,
            status=status,
            search=search
        )
        
        users, total = user_service.get_users(
            db=db,
            page=page,
            size=size,
            filters=filter_params
        )
        
        return {
            "users": [UserResponse.from_orm(user) for user in users],
            "total": total,
            "page": page,
            "size": size,
            "pages": (total + size - 1) // size
        }
        
    except Exception as e:
        logger.error(f"获取用户列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户列表失败"
        )


@router.get("/{user_id}", response_model=UserResponse, summary="获取用户信息（管理员）")
async def get_user(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    获取指定用户信息（仅管理员）
    """
    try:
        user = user_service.get_user_by_id(
            db=db,
            user_id=user_id
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        return UserResponse.from_orm(user)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户信息失败"
        )


@router.put("/{user_id}", response_model=UserResponse, summary="更新用户信息（管理员）")
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    更新指定用户信息（仅管理员）
    """
    try:
        updated_user = user_service.update_user(
            db=db,
            user_id=user_id,
            user_data=user_data
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        logger.info(f"管理员更新用户: {updated_user.username} (ID: {updated_user.id}, Admin: {current_user.id})")
        
        return UserResponse.from_orm(updated_user)
        
    except HTTPException:
        raise
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"更新用户信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新用户信息失败"
        )


@router.post("/{user_id}/activate", summary="激活用户（管理员）")
async def activate_user(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    激活用户账户（仅管理员）
    """
    try:
        success = user_service.activate_user(
            db=db,
            user_id=user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        logger.info(f"管理员激活用户: ID {user_id} (Admin: {current_user.id})")
        
        return create_response(message="用户激活成功")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"激活用户失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="激活用户失败"
        )


@router.post("/{user_id}/deactivate", summary="停用用户（管理员）")
async def deactivate_user(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    停用用户账户（仅管理员）
    """
    try:
        success = user_service.deactivate_user(
            db=db,
            user_id=user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        logger.info(f"管理员停用用户: ID {user_id} (Admin: {current_user.id})")
        
        return create_response(message="用户停用成功")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"停用用户失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="停用用户失败"
        )


@router.delete("/{user_id}", summary="删除用户（管理员）")
async def delete_user(
    user_id: int,
    permanent: bool = Query(False, description="是否永久删除"),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    删除用户账户（仅管理员）
    
    - **user_id**: 用户ID
    - **permanent**: 是否永久删除（默认为软删除）
    """
    try:
        success = user_service.delete_user(
            db=db,
            user_id=user_id,
            permanent=permanent
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        logger.info(f"管理员删除用户: ID {user_id} (Admin: {current_user.id}, Permanent: {permanent})")
        
        return create_response(
            message="用户删除成功" if permanent else "用户已移至回收站"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除用户失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除用户失败"
        )