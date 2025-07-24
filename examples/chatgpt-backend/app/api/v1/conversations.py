# -*- coding: utf-8 -*-
"""
对话API路由

提供对话创建、查询、更新、删除等管理功能。
"""

import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

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
from app.models.user import User
from app.models.conversation import (
    Conversation, ConversationCreate, ConversationUpdate, 
    ConversationResponse, ConversationDetail, ConversationList,
    ConversationSummary, ConversationStats, ConversationShare
)
from app.api.v1.auth import get_current_active_user
from app.services.conversation_service import ConversationService

logger = get_logger(__name__)
router = APIRouter()

# 对话服务实例
conversation_service = ConversationService()


class ConversationFilter(BaseModel):
    """对话过滤器"""
    status: Optional[str] = None
    model: Optional[str] = None
    is_pinned: Optional[bool] = None
    tags: Optional[List[str]] = None
    search: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


@router.get("/", response_model=ConversationList, summary="获取对话列表")
async def get_conversations(
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页数量"),
    status: Optional[str] = Query(None, description="对话状态过滤"),
    model: Optional[str] = Query(None, description="模型过滤"),
    is_pinned: Optional[bool] = Query(None, description="是否置顶过滤"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取当前用户的对话列表
    
    支持分页、过滤和搜索功能：
    - **page**: 页码（从1开始）
    - **size**: 每页数量（1-100）
    - **status**: 对话状态（active, archived, deleted）
    - **model**: AI模型过滤
    - **is_pinned**: 是否置顶
    - **search**: 搜索对话标题和描述
    """
    try:
        filter_params = ConversationFilter(
            status=status,
            model=model,
            is_pinned=is_pinned,
            search=search
        )
        
        conversations, total = conversation_service.get_user_conversations(
            db=db,
            user_id=current_user.id,
            page=page,
            size=size,
            filters=filter_params
        )
        
        return ConversationList(
            conversations=[ConversationResponse.from_orm(conv) for conv in conversations],
            total=total,
            page=page,
            size=size,
            pages=(total + size - 1) // size
        )
        
    except Exception as e:
        logger.error(f"获取对话列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取对话列表失败"
        )


@router.post("/", response_model=ConversationResponse, summary="创建新对话")
async def create_conversation(
    conversation_data: ConversationCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    创建新对话
    
    - **title**: 对话标题
    - **description**: 对话描述（可选）
    - **model**: AI模型（默认gpt-3.5-turbo）
    - **system_prompt**: 系统提示（可选）
    - **temperature**: 温度参数（0-2）
    - **max_tokens**: 最大token数
    - **tags**: 标签列表
    """
    try:
        conversation = conversation_service.create_conversation(
            db=db,
            user_id=current_user.id,
            conversation_data=conversation_data
        )
        
        logger.info(f"创建对话: {conversation.title} (ID: {conversation.id}, User: {current_user.id})")
        
        return ConversationResponse.from_orm(conversation)
        
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"创建对话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="创建对话失败"
        )


@router.get("/{conversation_id}", response_model=ConversationDetail, summary="获取对话详情")
async def get_conversation(
    conversation_id: int,
    include_messages: bool = Query(True, description="是否包含消息列表"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取对话详情
    
    - **conversation_id**: 对话ID
    - **include_messages**: 是否包含消息列表
    """
    try:
        conversation = conversation_service.get_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id,
            include_messages=include_messages
        )
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="对话不存在"
            )
        
        return ConversationDetail.from_orm(conversation)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取对话详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取对话详情失败"
        )


@router.put("/{conversation_id}", response_model=ConversationResponse, summary="更新对话")
async def update_conversation(
    conversation_id: int,
    conversation_data: ConversationUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    更新对话信息
    
    - **title**: 对话标题
    - **description**: 对话描述
    - **system_prompt**: 系统提示
    - **temperature**: 温度参数
    - **max_tokens**: 最大token数
    - **tags**: 标签列表
    """
    try:
        conversation = conversation_service.update_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id,
            conversation_data=conversation_data
        )
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="对话不存在"
            )
        
        logger.info(f"更新对话: {conversation.title} (ID: {conversation.id}, User: {current_user.id})")
        
        return ConversationResponse.from_orm(conversation)
        
    except HTTPException:
        raise
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"更新对话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新对话失败"
        )


@router.delete("/{conversation_id}", summary="删除对话")
async def delete_conversation(
    conversation_id: int,
    permanent: bool = Query(False, description="是否永久删除"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    删除对话
    
    - **conversation_id**: 对话ID
    - **permanent**: 是否永久删除（默认为软删除）
    """
    try:
        success = conversation_service.delete_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id,
            permanent=permanent
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="对话不存在"
            )
        
        logger.info(f"删除对话: ID {conversation_id} (User: {current_user.id}, Permanent: {permanent})")
        
        return create_response(
            message="对话删除成功" if permanent else "对话已移至回收站"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除对话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除对话失败"
        )


@router.post("/{conversation_id}/pin", summary="置顶对话")
async def pin_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    置顶对话
    """
    try:
        success = conversation_service.pin_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="对话不存在"
            )
        
        return create_response(message="对话置顶成功")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"置顶对话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="置顶对话失败"
        )


@router.delete("/{conversation_id}/pin", summary="取消置顶对话")
async def unpin_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    取消置顶对话
    """
    try:
        success = conversation_service.unpin_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="对话不存在"
            )
        
        return create_response(message="取消置顶成功")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消置顶失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="取消置顶失败"
        )


@router.post("/{conversation_id}/archive", summary="归档对话")
async def archive_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    归档对话
    """
    try:
        success = conversation_service.archive_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="对话不存在"
            )
        
        return create_response(message="对话归档成功")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"归档对话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="归档对话失败"
        )


@router.post("/{conversation_id}/restore", summary="恢复对话")
async def restore_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    恢复已归档或已删除的对话
    """
    try:
        success = conversation_service.restore_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="对话不存在"
            )
        
        return create_response(message="对话恢复成功")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"恢复对话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="恢复对话失败"
        )


@router.get("/{conversation_id}/share", response_model=ConversationShare, summary="获取对话分享信息")
async def get_conversation_share(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取对话分享信息
    """
    try:
        share_info = conversation_service.get_conversation_share(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id
        )
        
        if not share_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="对话不存在"
            )
        
        return share_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取分享信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取分享信息失败"
        )


@router.post("/{conversation_id}/share", response_model=ConversationShare, summary="分享对话")
async def share_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    创建对话分享链接
    """
    try:
        share_info = conversation_service.create_conversation_share(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id
        )
        
        if not share_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="对话不存在"
            )
        
        logger.info(f"分享对话: ID {conversation_id} (User: {current_user.id})")
        
        return share_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分享对话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="分享对话失败"
        )


@router.delete("/{conversation_id}/share", summary="取消分享对话")
async def unshare_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    取消对话分享
    """
    try:
        success = conversation_service.remove_conversation_share(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="对话不存在"
            )
        
        return create_response(message="取消分享成功")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消分享失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="取消分享失败"
        )


@router.get("/stats/summary", response_model=ConversationStats, summary="获取对话统计")
async def get_conversation_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取当前用户的对话统计信息
    """
    try:
        stats = conversation_service.get_user_conversation_stats(
            db=db,
            user_id=current_user.id
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"获取对话统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取对话统计失败"
        )


@router.get("/summaries", response_model=List[ConversationSummary], summary="获取对话摘要列表")
async def get_conversation_summaries(
    limit: int = Query(50, ge=1, le=100, description="返回数量限制"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取对话摘要列表（用于快速浏览）
    """
    try:
        summaries = conversation_service.get_conversation_summaries(
            db=db,
            user_id=current_user.id,
            limit=limit
        )
        
        return [ConversationSummary.from_orm(summary) for summary in summaries]
        
    except Exception as e:
        logger.error(f"获取对话摘要失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取对话摘要失败"
        )