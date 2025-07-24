# -*- coding: utf-8 -*-
"""
消息API路由

提供消息发送、接收、管理等功能，包括流式响应支持。
"""

import sys
from pathlib import Path
from typing import List, Optional, AsyncGenerator
from datetime import datetime

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
import json

# 导入模板库组件
from src.utils.response import create_response, create_error_response
from src.utils.pagination import paginate
from src.utils.exceptions import ValidationException, AuthorizationException
from src.core.logging import get_logger

# 导入应用组件
from app.core.database import get_db
from app.core.openai_client import openai_client
from app.models.user import User
from app.models.message import (
    Message, MessageCreate, MessageUpdate, MessageResponse, 
    MessageDetail, MessageList, MessageStream, MessageStats,
    MessageRole, MessageStatus
)
from app.models.conversation import Conversation
from app.api.v1.auth import get_current_active_user
from app.services.message_service import MessageService
from app.services.usage_service import UsageService

logger = get_logger(__name__)
router = APIRouter()

# 服务实例
message_service = MessageService()
usage_service = UsageService()


class MessageFilter(BaseModel):
    """消息过滤器"""
    role: Optional[MessageRole] = None
    status: Optional[MessageStatus] = None
    search: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    has_attachments: Optional[bool] = None


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    conversation_id: Optional[int] = None
    model: Optional[str] = "gpt-3.5-turbo"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    system_prompt: Optional[str] = None


@router.get("/conversations/{conversation_id}/messages", response_model=MessageList, summary="获取对话消息")
async def get_conversation_messages(
    conversation_id: int,
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(50, ge=1, le=100, description="每页数量"),
    role: Optional[MessageRole] = Query(None, description="消息角色过滤"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取指定对话的消息列表
    
    支持分页、过滤和搜索功能：
    - **page**: 页码（从1开始）
    - **size**: 每页数量（1-100）
    - **role**: 消息角色（user, assistant, system）
    - **search**: 搜索消息内容
    """
    try:
        # 验证对话权限
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="对话不存在"
            )
        
        filter_params = MessageFilter(
            role=role,
            search=search
        )
        
        messages, total = message_service.get_conversation_messages(
            db=db,
            conversation_id=conversation_id,
            page=page,
            size=size,
            filters=filter_params
        )
        
        return MessageList(
            messages=[MessageResponse.from_orm(msg) for msg in messages],
            total=total,
            page=page,
            size=size,
            pages=(total + size - 1) // size
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取消息列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取消息列表失败"
        )


@router.post("/chat", summary="发送聊天消息")
async def send_chat_message(
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    发送聊天消息并获取AI回复
    
    支持流式和非流式响应：
    - **message**: 用户消息内容
    - **conversation_id**: 对话ID（可选，不提供则创建新对话）
    - **model**: AI模型（默认gpt-3.5-turbo）
    - **temperature**: 温度参数（0-2）
    - **max_tokens**: 最大token数
    - **stream**: 是否流式响应
    - **system_prompt**: 系统提示（可选）
    """
    try:
        # 检查用户配额
        if not usage_service.check_user_quota(db, current_user.id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="已达到使用配额限制"
            )
        
        # 创建用户消息
        user_message = message_service.create_message(
            db=db,
            conversation_id=chat_request.conversation_id,
            user_id=current_user.id,
            message_data=MessageCreate(
                content=chat_request.message,
                role=MessageRole.user
            )
        )
        
        # 准备对话历史
        conversation_history = message_service.get_conversation_history(
            db=db,
            conversation_id=user_message.conversation_id,
            limit=20  # 限制历史消息数量
        )
        
        # 准备OpenAI消息格式
        openai_messages = []
        
        # 添加系统提示
        if chat_request.system_prompt:
            openai_messages.append({
                "role": "system",
                "content": chat_request.system_prompt
            })
        
        # 添加历史消息
        for msg in conversation_history:
            openai_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        # 流式响应
        if chat_request.stream:
            return StreamingResponse(
                stream_chat_response(
                    openai_messages=openai_messages,
                    model=chat_request.model,
                    temperature=chat_request.temperature,
                    max_tokens=chat_request.max_tokens,
                    conversation_id=user_message.conversation_id,
                    user_id=current_user.id,
                    db=db,
                    background_tasks=background_tasks
                ),
                media_type="text/plain"
            )
        
        # 非流式响应
        else:
            response = await openai_client.create_chat_completion(
                messages=openai_messages,
                model=chat_request.model,
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens,
                stream=False
            )
            
            # 创建AI回复消息
            ai_message = message_service.create_message(
                db=db,
                conversation_id=user_message.conversation_id,
                user_id=current_user.id,
                message_data=MessageCreate(
                    content=response.choices[0].message.content,
                    role=MessageRole.assistant,
                    model=chat_request.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
            )
            
            # 记录使用量
            background_tasks.add_task(
                usage_service.record_usage,
                db=db,
                user_id=current_user.id,
                usage_type="chat",
                model=chat_request.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                conversation_id=user_message.conversation_id,
                message_id=ai_message.id
            )
            
            return {
                "user_message": MessageResponse.from_orm(user_message),
                "ai_message": MessageResponse.from_orm(ai_message),
                "conversation_id": user_message.conversation_id
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"发送消息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="发送消息失败"
        )


async def stream_chat_response(
    openai_messages: List[dict],
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    conversation_id: int,
    user_id: int,
    db: Session,
    background_tasks: BackgroundTasks
) -> AsyncGenerator[str, None]:
    """
    流式聊天响应生成器
    """
    try:
        # 创建流式响应
        stream = await openai_client.create_chat_completion(
            messages=openai_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        full_content = ""
        prompt_tokens = 0
        completion_tokens = 0
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content += content
                completion_tokens += 1
                
                # 发送流式数据
                yield f"data: {json.dumps({'content': content, 'type': 'content'})}\n\n"
        
        # 创建AI回复消息
        ai_message = message_service.create_message(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            message_data=MessageCreate(
                content=full_content,
                role=MessageRole.assistant,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
        # 发送完成信号
        yield f"data: {json.dumps({'type': 'done', 'message_id': ai_message.id})}\n\n"
        
        # 记录使用量
        background_tasks.add_task(
            usage_service.record_usage,
            db=db,
            user_id=user_id,
            usage_type="chat",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            conversation_id=conversation_id,
            message_id=ai_message.id
        )
        
    except Exception as e:
        logger.error(f"流式响应失败: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': '生成回复失败'})}\n\n"


@router.get("/messages/{message_id}", response_model=MessageDetail, summary="获取消息详情")
async def get_message(
    message_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取消息详情
    """
    try:
        message = message_service.get_message(
            db=db,
            message_id=message_id,
            user_id=current_user.id
        )
        
        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="消息不存在"
            )
        
        return MessageDetail.from_orm(message)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取消息详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取消息详情失败"
        )


@router.put("/messages/{message_id}", response_model=MessageResponse, summary="更新消息")
async def update_message(
    message_id: int,
    message_data: MessageUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    更新消息内容
    
    - **content**: 消息内容
    - **is_edited**: 是否标记为已编辑
    """
    try:
        message = message_service.update_message(
            db=db,
            message_id=message_id,
            user_id=current_user.id,
            message_data=message_data
        )
        
        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="消息不存在"
            )
        
        logger.info(f"更新消息: ID {message_id} (User: {current_user.id})")
        
        return MessageResponse.from_orm(message)
        
    except HTTPException:
        raise
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"更新消息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新消息失败"
        )


@router.delete("/messages/{message_id}", summary="删除消息")
async def delete_message(
    message_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    删除消息
    """
    try:
        success = message_service.delete_message(
            db=db,
            message_id=message_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="消息不存在"
            )
        
        logger.info(f"删除消息: ID {message_id} (User: {current_user.id})")
        
        return create_response(message="消息删除成功")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除消息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除消息失败"
        )


@router.post("/messages/{message_id}/regenerate", response_model=MessageResponse, summary="重新生成消息")
async def regenerate_message(
    message_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    重新生成AI消息
    """
    try:
        # 检查用户配额
        if not usage_service.check_user_quota(db, current_user.id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="已达到使用配额限制"
            )
        
        new_message = message_service.regenerate_message(
            db=db,
            message_id=message_id,
            user_id=current_user.id
        )
        
        if not new_message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="消息不存在或无法重新生成"
            )
        
        logger.info(f"重新生成消息: ID {message_id} -> {new_message.id} (User: {current_user.id})")
        
        return MessageResponse.from_orm(new_message)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重新生成消息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="重新生成消息失败"
        )


@router.post("/messages/{message_id}/pin", summary="置顶消息")
async def pin_message(
    message_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    置顶消息
    """
    try:
        success = message_service.pin_message(
            db=db,
            message_id=message_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="消息不存在"
            )
        
        return create_response(message="消息置顶成功")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"置顶消息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="置顶消息失败"
        )


@router.delete("/messages/{message_id}/pin", summary="取消置顶消息")
async def unpin_message(
    message_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    取消置顶消息
    """
    try:
        success = message_service.unpin_message(
            db=db,
            message_id=message_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="消息不存在"
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


@router.get("/messages/stats", response_model=MessageStats, summary="获取消息统计")
async def get_message_stats(
    conversation_id: Optional[int] = Query(None, description="对话ID过滤"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取消息统计信息
    """
    try:
        stats = message_service.get_message_stats(
            db=db,
            user_id=current_user.id,
            conversation_id=conversation_id
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"获取消息统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取消息统计失败"
        )


@router.get("/messages/search", response_model=MessageList, summary="搜索消息")
async def search_messages(
    query: str = Query(..., description="搜索关键词"),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页数量"),
    conversation_id: Optional[int] = Query(None, description="对话ID过滤"),
    role: Optional[MessageRole] = Query(None, description="消息角色过滤"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    搜索消息
    
    - **query**: 搜索关键词
    - **conversation_id**: 限制在特定对话中搜索
    - **role**: 消息角色过滤
    """
    try:
        messages, total = message_service.search_messages(
            db=db,
            user_id=current_user.id,
            query=query,
            page=page,
            size=size,
            conversation_id=conversation_id,
            role=role
        )
        
        return MessageList(
            messages=[MessageResponse.from_orm(msg) for msg in messages],
            total=total,
            page=page,
            size=size,
            pages=(total + size - 1) // size
        )
        
    except Exception as e:
        logger.error(f"搜索消息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="搜索消息失败"
        )