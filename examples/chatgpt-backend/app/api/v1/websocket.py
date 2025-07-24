# -*- coding: utf-8 -*-
"""
WebSocket路由

提供实时通信功能，支持实时聊天、状态更新等。
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import json
import asyncio
from datetime import datetime

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, ValidationError

# 导入模板库组件
from src.core.logging import get_logger
from src.utils.exceptions import ValidationException, AuthorizationException

# 导入应用组件
from app.core.database import get_db
from app.core.openai_client import openai_client
from app.models.user import User
from app.models.message import MessageCreate, MessageRole
from app.api.v1.auth import get_user_from_token
from app.services.message_service import MessageService
from app.services.conversation_service import ConversationService
from app.services.usage_service import UsageService

logger = get_logger(__name__)
router = APIRouter()

# 服务实例
message_service = MessageService()
conversation_service = ConversationService()
usage_service = UsageService()


class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        # 存储活跃连接: {user_id: {connection_id: websocket}}
        self.active_connections: Dict[int, Dict[str, WebSocket]] = {}
        # 存储用户会话信息: {user_id: {connection_id: session_info}}
        self.user_sessions: Dict[int, Dict[str, dict]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: int, connection_id: str):
        """建立连接"""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = {}
            self.user_sessions[user_id] = {}
        
        self.active_connections[user_id][connection_id] = websocket
        self.user_sessions[user_id][connection_id] = {
            "connected_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        
        logger.info(f"WebSocket连接建立: User {user_id}, Connection {connection_id}")
        
        # 发送连接成功消息
        await self.send_personal_message({
            "type": "connection",
            "status": "connected",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat()
        }, user_id, connection_id)
    
    def disconnect(self, user_id: int, connection_id: str):
        """断开连接"""
        if user_id in self.active_connections:
            if connection_id in self.active_connections[user_id]:
                del self.active_connections[user_id][connection_id]
            if connection_id in self.user_sessions[user_id]:
                del self.user_sessions[user_id][connection_id]
            
            # 如果用户没有其他连接，清理用户记录
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                del self.user_sessions[user_id]
        
        logger.info(f"WebSocket连接断开: User {user_id}, Connection {connection_id}")
    
    async def send_personal_message(self, message: dict, user_id: int, connection_id: Optional[str] = None):
        """发送个人消息"""
        if user_id not in self.active_connections:
            return
        
        message_text = json.dumps(message, ensure_ascii=False)
        
        if connection_id:
            # 发送给特定连接
            if connection_id in self.active_connections[user_id]:
                try:
                    await self.active_connections[user_id][connection_id].send_text(message_text)
                    # 更新最后活动时间
                    if user_id in self.user_sessions and connection_id in self.user_sessions[user_id]:
                        self.user_sessions[user_id][connection_id]["last_activity"] = datetime.utcnow()
                except Exception as e:
                    logger.error(f"发送消息失败: {e}")
                    self.disconnect(user_id, connection_id)
        else:
            # 发送给用户的所有连接
            disconnected_connections = []
            for conn_id, websocket in self.active_connections[user_id].items():
                try:
                    await websocket.send_text(message_text)
                    # 更新最后活动时间
                    if user_id in self.user_sessions and conn_id in self.user_sessions[user_id]:
                        self.user_sessions[user_id][conn_id]["last_activity"] = datetime.utcnow()
                except Exception as e:
                    logger.error(f"发送消息失败: {e}")
                    disconnected_connections.append(conn_id)
            
            # 清理断开的连接
            for conn_id in disconnected_connections:
                self.disconnect(user_id, conn_id)
    
    async def broadcast_to_user(self, message: dict, user_id: int):
        """向用户的所有连接广播消息"""
        await self.send_personal_message(message, user_id)
    
    def get_user_connections(self, user_id: int) -> List[str]:
        """获取用户的所有连接ID"""
        if user_id in self.active_connections:
            return list(self.active_connections[user_id].keys())
        return []
    
    def get_active_users(self) -> List[int]:
        """获取所有活跃用户ID"""
        return list(self.active_connections.keys())
    
    def get_connection_count(self) -> int:
        """获取总连接数"""
        total = 0
        for connections in self.active_connections.values():
            total += len(connections)
        return total


# 全局连接管理器
manager = ConnectionManager()


class WebSocketMessage(BaseModel):
    """WebSocket消息格式"""
    type: str
    data: dict
    conversation_id: Optional[int] = None
    message_id: Optional[int] = None


class ChatMessage(BaseModel):
    """聊天消息"""
    content: str
    conversation_id: Optional[int] = None
    model: Optional[str] = "gpt-3.5-turbo"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None


@router.websocket("/ws/{token}")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str,
    connection_id: str = "default"
):
    """
    WebSocket连接端点
    
    - **token**: JWT认证令牌
    - **connection_id**: 连接标识符（可选，用于多连接管理）
    """
    user = None
    try:
        # 验证用户身份
        user = await get_user_from_token(token)
        if not user:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
            return
        
        # 建立连接
        await manager.connect(websocket, user.id, connection_id)
        
        # 获取数据库会话
        db = next(get_db())
        
        try:
            while True:
                # 接收消息
                data = await websocket.receive_text()
                
                try:
                    # 解析消息
                    message_data = json.loads(data)
                    ws_message = WebSocketMessage(**message_data)
                    
                    # 处理不同类型的消息
                    await handle_websocket_message(
                        ws_message, user, connection_id, db
                    )
                    
                except ValidationError as e:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": "消息格式错误",
                        "details": str(e)
                    }, user.id, connection_id)
                    
                except Exception as e:
                    logger.error(f"处理WebSocket消息失败: {e}")
                    await manager.send_personal_message({
                        "type": "error",
                        "message": "处理消息失败",
                        "details": str(e)
                    }, user.id, connection_id)
        
        finally:
            db.close()
            
    except WebSocketDisconnect:
        if user:
            manager.disconnect(user.id, connection_id)
    except Exception as e:
        logger.error(f"WebSocket连接错误: {e}")
        if user:
            manager.disconnect(user.id, connection_id)


async def handle_websocket_message(
    ws_message: WebSocketMessage,
    user: User,
    connection_id: str,
    db: Session
):
    """
    处理WebSocket消息
    """
    message_type = ws_message.type
    
    if message_type == "ping":
        # 心跳检测
        await manager.send_personal_message({
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        }, user.id, connection_id)
    
    elif message_type == "chat":
        # 聊天消息
        await handle_chat_message(ws_message, user, connection_id, db)
    
    elif message_type == "typing":
        # 输入状态
        await handle_typing_status(ws_message, user, connection_id, db)
    
    elif message_type == "join_conversation":
        # 加入对话
        await handle_join_conversation(ws_message, user, connection_id, db)
    
    elif message_type == "leave_conversation":
        # 离开对话
        await handle_leave_conversation(ws_message, user, connection_id, db)
    
    elif message_type == "get_status":
        # 获取状态
        await handle_get_status(ws_message, user, connection_id, db)
    
    else:
        await manager.send_personal_message({
            "type": "error",
            "message": f"未知消息类型: {message_type}"
        }, user.id, connection_id)


async def handle_chat_message(
    ws_message: WebSocketMessage,
    user: User,
    connection_id: str,
    db: Session
):
    """
    处理聊天消息
    """
    try:
        chat_data = ChatMessage(**ws_message.data)
        
        # 检查用户配额
        if not usage_service.check_user_quota(db, user.id):
            await manager.send_personal_message({
                "type": "error",
                "message": "已达到使用配额限制"
            }, user.id, connection_id)
            return
        
        # 创建用户消息
        user_message = message_service.create_message(
            db=db,
            conversation_id=chat_data.conversation_id,
            user_id=user.id,
            message_data=MessageCreate(
                content=chat_data.content,
                role=MessageRole.user
            )
        )
        
        # 发送用户消息确认
        await manager.send_personal_message({
            "type": "message_created",
            "message": {
                "id": user_message.id,
                "content": user_message.content,
                "role": user_message.role.value,
                "conversation_id": user_message.conversation_id,
                "created_at": user_message.created_at.isoformat()
            }
        }, user.id, connection_id)
        
        # 准备对话历史
        conversation_history = message_service.get_conversation_history(
            db=db,
            conversation_id=user_message.conversation_id,
            limit=20
        )
        
        # 准备OpenAI消息格式
        openai_messages = []
        
        # 添加系统提示
        if chat_data.system_prompt:
            openai_messages.append({
                "role": "system",
                "content": chat_data.system_prompt
            })
        
        # 添加历史消息
        for msg in conversation_history:
            openai_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        # 发送开始生成信号
        await manager.send_personal_message({
            "type": "ai_response_start",
            "conversation_id": user_message.conversation_id
        }, user.id, connection_id)
        
        # 创建流式响应
        stream = await openai_client.create_chat_completion(
            messages=openai_messages,
            model=chat_data.model,
            temperature=chat_data.temperature,
            max_tokens=chat_data.max_tokens,
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
                
                # 发送流式内容
                await manager.send_personal_message({
                    "type": "ai_response_chunk",
                    "content": content,
                    "conversation_id": user_message.conversation_id
                }, user.id, connection_id)
        
        # 创建AI回复消息
        ai_message = message_service.create_message(
            db=db,
            conversation_id=user_message.conversation_id,
            user_id=user.id,
            message_data=MessageCreate(
                content=full_content,
                role=MessageRole.assistant,
                model=chat_data.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
        # 发送完成信号
        await manager.send_personal_message({
            "type": "ai_response_complete",
            "message": {
                "id": ai_message.id,
                "content": ai_message.content,
                "role": ai_message.role.value,
                "conversation_id": ai_message.conversation_id,
                "model": ai_message.model,
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": prompt_tokens + completion_tokens
                },
                "created_at": ai_message.created_at.isoformat()
            }
        }, user.id, connection_id)
        
        # 记录使用量
        usage_service.record_usage(
            db=db,
            user_id=user.id,
            usage_type="chat",
            model=chat_data.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            conversation_id=user_message.conversation_id,
            message_id=ai_message.id
        )
        
    except Exception as e:
        logger.error(f"处理聊天消息失败: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": "发送消息失败",
            "details": str(e)
        }, user.id, connection_id)


async def handle_typing_status(
    ws_message: WebSocketMessage,
    user: User,
    connection_id: str,
    db: Session
):
    """
    处理输入状态
    """
    # 这里可以实现输入状态的广播逻辑
    # 例如在多用户对话中通知其他用户某人正在输入
    pass


async def handle_join_conversation(
    ws_message: WebSocketMessage,
    user: User,
    connection_id: str,
    db: Session
):
    """
    处理加入对话
    """
    conversation_id = ws_message.conversation_id
    if conversation_id:
        # 验证用户是否有权限访问该对话
        conversation = conversation_service.get_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user.id
        )
        
        if conversation:
            await manager.send_personal_message({
                "type": "conversation_joined",
                "conversation_id": conversation_id,
                "conversation": {
                    "id": conversation.id,
                    "title": conversation.title,
                    "description": conversation.description,
                    "model": conversation.model,
                    "created_at": conversation.created_at.isoformat()
                }
            }, user.id, connection_id)
        else:
            await manager.send_personal_message({
                "type": "error",
                "message": "对话不存在或无权限访问"
            }, user.id, connection_id)


async def handle_leave_conversation(
    ws_message: WebSocketMessage,
    user: User,
    connection_id: str,
    db: Session
):
    """
    处理离开对话
    """
    conversation_id = ws_message.conversation_id
    if conversation_id:
        await manager.send_personal_message({
            "type": "conversation_left",
            "conversation_id": conversation_id
        }, user.id, connection_id)


async def handle_get_status(
    ws_message: WebSocketMessage,
    user: User,
    connection_id: str,
    db: Session
):
    """
    处理获取状态请求
    """
    # 获取用户配额信息
    quota = usage_service.get_user_quota(db, user.id)
    
    # 获取连接信息
    connections = manager.get_user_connections(user.id)
    
    await manager.send_personal_message({
        "type": "status",
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role.value
        },
        "quota": {
            "daily_messages_used": quota.daily_messages_used,
            "daily_messages_limit": quota.daily_messages_limit,
            "monthly_tokens_used": quota.monthly_tokens_used,
            "monthly_tokens_limit": quota.monthly_tokens_limit
        },
        "connections": connections,
        "timestamp": datetime.utcnow().isoformat()
    }, user.id, connection_id)


# 管理员专用WebSocket端点
@router.websocket("/ws/admin/{token}")
async def admin_websocket_endpoint(
    websocket: WebSocket,
    token: str
):
    """
    管理员WebSocket连接端点
    """
    user = None
    try:
        # 验证管理员身份
        user = await get_user_from_token(token)
        if not user or user.role.value != "admin":
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Admin access required")
            return
        
        await websocket.accept()
        logger.info(f"管理员WebSocket连接建立: {user.username} (ID: {user.id})")
        
        # 发送系统状态
        await websocket.send_text(json.dumps({
            "type": "admin_connected",
            "system_status": {
                "active_users": len(manager.get_active_users()),
                "total_connections": manager.get_connection_count(),
                "timestamp": datetime.utcnow().isoformat()
            }
        }, ensure_ascii=False))
        
        while True:
            data = await websocket.receive_text()
            # 处理管理员命令
            try:
                command = json.loads(data)
                if command.get("type") == "get_system_status":
                    await websocket.send_text(json.dumps({
                        "type": "system_status",
                        "active_users": manager.get_active_users(),
                        "total_connections": manager.get_connection_count(),
                        "timestamp": datetime.utcnow().isoformat()
                    }, ensure_ascii=False))
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e)
                }, ensure_ascii=False))
    
    except WebSocketDisconnect:
        if user:
            logger.info(f"管理员WebSocket连接断开: {user.username} (ID: {user.id})")
    except Exception as e:
        logger.error(f"管理员WebSocket连接错误: {e}")


# 获取WebSocket状态的HTTP端点
@router.get("/ws/status")
async def get_websocket_status():
    """
    获取WebSocket连接状态
    """
    return {
        "active_users": len(manager.get_active_users()),
        "total_connections": manager.get_connection_count(),
        "active_user_ids": manager.get_active_users(),
        "timestamp": datetime.utcnow().isoformat()
    }