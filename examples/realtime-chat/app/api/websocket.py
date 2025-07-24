#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket聊天功能

提供实时聊天、房间管理、消息广播等功能。
"""

import json
import asyncio
from typing import Dict, List, Set
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..models.chat import ChatRoom, Message, UserRoom
from ..services.chat_service import ChatService
from ..core.auth import get_current_user_from_token
from ..core.redis_manager import RedisManager


class ConnectionManager:
    """
    WebSocket连接管理器
    
    管理用户连接、房间订阅、消息广播等功能。
    """
    
    def __init__(self):
        # 活跃连接：{user_id: websocket}
        self.active_connections: Dict[int, WebSocket] = {}
        # 房间订阅：{room_id: {user_id, ...}}
        self.room_subscriptions: Dict[int, Set[int]] = {}
        # 用户房间：{user_id: {room_id, ...}}
        self.user_rooms: Dict[int, Set[int]] = {}
        # Redis管理器
        self.redis_manager = RedisManager()
    
    async def connect(self, websocket: WebSocket, user_id: int):
        """
        建立WebSocket连接
        """
        await websocket.accept()
        self.active_connections[user_id] = websocket
        
        # 设置用户在线状态
        await self.redis_manager.set_user_online(user_id)
        
        # 通知其他用户该用户上线
        await self.broadcast_user_status(user_id, "online")
        
        print(f"用户 {user_id} 已连接")
    
    def disconnect(self, user_id: int):
        """
        断开WebSocket连接
        """
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        
        # 从所有房间取消订阅
        if user_id in self.user_rooms:
            for room_id in self.user_rooms[user_id].copy():
                self.leave_room(user_id, room_id)
            del self.user_rooms[user_id]
        
        # 设置用户离线状态
        asyncio.create_task(self.redis_manager.set_user_offline(user_id))
        
        # 通知其他用户该用户下线
        asyncio.create_task(self.broadcast_user_status(user_id, "offline"))
        
        print(f"用户 {user_id} 已断开连接")
    
    def join_room(self, user_id: int, room_id: int):
        """
        加入聊天室
        """
        if room_id not in self.room_subscriptions:
            self.room_subscriptions[room_id] = set()
        
        self.room_subscriptions[room_id].add(user_id)
        
        if user_id not in self.user_rooms:
            self.user_rooms[user_id] = set()
        
        self.user_rooms[user_id].add(room_id)
        
        print(f"用户 {user_id} 加入房间 {room_id}")
    
    def leave_room(self, user_id: int, room_id: int):
        """
        离开聊天室
        """
        if room_id in self.room_subscriptions:
            self.room_subscriptions[room_id].discard(user_id)
            
            # 如果房间没有用户了，删除房间订阅
            if not self.room_subscriptions[room_id]:
                del self.room_subscriptions[room_id]
        
        if user_id in self.user_rooms:
            self.user_rooms[user_id].discard(room_id)
        
        print(f"用户 {user_id} 离开房间 {room_id}")
    
    async def send_personal_message(self, message: str, user_id: int):
        """
        发送个人消息
        """
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id]
            await websocket.send_text(message)
    
    async def broadcast_to_room(self, message: str, room_id: int, exclude_user: int = None):
        """
        向房间广播消息
        """
        if room_id in self.room_subscriptions:
            for user_id in self.room_subscriptions[room_id]:
                if exclude_user and user_id == exclude_user:
                    continue
                
                if user_id in self.active_connections:
                    websocket = self.active_connections[user_id]
                    try:
                        await websocket.send_text(message)
                    except Exception as e:
                        print(f"发送消息到用户 {user_id} 失败: {e}")
                        # 连接可能已断开，清理连接
                        self.disconnect(user_id)
    
    async def broadcast_user_status(self, user_id: int, status: str):
        """
        广播用户状态变化
        """
        message = {
            "type": "user_status",
            "user_id": user_id,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        # 向所有在线用户广播
        for connected_user_id, websocket in self.active_connections.items():
            if connected_user_id != user_id:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception:
                    pass
    
    async def get_online_users(self, room_id: int = None) -> List[int]:
        """
        获取在线用户列表
        """
        if room_id:
            # 获取房间内在线用户
            if room_id in self.room_subscriptions:
                return [user_id for user_id in self.room_subscriptions[room_id] 
                       if user_id in self.active_connections]
            return []
        else:
            # 获取所有在线用户
            return list(self.active_connections.keys())


# 全局连接管理器实例
manager = ConnectionManager()


async def websocket_endpoint(
    websocket: WebSocket,
    token: str,
    db: Session = Depends(get_db)
):
    """
    WebSocket端点
    
    处理WebSocket连接和消息。
    """
    # 验证用户身份
    try:
        user = await get_current_user_from_token(token, db)
        if not user:
            await websocket.close(code=4001, reason="未授权")
            return
    except Exception:
        await websocket.close(code=4001, reason="令牌无效")
        return
    
    # 建立连接
    await manager.connect(websocket, user.id)
    chat_service = ChatService(db)
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            message_type = message_data.get("type")
            
            if message_type == "join_room":
                # 加入房间
                room_id = message_data.get("room_id")
                if room_id:
                    # 检查用户是否有权限加入房间
                    if chat_service.can_join_room(user.id, room_id):
                        manager.join_room(user.id, room_id)
                        
                        # 通知房间内其他用户
                        join_message = {
                            "type": "user_joined",
                            "user_id": user.id,
                            "username": user.username,
                            "room_id": room_id,
                            "timestamp": datetime.now().isoformat()
                        }
                        await manager.broadcast_to_room(
                            json.dumps(join_message), 
                            room_id, 
                            exclude_user=user.id
                        )
                        
                        # 发送房间信息给用户
                        room_info = chat_service.get_room_info(room_id)
                        online_users = await manager.get_online_users(room_id)
                        
                        response = {
                            "type": "room_joined",
                            "room_info": room_info,
                            "online_users": online_users
                        }
                        await manager.send_personal_message(
                            json.dumps(response), user.id
                        )
                    else:
                        # 权限不足
                        error_message = {
                            "type": "error",
                            "message": "无权限加入该房间"
                        }
                        await manager.send_personal_message(
                            json.dumps(error_message), user.id
                        )
            
            elif message_type == "leave_room":
                # 离开房间
                room_id = message_data.get("room_id")
                if room_id:
                    manager.leave_room(user.id, room_id)
                    
                    # 通知房间内其他用户
                    leave_message = {
                        "type": "user_left",
                        "user_id": user.id,
                        "username": user.username,
                        "room_id": room_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    await manager.broadcast_to_room(
                        json.dumps(leave_message), 
                        room_id
                    )
            
            elif message_type == "send_message":
                # 发送消息
                room_id = message_data.get("room_id")
                content = message_data.get("content")
                message_type_sub = message_data.get("message_type", "text")
                
                if room_id and content:
                    # 检查用户是否在房间内
                    if (user.id in manager.user_rooms and 
                        room_id in manager.user_rooms[user.id]):
                        
                        # 保存消息到数据库
                        message = chat_service.create_message(
                            user_id=user.id,
                            room_id=room_id,
                            content=content,
                            message_type=message_type_sub
                        )
                        
                        # 广播消息
                        broadcast_message = {
                            "type": "new_message",
                            "message_id": message.id,
                            "user_id": user.id,
                            "username": user.username,
                            "room_id": room_id,
                            "content": content,
                            "message_type": message_type_sub,
                            "timestamp": message.created_at.isoformat()
                        }
                        
                        await manager.broadcast_to_room(
                            json.dumps(broadcast_message), 
                            room_id
                        )
                        
                        # 如果有离线用户，推送通知
                        offline_users = chat_service.get_offline_room_users(room_id)
                        if offline_users:
                            # 这里可以集成推送服务
                            pass
            
            elif message_type == "private_message":
                # 私聊消息
                target_user_id = message_data.get("target_user_id")
                content = message_data.get("content")
                
                if target_user_id and content:
                    # 保存私聊消息
                    message = chat_service.create_private_message(
                        sender_id=user.id,
                        receiver_id=target_user_id,
                        content=content
                    )
                    
                    # 发送给目标用户
                    private_message = {
                        "type": "private_message",
                        "message_id": message.id,
                        "sender_id": user.id,
                        "sender_username": user.username,
                        "content": content,
                        "timestamp": message.created_at.isoformat()
                    }
                    
                    await manager.send_personal_message(
                        json.dumps(private_message), 
                        target_user_id
                    )
                    
                    # 发送确认给发送者
                    confirmation = {
                        "type": "message_sent",
                        "message_id": message.id
                    }
                    await manager.send_personal_message(
                        json.dumps(confirmation), 
                        user.id
                    )
            
            elif message_type == "typing":
                # 输入状态
                room_id = message_data.get("room_id")
                is_typing = message_data.get("is_typing", False)
                
                if room_id:
                    typing_message = {
                        "type": "typing",
                        "user_id": user.id,
                        "username": user.username,
                        "room_id": room_id,
                        "is_typing": is_typing
                    }
                    
                    await manager.broadcast_to_room(
                        json.dumps(typing_message), 
                        room_id, 
                        exclude_user=user.id
                    )
            
            elif message_type == "get_online_users":
                # 获取在线用户
                room_id = message_data.get("room_id")
                online_users = await manager.get_online_users(room_id)
                
                response = {
                    "type": "online_users",
                    "room_id": room_id,
                    "users": online_users
                }
                
                await manager.send_personal_message(
                    json.dumps(response), 
                    user.id
                )
    
    except WebSocketDisconnect:
        manager.disconnect(user.id)
    except Exception as e:
        print(f"WebSocket错误: {e}")
        manager.disconnect(user.id)


async def broadcast_system_message(message: str, room_id: int = None):
    """
    广播系统消息
    
    用于系统通知、维护公告等。
    """
    system_message = {
        "type": "system_message",
        "content": message,
        "timestamp": datetime.now().isoformat()
    }
    
    if room_id:
        await manager.broadcast_to_room(json.dumps(system_message), room_id)
    else:
        # 广播给所有在线用户
        for user_id, websocket in manager.active_connections.items():
            try:
                await websocket.send_text(json.dumps(system_message))
            except Exception:
                pass


async def kick_user_from_room(user_id: int, room_id: int, reason: str = "被管理员踢出"):
    """
    将用户踢出房间
    
    管理员功能。
    """
    if user_id in manager.active_connections:
        # 发送踢出通知
        kick_message = {
            "type": "kicked",
            "room_id": room_id,
            "reason": reason
        }
        
        await manager.send_personal_message(
            json.dumps(kick_message), 
            user_id
        )
        
        # 从房间移除
        manager.leave_room(user_id, room_id)
        
        # 通知房间内其他用户
        leave_message = {
            "type": "user_kicked",
            "user_id": user_id,
            "room_id": room_id,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        await manager.broadcast_to_room(
            json.dumps(leave_message), 
            room_id
        )