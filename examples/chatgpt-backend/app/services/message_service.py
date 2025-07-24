# -*- coding: utf-8 -*-
"""
消息服务

提供消息管理相关的业务逻辑。
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

# 导入模板库组件
from src.utils.exceptions import ValidationException, AuthorizationException
from src.core.logging import get_logger

# 导入应用组件
from app.models.message import (
    Message, MessageCreate, MessageUpdate, MessageRole, 
    MessageStatus, MessageStats
)
from app.models.conversation import Conversation, ConversationCreate
from app.models.user import User
from app.services.conversation_service import ConversationService

logger = get_logger(__name__)


class MessageService:
    """消息服务类"""
    
    def __init__(self):
        self.conversation_service = ConversationService()
    
    def create_message(self, db: Session, conversation_id: Optional[int], user_id: int, message_data: MessageCreate) -> Message:
        """
        创建新消息
        """
        try:
            # 如果没有提供对话ID，创建新对话
            if not conversation_id:
                conversation = self.conversation_service.create_conversation(
                    db=db,
                    user_id=user_id,
                    conversation_data=ConversationCreate(
                        title="新对话",
                        model=message_data.model or "gpt-3.5-turbo"
                    )
                )
                conversation_id = conversation.id
            else:
                # 验证对话存在且属于用户
                conversation = self.conversation_service.get_conversation(
                    db=db,
                    conversation_id=conversation_id,
                    user_id=user_id
                )
                if not conversation:
                    raise ValidationException("对话不存在或无权限访问")
            
            # 创建消息
            db_message = Message(
                content=message_data.content,
                role=message_data.role,
                conversation_id=conversation_id,
                user_id=user_id,
                status=MessageStatus.sent,
                model=message_data.model,
                prompt_tokens=message_data.prompt_tokens or 0,
                completion_tokens=message_data.completion_tokens or 0,
                total_tokens=message_data.total_tokens or 0,
                finish_reason=message_data.finish_reason,
                metadata=message_data.metadata or {},
                attachments=message_data.attachments or [],
                is_edited=False,
                is_deleted=False,
                is_pinned=False,
                parent_message_id=message_data.parent_message_id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            db.add(db_message)
            db.commit()
            db.refresh(db_message)
            
            # 更新对话统计
            self.conversation_service.update_conversation_stats(db, conversation_id)
            
            logger.info(f"创建消息: {db_message.role.value} (ID: {db_message.id}, Conversation: {conversation_id})")
            
            return db_message
            
        except ValidationException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"创建消息失败: {e}")
            raise ValidationException("创建消息失败")
    
    def get_message(self, db: Session, message_id: int, user_id: int) -> Optional[Message]:
        """
        获取消息详情
        """
        return db.query(Message).filter(
            Message.id == message_id,
            Message.user_id == user_id,
            Message.deleted_at.is_(None)
        ).first()
    
    def get_conversation_messages(self, db: Session, conversation_id: int, page: int = 1, size: int = 50, filters: Optional[dict] = None) -> Tuple[List[Message], int]:
        """
        获取对话消息列表
        """
        query = db.query(Message).filter(
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None)
        )
        
        # 应用过滤器
        if filters:
            if filters.get("role"):
                query = query.filter(Message.role == filters["role"])
            
            if filters.get("status"):
                query = query.filter(Message.status == filters["status"])
            
            if filters.get("search"):
                search_term = f"%{filters['search']}%"
                query = query.filter(Message.content.ilike(search_term))
            
            if filters.get("date_from"):
                query = query.filter(Message.created_at >= filters["date_from"])
            
            if filters.get("date_to"):
                query = query.filter(Message.created_at <= filters["date_to"])
            
            if filters.get("has_attachments") is not None:
                if filters["has_attachments"]:
                    query = query.filter(func.json_array_length(Message.attachments) > 0)
                else:
                    query = query.filter(func.json_array_length(Message.attachments) == 0)
        
        # 获取总数
        total = query.count()
        
        # 分页，按创建时间正序排列
        messages = query.order_by(Message.created_at.asc()).offset((page - 1) * size).limit(size).all()
        
        return messages, total
    
    def get_conversation_history(self, db: Session, conversation_id: int, limit: int = 20) -> List[Message]:
        """
        获取对话历史（用于AI上下文）
        """
        return db.query(Message).filter(
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None)
        ).order_by(Message.created_at.asc()).limit(limit).all()
    
    def update_message(self, db: Session, message_id: int, user_id: int, message_data: MessageUpdate) -> Optional[Message]:
        """
        更新消息
        """
        try:
            message = self.get_message(db, message_id, user_id)
            if not message:
                return None
            
            # 只允许用户更新自己的消息
            if message.role != MessageRole.user:
                raise ValidationException("只能编辑用户消息")
            
            # 更新字段
            update_data = message_data.dict(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(message, field):
                    setattr(message, field, value)
            
            # 标记为已编辑
            if message_data.content and message_data.content != message.content:
                message.is_edited = True
            
            message.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(message)
            
            return message
            
        except ValidationException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"更新消息失败: {e}")
            raise ValidationException("更新消息失败")
    
    def delete_message(self, db: Session, message_id: int, user_id: int) -> bool:
        """
        删除消息
        """
        try:
            message = self.get_message(db, message_id, user_id)
            if not message:
                return False
            
            # 软删除
            message.deleted_at = datetime.utcnow()
            message.is_deleted = True
            message.updated_at = datetime.utcnow()
            
            db.commit()
            
            # 更新对话统计
            self.conversation_service.update_conversation_stats(db, message.conversation_id)
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"删除消息失败: {e}")
            return False
    
    def regenerate_message(self, db: Session, message_id: int, user_id: int) -> Optional[Message]:
        """
        重新生成AI消息
        """
        try:
            # 获取原消息
            original_message = self.get_message(db, message_id, user_id)
            if not original_message:
                return None
            
            # 只能重新生成AI消息
            if original_message.role != MessageRole.assistant:
                raise ValidationException("只能重新生成AI消息")
            
            # 标记原消息为已删除
            original_message.deleted_at = datetime.utcnow()
            original_message.is_deleted = True
            
            # 这里应该调用OpenAI API重新生成，暂时返回一个占位消息
            new_message = Message(
                content="[重新生成的消息]",
                role=MessageRole.assistant,
                conversation_id=original_message.conversation_id,
                user_id=user_id,
                status=MessageStatus.sent,
                model=original_message.model,
                parent_message_id=original_message.parent_message_id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            db.add(new_message)
            db.commit()
            db.refresh(new_message)
            
            return new_message
            
        except ValidationException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"重新生成消息失败: {e}")
            return None
    
    def pin_message(self, db: Session, message_id: int, user_id: int) -> bool:
        """
        置顶消息
        """
        try:
            message = self.get_message(db, message_id, user_id)
            if not message:
                return False
            
            message.is_pinned = True
            message.updated_at = datetime.utcnow()
            
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"置顶消息失败: {e}")
            return False
    
    def unpin_message(self, db: Session, message_id: int, user_id: int) -> bool:
        """
        取消置顶消息
        """
        try:
            message = self.get_message(db, message_id, user_id)
            if not message:
                return False
            
            message.is_pinned = False
            message.updated_at = datetime.utcnow()
            
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"取消置顶失败: {e}")
            return False
    
    def get_message_stats(self, db: Session, user_id: int, conversation_id: Optional[int] = None) -> MessageStats:
        """
        获取消息统计
        """
        base_query = db.query(Message).filter(
            Message.user_id == user_id,
            Message.deleted_at.is_(None)
        )
        
        if conversation_id:
            base_query = base_query.filter(Message.conversation_id == conversation_id)
        
        # 总消息数
        total_messages = base_query.count()
        
        # 用户消息数
        user_messages = base_query.filter(Message.role == MessageRole.user).count()
        
        # AI消息数
        assistant_messages = base_query.filter(Message.role == MessageRole.assistant).count()
        
        # 系统消息数
        system_messages = base_query.filter(Message.role == MessageRole.system).count()
        
        # 总token数
        total_tokens = base_query.with_entities(func.sum(Message.total_tokens)).scalar() or 0
        
        # 平均消息长度
        avg_message_length = base_query.with_entities(func.avg(func.length(Message.content))).scalar() or 0
        
        # 置顶消息数
        pinned_messages = base_query.filter(Message.is_pinned == True).count()
        
        # 编辑过的消息数
        edited_messages = base_query.filter(Message.is_edited == True).count()
        
        # 最近消息时间
        latest_message = base_query.order_by(desc(Message.created_at)).first()
        
        return MessageStats(
            total_messages=total_messages,
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            system_messages=system_messages,
            total_tokens=total_tokens,
            average_message_length=round(avg_message_length, 2),
            pinned_messages=pinned_messages,
            edited_messages=edited_messages,
            latest_message_at=latest_message.created_at if latest_message else None
        )
    
    def search_messages(self, db: Session, user_id: int, query: str, page: int = 1, size: int = 20, conversation_id: Optional[int] = None, role: Optional[MessageRole] = None) -> Tuple[List[Message], int]:
        """
        搜索消息
        """
        search_term = f"%{query}%"
        
        db_query = db.query(Message).filter(
            Message.user_id == user_id,
            Message.deleted_at.is_(None),
            Message.content.ilike(search_term)
        )
        
        if conversation_id:
            db_query = db_query.filter(Message.conversation_id == conversation_id)
        
        if role:
            db_query = db_query.filter(Message.role == role)
        
        # 获取总数
        total = db_query.count()
        
        # 分页
        messages = db_query.order_by(
            desc(Message.created_at)
        ).offset((page - 1) * size).limit(size).all()
        
        return messages, total
    
    def get_messages_by_role(self, db: Session, user_id: int, role: MessageRole, conversation_id: Optional[int] = None) -> List[Message]:
        """
        根据角色获取消息
        """
        query = db.query(Message).filter(
            Message.user_id == user_id,
            Message.role == role,
            Message.deleted_at.is_(None)
        )
        
        if conversation_id:
            query = query.filter(Message.conversation_id == conversation_id)
        
        return query.order_by(Message.created_at.desc()).all()
    
    def get_pinned_messages(self, db: Session, user_id: int, conversation_id: Optional[int] = None) -> List[Message]:
        """
        获取置顶消息
        """
        query = db.query(Message).filter(
            Message.user_id == user_id,
            Message.is_pinned == True,
            Message.deleted_at.is_(None)
        )
        
        if conversation_id:
            query = query.filter(Message.conversation_id == conversation_id)
        
        return query.order_by(Message.created_at.desc()).all()
    
    def get_messages_with_attachments(self, db: Session, user_id: int, conversation_id: Optional[int] = None) -> List[Message]:
        """
        获取包含附件的消息
        """
        query = db.query(Message).filter(
            Message.user_id == user_id,
            func.json_array_length(Message.attachments) > 0,
            Message.deleted_at.is_(None)
        )
        
        if conversation_id:
            query = query.filter(Message.conversation_id == conversation_id)
        
        return query.order_by(Message.created_at.desc()).all()
    
    def export_conversation_messages(self, db: Session, conversation_id: int, user_id: int, format: str = "json") -> dict:
        """
        导出对话消息
        """
        # 验证对话权限
        conversation = self.conversation_service.get_conversation(db, conversation_id, user_id)
        if not conversation:
            raise ValidationException("对话不存在或无权限访问")
        
        # 获取所有消息
        messages = db.query(Message).filter(
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None)
        ).order_by(Message.created_at.asc()).all()
        
        export_data = {
            "conversation": {
                "id": conversation.id,
                "title": conversation.title,
                "description": conversation.description,
                "model": conversation.model,
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat()
            },
            "messages": [
                {
                    "id": msg.id,
                    "content": msg.content,
                    "role": msg.role.value,
                    "model": msg.model,
                    "tokens": {
                        "prompt": msg.prompt_tokens,
                        "completion": msg.completion_tokens,
                        "total": msg.total_tokens
                    },
                    "created_at": msg.created_at.isoformat(),
                    "is_edited": msg.is_edited,
                    "is_pinned": msg.is_pinned
                }
                for msg in messages
            ],
            "export_info": {
                "exported_at": datetime.utcnow().isoformat(),
                "format": format,
                "total_messages": len(messages)
            }
        }
        
        return export_data
    
    def get_message_thread(self, db: Session, message_id: int, user_id: int) -> List[Message]:
        """
        获取消息线程（包括父消息和子消息）
        """
        message = self.get_message(db, message_id, user_id)
        if not message:
            return []
        
        thread_messages = []
        
        # 获取父消息链
        current_message = message
        while current_message and current_message.parent_message_id:
            parent = self.get_message(db, current_message.parent_message_id, user_id)
            if parent:
                thread_messages.insert(0, parent)
                current_message = parent
            else:
                break
        
        # 添加当前消息
        thread_messages.append(message)
        
        # 获取子消息
        child_messages = db.query(Message).filter(
            Message.parent_message_id == message_id,
            Message.user_id == user_id,
            Message.deleted_at.is_(None)
        ).order_by(Message.created_at.asc()).all()
        
        thread_messages.extend(child_messages)
        
        return thread_messages
    
    def bulk_delete_messages(self, db: Session, message_ids: List[int], user_id: int) -> int:
        """
        批量删除消息
        """
        try:
            deleted_count = 0
            
            for message_id in message_ids:
                if self.delete_message(db, message_id, user_id):
                    deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"批量删除消息失败: {e}")
            return 0