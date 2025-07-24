# -*- coding: utf-8 -*-
"""
对话服务

提供对话管理相关的业务逻辑。
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
import uuid

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func, desc

# 导入模板库组件
from src.utils.exceptions import ValidationException, AuthorizationException
from src.core.logging import get_logger

# 导入应用组件
from app.models.conversation import (
    Conversation, ConversationCreate, ConversationUpdate,
    ConversationStatus, ConversationStats, ConversationShare
)
from app.models.message import Message, MessageRole
from app.models.user import User

logger = get_logger(__name__)


class ConversationService:
    """对话服务类"""
    
    def create_conversation(self, db: Session, user_id: int, conversation_data: ConversationCreate) -> Conversation:
        """
        创建新对话
        """
        try:
            # 验证用户存在
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValidationException("用户不存在")
            
            # 创建对话
            db_conversation = Conversation(
                title=conversation_data.title,
                description=conversation_data.description,
                user_id=user_id,
                model=conversation_data.model or "gpt-3.5-turbo",
                system_prompt=conversation_data.system_prompt,
                temperature=conversation_data.temperature or 0.7,
                max_tokens=conversation_data.max_tokens,
                status=ConversationStatus.active,
                is_pinned=False,
                is_shared=False,
                message_count=0,
                total_tokens=0,
                metadata=conversation_data.metadata or {},
                tags=conversation_data.tags or [],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            db.add(db_conversation)
            db.commit()
            db.refresh(db_conversation)
            
            logger.info(f"创建对话: {db_conversation.title} (ID: {db_conversation.id}, User: {user_id})")
            
            return db_conversation
            
        except ValidationException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"创建对话失败: {e}")
            raise ValidationException("创建对话失败")
    
    def get_conversation(self, db: Session, conversation_id: int, user_id: int, include_messages: bool = False) -> Optional[Conversation]:
        """
        获取对话详情
        """
        query = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None)
        )
        
        if include_messages:
            query = query.options(joinedload(Conversation.messages))
        
        return query.first()
    
    def get_user_conversations(self, db: Session, user_id: int, page: int = 1, size: int = 20, filters: Optional[dict] = None) -> Tuple[List[Conversation], int]:
        """
        获取用户对话列表
        """
        query = db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None)
        )
        
        # 应用过滤器
        if filters:
            if filters.get("status"):
                query = query.filter(Conversation.status == filters["status"])
            
            if filters.get("model"):
                query = query.filter(Conversation.model == filters["model"])
            
            if filters.get("is_pinned") is not None:
                query = query.filter(Conversation.is_pinned == filters["is_pinned"])
            
            if filters.get("tags"):
                # 搜索包含任一标签的对话
                for tag in filters["tags"]:
                    query = query.filter(Conversation.tags.contains([tag]))
            
            if filters.get("search"):
                search_term = f"%{filters['search']}%"
                query = query.filter(
                    or_(
                        Conversation.title.ilike(search_term),
                        Conversation.description.ilike(search_term)
                    )
                )
            
            if filters.get("date_from"):
                query = query.filter(Conversation.created_at >= filters["date_from"])
            
            if filters.get("date_to"):
                query = query.filter(Conversation.created_at <= filters["date_to"])
        
        # 获取总数
        total = query.count()
        
        # 排序：置顶的在前，然后按更新时间倒序
        conversations = query.order_by(
            desc(Conversation.is_pinned),
            desc(Conversation.updated_at)
        ).offset((page - 1) * size).limit(size).all()
        
        return conversations, total
    
    def update_conversation(self, db: Session, conversation_id: int, user_id: int, conversation_data: ConversationUpdate) -> Optional[Conversation]:
        """
        更新对话信息
        """
        try:
            conversation = self.get_conversation(db, conversation_id, user_id)
            if not conversation:
                return None
            
            # 更新字段
            update_data = conversation_data.dict(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(conversation, field):
                    setattr(conversation, field, value)
            
            conversation.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(conversation)
            
            return conversation
            
        except Exception as e:
            db.rollback()
            logger.error(f"更新对话失败: {e}")
            raise ValidationException("更新对话失败")
    
    def delete_conversation(self, db: Session, conversation_id: int, user_id: int, permanent: bool = False) -> bool:
        """
        删除对话
        """
        try:
            conversation = self.get_conversation(db, conversation_id, user_id)
            if not conversation:
                return False
            
            if permanent:
                # 永久删除
                db.delete(conversation)
            else:
                # 软删除
                conversation.deleted_at = datetime.utcnow()
                conversation.status = ConversationStatus.deleted
            
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"删除对话失败: {e}")
            return False
    
    def pin_conversation(self, db: Session, conversation_id: int, user_id: int) -> bool:
        """
        置顶对话
        """
        try:
            conversation = self.get_conversation(db, conversation_id, user_id)
            if not conversation:
                return False
            
            conversation.is_pinned = True
            conversation.updated_at = datetime.utcnow()
            
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"置顶对话失败: {e}")
            return False
    
    def unpin_conversation(self, db: Session, conversation_id: int, user_id: int) -> bool:
        """
        取消置顶对话
        """
        try:
            conversation = self.get_conversation(db, conversation_id, user_id)
            if not conversation:
                return False
            
            conversation.is_pinned = False
            conversation.updated_at = datetime.utcnow()
            
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"取消置顶失败: {e}")
            return False
    
    def archive_conversation(self, db: Session, conversation_id: int, user_id: int) -> bool:
        """
        归档对话
        """
        try:
            conversation = self.get_conversation(db, conversation_id, user_id)
            if not conversation:
                return False
            
            conversation.status = ConversationStatus.archived
            conversation.updated_at = datetime.utcnow()
            
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"归档对话失败: {e}")
            return False
    
    def restore_conversation(self, db: Session, conversation_id: int, user_id: int) -> bool:
        """
        恢复对话
        """
        try:
            # 查询包括已删除的对话
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            ).first()
            
            if not conversation:
                return False
            
            conversation.status = ConversationStatus.active
            conversation.deleted_at = None
            conversation.updated_at = datetime.utcnow()
            
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"恢复对话失败: {e}")
            return False
    
    def get_conversation_share(self, db: Session, conversation_id: int, user_id: int) -> Optional[ConversationShare]:
        """
        获取对话分享信息
        """
        conversation = self.get_conversation(db, conversation_id, user_id)
        if not conversation:
            return None
        
        if not conversation.is_shared:
            return None
        
        return ConversationShare(
            conversation_id=conversation.id,
            title=conversation.title,
            description=conversation.description,
            share_url=f"/shared/conversations/{conversation.share_token}",
            share_token=conversation.share_token,
            is_public=conversation.is_shared,
            created_at=conversation.created_at,
            shared_at=conversation.updated_at
        )
    
    def create_conversation_share(self, db: Session, conversation_id: int, user_id: int) -> Optional[ConversationShare]:
        """
        创建对话分享
        """
        try:
            conversation = self.get_conversation(db, conversation_id, user_id)
            if not conversation:
                return None
            
            # 生成分享令牌
            if not conversation.share_token:
                conversation.share_token = str(uuid.uuid4())
            
            conversation.is_shared = True
            conversation.updated_at = datetime.utcnow()
            
            db.commit()
            
            return ConversationShare(
                conversation_id=conversation.id,
                title=conversation.title,
                description=conversation.description,
                share_url=f"/shared/conversations/{conversation.share_token}",
                share_token=conversation.share_token,
                is_public=True,
                created_at=conversation.created_at,
                shared_at=conversation.updated_at
            )
            
        except Exception as e:
            db.rollback()
            logger.error(f"创建分享失败: {e}")
            return None
    
    def remove_conversation_share(self, db: Session, conversation_id: int, user_id: int) -> bool:
        """
        取消对话分享
        """
        try:
            conversation = self.get_conversation(db, conversation_id, user_id)
            if not conversation:
                return False
            
            conversation.is_shared = False
            conversation.updated_at = datetime.utcnow()
            
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"取消分享失败: {e}")
            return False
    
    def get_shared_conversation(self, db: Session, share_token: str) -> Optional[Conversation]:
        """
        通过分享令牌获取对话
        """
        return db.query(Conversation).filter(
            Conversation.share_token == share_token,
            Conversation.is_shared == True,
            Conversation.deleted_at.is_(None)
        ).options(joinedload(Conversation.messages)).first()
    
    def get_user_conversation_stats(self, db: Session, user_id: int) -> ConversationStats:
        """
        获取用户对话统计
        """
        # 总对话数
        total_conversations = db.query(func.count(Conversation.id)).filter(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None)
        ).scalar() or 0
        
        # 活跃对话数
        active_conversations = db.query(func.count(Conversation.id)).filter(
            Conversation.user_id == user_id,
            Conversation.status == ConversationStatus.active,
            Conversation.deleted_at.is_(None)
        ).scalar() or 0
        
        # 归档对话数
        archived_conversations = db.query(func.count(Conversation.id)).filter(
            Conversation.user_id == user_id,
            Conversation.status == ConversationStatus.archived,
            Conversation.deleted_at.is_(None)
        ).scalar() or 0
        
        # 置顶对话数
        pinned_conversations = db.query(func.count(Conversation.id)).filter(
            Conversation.user_id == user_id,
            Conversation.is_pinned == True,
            Conversation.deleted_at.is_(None)
        ).scalar() or 0
        
        # 分享对话数
        shared_conversations = db.query(func.count(Conversation.id)).filter(
            Conversation.user_id == user_id,
            Conversation.is_shared == True,
            Conversation.deleted_at.is_(None)
        ).scalar() or 0
        
        # 总消息数
        total_messages = db.query(func.sum(Conversation.message_count)).filter(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None)
        ).scalar() or 0
        
        # 总token数
        total_tokens = db.query(func.sum(Conversation.total_tokens)).filter(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None)
        ).scalar() or 0
        
        # 最近活跃对话
        recent_conversation = db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None)
        ).order_by(desc(Conversation.updated_at)).first()
        
        return ConversationStats(
            total_conversations=total_conversations,
            active_conversations=active_conversations,
            archived_conversations=archived_conversations,
            pinned_conversations=pinned_conversations,
            shared_conversations=shared_conversations,
            total_messages=total_messages,
            total_tokens=total_tokens,
            last_conversation_at=recent_conversation.updated_at if recent_conversation else None
        )
    
    def get_conversation_summaries(self, db: Session, user_id: int, limit: int = 50) -> List[Conversation]:
        """
        获取对话摘要列表
        """
        return db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None)
        ).order_by(
            desc(Conversation.is_pinned),
            desc(Conversation.updated_at)
        ).limit(limit).all()
    
    def update_conversation_stats(self, db: Session, conversation_id: int):
        """
        更新对话统计信息
        """
        try:
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            
            if not conversation:
                return
            
            # 统计消息数量
            message_count = db.query(func.count(Message.id)).filter(
                Message.conversation_id == conversation_id,
                Message.deleted_at.is_(None)
            ).scalar() or 0
            
            # 统计token数量
            total_tokens = db.query(func.sum(Message.total_tokens)).filter(
                Message.conversation_id == conversation_id,
                Message.deleted_at.is_(None)
            ).scalar() or 0
            
            # 更新统计信息
            conversation.message_count = message_count
            conversation.total_tokens = total_tokens
            conversation.updated_at = datetime.utcnow()
            
            db.commit()
            
        except Exception as e:
            db.rollback()
            logger.error(f"更新对话统计失败: {e}")
    
    def search_conversations(self, db: Session, user_id: int, query: str, page: int = 1, size: int = 20) -> Tuple[List[Conversation], int]:
        """
        搜索对话
        """
        search_term = f"%{query}%"
        
        db_query = db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None),
            or_(
                Conversation.title.ilike(search_term),
                Conversation.description.ilike(search_term)
            )
        )
        
        # 获取总数
        total = db_query.count()
        
        # 分页
        conversations = db_query.order_by(
            desc(Conversation.updated_at)
        ).offset((page - 1) * size).limit(size).all()
        
        return conversations, total
    
    def get_conversations_by_model(self, db: Session, user_id: int, model: str) -> List[Conversation]:
        """
        根据模型获取对话列表
        """
        return db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.model == model,
            Conversation.deleted_at.is_(None)
        ).order_by(desc(Conversation.updated_at)).all()
    
    def get_conversations_by_tags(self, db: Session, user_id: int, tags: List[str]) -> List[Conversation]:
        """
        根据标签获取对话列表
        """
        query = db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None)
        )
        
        # 搜索包含任一标签的对话
        for tag in tags:
            query = query.filter(Conversation.tags.contains([tag]))
        
        return query.order_by(desc(Conversation.updated_at)).all()