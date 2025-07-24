# -*- coding: utf-8 -*-
"""
用户服务

提供用户管理相关的业务逻辑。
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, timedelta

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

# 导入模板库组件
from src.utils.security import verify_password, get_password_hash
from src.utils.exceptions import ValidationException, AuthorizationException
from src.core.logging import get_logger

# 导入应用组件
from app.models.user import (
    User, UserCreate, UserUpdate, UserPasswordUpdate, 
    UserProfile, UserStats, UserRole, UserStatus
)
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.usage import Usage, UserQuota

logger = get_logger(__name__)


class UserService:
    """用户服务类"""
    
    def create_user(self, db: Session, user_data: UserCreate) -> User:
        """
        创建新用户
        """
        try:
            # 检查用户名是否已存在
            existing_user = db.query(User).filter(
                or_(
                    User.username == user_data.username,
                    User.email == user_data.email
                )
            ).first()
            
            if existing_user:
                if existing_user.username == user_data.username:
                    raise ValidationException("用户名已存在")
                if existing_user.email == user_data.email:
                    raise ValidationException("邮箱已存在")
            
            # 创建用户
            db_user = User(
                username=user_data.username,
                email=user_data.email,
                full_name=user_data.full_name,
                hashed_password=get_password_hash(user_data.password),
                role=user_data.role or UserRole.user,
                status=UserStatus.active,
                avatar_url=user_data.avatar_url,
                bio=user_data.bio,
                preferences=user_data.preferences or {},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            
            # 创建用户配额
            self._create_user_quota(db, db_user.id)
            
            logger.info(f"创建用户: {db_user.username} (ID: {db_user.id})")
            
            return db_user
            
        except ValidationException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"创建用户失败: {e}")
            raise ValidationException("创建用户失败")
    
    def get_user_by_id(self, db: Session, user_id: int) -> Optional[User]:
        """
        根据ID获取用户
        """
        return db.query(User).filter(
            User.id == user_id,
            User.deleted_at.is_(None)
        ).first()
    
    def get_user_by_username(self, db: Session, username: str) -> Optional[User]:
        """
        根据用户名获取用户
        """
        return db.query(User).filter(
            User.username == username,
            User.deleted_at.is_(None)
        ).first()
    
    def get_user_by_email(self, db: Session, email: str) -> Optional[User]:
        """
        根据邮箱获取用户
        """
        return db.query(User).filter(
            User.email == email,
            User.deleted_at.is_(None)
        ).first()
    
    def authenticate_user(self, db: Session, username: str, password: str) -> Optional[User]:
        """
        用户认证
        """
        user = self.get_user_by_username(db, username)
        if not user:
            user = self.get_user_by_email(db, username)
        
        if not user or not verify_password(password, user.hashed_password):
            return None
        
        if user.status != UserStatus.active:
            raise AuthorizationException("账户已被禁用")
        
        # 更新最后登录时间
        user.last_login_at = datetime.utcnow()
        db.commit()
        
        return user
    
    def update_user(self, db: Session, user_id: int, user_data: UserUpdate) -> Optional[User]:
        """
        更新用户信息
        """
        try:
            user = self.get_user_by_id(db, user_id)
            if not user:
                return None
            
            # 检查用户名和邮箱唯一性
            if user_data.username and user_data.username != user.username:
                existing_user = db.query(User).filter(
                    User.username == user_data.username,
                    User.id != user_id,
                    User.deleted_at.is_(None)
                ).first()
                if existing_user:
                    raise ValidationException("用户名已存在")
            
            if user_data.email and user_data.email != user.email:
                existing_user = db.query(User).filter(
                    User.email == user_data.email,
                    User.id != user_id,
                    User.deleted_at.is_(None)
                ).first()
                if existing_user:
                    raise ValidationException("邮箱已存在")
            
            # 更新字段
            update_data = user_data.dict(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(user, field):
                    setattr(user, field, value)
            
            user.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(user)
            
            return user
            
        except ValidationException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"更新用户失败: {e}")
            raise ValidationException("更新用户失败")
    
    def change_password(self, db: Session, user_id: int, password_data: UserPasswordUpdate) -> bool:
        """
        修改用户密码
        """
        try:
            user = self.get_user_by_id(db, user_id)
            if not user:
                return False
            
            # 验证当前密码
            if not verify_password(password_data.current_password, user.hashed_password):
                raise ValidationException("当前密码错误")
            
            # 验证新密码确认
            if password_data.new_password != password_data.confirm_password:
                raise ValidationException("新密码确认不匹配")
            
            # 更新密码
            user.hashed_password = get_password_hash(password_data.new_password)
            user.updated_at = datetime.utcnow()
            
            db.commit()
            
            return True
            
        except ValidationException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"修改密码失败: {e}")
            return False
    
    def get_users(self, db: Session, page: int = 1, size: int = 20, filters: Optional[dict] = None) -> Tuple[List[User], int]:
        """
        获取用户列表（分页）
        """
        query = db.query(User).filter(User.deleted_at.is_(None))
        
        # 应用过滤器
        if filters:
            if filters.get("role"):
                query = query.filter(User.role == filters["role"])
            
            if filters.get("status"):
                query = query.filter(User.status == filters["status"])
            
            if filters.get("search"):
                search_term = f"%{filters['search']}%"
                query = query.filter(
                    or_(
                        User.username.ilike(search_term),
                        User.email.ilike(search_term),
                        User.full_name.ilike(search_term)
                    )
                )
            
            if filters.get("date_from"):
                query = query.filter(User.created_at >= filters["date_from"])
            
            if filters.get("date_to"):
                query = query.filter(User.created_at <= filters["date_to"])
        
        # 获取总数
        total = query.count()
        
        # 分页
        users = query.order_by(User.created_at.desc()).offset((page - 1) * size).limit(size).all()
        
        return users, total
    
    def get_user_profile(self, db: Session, user_id: int) -> Optional[UserProfile]:
        """
        获取用户档案
        """
        user = self.get_user_by_id(db, user_id)
        if not user:
            return None
        
        # 获取统计信息
        conversation_count = db.query(func.count(Conversation.id)).filter(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None)
        ).scalar() or 0
        
        message_count = db.query(func.count(Message.id)).filter(
            Message.user_id == user_id,
            Message.deleted_at.is_(None)
        ).scalar() or 0
        
        total_tokens = db.query(func.sum(Usage.total_tokens)).filter(
            Usage.user_id == user_id
        ).scalar() or 0
        
        return UserProfile(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            avatar_url=user.avatar_url,
            bio=user.bio,
            role=user.role,
            status=user.status,
            preferences=user.preferences,
            conversation_count=conversation_count,
            message_count=message_count,
            total_tokens_used=total_tokens,
            created_at=user.created_at,
            last_login_at=user.last_login_at
        )
    
    def get_user_stats(self, db: Session, user_id: int) -> Optional[UserStats]:
        """
        获取用户统计信息
        """
        user = self.get_user_by_id(db, user_id)
        if not user:
            return None
        
        # 今天的统计
        today = datetime.utcnow().date()
        today_start = datetime.combine(today, datetime.min.time())
        today_end = datetime.combine(today, datetime.max.time())
        
        # 本月的统计
        month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # 对话统计
        total_conversations = db.query(func.count(Conversation.id)).filter(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None)
        ).scalar() or 0
        
        today_conversations = db.query(func.count(Conversation.id)).filter(
            Conversation.user_id == user_id,
            Conversation.created_at >= today_start,
            Conversation.created_at <= today_end,
            Conversation.deleted_at.is_(None)
        ).scalar() or 0
        
        # 消息统计
        total_messages = db.query(func.count(Message.id)).filter(
            Message.user_id == user_id,
            Message.deleted_at.is_(None)
        ).scalar() or 0
        
        today_messages = db.query(func.count(Message.id)).filter(
            Message.user_id == user_id,
            Message.created_at >= today_start,
            Message.created_at <= today_end,
            Message.deleted_at.is_(None)
        ).scalar() or 0
        
        # Token统计
        total_tokens = db.query(func.sum(Usage.total_tokens)).filter(
            Usage.user_id == user_id
        ).scalar() or 0
        
        month_tokens = db.query(func.sum(Usage.total_tokens)).filter(
            Usage.user_id == user_id,
            Usage.created_at >= month_start
        ).scalar() or 0
        
        # 费用统计
        total_cost = db.query(func.sum(Usage.cost)).filter(
            Usage.user_id == user_id
        ).scalar() or 0.0
        
        month_cost = db.query(func.sum(Usage.cost)).filter(
            Usage.user_id == user_id,
            Usage.created_at >= month_start
        ).scalar() or 0.0
        
        return UserStats(
            total_conversations=total_conversations,
            today_conversations=today_conversations,
            total_messages=total_messages,
            today_messages=today_messages,
            total_tokens_used=total_tokens,
            month_tokens_used=month_tokens,
            total_cost=total_cost,
            month_cost=month_cost,
            member_since=user.created_at,
            last_active=user.last_login_at
        )
    
    def activate_user(self, db: Session, user_id: int) -> bool:
        """
        激活用户
        """
        try:
            user = self.get_user_by_id(db, user_id)
            if not user:
                return False
            
            user.status = UserStatus.active
            user.updated_at = datetime.utcnow()
            
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"激活用户失败: {e}")
            return False
    
    def deactivate_user(self, db: Session, user_id: int) -> bool:
        """
        停用用户
        """
        try:
            user = self.get_user_by_id(db, user_id)
            if not user:
                return False
            
            user.status = UserStatus.inactive
            user.updated_at = datetime.utcnow()
            
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"停用用户失败: {e}")
            return False
    
    def delete_user(self, db: Session, user_id: int, permanent: bool = False) -> bool:
        """
        删除用户
        """
        try:
            user = self.get_user_by_id(db, user_id)
            if not user:
                return False
            
            if permanent:
                # 永久删除
                db.delete(user)
            else:
                # 软删除
                user.deleted_at = datetime.utcnow()
                user.status = UserStatus.deleted
            
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"删除用户失败: {e}")
            return False
    
    def _create_user_quota(self, db: Session, user_id: int):
        """
        创建用户配额
        """
        try:
            quota = UserQuota(
                user_id=user_id,
                daily_messages_limit=100,  # 默认每日100条消息
                daily_messages_used=0,
                monthly_tokens_limit=100000,  # 默认每月10万tokens
                monthly_tokens_used=0,
                reset_date=datetime.utcnow().date(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            db.add(quota)
            db.commit()
            
        except Exception as e:
            logger.error(f"创建用户配额失败: {e}")
            # 不抛出异常，因为这不应该影响用户创建
    
    def reset_password(self, db: Session, email: str, new_password: str) -> bool:
        """
        重置密码
        """
        try:
            user = self.get_user_by_email(db, email)
            if not user:
                return False
            
            user.hashed_password = get_password_hash(new_password)
            user.updated_at = datetime.utcnow()
            
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"重置密码失败: {e}")
            return False
    
    def update_last_activity(self, db: Session, user_id: int):
        """
        更新用户最后活动时间
        """
        try:
            user = self.get_user_by_id(db, user_id)
            if user:
                user.last_login_at = datetime.utcnow()
                db.commit()
        except Exception as e:
            logger.error(f"更新用户活动时间失败: {e}")