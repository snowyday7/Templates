# -*- coding: utf-8 -*-
"""
用户数据模型

定义用户相关的数据库模型和Pydantic模式。
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from enum import Enum

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from pydantic import BaseModel, EmailStr, validator
from passlib.context import CryptContext

# 导入模板库组件
from src.models.base import BaseDBModel
from src.utils.security import hash_password, verify_password

# 导入数据库基类
from app.core.database import Base

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserRole(str, Enum):
    """用户角色枚举"""
    ADMIN = "admin"
    USER = "user"
    PREMIUM = "premium"


class UserStatus(str, Enum):
    """用户状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class User(Base):
    """用户数据库模型"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    avatar_url = Column(String(500))
    
    # 用户状态
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    status = Column(SQLEnum(UserStatus), default=UserStatus.ACTIVE, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # 配额设置
    daily_message_limit = Column(Integer, default=100)
    monthly_token_limit = Column(Integer, default=100000)
    
    # 个人设置
    preferred_model = Column(String(50), default="gpt-3.5-turbo")
    system_prompt = Column(Text)
    language = Column(String(10), default="zh-CN")
    timezone = Column(String(50), default="Asia/Shanghai")
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login_at = Column(DateTime)
    
    # 关系
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    usage_records = relationship("Usage", back_populates="user", cascade="all, delete-orphan")
    
    def set_password(self, password: str):
        """设置密码"""
        self.hashed_password = pwd_context.hash(password)
    
    def verify_password(self, password: str) -> bool:
        """验证密码"""
        return pwd_context.verify(password, self.hashed_password)
    
    def is_active(self) -> bool:
        """检查用户是否激活"""
        return self.status == UserStatus.ACTIVE
    
    def is_admin(self) -> bool:
        """检查是否为管理员"""
        return self.role == UserRole.ADMIN
    
    def is_premium(self) -> bool:
        """检查是否为高级用户"""
        return self.role in [UserRole.PREMIUM, UserRole.ADMIN]
    
    def update_last_login(self):
        """更新最后登录时间"""
        self.last_login_at = datetime.utcnow()
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"


# Pydantic模式
class UserBase(BaseModel):
    """用户基础模式"""
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    language: str = "zh-CN"
    timezone: str = "Asia/Shanghai"
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3 or len(v) > 50:
            raise ValueError('用户名长度必须在3-50个字符之间')
        if not v.isalnum() and '_' not in v:
            raise ValueError('用户名只能包含字母、数字和下划线')
        return v


class UserCreate(UserBase):
    """用户创建模式"""
    password: str
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('密码长度至少8个字符')
        if not any(c.isupper() for c in v):
            raise ValueError('密码必须包含至少一个大写字母')
        if not any(c.islower() for c in v):
            raise ValueError('密码必须包含至少一个小写字母')
        if not any(c.isdigit() for c in v):
            raise ValueError('密码必须包含至少一个数字')
        return v


class UserUpdate(BaseModel):
    """用户更新模式"""
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    preferred_model: Optional[str] = None
    system_prompt: Optional[str] = None
    language: Optional[str] = None
    timezone: Optional[str] = None


class UserPasswordUpdate(BaseModel):
    """用户密码更新模式"""
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('密码长度至少8个字符')
        if not any(c.isupper() for c in v):
            raise ValueError('密码必须包含至少一个大写字母')
        if not any(c.islower() for c in v):
            raise ValueError('密码必须包含至少一个小写字母')
        if not any(c.isdigit() for c in v):
            raise ValueError('密码必须包含至少一个数字')
        return v


class UserResponse(UserBase):
    """用户响应模式"""
    id: int
    role: UserRole
    status: UserStatus
    is_verified: bool
    daily_message_limit: int
    monthly_token_limit: int
    preferred_model: str
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserProfile(BaseModel):
    """用户档案模式"""
    id: int
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    role: UserRole
    status: UserStatus
    is_verified: bool
    preferred_model: str
    language: str
    timezone: str
    created_at: datetime
    last_login_at: Optional[datetime] = None
    
    # 统计信息
    total_conversations: int = 0
    total_messages: int = 0
    total_tokens_used: int = 0
    
    class Config:
        from_attributes = True


class UserStats(BaseModel):
    """用户统计模式"""
    user_id: int
    today_messages: int = 0
    today_tokens: int = 0
    month_messages: int = 0
    month_tokens: int = 0
    total_conversations: int = 0
    total_messages: int = 0
    total_tokens: int = 0
    daily_limit_remaining: int = 0
    monthly_limit_remaining: int = 0
    
    class Config:
        from_attributes = True


class UserList(BaseModel):
    """用户列表模式"""
    users: List[UserResponse]
    total: int
    page: int
    size: int
    pages: int
    
    class Config:
        from_attributes = True