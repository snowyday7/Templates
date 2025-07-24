# -*- coding: utf-8 -*-
"""
对话数据模型

定义对话相关的数据库模型和Pydantic模式。
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from enum import Enum

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from pydantic import BaseModel, validator

# 导入数据库基类
from app.core.database import Base


class ConversationStatus(str, Enum):
    """对话状态枚举"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class Conversation(Base):
    """对话数据库模型"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    
    # 用户关联
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # 对话配置
    model = Column(String(50), default="gpt-3.5-turbo", nullable=False)
    system_prompt = Column(Text)
    temperature = Column(Integer, default=70)  # 存储为整数，实际值除以100
    max_tokens = Column(Integer, default=2048)
    
    # 对话状态
    status = Column(String(20), default=ConversationStatus.ACTIVE, nullable=False)
    is_pinned = Column(Boolean, default=False, nullable=False)
    is_shared = Column(Boolean, default=False, nullable=False)
    share_token = Column(String(64), unique=True, index=True)
    
    # 统计信息
    message_count = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    
    # 元数据
    metadata = Column(JSON, default=dict)
    tags = Column(JSON, default=list)  # 标签列表
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_message_at = Column(DateTime)
    
    # 关系
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")
    
    def update_stats(self):
        """更新统计信息"""
        self.message_count = len(self.messages)
        self.total_tokens = sum(msg.tokens_used or 0 for msg in self.messages)
        if self.messages:
            self.last_message_at = max(msg.created_at for msg in self.messages)
        self.updated_at = datetime.utcnow()
    
    def is_active(self) -> bool:
        """检查对话是否激活"""
        return self.status == ConversationStatus.ACTIVE
    
    def archive(self):
        """归档对话"""
        self.status = ConversationStatus.ARCHIVED
        self.updated_at = datetime.utcnow()
    
    def delete(self):
        """删除对话（软删除）"""
        self.status = ConversationStatus.DELETED
        self.updated_at = datetime.utcnow()
    
    def pin(self):
        """置顶对话"""
        self.is_pinned = True
        self.updated_at = datetime.utcnow()
    
    def unpin(self):
        """取消置顶"""
        self.is_pinned = False
        self.updated_at = datetime.utcnow()
    
    def add_tag(self, tag: str):
        """添加标签"""
        if self.tags is None:
            self.tags = []
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str):
        """移除标签"""
        if self.tags and tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def get_temperature(self) -> float:
        """获取实际温度值"""
        return self.temperature / 100.0 if self.temperature is not None else 0.7
    
    def set_temperature(self, temp: float):
        """设置温度值"""
        self.temperature = int(temp * 100)
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, title='{self.title}', user_id={self.user_id})>"


# Pydantic模式
class ConversationBase(BaseModel):
    """对话基础模式"""
    title: str
    description: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    tags: List[str] = []
    
    @validator('title')
    def validate_title(cls, v):
        if len(v.strip()) < 1:
            raise ValueError('对话标题不能为空')
        if len(v) > 200:
            raise ValueError('对话标题不能超过200个字符')
        return v.strip()
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('温度值必须在0-2之间')
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if not 1 <= v <= 8192:
            raise ValueError('最大token数必须在1-8192之间')
        return v


class ConversationCreate(ConversationBase):
    """对话创建模式"""
    pass


class ConversationUpdate(BaseModel):
    """对话更新模式"""
    title: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tags: Optional[List[str]] = None
    
    @validator('title')
    def validate_title(cls, v):
        if v is not None:
            if len(v.strip()) < 1:
                raise ValueError('对话标题不能为空')
            if len(v) > 200:
                raise ValueError('对话标题不能超过200个字符')
            return v.strip()
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v is not None and not 0 <= v <= 2:
            raise ValueError('温度值必须在0-2之间')
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v is not None and not 1 <= v <= 8192:
            raise ValueError('最大token数必须在1-8192之间')
        return v


class ConversationResponse(ConversationBase):
    """对话响应模式"""
    id: int
    user_id: int
    status: ConversationStatus
    is_pinned: bool
    is_shared: bool
    share_token: Optional[str] = None
    message_count: int
    total_tokens: int
    created_at: datetime
    updated_at: datetime
    last_message_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ConversationDetail(ConversationResponse):
    """对话详情模式（包含消息）"""
    messages: List['MessageResponse'] = []
    
    class Config:
        from_attributes = True


class ConversationSummary(BaseModel):
    """对话摘要模式"""
    id: int
    title: str
    message_count: int
    last_message_at: Optional[datetime] = None
    is_pinned: bool
    tags: List[str] = []
    
    class Config:
        from_attributes = True


class ConversationList(BaseModel):
    """对话列表模式"""
    conversations: List[ConversationResponse]
    total: int
    page: int
    size: int
    pages: int
    
    class Config:
        from_attributes = True


class ConversationStats(BaseModel):
    """对话统计模式"""
    total_conversations: int
    active_conversations: int
    archived_conversations: int
    total_messages: int
    total_tokens: int
    avg_messages_per_conversation: float
    most_used_model: str
    
    class Config:
        from_attributes = True


class ConversationShare(BaseModel):
    """对话分享模式"""
    share_token: str
    is_shared: bool
    share_url: str
    
    class Config:
        from_attributes = True


# 前向引用解决
from app.models.message import MessageResponse
ConversationDetail.model_rebuild()