# -*- coding: utf-8 -*-
"""
消息数据模型

定义消息相关的数据库模型和Pydantic模式。
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from pydantic import BaseModel, validator

# 导入数据库基类
from app.core.database import Base


class MessageRole(str, Enum):
    """消息角色枚举"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageStatus(str, Enum):
    """消息状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Message(Base):
    """消息数据库模型"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # 关联
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # 消息内容
    role = Column(String(20), nullable=False, index=True)
    content = Column(Text, nullable=False)
    raw_content = Column(Text)  # 原始内容（用于编辑历史）
    
    # 消息状态
    status = Column(String(20), default=MessageStatus.COMPLETED, nullable=False)
    error_message = Column(Text)
    
    # AI相关信息
    model = Column(String(50))
    tokens_used = Column(Integer, default=0)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    temperature = Column(Float)
    
    # 消息元数据
    metadata = Column(JSON, default=dict)
    attachments = Column(JSON, default=list)  # 附件信息
    
    # 消息特性
    is_edited = Column(Boolean, default=False, nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False)
    is_pinned = Column(Boolean, default=False, nullable=False)
    
    # 父消息（用于消息分支）
    parent_id = Column(Integer, ForeignKey("messages.id"), index=True)
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime)
    
    # 关系
    conversation = relationship("Conversation", back_populates="messages")
    user = relationship("User")
    parent = relationship("Message", remote_side=[id], backref="children")
    
    def mark_as_processing(self):
        """标记为处理中"""
        self.status = MessageStatus.PROCESSING
        self.updated_at = datetime.utcnow()
    
    def mark_as_completed(self, tokens_used: int = 0, prompt_tokens: int = 0, completion_tokens: int = 0):
        """标记为完成"""
        self.status = MessageStatus.COMPLETED
        self.tokens_used = tokens_used
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def mark_as_failed(self, error_message: str):
        """标记为失败"""
        self.status = MessageStatus.FAILED
        self.error_message = error_message
        self.updated_at = datetime.utcnow()
    
    def edit_content(self, new_content: str):
        """编辑消息内容"""
        if not self.raw_content:
            self.raw_content = self.content
        self.content = new_content
        self.is_edited = True
        self.updated_at = datetime.utcnow()
    
    def soft_delete(self):
        """软删除消息"""
        self.is_deleted = True
        self.updated_at = datetime.utcnow()
    
    def pin(self):
        """置顶消息"""
        self.is_pinned = True
        self.updated_at = datetime.utcnow()
    
    def unpin(self):
        """取消置顶"""
        self.is_pinned = False
        self.updated_at = datetime.utcnow()
    
    def add_attachment(self, attachment: Dict[str, Any]):
        """添加附件"""
        if self.attachments is None:
            self.attachments = []
        self.attachments.append(attachment)
        self.updated_at = datetime.utcnow()
    
    def is_user_message(self) -> bool:
        """检查是否为用户消息"""
        return self.role == MessageRole.USER
    
    def is_assistant_message(self) -> bool:
        """检查是否为助手消息"""
        return self.role == MessageRole.ASSISTANT
    
    def is_system_message(self) -> bool:
        """检查是否为系统消息"""
        return self.role == MessageRole.SYSTEM
    
    def get_display_content(self) -> str:
        """获取显示内容（处理删除状态）"""
        if self.is_deleted:
            return "[此消息已被删除]"
        return self.content
    
    def __repr__(self):
        return f"<Message(id={self.id}, role='{self.role}', conversation_id={self.conversation_id})>"


# Pydantic模式
class MessageBase(BaseModel):
    """消息基础模式"""
    content: str
    role: MessageRole = MessageRole.USER
    
    @validator('content')
    def validate_content(cls, v):
        if len(v.strip()) < 1:
            raise ValueError('消息内容不能为空')
        if len(v) > 10000:
            raise ValueError('消息内容不能超过10000个字符')
        return v.strip()


class MessageCreate(MessageBase):
    """消息创建模式"""
    conversation_id: int
    parent_id: Optional[int] = None
    attachments: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}


class MessageUpdate(BaseModel):
    """消息更新模式"""
    content: Optional[str] = None
    
    @validator('content')
    def validate_content(cls, v):
        if v is not None:
            if len(v.strip()) < 1:
                raise ValueError('消息内容不能为空')
            if len(v) > 10000:
                raise ValueError('消息内容不能超过10000个字符')
            return v.strip()
        return v


class MessageResponse(MessageBase):
    """消息响应模式"""
    id: int
    conversation_id: int
    user_id: int
    status: MessageStatus
    error_message: Optional[str] = None
    model: Optional[str] = None
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    temperature: Optional[float] = None
    metadata: Dict[str, Any] = {}
    attachments: List[Dict[str, Any]] = []
    is_edited: bool = False
    is_deleted: bool = False
    is_pinned: bool = False
    parent_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class MessageDetail(MessageResponse):
    """消息详情模式（包含子消息）"""
    children: List['MessageResponse'] = []
    raw_content: Optional[str] = None
    
    class Config:
        from_attributes = True


class MessageList(BaseModel):
    """消息列表模式"""
    messages: List[MessageResponse]
    total: int
    page: int
    size: int
    pages: int
    
    class Config:
        from_attributes = True


class MessageStream(BaseModel):
    """流式消息模式"""
    id: int
    conversation_id: int
    role: MessageRole
    content: str
    delta: str  # 增量内容
    is_complete: bool = False
    tokens_used: int = 0
    
    class Config:
        from_attributes = True


class MessageStats(BaseModel):
    """消息统计模式"""
    total_messages: int
    user_messages: int
    assistant_messages: int
    system_messages: int
    total_tokens: int
    avg_tokens_per_message: float
    most_used_model: str
    
    class Config:
        from_attributes = True


class MessageAttachment(BaseModel):
    """消息附件模式"""
    id: str
    name: str
    type: str  # image, file, url, etc.
    url: str
    size: Optional[int] = None
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        from_attributes = True


class MessageReaction(BaseModel):
    """消息反应模式"""
    message_id: int
    user_id: int
    reaction: str  # 👍, 👎, ❤️, etc.
    created_at: datetime
    
    class Config:
        from_attributes = True


class MessageExport(BaseModel):
    """消息导出模式"""
    conversation_title: str
    messages: List[Dict[str, Any]]
    export_format: str  # json, markdown, txt
    created_at: datetime
    
    class Config:
        from_attributes = True


# 解决前向引用
MessageDetail.model_rebuild()