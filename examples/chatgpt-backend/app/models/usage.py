# -*- coding: utf-8 -*-
"""
使用量统计模型

定义用户使用量统计相关的数据库模型和Pydantic模式。
"""

import sys
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from enum import Enum

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import Column, Integer, String, DateTime, Date, ForeignKey, JSON, Float, Index
from sqlalchemy.orm import relationship
from pydantic import BaseModel, validator

# 导入数据库基类
from app.core.database import Base


class UsageType(str, Enum):
    """使用类型枚举"""
    MESSAGE = "message"
    TOKEN = "token"
    API_CALL = "api_call"
    UPLOAD = "upload"
    EXPORT = "export"


class UsagePeriod(str, Enum):
    """使用周期枚举"""
    DAILY = "daily"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class Usage(Base):
    """使用量统计数据库模型"""
    __tablename__ = "usage"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # 用户关联
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # 使用信息
    usage_type = Column(String(20), nullable=False, index=True)
    usage_date = Column(Date, nullable=False, index=True)
    
    # 统计数据
    count = Column(Integer, default=0, nullable=False)
    tokens_used = Column(Integer, default=0, nullable=False)
    cost = Column(Float, default=0.0, nullable=False)  # 成本（美元）
    
    # 详细信息
    model = Column(String(50))
    conversation_id = Column(Integer, ForeignKey("conversations.id"), index=True)
    message_id = Column(Integer, ForeignKey("messages.id"), index=True)
    
    # 元数据
    metadata = Column(JSON, default=dict)
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # 关系
    user = relationship("User", back_populates="usage_records")
    conversation = relationship("Conversation")
    message = relationship("Message")
    
    # 索引
    __table_args__ = (
        Index('idx_user_date_type', 'user_id', 'usage_date', 'usage_type'),
        Index('idx_user_model_date', 'user_id', 'model', 'usage_date'),
    )
    
    def __repr__(self):
        return f"<Usage(id={self.id}, user_id={self.user_id}, type='{self.usage_type}', date={self.usage_date})>"


class UserQuota(Base):
    """用户配额数据库模型"""
    __tablename__ = "user_quotas"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # 用户关联
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)
    
    # 配额设置
    daily_message_limit = Column(Integer, default=100, nullable=False)
    monthly_message_limit = Column(Integer, default=3000, nullable=False)
    daily_token_limit = Column(Integer, default=50000, nullable=False)
    monthly_token_limit = Column(Integer, default=1500000, nullable=False)
    
    # 当前使用量
    daily_messages_used = Column(Integer, default=0, nullable=False)
    monthly_messages_used = Column(Integer, default=0, nullable=False)
    daily_tokens_used = Column(Integer, default=0, nullable=False)
    monthly_tokens_used = Column(Integer, default=0, nullable=False)
    
    # 重置时间
    daily_reset_at = Column(DateTime, nullable=False)
    monthly_reset_at = Column(DateTime, nullable=False)
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # 关系
    user = relationship("User")
    
    def is_daily_message_limit_exceeded(self) -> bool:
        """检查是否超过每日消息限制"""
        return self.daily_messages_used >= self.daily_message_limit
    
    def is_monthly_message_limit_exceeded(self) -> bool:
        """检查是否超过每月消息限制"""
        return self.monthly_messages_used >= self.monthly_message_limit
    
    def is_daily_token_limit_exceeded(self) -> bool:
        """检查是否超过每日token限制"""
        return self.daily_tokens_used >= self.daily_token_limit
    
    def is_monthly_token_limit_exceeded(self) -> bool:
        """检查是否超过每月token限制"""
        return self.monthly_tokens_used >= self.monthly_token_limit
    
    def can_send_message(self) -> bool:
        """检查是否可以发送消息"""
        return not (self.is_daily_message_limit_exceeded() or self.is_monthly_message_limit_exceeded())
    
    def can_use_tokens(self, tokens: int) -> bool:
        """检查是否可以使用指定数量的token"""
        return (self.daily_tokens_used + tokens <= self.daily_token_limit and
                self.monthly_tokens_used + tokens <= self.monthly_token_limit)
    
    def add_message_usage(self, tokens: int = 0):
        """添加消息使用量"""
        self.daily_messages_used += 1
        self.monthly_messages_used += 1
        self.daily_tokens_used += tokens
        self.monthly_tokens_used += tokens
        self.updated_at = datetime.utcnow()
    
    def reset_daily_usage(self):
        """重置每日使用量"""
        self.daily_messages_used = 0
        self.daily_tokens_used = 0
        self.daily_reset_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def reset_monthly_usage(self):
        """重置每月使用量"""
        self.monthly_messages_used = 0
        self.monthly_tokens_used = 0
        self.monthly_reset_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def __repr__(self):
        return f"<UserQuota(id={self.id}, user_id={self.user_id})>"


# Pydantic模式
class UsageBase(BaseModel):
    """使用量基础模式"""
    usage_type: UsageType
    count: int = 1
    tokens_used: int = 0
    cost: float = 0.0
    model: Optional[str] = None
    metadata: Dict[str, Any] = {}


class UsageCreate(UsageBase):
    """使用量创建模式"""
    user_id: int
    conversation_id: Optional[int] = None
    message_id: Optional[int] = None
    usage_date: Optional[date] = None


class UsageResponse(UsageBase):
    """使用量响应模式"""
    id: int
    user_id: int
    usage_date: date
    conversation_id: Optional[int] = None
    message_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class UsageStats(BaseModel):
    """使用量统计模式"""
    user_id: int
    period: UsagePeriod
    start_date: date
    end_date: date
    
    # 消息统计
    total_messages: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    
    # Token统计
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    # 成本统计
    total_cost: float = 0.0
    
    # 模型使用统计
    model_usage: Dict[str, int] = {}
    
    # 每日使用量（用于图表）
    daily_usage: List[Dict[str, Any]] = []
    
    class Config:
        from_attributes = True


class UserQuotaResponse(BaseModel):
    """用户配额响应模式"""
    id: int
    user_id: int
    
    # 配额限制
    daily_message_limit: int
    monthly_message_limit: int
    daily_token_limit: int
    monthly_token_limit: int
    
    # 当前使用量
    daily_messages_used: int
    monthly_messages_used: int
    daily_tokens_used: int
    monthly_tokens_used: int
    
    # 剩余配额
    daily_messages_remaining: int
    monthly_messages_remaining: int
    daily_tokens_remaining: int
    monthly_tokens_remaining: int
    
    # 使用率
    daily_message_usage_rate: float
    monthly_message_usage_rate: float
    daily_token_usage_rate: float
    monthly_token_usage_rate: float
    
    # 重置时间
    daily_reset_at: datetime
    monthly_reset_at: datetime
    
    class Config:
        from_attributes = True
    
    @classmethod
    def from_quota(cls, quota: UserQuota):
        """从UserQuota对象创建响应"""
        return cls(
            id=quota.id,
            user_id=quota.user_id,
            daily_message_limit=quota.daily_message_limit,
            monthly_message_limit=quota.monthly_message_limit,
            daily_token_limit=quota.daily_token_limit,
            monthly_token_limit=quota.monthly_token_limit,
            daily_messages_used=quota.daily_messages_used,
            monthly_messages_used=quota.monthly_messages_used,
            daily_tokens_used=quota.daily_tokens_used,
            monthly_tokens_used=quota.monthly_tokens_used,
            daily_messages_remaining=max(0, quota.daily_message_limit - quota.daily_messages_used),
            monthly_messages_remaining=max(0, quota.monthly_message_limit - quota.monthly_messages_used),
            daily_tokens_remaining=max(0, quota.daily_token_limit - quota.daily_tokens_used),
            monthly_tokens_remaining=max(0, quota.monthly_token_limit - quota.monthly_tokens_used),
            daily_message_usage_rate=quota.daily_messages_used / quota.daily_message_limit if quota.daily_message_limit > 0 else 0,
            monthly_message_usage_rate=quota.monthly_messages_used / quota.monthly_message_limit if quota.monthly_message_limit > 0 else 0,
            daily_token_usage_rate=quota.daily_tokens_used / quota.daily_token_limit if quota.daily_token_limit > 0 else 0,
            monthly_token_usage_rate=quota.monthly_tokens_used / quota.monthly_token_limit if quota.monthly_token_limit > 0 else 0,
            daily_reset_at=quota.daily_reset_at,
            monthly_reset_at=quota.monthly_reset_at
        )


class UsageReport(BaseModel):
    """使用量报告模式"""
    user_id: int
    report_period: UsagePeriod
    start_date: date
    end_date: date
    
    # 总体统计
    summary: UsageStats
    
    # 详细数据
    daily_breakdown: List[Dict[str, Any]]
    model_breakdown: List[Dict[str, Any]]
    conversation_breakdown: List[Dict[str, Any]]
    
    # 趋势分析
    trends: Dict[str, Any]
    
    # 建议
    recommendations: List[str]
    
    class Config:
        from_attributes = True


class UsageList(BaseModel):
    """使用量列表模式"""
    usage_records: List[UsageResponse]
    total: int
    page: int
    size: int
    pages: int
    
    class Config:
        from_attributes = True