"""API密钥管理系统

提供完整的API密钥管理功能，包括：
- API密钥生成和验证
- 密钥权限范围管理
- 密钥生命周期管理
- 使用统计和监控
"""

import secrets
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Set
from enum import Enum
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON
from sqlalchemy.orm import Session

from ..database.sqlalchemy_template import BaseModel as DBBaseModel


class APIKeyScope(str, Enum):
    """API密钥权限范围"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    USER_MANAGEMENT = "user_management"
    SYSTEM_CONFIG = "system_config"
    ANALYTICS = "analytics"
    BILLING = "billing"


class APIKeyStatus(str, Enum):
    """API密钥状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    REVOKED = "revoked"
    EXPIRED = "expired"


@dataclass
class APIKeyUsage:
    """API密钥使用统计"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_used: Optional[datetime] = None
    rate_limit_hits: int = 0
    bandwidth_used: int = 0  # bytes


class APIKey(DBBaseModel):
    """API密钥数据模型"""
    __tablename__ = "api_keys"

    key_id = Column(String(64), unique=True, nullable=False, index=True)
    key_hash = Column(String(128), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # 权限和范围
    scopes = Column(JSON, default=list)  # List[APIKeyScope]
    allowed_ips = Column(JSON, default=list)  # IP白名单
    allowed_domains = Column(JSON, default=list)  # 域名白名单
    
    # 生命周期
    status = Column(String(20), default=APIKeyStatus.ACTIVE.value)
    expires_at = Column(DateTime(timezone=True))
    last_used_at = Column(DateTime(timezone=True))
    
    # 使用限制
    rate_limit = Column(Integer, default=1000)  # 每小时请求数
    daily_limit = Column(Integer, default=10000)  # 每日请求数
    monthly_limit = Column(Integer, default=100000)  # 每月请求数
    
    # 使用统计
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    
    # 关联用户
    user_id = Column(Integer, nullable=True)
    created_by = Column(Integer, nullable=False)
    
    def is_valid(self) -> bool:
        """检查密钥是否有效"""
        if self.status != APIKeyStatus.ACTIVE.value:
            return False
        
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
            
        return True
    
    def has_scope(self, scope: APIKeyScope) -> bool:
        """检查是否有指定权限"""
        return scope.value in (self.scopes or [])
    
    def is_ip_allowed(self, ip: str) -> bool:
        """检查IP是否在白名单中"""
        if not self.allowed_ips:
            return True
        return ip in self.allowed_ips
    
    def is_domain_allowed(self, domain: str) -> bool:
        """检查域名是否在白名单中"""
        if not self.allowed_domains:
            return True
        return domain in self.allowed_domains


class APIKeyCreateRequest(BaseModel):
    """API密钥创建请求"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    scopes: List[APIKeyScope] = Field(default_factory=list)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)
    rate_limit: int = Field(1000, ge=1, le=10000)
    daily_limit: int = Field(10000, ge=1, le=100000)
    monthly_limit: int = Field(100000, ge=1, le=1000000)
    allowed_ips: Optional[List[str]] = None
    allowed_domains: Optional[List[str]] = None


class APIKeyResponse(BaseModel):
    """API密钥响应"""
    key_id: str
    name: str
    description: Optional[str]
    scopes: List[str]
    status: str
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    rate_limit: int
    daily_limit: int
    monthly_limit: int
    total_requests: int
    successful_requests: int
    failed_requests: int


class APIKeyManager:
    """API密钥管理器"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.usage_cache: Dict[str, APIKeyUsage] = {}
    
    def generate_api_key(self, session: Session, request: APIKeyCreateRequest, created_by: int) -> tuple[str, APIKey]:
        """生成新的API密钥"""
        # 生成密钥
        key = self._generate_key()
        key_id = self._generate_key_id()
        key_hash = self._hash_key(key)
        
        # 计算过期时间
        expires_at = None
        if request.expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=request.expires_in_days)
        
        # 创建数据库记录
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=request.name,
            description=request.description,
            scopes=[scope.value for scope in request.scopes],
            expires_at=expires_at,
            rate_limit=request.rate_limit,
            daily_limit=request.daily_limit,
            monthly_limit=request.monthly_limit,
            allowed_ips=request.allowed_ips,
            allowed_domains=request.allowed_domains,
            created_by=created_by
        )
        
        session.add(api_key)
        session.flush()
        
        return key, api_key
    
    def validate_api_key(self, session: Session, key: str, required_scope: Optional[APIKeyScope] = None, 
                        client_ip: Optional[str] = None, domain: Optional[str] = None) -> Optional[APIKey]:
        """验证API密钥"""
        try:
            # 解析密钥
            key_id = self._extract_key_id(key)
            if not key_id:
                return None
            
            # 查找密钥记录
            api_key = session.query(APIKey).filter(
                APIKey.key_id == key_id,
                APIKey.is_active == True
            ).first()
            
            if not api_key:
                return None
            
            # 验证密钥哈希
            if not self._verify_key(key, api_key.key_hash):
                return None
            
            # 检查密钥状态和有效性
            if not api_key.is_valid():
                return None
            
            # 检查权限范围
            if required_scope and not api_key.has_scope(required_scope):
                return None
            
            # 检查IP白名单
            if client_ip and not api_key.is_ip_allowed(client_ip):
                return None
            
            # 检查域名白名单
            if domain and not api_key.is_domain_allowed(domain):
                return None
            
            # 检查使用限制
            if not self._check_rate_limits(api_key):
                return None
            
            # 更新使用统计
            self._update_usage_stats(session, api_key)
            
            return api_key
            
        except Exception:
            return None
    
    def revoke_api_key(self, session: Session, key_id: str) -> bool:
        """撤销API密钥"""
        api_key = session.query(APIKey).filter(
            APIKey.key_id == key_id,
            APIKey.is_active == True
        ).first()
        
        if api_key:
            api_key.status = APIKeyStatus.REVOKED.value
            api_key.updated_at = datetime.now(timezone.utc)
            session.commit()
            return True
        
        return False
    
    def list_api_keys(self, session: Session, user_id: Optional[int] = None, 
                     status: Optional[APIKeyStatus] = None) -> List[APIKey]:
        """列出API密钥"""
        query = session.query(APIKey).filter(APIKey.is_active == True)
        
        if user_id:
            query = query.filter(APIKey.created_by == user_id)
        
        if status:
            query = query.filter(APIKey.status == status.value)
        
        return query.order_by(APIKey.created_at.desc()).all()
    
    def get_usage_stats(self, session: Session, key_id: str) -> Optional[Dict[str, Any]]:
        """获取密钥使用统计"""
        api_key = session.query(APIKey).filter(
            APIKey.key_id == key_id,
            APIKey.is_active == True
        ).first()
        
        if not api_key:
            return None
        
        return {
            "total_requests": api_key.total_requests,
            "successful_requests": api_key.successful_requests,
            "failed_requests": api_key.failed_requests,
            "success_rate": api_key.successful_requests / max(api_key.total_requests, 1) * 100,
            "last_used_at": api_key.last_used_at,
            "rate_limit": api_key.rate_limit,
            "daily_limit": api_key.daily_limit,
            "monthly_limit": api_key.monthly_limit
        }
    
    def _generate_key(self) -> str:
        """生成API密钥"""
        # 生成32字节的随机数据
        random_bytes = secrets.token_bytes(32)
        # 转换为十六进制字符串
        return random_bytes.hex()
    
    def _generate_key_id(self) -> str:
        """生成密钥ID"""
        return secrets.token_urlsafe(32)
    
    def _hash_key(self, key: str) -> str:
        """哈希API密钥"""
        return hmac.new(
            self.secret_key.encode(),
            key.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def _verify_key(self, key: str, key_hash: str) -> bool:
        """验证API密钥"""
        expected_hash = self._hash_key(key)
        return hmac.compare_digest(expected_hash, key_hash)
    
    def _extract_key_id(self, key: str) -> Optional[str]:
        """从密钥中提取密钥ID"""
        # 这里简化处理，实际应该从密钥结构中解析
        # 可以考虑使用JWT格式或自定义格式
        return key[:32] if len(key) >= 32 else None
    
    def _check_rate_limits(self, api_key: APIKey) -> bool:
        """检查速率限制"""
        # 这里应该实现基于Redis的速率限制检查
        # 简化实现，实际应该检查小时、日、月限制
        return True
    
    def _update_usage_stats(self, session: Session, api_key: APIKey):
        """更新使用统计"""
        api_key.total_requests += 1
        api_key.last_used_at = datetime.now(timezone.utc)
        session.commit()


# 便捷函数
def generate_api_key(session: Session, request: APIKeyCreateRequest, 
                    created_by: int, secret_key: str) -> tuple[str, APIKey]:
    """生成API密钥的便捷函数"""
    manager = APIKeyManager(secret_key)
    return manager.generate_api_key(session, request, created_by)


def validate_api_key(session: Session, key: str, secret_key: str, 
                    required_scope: Optional[APIKeyScope] = None,
                    client_ip: Optional[str] = None,
                    domain: Optional[str] = None) -> Optional[APIKey]:
    """验证API密钥的便捷函数"""
    manager = APIKeyManager(secret_key)
    return manager.validate_api_key(session, key, required_scope, client_ip, domain)


def revoke_api_key(session: Session, key_id: str) -> bool:
    """撤销API密钥的便捷函数"""
    manager = APIKeyManager("dummy")  # 撤销不需要secret_key
    return manager.revoke_api_key(session, key_id)