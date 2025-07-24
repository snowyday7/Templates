"""基于角色的访问控制(RBAC)系统

提供完整的RBAC功能，包括：
- 角色和权限管理
- 用户角色分配
- 权限检查和验证
- 动态权限控制
"""

from typing import List, Set, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime, timezone
from functools import wraps

from fastapi import HTTPException, status, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Table
from sqlalchemy.orm import Session, relationship

from ..database.sqlalchemy_template import BaseModel as DBBaseModel, Base


class PermissionType(str, Enum):
    """权限类型"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


class ResourceType(str, Enum):
    """资源类型"""
    USER = "user"
    ROLE = "role"
    PERMISSION = "permission"
    API_KEY = "api_key"
    SYSTEM = "system"
    DATA = "data"
    REPORT = "report"
    BILLING = "billing"


# 关联表
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True),
    Column('assigned_at', DateTime(timezone=True), default=datetime.utcnow),
    Column('assigned_by', Integer, ForeignKey('users.id')),
    Column('expires_at', DateTime(timezone=True), nullable=True)
)

role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True),
    Column('permission_id', Integer, ForeignKey('permissions.id'), primary_key=True),
    Column('granted_at', DateTime(timezone=True), default=datetime.utcnow),
    Column('granted_by', Integer, ForeignKey('users.id'))
)


class Permission(DBBaseModel):
    """权限模型"""
    __tablename__ = "permissions"

    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(200), nullable=False)
    description = Column(Text)
    resource_type = Column(String(50), nullable=False)
    permission_type = Column(String(50), nullable=False)
    
    # 权限层级和依赖
    parent_id = Column(Integer, ForeignKey('permissions.id'), nullable=True)
    level = Column(Integer, default=0)  # 权限层级
    
    # 权限约束
    conditions = Column(Text)  # JSON格式的权限条件
    
    # 关系
    parent = relationship("Permission", remote_side="Permission.id")
    children = relationship("Permission", back_populates="parent")
    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")
    
    def __str__(self):
        return f"{self.resource_type}:{self.permission_type}"
    
    def get_full_name(self) -> str:
        """获取完整权限名称"""
        return f"{self.resource_type}.{self.permission_type}"


class Role(DBBaseModel):
    """角色模型"""
    __tablename__ = "roles"

    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # 角色层级
    level = Column(Integer, default=0)  # 角色层级，数字越大权限越高
    is_system_role = Column(Boolean, default=False)  # 是否为系统角色
    is_default = Column(Boolean, default=False)  # 是否为默认角色
    
    # 角色约束
    max_users = Column(Integer, nullable=True)  # 最大用户数限制
    
    # 关系
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")
    
    def __str__(self):
        return self.name
    
    def has_permission(self, permission: Union[str, Permission]) -> bool:
        """检查角色是否有指定权限"""
        if isinstance(permission, str):
            return any(p.name == permission for p in self.permissions)
        return permission in self.permissions
    
    def get_permission_names(self) -> Set[str]:
        """获取角色的所有权限名称"""
        return {p.name for p in self.permissions}


class UserRole(DBBaseModel):
    """用户角色关联模型（用于存储额外信息）"""
    __tablename__ = "user_role_assignments"
    
    user_id = Column(Integer, nullable=False, index=True)
    role_id = Column(Integer, ForeignKey('roles.id'), nullable=False)
    assigned_by = Column(Integer, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    conditions = Column(Text)  # 角色分配条件
    
    # 关系
    role = relationship("Role")


class RoleCreateRequest(BaseModel):
    """角色创建请求"""
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    level: int = Field(0, ge=0, le=100)
    permission_ids: List[int] = Field(default_factory=list)
    max_users: Optional[int] = Field(None, ge=1)


class PermissionCreateRequest(BaseModel):
    """权限创建请求"""
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    resource_type: ResourceType
    permission_type: PermissionType
    parent_id: Optional[int] = None
    conditions: Optional[str] = None


class UserRoleAssignRequest(BaseModel):
    """用户角色分配请求"""
    user_id: int
    role_ids: List[int]
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)
    conditions: Optional[str] = None


class RBACManager:
    """RBAC管理器"""
    
    def __init__(self):
        self.permission_cache: Dict[int, Set[str]] = {}  # 用户权限缓存
        self.role_cache: Dict[int, Set[str]] = {}  # 用户角色缓存
    
    def create_permission(self, session: Session, request: PermissionCreateRequest) -> Permission:
        """创建权限"""
        # 检查权限是否已存在
        existing = session.query(Permission).filter(
            Permission.name == request.name
        ).first()
        
        if existing:
            raise ValueError(f"Permission '{request.name}' already exists")
        
        # 计算权限层级
        level = 0
        if request.parent_id:
            parent = session.query(Permission).get(request.parent_id)
            if parent:
                level = parent.level + 1
        
        permission = Permission(
            name=request.name,
            display_name=request.display_name,
            description=request.description,
            resource_type=request.resource_type.value,
            permission_type=request.permission_type.value,
            parent_id=request.parent_id,
            level=level,
            conditions=request.conditions
        )
        
        session.add(permission)
        session.flush()
        return permission
    
    def create_role(self, session: Session, request: RoleCreateRequest) -> Role:
        """创建角色"""
        # 检查角色是否已存在
        existing = session.query(Role).filter(
            Role.name == request.name
        ).first()
        
        if existing:
            raise ValueError(f"Role '{request.name}' already exists")
        
        role = Role(
            name=request.name,
            display_name=request.display_name,
            description=request.description,
            level=request.level,
            max_users=request.max_users
        )
        
        # 分配权限
        if request.permission_ids:
            permissions = session.query(Permission).filter(
                Permission.id.in_(request.permission_ids)
            ).all()
            role.permissions.extend(permissions)
        
        session.add(role)
        session.flush()
        return role
    
    def assign_role_to_user(self, session: Session, request: UserRoleAssignRequest, 
                           assigned_by: int) -> List[UserRole]:
        """为用户分配角色"""
        # 检查用户是否存在（这里假设有User模型）
        # user = session.query(User).get(request.user_id)
        # if not user:
        #     raise ValueError(f"User {request.user_id} not found")
        
        # 计算过期时间
        expires_at = None
        if request.expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=request.expires_in_days)
        
        assignments = []
        for role_id in request.role_ids:
            # 检查角色是否存在
            role = session.query(Role).get(role_id)
            if not role:
                continue
            
            # 检查是否已分配
            existing = session.query(UserRole).filter(
                UserRole.user_id == request.user_id,
                UserRole.role_id == role_id,
                UserRole.is_active == True
            ).first()
            
            if existing:
                continue
            
            assignment = UserRole(
                user_id=request.user_id,
                role_id=role_id,
                assigned_by=assigned_by,
                expires_at=expires_at,
                conditions=request.conditions
            )
            
            session.add(assignment)
            assignments.append(assignment)
        
        session.flush()
        
        # 清除缓存
        self._clear_user_cache(request.user_id)
        
        return assignments
    
    def revoke_user_role(self, session: Session, user_id: int, role_id: int) -> bool:
        """撤销用户角色"""
        assignment = session.query(UserRole).filter(
            UserRole.user_id == user_id,
            UserRole.role_id == role_id,
            UserRole.is_active == True
        ).first()
        
        if assignment:
            assignment.is_active = False
            assignment.updated_at = datetime.now(timezone.utc)
            session.commit()
            
            # 清除缓存
            self._clear_user_cache(user_id)
            return True
        
        return False
    
    def get_user_permissions(self, session: Session, user_id: int) -> Set[str]:
        """获取用户的所有权限"""
        # 检查缓存
        if user_id in self.permission_cache:
            return self.permission_cache[user_id]
        
        permissions = set()
        
        # 获取用户的所有有效角色
        user_roles = session.query(UserRole).filter(
            UserRole.user_id == user_id,
            UserRole.is_active == True
        ).all()
        
        for user_role in user_roles:
            # 检查角色是否过期
            if user_role.expires_at and datetime.now(timezone.utc) > user_role.expires_at:
                continue
            
            # 获取角色的权限
            role_permissions = session.query(Permission).join(
                role_permissions
            ).filter(
                role_permissions.c.role_id == user_role.role_id
            ).all()
            
            for permission in role_permissions:
                permissions.add(permission.name)
        
        # 缓存结果
        self.permission_cache[user_id] = permissions
        
        return permissions
    
    def get_user_roles(self, session: Session, user_id: int) -> Set[str]:
        """获取用户的所有角色"""
        # 检查缓存
        if user_id in self.role_cache:
            return self.role_cache[user_id]
        
        roles = set()
        
        user_roles = session.query(UserRole).join(Role).filter(
            UserRole.user_id == user_id,
            UserRole.is_active == True
        ).all()
        
        for user_role in user_roles:
            # 检查角色是否过期
            if user_role.expires_at and datetime.now(timezone.utc) > user_role.expires_at:
                continue
            
            roles.add(user_role.role.name)
        
        # 缓存结果
        self.role_cache[user_id] = roles
        
        return roles
    
    def check_permission(self, session: Session, user_id: int, 
                        permission: str, resource_id: Optional[int] = None) -> bool:
        """检查用户是否有指定权限"""
        user_permissions = self.get_user_permissions(session, user_id)
        
        # 直接权限检查
        if permission in user_permissions:
            return True
        
        # 通配符权限检查
        if "*" in user_permissions or "admin" in user_permissions:
            return True
        
        # 资源级权限检查
        if resource_id:
            resource_permission = f"{permission}:{resource_id}"
            if resource_permission in user_permissions:
                return True
        
        return False
    
    def _clear_user_cache(self, user_id: int):
        """清除用户缓存"""
        self.permission_cache.pop(user_id, None)
        self.role_cache.pop(user_id, None)
    
    def clear_all_cache(self):
        """清除所有缓存"""
        self.permission_cache.clear()
        self.role_cache.clear()


# 全局RBAC管理器实例
rbac_manager = RBACManager()


# 装饰器和依赖注入
def require_permission(permission: str, resource_id: Optional[int] = None):
    """权限检查装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 这里需要从请求中获取用户ID和数据库会话
            # 实际实现需要根据具体的认证系统调整
            request = kwargs.get('request') or (args[0] if args and isinstance(args[0], Request) else None)
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found"
                )
            
            # 从请求中获取用户信息（需要先通过认证中间件）
            user_id = getattr(request.state, 'user_id', None)
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # 获取数据库会话（需要依赖注入）
            session = kwargs.get('session')
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database session not found"
                )
            
            # 检查权限
            if not rbac_manager.check_permission(session, user_id, permission, resource_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(role: str):
    """角色检查装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request') or (args[0] if args and isinstance(args[0], Request) else None)
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found"
                )
            
            user_id = getattr(request.state, 'user_id', None)
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            session = kwargs.get('session')
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database session not found"
                )
            
            user_roles = rbac_manager.get_user_roles(session, user_id)
            if role not in user_roles and "admin" not in user_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role '{role}' required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# 便捷函数
def check_permission(session: Session, user_id: int, permission: str, 
                    resource_id: Optional[int] = None) -> bool:
    """检查权限的便捷函数"""
    return rbac_manager.check_permission(session, user_id, permission, resource_id)


def get_user_permissions(session: Session, user_id: int) -> Set[str]:
    """获取用户权限的便捷函数"""
    return rbac_manager.get_user_permissions(session, user_id)


def get_user_roles(session: Session, user_id: int) -> Set[str]:
    """获取用户角色的便捷函数"""
    return rbac_manager.get_user_roles(session, user_id)