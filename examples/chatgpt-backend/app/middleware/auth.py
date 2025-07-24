# -*- coding: utf-8 -*-
"""
认证中间件

提供JWT令牌验证和用户认证功能。
"""

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

# 导入模板库组件
from src.core.security import verify_token, decode_token
from src.core.logging import get_logger
from src.utils.exceptions import AuthenticationException, AuthorizationException

# 导入应用组件
from app.core.database import get_db
from app.models.user import User, UserStatus
from app.services.user_service import UserService

logger = get_logger(__name__)


class AuthMiddleware:
    """认证中间件类"""
    
    def __init__(self):
        self.security = HTTPBearer(auto_error=False)
        self.user_service = UserService()
    
    async def __call__(self, request: Request, call_next):
        """
        中间件处理函数
        """
        # 跳过不需要认证的路径
        if self._should_skip_auth(request.url.path):
            return await call_next(request)
        
        try:
            # 获取令牌
            token = await self._get_token_from_request(request)
            if not token:
                raise AuthenticationException("缺少认证令牌")
            
            # 验证令牌并获取用户
            user = await self._authenticate_user(request, token)
            if not user:
                raise AuthenticationException("无效的认证令牌")
            
            # 检查用户状态
            if user.status != UserStatus.active:
                raise AuthorizationException("用户账户已被禁用")
            
            # 将用户信息添加到请求中
            request.state.user = user
            request.state.user_id = user.id
            
            # 更新用户最后活动时间
            await self._update_last_activity(request, user.id)
            
            response = await call_next(request)
            return response
            
        except (AuthenticationException, AuthorizationException) as e:
            logger.warning(f"认证失败: {e} (Path: {request.url.path})")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"}
            )
        except Exception as e:
            logger.error(f"认证中间件错误: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="认证服务错误"
            )
    
    def _should_skip_auth(self, path: str) -> bool:
        """
        判断是否跳过认证
        """
        # 不需要认证的路径
        skip_paths = [
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/register",
            "/api/v1/auth/login",
            "/api/v1/auth/refresh",
            "/api/v1/auth/reset-password",
            "/api/v1/auth/verify-token",
            "/static"
        ]
        
        # 检查精确匹配
        if path in skip_paths:
            return True
        
        # 检查前缀匹配
        skip_prefixes = [
            "/static/",
            "/docs",
            "/redoc"
        ]
        
        for prefix in skip_prefixes:
            if path.startswith(prefix):
                return True
        
        return False
    
    async def _get_token_from_request(self, request: Request) -> Optional[str]:
        """
        从请求中获取令牌
        """
        # 从Authorization头获取
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            return authorization[7:]
        
        # 从查询参数获取（用于WebSocket）
        token = request.query_params.get("token")
        if token:
            return token
        
        # 从Cookie获取
        token = request.cookies.get("access_token")
        if token:
            return token
        
        return None
    
    async def _authenticate_user(self, request: Request, token: str) -> Optional[User]:
        """
        验证用户令牌
        """
        try:
            # 验证令牌
            if not verify_token(token):
                return None
            
            # 解码令牌获取用户信息
            payload = decode_token(token)
            if not payload:
                return None
            
            user_id = payload.get("sub")
            if not user_id:
                return None
            
            # 获取数据库会话
            db: Session = next(get_db())
            
            try:
                # 获取用户信息
                user = self.user_service.get_user_by_id(db, int(user_id))
                return user
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"用户认证失败: {e}")
            return None
    
    async def _update_last_activity(self, request: Request, user_id: int):
        """
        更新用户最后活动时间
        """
        try:
            db: Session = next(get_db())
            try:
                self.user_service.update_last_activity(db, user_id)
            finally:
                db.close()
        except Exception as e:
            # 更新活动时间失败不应该影响请求处理
            logger.warning(f"更新用户活动时间失败: {e}")


def get_current_user(request: Request) -> User:
    """
    获取当前认证用户
    """
    if not hasattr(request.state, "user") or not request.state.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未认证的用户"
        )
    return request.state.user


def get_current_user_id(request: Request) -> int:
    """
    获取当前用户ID
    """
    if not hasattr(request.state, "user_id") or not request.state.user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未认证的用户"
        )
    return request.state.user_id


def require_admin(request: Request) -> User:
    """
    要求管理员权限
    """
    user = get_current_user(request)
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限"
        )
    return user


def require_active_user(request: Request) -> User:
    """
    要求活跃用户
    """
    user = get_current_user(request)
    if user.status != UserStatus.active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户账户已被禁用"
        )
    return user


class OptionalAuthMiddleware:
    """可选认证中间件（不强制要求认证）"""
    
    def __init__(self):
        self.auth_middleware = AuthMiddleware()
    
    async def __call__(self, request: Request, call_next):
        """
        中间件处理函数
        """
        try:
            # 尝试认证，但不抛出异常
            token = await self.auth_middleware._get_token_from_request(request)
            if token:
                user = await self.auth_middleware._authenticate_user(request, token)
                if user and user.status == UserStatus.active:
                    request.state.user = user
                    request.state.user_id = user.id
                    await self.auth_middleware._update_last_activity(request, user.id)
            
            # 如果没有认证成功，设置为None
            if not hasattr(request.state, "user"):
                request.state.user = None
                request.state.user_id = None
            
            response = await call_next(request)
            return response
            
        except Exception as e:
            logger.warning(f"可选认证失败: {e}")
            request.state.user = None
            request.state.user_id = None
            response = await call_next(request)
            return response


def get_optional_current_user(request: Request) -> Optional[User]:
    """
    获取当前用户（可选）
    """
    return getattr(request.state, "user", None)


def get_optional_current_user_id(request: Request) -> Optional[int]:
    """
    获取当前用户ID（可选）
    """
    return getattr(request.state, "user_id",