"""FastAPI应用模板

提供完整的FastAPI应用开发模板，包括：
- 应用配置和初始化
- 路由和端点定义
- 中间件配置
- 依赖注入
- 错误处理
- API文档配置
"""

from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    APIRouter,
    Depends,
    HTTPException,
    status,
    Request,
    Response,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import uvicorn
import logging
import time
import uuid


class APISettings(BaseSettings):
    """API配置类"""

    # 应用基础配置
    APP_NAME: str = "FastAPI Application"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "A FastAPI application template"
    DEBUG: bool = False

    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False

    # 安全配置
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"

    # CORS配置
    ALLOWED_ORIGINS: List[str] = ["*"]
    ALLOWED_METHODS: List[str] = ["*"]
    ALLOWED_HEADERS: List[str] = ["*"]

    # 限流配置
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60

    # 数据库配置
    DATABASE_URL: str = "sqlite:///./app.db"

    class Config:
        env_file = ".env"


class BaseResponse(BaseModel):
    """基础响应模型"""

    success: bool = True
    message: str = "Success"
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """错误响应模型"""

    success: bool = False
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class PaginationParams(BaseModel):
    """分页参数模型"""

    page: int = Field(1, ge=1, description="页码")
    size: int = Field(10, ge=1, le=100, description="每页大小")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


class PaginatedResponse(BaseModel):
    """分页响应模型"""

    items: List[Any]
    total: int
    page: int
    size: int
    pages: int

    @validator("pages", always=True)
    def calculate_pages(cls, v, values):
        total = values.get("total", 0)
        size = values.get("size", 10)
        return (total + size - 1) // size if total > 0 else 0


class RequestMiddleware:
    """请求中间件"""

    def __init__(self, app: FastAPI):
        self.app = app

    async def __call__(self, request: Request, call_next: Callable):
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # 记录请求开始时间
        start_time = time.time()

        # 记录请求日志
        logging.info(
            f"Request started: {request.method} {request.url} " f"[{request_id}]"
        )

        try:
            # 处理请求
            response = await call_next(request)

            # 计算处理时间
            process_time = time.time() - start_time

            # 添加响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)

            # 记录响应日志
            logging.info(
                f"Request completed: {request.method} {request.url} "
                f"[{request_id}] - {response.status_code} - {process_time:.3f}s"
            )

            return response

        except Exception as e:
            # 记录错误日志
            process_time = time.time() - start_time
            logging.error(
                f"Request failed: {request.method} {request.url} "
                f"[{request_id}] - {str(e)} - {process_time:.3f}s"
            )
            raise


class RateLimitMiddleware:
    """限流中间件"""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = {}

    async def __call__(self, request: Request, call_next: Callable):
        client_ip = request.client.host
        current_time = time.time()

        # 清理过期记录
        self.requests = {
            ip: times
            for ip, times in self.requests.items()
            if any(t > current_time - 60 for t in times)
        }

        # 检查当前IP的请求次数
        if client_ip in self.requests:
            recent_requests = [
                t for t in self.requests[client_ip] if t > current_time - 60
            ]

            if len(recent_requests) >= self.requests_per_minute:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                )

            self.requests[client_ip] = recent_requests + [current_time]
        else:
            self.requests[client_ip] = [current_time]

        return await call_next(request)


class FastAPIApp:
    """FastAPI应用包装器"""

    def __init__(self, settings: Optional[APISettings] = None):
        self.settings = settings or APISettings()
        self.app = None
        self._setup_logging()
        self._create_app()

    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.DEBUG if self.settings.DEBUG else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """应用生命周期管理"""
        # 启动时执行
        logging.info(f"Starting {self.settings.APP_NAME} v{self.settings.APP_VERSION}")
        yield
        # 关闭时执行
        logging.info(f"Shutting down {self.settings.APP_NAME}")

    def _create_app(self):
        """创建FastAPI应用"""
        self.app = FastAPI(
            title=self.settings.APP_NAME,
            version=self.settings.APP_VERSION,
            description=self.settings.APP_DESCRIPTION,
            debug=self.settings.DEBUG,
            lifespan=self.lifespan,
        )

        # 添加中间件
        self._add_middleware()

        # 添加异常处理器
        self._add_exception_handlers()

        # 添加路由
        self._add_routes()

    def _add_middleware(self):
        """添加中间件"""
        # CORS中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=self.settings.ALLOWED_METHODS,
            allow_headers=self.settings.ALLOWED_HEADERS,
        )

        # 信任主机中间件
        if not self.settings.DEBUG:
            self.app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

        # 自定义中间件
        self.app.middleware("http")(RequestMiddleware(self.app))

    def _add_exception_handlers(self):
        """添加异常处理器"""

        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                    error_code=str(exc.status_code),
                    error_message=exc.detail,
                    request_id=getattr(request.state, "request_id", None),
                ).dict(),
            )

        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error_code="INTERNAL_SERVER_ERROR",
                    error_message="Internal server error",
                    request_id=getattr(request.state, "request_id", None),
                ).dict(),
            )

    def _add_routes(self):
        """添加基础路由"""

        @self.app.get("/", response_model=BaseResponse)
        async def root():
            return BaseResponse(
                message=f"Welcome to {self.settings.APP_NAME}",
                data={"version": self.settings.APP_VERSION},
            )

        @self.app.get("/health", response_model=BaseResponse)
        async def health_check():
            return BaseResponse(
                message="Service is healthy",
                data={
                    "status": "ok",
                    "timestamp": datetime.utcnow(),
                    "version": self.settings.APP_VERSION,
                },
            )

    def include_router(self, router: APIRouter, **kwargs):
        """包含路由"""
        self.app.include_router(router, **kwargs)

    def run(self, **kwargs):
        """运行应用"""
        config = {
            "app": self.app,
            "host": self.settings.HOST,
            "port": self.settings.PORT,
            "reload": self.settings.RELOAD,
            **kwargs,
        }
        uvicorn.run(**config)


# 认证相关
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """获取当前用户（示例实现）"""
    # 这里应该实现JWT token验证逻辑
    # 示例实现
    token = credentials.credentials

    # 验证token（这里需要实际的JWT验证逻辑）
    if not token or token == "invalid":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 返回用户信息
    return {"user_id": 1, "username": "testuser", "email": "test@example.com"}


def create_pagination_dependency():
    """创建分页依赖"""

    def get_pagination(page: int = 1, size: int = 10) -> PaginationParams:
        return PaginationParams(page=page, size=size)

    return get_pagination


# 示例路由
def create_user_router() -> APIRouter:
    """创建用户路由示例"""
    router = APIRouter(prefix="/users", tags=["users"])

    class UserCreate(BaseModel):
        username: str = Field(..., min_length=3, max_length=50)
        email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
        full_name: Optional[str] = None

    class UserResponse(BaseModel):
        id: int
        username: str
        email: str
        full_name: Optional[str]
        created_at: datetime
        is_active: bool

    @router.post("/", response_model=BaseResponse)
    async def create_user(
        user_data: UserCreate, current_user: Dict = Depends(get_current_user)
    ):
        # 创建用户逻辑
        user = {
            "id": 1,
            "username": user_data.username,
            "email": user_data.email,
            "full_name": user_data.full_name,
            "created_at": datetime.utcnow(),
            "is_active": True,
        }

        return BaseResponse(message="User created successfully", data=user)

    @router.get("/", response_model=BaseResponse)
    async def get_users(
        pagination: PaginationParams = Depends(create_pagination_dependency()),
        current_user: Dict = Depends(get_current_user),
    ):
        # 获取用户列表逻辑
        users = [
            {
                "id": i,
                "username": f"user{i}",
                "email": f"user{i}@example.com",
                "full_name": f"User {i}",
                "created_at": datetime.utcnow(),
                "is_active": True,
            }
            for i in range(1, 6)
        ]

        paginated_data = PaginatedResponse(
            items=users, total=100, page=pagination.page, size=pagination.size
        )

        return BaseResponse(
            message="Users retrieved successfully", data=paginated_data.dict()
        )

    @router.get("/{user_id}", response_model=BaseResponse)
    async def get_user(user_id: int, current_user: Dict = Depends(get_current_user)):
        # 获取单个用户逻辑
        if user_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        user = {
            "id": user_id,
            "username": f"user{user_id}",
            "email": f"user{user_id}@example.com",
            "full_name": f"User {user_id}",
            "created_at": datetime.utcnow(),
            "is_active": True,
        }

        return BaseResponse(message="User retrieved successfully", data=user)

    return router


def setup_cors(app: FastAPI, settings: Optional[APISettings] = None) -> None:
    """设置CORS中间件"""
    if settings is None:
        settings = APISettings()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=settings.ALLOWED_METHODS,
        allow_headers=settings.ALLOWED_HEADERS,
    )


def setup_middleware(app: FastAPI, settings: Optional[APISettings] = None) -> None:
    """设置所有中间件"""
    if settings is None:
        settings = APISettings()
    
    # 添加CORS中间件
    setup_cors(app, settings)
    
    # 添加信任主机中间件
    if not settings.DEBUG:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    
    # 添加请求中间件
    app.middleware("http")(RequestMiddleware(app))
    
    # 添加限流中间件
    app.middleware("http")(RateLimitMiddleware(settings.RATE_LIMIT_REQUESTS))


def create_fastapi_app(settings: Optional[APISettings] = None) -> FastAPI:
    """创建FastAPI应用的工厂函数"""
    app_wrapper = FastAPIApp(settings)

    # 添加用户路由
    user_router = create_user_router()
    app_wrapper.include_router(user_router, prefix="/api/v1")

    return app_wrapper.app


# 使用示例
if __name__ == "__main__":
    # 创建设置
    settings = APISettings(APP_NAME="My FastAPI App", DEBUG=True, RELOAD=True)

    # 创建应用
    app_wrapper = FastAPIApp(settings)

    # 添加路由
    user_router = create_user_router()
    app_wrapper.include_router(user_router, prefix="/api/v1")

    # 运行应用
    app_wrapper.run()
