# API开发模块使用指南

API开发模块提供了基于FastAPI的完整Web API开发框架，包括路由管理、中间件、异常处理、文档生成等功能。

## 📋 目录

- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [基础使用](#基础使用)
- [高级功能](#高级功能)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

## 🚀 快速开始

### 安装依赖

```bash
pip install fastapi uvicorn python-multipart
```

### 创建基础应用

```python
from templates.api import FastAPIApp, APISettings

# 创建API配置
settings = APISettings(
    APP_NAME="My API",
    APP_VERSION="1.0.0",
    DEBUG=True
)

# 创建FastAPI应用
app = FastAPIApp(settings)

# 获取FastAPI实例
fastapi_app = app.get_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
```

## ⚙️ 配置说明

### APISettings 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `APP_NAME` | str | "FastAPI App" | 应用名称 |
| `APP_VERSION` | str | "1.0.0" | 应用版本 |
| `APP_DESCRIPTION` | str | "" | 应用描述 |
| `DEBUG` | bool | False | 调试模式 |
| `API_PREFIX` | str | "/api/v1" | API前缀 |
| `DOCS_URL` | str | "/docs" | 文档URL |
| `REDOC_URL` | str | "/redoc" | ReDoc URL |
| `OPENAPI_URL` | str | "/openapi.json" | OpenAPI JSON URL |
| `CORS_ORIGINS` | list | ["*"] | CORS允许的源 |
| `CORS_METHODS` | list | ["*"] | CORS允许的方法 |
| `CORS_HEADERS` | list | ["*"] | CORS允许的头部 |
| `MAX_REQUEST_SIZE` | int | 16777216 | 最大请求大小(字节) |
| `REQUEST_TIMEOUT` | int | 30 | 请求超时时间(秒) |

### 环境变量配置

创建 `.env` 文件：

```env
APP_NAME=My API
APP_VERSION=1.0.0
APP_DESCRIPTION=My awesome API
DEBUG=true
API_PREFIX=/api/v1
CORS_ORIGINS=["http://localhost:3000", "https://myapp.com"]
MAX_REQUEST_SIZE=16777216
REQUEST_TIMEOUT=30
```

## 💻 基础使用

### 1. 创建路由

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List

# 创建路由器
router = APIRouter(prefix="/users", tags=["users"])

# 定义数据模型
class UserCreate(BaseModel):
    username: str
    email: str
    full_name: str = None

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: str = None
    is_active: bool = True

# 定义路由
@router.post("/", response_model=UserResponse)
async def create_user(user: UserCreate):
    # 创建用户逻辑
    return {"id": 1, **user.dict()}

@router.get("/", response_model=List[UserResponse])
async def get_users(skip: int = 0, limit: int = 100):
    # 获取用户列表逻辑
    return [{"id": 1, "username": "test", "email": "test@example.com"}]

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    if user_id == 1:
        return {"id": 1, "username": "test", "email": "test@example.com"}
    raise HTTPException(status_code=404, detail="User not found")

# 注册路由
app.include_router(router)
```

### 2. 中间件使用

```python
from templates.api import RequestLoggingMiddleware, CORSMiddleware
import time

# 添加请求日志中间件
app.add_middleware(RequestLoggingMiddleware)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 自定义中间件
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

### 3. 异常处理

```python
from templates.api import APIException, ErrorResponse
from fastapi import Request
from fastapi.responses import JSONResponse

# 自定义异常
class UserNotFoundError(APIException):
    def __init__(self, user_id: int):
        super().__init__(
            status_code=404,
            detail=f"User with id {user_id} not found",
            error_code="USER_NOT_FOUND"
        )

# 异常处理器
@app.exception_handler(UserNotFoundError)
async def user_not_found_handler(request: Request, exc: UserNotFoundError):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.error_code,
            message=exc.detail,
            details=exc.details
        ).dict()
    )

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            details=str(exc) if app.settings.DEBUG else None
        ).dict()
    )
```

## 🔧 高级功能

### 1. 依赖注入

```python
from fastapi import Depends
from templates.database import DatabaseManager, get_db_session
from templates.auth import get_current_user

# 数据库依赖
def get_db():
    db = DatabaseManager()
    try:
        with db.get_session() as session:
            yield session
    finally:
        pass

# 认证依赖
def get_current_active_user(current_user = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# 使用依赖
@router.get("/profile", response_model=UserResponse)
async def get_user_profile(
    current_user = Depends(get_current_active_user),
    db = Depends(get_db)
):
    return current_user
```

### 2. 请求验证

```python
from pydantic import BaseModel, validator, Field
from typing import Optional
import re

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None, max_length=100)
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Username must be alphanumeric')
        return v
    
    @validator('password')
    def password_strength(cls, v):
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain digit')
        return v
```

### 3. 文件上传

```python
from fastapi import File, UploadFile, Form
from templates.api import FileUploadManager
import shutil
from pathlib import Path

# 文件上传管理器
file_manager = FileUploadManager(
    upload_dir="uploads",
    max_file_size=10 * 1024 * 1024,  # 10MB
    allowed_extensions=[".jpg", ".png", ".pdf"]
)

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Form(None)
):
    # 验证文件
    file_manager.validate_file(file)
    
    # 保存文件
    file_path = await file_manager.save_file(file)
    
    return {
        "filename": file.filename,
        "file_path": str(file_path),
        "description": description,
        "size": file.size
    }

@router.post("/upload-multiple")
async def upload_multiple_files(
    files: List[UploadFile] = File(...)
):
    results = []
    for file in files:
        file_manager.validate_file(file)
        file_path = await file_manager.save_file(file)
        results.append({
            "filename": file.filename,
            "file_path": str(file_path)
        })
    return {"files": results}
```

### 4. 响应缓存

```python
from templates.api import CacheManager
from fastapi import Depends
import json

# 缓存管理器
cache_manager = CacheManager()

# 缓存装饰器
def cache_response(expire_time: int = 300):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # 尝试从缓存获取
            cached_result = await cache_manager.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 存储到缓存
            await cache_manager.set(
                cache_key, 
                json.dumps(result, default=str), 
                expire_time
            )
            
            return result
        return wrapper
    return decorator

# 使用缓存
@router.get("/users/{user_id}")
@cache_response(expire_time=600)  # 缓存10分钟
async def get_user_cached(user_id: int):
    # 模拟数据库查询
    await asyncio.sleep(1)
    return {"id": user_id, "username": f"user_{user_id}"}
```

## 📝 最佳实践

### 1. 项目结构

```
api/
├── __init__.py
├── main.py              # 应用入口
├── core/
│   ├── __init__.py
│   ├── config.py        # 配置
│   ├── security.py      # 安全相关
│   └── dependencies.py  # 依赖注入
├── routers/
│   ├── __init__.py
│   ├── users.py         # 用户路由
│   ├── auth.py          # 认证路由
│   └── admin.py         # 管理路由
├── models/
│   ├── __init__.py
│   ├── user.py          # 用户模型
│   └── base.py          # 基础模型
├── schemas/
│   ├── __init__.py
│   ├── user.py          # 用户模式
│   └── common.py        # 通用模式
└── services/
    ├── __init__.py
    ├── user_service.py  # 用户服务
    └── email_service.py # 邮件服务
```

### 2. 错误处理

```python
# 统一错误响应格式
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    path: Optional[str] = None

# 业务异常基类
class BusinessException(Exception):
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code or "BUSINESS_ERROR"
        super().__init__(self.message)

# 具体业务异常
class UserAlreadyExistsError(BusinessException):
    def __init__(self, username: str):
        super().__init__(
            message=f"User '{username}' already exists",
            error_code="USER_ALREADY_EXISTS"
        )
```

### 3. 数据验证

```python
# 使用Pydantic进行数据验证
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = Field(None, max_length=100)
    
    class Config:
        # 允许从ORM对象创建
        from_attributes = True
        # JSON编码配置
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    
class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    
class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
```

### 4. 性能优化

```python
# 使用异步操作
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 异步数据库操作
async def get_users_async(db, skip: int = 0, limit: int = 100):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor, 
            lambda: db.query(User).offset(skip).limit(limit).all()
        )
    return result

# 批量操作
@router.post("/users/batch")
async def create_users_batch(users: List[UserCreate]):
    tasks = [create_user_async(user) for user in users]
    results = await asyncio.gather(*tasks)
    return {"created_users": results}

# 响应压缩
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

## ❓ 常见问题

### Q: 如何处理跨域请求？

A: 配置CORS中间件：

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://myapp.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

### Q: 如何实现API版本控制？

A: 使用路由前缀：

```python
# v1 API
v1_router = APIRouter(prefix="/api/v1")

# v2 API
v2_router = APIRouter(prefix="/api/v2")

app.include_router(v1_router)
app.include_router(v2_router)
```

### Q: 如何限制请求频率？

A: 使用限流中间件：

```python
from templates.api import RateLimitMiddleware

app.add_middleware(
    RateLimitMiddleware,
    calls=100,  # 100次请求
    period=60   # 每60秒
)
```

### Q: 如何生成API文档？

A: FastAPI自动生成文档，访问 `/docs` 或 `/redoc`：

```python
# 自定义文档信息
app = FastAPI(
    title="My API",
    description="This is my awesome API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
```

## 📚 相关文档

- [数据库模块使用指南](database.md)
- [认证授权模块使用指南](auth.md)
- [监控日志模块使用指南](monitoring.md)
- [安全开发指南](best-practices/security.md)
- [性能优化建议](best-practices/performance.md)

---

如有其他问题，请查看 [GitHub Issues](https://github.com/your-username/python-backend-templates/issues) 或提交新的问题。