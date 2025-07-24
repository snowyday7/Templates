# ChatGPT Backend 架构文档

## 📋 目录

- [系统概览](#系统概览)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [核心组件](#核心组件)
- [数据流](#数据流)
- [API 设计](#api-设计)
- [数据库设计](#数据库设计)
- [安全架构](#安全架构)
- [缓存策略](#缓存策略)
- [监控和日志](#监控和日志)
- [部署架构](#部署架构)
- [扩展性考虑](#扩展性考虑)

## 🏗️ 系统概览

### 架构图

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   前端客户端     │    │   移动应用       │    │   第三方集成     │
│   (React/Vue)   │    │   (iOS/Android) │    │   (API调用)     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      Nginx (可选)        │
                    │    反向代理 + 负载均衡     │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │     FastAPI 应用         │
                    │   ┌─────────────────┐    │
                    │   │   API 路由      │    │
                    │   │   中间件        │    │
                    │   │   WebSocket     │    │
                    │   │   异常处理      │    │
                    │   └─────────────────┘    │
                    └─────────────┬─────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                       │                       │
┌───────┴────────┐    ┌─────────┴─────────┐    ┌────────┴────────┐
│   PostgreSQL   │    │      Redis        │    │   OpenAI API    │
│   主数据库      │    │   缓存 + 会话     │    │   AI 服务       │
└────────────────┘    └───────────────────┘    └─────────────────┘
```

### 核心特性

- **RESTful API**: 标准的 REST 接口设计
- **WebSocket**: 实时双向通信
- **JWT 认证**: 无状态的用户认证
- **异步处理**: 基于 asyncio 的高性能处理
- **缓存机制**: Redis 缓存提升性能
- **限流保护**: 防止 API 滥用
- **监控指标**: Prometheus 指标收集
- **容器化**: Docker 容器化部署

## 🛠️ 技术栈

### 后端框架
- **FastAPI**: 现代、快速的 Web 框架
- **Uvicorn**: ASGI 服务器
- **Pydantic**: 数据验证和序列化

### 数据库
- **PostgreSQL**: 主数据库
- **SQLAlchemy**: ORM 框架
- **Alembic**: 数据库迁移工具

### 缓存
- **Redis**: 内存数据库，用于缓存和会话存储

### 认证和安全
- **JWT**: JSON Web Tokens
- **bcrypt**: 密码哈希
- **python-jose**: JWT 处理

### 外部服务
- **OpenAI API**: GPT 模型服务
- **SMTP**: 邮件服务（可选）

### 监控和日志
- **Prometheus**: 指标收集
- **Grafana**: 监控面板（可选）
- **Python logging**: 应用日志

### 部署
- **Docker**: 容器化
- **Docker Compose**: 多容器编排
- **Nginx**: 反向代理（可选）

## 📁 项目结构

```
chatgpt-backend/
├── app/                          # 应用主目录
│   ├── __init__.py
│   ├── main.py                   # 应用入口
│   ├── api/                      # API 路由
│   │   ├── __init__.py
│   │   ├── deps.py              # 依赖注入
│   │   └── v1/                  # API v1 版本
│   │       ├── __init__.py
│   │       ├── auth.py          # 认证路由
│   │       ├── users.py         # 用户管理
│   │       ├── conversations.py # 对话管理
│   │       ├── messages.py      # 消息处理
│   │       └── websocket.py     # WebSocket 路由
│   ├── core/                    # 核心组件
│   │   ├── __init__.py
│   │   ├── config.py           # 配置管理
│   │   ├── database.py         # 数据库连接
│   │   ├── security.py         # 安全工具
│   │   └── cache.py            # 缓存管理
│   ├── models/                  # 数据模型
│   │   ├── __init__.py
│   │   ├── user.py             # 用户模型
│   │   ├── conversation.py     # 对话模型
│   │   ├── message.py          # 消息模型
│   │   └── quota.py            # 配额模型
│   ├── schemas/                 # Pydantic 模式
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── conversation.py
│   │   ├── message.py
│   │   └── common.py
│   ├── services/                # 业务逻辑
│   │   ├── __init__.py
│   │   ├── auth_service.py
│   │   ├── user_service.py
│   │   ├── conversation_service.py
│   │   ├── message_service.py
│   │   ├── openai_service.py
│   │   └── quota_service.py
│   ├── middleware/              # 中间件
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── cors.py
│   │   ├── rate_limit.py
│   │   └── logging.py
│   ├── exceptions/              # 异常处理
│   │   ├── __init__.py
│   │   ├── custom_exceptions.py
│   │   └── handlers.py
│   └── utils/                   # 工具函数
│       ├── __init__.py
│       ├── helpers.py
│       ├── validators.py
│       └── formatters.py
├── scripts/                     # 脚本文件
│   ├── init_db.py              # 数据库初始化
│   └── backup.py               # 备份脚本
├── tests/                       # 测试文件
├── docs/                        # 文档
├── docker/                      # Docker 配置
├── nginx/                       # Nginx 配置
├── monitoring/                  # 监控配置
├── requirements.txt             # Python 依赖
├── .env.example                # 环境变量示例
├── docker-compose.yml          # 生产环境
├── docker-compose.dev.yml      # 开发环境
└── README.md                   # 项目说明
```

## 🔧 核心组件

### 1. 应用入口 (main.py)

```python
# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化
    await init_database()
    await init_cache()
    yield
    # 关闭时清理
    await cleanup_resources()

# FastAPI 应用实例
app = FastAPI(
    title="ChatGPT Backend API",
    lifespan=lifespan
)
```

### 2. 配置管理 (config.py)

```python
class Settings(BaseSettings):
    # 应用配置
    app_name: str = "ChatGPT Backend"
    environment: str = "development"
    
    # 数据库配置
    database_url: str
    
    # OpenAI 配置
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    
    # 安全配置
    secret_key: str
    access_token_expire_minutes: int = 30
```

### 3. 数据库管理 (database.py)

```python
class Database:
    def __init__(self):
        self.engine = None
        self.session_factory = None
    
    async def init_database(self):
        # 创建数据库引擎
        # 设置会话工厂
        # 创建表结构
    
    async def get_session(self):
        # 返回数据库会话
```

### 4. 缓存管理 (cache.py)

```python
class Cache:
    async def get(self, key: str) -> Optional[str]:
        # 获取缓存值
    
    async def set(self, key: str, value: str, ttl: int = 3600):
        # 设置缓存值
    
    async def delete(self, key: str):
        # 删除缓存
```

### 5. 安全管理 (security.py)

```python
class Security:
    def hash_password(self, password: str) -> str:
        # 密码哈希
    
    def verify_password(self, password: str, hashed: str) -> bool:
        # 密码验证
    
    def create_access_token(self, data: dict) -> str:
        # 创建 JWT 令牌
    
    def verify_token(self, token: str) -> dict:
        # 验证 JWT 令牌
```

## 🔄 数据流

### 1. 用户认证流程

```
客户端 → 登录请求 → 验证凭据 → 生成 JWT → 返回令牌
       ↓
后续请求 → 验证 JWT → 提取用户信息 → 处理请求
```

### 2. 对话消息流程

```
客户端 → 发送消息 → 验证权限 → 保存消息 → 调用 OpenAI → 保存回复 → 返回结果
       ↓
WebSocket → 实时推送 → 客户端接收
```

### 3. 缓存策略

```
请求 → 检查缓存 → 缓存命中？ → 返回缓存数据
     ↓              ↓
   缓存未命中 → 查询数据库 → 更新缓存 → 返回数据
```

## 🗄️ 数据库设计

### 核心表结构

```sql
-- 用户表
CREATE TABLE users (
    id UUID PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_vip BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 对话表
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    title VARCHAR(200),
    model VARCHAR(50) NOT NULL,
    is_archived BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 消息表
CREATE TABLE messages (
    id UUID PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id),
    role VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    tokens_used INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 用户配额表
CREATE TABLE user_quotas (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    daily_requests INTEGER DEFAULT 0,
    monthly_requests INTEGER DEFAULT 0,
    daily_tokens INTEGER DEFAULT 0,
    monthly_tokens INTEGER DEFAULT 0,
    last_reset_date DATE DEFAULT CURRENT_DATE
);
```

### 索引策略

```sql
-- 性能优化索引
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
CREATE INDEX idx_user_quotas_user_id ON user_quotas(user_id);
```

## 🔒 安全架构

### 1. 认证机制

- **JWT 令牌**: 无状态认证
- **刷新令牌**: 长期会话管理
- **密码哈希**: bcrypt 加密存储

### 2. 授权控制

```python
# 权限装饰器
@require_auth
@require_active_user
@check_quota
async def protected_endpoint():
    pass
```

### 3. 安全中间件

- **CORS**: 跨域请求控制
- **Rate Limiting**: 请求频率限制
- **Input Validation**: 输入数据验证
- **SQL Injection Prevention**: ORM 防护

### 4. 数据保护

- **敏感数据加密**: 关键信息加密存储
- **API Key 管理**: 安全的密钥存储
- **日志脱敏**: 敏感信息过滤

## 💾 缓存策略

### 1. 缓存层级

```
应用缓存 (内存) → Redis 缓存 → 数据库
```

### 2. 缓存类型

- **用户会话**: 30分钟 TTL
- **对话列表**: 5分钟 TTL
- **模型配置**: 1小时 TTL
- **用户配额**: 实时更新

### 3. 缓存键设计

```python
# 缓存键命名规范
user_session = f"session:{user_id}"
user_conversations = f"conversations:{user_id}"
user_quota = f"quota:{user_id}:{date}"
model_config = f"model:{model_name}"
```

## 📊 监控和日志

### 1. 应用指标

```python
# Prometheus 指标
request_count = Counter('http_requests_total')
request_duration = Histogram('http_request_duration_seconds')
openai_requests = Counter('openai_requests_total')
active_connections = Gauge('websocket_connections_active')
```

### 2. 日志级别

- **DEBUG**: 详细调试信息
- **INFO**: 一般操作信息
- **WARNING**: 警告信息
- **ERROR**: 错误信息
- **CRITICAL**: 严重错误

### 3. 日志格式

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "logger": "app.services.openai",
  "message": "OpenAI API call completed",
  "user_id": "uuid",
  "request_id": "uuid",
  "duration": 1.23,
  "tokens_used": 150
}
```

## 🚀 部署架构

### 1. 开发环境

```yaml
# docker-compose.dev.yml
services:
  app: # 应用服务 + 热重载
  db: # PostgreSQL
  redis: # Redis
  adminer: # 数据库管理
  redis-commander: # Redis 管理
```

### 2. 生产环境

```yaml
# docker-compose.yml
services:
  app: # 应用服务
  db: # PostgreSQL + 持久化
  redis: # Redis + 持久化
  nginx: # 反向代理
  prometheus: # 监控
  grafana: # 仪表板
```

### 3. 高可用部署

```
负载均衡器 (Nginx/HAProxy)
    ↓
应用实例 1, 2, 3... (Docker Swarm/Kubernetes)
    ↓
数据库集群 (PostgreSQL Master/Slave)
    ↓
Redis 集群 (Redis Cluster/Sentinel)
```

## 📈 扩展性考虑

### 1. 水平扩展

- **无状态设计**: 应用实例可任意扩展
- **数据库分片**: 按用户 ID 分片
- **缓存集群**: Redis 集群模式

### 2. 垂直扩展

- **资源监控**: CPU、内存、磁盘使用率
- **性能调优**: 数据库查询优化
- **连接池**: 数据库连接池管理

### 3. 微服务拆分

```
当前单体架构 → 微服务架构

用户服务 (User Service)
对话服务 (Conversation Service)
AI 服务 (AI Service)
通知服务 (Notification Service)
```

### 4. 性能优化

- **异步处理**: 所有 I/O 操作异步化
- **连接复用**: HTTP 连接池
- **批量操作**: 数据库批量插入/更新
- **CDN**: 静态资源分发

## 🔧 开发最佳实践

### 1. 代码组织

- **单一职责**: 每个模块专注单一功能
- **依赖注入**: 使用 FastAPI 的依赖系统
- **类型提示**: 完整的类型注解
- **文档字符串**: 详细的函数文档

### 2. 错误处理

- **自定义异常**: 业务相关的异常类
- **全局异常处理**: 统一的错误响应格式
- **日志记录**: 详细的错误日志

### 3. 测试策略

- **单元测试**: 核心业务逻辑测试
- **集成测试**: API 端点测试
- **性能测试**: 负载和压力测试
- **安全测试**: 安全漏洞扫描

### 4. 代码质量

- **代码格式化**: Black, isort
- **静态分析**: mypy, flake8
- **代码审查**: Pull Request 流程
- **持续集成**: GitHub Actions/GitLab CI

这个架构设计确保了系统的可扩展性、可维护性和高性能，为 ChatGPT 后端服务提供了坚实的技术基础。