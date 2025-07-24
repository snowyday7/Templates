# ChatGPT Backend

一个功能完整的ChatGPT客户端后端API服务，基于FastAPI构建，提供用户管理、对话管理、消息处理、实时通信等功能。

## 功能特性

### 🔐 用户认证与授权
- JWT令牌认证
- 用户注册、登录、密码重置
- 基于角色的访问控制（RBAC）
- API密钥管理

### 💬 对话管理
- 创建、更新、删除对话
- 对话历史记录
- 对话分享功能
- 对话搜索和过滤

### 📝 消息处理
- 支持多种消息类型（文本、图片等）
- 消息流式响应
- 消息重新生成
- 消息搜索和导出

### 🌐 实时通信
- WebSocket支持
- 实时消息推送
- 连接状态管理
- 心跳检测

### 📊 配额管理
- 用户请求限制
- Token使用统计
- 使用量报告
- VIP用户支持

### 🛡️ 安全特性
- 请求限流
- CORS配置
- 数据加密
- 安全日志记录

### 📈 监控与日志
- 健康检查端点
- 性能监控
- 详细日志记录
- 指标收集

## 技术栈

- **框架**: FastAPI
- **数据库**: SQLAlchemy (支持SQLite、PostgreSQL、MySQL)
- **缓存**: Redis (可选，支持内存缓存)
- **认证**: JWT + Passlib
- **AI服务**: OpenAI API
- **异步**: asyncio + uvicorn
- **日志**: 结构化日志
- **测试**: pytest

## 快速开始

### 1. 环境要求

- Python 3.8+
- Redis (可选，用于缓存)
- PostgreSQL/MySQL (可选，默认使用SQLite)

### 2. 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd chatgpt-backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境

```bash
# 复制环境配置文件
cp .env.example .env

# 编辑配置文件
vim .env
```

**重要配置项：**

```env
# 必须配置
SECRET_KEY="your-super-secret-key"
OPENAI_API_KEY="sk-your-openai-api-key"

# 数据库配置（可选，默认SQLite）
DATABASE_URL="sqlite:///./chatgpt_backend.db"

# Redis配置（可选）
REDIS_URL="redis://localhost:6379/0"
```

### 4. 初始化数据库

```bash
# 运行数据库迁移
python -m app.scripts.init_db
```

### 5. 启动服务

```bash
# 开发环境
python -m app.main

# 或使用uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 生产环境
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 6. 访问API文档

启动服务后，访问以下地址查看API文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- 健康检查: http://localhost:8000/health

## API 使用示例

### 用户注册

```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "password123"
  }'
```

### 用户登录

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "password123"
  }'
```

### 创建对话

```bash
curl -X POST "http://localhost:8000/api/v1/conversations" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "我的第一个对话",
    "model": "gpt-3.5-turbo"
  }'
```

### 发送消息

```bash
curl -X POST "http://localhost:8000/api/v1/messages" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": 1,
    "content": "你好，请介绍一下自己",
    "role": "user"
  }'
```

### WebSocket连接

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/YOUR_ACCESS_TOKEN');

ws.onopen = function(event) {
    console.log('WebSocket连接已建立');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('收到消息:', data);
};

// 发送消息
ws.send(JSON.stringify({
    type: 'chat_message',
    conversation_id: 1,
    content: '你好！',
    model: 'gpt-3.5-turbo'
}));
```

## 项目结构

```
chatgpt-backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── auth.py          # 认证路由
│   │       ├── conversations.py # 对话路由
│   │       ├── messages.py      # 消息路由
│   │       ├── users.py         # 用户路由
│   │       └── websocket.py     # WebSocket路由
│   ├── core/
│   │   ├── config.py           # 配置管理
│   │   ├── database.py         # 数据库连接
│   │   ├── security.py         # 安全功能
│   │   └── cache.py            # 缓存管理
│   ├── models/
│   │   ├── user.py             # 用户模型
│   │   ├── conversation.py     # 对话模型
│   │   ├── message.py          # 消息模型
│   │   └── usage.py            # 使用量模型
│   ├── schemas/
│   │   ├── user.py             # 用户模式
│   │   ├── conversation.py     # 对话模式
│   │   └── message.py          # 消息模式
│   ├── services/
│   │   ├── user_service.py     # 用户服务
│   │   ├── conversation_service.py # 对话服务
│   │   ├── message_service.py  # 消息服务
│   │   └── usage_service.py    # 使用量服务
│   ├── middleware/
│   │   ├── auth.py             # 认证中间件
│   │   ├── rate_limit.py       # 限流中间件
│   │   └── cors.py             # CORS中间件
│   ├── exceptions/
│   │   ├── custom_exceptions.py # 自定义异常
│   │   └── handlers.py         # 异常处理器
│   └── main.py                 # 应用入口
├── tests/                      # 测试文件
├── docs/                       # 文档
├── scripts/                    # 脚本文件
├── requirements.txt            # 依赖列表
├── .env.example               # 环境配置示例
└── README.md                  # 项目说明
```

## 配置说明

### 数据库配置

支持多种数据库：

```env
# SQLite (默认)
DATABASE_URL="sqlite:///./chatgpt_backend.db"

# PostgreSQL
DATABASE_URL="postgresql://user:password@localhost:5432/chatgpt_backend"

# MySQL
DATABASE_URL="mysql+pymysql://user:password@localhost:3306/chatgpt_backend"
```

### Redis配置

```env
# 单机Redis
REDIS_URL="redis://localhost:6379/0"

# Redis集群
REDIS_URL="redis://localhost:6379,localhost:6380,localhost:6381/0"

# Redis with密码
REDIS_URL="redis://:password@localhost:6379/0"
```

### OpenAI配置

```env
# 官方API
OPENAI_API_KEY="sk-your-api-key"
OPENAI_API_BASE="https://api.openai.com/v1"

# 自定义端点
OPENAI_API_BASE="https://your-custom-endpoint.com/v1"
```

## 部署

### Docker部署

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/chatgpt_backend
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=chatgpt_backend
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### 生产环境部署

1. **使用Gunicorn + Nginx**

```bash
# 安装Gunicorn
pip install gunicorn

# 启动服务
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

2. **Nginx配置**

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_auth.py

# 生成覆盖率报告
pytest --cov=app --cov-report=html
```

## 开发

### 代码格式化

```bash
# 格式化代码
black app/
isort app/

# 检查代码质量
flake8 app/
mypy app/
```

### 数据库迁移

```bash
# 生成迁移文件
alembic revision --autogenerate -m "Add new table"

# 执行迁移
alembic upgrade head

# 回滚迁移
alembic downgrade -1
```

## 监控

### 健康检查

```bash
curl http://localhost:8000/health
```

### 应用信息

```bash
curl http://localhost:8000/info
```

### 指标收集

```bash
curl http://localhost:8000/metrics
```

## 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查数据库URL配置
   - 确认数据库服务正在运行
   - 检查网络连接

2. **Redis连接失败**
   - 检查Redis URL配置
   - 确认Redis服务正在运行
   - 应用会自动降级到内存缓存

3. **OpenAI API错误**
   - 检查API密钥是否正确
   - 确认账户有足够余额
   - 检查网络连接

4. **JWT令牌错误**
   - 检查SECRET_KEY配置
   - 确认令牌未过期
   - 检查令牌格式

### 日志查看

```bash
# 查看应用日志
tail -f logs/app.log

# 查看错误日志
grep ERROR logs/app.log
```

## 贡献

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 支持

如果您遇到问题或有疑问，请：

1. 查看文档和FAQ
2. 搜索现有的Issues
3. 创建新的Issue
4. 联系维护者

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 基础用户认证功能
- 对话和消息管理
- WebSocket实时通信
- OpenAI API集成
- 配额管理系统
- 监控和日志功能