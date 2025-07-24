# ChatGPT Backend 快速上手指南

本指南将帮助您快速搭建和使用 ChatGPT 后端服务。

## 📋 前置要求

### 必需
- **Docker** 和 **Docker Compose**
- **OpenAI API Key**

### 可选
- Python 3.11+ (本地开发)
- PostgreSQL (本地开发)
- Redis (本地开发)

## 🚀 快速开始

### 方式一：使用快速启动脚本（推荐）

1. **克隆项目**
   ```bash
   git clone <your-repo-url>
   cd chatgpt-backend
   ```

2. **配置环境变量**
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，设置你的 OPENAI_API_KEY
   nano .env
   ```

3. **启动开发环境**
   ```bash
   ./quick-start.sh dev
   ```

4. **访问服务**
   - API 文档: http://localhost:8000/docs
   - 数据库管理: http://localhost:8080
   - Redis 管理: http://localhost:8081

### 方式二：手动使用 Docker Compose

```bash
# 开发环境
docker-compose -f docker-compose.dev.yml up --build

# 生产环境
docker-compose up --build
```

### 方式三：本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件

# 初始化数据库
python scripts/init_db.py

# 启动服务
python start.py --dev
```

## 🔧 配置说明

### 核心配置项

在 `.env` 文件中设置以下关键配置：

```bash
# OpenAI API 配置（必需）
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，使用代理时修改

# 应用配置
SECRET_KEY=your-super-secret-key-change-this-in-production
ENVIRONMENT=development  # development 或 production

# 数据库配置
DATABASE_URL=postgresql://postgres:password@localhost:5432/chatgpt_backend

# Redis 配置
REDIS_URL=redis://localhost:6379/0

# 管理员用户（可选）
ADMIN_USERNAME=admin
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=admin123
```

### 支持的模型

默认支持以下 OpenAI 模型：
- GPT-4 系列: `gpt-4`, `gpt-4-turbo`, `gpt-4o`
- GPT-3.5 系列: `gpt-3.5-turbo`

可在配置中自定义模型列表和参数。

## 📚 API 使用示例

### 1. 用户注册

```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "password123"
  }'
```

### 2. 用户登录

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "password123"
  }'
```

### 3. 创建对话

```bash
curl -X POST "http://localhost:8000/api/v1/conversations" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "我的第一个对话",
    "model": "gpt-3.5-turbo"
  }'
```

### 4. 发送消息

```bash
curl -X POST "http://localhost:8000/api/v1/conversations/{conversation_id}/messages" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "你好，请介绍一下自己",
    "role": "user"
  }'
```

### 5. WebSocket 实时通信

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/chat?token=YOUR_ACCESS_TOKEN');

ws.onopen = function() {
    console.log('WebSocket 连接已建立');
    
    // 发送消息
    ws.send(JSON.stringify({
        type: 'chat_message',
        conversation_id: 'your_conversation_id',
        content: '你好！',
        model: 'gpt-3.5-turbo'
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('收到消息:', data);
};
```

## 🛠️ 开发工具

### 数据库管理
- **Adminer**: http://localhost:8080
  - 服务器: `db`
  - 用户名: `postgres`
  - 密码: `password`
  - 数据库: `chatgpt_backend_dev`

### Redis 管理
- **Redis Commander**: http://localhost:8081

### 日志查看
```bash
# 查看应用日志
docker-compose -f docker-compose.dev.yml logs -f app

# 查看数据库日志
docker-compose -f docker-compose.dev.yml logs -f db

# 查看所有服务日志
docker-compose -f docker-compose.dev.yml logs -f
```

## 🔍 健康检查

```bash
# 检查服务状态
curl http://localhost:8000/health

# 检查应用信息
curl http://localhost:8000/info

# 检查指标（如果启用）
curl http://localhost:8000/metrics
```

## 🚀 部署到生产环境

### 1. 使用 Docker Compose

```bash
# 启动生产环境
./quick-start.sh prod

# 或手动启动
docker-compose up -d --build
```

### 2. 环境变量配置

生产环境需要设置以下环境变量：

```bash
ENVIRONMENT=production
SECRET_KEY=your-production-secret-key
DATABASE_URL=postgresql://user:password@host:port/database
REDIS_URL=redis://host:port/0
OPENAI_API_KEY=your_production_openai_key

# 安全配置
CORS_ORIGINS=https://yourdomain.com
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

# 可选：邮件服务
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

### 3. 使用 Nginx 反向代理

```bash
# 启动包含 Nginx 的完整生产环境
docker-compose --profile production up -d
```

### 4. 监控和日志

```bash
# 启动监控服务
docker-compose --profile monitoring up -d

# 访问监控面板
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

## 🔧 常见问题

### Q: 服务启动失败
A: 检查以下项目：
1. Docker 和 Docker Compose 是否正确安装
2. 端口是否被占用
3. `.env` 文件是否正确配置
4. OpenAI API Key 是否有效

### Q: 数据库连接失败
A: 确保：
1. 数据库服务已启动
2. 数据库连接字符串正确
3. 网络连接正常

### Q: OpenAI API 调用失败
A: 检查：
1. API Key 是否正确
2. 账户是否有足够余额
3. 网络是否能访问 OpenAI API
4. 是否需要使用代理

### Q: 如何重置数据库
A: 运行以下命令：
```bash
# 停止服务
./quick-start.sh stop

# 清理所有数据
./quick-start.sh clean

# 重新启动
./quick-start.sh dev
```

## 📞 获取帮助

- 查看 [API 文档](http://localhost:8000/docs)
- 阅读 [README.md](./README.md)
- 提交 [Issue](https://github.com/your-repo/issues)

## 🎯 下一步

1. 探索 API 文档了解所有可用端点
2. 集成到你的前端应用
3. 自定义模型和参数
4. 设置监控和日志
5. 部署到生产环境

祝你使用愉快！🎉