# 实时聊天应用示例

💬 **高性能实时通信系统**

基于Python后端开发功能组件模板库构建的实时聊天应用，支持WebSocket实时通信、群聊、私聊等功能。

## 🚀 功能特性

### 核心功能
- **实时消息**：WebSocket双向通信、消息实时推送
- **聊天室管理**：创建房间、加入/退出、房间权限
- **私聊功能**：一对一聊天、消息加密
- **群聊功能**：多人群聊、@提醒、群管理
- **消息类型**：文本、图片、文件、表情包
- **在线状态**：用户在线/离线状态显示
- **消息历史**：聊天记录存储和检索
- **推送通知**：离线消息推送

### 技术特性
- **高并发**：支持大量并发连接
- **消息队列**：Redis消息分发和缓存
- **负载均衡**：多实例部署支持
- **数据持久化**：消息历史存储

### 技术模块
- **api**: FastAPI + WebSocket实时通信
- **database**: 消息存储和用户关系管理
- **auth**: 用户认证和聊天权限控制
- **cache**: Redis消息队列和在线状态缓存
- **monitoring**: 连接监控和性能指标

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 环境配置

复制环境变量模板文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置相应的环境变量。

### 运行应用

```bash
python main.py
```

应用将在 http://localhost:8000 启动。

### API文档

启动应用后，访问以下地址查看API文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 项目结构

```
realtime-chat/
├── app/                    # 应用代码
│   ├── api/               # API路由
│   ├── core/              # 核心配置
│   ├── models/            # 数据模型
│   ├── services/          # 业务逻辑
│   └── utils/             # 工具函数
├── tests/                 # 测试代码
├── docs/                  # 文档
├── scripts/               # 脚本文件
├── config/                # 配置文件
├── logs/                  # 日志文件
├── data/                  # 数据文件
├── main.py                # 应用入口
├── requirements.txt       # 依赖列表
└── README.md             # 项目说明
```

## 开发指南

### 添加新的API端点

在 `app/api/` 目录下创建新的路由文件，然后在主应用中注册。

### 数据库迁移

```bash
# 创建迁移文件
alembic revision --autogenerate -m "描述"

# 执行迁移
alembic upgrade head
```

### 运行测试

```bash
pytest tests/
```

## 部署

### Docker部署

```bash
# 构建镜像
docker build -t realtime-chat .

# 运行容器
docker run -p 8000:8000 realtime-chat
```

### 生产环境

建议使用以下配置进行生产部署：

- 使用Gunicorn或uWSGI作为WSGI服务器
- 配置Nginx作为反向代理
- 使用PostgreSQL或MySQL作为数据库
- 配置Redis用于缓存和会话存储
- 设置监控和日志收集

## 许可证

MIT License
