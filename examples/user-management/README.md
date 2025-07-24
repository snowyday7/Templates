# 用户管理系统示例

👥 **企业级用户管理系统**

基于Python后端开发功能组件模板库构建的完整用户管理系统，包含认证、授权、用户生命周期管理等功能。

## 🚀 功能特性

### 核心功能
- **用户注册**：邮箱验证、手机验证、社交登录
- **用户认证**：密码登录、双因子认证、SSO集成
- **权限管理**：基于角色的访问控制(RBAC)
- **用户档案**：个人信息管理、头像上传
- **安全功能**：密码策略、登录日志、异常检测
- **组织管理**：部门管理、团队协作
- **审计日志**：操作记录、合规性报告

### 技术模块
- **api**: FastAPI RESTful API框架
- **database**: SQLAlchemy ORM和用户数据模型
- **auth**: JWT认证和权限控制系统
- **monitoring**: 用户行为监控和安全日志

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
user-management/
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
docker build -t user-management .

# 运行容器
docker run -p 8000:8000 user-management
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
