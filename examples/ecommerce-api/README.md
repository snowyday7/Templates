# 电商API服务示例

🛒 **完整的电商后端API实现**

基于Python后端开发功能组件模板库构建的电商平台后端服务，提供完整的电商业务功能。

## 🚀 功能特性

### 核心业务功能
- **商品管理**：商品CRUD、分类管理、库存管理
- **用户系统**：用户注册、登录、个人信息管理
- **购物车**：添加商品、修改数量、清空购物车
- **订单系统**：下单、支付、订单状态管理
- **支付集成**：支持多种支付方式
- **优惠券**：优惠券创建、使用、验证
- **评价系统**：商品评价、评分统计

### 技术模块
- **api**: FastAPI RESTful API框架
- **database**: SQLAlchemy ORM和PostgreSQL
- **auth**: JWT认证和RBAC权限控制
- **cache**: Redis缓存和Celery异步任务
- **monitoring**: 结构化日志和性能监控
- **deployment**: Docker容器化和K8s部署

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
ecommerce-api/
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
docker build -t ecommerce-api .

# 运行容器
docker run -p 8000:8000 ecommerce-api
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
