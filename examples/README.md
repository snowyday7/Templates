# 示例项目

🚀 **完整的示例项目集合**

本目录包含基于Python后端开发功能组件模板库构建的完整示例项目，展示了不同场景下的最佳实践和架构设计。

## 📋 项目列表

### 1. 电商API服务 (ecommerce-api)
🛒 **完整的电商后端API实现**

- **技术栈**：FastAPI + SQLAlchemy + Redis + Celery
- **核心功能**：商品管理、用户系统、购物车、订单处理、支付集成
- **适用场景**：电商平台、在线商城、B2C业务
- **学习重点**：复杂业务逻辑、事务处理、支付集成

[查看详情](./ecommerce-api/README.md)

### 2. 用户管理系统 (user-management)
👥 **企业级用户管理系统**

- **技术栈**：FastAPI + SQLAlchemy + JWT + RBAC
- **核心功能**：用户认证、权限管理、组织架构、审计日志
- **适用场景**：企业内部系统、SaaS平台、管理后台
- **学习重点**：认证授权、权限控制、安全设计

[查看详情](./user-management/README.md)

### 3. 实时聊天应用 (realtime-chat)
💬 **高性能实时通信系统**

- **技术栈**：FastAPI + WebSocket + Redis + PostgreSQL
- **核心功能**：实时消息、群聊私聊、在线状态、消息历史
- **适用场景**：即时通讯、在线客服、协作工具
- **学习重点**：WebSocket通信、实时架构、高并发处理

[查看详情](./realtime-chat/README.md)

### 4. 微服务架构 (microservice-demo)
🏗️ **云原生微服务架构演示**

- **技术栈**：FastAPI + Docker + Kubernetes + 微服务
- **核心功能**：服务拆分、API网关、服务发现、分布式追踪
- **适用场景**：大型系统、云原生应用、分布式架构
- **学习重点**：微服务设计、容器化部署、云原生实践

[查看详情](./microservice-demo/README.md)

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Docker & Docker Compose
- Redis (可选)
- PostgreSQL (可选)

### 运行示例项目

1. **选择项目**：进入任意示例项目目录
```bash
cd ecommerce-api  # 或其他项目
```

2. **安装依赖**：
```bash
pip install -r requirements.txt
```

3. **配置环境**：
```bash
cp .env.example .env
# 编辑 .env 文件配置数据库等信息
```

4. **启动服务**：
```bash
python main.py
```

5. **访问应用**：
- API文档：http://localhost:8000/docs
- 应用界面：http://localhost:8000

### Docker 快速启动

每个项目都支持Docker部署：

```bash
# 构建镜像
docker build -t project-name .

# 运行容器
docker run -p 8000:8000 project-name
```

## 📚 学习路径

### 初学者推荐顺序

1. **用户管理系统** - 学习基础的CRUD操作和认证
2. **电商API服务** - 掌握复杂业务逻辑和数据关系
3. **实时聊天应用** - 了解实时通信和WebSocket
4. **微服务架构** - 学习分布式系统和云原生

### 技能重点

- **API设计**：RESTful设计、文档生成、版本管理
- **数据库**：ORM使用、迁移管理、性能优化
- **认证授权**：JWT、OAuth、RBAC权限模型
- **缓存策略**：Redis使用、缓存模式、性能优化
- **异步处理**：Celery任务队列、后台任务
- **监控日志**：结构化日志、性能监控、错误追踪
- **部署运维**：Docker容器化、K8s编排、CI/CD

## 🛠️ 开发指南

### 代码结构

所有示例项目都遵循统一的代码结构：

```
project-name/
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
├── main.py                # 应用入口
├── requirements.txt       # 依赖列表
└── README.md             # 项目说明
```

### 最佳实践

- **模块化设计**：清晰的模块划分和依赖关系
- **配置管理**：环境变量和配置文件分离
- **错误处理**：统一的异常处理和错误响应
- **日志记录**：结构化日志和操作审计
- **测试覆盖**：单元测试和集成测试
- **文档完善**：API文档和使用说明

## 🤝 贡献指南

欢迎为示例项目贡献代码和改进建议：

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 创建 Pull Request

## 📄 许可证

所有示例项目均采用 MIT 许可证。

---

💡 **提示**：这些示例项目不仅可以作为学习材料，也可以作为实际项目的起始模板。根据具体需求进行修改和扩展即可快速构建生产级应用。