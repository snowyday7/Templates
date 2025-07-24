# ChatGPT Backend 部署指南

## 📋 目录

- [部署概览](#部署概览)
- [环境准备](#环境准备)
- [本地开发部署](#本地开发部署)
- [Docker 部署](#docker-部署)
- [生产环境部署](#生产环境部署)
- [云平台部署](#云平台部署)
- [监控和日志](#监控和日志)
- [备份和恢复](#备份和恢复)
- [故障排除](#故障排除)
- [性能优化](#性能优化)

## 🌐 部署概览

### 支持的部署方式

1. **本地开发**: 直接运行 Python 应用
2. **Docker 容器**: 单机容器化部署
3. **Docker Compose**: 多服务编排部署
4. **云平台**: AWS、Azure、GCP 等
5. **Kubernetes**: 容器编排平台
6. **传统服务器**: VPS、专用服务器

### 架构选择

| 场景 | 推荐方案 | 特点 |
|------|----------|------|
| 开发测试 | Docker Compose | 快速启动，完整环境 |
| 小型生产 | Docker + Nginx | 简单可靠，易维护 |
| 中型生产 | Kubernetes | 自动扩缩，高可用 |
| 大型生产 | 微服务 + K8s | 高性能，高可用 |

## 🛠️ 环境准备

### 系统要求

#### 最低配置
- **CPU**: 1 核心
- **内存**: 2GB RAM
- **存储**: 10GB 可用空间
- **网络**: 稳定的互联网连接

#### 推荐配置
- **CPU**: 2+ 核心
- **内存**: 4GB+ RAM
- **存储**: 50GB+ SSD
- **网络**: 高速互联网连接

### 软件依赖

#### 必需软件
```bash
# Docker 和 Docker Compose
sudo apt update
sudo apt install docker.io docker-compose

# 或者使用官方安装脚本
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

#### 可选软件
```bash
# Python 3.11+ (本地开发)
sudo apt install python3.11 python3.11-venv python3-pip

# PostgreSQL (本地数据库)
sudo apt install postgresql postgresql-contrib

# Redis (本地缓存)
sudo apt install redis-server

# Nginx (反向代理)
sudo apt install nginx
```

## 💻 本地开发部署

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd chatgpt-backend
```

### 2. 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，设置必要的配置
nano .env
```

### 5. 初始化数据库

```bash
# 如果使用 SQLite（默认）
python scripts/init_db.py

# 如果使用 PostgreSQL
sudo -u postgres createdb chatgpt_backend
python scripts/init_db.py
```

### 6. 启动应用

```bash
# 开发模式
python start.py --dev

# 或直接使用 uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. 验证部署

```bash
# 检查健康状态
curl http://localhost:8000/health

# 访问 API 文档
open http://localhost:8000/docs
```

## 🐳 Docker 部署

### 单容器部署

#### 1. 构建镜像

```bash
docker build -t chatgpt-backend .
```

#### 2. 运行容器

```bash
docker run -d \
  --name chatgpt-backend \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_api_key \
  -e DATABASE_URL=sqlite:///./app.db \
  -v $(pwd)/data:/app/data \
  chatgpt-backend
```

### Docker Compose 部署

#### 1. 开发环境

```bash
# 启动开发环境
./quick-start.sh dev

# 或手动启动
docker-compose -f docker-compose.dev.yml up -d
```

#### 2. 生产环境

```bash
# 启动生产环境
./quick-start.sh prod

# 或手动启动
docker-compose up -d
```

#### 3. 查看服务状态

```bash
# 查看容器状态
docker-compose ps

# 查看日志
docker-compose logs -f app

# 进入容器
docker-compose exec app bash
```

## 🏭 生产环境部署

### 使用 Docker Compose + Nginx

#### 1. 准备生产配置

```bash
# 创建生产环境配置
cp .env.example .env.prod

# 编辑生产配置
nano .env.prod
```

生产环境 `.env.prod` 示例：

```bash
# 应用配置
ENVIRONMENT=production
SECRET_KEY=your-super-secure-secret-key-for-production
DEBUG=false

# 数据库配置
DATABASE_URL=postgresql://username:password@db:5432/chatgpt_backend

# Redis 配置
REDIS_URL=redis://redis:6379/0

# OpenAI 配置
OPENAI_API_KEY=your_production_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# 安全配置
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

# 日志配置
LOG_LEVEL=INFO
LOG_FORMAT=json

# 邮件配置（可选）
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_FROM_EMAIL=noreply@yourdomain.com
```

#### 2. 创建 Nginx 配置

```bash
mkdir -p nginx/conf.d
```

`nginx/conf.d/default.conf`:

```nginx
upstream backend {
    server app:8000;
}

server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    
    # 重定向到 HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;
    
    # SSL 配置
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    
    # 安全头
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # 客户端最大请求大小
    client_max_body_size 10M;
    
    # API 路由
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket 支持
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # WebSocket 路由
    location /ws/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # 健康检查
    location /health {
        proxy_pass http://backend;
        access_log off;
    }
    
    # 静态文件（如果有）
    location /static/ {
        alias /var/www/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # 默认路由
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### 3. 启动生产环境

```bash
# 使用生产配置启动
docker-compose --env-file .env.prod up -d

# 或使用 profile
docker-compose --profile production up -d
```

### SSL 证书配置

#### 使用 Let's Encrypt

```bash
# 安装 certbot
sudo apt install certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# 自动续期
sudo crontab -e
# 添加以下行
0 12 * * * /usr/bin/certbot renew --quiet
```

#### 手动证书配置

```bash
# 创建 SSL 目录
mkdir -p nginx/ssl

# 复制证书文件
cp your_cert.pem nginx/ssl/cert.pem
cp your_key.pem nginx/ssl/key.pem

# 设置权限
chmod 600 nginx/ssl/key.pem
```

## ☁️ 云平台部署

### AWS 部署

#### 使用 ECS (Elastic Container Service)

1. **创建 ECR 仓库**

```bash
# 创建仓库
aws ecr create-repository --repository-name chatgpt-backend

# 获取登录令牌
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com

# 构建并推送镜像
docker build -t chatgpt-backend .
docker tag chatgpt-backend:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/chatgpt-backend:latest
docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/chatgpt-backend:latest
```

2. **创建 ECS 任务定义**

`ecs-task-definition.json`:

```json
{
  "family": "chatgpt-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "chatgpt-backend",
      "image": "<account-id>.dkr.ecr.us-west-2.amazonaws.com/chatgpt-backend:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-west-2:<account-id>:secret:openai-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/chatgpt-backend",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### 使用 EC2

```bash
# 连接到 EC2 实例
ssh -i your-key.pem ubuntu@your-ec2-ip

# 安装 Docker
sudo apt update
sudo apt install docker.io docker-compose
sudo usermod -aG docker ubuntu

# 克隆项目
git clone <your-repo-url>
cd chatgpt-backend

# 配置环境变量
cp .env.example .env
nano .env

# 启动服务
docker-compose up -d
```

### Google Cloud Platform 部署

#### 使用 Cloud Run

```bash
# 构建并推送到 GCR
gcloud builds submit --tag gcr.io/your-project-id/chatgpt-backend

# 部署到 Cloud Run
gcloud run deploy chatgpt-backend \
  --image gcr.io/your-project-id/chatgpt-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars ENVIRONMENT=production \
  --set-env-vars OPENAI_API_KEY=your_api_key
```

### Azure 部署

#### 使用 Container Instances

```bash
# 创建资源组
az group create --name chatgpt-backend-rg --location eastus

# 创建容器实例
az container create \
  --resource-group chatgpt-backend-rg \
  --name chatgpt-backend \
  --image your-registry/chatgpt-backend:latest \
  --dns-name-label chatgpt-backend \
  --ports 8000 \
  --environment-variables ENVIRONMENT=production \
  --secure-environment-variables OPENAI_API_KEY=your_api_key
```

## 🎛️ Kubernetes 部署

### 1. 创建命名空间

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: chatgpt-backend
```

### 2. 创建 ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: chatgpt-backend-config
  namespace: chatgpt-backend
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  DATABASE_URL: "postgresql://username:password@postgres:5432/chatgpt_backend"
  REDIS_URL: "redis://redis:6379/0"
```

### 3. 创建 Secret

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: chatgpt-backend-secret
  namespace: chatgpt-backend
type: Opaque
data:
  OPENAI_API_KEY: <base64-encoded-api-key>
  SECRET_KEY: <base64-encoded-secret-key>
```

### 4. 创建 Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chatgpt-backend
  namespace: chatgpt-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chatgpt-backend
  template:
    metadata:
      labels:
        app: chatgpt-backend
    spec:
      containers:
      - name: chatgpt-backend
        image: your-registry/chatgpt-backend:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: chatgpt-backend-config
        - secretRef:
            name: chatgpt-backend-secret
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### 5. 创建 Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: chatgpt-backend-service
  namespace: chatgpt-backend
spec:
  selector:
    app: chatgpt-backend
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

### 6. 创建 Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: chatgpt-backend-ingress
  namespace: chatgpt-backend
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: chatgpt-backend-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: chatgpt-backend-service
            port:
              number: 80
```

### 7. 部署到 Kubernetes

```bash
# 应用所有配置
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# 查看部署状态
kubectl get pods -n chatgpt-backend
kubectl get services -n chatgpt-backend
kubectl get ingress -n chatgpt-backend
```

## 📊 监控和日志

### Prometheus + Grafana 监控

#### 1. 启动监控服务

```bash
# 启动监控 profile
docker-compose --profile monitoring up -d
```

#### 2. Prometheus 配置

`monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'chatgpt-backend'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

#### 3. Grafana 仪表板

访问 http://localhost:3000 (admin/admin) 导入预配置的仪表板。

### 日志管理

#### 1. 集中化日志收集

```yaml
# docker-compose.yml 中添加
services:
  app:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # ELK Stack (可选)
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.14.0
    environment:
      - discovery.type=single-node
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:7.14.0
    volumes:
      - ./logstash/config:/usr/share/logstash/pipeline

  kibana:
    image: docker.elastic.co/kibana/kibana:7.14.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
```

#### 2. 日志查看命令

```bash
# 查看应用日志
docker-compose logs -f app

# 查看特定时间段的日志
docker-compose logs --since="2024-01-01T12:00:00" app

# 查看错误日志
docker-compose logs app | grep ERROR

# 实时监控日志
tail -f logs/app.log
```

## 💾 备份和恢复

### 数据库备份

#### 1. 自动备份脚本

```bash
#!/bin/bash
# backup.sh

DATE=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="/backups"
DB_NAME="chatgpt_backend"

# PostgreSQL 备份
docker-compose exec -T db pg_dump -U postgres $DB_NAME > "$BACKUP_DIR/db_backup_$DATE.sql"

# 压缩备份文件
gzip "$BACKUP_DIR/db_backup_$DATE.sql"

# 删除 7 天前的备份
find $BACKUP_DIR -name "db_backup_*.sql.gz" -mtime +7 -delete

echo "Backup completed: db_backup_$DATE.sql.gz"
```

#### 2. 设置定时备份

```bash
# 添加到 crontab
crontab -e

# 每天凌晨 2 点备份
0 2 * * * /path/to/backup.sh
```

### 数据恢复

```bash
# 恢复数据库
gunzip -c /backups/db_backup_20240101_020000.sql.gz | docker-compose exec -T db psql -U postgres -d chatgpt_backend

# 或使用脚本中的恢复函数
python scripts/backup.py restore --file /backups/db_backup_20240101_020000.sql.gz
```

## 🔧 故障排除

### 常见问题

#### 1. 容器启动失败

```bash
# 查看容器日志
docker-compose logs app

# 检查容器状态
docker-compose ps

# 重启服务
docker-compose restart app
```

#### 2. 数据库连接失败

```bash
# 检查数据库状态
docker-compose exec db pg_isready -U postgres

# 查看数据库日志
docker-compose logs db

# 重置数据库
docker-compose down
docker volume rm chatgpt-backend_postgres_data
docker-compose up -d
```

#### 3. OpenAI API 调用失败

```bash
# 测试 API 连接
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models

# 检查环境变量
docker-compose exec app env | grep OPENAI

# 查看应用日志中的 OpenAI 相关错误
docker-compose logs app | grep -i openai
```

#### 4. 内存不足

```bash
# 查看系统资源使用
docker stats

# 增加 swap 空间
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 优化 Docker 内存限制
docker-compose.yml 中添加:
services:
  app:
    mem_limit: 512m
    memswap_limit: 1g
```

### 性能问题诊断

```bash
# 查看应用性能指标
curl http://localhost:8000/metrics

# 数据库性能分析
docker-compose exec db psql -U postgres -d chatgpt_backend -c "SELECT * FROM pg_stat_activity;"

# Redis 性能监控
docker-compose exec redis redis-cli info stats
```

## ⚡ 性能优化

### 应用层优化

#### 1. 连接池配置

```python
# database.py
engine = create_async_engine(
    database_url,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

#### 2. 缓存优化

```python
# 增加缓存 TTL
USER_CACHE_TTL = 3600  # 1 小时
CONVERSATION_CACHE_TTL = 1800  # 30 分钟
MODEL_CACHE_TTL = 86400  # 24 小时
```

### 数据库优化

#### 1. PostgreSQL 配置

```sql
-- postgresql.conf 优化
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
max_connections = 100
```

#### 2. 索引优化

```sql
-- 添加复合索引
CREATE INDEX idx_messages_conversation_created ON messages(conversation_id, created_at DESC);
CREATE INDEX idx_conversations_user_updated ON conversations(user_id, updated_at DESC);
```

### 系统层优化

#### 1. Nginx 优化

```nginx
# nginx.conf
worker_processes auto;
worker_connections 1024;

# 启用 gzip 压缩
gzip on;
gzip_vary on;
gzip_min_length 1024;
gzip_types text/plain text/css application/json application/javascript;

# 缓存配置
location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

#### 2. Docker 优化

```yaml
# docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
```

### 监控和告警

#### 1. 设置告警规则

```yaml
# prometheus/alert.rules.yml
groups:
- name: chatgpt-backend
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    annotations:
      summary: "High error rate detected"
      
  - alert: HighMemoryUsage
    expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.8
    for: 5m
    annotations:
      summary: "High memory usage detected"
```

#### 2. 健康检查

```bash
# 创建健康检查脚本
#!/bin/bash
# health_check.sh

HEALTH_URL="http://localhost:8000/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)

if [ $RESPONSE -eq 200 ]; then
    echo "Service is healthy"
    exit 0
else
    echo "Service is unhealthy (HTTP $RESPONSE)"
    exit 1
fi
```

通过以上部署指南，您可以根据不同的需求和环境选择合适的部署方式，确保 ChatGPT Backend 服务稳定、高效地运行。