# 部署配置模块使用指南

部署配置模块提供了完整的应用部署解决方案，包括Docker容器化、Kubernetes编排、CI/CD流水线、服务器配置等功能。

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
pip install docker kubernetes pyyaml jinja2
```

### 基础配置

```python
from templates.deployment import DockerConfig, DockerfileGenerator, DockerComposeGenerator

# 创建Docker配置
docker_config = DockerConfig(
    base_image="python:3.11-slim",
    working_dir="/app",
    port=8000,
    environment="production"
)

# 创建Dockerfile生成器
dockerfile_generator = DockerfileGenerator(docker_config)

# 创建Docker Compose生成器
compose_generator = DockerComposeGenerator()
```

## ⚙️ 配置说明

### DockerConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_image` | str | "python:3.11-slim" | 基础镜像 |
| `working_dir` | str | "/app" | 工作目录 |
| `port` | int | 8000 | 应用端口 |
| `environment` | str | "production" | 环境类型 |
| `python_version` | str | "3.11" | Python版本 |
| `requirements_file` | str | "requirements.txt" | 依赖文件 |
| `app_module` | str | "main:app" | 应用模块 |
| `user` | str | "appuser" | 运行用户 |
| `group` | str | "appgroup" | 运行用户组 |
| `uid` | int | 1000 | 用户ID |
| `gid` | int | 1000 | 用户组ID |

### Kubernetes配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `namespace` | str | "default" | 命名空间 |
| `replicas` | int | 3 | 副本数量 |
| `cpu_request` | str | "100m" | CPU请求 |
| `cpu_limit` | str | "500m" | CPU限制 |
| `memory_request` | str | "128Mi" | 内存请求 |
| `memory_limit` | str | "512Mi" | 内存限制 |
| `service_type` | str | "ClusterIP" | 服务类型 |
| `ingress_enabled` | bool | True | 启用Ingress |
| `health_check_path` | str | "/health" | 健康检查路径 |
| `readiness_probe_path` | str | "/ready" | 就绪探针路径 |

### 环境变量配置

创建 `.env` 文件：

```env
DOCKER_BASE_IMAGE=python:3.11-slim
DOCKER_WORKING_DIR=/app
DOCKER_PORT=8000
DOCKER_ENVIRONMENT=production

KUBERNETES_NAMESPACE=myapp
KUBERNETES_REPLICAS=3
KUBERNETES_CPU_REQUEST=100m
KUBERNETES_CPU_LIMIT=500m
KUBERNETES_MEMORY_REQUEST=128Mi
KUBERNETES_MEMORY_LIMIT=512Mi

CI_CD_PROVIDER=github
CI_CD_REGISTRY=ghcr.io
CI_CD_IMAGE_NAME=myapp
```

## 💻 基础使用

### 1. Docker容器化

```python
from templates.deployment import DockerfileGenerator, DockerConfig

# 创建配置
config = DockerConfig(
    base_image="python:3.11-slim",
    working_dir="/app",
    port=8000,
    requirements_file="requirements.txt",
    app_module="main:app"
)

# 生成Dockerfile
generator = DockerfileGenerator(config)
dockerfile_content = generator.generate()

# 保存Dockerfile
with open("Dockerfile", "w") as f:
    f.write(dockerfile_content)

print("Dockerfile generated successfully!")
```

生成的Dockerfile示例：

```dockerfile
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 创建非root用户
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash appuser

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 更改文件所有者
RUN chown -R appuser:appgroup /app

# 切换到非root用户
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose配置

```python
from templates.deployment import DockerComposeGenerator, ServiceConfig

# 创建服务配置
api_service = ServiceConfig(
    name="api",
    image="myapp:latest",
    ports=["8000:8000"],
    environment={
        "DATABASE_URL": "postgresql://user:pass@db:5432/myapp",
        "REDIS_URL": "redis://redis:6379/0"
    },
    depends_on=["db", "redis"],
    volumes=["./logs:/app/logs"]
)

db_service = ServiceConfig(
    name="db",
    image="postgres:15",
    environment={
        "POSTGRES_DB": "myapp",
        "POSTGRES_USER": "user",
        "POSTGRES_PASSWORD": "password"
    },
    volumes=["postgres_data:/var/lib/postgresql/data"],
    ports=["5432:5432"]
)

redis_service = ServiceConfig(
    name="redis",
    image="redis:7-alpine",
    ports=["6379:6379"],
    volumes=["redis_data:/data"]
)

# 生成docker-compose.yml
compose_generator = DockerComposeGenerator()
compose_generator.add_service(api_service)
compose_generator.add_service(db_service)
compose_generator.add_service(redis_service)

# 添加网络和卷
compose_generator.add_network("app_network", driver="bridge")
compose_generator.add_volume("postgres_data")
compose_generator.add_volume("redis_data")

compose_content = compose_generator.generate()

# 保存docker-compose.yml
with open("docker-compose.yml", "w") as f:
    f.write(compose_content)

print("Docker Compose file generated successfully!")
```

### 3. Kubernetes部署

```python
from templates.deployment import KubernetesGenerator, K8sConfig

# 创建Kubernetes配置
k8s_config = K8sConfig(
    app_name="myapp",
    namespace="production",
    image="myapp:v1.0.0",
    replicas=3,
    port=8000,
    cpu_request="100m",
    cpu_limit="500m",
    memory_request="128Mi",
    memory_limit="512Mi"
)

# 生成Kubernetes清单
k8s_generator = KubernetesGenerator(k8s_config)

# 生成Deployment
deployment = k8s_generator.generate_deployment()
with open("deployment.yaml", "w") as f:
    f.write(deployment)

# 生成Service
service = k8s_generator.generate_service()
with open("service.yaml", "w") as f:
    f.write(service)

# 生成Ingress
ingress = k8s_generator.generate_ingress(
    host="myapp.example.com",
    tls_enabled=True
)
with open("ingress.yaml", "w") as f:
    f.write(ingress)

# 生成ConfigMap
config_map = k8s_generator.generate_configmap({
    "DATABASE_HOST": "postgres-service",
    "REDIS_HOST": "redis-service",
    "LOG_LEVEL": "INFO"
})
with open("configmap.yaml", "w") as f:
    f.write(config_map)

print("Kubernetes manifests generated successfully!")
```

## 🔧 高级功能

### 1. CI/CD流水线

```python
from templates.deployment import CICDGenerator, GitHubActionsConfig

# GitHub Actions配置
gh_config = GitHubActionsConfig(
    python_version="3.11",
    registry="ghcr.io",
    image_name="myapp",
    deploy_environment="production",
    k8s_cluster="production-cluster"
)

# 生成GitHub Actions工作流
cicd_generator = CICDGenerator()
workflow = cicd_generator.generate_github_actions(gh_config)

# 保存工作流文件
import os
os.makedirs(".github/workflows", exist_ok=True)
with open(".github/workflows/deploy.yml", "w") as f:
    f.write(workflow)

print("GitHub Actions workflow generated successfully!")
```

生成的GitHub Actions工作流示例：

```yaml
name: Build and Deploy

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: myapp

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov
          
      - name: Run tests
        run: |
          pytest --cov=. --cov-report=xml
          
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      
      - name: Log in to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure kubectl
        uses: azure/k8s-set-context@v1
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG }}
          
      - name: Deploy to Kubernetes
        run: |
          sed -i 's|IMAGE_TAG|${{ github.sha }}|g' k8s/deployment.yaml
          kubectl apply -f k8s/
```

### 2. 多环境部署

```python
from templates.deployment import EnvironmentManager, EnvironmentConfig

# 定义环境配置
dev_config = EnvironmentConfig(
    name="development",
    replicas=1,
    cpu_limit="200m",
    memory_limit="256Mi",
    database_url="postgresql://dev_user:dev_pass@dev-db:5432/dev_db",
    debug=True
)

staging_config = EnvironmentConfig(
    name="staging",
    replicas=2,
    cpu_limit="300m",
    memory_limit="384Mi",
    database_url="postgresql://staging_user:staging_pass@staging-db:5432/staging_db",
    debug=False
)

prod_config = EnvironmentConfig(
    name="production",
    replicas=5,
    cpu_limit="500m",
    memory_limit="512Mi",
    database_url="postgresql://prod_user:prod_pass@prod-db:5432/prod_db",
    debug=False
)

# 环境管理器
env_manager = EnvironmentManager()
env_manager.add_environment(dev_config)
env_manager.add_environment(staging_config)
env_manager.add_environment(prod_config)

# 为每个环境生成配置
for env_name in ["development", "staging", "production"]:
    env_config = env_manager.get_environment(env_name)
    
    # 生成环境特定的Kubernetes清单
    k8s_config = K8sConfig(
        app_name="myapp",
        namespace=env_name,
        replicas=env_config.replicas,
        cpu_limit=env_config.cpu_limit,
        memory_limit=env_config.memory_limit
    )
    
    k8s_generator = KubernetesGenerator(k8s_config)
    
    # 创建环境目录
    os.makedirs(f"k8s/{env_name}", exist_ok=True)
    
    # 生成部署文件
    deployment = k8s_generator.generate_deployment()
    with open(f"k8s/{env_name}/deployment.yaml", "w") as f:
        f.write(deployment)
    
    # 生成环境变量配置
    env_vars = {
        "DATABASE_URL": env_config.database_url,
        "DEBUG": str(env_config.debug).lower(),
        "ENVIRONMENT": env_name
    }
    
    config_map = k8s_generator.generate_configmap(env_vars)
    with open(f"k8s/{env_name}/configmap.yaml", "w") as f:
        f.write(config_map)

print("Multi-environment configurations generated successfully!")
```

### 3. 蓝绿部署

```python
from templates.deployment import BlueGreenDeployment

class BlueGreenDeployment:
    def __init__(self, k8s_client, app_name: str, namespace: str):
        self.k8s_client = k8s_client
        self.app_name = app_name
        self.namespace = namespace
    
    async def deploy(self, new_image: str, health_check_url: str):
        """执行蓝绿部署"""
        # 1. 获取当前活跃版本
        current_version = await self._get_current_version()
        new_version = "blue" if current_version == "green" else "green"
        
        logger.info(f"Starting blue-green deployment: {current_version} -> {new_version}")
        
        try:
            # 2. 部署新版本
            await self._deploy_version(new_version, new_image)
            
            # 3. 等待新版本就绪
            await self._wait_for_ready(new_version)
            
            # 4. 健康检查
            if await self._health_check(new_version, health_check_url):
                # 5. 切换流量
                await self._switch_traffic(new_version)
                
                # 6. 验证切换成功
                await self._verify_traffic_switch(new_version)
                
                # 7. 清理旧版本
                await self._cleanup_old_version(current_version)
                
                logger.info(f"Blue-green deployment completed successfully")
                return True
            else:
                # 健康检查失败，回滚
                await self._rollback(current_version, new_version)
                return False
                
        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            await self._rollback(current_version, new_version)
            raise
    
    async def _deploy_version(self, version: str, image: str):
        """部署指定版本"""
        deployment_name = f"{self.app_name}-{version}"
        
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": deployment_name,
                "namespace": self.namespace,
                "labels": {
                    "app": self.app_name,
                    "version": version
                }
            },
            "spec": {
                "replicas": 3,
                "selector": {
                    "matchLabels": {
                        "app": self.app_name,
                        "version": version
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.app_name,
                            "version": version
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": self.app_name,
                            "image": image,
                            "ports": [{"containerPort": 8000}],
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 10,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
        
        await self.k8s_client.create_or_update_deployment(deployment_manifest)
    
    async def _switch_traffic(self, new_version: str):
        """切换服务流量到新版本"""
        service_manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": self.app_name,
                "namespace": self.namespace
            },
            "spec": {
                "selector": {
                    "app": self.app_name,
                    "version": new_version
                },
                "ports": [{
                    "port": 80,
                    "targetPort": 8000
                }]
            }
        }
        
        await self.k8s_client.update_service(service_manifest)

# 使用蓝绿部署
blue_green = BlueGreenDeployment(k8s_client, "myapp", "production")
await blue_green.deploy("myapp:v2.0.0", "http://myapp/health")
```

### 4. 金丝雀部署

```python
from templates.deployment import CanaryDeployment

class CanaryDeployment:
    def __init__(self, k8s_client, app_name: str, namespace: str):
        self.k8s_client = k8s_client
        self.app_name = app_name
        self.namespace = namespace
    
    async def deploy(self, new_image: str, traffic_percentages: list = [10, 25, 50, 100]):
        """执行金丝雀部署"""
        logger.info(f"Starting canary deployment with image: {new_image}")
        
        try:
            # 1. 部署金丝雀版本
            await self._deploy_canary(new_image)
            
            # 2. 逐步增加流量
            for percentage in traffic_percentages:
                logger.info(f"Routing {percentage}% traffic to canary")
                
                # 更新流量分配
                await self._update_traffic_split(percentage)
                
                # 监控指标
                await self._monitor_metrics(duration=300)  # 监控5分钟
                
                # 检查是否需要回滚
                if await self._should_rollback():
                    await self._rollback_canary()
                    return False
                
                # 等待下一阶段
                if percentage < 100:
                    await asyncio.sleep(600)  # 等待10分钟
            
            # 3. 完成部署
            await self._complete_canary_deployment()
            logger.info("Canary deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            await self._rollback_canary()
            raise
    
    async def _update_traffic_split(self, canary_percentage: int):
        """更新流量分配"""
        stable_percentage = 100 - canary_percentage
        
        # 使用Istio VirtualService进行流量分割
        virtual_service = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {
                "name": self.app_name,
                "namespace": self.namespace
            },
            "spec": {
                "hosts": [self.app_name],
                "http": [{
                    "match": [{"headers": {"canary": {"exact": "true"}}}],
                    "route": [{"destination": {"host": self.app_name, "subset": "canary"}}]
                }, {
                    "route": [
                        {
                            "destination": {"host": self.app_name, "subset": "stable"},
                            "weight": stable_percentage
                        },
                        {
                            "destination": {"host": self.app_name, "subset": "canary"},
                            "weight": canary_percentage
                        }
                    ]
                }]
            }
        }
        
        await self.k8s_client.apply_manifest(virtual_service)
    
    async def _monitor_metrics(self, duration: int):
        """监控金丝雀版本的指标"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # 获取错误率
            error_rate = await self._get_error_rate("canary")
            
            # 获取响应时间
            response_time = await self._get_avg_response_time("canary")
            
            logger.info(
                "Canary metrics",
                error_rate=error_rate,
                avg_response_time=response_time
            )
            
            await asyncio.sleep(30)  # 每30秒检查一次
    
    async def _should_rollback(self) -> bool:
        """判断是否需要回滚"""
        # 获取金丝雀版本的指标
        canary_error_rate = await self._get_error_rate("canary")
        stable_error_rate = await self._get_error_rate("stable")
        
        canary_response_time = await self._get_avg_response_time("canary")
        stable_response_time = await self._get_avg_response_time("stable")
        
        # 回滚条件
        if canary_error_rate > stable_error_rate * 2:  # 错误率超过稳定版本2倍
            logger.warning("High error rate detected, rolling back")
            return True
        
        if canary_response_time > stable_response_time * 1.5:  # 响应时间超过稳定版本1.5倍
            logger.warning("High response time detected, rolling back")
            return True
        
        return False

# 使用金丝雀部署
canary = CanaryDeployment(k8s_client, "myapp", "production")
await canary.deploy("myapp:v2.0.0", [5, 10, 25, 50, 100])
```

## 📝 最佳实践

### 1. 容器安全

```python
# 安全的Dockerfile模板
SECURE_DOCKERFILE_TEMPLATE = """
# 使用官方基础镜像
FROM python:3.11-slim

# 设置标签
LABEL maintainer="your-email@example.com"
LABEL version="1.0.0"
LABEL description="My secure application"

# 更新系统包
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 创建非特权用户
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# 设置工作目录
WORKDIR /app

# 复制并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY --chown=appuser:appgroup . .

# 设置正确的权限
RUN chmod -R 755 /app

# 切换到非特权用户
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# 安全扫描集成
class SecurityScanner:
    def __init__(self):
        self.scanners = {
            "trivy": self._run_trivy_scan,
            "clair": self._run_clair_scan,
            "snyk": self._run_snyk_scan
        }
    
    async def scan_image(self, image_name: str, scanner: str = "trivy"):
        """扫描Docker镜像安全漏洞"""
        if scanner not in self.scanners:
            raise ValueError(f"Unsupported scanner: {scanner}")
        
        return await self.scanners[scanner](image_name)
    
    async def _run_trivy_scan(self, image_name: str):
        """使用Trivy扫描镜像"""
        cmd = f"trivy image --format json {image_name}"
        result = await self._run_command(cmd)
        
        # 解析扫描结果
        scan_data = json.loads(result)
        vulnerabilities = []
        
        for target in scan_data.get("Results", []):
            for vuln in target.get("Vulnerabilities", []):
                vulnerabilities.append({
                    "id": vuln.get("VulnerabilityID"),
                    "severity": vuln.get("Severity"),
                    "title": vuln.get("Title"),
                    "description": vuln.get("Description")
                })
        
        return {
            "image": image_name,
            "vulnerabilities": vulnerabilities,
            "high_severity_count": len([v for v in vulnerabilities if v["severity"] == "HIGH"]),
            "critical_severity_count": len([v for v in vulnerabilities if v["severity"] == "CRITICAL"])
        }
```

### 2. 资源管理

```python
# Kubernetes资源配额
class ResourceQuotaManager:
    def __init__(self, k8s_client):
        self.k8s_client = k8s_client
    
    def create_namespace_quota(self, namespace: str, quotas: dict):
        """为命名空间创建资源配额"""
        quota_manifest = {
            "apiVersion": "v1",
            "kind": "ResourceQuota",
            "metadata": {
                "name": f"{namespace}-quota",
                "namespace": namespace
            },
            "spec": {
                "hard": quotas
            }
        }
        
        return quota_manifest
    
    def create_limit_range(self, namespace: str, limits: dict):
        """创建资源限制范围"""
        limit_range_manifest = {
            "apiVersion": "v1",
            "kind": "LimitRange",
            "metadata": {
                "name": f"{namespace}-limits",
                "namespace": namespace
            },
            "spec": {
                "limits": [{
                    "type": "Container",
                    "default": limits.get("default", {}),
                    "defaultRequest": limits.get("defaultRequest", {}),
                    "max": limits.get("max", {}),
                    "min": limits.get("min", {})
                }]
            }
        }
        
        return limit_range_manifest

# 使用资源管理
resource_manager = ResourceQuotaManager(k8s_client)

# 生产环境资源配额
prod_quotas = {
    "requests.cpu": "4",
    "requests.memory": "8Gi",
    "limits.cpu": "8",
    "limits.memory": "16Gi",
    "persistentvolumeclaims": "10",
    "services": "5",
    "secrets": "10",
    "configmaps": "10"
}

prod_limits = {
    "default": {
        "cpu": "500m",
        "memory": "512Mi"
    },
    "defaultRequest": {
        "cpu": "100m",
        "memory": "128Mi"
    },
    "max": {
        "cpu": "2",
        "memory": "4Gi"
    },
    "min": {
        "cpu": "50m",
        "memory": "64Mi"
    }
}

quota_manifest = resource_manager.create_namespace_quota("production", prod_quotas)
limit_manifest = resource_manager.create_limit_range("production", prod_limits)
```

### 3. 监控和告警

```python
# 部署监控配置
class DeploymentMonitoring:
    def __init__(self):
        self.prometheus_config = self._create_prometheus_config()
        self.grafana_dashboards = self._create_grafana_dashboards()
        self.alert_rules = self._create_alert_rules()
    
    def _create_prometheus_config(self):
        """创建Prometheus配置"""
        return {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "scrape_configs": [
                {
                    "job_name": "kubernetes-pods",
                    "kubernetes_sd_configs": [{
                        "role": "pod"
                    }],
                    "relabel_configs": [
                        {
                            "source_labels": ["__meta_kubernetes_pod_annotation_prometheus_io_scrape"],
                            "action": "keep",
                            "regex": "true"
                        },
                        {
                            "source_labels": ["__meta_kubernetes_pod_annotation_prometheus_io_path"],
                            "action": "replace",
                            "target_label": "__metrics_path__",
                            "regex": "(.+)"
                        }
                    ]
                }
            ]
        }
    
    def _create_alert_rules(self):
        """创建告警规则"""
        return {
            "groups": [
                {
                    "name": "deployment.rules",
                    "rules": [
                        {
                            "alert": "PodCrashLooping",
                            "expr": "rate(kube_pod_container_status_restarts_total[15m]) > 0",
                            "for": "5m",
                            "labels": {
                                "severity": "critical"
                            },
                            "annotations": {
                                "summary": "Pod {{ $labels.pod }} is crash looping",
                                "description": "Pod {{ $labels.pod }} in namespace {{ $labels.namespace }} is restarting frequently."
                            }
                        },
                        {
                            "alert": "DeploymentReplicasMismatch",
                            "expr": "kube_deployment_spec_replicas != kube_deployment_status_available_replicas",
                            "for": "10m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "Deployment {{ $labels.deployment }} has mismatched replicas",
                                "description": "Deployment {{ $labels.deployment }} in namespace {{ $labels.namespace }} has {{ $value }} available replicas, expected {{ $labels.spec_replicas }}."
                            }
                        }
                    ]
                }
            ]
        }
```

## ❓ 常见问题

### Q: 如何处理数据库迁移？

A: 使用初始化容器进行迁移：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  template:
    spec:
      initContainers:
      - name: db-migration
        image: myapp:latest
        command: ["python", "manage.py", "migrate"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
      containers:
      - name: app
        image: myapp:latest
```

### Q: 如何实现零停机部署？

A: 使用滚动更新策略：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0
      maxSurge: 1
  template:
    spec:
      containers:
      - name: app
        image: myapp:latest
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

### Q: 如何管理配置和密钥？

A: 使用ConfigMap和Secret：

```python
# 配置管理
class ConfigManager:
    def __init__(self, k8s_client):
        self.k8s_client = k8s_client
    
    def create_config_map(self, name: str, namespace: str, data: dict):
        config_map = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": name,
                "namespace": namespace
            },
            "data": data
        }
        return config_map
    
    def create_secret(self, name: str, namespace: str, data: dict, secret_type: str = "Opaque"):
        import base64
        
        # 对数据进行base64编码
        encoded_data = {}
        for key, value in data.items():
            encoded_data[key] = base64.b64encode(value.encode()).decode()
        
        secret = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": name,
                "namespace": namespace
            },
            "type": secret_type,
            "data": encoded_data
        }
        return secret

# 使用配置管理
config_manager = ConfigManager(k8s_client)

# 创建配置
app_config = config_manager.create_config_map(
    "app-config",
    "production",
    {
        "LOG_LEVEL": "INFO",
        "CACHE_TTL": "3600",
        "MAX_CONNECTIONS": "100"
    }
)

# 创建密钥
app_secrets = config_manager.create_secret(
    "app-secrets",
    "production",
    {
        "DATABASE_PASSWORD": "super-secret-password",
        "JWT_SECRET": "jwt-signing-key",
        "API_KEY": "external-api-key"
    }
)
```

## 📚 相关文档

- [API开发模块使用指南](api.md)
- [数据库模块使用指南](database.md)
- [监控日志模块使用指南](monitoring.md)
- [缓存消息模块使用指南](cache.md)
- [项目结构最佳实践](best-practices/project-structure.md)
- [安全开发指南](best-practices/security.md)
- [性能优化建议](best-practices/performance.md)

---

如有其他问题，请查看 [GitHub Issues](https://github.com/your-username/python-backend-templates/issues) 或提交新的问题。