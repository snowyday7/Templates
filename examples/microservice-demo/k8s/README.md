# Kubernetes 部署指南

本目录包含了微服务架构示例的完整 Kubernetes 部署配置。

## 📁 文件结构

```
k8s/
├── README.md              # 本文档
├── deploy.sh              # 自动化部署脚本
├── namespace.yaml         # 命名空间和资源配额
├── database.yaml          # PostgreSQL 和 Redis 部署
├── monitoring.yaml        # 监控栈 (Prometheus, Grafana, Jaeger)
└── api-gateway.yaml       # API 网关部署
```

## 🚀 快速开始

### 前置条件

1. **Kubernetes 集群**
   - 本地: Minikube, Kind, Docker Desktop
   - 云端: EKS, GKE, AKS
   - 最低版本: v1.20+

2. **工具安装**
   ```bash
   # kubectl
   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/darwin/amd64/kubectl"
   chmod +x kubectl
   sudo mv kubectl /usr/local/bin/
   
   # 验证安装
   kubectl version --client
   ```

3. **集群连接**
   ```bash
   # 检查集群连接
   kubectl cluster-info
   kubectl get nodes
   ```

### 一键部署

```bash
# 进入 k8s 目录
cd examples/microservice-demo/k8s

# 执行自动化部署
./deploy.sh
```

### 分步部署

1. **创建命名空间**
   ```bash
   ./deploy.sh namespace
   ```

2. **部署数据库**
   ```bash
   ./deploy.sh databases
   ```

3. **部署监控栈**
   ```bash
   ./deploy.sh monitoring
   ```

4. **部署 API 网关**
   ```bash
   ./deploy.sh gateway
   ```

## 📊 监控和观察

### Grafana 仪表板

```bash
# 端口转发
kubectl port-forward svc/grafana 3000:3000 -n microservice-demo

# 访问地址: http://localhost:3000
# 用户名: admin
# 密码: admin123
```

**预配置仪表板:**
- 微服务概览
- 请求速率和延迟
- 错误率监控
- 资源使用情况

### Prometheus 监控

```bash
# 端口转发
kubectl port-forward svc/prometheus 9090:9090 -n microservice-demo

# 访问地址: http://localhost:9090
```

**监控指标:**
- HTTP 请求指标
- 应用性能指标
- 基础设施指标
- 自定义业务指标

### Jaeger 链路追踪

```bash
# 端口转发
kubectl port-forward svc/jaeger 16686:16686 -n microservice-demo

# 访问地址: http://localhost:16686
```

**追踪功能:**
- 分布式请求追踪
- 服务依赖图
- 性能瓶颈分析
- 错误定位

## 🔧 配置说明

### 资源配额

```yaml
# namespace.yaml 中的配置
requests.cpu: "4"        # CPU 请求总量
requests.memory: 8Gi     # 内存请求总量
limits.cpu: "8"          # CPU 限制总量
limits.memory: 16Gi      # 内存限制总量
```

### 数据库配置

**PostgreSQL:**
- 版本: 15-alpine
- 存储: 10Gi PVC
- 多数据库支持
- 自动初始化脚本

**Redis:**
- 版本: 7-alpine
- 内存限制: 256MB
- LRU 淘汰策略
- 持久化配置

### 监控配置

**Prometheus:**
- 数据保留: 15天
- 抓取间隔: 15秒
- 自动服务发现
- 告警规则配置

**Grafana:**
- 自动配置数据源
- 预置仪表板
- 用户认证

## 🔍 故障排查

### 常见问题

1. **Pod 启动失败**
   ```bash
   # 查看 Pod 状态
   kubectl get pods -n microservice-demo
   
   # 查看 Pod 日志
   kubectl logs <pod-name> -n microservice-demo
   
   # 查看 Pod 事件
   kubectl describe pod <pod-name> -n microservice-demo
   ```

2. **服务无法访问**
   ```bash
   # 检查服务状态
   kubectl get svc -n microservice-demo
   
   # 检查端点
   kubectl get endpoints -n microservice-demo
   
   # 测试服务连通性
   kubectl run test-pod --image=busybox -it --rm -- /bin/sh
   ```

3. **存储问题**
   ```bash
   # 检查 PVC 状态
   kubectl get pvc -n microservice-demo
   
   # 检查存储类
   kubectl get storageclass
   ```

### 日志收集

```bash
# 收集所有 Pod 日志
for pod in $(kubectl get pods -n microservice-demo -o name); do
  echo "=== $pod ==="
  kubectl logs $pod -n microservice-demo
done
```

### 性能调优

1. **资源调整**
   ```yaml
   resources:
     requests:
       cpu: 200m      # 根据实际使用调整
       memory: 256Mi
     limits:
       cpu: 500m
       memory: 512Mi
   ```

2. **副本数调整**
   ```bash
   # 手动扩缩容
   kubectl scale deployment api-gateway --replicas=5 -n microservice-demo
   
   # 查看 HPA 状态
   kubectl get hpa -n microservice-demo
   ```

## 🔒 安全配置

### RBAC 权限

- ServiceAccount 配置
- ClusterRole 最小权限
- RoleBinding 绑定

### 密钥管理

```bash
# 查看密钥
kubectl get secrets -n microservice-demo

# 更新密钥
kubectl create secret generic jwt-secret \
  --from-literal=secret-key=your-new-secret \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 网络策略

```yaml
# 示例网络策略
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-gateway-policy
spec:
  podSelector:
    matchLabels:
      app: api-gateway
  policyTypes:
  - Ingress
  - Egress
```

## 📈 扩展和优化

### 水平扩缩容

- HPA 基于 CPU/内存
- 自定义指标扩缩容
- VPA 垂直扩缩容

### 滚动更新

```bash
# 更新镜像
kubectl set image deployment/api-gateway \
  api-gateway=microservice-demo/api-gateway:v2.0 \
  -n microservice-demo

# 查看更新状态
kubectl rollout status deployment/api-gateway -n microservice-demo

# 回滚更新
kubectl rollout undo deployment/api-gateway -n microservice-demo
```

### 健康检查

- Liveness Probe: 存活检查
- Readiness Probe: 就绪检查
- Startup Probe: 启动检查

## 🧹 清理资源

```bash
# 删除所有资源
./deploy.sh cleanup

# 或手动删除
kubectl delete namespace microservice-demo
```

## 📚 参考资料

- [Kubernetes 官方文档](https://kubernetes.io/docs/)
- [Prometheus 监控指南](https://prometheus.io/docs/)
- [Grafana 仪表板](https://grafana.com/docs/)
- [Jaeger 链路追踪](https://www.jaegertracing.io/docs/)
- [微服务最佳实践](https://microservices.io/)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个部署配置！

## 📄 许可证

本项目采用 MIT 许可证。