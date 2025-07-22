"""Kubernetes部署模板

提供完整的Kubernetes部署功能，包括：
- Deployment配置生成
- Service配置生成
- ConfigMap和Secret管理
- Ingress配置
- HPA自动扩缩容
- 命名空间管理
"""

import json
import yaml
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class ServiceType(str, Enum):
    """服务类型"""

    CLUSTER_IP = "ClusterIP"
    NODE_PORT = "NodePort"
    LOAD_BALANCER = "LoadBalancer"
    EXTERNAL_NAME = "ExternalName"


class RestartPolicy(str, Enum):
    """重启策略"""

    ALWAYS = "Always"
    ON_FAILURE = "OnFailure"
    NEVER = "Never"


class PullPolicy(str, Enum):
    """镜像拉取策略"""

    ALWAYS = "Always"
    IF_NOT_PRESENT = "IfNotPresent"
    NEVER = "Never"


class ProbeConfig(BaseModel):
    """探针配置"""

    path: str = "/health"
    port: int = 8000
    initial_delay_seconds: int = 30
    period_seconds: int = 10
    timeout_seconds: int = 5
    failure_threshold: int = 3
    success_threshold: int = 1


class ResourceConfig(BaseModel):
    """资源配置"""

    cpu_request: str = "100m"
    memory_request: str = "128Mi"
    cpu_limit: str = "500m"
    memory_limit: str = "512Mi"


class ContainerConfig(BaseModel):
    """容器配置"""

    name: str
    image: str
    tag: str = "latest"
    port: int = 8000
    pull_policy: PullPolicy = PullPolicy.IF_NOT_PRESENT

    # 环境变量
    env_vars: Dict[str, str] = Field(default_factory=dict)
    env_from_config_map: List[str] = Field(default_factory=list)
    env_from_secret: List[str] = Field(default_factory=list)

    # 资源配置
    resources: ResourceConfig = Field(default_factory=ResourceConfig)

    # 健康检查
    liveness_probe: Optional[ProbeConfig] = None
    readiness_probe: Optional[ProbeConfig] = None

    # 挂载卷
    volume_mounts: List[Dict[str, str]] = Field(default_factory=list)

    @property
    def full_image(self) -> str:
        """获取完整镜像名称"""
        return f"{self.image}:{self.tag}"


class DeploymentConfig(BaseModel):
    """Deployment配置"""

    name: str
    namespace: str = "default"
    replicas: int = 3

    # 容器配置
    containers: List[ContainerConfig]

    # 标签和选择器
    labels: Dict[str, str] = Field(default_factory=dict)
    selector_labels: Dict[str, str] = Field(default_factory=dict)

    # Pod配置
    restart_policy: RestartPolicy = RestartPolicy.ALWAYS

    # 卷配置
    volumes: List[Dict[str, Any]] = Field(default_factory=list)

    # 节点选择
    node_selector: Dict[str, str] = Field(default_factory=dict)

    # 容忍度
    tolerations: List[Dict[str, Any]] = Field(default_factory=list)

    # 亲和性
    affinity: Optional[Dict[str, Any]] = None

    @validator("replicas")
    def validate_replicas(cls, v):
        if v < 1:
            raise ValueError("Replicas must be at least 1")
        return v

    def __post_init__(self):
        if not self.selector_labels:
            self.selector_labels = {"app": self.name}
        if not self.labels:
            self.labels = self.selector_labels.copy()


class ServiceConfig(BaseModel):
    """Service配置"""

    name: str
    namespace: str = "default"
    service_type: ServiceType = ServiceType.CLUSTER_IP

    # 端口配置
    ports: List[Dict[str, Union[str, int]]] = Field(default_factory=list)

    # 选择器
    selector: Dict[str, str] = Field(default_factory=dict)

    # 标签
    labels: Dict[str, str] = Field(default_factory=dict)

    # NodePort特定配置
    node_port: Optional[int] = None

    # LoadBalancer特定配置
    load_balancer_ip: Optional[str] = None

    @validator("node_port")
    def validate_node_port(cls, v):
        if v is not None and not 30000 <= v <= 32767:
            raise ValueError("NodePort must be between 30000 and 32767")
        return v


class IngressConfig(BaseModel):
    """Ingress配置"""

    name: str
    namespace: str = "default"

    # 主机和路径规则
    rules: List[Dict[str, Any]] = Field(default_factory=list)

    # TLS配置
    tls: List[Dict[str, Any]] = Field(default_factory=list)

    # 注解
    annotations: Dict[str, str] = Field(default_factory=dict)

    # 标签
    labels: Dict[str, str] = Field(default_factory=dict)

    # Ingress类
    ingress_class: str = "nginx"


class HPAConfig(BaseModel):
    """HPA配置"""

    name: str
    namespace: str = "default"

    # 目标Deployment
    target_deployment: str

    # 副本数配置
    min_replicas: int = 1
    max_replicas: int = 10

    # CPU利用率目标
    target_cpu_utilization: int = 70

    # 内存利用率目标
    target_memory_utilization: Optional[int] = None

    # 自定义指标
    custom_metrics: List[Dict[str, Any]] = Field(default_factory=list)

    @validator("min_replicas")
    def validate_min_replicas(cls, v):
        if v < 1:
            raise ValueError("Min replicas must be at least 1")
        return v

    @validator("max_replicas")
    def validate_max_replicas(cls, v, values):
        if "min_replicas" in values and v < values["min_replicas"]:
            raise ValueError("Max replicas must be greater than min replicas")
        return v


class KubernetesGenerator:
    """Kubernetes配置生成器"""

    def __init__(self, output_dir: str = "k8s"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_namespace(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """生成Namespace配置"""
        namespace = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": name},
        }

        if labels:
            namespace["metadata"]["labels"] = labels

        return namespace

    def generate_config_map(
        self,
        name: str,
        namespace: str = "default",
        data: Optional[Dict[str, str]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """生成ConfigMap配置"""
        config_map = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": name, "namespace": namespace},
            "data": data or {},
        }

        if labels:
            config_map["metadata"]["labels"] = labels

        return config_map

    def generate_secret(
        self,
        name: str,
        namespace: str = "default",
        data: Optional[Dict[str, str]] = None,
        secret_type: str = "Opaque",
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """生成Secret配置"""
        secret = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": name, "namespace": namespace},
            "type": secret_type,
            "data": data or {},
        }

        if labels:
            secret["metadata"]["labels"] = labels

        return secret

    def generate_deployment(self, config: DeploymentConfig) -> Dict[str, Any]:
        """生成Deployment配置"""
        # 构建容器配置
        containers = []
        for container in config.containers:
            container_spec = {
                "name": container.name,
                "image": container.full_image,
                "imagePullPolicy": container.pull_policy.value,
                "ports": [{"containerPort": container.port, "protocol": "TCP"}],
            }

            # 环境变量
            env = []
            for key, value in container.env_vars.items():
                env.append({"name": key, "value": value})

            for config_map in container.env_from_config_map:
                env.append(
                    {
                        "name": config_map,
                        "valueFrom": {
                            "configMapKeyRef": {"name": config_map, "key": config_map}
                        },
                    }
                )

            for secret in container.env_from_secret:
                env.append(
                    {
                        "name": secret,
                        "valueFrom": {"secretKeyRef": {"name": secret, "key": secret}},
                    }
                )

            if env:
                container_spec["env"] = env

            # 资源配置
            container_spec["resources"] = {
                "requests": {
                    "cpu": container.resources.cpu_request,
                    "memory": container.resources.memory_request,
                },
                "limits": {
                    "cpu": container.resources.cpu_limit,
                    "memory": container.resources.memory_limit,
                },
            }

            # 健康检查
            if container.liveness_probe:
                container_spec["livenessProbe"] = self._build_probe(
                    container.liveness_probe
                )

            if container.readiness_probe:
                container_spec["readinessProbe"] = self._build_probe(
                    container.readiness_probe
                )

            # 卷挂载
            if container.volume_mounts:
                container_spec["volumeMounts"] = container.volume_mounts

            containers.append(container_spec)

        # 构建Pod模板
        pod_template = {
            "metadata": {"labels": config.labels},
            "spec": {
                "restartPolicy": config.restart_policy.value,
                "containers": containers,
            },
        }

        # 卷配置
        if config.volumes:
            pod_template["spec"]["volumes"] = config.volumes

        # 节点选择
        if config.node_selector:
            pod_template["spec"]["nodeSelector"] = config.node_selector

        # 容忍度
        if config.tolerations:
            pod_template["spec"]["tolerations"] = config.tolerations

        # 亲和性
        if config.affinity:
            pod_template["spec"]["affinity"] = config.affinity

        # 构建Deployment
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": config.name,
                "namespace": config.namespace,
                "labels": config.labels,
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {"matchLabels": config.selector_labels},
                "template": pod_template,
            },
        }

        return deployment

    def generate_service(self, config: ServiceConfig) -> Dict[str, Any]:
        """生成Service配置"""
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": config.name,
                "namespace": config.namespace,
                "labels": config.labels,
            },
            "spec": {
                "type": config.service_type.value,
                "selector": config.selector,
                "ports": config.ports,
            },
        }

        # NodePort特定配置
        if config.service_type == ServiceType.NODE_PORT and config.node_port:
            for port in service["spec"]["ports"]:
                port["nodePort"] = config.node_port

        # LoadBalancer特定配置
        if config.service_type == ServiceType.LOAD_BALANCER and config.load_balancer_ip:
            service["spec"]["loadBalancerIP"] = config.load_balancer_ip

        return service

    def generate_ingress(self, config: IngressConfig) -> Dict[str, Any]:
        """生成Ingress配置"""
        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": config.name,
                "namespace": config.namespace,
                "labels": config.labels,
                "annotations": config.annotations,
            },
            "spec": {"ingressClassName": config.ingress_class, "rules": config.rules},
        }

        if config.tls:
            ingress["spec"]["tls"] = config.tls

        return ingress

    def generate_hpa(self, config: HPAConfig) -> Dict[str, Any]:
        """生成HPA配置"""
        metrics = [
            {
                "type": "Resource",
                "resource": {
                    "name": "cpu",
                    "target": {
                        "type": "Utilization",
                        "averageUtilization": config.target_cpu_utilization,
                    },
                },
            }
        ]

        if config.target_memory_utilization:
            metrics.append(
                {
                    "type": "Resource",
                    "resource": {
                        "name": "memory",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": config.target_memory_utilization,
                        },
                    },
                }
            )

        # 添加自定义指标
        metrics.extend(config.custom_metrics)

        hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": config.name, "namespace": config.namespace},
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": config.target_deployment,
                },
                "minReplicas": config.min_replicas,
                "maxReplicas": config.max_replicas,
                "metrics": metrics,
            },
        }

        return hpa

    def _build_probe(self, probe: ProbeConfig) -> Dict[str, Any]:
        """构建探针配置"""
        return {
            "httpGet": {"path": probe.path, "port": probe.port},
            "initialDelaySeconds": probe.initial_delay_seconds,
            "periodSeconds": probe.period_seconds,
            "timeoutSeconds": probe.timeout_seconds,
            "failureThreshold": probe.failure_threshold,
            "successThreshold": probe.success_threshold,
        }

    def save_yaml(self, config: Dict[str, Any], filename: str) -> None:
        """保存YAML文件"""
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    def save_json(self, config: Dict[str, Any], filename: str) -> None:
        """保存JSON文件"""
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def generate_complete_app(
        self,
        app_name: str,
        image: str,
        namespace: str = "default",
        replicas: int = 3,
        port: int = 8000,
        domain: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """生成完整应用的所有配置"""
        configs = {}

        # 容器配置
        container = ContainerConfig(
            name=app_name,
            image=image,
            port=port,
            liveness_probe=ProbeConfig(port=port),
            readiness_probe=ProbeConfig(port=port),
        )

        # Deployment配置
        deployment_config = DeploymentConfig(
            name=app_name,
            namespace=namespace,
            replicas=replicas,
            containers=[container],
            labels={"app": app_name},
        )
        configs["deployment"] = self.generate_deployment(deployment_config)

        # Service配置
        service_config = ServiceConfig(
            name=f"{app_name}-service",
            namespace=namespace,
            ports=[{"name": "http", "port": 80, "targetPort": port, "protocol": "TCP"}],
            selector={"app": app_name},
            labels={"app": app_name},
        )
        configs["service"] = self.generate_service(service_config)

        # Ingress配置（如果提供了域名）
        if domain:
            ingress_config = IngressConfig(
                name=f"{app_name}-ingress",
                namespace=namespace,
                rules=[
                    {
                        "host": domain,
                        "http": {
                            "paths": [
                                {
                                    "path": "/",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": f"{app_name}-service",
                                            "port": {"number": 80},
                                        }
                                    },
                                }
                            ]
                        },
                    }
                ],
                labels={"app": app_name},
            )
            configs["ingress"] = self.generate_ingress(ingress_config)

        # HPA配置
        hpa_config = HPAConfig(
            name=f"{app_name}-hpa",
            namespace=namespace,
            target_deployment=app_name,
            min_replicas=1,
            max_replicas=10,
        )
        configs["hpa"] = self.generate_hpa(hpa_config)

        return configs

    def save_complete_app(
        self,
        app_name: str,
        image: str,
        namespace: str = "default",
        replicas: int = 3,
        port: int = 8000,
        domain: Optional[str] = None,
    ) -> None:
        """生成并保存完整应用配置"""
        configs = self.generate_complete_app(
            app_name, image, namespace, replicas, port, domain
        )

        for config_type, config in configs.items():
            filename = f"{app_name}-{config_type}.yaml"
            self.save_yaml(config, filename)


# 全局生成器
k8s_generator = KubernetesGenerator()


# 便捷函数
def generate_deployment(config: DeploymentConfig) -> Dict[str, Any]:
    """生成Deployment配置"""
    return k8s_generator.generate_deployment(config)


def generate_service(config: ServiceConfig) -> Dict[str, Any]:
    """生成Service配置"""
    return k8s_generator.generate_service(config)


def generate_ingress(config: IngressConfig) -> Dict[str, Any]:
    """生成Ingress配置"""
    return k8s_generator.generate_ingress(config)


def generate_hpa(config: HPAConfig) -> Dict[str, Any]:
    """生成HPA配置"""
    return k8s_generator.generate_hpa(config)


def generate_complete_app(
    app_name: str, image: str, **kwargs
) -> Dict[str, Dict[str, Any]]:
    """生成完整应用配置"""
    return k8s_generator.generate_complete_app(app_name, image, **kwargs)


# 使用示例
if __name__ == "__main__":
    # 创建生成器
    generator = KubernetesGenerator("k8s-configs")

    # 生成完整应用配置
    app_configs = generator.generate_complete_app(
        app_name="my-python-app",
        image="my-python-app",
        namespace="production",
        replicas=3,
        port=8000,
        domain="api.example.com",
    )

    # 保存配置文件
    for config_type, config in app_configs.items():
        filename = f"my-python-app-{config_type}.yaml"
        generator.save_yaml(config, filename)
        print(f"Generated {filename}")

    # 生成ConfigMap
    config_map = generator.generate_config_map(
        name="app-config",
        namespace="production",
        data={
            "DATABASE_URL": "postgresql://user:pass@db:5432/app",
            "REDIS_URL": "redis://redis:6379/0",
        },
    )
    generator.save_yaml(config_map, "app-configmap.yaml")

    # 生成Secret
    secret = generator.generate_secret(
        name="app-secrets",
        namespace="production",
        data={
            "SECRET_KEY": "base64-encoded-secret-key",
            "DB_PASSWORD": "base64-encoded-password",
        },
    )
    generator.save_yaml(secret, "app-secret.yaml")

    print("All Kubernetes configurations generated successfully!")
