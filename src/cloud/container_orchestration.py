# -*- coding: utf-8 -*-
"""
云原生和容器化模块
提供Docker、Kubernetes集成、容器编排、服务网格等功能
"""

import asyncio
import json
import yaml
import time
import uuid
import base64
import tempfile
import shutil
from typing import (
    Dict, List, Any, Optional, Union, Tuple, Callable,
    Set, AsyncIterator, Iterator
)
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import os
from contextlib import asynccontextmanager
import tarfile
import io

# Docker SDK
import docker
from docker.models.containers import Container
from docker.models.images import Image
from docker.models.networks import Network
from docker.models.volumes import Volume

# Kubernetes客户端
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream

# 云服务SDK
import boto3
from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from google.cloud import container_v1

# 监控和日志
import structlog
from prometheus_client import Counter, Histogram, Gauge

# 配置管理
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

# 网络和HTTP
import aiohttp
from aiohttp import ClientSession, ClientTimeout

# 其他工具
import jinja2
from tenacity import retry, stop_after_attempt, wait_exponential
import schedule
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = structlog.get_logger(__name__)


class ContainerStatus(Enum):
    """容器状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"
    TERMINATING = "terminating"
    TERMINATED = "terminated"


class DeploymentStrategy(Enum):
    """部署策略枚举"""
    ROLLING_UPDATE = "rolling_update"
    RECREATE = "recreate"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    A_B_TESTING = "a_b_testing"


class ScalingPolicy(Enum):
    """扩缩容策略枚举"""
    MANUAL = "manual"
    AUTO_CPU = "auto_cpu"
    AUTO_MEMORY = "auto_memory"
    AUTO_CUSTOM = "auto_custom"
    PREDICTIVE = "predictive"


class CloudProvider(Enum):
    """云服务提供商枚举"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ALIBABA = "alibaba"
    TENCENT = "tencent"
    LOCAL = "local"


class NetworkPolicy(Enum):
    """网络策略枚举"""
    ALLOW_ALL = "allow_all"
    DENY_ALL = "deny_all"
    CUSTOM = "custom"


@dataclass
class ContainerConfig:
    """容器配置"""
    name: str
    image: str
    tag: str = "latest"
    command: Optional[List[str]] = None
    args: Optional[List[str]] = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    ports: List[int] = field(default_factory=list)
    volumes: Dict[str, str] = field(default_factory=dict)  # host_path: container_path
    cpu_request: Optional[str] = None
    cpu_limit: Optional[str] = None
    memory_request: Optional[str] = None
    memory_limit: Optional[str] = None
    restart_policy: str = "Always"
    health_check: Optional[Dict[str, Any]] = None
    security_context: Optional[Dict[str, Any]] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    @property
    def full_image(self) -> str:
        """获取完整镜像名"""
        return f"{self.image}:{self.tag}"


@dataclass
class ServiceConfig:
    """服务配置"""
    name: str
    selector: Dict[str, str]
    ports: List[Dict[str, Any]]  # [{"port": 80, "target_port": 8080, "protocol": "TCP"}]
    service_type: str = "ClusterIP"  # ClusterIP, NodePort, LoadBalancer
    external_ips: List[str] = field(default_factory=list)
    load_balancer_ip: Optional[str] = None
    session_affinity: str = "None"
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeploymentConfig:
    """部署配置"""
    name: str
    namespace: str = "default"
    replicas: int = 1
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE
    containers: List[ContainerConfig] = field(default_factory=list)
    service: Optional[ServiceConfig] = None
    ingress: Optional[Dict[str, Any]] = None
    config_maps: List[Dict[str, Any]] = field(default_factory=list)
    secrets: List[Dict[str, Any]] = field(default_factory=list)
    persistent_volumes: List[Dict[str, Any]] = field(default_factory=list)
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)
    affinity: Optional[Dict[str, Any]] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class AutoScalingConfig:
    """自动扩缩容配置"""
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: Optional[int] = 70
    target_memory_utilization: Optional[int] = 80
    custom_metrics: List[Dict[str, Any]] = field(default_factory=list)
    scale_up_policy: Optional[Dict[str, Any]] = None
    scale_down_policy: Optional[Dict[str, Any]] = None


@dataclass
class ClusterConfig:
    """集群配置"""
    name: str
    provider: CloudProvider
    region: str
    node_pools: List[Dict[str, Any]] = field(default_factory=list)
    network_config: Optional[Dict[str, Any]] = None
    security_config: Optional[Dict[str, Any]] = None
    addons: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)


class ContainerOrchestrator(ABC):
    """容器编排器抽象基类"""
    
    @abstractmethod
    async def deploy_application(self, config: DeploymentConfig) -> bool:
        """部署应用"""
        pass
    
    @abstractmethod
    async def update_application(self, config: DeploymentConfig) -> bool:
        """更新应用"""
        pass
    
    @abstractmethod
    async def delete_application(self, name: str, namespace: str = "default") -> bool:
        """删除应用"""
        pass
    
    @abstractmethod
    async def scale_application(self, name: str, replicas: int, namespace: str = "default") -> bool:
        """扩缩容应用"""
        pass
    
    @abstractmethod
    async def get_application_status(self, name: str, namespace: str = "default") -> Dict[str, Any]:
        """获取应用状态"""
        pass
    
    @abstractmethod
    async def get_logs(self, name: str, namespace: str = "default", lines: int = 100) -> str:
        """获取日志"""
        pass


class DockerOrchestrator(ContainerOrchestrator):
    """Docker容器编排器"""
    
    def __init__(self):
        self.client = docker.from_env()
        self.containers: Dict[str, Container] = {}
        self.networks: Dict[str, Network] = {}
        self.volumes: Dict[str, Volume] = {}
    
    async def deploy_application(self, config: DeploymentConfig) -> bool:
        """部署应用到Docker"""
        try:
            # 创建网络
            network_name = f"{config.name}-network"
            if network_name not in self.networks:
                network = self.client.networks.create(
                    name=network_name,
                    driver="bridge"
                )
                self.networks[network_name] = network
            
            # 部署容器
            for i in range(config.replicas):
                for container_config in config.containers:
                    container_name = f"{config.name}-{container_config.name}-{i}"
                    
                    # 构建容器参数
                    container_params = {
                        'image': container_config.full_image,
                        'name': container_name,
                        'environment': container_config.env_vars,
                        'labels': container_config.labels,
                        'network': network_name,
                        'detach': True,
                        'restart_policy': {'Name': container_config.restart_policy.lower()}
                    }
                    
                    # 添加端口映射
                    if container_config.ports:
                        ports = {}
                        for port in container_config.ports:
                            ports[f"{port}/tcp"] = port
                        container_params['ports'] = ports
                    
                    # 添加卷挂载
                    if container_config.volumes:
                        volumes = {}
                        for host_path, container_path in container_config.volumes.items():
                            volumes[host_path] = {'bind': container_path, 'mode': 'rw'}
                        container_params['volumes'] = volumes
                    
                    # 添加命令和参数
                    if container_config.command:
                        container_params['command'] = container_config.command
                    
                    # 创建并启动容器
                    container = self.client.containers.run(**container_params)
                    self.containers[container_name] = container
                    
                    logger.info(f"容器已部署: {container_name}")
            
            return True
        
        except Exception as e:
            logger.error(f"应用部署失败: {e}")
            return False
    
    async def update_application(self, config: DeploymentConfig) -> bool:
        """更新应用"""
        try:
            # 先删除旧容器
            await self.delete_application(config.name)
            
            # 重新部署
            return await self.deploy_application(config)
        
        except Exception as e:
            logger.error(f"应用更新失败: {e}")
            return False
    
    async def delete_application(self, name: str, namespace: str = "default") -> bool:
        """删除应用"""
        try:
            # 删除相关容器
            containers_to_remove = []
            for container_name, container in self.containers.items():
                if container_name.startswith(name):
                    containers_to_remove.append(container_name)
            
            for container_name in containers_to_remove:
                container = self.containers[container_name]
                container.stop()
                container.remove()
                del self.containers[container_name]
                logger.info(f"容器已删除: {container_name}")
            
            # 删除网络
            network_name = f"{name}-network"
            if network_name in self.networks:
                network = self.networks[network_name]
                network.remove()
                del self.networks[network_name]
                logger.info(f"网络已删除: {network_name}")
            
            return True
        
        except Exception as e:
            logger.error(f"应用删除失败: {e}")
            return False
    
    async def scale_application(self, name: str, replicas: int, namespace: str = "default") -> bool:
        """扩缩容应用"""
        try:
            # 获取当前容器数量
            current_containers = [c for c in self.containers.keys() if c.startswith(name)]
            current_replicas = len(current_containers)
            
            if replicas > current_replicas:
                # 扩容：创建新容器
                # 这里需要获取原始配置，简化处理
                logger.info(f"扩容应用 {name} 从 {current_replicas} 到 {replicas}")
            elif replicas < current_replicas:
                # 缩容：删除多余容器
                containers_to_remove = current_containers[replicas:]
                for container_name in containers_to_remove:
                    container = self.containers[container_name]
                    container.stop()
                    container.remove()
                    del self.containers[container_name]
                    logger.info(f"容器已删除: {container_name}")
            
            return True
        
        except Exception as e:
            logger.error(f"应用扩缩容失败: {e}")
            return False
    
    async def get_application_status(self, name: str, namespace: str = "default") -> Dict[str, Any]:
        """获取应用状态"""
        try:
            status = {
                'name': name,
                'containers': [],
                'total_containers': 0,
                'running_containers': 0
            }
            
            for container_name, container in self.containers.items():
                if container_name.startswith(name):
                    container.reload()
                    container_status = {
                        'name': container_name,
                        'status': container.status,
                        'image': container.image.tags[0] if container.image.tags else 'unknown',
                        'created': container.attrs['Created'],
                        'ports': container.ports
                    }
                    status['containers'].append(container_status)
                    status['total_containers'] += 1
                    
                    if container.status == 'running':
                        status['running_containers'] += 1
            
            return status
        
        except Exception as e:
            logger.error(f"获取应用状态失败: {e}")
            return {}
    
    async def get_logs(self, name: str, namespace: str = "default", lines: int = 100) -> str:
        """获取日志"""
        try:
            logs = []
            
            for container_name, container in self.containers.items():
                if container_name.startswith(name):
                    container_logs = container.logs(tail=lines, timestamps=True).decode('utf-8')
                    logs.append(f"=== {container_name} ===")
                    logs.append(container_logs)
                    logs.append("")
            
            return "\n".join(logs)
        
        except Exception as e:
            logger.error(f"获取日志失败: {e}")
            return ""


class KubernetesOrchestrator(ContainerOrchestrator):
    """Kubernetes容器编排器"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()
        
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        self.networking_v1 = client.NetworkingV1Api()
        self.autoscaling_v1 = client.AutoscalingV1Api()
        self.metrics_v1beta1 = client.CustomObjectsApi()
    
    async def deploy_application(self, config: DeploymentConfig) -> bool:
        """部署应用到Kubernetes"""
        try:
            # 创建命名空间（如果不存在）
            await self._ensure_namespace(config.namespace)
            
            # 创建ConfigMaps
            for cm_config in config.config_maps:
                await self._create_config_map(cm_config, config.namespace)
            
            # 创建Secrets
            for secret_config in config.secrets:
                await self._create_secret(secret_config, config.namespace)
            
            # 创建PersistentVolumes
            for pv_config in config.persistent_volumes:
                await self._create_persistent_volume(pv_config, config.namespace)
            
            # 创建Deployment
            deployment = self._build_deployment(config)
            self.apps_v1.create_namespaced_deployment(
                namespace=config.namespace,
                body=deployment
            )
            
            logger.info(f"Deployment已创建: {config.name}")
            
            # 创建Service
            if config.service:
                service = self._build_service(config.service, config.namespace)
                self.core_v1.create_namespaced_service(
                    namespace=config.namespace,
                    body=service
                )
                logger.info(f"Service已创建: {config.service.name}")
            
            # 创建Ingress
            if config.ingress:
                ingress = self._build_ingress(config.ingress, config.namespace)
                self.networking_v1.create_namespaced_ingress(
                    namespace=config.namespace,
                    body=ingress
                )
                logger.info(f"Ingress已创建: {config.ingress['name']}")
            
            return True
        
        except ApiException as e:
            logger.error(f"Kubernetes API错误: {e}")
            return False
        except Exception as e:
            logger.error(f"应用部署失败: {e}")
            return False
    
    async def update_application(self, config: DeploymentConfig) -> bool:
        """更新应用"""
        try:
            deployment = self._build_deployment(config)
            
            self.apps_v1.patch_namespaced_deployment(
                name=config.name,
                namespace=config.namespace,
                body=deployment
            )
            
            logger.info(f"Deployment已更新: {config.name}")
            return True
        
        except ApiException as e:
            logger.error(f"Kubernetes API错误: {e}")
            return False
        except Exception as e:
            logger.error(f"应用更新失败: {e}")
            return False
    
    async def delete_application(self, name: str, namespace: str = "default") -> bool:
        """删除应用"""
        try:
            # 删除Deployment
            self.apps_v1.delete_namespaced_deployment(
                name=name,
                namespace=namespace
            )
            
            # 删除Service
            try:
                self.core_v1.delete_namespaced_service(
                    name=name,
                    namespace=namespace
                )
            except ApiException:
                pass  # Service可能不存在
            
            # 删除Ingress
            try:
                self.networking_v1.delete_namespaced_ingress(
                    name=name,
                    namespace=namespace
                )
            except ApiException:
                pass  # Ingress可能不存在
            
            logger.info(f"应用已删除: {name}")
            return True
        
        except ApiException as e:
            logger.error(f"Kubernetes API错误: {e}")
            return False
        except Exception as e:
            logger.error(f"应用删除失败: {e}")
            return False
    
    async def scale_application(self, name: str, replicas: int, namespace: str = "default") -> bool:
        """扩缩容应用"""
        try:
            # 更新Deployment的副本数
            body = {'spec': {'replicas': replicas}}
            
            self.apps_v1.patch_namespaced_deployment_scale(
                name=name,
                namespace=namespace,
                body=body
            )
            
            logger.info(f"应用已扩缩容: {name} -> {replicas} replicas")
            return True
        
        except ApiException as e:
            logger.error(f"Kubernetes API错误: {e}")
            return False
        except Exception as e:
            logger.error(f"应用扩缩容失败: {e}")
            return False
    
    async def get_application_status(self, name: str, namespace: str = "default") -> Dict[str, Any]:
        """获取应用状态"""
        try:
            # 获取Deployment状态
            deployment = self.apps_v1.read_namespaced_deployment(
                name=name,
                namespace=namespace
            )
            
            # 获取Pods状态
            pods = self.core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"app={name}"
            )
            
            status = {
                'name': name,
                'namespace': namespace,
                'replicas': {
                    'desired': deployment.spec.replicas,
                    'current': deployment.status.replicas or 0,
                    'ready': deployment.status.ready_replicas or 0,
                    'available': deployment.status.available_replicas or 0
                },
                'conditions': [],
                'pods': []
            }
            
            # 添加Deployment条件
            if deployment.status.conditions:
                for condition in deployment.status.conditions:
                    status['conditions'].append({
                        'type': condition.type,
                        'status': condition.status,
                        'reason': condition.reason,
                        'message': condition.message,
                        'last_update_time': condition.last_update_time
                    })
            
            # 添加Pod信息
            for pod in pods.items:
                pod_status = {
                    'name': pod.metadata.name,
                    'phase': pod.status.phase,
                    'ready': False,
                    'restart_count': 0,
                    'node': pod.spec.node_name,
                    'created': pod.metadata.creation_timestamp
                }
                
                # 检查容器状态
                if pod.status.container_statuses:
                    for container_status in pod.status.container_statuses:
                        pod_status['restart_count'] += container_status.restart_count
                        if container_status.ready:
                            pod_status['ready'] = True
                
                status['pods'].append(pod_status)
            
            return status
        
        except ApiException as e:
            logger.error(f"Kubernetes API错误: {e}")
            return {}
        except Exception as e:
            logger.error(f"获取应用状态失败: {e}")
            return {}
    
    async def get_logs(self, name: str, namespace: str = "default", lines: int = 100) -> str:
        """获取日志"""
        try:
            # 获取Pods
            pods = self.core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"app={name}"
            )
            
            logs = []
            
            for pod in pods.items:
                pod_name = pod.metadata.name
                
                try:
                    pod_logs = self.core_v1.read_namespaced_pod_log(
                        name=pod_name,
                        namespace=namespace,
                        tail_lines=lines,
                        timestamps=True
                    )
                    
                    logs.append(f"=== Pod: {pod_name} ===")
                    logs.append(pod_logs)
                    logs.append("")
                
                except ApiException as e:
                    logs.append(f"=== Pod: {pod_name} (Error: {e.reason}) ===")
                    logs.append("")
            
            return "\n".join(logs)
        
        except ApiException as e:
            logger.error(f"Kubernetes API错误: {e}")
            return ""
        except Exception as e:
            logger.error(f"获取日志失败: {e}")
            return ""
    
    async def _ensure_namespace(self, namespace: str):
        """确保命名空间存在"""
        try:
            self.core_v1.read_namespace(name=namespace)
        except ApiException as e:
            if e.status == 404:
                # 创建命名空间
                ns_body = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=namespace)
                )
                self.core_v1.create_namespace(body=ns_body)
                logger.info(f"命名空间已创建: {namespace}")
            else:
                raise
    
    async def _create_config_map(self, cm_config: Dict[str, Any], namespace: str):
        """创建ConfigMap"""
        try:
            config_map = client.V1ConfigMap(
                metadata=client.V1ObjectMeta(
                    name=cm_config['name'],
                    namespace=namespace
                ),
                data=cm_config.get('data', {})
            )
            
            self.core_v1.create_namespaced_config_map(
                namespace=namespace,
                body=config_map
            )
            
            logger.info(f"ConfigMap已创建: {cm_config['name']}")
        
        except ApiException as e:
            if e.status != 409:  # 忽略已存在错误
                raise
    
    async def _create_secret(self, secret_config: Dict[str, Any], namespace: str):
        """创建Secret"""
        try:
            # 对数据进行base64编码
            encoded_data = {}
            for key, value in secret_config.get('data', {}).items():
                encoded_data[key] = base64.b64encode(value.encode()).decode()
            
            secret = client.V1Secret(
                metadata=client.V1ObjectMeta(
                    name=secret_config['name'],
                    namespace=namespace
                ),
                type=secret_config.get('type', 'Opaque'),
                data=encoded_data
            )
            
            self.core_v1.create_namespaced_secret(
                namespace=namespace,
                body=secret
            )
            
            logger.info(f"Secret已创建: {secret_config['name']}")
        
        except ApiException as e:
            if e.status != 409:  # 忽略已存在错误
                raise
    
    async def _create_persistent_volume(self, pv_config: Dict[str, Any], namespace: str):
        """创建PersistentVolume"""
        try:
            # 创建PVC
            pvc = client.V1PersistentVolumeClaim(
                metadata=client.V1ObjectMeta(
                    name=pv_config['name'],
                    namespace=namespace
                ),
                spec=client.V1PersistentVolumeClaimSpec(
                    access_modes=pv_config.get('access_modes', ['ReadWriteOnce']),
                    resources=client.V1ResourceRequirements(
                        requests={'storage': pv_config.get('size', '1Gi')}
                    ),
                    storage_class_name=pv_config.get('storage_class')
                )
            )
            
            self.core_v1.create_namespaced_persistent_volume_claim(
                namespace=namespace,
                body=pvc
            )
            
            logger.info(f"PVC已创建: {pv_config['name']}")
        
        except ApiException as e:
            if e.status != 409:  # 忽略已存在错误
                raise
    
    def _build_deployment(self, config: DeploymentConfig) -> client.V1Deployment:
        """构建Deployment对象"""
        # 构建容器列表
        containers = []
        for container_config in config.containers:
            container = client.V1Container(
                name=container_config.name,
                image=container_config.full_image,
                command=container_config.command,
                args=container_config.args,
                env=[
                    client.V1EnvVar(name=k, value=v)
                    for k, v in container_config.env_vars.items()
                ],
                ports=[
                    client.V1ContainerPort(container_port=port)
                    for port in container_config.ports
                ],
                resources=client.V1ResourceRequirements(
                    requests={
                        k: v for k, v in {
                            'cpu': container_config.cpu_request,
                            'memory': container_config.memory_request
                        }.items() if v
                    },
                    limits={
                        k: v for k, v in {
                            'cpu': container_config.cpu_limit,
                            'memory': container_config.memory_limit
                        }.items() if v
                    }
                )
            )
            
            # 添加健康检查
            if container_config.health_check:
                hc = container_config.health_check
                if hc.get('http'):
                    container.liveness_probe = client.V1Probe(
                        http_get=client.V1HTTPGetAction(
                            path=hc['http']['path'],
                            port=hc['http']['port']
                        ),
                        initial_delay_seconds=hc.get('initial_delay', 30),
                        period_seconds=hc.get('period', 10)
                    )
            
            containers.append(container)
        
        # 构建Pod模板
        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels=dict(config.labels, app=config.name),
                annotations=config.annotations
            ),
            spec=client.V1PodSpec(
                containers=containers,
                restart_policy=config.containers[0].restart_policy if config.containers else "Always",
                node_selector=config.node_selector,
                tolerations=[
                    client.V1Toleration(**toleration)
                    for toleration in config.tolerations
                ] if config.tolerations else None,
                affinity=client.V1Affinity(**config.affinity) if config.affinity else None
            )
        )
        
        # 构建部署策略
        strategy = None
        if config.strategy == DeploymentStrategy.ROLLING_UPDATE:
            strategy = client.V1DeploymentStrategy(
                type="RollingUpdate",
                rolling_update=client.V1RollingUpdateDeployment(
                    max_surge="25%",
                    max_unavailable="25%"
                )
            )
        elif config.strategy == DeploymentStrategy.RECREATE:
            strategy = client.V1DeploymentStrategy(type="Recreate")
        
        # 构建Deployment
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(
                name=config.name,
                namespace=config.namespace,
                labels=dict(config.labels, app=config.name),
                annotations=config.annotations
            ),
            spec=client.V1DeploymentSpec(
                replicas=config.replicas,
                selector=client.V1LabelSelector(
                    match_labels={'app': config.name}
                ),
                template=pod_template,
                strategy=strategy
            )
        )
        
        return deployment
    
    def _build_service(self, service_config: ServiceConfig, namespace: str) -> client.V1Service:
        """构建Service对象"""
        ports = [
            client.V1ServicePort(
                name=port_config.get('name', f"port-{port_config['port']}"),
                port=port_config['port'],
                target_port=port_config.get('target_port', port_config['port']),
                protocol=port_config.get('protocol', 'TCP')
            )
            for port_config in service_config.ports
        ]
        
        service = client.V1Service(
            metadata=client.V1ObjectMeta(
                name=service_config.name,
                namespace=namespace,
                labels=service_config.labels,
                annotations=service_config.annotations
            ),
            spec=client.V1ServiceSpec(
                selector=service_config.selector,
                ports=ports,
                type=service_config.service_type,
                external_i_ps=service_config.external_ips if service_config.external_ips else None,
                load_balancer_ip=service_config.load_balancer_ip,
                session_affinity=service_config.session_affinity
            )
        )
        
        return service
    
    def _build_ingress(self, ingress_config: Dict[str, Any], namespace: str) -> client.V1Ingress:
        """构建Ingress对象"""
        rules = []
        
        for rule_config in ingress_config.get('rules', []):
            paths = []
            for path_config in rule_config.get('paths', []):
                path = client.V1HTTPIngressPath(
                    path=path_config['path'],
                    path_type=path_config.get('path_type', 'Prefix'),
                    backend=client.V1IngressBackend(
                        service=client.V1IngressServiceBackend(
                            name=path_config['service']['name'],
                            port=client.V1ServiceBackendPort(
                                number=path_config['service']['port']
                            )
                        )
                    )
                )
                paths.append(path)
            
            rule = client.V1IngressRule(
                host=rule_config.get('host'),
                http=client.V1HTTPIngressRuleValue(paths=paths)
            )
            rules.append(rule)
        
        ingress = client.V1Ingress(
            metadata=client.V1ObjectMeta(
                name=ingress_config['name'],
                namespace=namespace,
                labels=ingress_config.get('labels', {}),
                annotations=ingress_config.get('annotations', {})
            ),
            spec=client.V1IngressSpec(
                rules=rules,
                tls=[
                    client.V1IngressTLS(
                        hosts=tls_config['hosts'],
                        secret_name=tls_config['secret_name']
                    )
                    for tls_config in ingress_config.get('tls', [])
                ] if ingress_config.get('tls') else None
            )
        )
        
        return ingress


class AutoScaler:
    """自动扩缩容器"""
    
    def __init__(self, orchestrator: ContainerOrchestrator):
        self.orchestrator = orchestrator
        self.scaling_configs: Dict[str, AutoScalingConfig] = {}
        self.metrics_client = None
        self.scheduler = AsyncIOScheduler()
        self.scaling_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_scaling_config(self, app_name: str, config: AutoScalingConfig):
        """添加扩缩容配置"""
        self.scaling_configs[app_name] = config
        logger.info(f"扩缩容配置已添加: {app_name}")
    
    async def start_monitoring(self):
        """开始监控"""
        self.scheduler.add_job(
            self._check_scaling,
            'interval',
            seconds=30,
            id='auto_scaling_check'
        )
        
        self.scheduler.start()
        logger.info("自动扩缩容监控已启动")
    
    async def stop_monitoring(self):
        """停止监控"""
        self.scheduler.shutdown()
        logger.info("自动扩缩容监控已停止")
    
    async def _check_scaling(self):
        """检查扩缩容"""
        for app_name, config in self.scaling_configs.items():
            try:
                await self._evaluate_scaling(app_name, config)
            except Exception as e:
                logger.error(f"扩缩容检查失败: {app_name}, {e}")
    
    async def _evaluate_scaling(self, app_name: str, config: AutoScalingConfig):
        """评估扩缩容"""
        # 获取当前状态
        status = await self.orchestrator.get_application_status(app_name)
        if not status:
            return
        
        current_replicas = status.get('replicas', {}).get('current', 0)
        if current_replicas == 0:
            return
        
        # 获取指标
        metrics = await self._get_metrics(app_name)
        if not metrics:
            return
        
        # 计算目标副本数
        target_replicas = await self._calculate_target_replicas(
            current_replicas, metrics, config
        )
        
        # 应用扩缩容限制
        target_replicas = max(config.min_replicas, min(config.max_replicas, target_replicas))
        
        # 检查是否需要扩缩容
        if target_replicas != current_replicas:
            # 记录扩缩容历史
            scaling_event = {
                'timestamp': datetime.now(),
                'from_replicas': current_replicas,
                'to_replicas': target_replicas,
                'reason': 'auto_scaling',
                'metrics': metrics
            }
            
            if app_name not in self.scaling_history:
                self.scaling_history[app_name] = []
            
            self.scaling_history[app_name].append(scaling_event)
            
            # 执行扩缩容
            success = await self.orchestrator.scale_application(app_name, target_replicas)
            
            if success:
                logger.info(
                    f"自动扩缩容完成: {app_name} {current_replicas} -> {target_replicas}",
                    metrics=metrics
                )
            else:
                logger.error(f"自动扩缩容失败: {app_name}")
    
    async def _get_metrics(self, app_name: str) -> Dict[str, float]:
        """获取指标"""
        # 这里应该从监控系统获取实际指标
        # 简化实现，返回模拟数据
        import random
        
        return {
            'cpu_utilization': random.uniform(20, 90),
            'memory_utilization': random.uniform(30, 80),
            'request_rate': random.uniform(10, 100),
            'response_time': random.uniform(100, 1000)
        }
    
    async def _calculate_target_replicas(self, current_replicas: int, 
                                       metrics: Dict[str, float], 
                                       config: AutoScalingConfig) -> int:
        """计算目标副本数"""
        target_replicas = current_replicas
        
        # CPU扩缩容
        if config.target_cpu_utilization and 'cpu_utilization' in metrics:
            cpu_ratio = metrics['cpu_utilization'] / config.target_cpu_utilization
            cpu_target = int(current_replicas * cpu_ratio)
            target_replicas = max(target_replicas, cpu_target)
        
        # 内存扩缩容
        if config.target_memory_utilization and 'memory_utilization' in metrics:
            memory_ratio = metrics['memory_utilization'] / config.target_memory_utilization
            memory_target = int(current_replicas * memory_ratio)
            target_replicas = max(target_replicas, memory_target)
        
        # 自定义指标扩缩容
        for custom_metric in config.custom_metrics:
            metric_name = custom_metric['name']
            target_value = custom_metric['target_value']
            
            if metric_name in metrics:
                metric_ratio = metrics[metric_name] / target_value
                metric_target = int(current_replicas * metric_ratio)
                target_replicas = max(target_replicas, metric_target)
        
        return target_replicas


class ContainerManager:
    """容器管理器"""
    
    def __init__(self, orchestrator: ContainerOrchestrator):
        self.orchestrator = orchestrator
        self.auto_scaler = AutoScaler(orchestrator)
        self.deployments: Dict[str, DeploymentConfig] = {}
        self.metrics = {
            'deployments_total': Counter('container_deployments_total'),
            'deployment_duration': Histogram('container_deployment_duration_seconds'),
            'scaling_events': Counter('container_scaling_events_total')
        }
    
    async def deploy(self, config: DeploymentConfig) -> bool:
        """部署应用"""
        start_time = time.time()
        
        try:
            success = await self.orchestrator.deploy_application(config)
            
            if success:
                self.deployments[config.name] = config
                self.metrics['deployments_total'].labels(status='success').inc()
                logger.info(f"应用部署成功: {config.name}")
            else:
                self.metrics['deployments_total'].labels(status='failure').inc()
                logger.error(f"应用部署失败: {config.name}")
            
            duration = time.time() - start_time
            self.metrics['deployment_duration'].observe(duration)
            
            return success
        
        except Exception as e:
            self.metrics['deployments_total'].labels(status='error').inc()
            logger.error(f"应用部署异常: {config.name}, {e}")
            return False
    
    async def update(self, config: DeploymentConfig) -> bool:
        """更新应用"""
        try:
            success = await self.orchestrator.update_application(config)
            
            if success:
                self.deployments[config.name] = config
                logger.info(f"应用更新成功: {config.name}")
            else:
                logger.error(f"应用更新失败: {config.name}")
            
            return success
        
        except Exception as e:
            logger.error(f"应用更新异常: {config.name}, {e}")
            return False
    
    async def delete(self, name: str, namespace: str = "default") -> bool:
        """删除应用"""
        try:
            success = await self.orchestrator.delete_application(name, namespace)
            
            if success:
                self.deployments.pop(name, None)
                logger.info(f"应用删除成功: {name}")
            else:
                logger.error(f"应用删除失败: {name}")
            
            return success
        
        except Exception as e:
            logger.error(f"应用删除异常: {name}, {e}")
            return False
    
    async def scale(self, name: str, replicas: int, namespace: str = "default") -> bool:
        """扩缩容应用"""
        try:
            success = await self.orchestrator.scale_application(name, replicas, namespace)
            
            if success:
                self.metrics['scaling_events'].labels(app=name).inc()
                logger.info(f"应用扩缩容成功: {name} -> {replicas}")
            else:
                logger.error(f"应用扩缩容失败: {name}")
            
            return success
        
        except Exception as e:
            logger.error(f"应用扩缩容异常: {name}, {e}")
            return False
    
    async def get_status(self, name: str, namespace: str = "default") -> Dict[str, Any]:
        """获取应用状态"""
        return await self.orchestrator.get_application_status(name, namespace)
    
    async def get_logs(self, name: str, namespace: str = "default", lines: int = 100) -> str:
        """获取应用日志"""
        return await self.orchestrator.get_logs(name, namespace, lines)
    
    def enable_auto_scaling(self, app_name: str, config: AutoScalingConfig):
        """启用自动扩缩容"""
        self.auto_scaler.add_scaling_config(app_name, config)
    
    async def start_auto_scaling(self):
        """启动自动扩缩容"""
        await self.auto_scaler.start_monitoring()
    
    async def stop_auto_scaling(self):
        """停止自动扩缩容"""
        await self.auto_scaler.stop_monitoring()
    
    def list_deployments(self) -> List[str]:
        """列出所有部署"""
        return list(self.deployments.keys())
    
    def get_deployment_config(self, name: str) -> Optional[DeploymentConfig]:
        """获取部署配置"""
        return self.deployments.get(name)


# 示例使用
async def example_usage():
    """示例用法"""
    # 创建容器配置
    container_config = ContainerConfig(
        name="web-server",
        image="nginx",
        tag="1.21",
        ports=[80],
        env_vars={"ENV": "production"},
        cpu_request="100m",
        cpu_limit="500m",
        memory_request="128Mi",
        memory_limit="512Mi",
        health_check={
            "http": {"path": "/health", "port": 80},
            "initial_delay": 30,
            "period": 10
        }
    )
    
    # 创建服务配置
    service_config = ServiceConfig(
        name="web-service",
        selector={"app": "web-app"},
        ports=[{"port": 80, "target_port": 80}],
        service_type="LoadBalancer"
    )
    
    # 创建部署配置
    deployment_config = DeploymentConfig(
        name="web-app",
        namespace="default",
        replicas=3,
        strategy=DeploymentStrategy.ROLLING_UPDATE,
        containers=[container_config],
        service=service_config,
        labels={"app": "web-app", "version": "v1"}
    )
    
    # 创建自动扩缩容配置
    auto_scaling_config = AutoScalingConfig(
        min_replicas=2,
        max_replicas=10,
        target_cpu_utilization=70,
        target_memory_utilization=80
    )
    
    # 选择编排器（Kubernetes或Docker）
    # orchestrator = DockerOrchestrator()
    orchestrator = KubernetesOrchestrator()
    
    # 创建容器管理器
    manager = ContainerManager(orchestrator)
    
    # 部署应用
    success = await manager.deploy(deployment_config)
    if success:
        print("应用部署成功")
        
        # 启用自动扩缩容
        manager.enable_auto_scaling("web-app", auto_scaling_config)
        await manager.start_auto_scaling()
        
        # 获取应用状态
        status = await manager.get_status("web-app")
        print(f"应用状态: {status}")
        
        # 手动扩缩容
        await manager.scale("web-app", 5)
        
        # 获取日志
        logs = await manager.get_logs("web-app", lines=50)
        print(f"应用日志: {logs[:500]}...")  # 只显示前500字符
        
        # 等待一段时间
        await asyncio.sleep(60)
        
        # 停止自动扩缩容
        await manager.stop_auto_scaling()
        
        # 删除应用
        await manager.delete("web-app")
        print("应用已删除")
    else:
        print("应用部署失败")


if __name__ == "__main__":
    asyncio.run(example_usage())