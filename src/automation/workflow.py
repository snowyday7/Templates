# -*- coding: utf-8 -*-
"""
工作流和自动化模块
提供任务调度、工作流编排、自动化脚本执行等功能
"""

import asyncio
import json
import yaml
import uuid
import time
import threading
from typing import (
    Dict, List, Any, Optional, Union, Callable, Awaitable,
    TypeVar, Generic, Set, Tuple
)
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import shlex
import os
import signal
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import multiprocessing
from queue import Queue, PriorityQueue
import heapq
from contextlib import asynccontextmanager
import croniter
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor as APSThreadPoolExecutor
from apscheduler.executors.asyncio import AsyncIOExecutor
import redis
import aioredis
from celery import Celery
from kombu import Queue as KombuQueue
import pika
import requests
import aiohttp
from jinja2 import Template, Environment, FileSystemLoader
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import slack_sdk
from slack_sdk.web.async_client import AsyncWebClient
import telegram
from telegram.ext import Application
import discord
from discord.ext import commands
import paramiko
from fabric import Connection
import docker
import kubernetes
from kubernetes import client, config
import boto3
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from google.cloud import compute_v1
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from motor.motor_asyncio import AsyncIOMotorClient
import pymongo
from elasticsearch import AsyncElasticsearch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
from passlib.context import CryptContext
import pyotp
from functools import wraps, partial
from contextlib import contextmanager
import tempfile
import shutil
import zipfile
import tarfile
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import schedule
from retrying import retry
from tenacity import retry as tenacity_retry, stop_after_attempt, wait_exponential
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

logger = structlog.get_logger(__name__)
console = Console()

T = TypeVar('T')


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class WorkflowStatus(Enum):
    """工作流状态枚举"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TriggerType(Enum):
    """触发器类型枚举"""
    MANUAL = "manual"
    CRON = "cron"
    INTERVAL = "interval"
    EVENT = "event"
    WEBHOOK = "webhook"
    FILE_CHANGE = "file_change"
    API_CALL = "api_call"
    DATABASE_CHANGE = "database_change"


class ExecutionMode(Enum):
    """执行模式枚举"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    BATCH = "batch"


class NotificationType(Enum):
    """通知类型枚举"""
    EMAIL = "email"
    SLACK = "slack"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    SMS = "sms"
    PUSH = "push"


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    logs: List[str] = field(default_factory=list)


@dataclass
class WorkflowResult:
    """工作流执行结果"""
    workflow_id: str
    status: WorkflowStatus
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class TaskConfig:
    """任务配置"""
    name: str
    command: Optional[str] = None
    function: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    retry_count: int = 0
    retry_delay: int = 1
    dependencies: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    working_directory: Optional[str] = None
    priority: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowConfig:
    """工作流配置"""
    name: str
    description: str = ""
    tasks: List[TaskConfig] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    max_parallel_tasks: int = 5
    timeout: Optional[int] = None
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    notifications: List[Dict[str, Any]] = field(default_factory=list)
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)


class Task(ABC):
    """任务抽象基类"""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.id = str(uuid.uuid4())
        self.status = TaskStatus.PENDING
        self.result: Optional[TaskResult] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.logs: List[str] = []
        self.retry_count = 0
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any] = None) -> TaskResult:
        """执行任务"""
        pass
    
    def add_log(self, message: str, level: str = "INFO"):
        """添加日志"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)
        logger.info(message, task_id=self.id, task_name=self.config.name)
    
    def can_execute(self, completed_tasks: Set[str]) -> bool:
        """检查是否可以执行"""
        return all(dep in completed_tasks for dep in self.config.dependencies)
    
    def should_retry(self) -> bool:
        """检查是否应该重试"""
        return (
            self.status == TaskStatus.FAILED and
            self.retry_count < self.config.retry_count
        )


class CommandTask(Task):
    """命令行任务"""
    
    async def execute(self, context: Dict[str, Any] = None) -> TaskResult:
        """执行命令行任务"""
        self.status = TaskStatus.RUNNING
        self.start_time = datetime.now()
        
        try:
            # 准备环境变量
            env = os.environ.copy()
            env.update(self.config.environment)
            
            # 替换命令中的变量
            command = self.config.command
            if context:
                for key, value in context.items():
                    command = command.replace(f"${{{key}}}", str(value))
            
            self.add_log(f"执行命令: {command}")
            
            # 执行命令
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.working_directory
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.timeout
                )
                
                output = stdout.decode('utf-8') if stdout else ""
                error = stderr.decode('utf-8') if stderr else ""
                
                if process.returncode == 0:
                    self.status = TaskStatus.SUCCESS
                    self.add_log(f"命令执行成功: {output}")
                else:
                    self.status = TaskStatus.FAILED
                    self.add_log(f"命令执行失败: {error}", "ERROR")
                
                self.result = TaskResult(
                    task_id=self.id,
                    status=self.status,
                    output=output,
                    error=error if process.returncode != 0 else None,
                    start_time=self.start_time,
                    end_time=datetime.now(),
                    logs=self.logs.copy(),
                    retry_count=self.retry_count
                )
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                self.status = TaskStatus.TIMEOUT
                self.add_log("任务执行超时", "ERROR")
                
                self.result = TaskResult(
                    task_id=self.id,
                    status=self.status,
                    error="Task execution timeout",
                    start_time=self.start_time,
                    end_time=datetime.now(),
                    logs=self.logs.copy(),
                    retry_count=self.retry_count
                )
        
        except Exception as e:
            self.status = TaskStatus.FAILED
            error_msg = str(e)
            self.add_log(f"任务执行异常: {error_msg}", "ERROR")
            
            self.result = TaskResult(
                task_id=self.id,
                status=self.status,
                error=error_msg,
                start_time=self.start_time,
                end_time=datetime.now(),
                logs=self.logs.copy(),
                retry_count=self.retry_count
            )
        
        finally:
            self.end_time = datetime.now()
            if self.result:
                self.result.duration = (self.end_time - self.start_time).total_seconds()
        
        return self.result


class FunctionTask(Task):
    """函数任务"""
    
    async def execute(self, context: Dict[str, Any] = None) -> TaskResult:
        """执行函数任务"""
        self.status = TaskStatus.RUNNING
        self.start_time = datetime.now()
        
        try:
            self.add_log(f"执行函数: {self.config.function.__name__}")
            
            # 准备参数
            kwargs = self.config.parameters.copy()
            if context:
                kwargs.update(context)
            
            # 执行函数
            if asyncio.iscoroutinefunction(self.config.function):
                if self.config.timeout:
                    output = await asyncio.wait_for(
                        self.config.function(**kwargs),
                        timeout=self.config.timeout
                    )
                else:
                    output = await self.config.function(**kwargs)
            else:
                # 在线程池中执行同步函数
                loop = asyncio.get_event_loop()
                if self.config.timeout:
                    output = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            partial(self.config.function, **kwargs)
                        ),
                        timeout=self.config.timeout
                    )
                else:
                    output = await loop.run_in_executor(
                        None,
                        partial(self.config.function, **kwargs)
                    )
            
            self.status = TaskStatus.SUCCESS
            self.add_log(f"函数执行成功")
            
            self.result = TaskResult(
                task_id=self.id,
                status=self.status,
                output=output,
                start_time=self.start_time,
                end_time=datetime.now(),
                logs=self.logs.copy(),
                retry_count=self.retry_count
            )
        
        except asyncio.TimeoutError:
            self.status = TaskStatus.TIMEOUT
            self.add_log("函数执行超时", "ERROR")
            
            self.result = TaskResult(
                task_id=self.id,
                status=self.status,
                error="Function execution timeout",
                start_time=self.start_time,
                end_time=datetime.now(),
                logs=self.logs.copy(),
                retry_count=self.retry_count
            )
        
        except Exception as e:
            self.status = TaskStatus.FAILED
            error_msg = str(e)
            self.add_log(f"函数执行异常: {error_msg}", "ERROR")
            
            self.result = TaskResult(
                task_id=self.id,
                status=self.status,
                error=error_msg,
                start_time=self.start_time,
                end_time=datetime.now(),
                logs=self.logs.copy(),
                retry_count=self.retry_count
            )
        
        finally:
            self.end_time = datetime.now()
            if self.result:
                self.result.duration = (self.end_time - self.start_time).total_seconds()
        
        return self.result


class HTTPTask(Task):
    """HTTP请求任务"""
    
    async def execute(self, context: Dict[str, Any] = None) -> TaskResult:
        """执行HTTP请求任务"""
        self.status = TaskStatus.RUNNING
        self.start_time = datetime.now()
        
        try:
            # 准备请求参数
            params = self.config.parameters.copy()
            url = params.get('url')
            method = params.get('method', 'GET').upper()
            headers = params.get('headers', {})
            data = params.get('data')
            json_data = params.get('json')
            
            if not url:
                raise ValueError("URL is required for HTTP task")
            
            self.add_log(f"发送HTTP请求: {method} {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=data,
                    json=json_data,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    response_text = await response.text()
                    
                    if response.status < 400:
                        self.status = TaskStatus.SUCCESS
                        self.add_log(f"HTTP请求成功: {response.status}")
                        
                        output = {
                            'status_code': response.status,
                            'headers': dict(response.headers),
                            'content': response_text
                        }
                        
                        try:
                            output['json'] = await response.json()
                        except:
                            pass
                    else:
                        self.status = TaskStatus.FAILED
                        self.add_log(f"HTTP请求失败: {response.status}", "ERROR")
                        output = None
            
            self.result = TaskResult(
                task_id=self.id,
                status=self.status,
                output=output,
                error=None if self.status == TaskStatus.SUCCESS else f"HTTP {response.status}",
                start_time=self.start_time,
                end_time=datetime.now(),
                logs=self.logs.copy(),
                retry_count=self.retry_count
            )
        
        except Exception as e:
            self.status = TaskStatus.FAILED
            error_msg = str(e)
            self.add_log(f"HTTP请求异常: {error_msg}", "ERROR")
            
            self.result = TaskResult(
                task_id=self.id,
                status=self.status,
                error=error_msg,
                start_time=self.start_time,
                end_time=datetime.now(),
                logs=self.logs.copy(),
                retry_count=self.retry_count
            )
        
        finally:
            self.end_time = datetime.now()
            if self.result:
                self.result.duration = (self.end_time - self.start_time).total_seconds()
        
        return self.result


class DatabaseTask(Task):
    """数据库任务"""
    
    async def execute(self, context: Dict[str, Any] = None) -> TaskResult:
        """执行数据库任务"""
        self.status = TaskStatus.RUNNING
        self.start_time = datetime.now()
        
        try:
            params = self.config.parameters.copy()
            connection_string = params.get('connection_string')
            query = params.get('query')
            operation = params.get('operation', 'select')  # select, insert, update, delete
            
            if not connection_string or not query:
                raise ValueError("Connection string and query are required")
            
            self.add_log(f"执行数据库操作: {operation}")
            
            # 创建数据库连接
            engine = sa.create_engine(connection_string)
            
            with engine.connect() as conn:
                if operation.lower() == 'select':
                    result = conn.execute(sa.text(query))
                    output = [dict(row) for row in result]
                else:
                    result = conn.execute(sa.text(query))
                    conn.commit()
                    output = {'affected_rows': result.rowcount}
            
            self.status = TaskStatus.SUCCESS
            self.add_log(f"数据库操作成功")
            
            self.result = TaskResult(
                task_id=self.id,
                status=self.status,
                output=output,
                start_time=self.start_time,
                end_time=datetime.now(),
                logs=self.logs.copy(),
                retry_count=self.retry_count
            )
        
        except Exception as e:
            self.status = TaskStatus.FAILED
            error_msg = str(e)
            self.add_log(f"数据库操作异常: {error_msg}", "ERROR")
            
            self.result = TaskResult(
                task_id=self.id,
                status=self.status,
                error=error_msg,
                start_time=self.start_time,
                end_time=datetime.now(),
                logs=self.logs.copy(),
                retry_count=self.retry_count
            )
        
        finally:
            self.end_time = datetime.now()
            if self.result:
                self.result.duration = (self.end_time - self.start_time).total_seconds()
        
        return self.result


class FileTask(Task):
    """文件操作任务"""
    
    async def execute(self, context: Dict[str, Any] = None) -> TaskResult:
        """执行文件操作任务"""
        self.status = TaskStatus.RUNNING
        self.start_time = datetime.now()
        
        try:
            params = self.config.parameters.copy()
            operation = params.get('operation')  # copy, move, delete, create, read, write
            source = params.get('source')
            destination = params.get('destination')
            content = params.get('content')
            
            self.add_log(f"执行文件操作: {operation}")
            
            output = None
            
            if operation == 'copy':
                shutil.copy2(source, destination)
                output = f"文件已复制: {source} -> {destination}"
            
            elif operation == 'move':
                shutil.move(source, destination)
                output = f"文件已移动: {source} -> {destination}"
            
            elif operation == 'delete':
                if os.path.isfile(source):
                    os.remove(source)
                elif os.path.isdir(source):
                    shutil.rmtree(source)
                output = f"文件已删除: {source}"
            
            elif operation == 'create':
                Path(source).touch()
                output = f"文件已创建: {source}"
            
            elif operation == 'read':
                with open(source, 'r', encoding='utf-8') as f:
                    output = f.read()
            
            elif operation == 'write':
                with open(destination, 'w', encoding='utf-8') as f:
                    f.write(content)
                output = f"内容已写入: {destination}"
            
            elif operation == 'compress':
                if destination.endswith('.zip'):
                    with zipfile.ZipFile(destination, 'w') as zf:
                        if os.path.isfile(source):
                            zf.write(source, os.path.basename(source))
                        else:
                            for root, dirs, files in os.walk(source):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arc_name = os.path.relpath(file_path, source)
                                    zf.write(file_path, arc_name)
                elif destination.endswith(('.tar', '.tar.gz', '.tar.bz2')):
                    mode = 'w:gz' if destination.endswith('.gz') else 'w:bz2' if destination.endswith('.bz2') else 'w'
                    with tarfile.open(destination, mode) as tf:
                        tf.add(source, arcname=os.path.basename(source))
                
                output = f"文件已压缩: {source} -> {destination}"
            
            else:
                raise ValueError(f"Unsupported file operation: {operation}")
            
            self.status = TaskStatus.SUCCESS
            self.add_log(f"文件操作成功: {output}")
            
            self.result = TaskResult(
                task_id=self.id,
                status=self.status,
                output=output,
                start_time=self.start_time,
                end_time=datetime.now(),
                logs=self.logs.copy(),
                retry_count=self.retry_count
            )
        
        except Exception as e:
            self.status = TaskStatus.FAILED
            error_msg = str(e)
            self.add_log(f"文件操作异常: {error_msg}", "ERROR")
            
            self.result = TaskResult(
                task_id=self.id,
                status=self.status,
                error=error_msg,
                start_time=self.start_time,
                end_time=datetime.now(),
                logs=self.logs.copy(),
                retry_count=self.retry_count
            )
        
        finally:
            self.end_time = datetime.now()
            if self.result:
                self.result.duration = (self.end_time - self.start_time).total_seconds()
        
        return self.result


class TaskFactory:
    """任务工厂"""
    
    @staticmethod
    def create_task(config: TaskConfig) -> Task:
        """创建任务实例"""
        if config.command:
            return CommandTask(config)
        elif config.function:
            return FunctionTask(config)
        elif config.parameters.get('url'):
            return HTTPTask(config)
        elif config.parameters.get('connection_string'):
            return DatabaseTask(config)
        elif config.parameters.get('operation') in ['copy', 'move', 'delete', 'create', 'read', 'write', 'compress']:
            return FileTask(config)
        else:
            raise ValueError(f"Cannot determine task type for config: {config.name}")


class WorkflowEngine:
    """工作流引擎"""
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowConfig] = {}
        self.running_workflows: Dict[str, WorkflowResult] = {}
        self.task_queue = asyncio.Queue()
        self.max_concurrent_workflows = 10
        self.scheduler = AsyncIOScheduler()
        self.notification_manager = NotificationManager()
        self.metrics = WorkflowMetrics()
        self._running = False
    
    async def start(self):
        """启动工作流引擎"""
        self._running = True
        self.scheduler.start()
        
        # 启动工作流处理器
        asyncio.create_task(self._workflow_processor())
        
        logger.info("工作流引擎已启动")
    
    async def stop(self):
        """停止工作流引擎"""
        self._running = False
        self.scheduler.shutdown()
        
        logger.info("工作流引擎已停止")
    
    def register_workflow(self, config: WorkflowConfig) -> str:
        """注册工作流"""
        workflow_id = str(uuid.uuid4())
        self.workflows[workflow_id] = config
        
        # 注册触发器
        for trigger_config in config.triggers:
            self._register_trigger(workflow_id, trigger_config)
        
        logger.info(f"工作流已注册: {config.name}", workflow_id=workflow_id)
        return workflow_id
    
    def _register_trigger(self, workflow_id: str, trigger_config: Dict[str, Any]):
        """注册触发器"""
        trigger_type = TriggerType(trigger_config.get('type'))
        
        if trigger_type == TriggerType.CRON:
            cron_expr = trigger_config.get('cron')
            self.scheduler.add_job(
                self._trigger_workflow,
                CronTrigger.from_crontab(cron_expr),
                args=[workflow_id],
                id=f"{workflow_id}_cron"
            )
        
        elif trigger_type == TriggerType.INTERVAL:
            interval = trigger_config.get('interval', 60)
            self.scheduler.add_job(
                self._trigger_workflow,
                IntervalTrigger(seconds=interval),
                args=[workflow_id],
                id=f"{workflow_id}_interval"
            )
        
        elif trigger_type == TriggerType.FILE_CHANGE:
            path = trigger_config.get('path')
            if path:
                self._setup_file_watcher(workflow_id, path)
    
    def _setup_file_watcher(self, workflow_id: str, path: str):
        """设置文件监控"""
        class WorkflowFileHandler(FileSystemEventHandler):
            def __init__(self, engine, wf_id):
                self.engine = engine
                self.workflow_id = wf_id
            
            def on_modified(self, event):
                if not event.is_directory:
                    asyncio.create_task(self.engine._trigger_workflow(self.workflow_id))
        
        observer = Observer()
        observer.schedule(WorkflowFileHandler(self, workflow_id), path, recursive=True)
        observer.start()
    
    async def _trigger_workflow(self, workflow_id: str, context: Dict[str, Any] = None):
        """触发工作流执行"""
        if workflow_id not in self.workflows:
            logger.error(f"工作流不存在: {workflow_id}")
            return
        
        await self.task_queue.put((workflow_id, context or {}))
    
    async def execute_workflow(self, workflow_id: str, context: Dict[str, Any] = None) -> WorkflowResult:
        """执行工作流"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        config = self.workflows[workflow_id]
        execution_id = str(uuid.uuid4())
        
        result = WorkflowResult(
            workflow_id=execution_id,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now()
        )
        
        self.running_workflows[execution_id] = result
        
        try:
            logger.info(f"开始执行工作流: {config.name}", execution_id=execution_id)
            
            # 创建任务实例
            tasks = {}
            for task_config in config.tasks:
                task = TaskFactory.create_task(task_config)
                tasks[task_config.name] = task
            
            # 执行任务
            if config.execution_mode == ExecutionMode.SEQUENTIAL:
                await self._execute_sequential(tasks, config, context or {})
            elif config.execution_mode == ExecutionMode.PARALLEL:
                await self._execute_parallel(tasks, config, context or {})
            elif config.execution_mode == ExecutionMode.CONDITIONAL:
                await self._execute_conditional(tasks, config, context or {})
            
            # 收集任务结果
            for task_name, task in tasks.items():
                if task.result:
                    result.task_results[task_name] = task.result
            
            # 判断工作流状态
            failed_tasks = [t for t in tasks.values() if t.status == TaskStatus.FAILED]
            if failed_tasks:
                result.status = WorkflowStatus.FAILED
                result.error = f"Failed tasks: {[t.config.name for t in failed_tasks]}"
            else:
                result.status = WorkflowStatus.COMPLETED
            
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            
            # 发送通知
            await self._send_notifications(config, result)
            
            # 更新指标
            self.metrics.record_workflow_execution(result)
            
            logger.info(
                f"工作流执行完成: {config.name}",
                execution_id=execution_id,
                status=result.status.value,
                duration=result.duration
            )
        
        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.error = str(e)
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            
            logger.error(
                f"工作流执行失败: {config.name}",
                execution_id=execution_id,
                error=str(e)
            )
        
        finally:
            if execution_id in self.running_workflows:
                del self.running_workflows[execution_id]
        
        return result
    
    async def _execute_sequential(self, tasks: Dict[str, Task], config: WorkflowConfig, context: Dict[str, Any]):
        """顺序执行任务"""
        completed_tasks = set()
        
        while len(completed_tasks) < len(tasks):
            # 找到可以执行的任务
            ready_tasks = [
                task for task in tasks.values()
                if task.status == TaskStatus.PENDING and task.can_execute(completed_tasks)
            ]
            
            if not ready_tasks:
                # 检查是否有失败的任务需要重试
                retry_tasks = [
                    task for task in tasks.values()
                    if task.should_retry()
                ]
                
                if retry_tasks:
                    for task in retry_tasks:
                        task.retry_count += 1
                        task.status = TaskStatus.PENDING
                        await asyncio.sleep(task.config.retry_delay)
                else:
                    break
            
            # 执行第一个就绪的任务
            if ready_tasks:
                task = ready_tasks[0]
                await task.execute(context)
                
                if task.status == TaskStatus.SUCCESS:
                    completed_tasks.add(task.config.name)
                    # 将任务输出添加到上下文
                    if task.result and task.result.output:
                        context[f"{task.config.name}_output"] = task.result.output
    
    async def _execute_parallel(self, tasks: Dict[str, Task], config: WorkflowConfig, context: Dict[str, Any]):
        """并行执行任务"""
        completed_tasks = set()
        running_tasks = set()
        
        while len(completed_tasks) < len(tasks):
            # 找到可以执行的任务
            ready_tasks = [
                task for task in tasks.values()
                if (task.status == TaskStatus.PENDING and 
                    task.can_execute(completed_tasks) and 
                    task not in running_tasks)
            ]
            
            # 启动新任务（不超过最大并发数）
            while ready_tasks and len(running_tasks) < config.max_parallel_tasks:
                task = ready_tasks.pop(0)
                running_tasks.add(task)
                asyncio.create_task(self._execute_task_with_callback(task, context, running_tasks, completed_tasks))
            
            # 等待一些任务完成
            if running_tasks:
                await asyncio.sleep(0.1)
            else:
                break
    
    async def _execute_task_with_callback(
        self,
        task: Task,
        context: Dict[str, Any],
        running_tasks: set,
        completed_tasks: set
    ):
        """执行任务并处理回调"""
        try:
            await task.execute(context)
            
            if task.status == TaskStatus.SUCCESS:
                completed_tasks.add(task.config.name)
                # 将任务输出添加到上下文
                if task.result and task.result.output:
                    context[f"{task.config.name}_output"] = task.result.output
        
        finally:
            running_tasks.discard(task)
    
    async def _execute_conditional(self, tasks: Dict[str, Task], config: WorkflowConfig, context: Dict[str, Any]):
        """条件执行任务"""
        completed_tasks = set()
        
        for task in tasks.values():
            if task.can_execute(completed_tasks):
                # 检查条件
                conditions = task.config.conditions
                if self._evaluate_conditions(conditions, context):
                    await task.execute(context)
                    
                    if task.status == TaskStatus.SUCCESS:
                        completed_tasks.add(task.config.name)
                        if task.result and task.result.output:
                            context[f"{task.config.name}_output"] = task.result.output
                else:
                    task.status = TaskStatus.SKIPPED
                    completed_tasks.add(task.config.name)
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """评估条件"""
        if not conditions:
            return True
        
        # 简单的条件评估逻辑
        for key, expected_value in conditions.items():
            if key not in context or context[key] != expected_value:
                return False
        
        return True
    
    async def _send_notifications(self, config: WorkflowConfig, result: WorkflowResult):
        """发送通知"""
        for notification_config in config.notifications:
            try:
                await self.notification_manager.send_notification(
                    notification_config,
                    {
                        'workflow_name': config.name,
                        'status': result.status.value,
                        'duration': result.duration,
                        'error': result.error
                    }
                )
            except Exception as e:
                logger.error(f"发送通知失败: {e}")
    
    async def _workflow_processor(self):
        """工作流处理器"""
        while self._running:
            try:
                workflow_id, context = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                if len(self.running_workflows) < self.max_concurrent_workflows:
                    asyncio.create_task(self.execute_workflow(workflow_id, context))
                else:
                    # 重新放回队列
                    await self.task_queue.put((workflow_id, context))
                    await asyncio.sleep(1)
            
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"工作流处理器错误: {e}")
    
    def get_workflow_status(self, execution_id: str) -> Optional[WorkflowResult]:
        """获取工作流状态"""
        return self.running_workflows.get(execution_id)
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """列出所有工作流"""
        return [
            {
                'id': wf_id,
                'name': config.name,
                'description': config.description,
                'task_count': len(config.tasks),
                'created_at': config.created_at.isoformat()
            }
            for wf_id, config in self.workflows.items()
        ]


class NotificationManager:
    """通知管理器"""
    
    def __init__(self):
        self.notification_handlers = {
            NotificationType.EMAIL: self._send_email,
            NotificationType.SLACK: self._send_slack,
            NotificationType.WEBHOOK: self._send_webhook
        }
    
    async def send_notification(self, config: Dict[str, Any], data: Dict[str, Any]):
        """发送通知"""
        notification_type = NotificationType(config.get('type'))
        
        if notification_type in self.notification_handlers:
            await self.notification_handlers[notification_type](config, data)
        else:
            logger.warning(f"不支持的通知类型: {notification_type}")
    
    async def _send_email(self, config: Dict[str, Any], data: Dict[str, Any]):
        """发送邮件通知"""
        smtp_server = config.get('smtp_server')
        smtp_port = config.get('smtp_port', 587)
        username = config.get('username')
        password = config.get('password')
        to_emails = config.get('to')
        subject = config.get('subject', '工作流通知')
        
        # 渲染邮件内容
        template = Template(config.get('template', '工作流 {{workflow_name}} 状态: {{status}}'))
        content = template.render(**data)
        
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = ', '.join(to_emails)
        msg['Subject'] = subject
        
        msg.attach(MIMEText(content, 'html' if '<' in content else 'plain'))
        
        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            logger.info("邮件通知发送成功")
        
        except Exception as e:
            logger.error(f"邮件通知发送失败: {e}")
    
    async def _send_slack(self, config: Dict[str, Any], data: Dict[str, Any]):
        """发送Slack通知"""
        token = config.get('token')
        channel = config.get('channel')
        
        template = Template(config.get('template', '工作流 {{workflow_name}} 状态: {{status}}'))
        message = template.render(**data)
        
        try:
            client = AsyncWebClient(token=token)
            await client.chat_postMessage(
                channel=channel,
                text=message
            )
            
            logger.info("Slack通知发送成功")
        
        except Exception as e:
            logger.error(f"Slack通知发送失败: {e}")
    
    async def _send_webhook(self, config: Dict[str, Any], data: Dict[str, Any]):
        """发送Webhook通知"""
        url = config.get('url')
        method = config.get('method', 'POST')
        headers = config.get('headers', {})
        
        payload = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status < 400:
                        logger.info("Webhook通知发送成功")
                    else:
                        logger.error(f"Webhook通知发送失败: {response.status}")
        
        except Exception as e:
            logger.error(f"Webhook通知发送失败: {e}")


class WorkflowMetrics:
    """工作流指标"""
    
    def __init__(self):
        self.workflow_executions = Counter('workflow_executions_total', 'Total workflow executions', ['workflow_name', 'status'])
        self.workflow_duration = Histogram('workflow_duration_seconds', 'Workflow execution duration', ['workflow_name'])
        self.task_executions = Counter('task_executions_total', 'Total task executions', ['task_name', 'status'])
        self.task_duration = Histogram('task_duration_seconds', 'Task execution duration', ['task_name'])
        self.active_workflows = Gauge('active_workflows', 'Number of active workflows')
    
    def record_workflow_execution(self, result: WorkflowResult):
        """记录工作流执行指标"""
        workflow_name = result.workflow_id  # 这里应该是工作流名称
        
        self.workflow_executions.labels(
            workflow_name=workflow_name,
            status=result.status.value
        ).inc()
        
        if result.duration:
            self.workflow_duration.labels(workflow_name=workflow_name).observe(result.duration)
        
        # 记录任务指标
        for task_name, task_result in result.task_results.items():
            self.task_executions.labels(
                task_name=task_name,
                status=task_result.status.value
            ).inc()
            
            if task_result.duration:
                self.task_duration.labels(task_name=task_name).observe(task_result.duration)


class WorkflowBuilder:
    """工作流构建器"""
    
    def __init__(self, name: str, description: str = ""):
        self.config = WorkflowConfig(name=name, description=description)
    
    def add_task(self, task_config: TaskConfig) -> 'WorkflowBuilder':
        """添加任务"""
        self.config.tasks.append(task_config)
        return self
    
    def add_command_task(
        self,
        name: str,
        command: str,
        dependencies: List[str] = None,
        timeout: int = None,
        retry_count: int = 0
    ) -> 'WorkflowBuilder':
        """添加命令任务"""
        task_config = TaskConfig(
            name=name,
            command=command,
            dependencies=dependencies or [],
            timeout=timeout,
            retry_count=retry_count
        )
        return self.add_task(task_config)
    
    def add_function_task(
        self,
        name: str,
        function: Callable,
        parameters: Dict[str, Any] = None,
        dependencies: List[str] = None,
        timeout: int = None,
        retry_count: int = 0
    ) -> 'WorkflowBuilder':
        """添加函数任务"""
        task_config = TaskConfig(
            name=name,
            function=function,
            parameters=parameters or {},
            dependencies=dependencies or [],
            timeout=timeout,
            retry_count=retry_count
        )
        return self.add_task(task_config)
    
    def add_http_task(
        self,
        name: str,
        url: str,
        method: str = "GET",
        headers: Dict[str, str] = None,
        data: Any = None,
        dependencies: List[str] = None,
        timeout: int = None,
        retry_count: int = 0
    ) -> 'WorkflowBuilder':
        """添加HTTP任务"""
        parameters = {
            'url': url,
            'method': method,
            'headers': headers or {},
            'data': data
        }
        
        task_config = TaskConfig(
            name=name,
            parameters=parameters,
            dependencies=dependencies or [],
            timeout=timeout,
            retry_count=retry_count
        )
        return self.add_task(task_config)
    
    def set_execution_mode(self, mode: ExecutionMode) -> 'WorkflowBuilder':
        """设置执行模式"""
        self.config.execution_mode = mode
        return self
    
    def set_max_parallel_tasks(self, max_tasks: int) -> 'WorkflowBuilder':
        """设置最大并行任务数"""
        self.config.max_parallel_tasks = max_tasks
        return self
    
    def add_cron_trigger(self, cron_expression: str) -> 'WorkflowBuilder':
        """添加Cron触发器"""
        trigger = {
            'type': TriggerType.CRON.value,
            'cron': cron_expression
        }
        self.config.triggers.append(trigger)
        return self
    
    def add_interval_trigger(self, interval_seconds: int) -> 'WorkflowBuilder':
        """添加间隔触发器"""
        trigger = {
            'type': TriggerType.INTERVAL.value,
            'interval': interval_seconds
        }
        self.config.triggers.append(trigger)
        return self
    
    def add_email_notification(
        self,
        smtp_server: str,
        username: str,
        password: str,
        to_emails: List[str],
        subject: str = "工作流通知",
        template: str = None
    ) -> 'WorkflowBuilder':
        """添加邮件通知"""
        notification = {
            'type': NotificationType.EMAIL.value,
            'smtp_server': smtp_server,
            'username': username,
            'password': password,
            'to': to_emails,
            'subject': subject,
            'template': template or '工作流 {{workflow_name}} 状态: {{status}}'
        }
        self.config.notifications.append(notification)
        return self
    
    def add_slack_notification(
        self,
        token: str,
        channel: str,
        template: str = None
    ) -> 'WorkflowBuilder':
        """添加Slack通知"""
        notification = {
            'type': NotificationType.SLACK.value,
            'token': token,
            'channel': channel,
            'template': template or '工作流 {{workflow_name}} 状态: {{status}}'
        }
        self.config.notifications.append(notification)
        return self
    
    def build(self) -> WorkflowConfig:
        """构建工作流配置"""
        return self.config


# CLI工具
@click.group()
def cli():
    """工作流管理CLI"""
    pass


@cli.command()
@click.argument('config_file')
def run(config_file):
    """运行工作流"""
    async def _run():
        # 加载配置
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        # 创建工作流引擎
        engine = WorkflowEngine()
        await engine.start()
        
        try:
            # 构建工作流配置
            config = WorkflowConfig(**config_data)
            
            # 注册并执行工作流
            workflow_id = engine.register_workflow(config)
            result = await engine.execute_workflow(workflow_id)
            
            # 显示结果
            console.print(Panel(
                f"工作流执行完成\n"
                f"状态: {result.status.value}\n"
                f"持续时间: {result.duration:.2f}秒\n"
                f"任务数: {len(result.task_results)}",
                title=config.name
            ))
            
            # 显示任务结果表格
            table = Table(title="任务执行结果")
            table.add_column("任务名称")
            table.add_column("状态")
            table.add_column("持续时间")
            table.add_column("错误")
            
            for task_name, task_result in result.task_results.items():
                status_color = "green" if task_result.status == TaskStatus.SUCCESS else "red"
                table.add_row(
                    task_name,
                    f"[{status_color}]{task_result.status.value}[/{status_color}]",
                    f"{task_result.duration:.2f}s" if task_result.duration else "N/A",
                    task_result.error or ""
                )
            
            console.print(table)
        
        finally:
            await engine.stop()
    
    asyncio.run(_run())


@cli.command()
@click.argument('name')
@click.option('--description', default="", help="工作流描述")
def create(name, description):
    """创建工作流模板"""
    template = {
        "name": name,
        "description": description,
        "execution_mode": "sequential",
        "tasks": [
            {
                "name": "example_task",
                "command": "echo 'Hello World'",
                "timeout": 30,
                "retry_count": 1
            }
        ],
        "triggers": [
            {
                "type": "manual"
            }
        ],
        "notifications": []
    }
    
    filename = f"{name.lower().replace(' ', '_')}_workflow.yaml"
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(template, f, default_flow_style=False, allow_unicode=True)
    
    console.print(f"工作流模板已创建: {filename}")


# 使用示例
if __name__ == "__main__":
    async def example_usage():
        # 创建工作流引擎
        engine = WorkflowEngine()
        await engine.start()
        
        try:
            # 使用构建器创建工作流
            builder = WorkflowBuilder("数据处理工作流", "处理和分析数据的工作流")
            
            # 添加任务
            builder.add_command_task(
                "下载数据",
                "curl -o data.csv https://example.com/data.csv",
                timeout=60
            ).add_function_task(
                "处理数据",
                lambda: print("处理数据..."),
                dependencies=["下载数据"]
            ).add_http_task(
                "上传结果",
                "https://api.example.com/upload",
                method="POST",
                dependencies=["处理数据"]
            )
            
            # 设置执行模式和触发器
            builder.set_execution_mode(ExecutionMode.SEQUENTIAL)
            builder.add_cron_trigger("0 2 * * *")  # 每天凌晨2点执行
            
            # 添加通知
            builder.add_email_notification(
                "smtp.example.com",
                "user@example.com",
                "password",
                ["admin@example.com"]
            )
            
            # 构建并注册工作流
            config = builder.build()
            workflow_id = engine.register_workflow(config)
            
            # 手动执行工作流
            result = await engine.execute_workflow(workflow_id)
            
            print(f"工作流执行结果: {result.status}")
            print(f"执行时间: {result.duration}秒")
            
            for task_name, task_result in result.task_results.items():
                print(f"任务 {task_name}: {task_result.status}")
        
        finally:
            await engine.stop()
    
    # 运行示例
    asyncio.run(example_usage())