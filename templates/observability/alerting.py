"""告警系统

提供完整的告警功能，包括：
- 告警规则定义
- 指标监控
- 告警触发
- 通知发送
- 告警抑制
"""

import time
import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    import requests
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False


class AlertSeverity(str, Enum):
    """告警严重级别"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(str, Enum):
    """告警状态"""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"


class ComparisonOperator(str, Enum):
    """比较操作符"""
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NEQ = "!="


class AggregationFunction(str, Enum):
    """聚合函数"""
    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    RATE = "rate"
    INCREASE = "increase"


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    description: str
    metric_name: str
    operator: ComparisonOperator
    threshold: float
    severity: AlertSeverity = AlertSeverity.MEDIUM
    
    # 时间窗口配置
    evaluation_interval: int = 60  # seconds
    for_duration: int = 300  # seconds, 持续时间
    
    # 聚合配置
    aggregation: AggregationFunction = AggregationFunction.AVG
    time_window: int = 300  # seconds
    
    # 标签过滤
    label_filters: Dict[str, str] = field(default_factory=dict)
    
    # 告警配置
    enabled: bool = True
    annotations: Dict[str, str] = field(default_factory=dict)
    runbook_url: Optional[str] = None
    
    # 抑制配置
    inhibit_rules: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Alert rule name cannot be empty")
        if self.evaluation_interval <= 0:
            raise ValueError("Evaluation interval must be positive")
        if self.for_duration < 0:
            raise ValueError("For duration cannot be negative")


@dataclass
class Alert:
    """告警实例"""
    rule_name: str
    metric_name: str
    value: float
    threshold: float
    operator: ComparisonOperator
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.PENDING
    
    # 时间信息
    started_at: float = field(default_factory=time.time)
    fired_at: Optional[float] = None
    resolved_at: Optional[float] = None
    acknowledged_at: Optional[float] = None
    
    # 标签和注释
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    # 元数据
    fingerprint: str = field(default="")
    generation: int = 0
    
    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = self._generate_fingerprint()
    
    def _generate_fingerprint(self) -> str:
        """生成告警指纹"""
        import hashlib
        
        data = f"{self.rule_name}:{self.metric_name}:{sorted(self.labels.items())}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def fire(self):
        """触发告警"""
        if self.status == AlertStatus.PENDING:
            self.status = AlertStatus.FIRING
            self.fired_at = time.time()
    
    def resolve(self):
        """解决告警"""
        if self.status in [AlertStatus.PENDING, AlertStatus.FIRING]:
            self.status = AlertStatus.RESOLVED
            self.resolved_at = time.time()
    
    def acknowledge(self):
        """确认告警"""
        if self.status == AlertStatus.FIRING:
            self.status = AlertStatus.ACKNOWLEDGED
            self.acknowledged_at = time.time()
    
    def suppress(self):
        """抑制告警"""
        if self.status in [AlertStatus.PENDING, AlertStatus.FIRING]:
            self.status = AlertStatus.SUPPRESSED
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "rule_name": self.rule_name,
            "metric_name": self.metric_name,
            "value": self.value,
            "threshold": self.threshold,
            "operator": self.operator.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "started_at": self.started_at,
            "fired_at": self.fired_at,
            "resolved_at": self.resolved_at,
            "acknowledged_at": self.acknowledged_at,
            "labels": self.labels,
            "annotations": self.annotations,
            "fingerprint": self.fingerprint,
            "generation": self.generation
        }


class NotificationChannel(ABC):
    """通知渠道基类"""
    
    @abstractmethod
    async def send_notification(self, alert: Alert, message: str) -> bool:
        """发送通知"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查渠道是否可用"""
        pass


class EmailNotificationChannel(NotificationChannel):
    """邮件通知渠道"""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str,
                 from_email: str, to_emails: List[str], use_tls: bool = True):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.use_tls = use_tls
    
    async def send_notification(self, alert: Alert, message: str) -> bool:
        """发送邮件通知"""
        if not EMAIL_AVAILABLE:
            return False
        
        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.rule_name}"
            
            # 邮件内容
            body = f"""
告警名称: {alert.rule_name}
指标名称: {alert.metric_name}
当前值: {alert.value}
阈值: {alert.threshold}
操作符: {alert.operator.value}
严重级别: {alert.severity.value}
状态: {alert.status.value}
开始时间: {datetime.fromtimestamp(alert.started_at)}

{message}

标签:
{chr(10).join(f'  {k}: {v}' for k, v in alert.labels.items())}

注释:
{chr(10).join(f'  {k}: {v}' for k, v in alert.annotations.items())}
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 发送邮件
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            print(f"Failed to send email notification: {e}")
            return False
    
    def is_available(self) -> bool:
        """检查邮件服务是否可用"""
        return EMAIL_AVAILABLE and bool(self.smtp_host and self.username and self.password)


class WebhookNotificationChannel(NotificationChannel):
    """Webhook通知渠道"""
    
    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None,
                 timeout: int = 30, retry_count: int = 3):
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.retry_count = retry_count
    
    async def send_notification(self, alert: Alert, message: str) -> bool:
        """发送Webhook通知"""
        if not HTTP_AVAILABLE:
            return False
        
        payload = {
            "alert": alert.to_dict(),
            "message": message,
            "timestamp": time.time()
        }
        
        headers = {
            "Content-Type": "application/json",
            **self.headers
        }
        
        for attempt in range(self.retry_count):
            try:
                response = requests.post(
                    self.url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code < 400:
                    return True
                
                print(f"Webhook notification failed with status {response.status_code}: {response.text}")
                
            except Exception as e:
                print(f"Webhook notification attempt {attempt + 1} failed: {e}")
                
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避
        
        return False
    
    def is_available(self) -> bool:
        """检查Webhook是否可用"""
        return HTTP_AVAILABLE and bool(self.url)


class SlackNotificationChannel(NotificationChannel):
    """Slack通知渠道"""
    
    def __init__(self, webhook_url: str, channel: Optional[str] = None,
                 username: str = "AlertBot", icon_emoji: str = ":warning:"):
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji
    
    async def send_notification(self, alert: Alert, message: str) -> bool:
        """发送Slack通知"""
        if not HTTP_AVAILABLE:
            return False
        
        # 根据严重级别选择颜色
        color_map = {
            AlertSeverity.CRITICAL: "danger",
            AlertSeverity.HIGH: "warning",
            AlertSeverity.MEDIUM: "#ff9500",
            AlertSeverity.LOW: "good",
            AlertSeverity.INFO: "#439FE0"
        }
        
        color = color_map.get(alert.severity, "#439FE0")
        
        # 构建Slack消息
        payload = {
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "attachments": [
                {
                    "color": color,
                    "title": f"[{alert.severity.upper()}] {alert.rule_name}",
                    "text": message,
                    "fields": [
                        {
                            "title": "指标",
                            "value": alert.metric_name,
                            "short": True
                        },
                        {
                            "title": "当前值",
                            "value": str(alert.value),
                            "short": True
                        },
                        {
                            "title": "阈值",
                            "value": f"{alert.operator.value} {alert.threshold}",
                            "short": True
                        },
                        {
                            "title": "状态",
                            "value": alert.status.value,
                            "short": True
                        }
                    ],
                    "timestamp": int(alert.started_at)
                }
            ]
        }
        
        if self.channel:
            payload["channel"] = self.channel
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=30
            )
            
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to send Slack notification: {e}")
            return False
    
    def is_available(self) -> bool:
        """检查Slack是否可用"""
        return HTTP_AVAILABLE and bool(self.webhook_url)


class AlertingConfig(BaseSettings):
    """告警配置"""
    enabled: bool = True
    
    # 评估配置
    evaluation_interval: int = 60  # seconds
    max_alerts: int = 1000
    alert_retention: int = 86400  # seconds (24 hours)
    
    # 通知配置
    notification_timeout: int = 30  # seconds
    max_notification_retries: int = 3
    notification_rate_limit: int = 10  # per minute
    
    # 抑制配置
    enable_inhibition: bool = True
    inhibition_duration: int = 3600  # seconds
    
    # 邮件配置
    email_enabled: bool = False
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from_email: Optional[str] = None
    smtp_to_emails: List[str] = Field(default_factory=list)
    smtp_use_tls: bool = True
    
    # Webhook配置
    webhook_enabled: bool = False
    webhook_url: Optional[str] = None
    webhook_headers: Dict[str, str] = Field(default_factory=dict)
    
    # Slack配置
    slack_enabled: bool = False
    slack_webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None
    slack_username: str = "AlertBot"
    
    class Config:
        env_prefix = "ALERT_"


class AlertEvaluator:
    """告警评估器"""
    
    def __init__(self, metrics_manager):
        self.metrics_manager = metrics_manager
        self._metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    def evaluate_rule(self, rule: AlertRule) -> Optional[Alert]:
        """评估告警规则"""
        if not rule.enabled:
            return None
        
        # 获取指标数据
        metric = self.metrics_manager.registry.get_metric(rule.metric_name)
        if not metric:
            return None
        
        # 获取当前值
        current_value = self._get_aggregated_value(metric, rule)
        if current_value is None:
            return None
        
        # 检查条件
        condition_met = self._check_condition(current_value, rule.operator, rule.threshold)
        
        if condition_met:
            # 创建告警
            alert = Alert(
                rule_name=rule.name,
                metric_name=rule.metric_name,
                value=current_value,
                threshold=rule.threshold,
                operator=rule.operator,
                severity=rule.severity,
                annotations=rule.annotations.copy()
            )
            
            # 添加标签过滤
            alert.labels.update(rule.label_filters)
            
            return alert
        
        return None
    
    def _get_aggregated_value(self, metric, rule: AlertRule) -> Optional[float]:
        """获取聚合值"""
        # 获取最近的样本
        samples = metric.get_samples()
        if not samples:
            return None
        
        # 过滤时间窗口内的样本
        now = time.time()
        window_start = now - rule.time_window
        
        filtered_samples = [
            sample for sample in samples
            if sample.timestamp >= window_start
        ]
        
        if not filtered_samples:
            return None
        
        # 应用聚合函数
        values = [sample.value for sample in filtered_samples]
        
        if rule.aggregation == AggregationFunction.AVG:
            return sum(values) / len(values)
        elif rule.aggregation == AggregationFunction.SUM:
            return sum(values)
        elif rule.aggregation == AggregationFunction.MIN:
            return min(values)
        elif rule.aggregation == AggregationFunction.MAX:
            return max(values)
        elif rule.aggregation == AggregationFunction.COUNT:
            return len(values)
        elif rule.aggregation == AggregationFunction.RATE:
            if len(values) < 2:
                return 0.0
            time_diff = filtered_samples[-1].timestamp - filtered_samples[0].timestamp
            value_diff = filtered_samples[-1].value - filtered_samples[0].value
            return value_diff / time_diff if time_diff > 0 else 0.0
        elif rule.aggregation == AggregationFunction.INCREASE:
            if len(values) < 2:
                return 0.0
            return filtered_samples[-1].value - filtered_samples[0].value
        
        return values[-1]  # 默认返回最新值
    
    def _check_condition(self, value: float, operator: ComparisonOperator, threshold: float) -> bool:
        """检查条件"""
        if operator == ComparisonOperator.GT:
            return value > threshold
        elif operator == ComparisonOperator.GTE:
            return value >= threshold
        elif operator == ComparisonOperator.LT:
            return value < threshold
        elif operator == ComparisonOperator.LTE:
            return value <= threshold
        elif operator == ComparisonOperator.EQ:
            return abs(value - threshold) < 1e-9
        elif operator == ComparisonOperator.NEQ:
            return abs(value - threshold) >= 1e-9
        
        return False


class AlertManager:
    """告警管理器"""
    
    def __init__(self, config: AlertingConfig, metrics_manager):
        self.config = config
        self.metrics_manager = metrics_manager
        self.evaluator = AlertEvaluator(metrics_manager)
        
        # 告警存储
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: Dict[str, Alert] = {}  # fingerprint -> alert
        self.alert_history: deque = deque(maxlen=config.max_alerts)
        
        # 通知渠道
        self.notification_channels: List[NotificationChannel] = []
        self._setup_notification_channels()
        
        # 运行状态
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # 通知限流
        self._notification_timestamps: deque = deque(maxlen=config.notification_rate_limit)
    
    def _setup_notification_channels(self):
        """设置通知渠道"""
        # 邮件通知
        if (self.config.email_enabled and self.config.smtp_host and 
            self.config.smtp_username and self.config.smtp_password):
            email_channel = EmailNotificationChannel(
                smtp_host=self.config.smtp_host,
                smtp_port=self.config.smtp_port,
                username=self.config.smtp_username,
                password=self.config.smtp_password,
                from_email=self.config.smtp_from_email or self.config.smtp_username,
                to_emails=self.config.smtp_to_emails,
                use_tls=self.config.smtp_use_tls
            )
            self.notification_channels.append(email_channel)
        
        # Webhook通知
        if self.config.webhook_enabled and self.config.webhook_url:
            webhook_channel = WebhookNotificationChannel(
                url=self.config.webhook_url,
                headers=self.config.webhook_headers
            )
            self.notification_channels.append(webhook_channel)
        
        # Slack通知
        if self.config.slack_enabled and self.config.slack_webhook_url:
            slack_channel = SlackNotificationChannel(
                webhook_url=self.config.slack_webhook_url,
                channel=self.config.slack_channel,
                username=self.config.slack_username
            )
            self.notification_channels.append(slack_channel)
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        with self._lock:
            self.rules[rule.name] = rule
    
    def remove_rule(self, rule_name: str):
        """移除告警规则"""
        with self._lock:
            self.rules.pop(rule_name, None)
    
    def get_rule(self, rule_name: str) -> Optional[AlertRule]:
        """获取告警规则"""
        return self.rules.get(rule_name)
    
    def get_all_rules(self) -> Dict[str, AlertRule]:
        """获取所有告警规则"""
        return self.rules.copy()
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        with self._lock:
            return [
                alert for alert in self.alerts.values()
                if alert.status in [AlertStatus.PENDING, AlertStatus.FIRING]
            ]
    
    def get_all_alerts(self) -> List[Alert]:
        """获取所有告警"""
        with self._lock:
            return list(self.alerts.values())
    
    def acknowledge_alert(self, fingerprint: str) -> bool:
        """确认告警"""
        with self._lock:
            alert = self.alerts.get(fingerprint)
            if alert and alert.status == AlertStatus.FIRING:
                alert.acknowledge()
                return True
            return False
    
    def start(self):
        """启动告警管理器"""
        if not self.config.enabled or self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """停止告警管理器"""
        self._running = False
        if self._thread:
            self._thread.join()
    
    def _evaluation_loop(self):
        """评估循环"""
        while self._running:
            try:
                self._evaluate_rules()
                self._cleanup_old_alerts()
            except Exception as e:
                print(f"Alert evaluation error: {e}")
            
            time.sleep(self.config.evaluation_interval)
    
    def _evaluate_rules(self):
        """评估所有规则"""
        current_time = time.time()
        
        for rule in self.rules.values():
            try:
                alert = self.evaluator.evaluate_rule(rule)
                if alert:
                    self._process_alert(alert, rule, current_time)
            except Exception as e:
                print(f"Error evaluating rule {rule.name}: {e}")
    
    def _process_alert(self, alert: Alert, rule: AlertRule, current_time: float):
        """处理告警"""
        with self._lock:
            existing_alert = self.alerts.get(alert.fingerprint)
            
            if existing_alert:
                # 更新现有告警
                existing_alert.value = alert.value
                existing_alert.generation += 1
                
                # 检查是否应该触发
                if (existing_alert.status == AlertStatus.PENDING and
                    current_time - existing_alert.started_at >= rule.for_duration):
                    existing_alert.fire()
                    asyncio.create_task(self._send_notifications(existing_alert, "告警触发"))
            else:
                # 新告警
                self.alerts[alert.fingerprint] = alert
                self.alert_history.append(alert)
                
                # 如果没有持续时间要求，立即触发
                if rule.for_duration == 0:
                    alert.fire()
                    asyncio.create_task(self._send_notifications(alert, "告警触发"))
    
    async def _send_notifications(self, alert: Alert, message: str):
        """发送通知"""
        # 检查通知限流
        if not self._check_notification_rate_limit():
            return
        
        # 检查告警抑制
        if self._is_alert_suppressed(alert):
            alert.suppress()
            return
        
        # 发送到所有可用的通知渠道
        for channel in self.notification_channels:
            if channel.is_available():
                try:
                    await channel.send_notification(alert, message)
                except Exception as e:
                    print(f"Failed to send notification via {type(channel).__name__}: {e}")
    
    def _check_notification_rate_limit(self) -> bool:
        """检查通知限流"""
        now = time.time()
        
        # 清理过期的时间戳
        while (self._notification_timestamps and 
               now - self._notification_timestamps[0] > 60):
            self._notification_timestamps.popleft()
        
        # 检查是否超过限制
        if len(self._notification_timestamps) >= self.config.notification_rate_limit:
            return False
        
        self._notification_timestamps.append(now)
        return True
    
    def _is_alert_suppressed(self, alert: Alert) -> bool:
        """检查告警是否被抑制"""
        if not self.config.enable_inhibition:
            return False
        
        # 简单的抑制逻辑：相同指标的更高级别告警会抑制低级别告警
        severity_order = {
            AlertSeverity.CRITICAL: 5,
            AlertSeverity.HIGH: 4,
            AlertSeverity.MEDIUM: 3,
            AlertSeverity.LOW: 2,
            AlertSeverity.INFO: 1
        }
        
        alert_severity = severity_order.get(alert.severity, 0)
        
        for other_alert in self.alerts.values():
            if (other_alert.fingerprint != alert.fingerprint and
                other_alert.metric_name == alert.metric_name and
                other_alert.status == AlertStatus.FIRING and
                severity_order.get(other_alert.severity, 0) > alert_severity):
                return True
        
        return False
    
    def _cleanup_old_alerts(self):
        """清理过期告警"""
        current_time = time.time()
        retention_threshold = current_time - self.config.alert_retention
        
        with self._lock:
            expired_fingerprints = [
                fingerprint for fingerprint, alert in self.alerts.items()
                if (alert.status == AlertStatus.RESOLVED and
                    alert.resolved_at and alert.resolved_at < retention_threshold)
            ]
            
            for fingerprint in expired_fingerprints:
                del self.alerts[fingerprint]


# 全局告警管理器
_alert_manager: Optional[AlertManager] = None


def initialize_alerting(config: AlertingConfig, metrics_manager) -> AlertManager:
    """初始化告警系统"""
    global _alert_manager
    _alert_manager = AlertManager(config, metrics_manager)
    _alert_manager.start()
    return _alert_manager


def get_alert_manager() -> Optional[AlertManager]:
    """获取全局告警管理器"""
    return _alert_manager


# 便捷函数
def add_alert_rule(rule: AlertRule):
    """添加告警规则的便捷函数"""
    manager = get_alert_manager()
    if manager:
        manager.add_rule(rule)


def get_active_alerts() -> List[Alert]:
    """获取活跃告警的便捷函数"""
    manager = get_alert_manager()
    return manager.get_active_alerts() if manager else []


def acknowledge_alert(fingerprint: str) -> bool:
    """确认告警的便捷函数"""
    manager = get_alert_manager()
    return manager.acknowledge_alert(fingerprint) if manager else False