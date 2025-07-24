# -*- coding: utf-8 -*-
"""
高级安全模块
提供威胁检测、安全审计、访问控制、数据保护等高级安全功能
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import json
import re
import ipaddress
from typing import (
    Dict, List, Any, Optional, Union, Callable, Tuple,
    Set, Pattern, NamedTuple
)
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import bcrypt
import pyotp
import qrcode
from io import BytesIO
import base64
from collections import defaultdict, deque
import geoip2.database
import user_agents
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import redis
from passlib.context import CryptContext
from passlib.hash import argon2
import bleach
from urllib.parse import urlparse
import socket
import ssl
import certifi
import requests
from email_validator import validate_email, EmailNotValidError
import phonenumbers
from phonenumbers import NumberParseException
import magic
from PIL import Image
import exifread
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import zxcvbn

logger = logging.getLogger(__name__)

Base = declarative_base()


class ThreatLevel(Enum):
    """威胁等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """安全事件类型枚举"""
    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    BRUTE_FORCE = "brute_force"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALWARE_DETECTION = "malware_detection"
    INJECTION_ATTEMPT = "injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    CSRF_ATTEMPT = "csrf_attempt"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    ACCOUNT_LOCKOUT = "account_lockout"
    PASSWORD_CHANGE = "password_change"
    PERMISSION_CHANGE = "permission_change"
    FILE_UPLOAD = "file_upload"
    API_ABUSE = "api_abuse"


class AccessControlAction(Enum):
    """访问控制动作枚举"""
    ALLOW = "allow"
    DENY = "deny"
    CHALLENGE = "challenge"
    LOG = "log"
    RATE_LIMIT = "rate_limit"


class EncryptionAlgorithm(Enum):
    """加密算法枚举"""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    FERNET = "fernet"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    CHACHA20_POLY1305 = "chacha20_poly1305"


@dataclass
class SecurityEvent:
    """安全事件"""
    event_type: SecurityEventType
    timestamp: datetime
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    threat_level: ThreatLevel = ThreatLevel.LOW
    details: Dict[str, Any] = field(default_factory=dict)
    location: Optional[Dict[str, str]] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    fingerprint: Optional[str] = None


@dataclass
class ThreatIndicator:
    """威胁指标"""
    indicator_type: str  # ip, domain, hash, pattern
    value: str
    threat_level: ThreatLevel
    description: str
    source: str
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessRule:
    """访问规则"""
    name: str
    conditions: Dict[str, Any]
    action: AccessControlAction
    priority: int = 0
    enabled: bool = True
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SecurityPolicy:
    """安全策略"""
    name: str
    rules: List[AccessRule]
    default_action: AccessControlAction = AccessControlAction.DENY
    enabled: bool = True
    description: str = ""
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class SecurityAuditLog(Base):
    """安全审计日志模型"""
    __tablename__ = 'security_audit_logs'
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String(50), nullable=False)
    user_id = Column(String(100))
    ip_address = Column(String(45))
    user_agent = Column(Text)
    resource = Column(String(255))
    action = Column(String(100))
    threat_level = Column(String(20))
    details = Column(Text)
    location = Column(Text)
    session_id = Column(String(100))
    request_id = Column(String(100))
    fingerprint = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow)


class ThreatIntelligence:
    """威胁情报"""
    
    def __init__(self):
        self.indicators: Dict[str, List[ThreatIndicator]] = defaultdict(list)
        self.reputation_cache: Dict[str, Tuple[float, datetime]] = {}
        self.malicious_ips: Set[str] = set()
        self.malicious_domains: Set[str] = set()
        self.malicious_hashes: Set[str] = set()
        self.suspicious_patterns: List[Pattern] = []
    
    def add_indicator(self, indicator: ThreatIndicator):
        """添加威胁指标"""
        self.indicators[indicator.indicator_type].append(indicator)
        
        # 更新快速查找集合
        if indicator.indicator_type == "ip":
            self.malicious_ips.add(indicator.value)
        elif indicator.indicator_type == "domain":
            self.malicious_domains.add(indicator.value)
        elif indicator.indicator_type == "hash":
            self.malicious_hashes.add(indicator.value)
        elif indicator.indicator_type == "pattern":
            try:
                pattern = re.compile(indicator.value)
                self.suspicious_patterns.append(pattern)
            except re.error:
                logger.warning(f"Invalid regex pattern: {indicator.value}")
    
    def check_ip_reputation(self, ip_address: str) -> Tuple[bool, ThreatLevel]:
        """检查IP声誉"""
        # 检查是否在恶意IP列表中
        if ip_address in self.malicious_ips:
            return True, ThreatLevel.HIGH
        
        # 检查缓存
        if ip_address in self.reputation_cache:
            score, timestamp = self.reputation_cache[ip_address]
            if datetime.now() - timestamp < timedelta(hours=1):
                if score > 0.7:
                    return True, ThreatLevel.HIGH
                elif score > 0.4:
                    return True, ThreatLevel.MEDIUM
                else:
                    return False, ThreatLevel.LOW
        
        # 这里可以集成外部威胁情报API
        # 例如：VirusTotal, AbuseIPDB, etc.
        
        return False, ThreatLevel.LOW
    
    def check_domain_reputation(self, domain: str) -> Tuple[bool, ThreatLevel]:
        """检查域名声誉"""
        if domain in self.malicious_domains:
            return True, ThreatLevel.HIGH
        
        # 检查是否匹配可疑模式
        for pattern in self.suspicious_patterns:
            if pattern.search(domain):
                return True, ThreatLevel.MEDIUM
        
        return False, ThreatLevel.LOW
    
    def check_file_hash(self, file_hash: str) -> Tuple[bool, ThreatLevel]:
        """检查文件哈希"""
        if file_hash in self.malicious_hashes:
            return True, ThreatLevel.HIGH
        
        return False, ThreatLevel.LOW
    
    def cleanup_expired_indicators(self):
        """清理过期的威胁指标"""
        now = datetime.now()
        
        for indicator_type, indicators in self.indicators.items():
            valid_indicators = []
            for indicator in indicators:
                if indicator.expires_at is None or indicator.expires_at > now:
                    valid_indicators.append(indicator)
            
            self.indicators[indicator_type] = valid_indicators
        
        # 重建快速查找集合
        self.malicious_ips.clear()
        self.malicious_domains.clear()
        self.malicious_hashes.clear()
        self.suspicious_patterns.clear()
        
        for indicators in self.indicators.values():
            for indicator in indicators:
                if indicator.indicator_type == "ip":
                    self.malicious_ips.add(indicator.value)
                elif indicator.indicator_type == "domain":
                    self.malicious_domains.add(indicator.value)
                elif indicator.indicator_type == "hash":
                    self.malicious_hashes.add(indicator.value)
                elif indicator.indicator_type == "pattern":
                    try:
                        pattern = re.compile(indicator.value)
                        self.suspicious_patterns.append(pattern)
                    except re.error:
                        pass


class BehaviorAnalyzer:
    """行为分析器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.user_behaviors: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_behaviors: Dict[str, Dict[str, float]] = {}
        self.anomaly_threshold = 2.0  # 标准差倍数
    
    def record_behavior(self, user_id: str, event: SecurityEvent):
        """记录用户行为"""
        behavior_data = {
            "timestamp": event.timestamp,
            "event_type": event.event_type.value,
            "ip_address": event.ip_address,
            "user_agent": event.user_agent,
            "resource": event.resource,
            "action": event.action
        }
        
        self.user_behaviors[user_id].append(behavior_data)
    
    def analyze_user_behavior(self, user_id: str) -> Dict[str, Any]:
        """分析用户行为"""
        if user_id not in self.user_behaviors:
            return {"anomaly_score": 0.0, "anomalies": []}
        
        behaviors = list(self.user_behaviors[user_id])
        if len(behaviors) < 10:  # 需要足够的数据
            return {"anomaly_score": 0.0, "anomalies": []}
        
        anomalies = []
        total_anomaly_score = 0.0
        
        # 分析时间模式
        time_anomaly = self._analyze_time_pattern(behaviors)
        if time_anomaly["is_anomaly"]:
            anomalies.append(time_anomaly)
            total_anomaly_score += time_anomaly["score"]
        
        # 分析IP地址模式
        ip_anomaly = self._analyze_ip_pattern(behaviors)
        if ip_anomaly["is_anomaly"]:
            anomalies.append(ip_anomaly)
            total_anomaly_score += ip_anomaly["score"]
        
        # 分析用户代理模式
        ua_anomaly = self._analyze_user_agent_pattern(behaviors)
        if ua_anomaly["is_anomaly"]:
            anomalies.append(ua_anomaly)
            total_anomaly_score += ua_anomaly["score"]
        
        # 分析访问频率
        frequency_anomaly = self._analyze_access_frequency(behaviors)
        if frequency_anomaly["is_anomaly"]:
            anomalies.append(frequency_anomaly)
            total_anomaly_score += frequency_anomaly["score"]
        
        return {
            "anomaly_score": total_anomaly_score,
            "anomalies": anomalies,
            "behavior_count": len(behaviors)
        }
    
    def _analyze_time_pattern(self, behaviors: List[Dict]) -> Dict[str, Any]:
        """分析时间模式"""
        hours = [b["timestamp"].hour for b in behaviors if b["timestamp"]]
        
        if not hours:
            return {"is_anomaly": False, "score": 0.0}
        
        # 计算小时分布
        hour_counts = defaultdict(int)
        for hour in hours:
            hour_counts[hour] += 1
        
        # 检查是否在异常时间段（如深夜）大量活动
        night_hours = sum(hour_counts[h] for h in range(0, 6))  # 0-6点
        total_hours = len(hours)
        night_ratio = night_hours / total_hours if total_hours > 0 else 0
        
        if night_ratio > 0.5:  # 超过50%的活动在深夜
            return {
                "is_anomaly": True,
                "score": 1.0,
                "type": "unusual_time_pattern",
                "description": f"异常时间模式：{night_ratio:.1%}的活动发生在深夜"
            }
        
        return {"is_anomaly": False, "score": 0.0}
    
    def _analyze_ip_pattern(self, behaviors: List[Dict]) -> Dict[str, Any]:
        """分析IP地址模式"""
        ips = [b["ip_address"] for b in behaviors if b["ip_address"]]
        
        if not ips:
            return {"is_anomaly": False, "score": 0.0}
        
        unique_ips = set(ips)
        
        # 检查是否有过多不同的IP地址
        if len(unique_ips) > len(ips) * 0.8:  # 80%以上都是不同IP
            return {
                "is_anomaly": True,
                "score": 1.5,
                "type": "multiple_ip_addresses",
                "description": f"检测到{len(unique_ips)}个不同IP地址"
            }
        
        # 检查地理位置跳跃（需要GeoIP数据库）
        # 这里简化处理
        
        return {"is_anomaly": False, "score": 0.0}
    
    def _analyze_user_agent_pattern(self, behaviors: List[Dict]) -> Dict[str, Any]:
        """分析用户代理模式"""
        user_agents = [b["user_agent"] for b in behaviors if b["user_agent"]]
        
        if not user_agents:
            return {"is_anomaly": False, "score": 0.0}
        
        unique_uas = set(user_agents)
        
        # 检查是否频繁更换用户代理
        if len(unique_uas) > len(user_agents) * 0.5:
            return {
                "is_anomaly": True,
                "score": 1.0,
                "type": "multiple_user_agents",
                "description": f"检测到{len(unique_uas)}个不同用户代理"
            }
        
        return {"is_anomaly": False, "score": 0.0}
    
    def _analyze_access_frequency(self, behaviors: List[Dict]) -> Dict[str, Any]:
        """分析访问频率"""
        if len(behaviors) < 2:
            return {"is_anomaly": False, "score": 0.0}
        
        # 计算时间间隔
        intervals = []
        for i in range(1, len(behaviors)):
            if behaviors[i]["timestamp"] and behaviors[i-1]["timestamp"]:
                interval = (behaviors[i]["timestamp"] - behaviors[i-1]["timestamp"]).total_seconds()
                intervals.append(interval)
        
        if not intervals:
            return {"is_anomaly": False, "score": 0.0}
        
        avg_interval = sum(intervals) / len(intervals)
        
        # 检查是否有异常高频访问
        if avg_interval < 1.0:  # 平均间隔小于1秒
            return {
                "is_anomaly": True,
                "score": 2.0,
                "type": "high_frequency_access",
                "description": f"异常高频访问，平均间隔{avg_interval:.2f}秒"
            }
        
        return {"is_anomaly": False, "score": 0.0}


class AccessController:
    """访问控制器"""
    
    def __init__(self):
        self.policies: Dict[str, SecurityPolicy] = {}
        self.rate_limiters: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        self.blocked_ips: Set[str] = set()
        self.blocked_users: Set[str] = set()
    
    def add_policy(self, policy: SecurityPolicy):
        """添加安全策略"""
        self.policies[policy.name] = policy
    
    def evaluate_access(
        self,
        user_id: Optional[str],
        ip_address: str,
        resource: str,
        action: str,
        context: Dict[str, Any] = None
    ) -> Tuple[AccessControlAction, str]:
        """评估访问请求"""
        context = context or {}
        
        # 检查IP黑名单
        if ip_address in self.blocked_ips:
            return AccessControlAction.DENY, "IP地址已被阻止"
        
        # 检查用户黑名单
        if user_id and user_id in self.blocked_users:
            return AccessControlAction.DENY, "用户已被阻止"
        
        # 检查速率限制
        rate_limit_result = self._check_rate_limit(user_id, ip_address, resource)
        if rate_limit_result[0] == AccessControlAction.DENY:
            return rate_limit_result
        
        # 评估所有策略
        for policy_name, policy in self.policies.items():
            if not policy.enabled:
                continue
            
            # 按优先级排序规则
            sorted_rules = sorted(policy.rules, key=lambda r: r.priority, reverse=True)
            
            for rule in sorted_rules:
                if not rule.enabled:
                    continue
                
                if self._evaluate_rule(rule, user_id, ip_address, resource, action, context):
                    return rule.action, f"匹配规则: {rule.name}"
        
        # 如果没有规则匹配，使用默认策略
        default_policy = list(self.policies.values())[0] if self.policies else None
        if default_policy:
            return default_policy.default_action, "使用默认策略"
        
        return AccessControlAction.ALLOW, "无匹配策略，默认允许"
    
    def _evaluate_rule(
        self,
        rule: AccessRule,
        user_id: Optional[str],
        ip_address: str,
        resource: str,
        action: str,
        context: Dict[str, Any]
    ) -> bool:
        """评估单个规则"""
        conditions = rule.conditions
        
        # 检查用户ID条件
        if "user_ids" in conditions:
            if user_id not in conditions["user_ids"]:
                return False
        
        # 检查IP地址条件
        if "ip_ranges" in conditions:
            if not self._check_ip_in_ranges(ip_address, conditions["ip_ranges"]):
                return False
        
        # 检查资源条件
        if "resources" in conditions:
            if not any(re.match(pattern, resource) for pattern in conditions["resources"]):
                return False
        
        # 检查动作条件
        if "actions" in conditions:
            if action not in conditions["actions"]:
                return False
        
        # 检查时间条件
        if "time_ranges" in conditions:
            current_time = datetime.now().time()
            if not self._check_time_in_ranges(current_time, conditions["time_ranges"]):
                return False
        
        # 检查自定义条件
        if "custom" in conditions:
            for key, expected_value in conditions["custom"].items():
                if context.get(key) != expected_value:
                    return False
        
        return True
    
    def _check_ip_in_ranges(self, ip_address: str, ip_ranges: List[str]) -> bool:
        """检查IP是否在指定范围内"""
        try:
            ip = ipaddress.ip_address(ip_address)
            for ip_range in ip_ranges:
                if "/" in ip_range:
                    network = ipaddress.ip_network(ip_range, strict=False)
                    if ip in network:
                        return True
                else:
                    if ip == ipaddress.ip_address(ip_range):
                        return True
        except ValueError:
            return False
        
        return False
    
    def _check_time_in_ranges(self, current_time, time_ranges: List[Dict[str, str]]) -> bool:
        """检查时间是否在指定范围内"""
        for time_range in time_ranges:
            start_time = datetime.strptime(time_range["start"], "%H:%M").time()
            end_time = datetime.strptime(time_range["end"], "%H:%M").time()
            
            if start_time <= current_time <= end_time:
                return True
        
        return False
    
    def _check_rate_limit(
        self,
        user_id: Optional[str],
        ip_address: str,
        resource: str
    ) -> Tuple[AccessControlAction, str]:
        """检查速率限制"""
        now = time.time()
        window_size = 60  # 1分钟窗口
        max_requests = 100  # 最大请求数
        
        # 清理过期记录
        cutoff_time = now - window_size
        
        # 检查IP速率限制
        ip_requests = self.rate_limiters["ip"][ip_address]
        while ip_requests and ip_requests[0] < cutoff_time:
            ip_requests.popleft()
        
        if len(ip_requests) >= max_requests:
            return AccessControlAction.DENY, "IP速率限制超出"
        
        ip_requests.append(now)
        
        # 检查用户速率限制
        if user_id:
            user_requests = self.rate_limiters["user"][user_id]
            while user_requests and user_requests[0] < cutoff_time:
                user_requests.popleft()
            
            if len(user_requests) >= max_requests:
                return AccessControlAction.DENY, "用户速率限制超出"
            
            user_requests.append(now)
        
        return AccessControlAction.ALLOW, "速率限制检查通过"
    
    def block_ip(self, ip_address: str, duration: Optional[timedelta] = None):
        """阻止IP地址"""
        self.blocked_ips.add(ip_address)
        
        if duration:
            # 设置定时解除阻止
            async def unblock_later():
                await asyncio.sleep(duration.total_seconds())
                self.blocked_ips.discard(ip_address)
            
            asyncio.create_task(unblock_later())
    
    def block_user(self, user_id: str, duration: Optional[timedelta] = None):
        """阻止用户"""
        self.blocked_users.add(user_id)
        
        if duration:
            # 设置定时解除阻止
            async def unblock_later():
                await asyncio.sleep(duration.total_seconds())
                self.blocked_users.discard(user_id)
            
            asyncio.create_task(unblock_later())


class AdvancedCrypto:
    """高级加密工具"""
    
    def __init__(self):
        self.password_context = CryptContext(
            schemes=["argon2", "bcrypt"],
            deprecated="auto"
        )
    
    def generate_key(self, algorithm: EncryptionAlgorithm) -> bytes:
        """生成加密密钥"""
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            return get_random_bytes(32)  # 256位
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            return get_random_bytes(32)
        elif algorithm == EncryptionAlgorithm.FERNET:
            return Fernet.generate_key()
        elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
            key_size = 2048 if algorithm == EncryptionAlgorithm.RSA_2048 else 4096
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size
            )
            return private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def encrypt_data(
        self,
        data: bytes,
        key: bytes,
        algorithm: EncryptionAlgorithm
    ) -> Tuple[bytes, bytes]:
        """加密数据，返回(加密数据, 初始化向量/nonce)"""
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            nonce = get_random_bytes(12)
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce)
            )
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()
            return ciphertext + encryptor.tag, nonce
        
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            iv = get_random_bytes(16)
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv)
            )
            encryptor = cipher.encryptor()
            padded_data = pad(data, 16)
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            return ciphertext, iv
        
        elif algorithm == EncryptionAlgorithm.FERNET:
            f = Fernet(key)
            ciphertext = f.encrypt(data)
            return ciphertext, b''
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def decrypt_data(
        self,
        ciphertext: bytes,
        key: bytes,
        algorithm: EncryptionAlgorithm,
        iv_or_nonce: bytes = b''
    ) -> bytes:
        """解密数据"""
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            nonce = iv_or_nonce
            tag = ciphertext[-16:]  # 最后16字节是认证标签
            actual_ciphertext = ciphertext[:-16]
            
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce, tag)
            )
            decryptor = cipher.decryptor()
            return decryptor.update(actual_ciphertext) + decryptor.finalize()
        
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            iv = iv_or_nonce
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv)
            )
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            return unpad(padded_data, 16)
        
        elif algorithm == EncryptionAlgorithm.FERNET:
            f = Fernet(key)
            return f.decrypt(ciphertext)
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def hash_password(self, password: str) -> str:
        """哈希密码"""
        return self.password_context.hash(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """验证密码"""
        return self.password_context.verify(password, hashed)
    
    def generate_secure_token(self, length: int = 32) -> str:
        """生成安全令牌"""
        return secrets.token_urlsafe(length)
    
    def generate_otp_secret(self) -> str:
        """生成OTP密钥"""
        return pyotp.random_base32()
    
    def generate_otp_qr_code(self, secret: str, user_email: str, issuer: str) -> str:
        """生成OTP二维码"""
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=user_email,
            issuer_name=issuer
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # 转换为base64字符串
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def verify_otp(self, secret: str, token: str, window: int = 1) -> bool:
        """验证OTP令牌"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=window)


class InputSanitizer:
    """输入清理器"""
    
    def __init__(self):
        # SQL注入模式
        self.sql_injection_patterns = [
            re.compile(r"('|(\-\-)|(;)|(\||\|)|(\*|\*))", re.IGNORECASE),
            re.compile(r"(union|select|insert|delete|update|drop|create|alter|exec|execute)", re.IGNORECASE),
            re.compile(r"(script|javascript|vbscript|onload|onerror|onclick)", re.IGNORECASE)
        ]
        
        # XSS模式
        self.xss_patterns = [
            re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"on\w+\s*=", re.IGNORECASE),
            re.compile(r"<iframe[^>]*>.*?</iframe>", re.IGNORECASE | re.DOTALL)
        ]
        
        # 命令注入模式
        self.command_injection_patterns = [
            re.compile(r"[;&|`$(){}\[\]]"),
            re.compile(r"(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig)", re.IGNORECASE)
        ]
    
    def detect_sql_injection(self, input_text: str) -> bool:
        """检测SQL注入"""
        for pattern in self.sql_injection_patterns:
            if pattern.search(input_text):
                return True
        return False
    
    def detect_xss(self, input_text: str) -> bool:
        """检测XSS攻击"""
        for pattern in self.xss_patterns:
            if pattern.search(input_text):
                return True
        return False
    
    def detect_command_injection(self, input_text: str) -> bool:
        """检测命令注入"""
        for pattern in self.command_injection_patterns:
            if pattern.search(input_text):
                return True
        return False
    
    def sanitize_html(self, html_content: str) -> str:
        """清理HTML内容"""
        allowed_tags = [
            'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote'
        ]
        
        allowed_attributes = {
            '*': ['class'],
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'width', 'height']
        }
        
        return bleach.clean(
            html_content,
            tags=allowed_tags,
            attributes=allowed_attributes,
            strip=True
        )
    
    def validate_email(self, email: str) -> bool:
        """验证邮箱地址"""
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False
    
    def validate_phone(self, phone: str, region: str = "CN") -> bool:
        """验证电话号码"""
        try:
            parsed_number = phonenumbers.parse(phone, region)
            return phonenumbers.is_valid_number(parsed_number)
        except NumberParseException:
            return False
    
    def validate_url(self, url: str) -> bool:
        """验证URL"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def check_password_strength(self, password: str) -> Dict[str, Any]:
        """检查密码强度"""
        result = zxcvbn.zxcvbn(password)
        
        return {
            "score": result["score"],  # 0-4分
            "crack_time": result["crack_times_display"]["offline_slow_hashing_1e4_per_second"],
            "feedback": result["feedback"],
            "is_strong": result["score"] >= 3
        }


class FileSecurityScanner:
    """文件安全扫描器"""
    
    def __init__(self):
        self.dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs',
            '.js', '.jar', '.app', '.deb', '.pkg', '.dmg', '.iso'
        }
        
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        
        # 恶意文件签名（简化版）
        self.malware_signatures = {
            b'\x4d\x5a\x90\x00': 'PE_EXECUTABLE',
            b'\x50\x4b\x03\x04': 'ZIP_ARCHIVE',
            b'\x52\x61\x72\x21': 'RAR_ARCHIVE'
        }
    
    def scan_file(self, file_path: str) -> Dict[str, Any]:
        """扫描文件"""
        result = {
            "is_safe": True,
            "threats": [],
            "file_info": {},
            "scan_time": datetime.now()
        }
        
        try:
            # 获取文件信息
            file_info = self._get_file_info(file_path)
            result["file_info"] = file_info
            
            # 检查文件大小
            if file_info["size"] > self.max_file_size:
                result["threats"].append({
                    "type": "oversized_file",
                    "description": f"文件大小超出限制: {file_info['size']} bytes"
                })
                result["is_safe"] = False
            
            # 检查文件扩展名
            if file_info["extension"].lower() in self.dangerous_extensions:
                result["threats"].append({
                    "type": "dangerous_extension",
                    "description": f"危险文件扩展名: {file_info['extension']}"
                })
                result["is_safe"] = False
            
            # 检查文件签名
            signature_threat = self._check_file_signature(file_path)
            if signature_threat:
                result["threats"].append(signature_threat)
                result["is_safe"] = False
            
            # 检查EXIF数据（图片文件）
            if file_info["mime_type"].startswith("image/"):
                exif_threat = self._check_exif_data(file_path)
                if exif_threat:
                    result["threats"].append(exif_threat)
            
            # 计算文件哈希
            result["file_info"]["sha256"] = self._calculate_file_hash(file_path)
            
        except Exception as e:
            result["threats"].append({
                "type": "scan_error",
                "description": f"扫描错误: {str(e)}"
            })
            result["is_safe"] = False
        
        return result
    
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """获取文件信息"""
        import os
        from pathlib import Path
        
        path = Path(file_path)
        stat = path.stat()
        
        # 检测MIME类型
        try:
            mime_type = magic.from_file(file_path, mime=True)
        except:
            mime_type = "application/octet-stream"
        
        return {
            "name": path.name,
            "extension": path.suffix,
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "mime_type": mime_type
        }
    
    def _check_file_signature(self, file_path: str) -> Optional[Dict[str, Any]]:
        """检查文件签名"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
            
            for signature, sig_type in self.malware_signatures.items():
                if header.startswith(signature):
                    return {
                        "type": "suspicious_signature",
                        "description": f"检测到可疑文件签名: {sig_type}"
                    }
        except:
            pass
        
        return None
    
    def _check_exif_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """检查EXIF数据"""
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f)
            
            # 检查是否包含GPS信息
            gps_tags = [tag for tag in tags.keys() if 'GPS' in tag]
            if gps_tags:
                return {
                    "type": "privacy_risk",
                    "description": "图片包含GPS位置信息"
                }
        except:
            pass
        
        return None
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希"""
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            
            return hash_sha256.hexdigest()
        except:
            return ""


class SecurityAuditor:
    """安全审计器"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        self.threat_intelligence = ThreatIntelligence()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.access_controller = AccessController()
        self.input_sanitizer = InputSanitizer()
        self.file_scanner = FileSecurityScanner()
        self.crypto = AdvancedCrypto()
    
    async def log_security_event(self, event: SecurityEvent):
        """记录安全事件"""
        # 保存到数据库
        audit_log = SecurityAuditLog(
            event_type=event.event_type.value,
            user_id=event.user_id,
            ip_address=event.ip_address,
            user_agent=event.user_agent,
            resource=event.resource,
            action=event.action,
            threat_level=event.threat_level.value,
            details=json.dumps(event.details),
            location=json.dumps(event.location) if event.location else None,
            session_id=event.session_id,
            request_id=event.request_id,
            fingerprint=event.fingerprint,
            timestamp=event.timestamp
        )
        
        self.session.add(audit_log)
        self.session.commit()
        
        # 行为分析
        if event.user_id:
            self.behavior_analyzer.record_behavior(event.user_id, event)
            
            # 检查异常行为
            behavior_analysis = self.behavior_analyzer.analyze_user_behavior(event.user_id)
            if behavior_analysis["anomaly_score"] > 3.0:
                await self._handle_anomalous_behavior(event.user_id, behavior_analysis)
        
        # 威胁检测
        if event.ip_address:
            is_malicious, threat_level = self.threat_intelligence.check_ip_reputation(event.ip_address)
            if is_malicious and threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                await self._handle_threat_detection(event, threat_level)
    
    async def _handle_anomalous_behavior(self, user_id: str, analysis: Dict[str, Any]):
        """处理异常行为"""
        logger.warning(f"检测到用户 {user_id} 的异常行为: {analysis}")
        
        # 可以实施额外的安全措施
        # 例如：要求重新认证、临时限制访问等
        
        # 创建安全事件
        event = SecurityEvent(
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            timestamp=datetime.now(),
            user_id=user_id,
            threat_level=ThreatLevel.MEDIUM,
            details={
                "anomaly_score": analysis["anomaly_score"],
                "anomalies": analysis["anomalies"]
            }
        )
        
        await self.log_security_event(event)
    
    async def _handle_threat_detection(self, event: SecurityEvent, threat_level: ThreatLevel):
        """处理威胁检测"""
        logger.error(f"检测到威胁: IP {event.ip_address}, 威胁等级: {threat_level.value}")
        
        # 自动阻止高威胁IP
        if threat_level == ThreatLevel.CRITICAL:
            self.access_controller.block_ip(event.ip_address, timedelta(hours=24))
        elif threat_level == ThreatLevel.HIGH:
            self.access_controller.block_ip(event.ip_address, timedelta(hours=1))
    
    def generate_security_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """生成安全报告"""
        # 查询安全事件
        events = self.session.query(SecurityAuditLog).filter(
            SecurityAuditLog.timestamp >= start_date,
            SecurityAuditLog.timestamp <= end_date
        ).all()
        
        # 统计分析
        event_counts = defaultdict(int)
        threat_levels = defaultdict(int)
        top_ips = defaultdict(int)
        top_users = defaultdict(int)
        
        for event in events:
            event_counts[event.event_type] += 1
            threat_levels[event.threat_level] += 1
            if event.ip_address:
                top_ips[event.ip_address] += 1
            if event.user_id:
                top_users[event.user_id] += 1
        
        return {
            "period": {
                "start": start_date,
                "end": end_date
            },
            "total_events": len(events),
            "event_types": dict(event_counts),
            "threat_levels": dict(threat_levels),
            "top_ips": dict(sorted(top_ips.items(), key=lambda x: x[1], reverse=True)[:10]),
            "top_users": dict(sorted(top_users.items(), key=lambda x: x[1], reverse=True)[:10]),
            "security_score": self._calculate_security_score(events)
        }
    
    def _calculate_security_score(self, events: List[SecurityAuditLog]) -> float:
        """计算安全评分"""
        if not events:
            return 100.0
        
        # 基础分数
        base_score = 100.0
        
        # 根据威胁等级扣分
        for event in events:
            if event.threat_level == "critical":
                base_score -= 10
            elif event.threat_level == "high":
                base_score -= 5
            elif event.threat_level == "medium":
                base_score -= 2
            elif event.threat_level == "low":
                base_score -= 0.5
        
        return max(0.0, base_score)
    
    def close(self):
        """关闭数据库连接"""
        self.session.close()


# 使用示例
if __name__ == "__main__":
    async def example_usage():
        # 创建安全审计器
        auditor = SecurityAuditor("sqlite:///security_audit.db")
        
        # 创建安全事件
        event = SecurityEvent(
            event_type=SecurityEventType.LOGIN_ATTEMPT,
            timestamp=datetime.now(),
            user_id="user123",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0...",
            resource="/login",
            action="POST",
            threat_level=ThreatLevel.LOW
        )
        
        # 记录事件
        await auditor.log_security_event(event)
        
        # 生成安全报告
        report = auditor.generate_security_report(
            datetime.now() - timedelta(days=7),
            datetime.now()
        )
        
        print(f"安全报告: {report}")
        
        # 文件安全扫描
        scanner = FileSecurityScanner()
        # scan_result = scanner.scan_file("/path/to/file")
        # print(f"文件扫描结果: {scan_result}")
        
        # 密码强度检查
        sanitizer = InputSanitizer()
        password_strength = sanitizer.check_password_strength("MySecurePassword123!")
        print(f"密码强度: {password_strength}")
        
        # 加密工具
        crypto = AdvancedCrypto()
        key = crypto.generate_key(EncryptionAlgorithm.AES_256_GCM)
        encrypted_data, nonce = crypto.encrypt_data(b"Hello, World!", key, EncryptionAlgorithm.AES_256_GCM)
        decrypted_data = crypto.decrypt_data(encrypted_data, key, EncryptionAlgorithm.AES_256_GCM, nonce)
        print(f"加密解密测试: {decrypted_data}")
        
        auditor.close()
    
    # 运行示例
    asyncio.run(example_usage())