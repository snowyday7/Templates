#!/usr/bin/env python3
"""
通知服务

提供通知相关的业务逻辑，包括：
1. 邮件通知
2. 短信通知
3. 推送通知
4. 站内消息
5. 通知模板管理
6. 通知历史记录
"""

import smtplib
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from jinja2 import Template
import aiohttp
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func

from .base import BaseService
from ..models.notification import Notification, NotificationTemplate
from ..core.exceptions import NotificationException, ValidationException
from ..utils.logger import get_logger
from ..core.config import get_settings


# =============================================================================
# 常量定义
# =============================================================================

# 通知类型
class NotificationType:
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"
    WEBHOOK = "webhook"


# 通知状态
class NotificationStatus:
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


# 通知优先级
class NotificationPriority:
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# 预定义模板
DEFAULT_TEMPLATES = {
    "welcome": {
        "subject": "欢迎加入 {{app_name}}",
        "content": "亲爱的 {{user_name}}，欢迎加入我们的平台！",
        "type": "email"
    },
    "password_reset": {
        "subject": "密码重置请求",
        "content": "您的密码重置链接：{{reset_link}}，有效期30分钟。",
        "type": "email"
    },
    "email_verification": {
        "subject": "邮箱验证",
        "content": "请点击链接验证您的邮箱：{{verification_link}}",
        "type": "email"
    },
    "login_alert": {
        "subject": "登录提醒",
        "content": "您的账户在 {{login_time}} 从 {{login_location}} 登录。",
        "type": "email"
    },
    "security_alert": {
        "subject": "安全提醒",
        "content": "检测到您的账户存在异常活动，请及时检查。",
        "type": "email"
    }
}


# =============================================================================
# 通知服务类
# =============================================================================

class NotificationService(BaseService[Notification]):
    """
    通知服务类
    
    提供通知相关的业务逻辑操作
    """
    
    def __init__(self, db: AsyncSession):
        """
        初始化通知服务
        
        Args:
            db: 数据库会话
        """
        super().__init__(db, Notification)
        self.logger = get_logger(__name__)
        self.settings = get_settings()
        
        # 初始化发送器
        self._email_config = {
            'smtp_server': self.settings.SMTP_SERVER,
            'smtp_port': self.settings.SMTP_PORT,
            'smtp_username': self.settings.SMTP_USERNAME,
            'smtp_password': self.settings.SMTP_PASSWORD,
            'use_tls': self.settings.SMTP_USE_TLS,
            'from_email': self.settings.FROM_EMAIL,
            'from_name': self.settings.FROM_NAME
        }
        
        # 短信配置
        self._sms_config = {
            'api_key': getattr(self.settings, 'SMS_API_KEY', ''),
            'api_url': getattr(self.settings, 'SMS_API_URL', ''),
            'sender': getattr(self.settings, 'SMS_SENDER', '')
        }
        
        # 推送配置
        self._push_config = {
            'fcm_server_key': getattr(self.settings, 'FCM_SERVER_KEY', ''),
            'apns_key_id': getattr(self.settings, 'APNS_KEY_ID', ''),
            'apns_team_id': getattr(self.settings, 'APNS_TEAM_ID', '')
        }
        
        # 重试配置
        self._retry_config = {
            'max_retries': 3,
            'retry_delays': [60, 300, 900]  # 1分钟, 5分钟, 15分钟
        }
    
    # =========================================================================
    # 发送通知
    # =========================================================================
    
    async def send_notification(
        self,
        notification_type: str,
        recipient: str,
        template_name: str,
        template_data: Dict[str, Any],
        user_id: Optional[int] = None,
        priority: str = NotificationPriority.NORMAL,
        scheduled_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """
        发送通知
        
        Args:
            notification_type: 通知类型
            recipient: 接收者
            template_name: 模板名称
            template_data: 模板数据
            user_id: 用户ID
            priority: 优先级
            scheduled_at: 计划发送时间
            metadata: 元数据
            
        Returns:
            通知记录
        """
        try:
            # 获取模板
            template = await self._get_template(template_name, notification_type)
            if not template:
                raise NotificationException(f"模板不存在: {template_name}")
            
            # 渲染内容
            subject = await self._render_template(template.subject, template_data)
            content = await self._render_template(template.content, template_data)
            
            # 创建通知记录
            notification = Notification(
                type=notification_type,
                recipient=recipient,
                subject=subject,
                content=content,
                template_name=template_name,
                template_data=template_data,
                user_id=user_id,
                priority=priority,
                status=NotificationStatus.PENDING,
                scheduled_at=scheduled_at or datetime.utcnow(),
                metadata=metadata or {},
                retry_count=0
            )
            
            # 保存到数据库
            created_notification = await self.create(notification)
            
            # 如果是立即发送，则发送通知
            if not scheduled_at or scheduled_at <= datetime.utcnow():
                await self._send_notification_now(created_notification)
            
            return created_notification
            
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
            raise NotificationException(f"发送通知失败: {str(e)}")
    
    async def send_email(
        self,
        to_email: str,
        subject: str,
        content: str,
        content_type: str = "html",
        attachments: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[int] = None
    ) -> Notification:
        """
        发送邮件
        
        Args:
            to_email: 收件人邮箱
            subject: 邮件主题
            content: 邮件内容
            content_type: 内容类型（html/plain）
            attachments: 附件列表
            user_id: 用户ID
            
        Returns:
            通知记录
        """
        try:
            # 创建通知记录
            notification = Notification(
                type=NotificationType.EMAIL,
                recipient=to_email,
                subject=subject,
                content=content,
                user_id=user_id,
                status=NotificationStatus.PENDING,
                metadata={
                    'content_type': content_type,
                    'attachments': attachments or []
                }
            )
            
            # 保存到数据库
            created_notification = await self.create(notification)
            
            # 发送邮件
            await self._send_email_notification(created_notification)
            
            return created_notification
            
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")
            raise NotificationException(f"发送邮件失败: {str(e)}")
    
    async def send_sms(
        self,
        phone_number: str,
        message: str,
        user_id: Optional[int] = None
    ) -> Notification:
        """
        发送短信
        
        Args:
            phone_number: 手机号码
            message: 短信内容
            user_id: 用户ID
            
        Returns:
            通知记录
        """
        try:
            # 创建通知记录
            notification = Notification(
                type=NotificationType.SMS,
                recipient=phone_number,
                content=message,
                user_id=user_id,
                status=NotificationStatus.PENDING
            )
            
            # 保存到数据库
            created_notification = await self.create(notification)
            
            # 发送短信
            await self._send_sms_notification(created_notification)
            
            return created_notification
            
        except Exception as e:
            self.logger.error(f"Error sending SMS: {e}")
            raise NotificationException(f"发送短信失败: {str(e)}")
    
    async def send_push_notification(
        self,
        device_token: str,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None
    ) -> Notification:
        """
        发送推送通知
        
        Args:
            device_token: 设备令牌
            title: 通知标题
            body: 通知内容
            data: 附加数据
            user_id: 用户ID
            
        Returns:
            通知记录
        """
        try:
            # 创建通知记录
            notification = Notification(
                type=NotificationType.PUSH,
                recipient=device_token,
                subject=title,
                content=body,
                user_id=user_id,
                status=NotificationStatus.PENDING,
                metadata={'data': data or {}}
            )
            
            # 保存到数据库
            created_notification = await self.create(notification)
            
            # 发送推送
            await self._send_push_notification(created_notification)
            
            return created_notification
            
        except Exception as e:
            self.logger.error(f"Error sending push notification: {e}")
            raise NotificationException(f"发送推送通知失败: {str(e)}")
    
    async def send_in_app_notification(
        self,
        user_id: int,
        title: str,
        content: str,
        action_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """
        发送站内通知
        
        Args:
            user_id: 用户ID
            title: 通知标题
            content: 通知内容
            action_url: 操作链接
            metadata: 元数据
            
        Returns:
            通知记录
        """
        try:
            # 创建通知记录
            notification = Notification(
                type=NotificationType.IN_APP,
                recipient=str(user_id),
                subject=title,
                content=content,
                user_id=user_id,
                status=NotificationStatus.SENT,  # 站内通知直接标记为已发送
                metadata={
                    'action_url': action_url,
                    **(metadata or {})
                }
            )
            
            # 保存到数据库
            created_notification = await self.create(notification)
            
            # 缓存到Redis（用于实时推送）
            await self._cache_in_app_notification(created_notification)
            
            return created_notification
            
        except Exception as e:
            self.logger.error(f"Error sending in-app notification: {e}")
            raise NotificationException(f"发送站内通知失败: {str(e)}")
    
    # =========================================================================
    # 批量发送
    # =========================================================================
    
    async def send_bulk_notifications(
        self,
        notification_type: str,
        recipients: List[str],
        template_name: str,
        template_data: Dict[str, Any],
        user_ids: Optional[List[int]] = None,
        priority: str = NotificationPriority.NORMAL
    ) -> List[Notification]:
        """
        批量发送通知
        
        Args:
            notification_type: 通知类型
            recipients: 接收者列表
            template_name: 模板名称
            template_data: 模板数据
            user_ids: 用户ID列表
            priority: 优先级
            
        Returns:
            通知记录列表
        """
        try:
            notifications = []
            
            for i, recipient in enumerate(recipients):
                user_id = user_ids[i] if user_ids and i < len(user_ids) else None
                
                notification = await self.send_notification(
                    notification_type=notification_type,
                    recipient=recipient,
                    template_name=template_name,
                    template_data=template_data,
                    user_id=user_id,
                    priority=priority
                )
                
                notifications.append(notification)
                
                # 添加延迟以避免过载
                if i % 10 == 9:  # 每10个通知后暂停
                    await asyncio.sleep(0.1)
            
            self.logger.info(f"Sent {len(notifications)} bulk notifications")
            
            return notifications
            
        except Exception as e:
            self.logger.error(f"Error sending bulk notifications: {e}")
            raise NotificationException(f"批量发送通知失败: {str(e)}")
    
    # =========================================================================
    # 通知查询
    # =========================================================================
    
    async def get_user_notifications(
        self,
        user_id: int,
        notification_type: Optional[str] = None,
        status: Optional[str] = None,
        unread_only: bool = False,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        获取用户通知列表
        
        Args:
            user_id: 用户ID
            notification_type: 通知类型
            status: 通知状态
            unread_only: 只获取未读通知
            page: 页码
            page_size: 每页大小
            
        Returns:
            通知列表和分页信息
        """
        try:
            # 构建查询条件
            conditions = [Notification.user_id == user_id]
            
            if notification_type:
                conditions.append(Notification.type == notification_type)
            
            if status:
                conditions.append(Notification.status == status)
            
            if unread_only:
                conditions.append(Notification.read_at.is_(None))
            
            # 获取通知列表
            result = await self.get_with_pagination(
                conditions=and_(*conditions),
                page=page,
                page_size=page_size,
                order_by=[Notification.created_at.desc()]
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting user notifications: {e}")
            raise NotificationException(f"获取用户通知失败: {str(e)}")
    
    async def get_unread_count(self, user_id: int) -> int:
        """
        获取用户未读通知数量
        
        Args:
            user_id: 用户ID
            
        Returns:
            未读通知数量
        """
        try:
            query = select(func.count(Notification.id)).where(
                and_(
                    Notification.user_id == user_id,
                    Notification.read_at.is_(None)
                )
            )
            result = await self.db.execute(query)
            return result.scalar() or 0
            
        except Exception as e:
            self.logger.error(f"Error getting unread count: {e}")
            return 0
    
    async def mark_as_read(
        self,
        notification_id: int,
        user_id: Optional[int] = None
    ) -> bool:
        """
        标记通知为已读
        
        Args:
            notification_id: 通知ID
            user_id: 用户ID（用于权限检查）
            
        Returns:
            是否标记成功
        """
        try:
            # 构建查询条件
            conditions = [Notification.id == notification_id]
            if user_id:
                conditions.append(Notification.user_id == user_id)
            
            # 更新通知
            updated = await self.update(
                notification_id,
                {'read_at': datetime.utcnow()},
                conditions=and_(*conditions)
            )
            
            return updated is not None
            
        except Exception as e:
            self.logger.error(f"Error marking notification as read: {e}")
            return False
    
    async def mark_all_as_read(self, user_id: int) -> int:
        """
        标记用户所有通知为已读
        
        Args:
            user_id: 用户ID
            
        Returns:
            标记的通知数量
        """
        try:
            # 获取未读通知
            query = select(Notification).where(
                and_(
                    Notification.user_id == user_id,
                    Notification.read_at.is_(None)
                )
            )
            result = await self.db.execute(query)
            notifications = result.fetchall()
            
            # 批量更新
            count = 0
            for notification in notifications:
                await self.update(notification.id, {'read_at': datetime.utcnow()})
                count += 1
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error marking all notifications as read: {e}")
            return 0
    
    # =========================================================================
    # 模板管理
    # =========================================================================
    
    async def create_template(
        self,
        name: str,
        subject: str,
        content: str,
        notification_type: str,
        description: Optional[str] = None,
        variables: Optional[List[str]] = None
    ) -> NotificationTemplate:
        """
        创建通知模板
        
        Args:
            name: 模板名称
            subject: 主题模板
            content: 内容模板
            notification_type: 通知类型
            description: 模板描述
            variables: 模板变量
            
        Returns:
            模板记录
        """
        try:
            # 检查模板名称是否已存在
            existing = await self._get_template(name, notification_type)
            if existing:
                raise ValidationException(f"模板已存在: {name}")
            
            # 创建模板
            template = NotificationTemplate(
                name=name,
                subject=subject,
                content=content,
                type=notification_type,
                description=description,
                variables=variables or [],
                is_active=True
            )
            
            # 保存到数据库
            created_template = await self.create(template)
            
            self.logger.info(f"Notification template created: {name}")
            
            return created_template
            
        except Exception as e:
            self.logger.error(f"Error creating template: {e}")
            raise NotificationException(f"创建模板失败: {str(e)}")
    
    async def update_template(
        self,
        template_id: int,
        subject: Optional[str] = None,
        content: Optional[str] = None,
        description: Optional[str] = None,
        variables: Optional[List[str]] = None,
        is_active: Optional[bool] = None
    ) -> NotificationTemplate:
        """
        更新通知模板
        
        Args:
            template_id: 模板ID
            subject: 主题模板
            content: 内容模板
            description: 模板描述
            variables: 模板变量
            is_active: 是否激活
            
        Returns:
            更新后的模板
        """
        try:
            # 构建更新数据
            update_data = {}
            if subject is not None:
                update_data['subject'] = subject
            if content is not None:
                update_data['content'] = content
            if description is not None:
                update_data['description'] = description
            if variables is not None:
                update_data['variables'] = variables
            if is_active is not None:
                update_data['is_active'] = is_active
            
            # 更新模板
            updated_template = await self.update(template_id, update_data)
            
            self.logger.info(f"Notification template updated: {template_id}")
            
            return updated_template
            
        except Exception as e:
            self.logger.error(f"Error updating template: {e}")
            raise NotificationException(f"更新模板失败: {str(e)}")
    
    # =========================================================================
    # 重试和错误处理
    # =========================================================================
    
    async def retry_failed_notifications(self, max_age_hours: int = 24) -> int:
        """
        重试失败的通知
        
        Args:
            max_age_hours: 最大重试时间（小时）
            
        Returns:
            重试的通知数量
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            # 查询需要重试的通知
            query = select(Notification).where(
                and_(
                    Notification.status == NotificationStatus.FAILED,
                    Notification.retry_count < self._retry_config['max_retries'],
                    Notification.created_at >= cutoff_time
                )
            )
            result = await self.db.execute(query)
            failed_notifications = result.fetchall()
            
            retry_count = 0
            for notification in failed_notifications:
                try:
                    await self._send_notification_now(notification)
                    retry_count += 1
                except Exception as e:
                    self.logger.error(f"Error retrying notification {notification.id}: {e}")
                    continue
            
            self.logger.info(f"Retried {retry_count} failed notifications")
            
            return retry_count
            
        except Exception as e:
            self.logger.error(f"Error retrying failed notifications: {e}")
            return 0
    
    async def process_scheduled_notifications(self) -> int:
        """
        处理计划发送的通知
        
        Returns:
            处理的通知数量
        """
        try:
            # 查询需要发送的计划通知
            query = select(Notification).where(
                and_(
                    Notification.status == NotificationStatus.PENDING,
                    Notification.scheduled_at <= datetime.utcnow()
                )
            )
            result = await self.db.execute(query)
            scheduled_notifications = result.fetchall()
            
            processed_count = 0
            for notification in scheduled_notifications:
                try:
                    await self._send_notification_now(notification)
                    processed_count += 1
                except Exception as e:
                    self.logger.error(f"Error processing scheduled notification {notification.id}: {e}")
                    continue
            
            self.logger.info(f"Processed {processed_count} scheduled notifications")
            
            return processed_count
            
        except Exception as e:
            self.logger.error(f"Error processing scheduled notifications: {e}")
            return 0
    
    # =========================================================================
    # 统计和分析
    # =========================================================================
    
    async def get_notification_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        获取通知统计信息
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            统计信息
        """
        try:
            # 设置默认时间范围（最近30天）
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # 基础查询条件
            base_conditions = [
                Notification.created_at >= start_date,
                Notification.created_at <= end_date
            ]
            
            # 总通知数
            total_query = select(func.count(Notification.id)).where(and_(*base_conditions))
            total_result = await self.db.execute(total_query)
            total_count = total_result.scalar() or 0
            
            # 按类型统计
            type_query = select(
                Notification.type,
                func.count(Notification.id).label('count')
            ).where(and_(*base_conditions)).group_by(Notification.type)
            type_result = await self.db.execute(type_query)
            type_stats = {row.type: row.count for row in type_result.fetchall()}
            
            # 按状态统计
            status_query = select(
                Notification.status,
                func.count(Notification.id).label('count')
            ).where(and_(*base_conditions)).group_by(Notification.status)
            status_result = await self.db.execute(status_query)
            status_stats = {row.status: row.count for row in status_result.fetchall()}
            
            # 成功率
            sent_count = status_stats.get(NotificationStatus.SENT, 0) + status_stats.get(NotificationStatus.DELIVERED, 0)
            success_rate = (sent_count / total_count * 100) if total_count > 0 else 0
            
            return {
                'total_notifications': total_count,
                'success_rate': round(success_rate, 2),
                'by_type': type_stats,
                'by_status': status_stats,
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting notification statistics: {e}")
            raise NotificationException(f"获取通知统计失败: {str(e)}")
    
    # =========================================================================
    # 私有辅助方法
    # =========================================================================
    
    async def _get_template(
        self,
        name: str,
        notification_type: str
    ) -> Optional[NotificationTemplate]:
        """获取通知模板"""
        try:
            # 先从数据库查询
            query = select(NotificationTemplate).where(
                and_(
                    NotificationTemplate.name == name,
                    NotificationTemplate.type == notification_type,
                    NotificationTemplate.is_active == True
                )
            )
            result = await self.db.execute(query)
            template = result.scalar_one_or_none()
            
            if template:
                return template
            
            # 如果数据库中没有，检查默认模板
            if name in DEFAULT_TEMPLATES:
                default_template = DEFAULT_TEMPLATES[name]
                if default_template['type'] == notification_type:
                    # 创建临时模板对象
                    return NotificationTemplate(
                        name=name,
                        subject=default_template['subject'],
                        content=default_template['content'],
                        type=notification_type
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting template {name}: {e}")
            return None
    
    async def _render_template(self, template_str: str, data: Dict[str, Any]) -> str:
        """渲染模板"""
        try:
            template = Template(template_str)
            return template.render(**data)
        except Exception as e:
            self.logger.error(f"Error rendering template: {e}")
            return template_str
    
    async def _send_notification_now(self, notification: Notification):
        """立即发送通知"""
        try:
            if notification.type == NotificationType.EMAIL:
                await self._send_email_notification(notification)
            elif notification.type == NotificationType.SMS:
                await self._send_sms_notification(notification)
            elif notification.type == NotificationType.PUSH:
                await self._send_push_notification(notification)
            elif notification.type == NotificationType.IN_APP:
                await self._cache_in_app_notification(notification)
            else:
                raise NotificationException(f"不支持的通知类型: {notification.type}")
                
        except Exception as e:
            # 更新失败状态
            await self._update_notification_status(
                notification,
                NotificationStatus.FAILED,
                error_message=str(e)
            )
            raise
    
    async def _send_email_notification(self, notification: Notification):
        """发送邮件通知"""
        try:
            # 创建邮件消息
            msg = MIMEMultipart()
            msg['From'] = f"{self._email_config['from_name']} <{self._email_config['from_email']}>"
            msg['To'] = notification.recipient
            msg['Subject'] = notification.subject
            
            # 添加邮件内容
            content_type = notification.metadata.get('content_type', 'html')
            msg.attach(MIMEText(notification.content, content_type, 'utf-8'))
            
            # 添加附件
            attachments = notification.metadata.get('attachments', [])
            for attachment in attachments:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment['content'])
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {attachment["filename"]}'
                )
                msg.attach(part)
            
            # 发送邮件
            with smtplib.SMTP(self._email_config['smtp_server'], self._email_config['smtp_port']) as server:
                if self._email_config['use_tls']:
                    server.starttls()
                
                server.login(self._email_config['smtp_username'], self._email_config['smtp_password'])
                server.send_message(msg)
            
            # 更新状态
            await self._update_notification_status(notification, NotificationStatus.SENT)
            
        except Exception as e:
            self.logger.error(f"Error sending email notification {notification.id}: {e}")
            raise
    
    async def _send_sms_notification(self, notification: Notification):
        """发送短信通知"""
        try:
            if not self._sms_config['api_key']:
                raise NotificationException("短信服务未配置")
            
            # 构建请求数据
            data = {
                'to': notification.recipient,
                'message': notification.content,
                'sender': self._sms_config['sender']
            }
            
            # 发送HTTP请求
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self._sms_config["api_key"]}',
                    'Content-Type': 'application/json'
                }
                
                async with session.post(
                    self._sms_config['api_url'],
                    json=data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        await self._update_notification_status(notification, NotificationStatus.SENT)
                    else:
                        error_text = await response.text()
                        raise NotificationException(f"短信发送失败: {error_text}")
            
        except Exception as e:
            self.logger.error(f"Error sending SMS notification {notification.id}: {e}")
            raise
    
    async def _send_push_notification(self, notification: Notification):
        """发送推送通知"""
        try:
            if not self._push_config['fcm_server_key']:
                raise NotificationException("推送服务未配置")
            
            # 构建FCM请求数据
            data = {
                'to': notification.recipient,
                'notification': {
                    'title': notification.subject,
                    'body': notification.content
                },
                'data': notification.metadata.get('data', {})
            }
            
            # 发送FCM请求
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'key={self._push_config["fcm_server_key"]}',
                    'Content-Type': 'application/json'
                }
                
                async with session.post(
                    'https://fcm.googleapis.com/fcm/send',
                    json=data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        await self._update_notification_status(notification, NotificationStatus.SENT)
                    else:
                        error_text = await response.text()
                        raise NotificationException(f"推送发送失败: {error_text}")
            
        except Exception as e:
            self.logger.error(f"Error sending push notification {notification.id}: {e}")
            raise
    
    async def _cache_in_app_notification(self, notification: Notification):
        """缓存站内通知"""
        try:
            # 缓存到Redis用于实时推送
            cache_key = f"notifications:user:{notification.user_id}"
            notification_data = {
                'id': notification.id,
                'subject': notification.subject,
                'content': notification.content,
                'created_at': notification.created_at.isoformat(),
                'metadata': notification.metadata
            }
            
            # 添加到用户通知列表
            await self.cache.lpush(cache_key, json.dumps(notification_data))
            
            # 限制列表长度
            await self.cache.ltrim(cache_key, 0, 99)  # 保留最新100条
            
            # 设置过期时间
            await self.cache.expire(cache_key, 86400 * 7)  # 7天
            
            # 更新状态
            await self._update_notification_status(notification, NotificationStatus.SENT)
            
        except Exception as e:
            self.logger.error(f"Error caching in-app notification {notification.id}: {e}")
            raise
    
    async def _update_notification_status(
        self,
        notification: Notification,
        status: str,
        error_message: Optional[str] = None
    ):
        """更新通知状态"""
        try:
            update_data = {
                'status': status,
                'sent_at': datetime.utcnow() if status == NotificationStatus.SENT else None
            }
            
            if status == NotificationStatus.FAILED:
                update_data['retry_count'] = (notification.retry_count or 0) + 1
                update_data['error_message'] = error_message
            
            await self.update(notification.id, update_data)
            
        except Exception as e:
            self.logger.error(f"Error updating notification status: {e}")