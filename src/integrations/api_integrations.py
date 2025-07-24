# -*- coding: utf-8 -*-
"""
API集成模块
提供第三方服务集成功能，包括支付、短信、邮件、云存储、社交登录等
"""

import asyncio
import aiohttp
import json
import base64
import hmac
import hashlib
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from urllib.parse import urlencode, quote
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import xml.etree.ElementTree as ET
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import ssl
from pathlib import Path
import mimetypes

logger = logging.getLogger(__name__)


class PaymentProvider(Enum):
    """支付提供商枚举"""
    ALIPAY = "alipay"
    WECHAT = "wechat"
    STRIPE = "stripe"
    PAYPAL = "paypal"
    UNIONPAY = "unionpay"


class SMSProvider(Enum):
    """短信提供商枚举"""
    ALIYUN = "aliyun"
    TENCENT = "tencent"
    TWILIO = "twilio"
    HUAWEI = "huawei"


class EmailProvider(Enum):
    """邮件提供商枚举"""
    SMTP = "smtp"
    SENDGRID = "sendgrid"
    MAILGUN = "mailgun"
    ALIYUN_DM = "aliyun_dm"
    TENCENT_SES = "tencent_ses"


class CloudProvider(Enum):
    """云存储提供商枚举"""
    ALIYUN_OSS = "aliyun_oss"
    TENCENT_COS = "tencent_cos"
    AWS_S3 = "aws_s3"
    QINIU = "qiniu"
    UPYUN = "upyun"


class SocialProvider(Enum):
    """社交登录提供商枚举"""
    WECHAT = "wechat"
    QQ = "qq"
    WEIBO = "weibo"
    GITHUB = "github"
    GOOGLE = "google"
    FACEBOOK = "facebook"


@dataclass
class IntegrationConfig:
    """集成配置"""
    provider: str
    app_id: Optional[str] = None
    app_secret: Optional[str] = None
    api_key: Optional[str] = None
    access_token: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationResult:
    """集成结果"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    response_time: float = 0.0


class BaseIntegration(ABC):
    """基础集成类"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def authenticate(self) -> IntegrationResult:
        """认证"""
        pass
    
    async def make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        params: Optional[Dict[str, str]] = None
    ) -> IntegrationResult:
        """发送请求"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            if not self.session:
                raise ValueError("Session not initialized")
            
            async with self.session.request(
                method=method,
                url=url,
                headers=headers,
                json=data if isinstance(data, dict) else None,
                data=data if not isinstance(data, dict) else None,
                params=params
            ) as response:
                response_time = time.time() - start_time
                
                try:
                    response_data = await response.json()
                except:
                    response_data = await response.text()
                
                return IntegrationResult(
                    success=response.status < 400,
                    data=response_data,
                    error=None if response.status < 400 else f"HTTP {response.status}",
                    request_id=request_id,
                    response_time=response_time,
                    metadata={
                        "status_code": response.status,
                        "headers": dict(response.headers)
                    }
                )
        
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Request failed: {e}")
            return IntegrationResult(
                success=False,
                error=str(e),
                request_id=request_id,
                response_time=response_time
            )


class PaymentIntegration(BaseIntegration):
    """支付集成"""
    
    async def authenticate(self) -> IntegrationResult:
        """认证"""
        return IntegrationResult(success=True)
    
    async def create_order(
        self,
        amount: float,
        currency: str = "CNY",
        order_id: str = None,
        description: str = "",
        notify_url: str = None,
        return_url: str = None,
        extra_params: Dict[str, Any] = None
    ) -> IntegrationResult:
        """创建订单"""
        try:
            if not order_id:
                order_id = str(uuid.uuid4())
            
            if self.config.provider == PaymentProvider.ALIPAY.value:
                return await self._create_alipay_order(
                    amount, currency, order_id, description,
                    notify_url, return_url, extra_params
                )
            elif self.config.provider == PaymentProvider.WECHAT.value:
                return await self._create_wechat_order(
                    amount, currency, order_id, description,
                    notify_url, return_url, extra_params
                )
            elif self.config.provider == PaymentProvider.STRIPE.value:
                return await self._create_stripe_order(
                    amount, currency, order_id, description,
                    notify_url, return_url, extra_params
                )
            else:
                return IntegrationResult(
                    success=False,
                    error=f"Unsupported payment provider: {self.config.provider}"
                )
        
        except Exception as e:
            return IntegrationResult(
                success=False,
                error=str(e)
            )
    
    async def _create_alipay_order(
        self, amount, currency, order_id, description,
        notify_url, return_url, extra_params
    ) -> IntegrationResult:
        """创建支付宝订单"""
        params = {
            "app_id": self.config.app_id,
            "method": "alipay.trade.page.pay",
            "charset": "utf-8",
            "sign_type": "RSA2",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0",
            "biz_content": json.dumps({
                "out_trade_no": order_id,
                "total_amount": str(amount),
                "subject": description,
                "product_code": "FAST_INSTANT_TRADE_PAY"
            })
        }
        
        if notify_url:
            params["notify_url"] = notify_url
        if return_url:
            params["return_url"] = return_url
        
        # 这里需要实现签名逻辑
        sign = self._generate_alipay_sign(params)
        params["sign"] = sign
        
        payment_url = f"{self.config.base_url}?{urlencode(params)}"
        
        return IntegrationResult(
            success=True,
            data={
                "order_id": order_id,
                "payment_url": payment_url,
                "amount": amount,
                "currency": currency
            }
        )
    
    async def _create_wechat_order(
        self, amount, currency, order_id, description,
        notify_url, return_url, extra_params
    ) -> IntegrationResult:
        """创建微信订单"""
        # 微信支付统一下单API
        url = f"{self.config.base_url}/pay/unifiedorder"
        
        data = {
            "appid": self.config.app_id,
            "mch_id": self.config.extra_config.get("mch_id"),
            "nonce_str": str(uuid.uuid4()).replace("-", ""),
            "body": description,
            "out_trade_no": order_id,
            "total_fee": int(amount * 100),  # 微信金额单位为分
            "spbill_create_ip": "127.0.0.1",
            "trade_type": "NATIVE"
        }
        
        if notify_url:
            data["notify_url"] = notify_url
        
        # 生成签名
        sign = self._generate_wechat_sign(data)
        data["sign"] = sign
        
        # 转换为XML
        xml_data = self._dict_to_xml(data)
        
        result = await self.make_request(
            "POST", url, 
            headers={"Content-Type": "application/xml"},
            data=xml_data
        )
        
        if result.success:
            # 解析XML响应
            response_data = self._xml_to_dict(result.data)
            if response_data.get("return_code") == "SUCCESS":
                return IntegrationResult(
                    success=True,
                    data={
                        "order_id": order_id,
                        "prepay_id": response_data.get("prepay_id"),
                        "code_url": response_data.get("code_url"),
                        "amount": amount,
                        "currency": currency
                    }
                )
        
        return result
    
    async def _create_stripe_order(
        self, amount, currency, order_id, description,
        notify_url, return_url, extra_params
    ) -> IntegrationResult:
        """创建Stripe订单"""
        url = f"{self.config.base_url}/v1/payment_intents"
        
        data = {
            "amount": int(amount * 100),  # Stripe金额单位为分
            "currency": currency.lower(),
            "metadata": {"order_id": order_id},
            "description": description
        }
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        result = await self.make_request(
            "POST", url, headers=headers, data=urlencode(data)
        )
        
        if result.success:
            return IntegrationResult(
                success=True,
                data={
                    "order_id": order_id,
                    "payment_intent_id": result.data.get("id"),
                    "client_secret": result.data.get("client_secret"),
                    "amount": amount,
                    "currency": currency
                }
            )
        
        return result
    
    def _generate_alipay_sign(self, params: Dict[str, str]) -> str:
        """生成支付宝签名"""
        # 这里需要实现RSA签名逻辑
        # 简化实现，实际需要使用私钥签名
        sorted_params = sorted(params.items())
        sign_string = "&".join([f"{k}={v}" for k, v in sorted_params if k != "sign"])
        return hashlib.md5(sign_string.encode()).hexdigest()
    
    def _generate_wechat_sign(self, params: Dict[str, Any]) -> str:
        """生成微信签名"""
        sorted_params = sorted(params.items())
        sign_string = "&".join([f"{k}={v}" for k, v in sorted_params if k != "sign"])
        sign_string += f"&key={self.config.app_secret}"
        return hashlib.md5(sign_string.encode()).hexdigest().upper()
    
    def _dict_to_xml(self, data: Dict[str, Any]) -> str:
        """字典转XML"""
        xml_parts = ["<xml>"]
        for key, value in data.items():
            xml_parts.append(f"<{key}>{value}</{key}>")
        xml_parts.append("</xml>")
        return "".join(xml_parts)
    
    def _xml_to_dict(self, xml_string: str) -> Dict[str, str]:
        """XML转字典"""
        try:
            root = ET.fromstring(xml_string)
            return {child.tag: child.text for child in root}
        except:
            return {}


class SMSIntegration(BaseIntegration):
    """短信集成"""
    
    async def authenticate(self) -> IntegrationResult:
        """认证"""
        return IntegrationResult(success=True)
    
    async def send_sms(
        self,
        phone: str,
        message: str,
        template_id: str = None,
        template_params: Dict[str, str] = None
    ) -> IntegrationResult:
        """发送短信"""
        try:
            if self.config.provider == SMSProvider.ALIYUN.value:
                return await self._send_aliyun_sms(
                    phone, message, template_id, template_params
                )
            elif self.config.provider == SMSProvider.TENCENT.value:
                return await self._send_tencent_sms(
                    phone, message, template_id, template_params
                )
            elif self.config.provider == SMSProvider.TWILIO.value:
                return await self._send_twilio_sms(
                    phone, message, template_id, template_params
                )
            else:
                return IntegrationResult(
                    success=False,
                    error=f"Unsupported SMS provider: {self.config.provider}"
                )
        
        except Exception as e:
            return IntegrationResult(
                success=False,
                error=str(e)
            )
    
    async def _send_aliyun_sms(
        self, phone, message, template_id, template_params
    ) -> IntegrationResult:
        """发送阿里云短信"""
        url = "https://dysmsapi.aliyuncs.com/"
        
        params = {
            "Action": "SendSms",
            "Version": "2017-05-25",
            "RegionId": "cn-hangzhou",
            "PhoneNumbers": phone,
            "SignName": self.config.extra_config.get("sign_name"),
            "TemplateCode": template_id,
            "TemplateParam": json.dumps(template_params or {}),
            "AccessKeyId": self.config.api_key,
            "Timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "SignatureMethod": "HMAC-SHA1",
            "SignatureVersion": "1.0",
            "SignatureNonce": str(uuid.uuid4()),
            "Format": "JSON"
        }
        
        # 生成签名
        signature = self._generate_aliyun_signature(params)
        params["Signature"] = signature
        
        result = await self.make_request("GET", url, params=params)
        
        if result.success and result.data.get("Code") == "OK":
            return IntegrationResult(
                success=True,
                data={
                    "message_id": result.data.get("BizId"),
                    "phone": phone,
                    "status": "sent"
                }
            )
        
        return IntegrationResult(
            success=False,
            error=result.data.get("Message", "Unknown error")
        )
    
    async def _send_tencent_sms(
        self, phone, message, template_id, template_params
    ) -> IntegrationResult:
        """发送腾讯云短信"""
        url = "https://sms.tencentcloudapi.com/"
        
        data = {
            "PhoneNumberSet": [phone],
            "TemplateID": template_id,
            "Sign": self.config.extra_config.get("sign_name"),
            "TemplateParamSet": list(template_params.values()) if template_params else [],
            "SmsSdkAppid": self.config.app_id
        }
        
        headers = {
            "Authorization": self._generate_tencent_auth(data),
            "Content-Type": "application/json; charset=utf-8",
            "Host": "sms.tencentcloudapi.com",
            "X-TC-Action": "SendSms",
            "X-TC-Version": "2019-07-11",
            "X-TC-Region": "ap-guangzhou"
        }
        
        result = await self.make_request("POST", url, headers=headers, data=data)
        
        if result.success:
            response = result.data.get("Response", {})
            if "Error" not in response:
                return IntegrationResult(
                    success=True,
                    data={
                        "message_id": response.get("SendStatusSet", [{}])[0].get("SerialNo"),
                        "phone": phone,
                        "status": "sent"
                    }
                )
        
        return result
    
    async def _send_twilio_sms(
        self, phone, message, template_id, template_params
    ) -> IntegrationResult:
        """发送Twilio短信"""
        url = f"{self.config.base_url}/2010-04-01/Accounts/{self.config.app_id}/Messages.json"
        
        data = {
            "From": self.config.extra_config.get("from_number"),
            "To": phone,
            "Body": message
        }
        
        auth = base64.b64encode(
            f"{self.config.app_id}:{self.config.app_secret}".encode()
        ).decode()
        
        headers = {
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        result = await self.make_request(
            "POST", url, headers=headers, data=urlencode(data)
        )
        
        if result.success:
            return IntegrationResult(
                success=True,
                data={
                    "message_id": result.data.get("sid"),
                    "phone": phone,
                    "status": result.data.get("status")
                }
            )
        
        return result
    
    def _generate_aliyun_signature(self, params: Dict[str, str]) -> str:
        """生成阿里云签名"""
        sorted_params = sorted(params.items())
        query_string = "&".join([f"{quote(k)}={quote(str(v))}" for k, v in sorted_params if k != "Signature"])
        string_to_sign = f"GET&%2F&{quote(query_string)}"
        
        signature = hmac.new(
            f"{self.config.app_secret}&".encode(),
            string_to_sign.encode(),
            hashlib.sha1
        ).digest()
        
        return base64.b64encode(signature).decode()
    
    def _generate_tencent_auth(self, data: Dict[str, Any]) -> str:
        """生成腾讯云认证"""
        # 简化实现，实际需要按照腾讯云API签名规范
        return f"TC3-HMAC-SHA256 Credential={self.config.api_key}"


class EmailIntegration(BaseIntegration):
    """邮件集成"""
    
    async def authenticate(self) -> IntegrationResult:
        """认证"""
        return IntegrationResult(success=True)
    
    async def send_email(
        self,
        to_emails: Union[str, List[str]],
        subject: str,
        content: str,
        content_type: str = "text/html",
        from_email: str = None,
        from_name: str = None,
        attachments: List[str] = None
    ) -> IntegrationResult:
        """发送邮件"""
        try:
            if isinstance(to_emails, str):
                to_emails = [to_emails]
            
            if self.config.provider == EmailProvider.SMTP.value:
                return await self._send_smtp_email(
                    to_emails, subject, content, content_type,
                    from_email, from_name, attachments
                )
            elif self.config.provider == EmailProvider.SENDGRID.value:
                return await self._send_sendgrid_email(
                    to_emails, subject, content, content_type,
                    from_email, from_name, attachments
                )
            else:
                return IntegrationResult(
                    success=False,
                    error=f"Unsupported email provider: {self.config.provider}"
                )
        
        except Exception as e:
            return IntegrationResult(
                success=False,
                error=str(e)
            )
    
    async def _send_smtp_email(
        self, to_emails, subject, content, content_type,
        from_email, from_name, attachments
    ) -> IntegrationResult:
        """发送SMTP邮件"""
        try:
            msg = MIMEMultipart()
            msg['From'] = f"{from_name} <{from_email}>" if from_name else from_email
            msg['To'] = ", ".join(to_emails)
            msg['Subject'] = subject
            
            # 添加邮件内容
            msg.attach(MIMEText(content, content_type.split('/')[-1]))
            
            # 添加附件
            if attachments:
                for file_path in attachments:
                    with open(file_path, "rb") as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                    
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {Path(file_path).name}'
                    )
                    msg.attach(part)
            
            # 发送邮件
            context = ssl.create_default_context()
            with smtplib.SMTP(self.config.base_url, self.config.extra_config.get("port", 587)) as server:
                server.starttls(context=context)
                server.login(self.config.api_key, self.config.app_secret)
                server.sendmail(from_email, to_emails, msg.as_string())
            
            return IntegrationResult(
                success=True,
                data={
                    "to_emails": to_emails,
                    "subject": subject,
                    "status": "sent"
                }
            )
        
        except Exception as e:
            return IntegrationResult(
                success=False,
                error=str(e)
            )
    
    async def _send_sendgrid_email(
        self, to_emails, subject, content, content_type,
        from_email, from_name, attachments
    ) -> IntegrationResult:
        """发送SendGrid邮件"""
        url = f"{self.config.base_url}/v3/mail/send"
        
        data = {
            "personalizations": [{
                "to": [{"email": email} for email in to_emails]
            }],
            "from": {
                "email": from_email,
                "name": from_name
            },
            "subject": subject,
            "content": [{
                "type": content_type,
                "value": content
            }]
        }
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        result = await self.make_request("POST", url, headers=headers, data=data)
        
        if result.success:
            return IntegrationResult(
                success=True,
                data={
                    "to_emails": to_emails,
                    "subject": subject,
                    "status": "sent"
                }
            )
        
        return result


class CloudStorageIntegration(BaseIntegration):
    """云存储集成"""
    
    async def authenticate(self) -> IntegrationResult:
        """认证"""
        return IntegrationResult(success=True)
    
    async def upload_file(
        self,
        file_path: str,
        object_key: str,
        bucket: str = None,
        content_type: str = None
    ) -> IntegrationResult:
        """上传文件"""
        try:
            if not content_type:
                content_type, _ = mimetypes.guess_type(file_path)
                content_type = content_type or "application/octet-stream"
            
            if self.config.provider == CloudProvider.ALIYUN_OSS.value:
                return await self._upload_to_aliyun_oss(
                    file_path, object_key, bucket, content_type
                )
            elif self.config.provider == CloudProvider.AWS_S3.value:
                return await self._upload_to_aws_s3(
                    file_path, object_key, bucket, content_type
                )
            else:
                return IntegrationResult(
                    success=False,
                    error=f"Unsupported cloud provider: {self.config.provider}"
                )
        
        except Exception as e:
            return IntegrationResult(
                success=False,
                error=str(e)
            )
    
    async def _upload_to_aliyun_oss(
        self, file_path, object_key, bucket, content_type
    ) -> IntegrationResult:
        """上传到阿里云OSS"""
        url = f"https://{bucket}.{self.config.base_url}/{object_key}"
        
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        headers = {
            "Content-Type": content_type,
            "Authorization": self._generate_oss_auth("PUT", object_key, content_type)
        }
        
        result = await self.make_request(
            "PUT", url, headers=headers, data=file_data
        )
        
        if result.success:
            return IntegrationResult(
                success=True,
                data={
                    "object_key": object_key,
                    "bucket": bucket,
                    "url": url,
                    "size": len(file_data)
                }
            )
        
        return result
    
    async def _upload_to_aws_s3(
        self, file_path, object_key, bucket, content_type
    ) -> IntegrationResult:
        """上传到AWS S3"""
        url = f"https://{bucket}.s3.amazonaws.com/{object_key}"
        
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        headers = {
            "Content-Type": content_type,
            "Authorization": self._generate_s3_auth("PUT", object_key, content_type)
        }
        
        result = await self.make_request(
            "PUT", url, headers=headers, data=file_data
        )
        
        if result.success:
            return IntegrationResult(
                success=True,
                data={
                    "object_key": object_key,
                    "bucket": bucket,
                    "url": url,
                    "size": len(file_data)
                }
            )
        
        return result
    
    def _generate_oss_auth(self, method: str, object_key: str, content_type: str) -> str:
        """生成OSS认证"""
        # 简化实现，实际需要按照OSS签名规范
        return f"OSS {self.config.api_key}:signature"
    
    def _generate_s3_auth(self, method: str, object_key: str, content_type: str) -> str:
        """生成S3认证"""
        # 简化实现，实际需要按照AWS签名规范
        return f"AWS4-HMAC-SHA256 Credential={self.config.api_key}"


class SocialLoginIntegration(BaseIntegration):
    """社交登录集成"""
    
    async def authenticate(self) -> IntegrationResult:
        """认证"""
        return IntegrationResult(success=True)
    
    def get_auth_url(
        self,
        redirect_uri: str,
        state: str = None,
        scope: str = None
    ) -> IntegrationResult:
        """获取授权URL"""
        try:
            if not state:
                state = str(uuid.uuid4())
            
            if self.config.provider == SocialProvider.WECHAT.value:
                return self._get_wechat_auth_url(redirect_uri, state, scope)
            elif self.config.provider == SocialProvider.GITHUB.value:
                return self._get_github_auth_url(redirect_uri, state, scope)
            elif self.config.provider == SocialProvider.GOOGLE.value:
                return self._get_google_auth_url(redirect_uri, state, scope)
            else:
                return IntegrationResult(
                    success=False,
                    error=f"Unsupported social provider: {self.config.provider}"
                )
        
        except Exception as e:
            return IntegrationResult(
                success=False,
                error=str(e)
            )
    
    def _get_wechat_auth_url(self, redirect_uri, state, scope) -> IntegrationResult:
        """获取微信授权URL"""
        params = {
            "appid": self.config.app_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": scope or "snsapi_userinfo",
            "state": state
        }
        
        auth_url = f"{self.config.base_url}/connect/oauth2/authorize?{urlencode(params)}#wechat_redirect"
        
        return IntegrationResult(
            success=True,
            data={
                "auth_url": auth_url,
                "state": state
            }
        )
    
    def _get_github_auth_url(self, redirect_uri, state, scope) -> IntegrationResult:
        """获取GitHub授权URL"""
        params = {
            "client_id": self.config.app_id,
            "redirect_uri": redirect_uri,
            "scope": scope or "user:email",
            "state": state
        }
        
        auth_url = f"{self.config.base_url}/login/oauth/authorize?{urlencode(params)}"
        
        return IntegrationResult(
            success=True,
            data={
                "auth_url": auth_url,
                "state": state
            }
        )
    
    def _get_google_auth_url(self, redirect_uri, state, scope) -> IntegrationResult:
        """获取Google授权URL"""
        params = {
            "client_id": self.config.app_id,
            "redirect_uri": redirect_uri,
            "scope": scope or "openid email profile",
            "response_type": "code",
            "state": state
        }
        
        auth_url = f"{self.config.base_url}/o/oauth2/v2/auth?{urlencode(params)}"
        
        return IntegrationResult(
            success=True,
            data={
                "auth_url": auth_url,
                "state": state
            }
        )
    
    async def exchange_code_for_token(
        self,
        code: str,
        redirect_uri: str
    ) -> IntegrationResult:
        """用授权码换取访问令牌"""
        try:
            if self.config.provider == SocialProvider.WECHAT.value:
                return await self._exchange_wechat_token(code, redirect_uri)
            elif self.config.provider == SocialProvider.GITHUB.value:
                return await self._exchange_github_token(code, redirect_uri)
            elif self.config.provider == SocialProvider.GOOGLE.value:
                return await self._exchange_google_token(code, redirect_uri)
            else:
                return IntegrationResult(
                    success=False,
                    error=f"Unsupported social provider: {self.config.provider}"
                )
        
        except Exception as e:
            return IntegrationResult(
                success=False,
                error=str(e)
            )
    
    async def _exchange_wechat_token(self, code, redirect_uri) -> IntegrationResult:
        """微信换取令牌"""
        url = f"{self.config.base_url}/sns/oauth2/access_token"
        
        params = {
            "appid": self.config.app_id,
            "secret": self.config.app_secret,
            "code": code,
            "grant_type": "authorization_code"
        }
        
        result = await self.make_request("GET", url, params=params)
        
        if result.success and "access_token" in result.data:
            return IntegrationResult(
                success=True,
                data={
                    "access_token": result.data["access_token"],
                    "refresh_token": result.data.get("refresh_token"),
                    "openid": result.data.get("openid"),
                    "scope": result.data.get("scope"),
                    "expires_in": result.data.get("expires_in")
                }
            )
        
        return result
    
    async def _exchange_github_token(self, code, redirect_uri) -> IntegrationResult:
        """GitHub换取令牌"""
        url = f"{self.config.base_url}/login/oauth/access_token"
        
        data = {
            "client_id": self.config.app_id,
            "client_secret": self.config.app_secret,
            "code": code,
            "redirect_uri": redirect_uri
        }
        
        headers = {"Accept": "application/json"}
        
        result = await self.make_request("POST", url, headers=headers, data=data)
        
        if result.success and "access_token" in result.data:
            return IntegrationResult(
                success=True,
                data={
                    "access_token": result.data["access_token"],
                    "token_type": result.data.get("token_type"),
                    "scope": result.data.get("scope")
                }
            )
        
        return result
    
    async def _exchange_google_token(self, code, redirect_uri) -> IntegrationResult:
        """Google换取令牌"""
        url = f"{self.config.base_url}/oauth2/v4/token"
        
        data = {
            "client_id": self.config.app_id,
            "client_secret": self.config.app_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri
        }
        
        result = await self.make_request("POST", url, data=data)
        
        if result.success and "access_token" in result.data:
            return IntegrationResult(
                success=True,
                data={
                    "access_token": result.data["access_token"],
                    "refresh_token": result.data.get("refresh_token"),
                    "token_type": result.data.get("token_type"),
                    "expires_in": result.data.get("expires_in"),
                    "scope": result.data.get("scope")
                }
            )
        
        return result


# 集成管理器
class IntegrationManager:
    """集成管理器"""
    
    def __init__(self):
        self.integrations: Dict[str, BaseIntegration] = {}
    
    def register_integration(
        self,
        name: str,
        integration_class: type,
        config: IntegrationConfig
    ):
        """注册集成"""
        self.integrations[name] = integration_class(config)
    
    def get_integration(self, name: str) -> Optional[BaseIntegration]:
        """获取集成"""
        return self.integrations.get(name)
    
    async def test_integration(self, name: str) -> IntegrationResult:
        """测试集成"""
        integration = self.get_integration(name)
        if not integration:
            return IntegrationResult(
                success=False,
                error=f"Integration '{name}' not found"
            )
        
        async with integration:
            return await integration.authenticate()


# 使用示例
if __name__ == "__main__":
    async def example_usage():
        manager = IntegrationManager()
        
        # 注册支付集成
        payment_config = IntegrationConfig(
            provider=PaymentProvider.ALIPAY.value,
            app_id="your_app_id",
            app_secret="your_app_secret",
            base_url="https://openapi.alipay.com/gateway.do"
        )
        manager.register_integration("alipay", PaymentIntegration, payment_config)
        
        # 注册短信集成
        sms_config = IntegrationConfig(
            provider=SMSProvider.ALIYUN.value,
            api_key="your_access_key",
            app_secret="your_access_secret",
            extra_config={"sign_name": "your_sign_name"}
        )
        manager.register_integration("aliyun_sms", SMSIntegration, sms_config)
        
        # 测试集成
        payment_result = await manager.test_integration("alipay")
        print(f"Payment integration test: {payment_result.success}")
        
        sms_result = await manager.test_integration("aliyun_sms")
        print(f"SMS integration test: {sms_result.success}")
        
        # 使用集成
        payment_integration = manager.get_integration("alipay")
        if payment_integration:
            async with payment_integration:
                order_result = await payment_integration.create_order(
                    amount=100.0,
                    description="Test Order"
                )
                print(f"Order created: {order_result.success}")
    
    # 运行示例
    asyncio.run(example_usage())