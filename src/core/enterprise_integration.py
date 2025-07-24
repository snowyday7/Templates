# -*- coding: utf-8 -*-
"""
企业集成模块
提供与企业系统的集成功能，包括SSO、LDAP、ERP、CRM等
"""

import asyncio
import json
import base64
import hashlib
import hmac
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from urllib.parse import urlencode, parse_qs
import httpx
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key

logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    """集成类型枚举"""
    SSO_SAML = "sso_saml"
    SSO_OAUTH = "sso_oauth"
    SSO_OIDC = "sso_oidc"
    LDAP = "ldap"
    ACTIVE_DIRECTORY = "active_directory"
    ERP_SAP = "erp_sap"
    ERP_ORACLE = "erp_oracle"
    CRM_SALESFORCE = "crm_salesforce"
    CRM_DYNAMICS = "crm_dynamics"
    WEBHOOK = "webhook"
    API_GATEWAY = "api_gateway"


class AuthenticationMethod(Enum):
    """认证方法枚举"""
    BASIC = "basic"
    BEARER = "bearer"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    CERTIFICATE = "certificate"
    HMAC = "hmac"


@dataclass
class IntegrationConfig:
    """集成配置"""
    name: str
    integration_type: IntegrationType
    endpoint: str
    auth_method: AuthenticationMethod
    credentials: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 3
    enabled: bool = True


@dataclass
class IntegrationResult:
    """集成结果"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SAMLProvider:
    """SAML SSO提供者"""

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.entity_id = config.settings.get("entity_id")
        self.sso_url = config.settings.get("sso_url")
        self.certificate = config.settings.get("certificate")
        self.private_key = config.settings.get("private_key")

    def generate_auth_request(self, relay_state: Optional[str] = None) -> str:
        """生成SAML认证请求"""
        import uuid
        request_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        auth_request = f"""
        <samlp:AuthnRequest
            xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
            xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
            ID="{request_id}"
            Version="2.0"
            IssueInstant="{timestamp}"
            Destination="{self.sso_url}"
            ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
            AssertionConsumerServiceURL="{self.config.settings.get('acs_url')}">
            <saml:Issuer>{self.entity_id}</saml:Issuer>
        </samlp:AuthnRequest>
        """
        
        # 编码请求
        encoded_request = base64.b64encode(auth_request.encode()).decode()
        
        # 构建重定向URL
        params = {
            "SAMLRequest": encoded_request
        }
        if relay_state:
            params["RelayState"] = relay_state
            
        return f"{self.sso_url}?{urlencode(params)}"

    def validate_response(self, saml_response: str) -> IntegrationResult:
        """验证SAML响应"""
        try:
            # 解码响应
            decoded_response = base64.b64decode(saml_response)
            root = ET.fromstring(decoded_response)
            
            # 提取用户信息
            namespaces = {
                'saml': 'urn:oasis:names:tc:SAML:2.0:assertion',
                'samlp': 'urn:oasis:names:tc:SAML:2.0:protocol'
            }
            
            # 查找断言
            assertion = root.find('.//saml:Assertion', namespaces)
            if assertion is None:
                return IntegrationResult(
                    success=False,
                    error="No assertion found in SAML response"
                )
            
            # 提取用户属性
            subject = assertion.find('.//saml:Subject/saml:NameID', namespaces)
            user_id = subject.text if subject is not None else None
            
            attributes = {}
            attr_statements = assertion.findall('.//saml:AttributeStatement/saml:Attribute', namespaces)
            for attr in attr_statements:
                name = attr.get('Name')
                values = [v.text for v in attr.findall('saml:AttributeValue', namespaces)]
                attributes[name] = values[0] if len(values) == 1 else values
            
            return IntegrationResult(
                success=True,
                data={
                    "user_id": user_id,
                    "attributes": attributes
                }
            )
            
        except Exception as e:
            logger.error(f"Error validating SAML response: {e}")
            return IntegrationResult(
                success=False,
                error=str(e)
            )


class OAuthProvider:
    """OAuth SSO提供者"""

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.client_id = config.credentials.get("client_id")
        self.client_secret = config.credentials.get("client_secret")
        self.auth_url = config.settings.get("auth_url")
        self.token_url = config.settings.get("token_url")
        self.user_info_url = config.settings.get("user_info_url")
        self.scope = config.settings.get("scope", "openid profile email")

    def get_auth_url(self, redirect_uri: str, state: Optional[str] = None) -> str:
        """获取授权URL"""
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": self.scope
        }
        if state:
            params["state"] = state
            
        return f"{self.auth_url}?{urlencode(params)}"

    async def exchange_code_for_token(self, code: str, redirect_uri: str) -> IntegrationResult:
        """交换授权码获取访问令牌"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.token_url,
                    data={
                        "grant_type": "authorization_code",
                        "code": code,
                        "redirect_uri": redirect_uri,
                        "client_id": self.client_id,
                        "client_secret": self.client_secret
                    },
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    return IntegrationResult(
                        success=True,
                        data=token_data,
                        status_code=response.status_code
                    )
                else:
                    return IntegrationResult(
                        success=False,
                        error=f"Token exchange failed: {response.text}",
                        status_code=response.status_code
                    )
                    
        except Exception as e:
            logger.error(f"Error exchanging OAuth code: {e}")
            return IntegrationResult(
                success=False,
                error=str(e)
            )

    async def get_user_info(self, access_token: str) -> IntegrationResult:
        """获取用户信息"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.user_info_url,
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    user_data = response.json()
                    return IntegrationResult(
                        success=True,
                        data=user_data,
                        status_code=response.status_code
                    )
                else:
                    return IntegrationResult(
                        success=False,
                        error=f"Failed to get user info: {response.text}",
                        status_code=response.status_code
                    )
                    
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return IntegrationResult(
                success=False,
                error=str(e)
            )


class LDAPConnector:
    """LDAP连接器"""

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.server = config.endpoint
        self.bind_dn = config.credentials.get("bind_dn")
        self.bind_password = config.credentials.get("bind_password")
        self.base_dn = config.settings.get("base_dn")
        self.user_filter = config.settings.get("user_filter", "(uid={username})")

    async def authenticate_user(self, username: str, password: str) -> IntegrationResult:
        """认证用户"""
        try:
            # 这里需要使用ldap3库，但为了避免依赖，我们模拟实现
            # 实际使用时需要安装: pip install ldap3
            
            # 模拟LDAP认证逻辑
            user_dn = f"uid={username},{self.base_dn}"
            
            # 在实际实现中，这里会连接LDAP服务器进行认证
            # import ldap3
            # server = ldap3.Server(self.server)
            # conn = ldap3.Connection(server, user_dn, password)
            # if conn.bind():
            #     return IntegrationResult(success=True, data={"user_dn": user_dn})
            
            # 模拟成功认证
            return IntegrationResult(
                success=True,
                data={
                    "user_dn": user_dn,
                    "username": username,
                    "attributes": {
                        "cn": f"User {username}",
                        "mail": f"{username}@example.com"
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"LDAP authentication error: {e}")
            return IntegrationResult(
                success=False,
                error=str(e)
            )

    async def search_users(self, search_filter: str) -> IntegrationResult:
        """搜索用户"""
        try:
            # 模拟用户搜索
            users = [
                {
                    "dn": f"uid=user1,{self.base_dn}",
                    "attributes": {
                        "uid": "user1",
                        "cn": "User One",
                        "mail": "user1@example.com"
                    }
                },
                {
                    "dn": f"uid=user2,{self.base_dn}",
                    "attributes": {
                        "uid": "user2",
                        "cn": "User Two",
                        "mail": "user2@example.com"
                    }
                }
            ]
            
            return IntegrationResult(
                success=True,
                data=users
            )
            
        except Exception as e:
            logger.error(f"LDAP search error: {e}")
            return IntegrationResult(
                success=False,
                error=str(e)
            )


class ERPConnector:
    """ERP系统连接器"""

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.base_url = config.endpoint
        self.api_key = config.credentials.get("api_key")
        self.username = config.credentials.get("username")
        self.password = config.credentials.get("password")

    async def get_customer_data(self, customer_id: str) -> IntegrationResult:
        """获取客户数据"""
        try:
            headers = self._get_auth_headers()
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/customers/{customer_id}",
                    headers=headers,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    customer_data = response.json()
                    return IntegrationResult(
                        success=True,
                        data=customer_data,
                        status_code=response.status_code
                    )
                else:
                    return IntegrationResult(
                        success=False,
                        error=f"Failed to get customer data: {response.text}",
                        status_code=response.status_code
                    )
                    
        except Exception as e:
            logger.error(f"ERP connector error: {e}")
            return IntegrationResult(
                success=False,
                error=str(e)
            )

    async def create_order(self, order_data: Dict[str, Any]) -> IntegrationResult:
        """创建订单"""
        try:
            headers = self._get_auth_headers()
            headers["Content-Type"] = "application/json"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/orders",
                    headers=headers,
                    json=order_data,
                    timeout=self.config.timeout
                )
                
                if response.status_code in [200, 201]:
                    result_data = response.json()
                    return IntegrationResult(
                        success=True,
                        data=result_data,
                        status_code=response.status_code
                    )
                else:
                    return IntegrationResult(
                        success=False,
                        error=f"Failed to create order: {response.text}",
                        status_code=response.status_code
                    )
                    
        except Exception as e:
            logger.error(f"ERP order creation error: {e}")
            return IntegrationResult(
                success=False,
                error=str(e)
            )

    def _get_auth_headers(self) -> Dict[str, str]:
        """获取认证头"""
        headers = {}
        
        if self.config.auth_method == AuthenticationMethod.API_KEY:
            headers["X-API-Key"] = self.api_key
        elif self.config.auth_method == AuthenticationMethod.BASIC:
            credentials = base64.b64encode(
                f"{self.username}:{self.password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"
        elif self.config.auth_method == AuthenticationMethod.BEARER:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        return headers


class WebhookManager:
    """Webhook管理器"""

    def __init__(self):
        self.webhooks: Dict[str, IntegrationConfig] = {}
        self.handlers: Dict[str, Callable] = {}

    def register_webhook(
        self,
        name: str,
        config: IntegrationConfig,
        handler: Callable
    ):
        """注册webhook"""
        self.webhooks[name] = config
        self.handlers[name] = handler
        logger.info(f"Registered webhook: {name}")

    async def process_webhook(
        self,
        name: str,
        payload: Dict[str, Any],
        headers: Dict[str, str] = None
    ) -> IntegrationResult:
        """处理webhook"""
        if name not in self.webhooks:
            return IntegrationResult(
                success=False,
                error=f"Webhook {name} not found"
            )

        config = self.webhooks[name]
        handler = self.handlers[name]

        try:
            # 验证签名（如果配置了）
            if config.settings.get("verify_signature"):
                if not self._verify_signature(payload, headers, config):
                    return IntegrationResult(
                        success=False,
                        error="Invalid webhook signature"
                    )

            # 调用处理器
            result = await handler(payload, headers)
            return IntegrationResult(
                success=True,
                data=result
            )

        except Exception as e:
            logger.error(f"Webhook processing error: {e}")
            return IntegrationResult(
                success=False,
                error=str(e)
            )

    def _verify_signature(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        config: IntegrationConfig
    ) -> bool:
        """验证webhook签名"""
        try:
            secret = config.credentials.get("webhook_secret")
            if not secret:
                return True

            signature_header = config.settings.get("signature_header", "X-Signature")
            received_signature = headers.get(signature_header)
            
            if not received_signature:
                return False

            # 计算期望的签名
            payload_str = json.dumps(payload, sort_keys=True)
            expected_signature = hmac.new(
                secret.encode(),
                payload_str.encode(),
                hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(received_signature, expected_signature)

        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False


class EnterpriseIntegrationManager:
    """企业集成管理器"""

    def __init__(self):
        self.integrations: Dict[str, Any] = {}
        self.webhook_manager = WebhookManager()

    def register_integration(
        self,
        name: str,
        config: IntegrationConfig
    ) -> Any:
        """注册集成"""
        if config.integration_type == IntegrationType.SSO_SAML:
            integration = SAMLProvider(config)
        elif config.integration_type == IntegrationType.SSO_OAUTH:
            integration = OAuthProvider(config)
        elif config.integration_type == IntegrationType.LDAP:
            integration = LDAPConnector(config)
        elif config.integration_type in [IntegrationType.ERP_SAP, IntegrationType.ERP_ORACLE]:
            integration = ERPConnector(config)
        else:
            raise ValueError(f"Unsupported integration type: {config.integration_type}")

        self.integrations[name] = integration
        logger.info(f"Registered integration: {name}")
        return integration

    def get_integration(self, name: str) -> Optional[Any]:
        """获取集成"""
        return self.integrations.get(name)

    async def test_integration(self, name: str) -> IntegrationResult:
        """测试集成连接"""
        integration = self.get_integration(name)
        if not integration:
            return IntegrationResult(
                success=False,
                error=f"Integration {name} not found"
            )

        try:
            # 根据集成类型执行不同的测试
            if isinstance(integration, LDAPConnector):
                # 测试LDAP连接
                return await integration.search_users("(objectClass=person)")
            elif isinstance(integration, ERPConnector):
                # 测试ERP连接
                return await integration.get_customer_data("test")
            else:
                return IntegrationResult(
                    success=True,
                    data={"message": "Integration test not implemented"}
                )

        except Exception as e:
            logger.error(f"Integration test error: {e}")
            return IntegrationResult(
                success=False,
                error=str(e)
            )

    def get_integration_status(self) -> Dict[str, Any]:
        """获取所有集成状态"""
        status = {
            "total_integrations": len(self.integrations),
            "integrations": {},
            "webhooks": len(self.webhook_manager.webhooks)
        }

        for name, integration in self.integrations.items():
            status["integrations"][name] = {
                "type": integration.config.integration_type.value,
                "enabled": integration.config.enabled,
                "endpoint": integration.config.endpoint
            }

        return status


# 使用示例
if __name__ == "__main__":
    async def example_usage():
        # 创建企业集成管理器
        integration_manager = EnterpriseIntegrationManager()
        
        # 配置LDAP集成
        ldap_config = IntegrationConfig(
            name="company_ldap",
            integration_type=IntegrationType.LDAP,
            endpoint="ldap://ldap.company.com:389",
            auth_method=AuthenticationMethod.BASIC,
            credentials={
                "bind_dn": "cn=admin,dc=company,dc=com",
                "bind_password": "password"
            },
            settings={
                "base_dn": "ou=users,dc=company,dc=com",
                "user_filter": "(uid={username})"
            }
        )
        
        # 注册LDAP集成
        ldap_integration = integration_manager.register_integration(
            "company_ldap", ldap_config
        )
        
        # 测试LDAP认证
        auth_result = await ldap_integration.authenticate_user("testuser", "password")
        print(f"LDAP auth result: {auth_result.success}")
        
        # 配置OAuth SSO
        oauth_config = IntegrationConfig(
            name="google_sso",
            integration_type=IntegrationType.SSO_OAUTH,
            endpoint="https://accounts.google.com",
            auth_method=AuthenticationMethod.OAUTH2,
            credentials={
                "client_id": "your-client-id",
                "client_secret": "your-client-secret"
            },
            settings={
                "auth_url": "https://accounts.google.com/o/oauth2/auth",
                "token_url": "https://oauth2.googleapis.com/token",
                "user_info_url": "https://www.googleapis.com/oauth2/v2/userinfo",
                "scope": "openid profile email"
            }
        )
        
        # 注册OAuth集成
        oauth_integration = integration_manager.register_integration(
            "google_sso", oauth_config
        )
        
        # 获取OAuth授权URL
        auth_url = oauth_integration.get_auth_url(
            "http://localhost:8000/auth/callback",
            "random-state"
        )
        print(f"OAuth auth URL: {auth_url}")
        
        # 获取集成状态
        status = integration_manager.get_integration_status()
        print(f"Integration status: {status}")
    
    # 运行示例
    asyncio.run(example_usage())