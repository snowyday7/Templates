"""企业集成模块

提供完整的企业集成功能，包括：
- API网关集成
- 消息队列集成
- 数据库集成
- 第三方服务集成
- 企业服务总线(ESB)
- 工作流引擎
- 数据同步
- 身份认证集成
"""

from .api_gateway import (
    APIGateway,
    GatewayConfig,
    Route,
    RouteConfig,
    Middleware,
    RateLimiter,
    AuthenticationMiddleware,
    LoggingMiddleware,
    initialize_api_gateway,
    get_api_gateway
)

from .message_broker import (
    MessageBroker,
    BrokerConfig,
    Message,
    MessageHandler,
    Topic,
    Queue,
    Publisher,
    Subscriber,
    initialize_message_broker,
    get_message_broker
)

from .database_integration import (
    DatabaseIntegration,
    DatabaseConfig,
    ConnectionPool,
    QueryBuilder,
    DataMapper,
    TransactionManager,
    initialize_database_integration,
    get_database_integration
)

from .third_party_services import (
    ServiceIntegration,
    ServiceConfig,
    ServiceClient,
    ServiceRegistry,
    ServiceDiscovery,
    CircuitBreaker,
    RetryPolicy,
    initialize_service_integration,
    get_service_integration
)

from .esb import (
    EnterpriseServiceBus,
    ESBConfig,
    ServiceEndpoint,
    MessageRouter,
    DataTransformer,
    ProtocolAdapter,
    initialize_esb,
    get_esb
)

from .workflow_engine import (
    WorkflowEngine,
    WorkflowConfig,
    Workflow,
    WorkflowStep,
    WorkflowInstance,
    TaskExecutor,
    WorkflowScheduler,
    initialize_workflow_engine,
    get_workflow_engine
)

from .data_sync import (
    DataSynchronizer,
    SyncConfig,
    SyncJob,
    DataSource,
    DataTarget,
    SyncStrategy,
    ConflictResolver,
    initialize_data_sync,
    get_data_sync
)

from .identity_integration import (
    IdentityProvider,
    IdentityConfig,
    UserProvider,
    GroupProvider,
    RoleProvider,
    SAMLProvider,
    OAuthProvider,
    LDAPProvider,
    initialize_identity_integration,
    get_identity_integration
)

__all__ = [
    # API Gateway
    "APIGateway",
    "GatewayConfig",
    "Route",
    "RouteConfig",
    "Middleware",
    "RateLimiter",
    "AuthenticationMiddleware",
    "LoggingMiddleware",
    "initialize_api_gateway",
    "get_api_gateway",
    
    # Message Broker
    "MessageBroker",
    "BrokerConfig",
    "Message",
    "MessageHandler",
    "Topic",
    "Queue",
    "Publisher",
    "Subscriber",
    "initialize_message_broker",
    "get_message_broker",
    
    # Database Integration
    "DatabaseIntegration",
    "DatabaseConfig",
    "ConnectionPool",
    "QueryBuilder",
    "DataMapper",
    "TransactionManager",
    "initialize_database_integration",
    "get_database_integration",
    
    # Third Party Services
    "ServiceIntegration",
    "ServiceConfig",
    "ServiceClient",
    "ServiceRegistry",
    "ServiceDiscovery",
    "CircuitBreaker",
    "RetryPolicy",
    "initialize_service_integration",
    "get_service_integration",
    
    # Enterprise Service Bus
    "EnterpriseServiceBus",
    "ESBConfig",
    "ServiceEndpoint",
    "MessageRouter",
    "DataTransformer",
    "ProtocolAdapter",
    "initialize_esb",
    "get_esb",
    
    # Workflow Engine
    "WorkflowEngine",
    "WorkflowConfig",
    "Workflow",
    "WorkflowStep",
    "WorkflowInstance",
    "TaskExecutor",
    "WorkflowScheduler",
    "initialize_workflow_engine",
    "get_workflow_engine",
    
    # Data Sync
    "DataSynchronizer",
    "SyncConfig",
    "SyncJob",
    "DataSource",
    "DataTarget",
    "SyncStrategy",
    "ConflictResolver",
    "initialize_data_sync",
    "get_data_sync",
    
    # Identity Integration
    "IdentityProvider",
    "IdentityConfig",
    "UserProvider",
    "GroupProvider",
    "RoleProvider",
    "SAMLProvider",
    "OAuthProvider",
    "LDAPProvider",
    "initialize_identity_integration",
    "get_identity_integration",
]