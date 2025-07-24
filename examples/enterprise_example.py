"""企业级功能综合示例

展示如何使用所有新增的企业级功能：
- 安全管理（API密钥、RBAC）
- 性能优化（连接池）
- 可观测性（追踪、指标、告警）
- 数据治理（数据质量）
- 高可用性（负载均衡、故障转移）
- 企业集成（API网关）
"""

import asyncio
import time
from typing import Dict, Any, List
from datetime import datetime

# 导入企业级模块
from templates.security import (
    APIKeyManager, APIKeyConfig,
    RBACManager, RBACConfig,
    initialize_security
)
from templates.performance import (
    ConnectionPoolManager, PoolConfig,
    initialize_performance
)
from templates.observability import (
    TracingManager, TraceConfig,
    MetricsManager, MetricsConfig,
    AlertManager, AlertingConfig,
    initialize_observability
)
from templates.data_governance import (
    DataQualityManager, DataQualityConfig,
    initialize_data_governance
)
from templates.high_availability import (
    LoadBalancerManager, LoadBalancerConfig,
    FailoverManager, FailoverConfig,
    ServerInstance, ServiceInstance,
    initialize_load_balancer, initialize_failover
)
from templates.enterprise_integration import (
    APIGateway, GatewayConfig,
    RouteConfig, Backend,
    initialize_api_gateway
)


class EnterpriseApplication:
    """企业级应用示例"""
    
    def __init__(self):
        self.security_manager = None
        self.performance_manager = None
        self.observability_manager = None
        self.data_governance_manager = None
        self.load_balancer_manager = None
        self.failover_manager = None
        self.api_gateway = None
        
        # 初始化所有组件
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化所有企业级组件"""
        print("Initializing enterprise components...")
        
        # 1. 初始化安全管理
        self._initialize_security()
        
        # 2. 初始化性能优化
        self._initialize_performance()
        
        # 3. 初始化可观测性
        self._initialize_observability()
        
        # 4. 初始化数据治理
        self._initialize_data_governance()
        
        # 5. 初始化高可用性
        self._initialize_high_availability()
        
        # 6. 初始化企业集成
        self._initialize_enterprise_integration()
        
        print("All enterprise components initialized successfully!")
    
    def _initialize_security(self):
        """初始化安全管理"""
        print("  - Initializing security management...")
        
        # API密钥管理配置
        api_key_config = APIKeyConfig(
            default_expiry_days=30,
            max_keys_per_user=10,
            enable_usage_tracking=True,
            enable_rate_limiting=True
        )
        
        # RBAC配置
        rbac_config = RBACConfig(
            enable_inheritance=True,
            enable_dynamic_permissions=True,
            cache_permissions=True,
            permission_cache_ttl=300
        )
        
        # 初始化安全管理器
        self.security_manager = initialize_security({
            'api_key': api_key_config,
            'rbac': rbac_config
        })
        
        # 创建示例角色和权限
        self._setup_security_demo()
    
    def _setup_security_demo(self):
        """设置安全管理演示数据"""
        if not self.security_manager:
            return
        
        # 创建权限
        permissions = [
            "user:read", "user:write", "user:delete",
            "admin:read", "admin:write", "admin:delete",
            "system:monitor", "system:configure"
        ]
        
        for perm in permissions:
            self.security_manager['rbac'].create_permission(
                perm, f"Permission to {perm.replace(':', ' ')}"
            )
        
        # 创建角色
        roles = {
            "user": ["user:read"],
            "moderator": ["user:read", "user:write"],
            "admin": ["user:read", "user:write", "user:delete", "admin:read", "admin:write"],
            "super_admin": permissions  # 所有权限
        }
        
        for role_name, role_perms in roles.items():
            role = self.security_manager['rbac'].create_role(role_name, f"{role_name.title()} role")
            for perm in role_perms:
                self.security_manager['rbac'].assign_permission_to_role(role.id, perm)
        
        # 创建示例用户
        users = [
            ("user1", "user"),
            ("user2", "moderator"),
            ("admin1", "admin"),
            ("superadmin", "super_admin")
        ]
        
        for user_id, role_name in users:
            role = self.security_manager['rbac'].get_role_by_name(role_name)
            if role:
                self.security_manager['rbac'].assign_role_to_user(user_id, role.id)
        
        # 创建API密钥
        for user_id, _ in users:
            api_key = self.security_manager['api_key'].create_api_key(
                user_id=user_id,
                name=f"{user_id}_key",
                scopes=["api:read", "api:write"]
            )
            print(f"    Created API key for {user_id}: {api_key.key[:20]}...")
    
    def _initialize_performance(self):
        """初始化性能优化"""
        print("  - Initializing performance optimization...")
        
        # 连接池配置
        pool_config = PoolConfig(
            max_connections=100,
            min_connections=10,
            connection_timeout=30,
            idle_timeout=300,
            max_lifetime=3600
        )
        
        # 初始化性能管理器
        self.performance_manager = initialize_performance(pool_config)
        
        # 创建示例连接池
        self._setup_performance_demo()
    
    def _setup_performance_demo(self):
        """设置性能优化演示"""
        if not self.performance_manager:
            return
        
        # 创建数据库连接池
        db_pool = self.performance_manager.create_database_pool(
            "main_db",
            "postgresql://user:pass@localhost:5432/mydb"
        )
        
        # 创建Redis连接池
        redis_pool = self.performance_manager.create_redis_pool(
            "main_cache",
            "redis://localhost:6379/0"
        )
        
        print(f"    Created database pool: {db_pool.pool_id}")
        print(f"    Created Redis pool: {redis_pool.pool_id}")
    
    def _initialize_observability(self):
        """初始化可观测性"""
        print("  - Initializing observability...")
        
        # 追踪配置
        trace_config = TraceConfig(
            service_name="enterprise_app",
            service_version="1.0.0",
            environment="production",
            sampling_rate=1.0
        )
        
        # 指标配置
        metrics_config = MetricsConfig(
            namespace="enterprise_app",
            enable_default_metrics=True,
            export_interval=60
        )
        
        # 告警配置
        alerting_config = AlertingConfig(
            evaluation_interval=60,
            notification_timeout=30,
            max_alerts=1000
        )
        
        # 初始化可观测性管理器
        self.observability_manager = initialize_observability({
            'tracing': trace_config,
            'metrics': metrics_config,
            'alerting': alerting_config
        })
        
        # 设置示例指标和告警
        self._setup_observability_demo()
    
    def _setup_observability_demo(self):
        """设置可观测性演示"""
        if not self.observability_manager:
            return
        
        # 创建自定义指标
        metrics_manager = self.observability_manager['metrics']
        
        # 创建计数器指标
        request_counter = metrics_manager.create_counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"]
        )
        
        # 创建直方图指标
        response_time_histogram = metrics_manager.create_histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"]
        )
        
        # 创建仪表盘指标
        active_connections_gauge = metrics_manager.create_gauge(
            "active_connections",
            "Number of active connections",
            ["pool_name"]
        )
        
        print("    Created custom metrics: request_counter, response_time_histogram, active_connections_gauge")
        
        # 创建告警规则
        alert_manager = self.observability_manager['alerting']
        
        # 高错误率告警
        error_rate_rule = alert_manager.create_alert_rule(
            "high_error_rate",
            "High Error Rate",
            "rate(http_requests_total{status=~'5..'}[5m]) > 0.1",
            "critical",
            "Error rate is above 10%"
        )
        
        # 高响应时间告警
        response_time_rule = alert_manager.create_alert_rule(
            "high_response_time",
            "High Response Time",
            "histogram_quantile(0.95, http_request_duration_seconds) > 2.0",
            "warning",
            "95th percentile response time is above 2 seconds"
        )
        
        print("    Created alert rules: high_error_rate, high_response_time")
    
    def _initialize_data_governance(self):
        """初始化数据治理"""
        print("  - Initializing data governance...")
        
        # 数据质量配置
        data_quality_config = DataQualityConfig(
            enable_auto_profiling=True,
            profiling_sample_size=10000,
            enable_anomaly_detection=True,
            store_results=True
        )
        
        # 初始化数据治理管理器
        self.data_governance_manager = initialize_data_governance({
            'data_quality': data_quality_config
        })
        
        # 设置数据质量规则
        self._setup_data_governance_demo()
    
    def _setup_data_governance_demo(self):
        """设置数据治理演示"""
        if not self.data_governance_manager:
            return
        
        data_quality_manager = self.data_governance_manager['data_quality']
        
        # 创建数据质量规则
        rules = [
            {
                'name': 'user_email_completeness',
                'type': 'completeness',
                'column': 'email',
                'threshold': 0.95,
                'description': 'User email should be 95% complete'
            },
            {
                'name': 'user_id_uniqueness',
                'type': 'uniqueness',
                'column': 'user_id',
                'threshold': 1.0,
                'description': 'User ID should be 100% unique'
            },
            {
                'name': 'email_validity',
                'type': 'validity',
                'column': 'email',
                'pattern': r'^[\w\.-]+@[\w\.-]+\.\w+$',
                'threshold': 0.98,
                'description': 'Email should be valid format'
            }
        ]
        
        for rule_config in rules:
            rule = data_quality_manager.create_rule(
                rule_config['name'],
                rule_config['type'],
                rule_config['column'],
                rule_config['threshold'],
                rule_config['description']
            )
            print(f"    Created data quality rule: {rule.name}")
    
    def _initialize_high_availability(self):
        """初始化高可用性"""
        print("  - Initializing high availability...")
        
        # 负载均衡配置
        lb_config = LoadBalancerConfig(
            strategy="round_robin",
            health_check_enabled=True,
            health_check_interval=30
        )
        
        # 故障转移配置
        failover_config = FailoverConfig(
            strategy="priority_based",
            health_check_interval=30,
            failure_threshold=3,
            recovery_threshold=2
        )
        
        # 初始化高可用性管理器
        self.load_balancer_manager = initialize_load_balancer(lb_config)
        self.failover_manager = initialize_failover(failover_config)
        
        # 设置高可用性演示
        self._setup_high_availability_demo()
    
    def _setup_high_availability_demo(self):
        """设置高可用性演示"""
        if not self.load_balancer_manager or not self.failover_manager:
            return
        
        # 创建服务器实例
        servers = [
            ServerInstance(
                id="server1",
                host="192.168.1.10",
                port=8080,
                weight=1
            ),
            ServerInstance(
                id="server2",
                host="192.168.1.11",
                port=8080,
                weight=1
            ),
            ServerInstance(
                id="server3",
                host="192.168.1.12",
                port=8080,
                weight=2  # 更高权重
            )
        ]
        
        # 创建负载均衡器
        balancer = self.load_balancer_manager.create_balancer(
            "web_servers",
            servers
        )
        
        print(f"    Created load balancer with {len(servers)} servers")
        
        # 创建服务实例（用于故障转移）
        services = [
            ServiceInstance(
                id="service1",
                name="Web Service 1",
                host="192.168.1.10",
                port=8080,
                priority=1,
                health_check_url="http://192.168.1.10:8080/health"
            ),
            ServiceInstance(
                id="service2",
                name="Web Service 2",
                host="192.168.1.11",
                port=8080,
                priority=2,
                health_check_url="http://192.168.1.11:8080/health"
            )
        ]
        
        # 添加服务到故障转移管理器
        for service in services:
            self.failover_manager.add_service(service)
        
        # 设置主服务为活跃状态
        services[0].status = "active"
        
        print(f"    Added {len(services)} services to failover manager")
    
    def _initialize_enterprise_integration(self):
        """初始化企业集成"""
        print("  - Initializing enterprise integration...")
        
        # API网关配置
        gateway_config = GatewayConfig(
            host="0.0.0.0",
            port=8080,
            health_check_enabled=True,
            metrics_enabled=True
        )
        
        # 初始化API网关
        self.api_gateway = initialize_api_gateway(gateway_config)
        
        # 设置API网关路由
        self._setup_api_gateway_demo()
    
    def _setup_api_gateway_demo(self):
        """设置API网关演示"""
        if not self.api_gateway:
            return
        
        # 创建后端服务
        backends = [
            Backend(
                id="api_server1",
                host="192.168.1.20",
                port=8000,
                weight=1
            ),
            Backend(
                id="api_server2",
                host="192.168.1.21",
                port=8000,
                weight=1
            )
        ]
        
        # 创建路由配置
        routes = [
            RouteConfig(
                path="/api/v1/users",
                methods=["GET", "POST"],
                backends=backends,
                authentication="api_key",
                rate_limit_enabled=True,
                rate_limit_requests=100,
                rate_limit_window=60
            ),
            RouteConfig(
                path="/api/v1/admin",
                methods=["GET", "POST", "PUT", "DELETE"],
                backends=backends,
                authentication="jwt",
                rate_limit_enabled=True,
                rate_limit_requests=50,
                rate_limit_window=60
            ),
            RouteConfig(
                path="/health",
                methods=["GET"],
                backends=backends,
                authentication="none"
            )
        ]
        
        # 添加路由到网关
        for route_config in routes:
            route = self.api_gateway.add_route(route_config)
            print(f"    Added route: {route_config.path}")
    
    async def demonstrate_features(self):
        """演示企业级功能"""
        print("\n=== Demonstrating Enterprise Features ===")
        
        # 1. 演示安全功能
        await self._demo_security()
        
        # 2. 演示性能优化
        await self._demo_performance()
        
        # 3. 演示可观测性
        await self._demo_observability()
        
        # 4. 演示数据治理
        await self._demo_data_governance()
        
        # 5. 演示高可用性
        await self._demo_high_availability()
        
        # 6. 演示企业集成
        await self._demo_enterprise_integration()
    
    async def _demo_security(self):
        """演示安全功能"""
        print("\n--- Security Demo ---")
        
        if not self.security_manager:
            print("Security manager not initialized")
            return
        
        # 演示API密钥验证
        api_key_manager = self.security_manager['api_key']
        rbac_manager = self.security_manager['rbac']
        
        # 获取用户的API密钥
        user_keys = api_key_manager.get_user_keys("user1")
        if user_keys:
            api_key = user_keys[0]
            print(f"Validating API key for user1: {api_key.key[:20]}...")
            
            # 验证API密钥
            is_valid = api_key_manager.validate_key(api_key.key)
            print(f"API key validation result: {is_valid}")
            
            # 记录使用情况
            api_key_manager.record_usage(api_key.key, "/api/v1/users", "GET")
            print("API key usage recorded")
        
        # 演示RBAC权限检查
        print("\nChecking user permissions:")
        users = ["user1", "admin1", "superadmin"]
        permissions = ["user:read", "user:write", "admin:delete"]
        
        for user in users:
            print(f"  {user}:")
            for perm in permissions:
                has_permission = rbac_manager.check_permission(user, perm)
                print(f"    {perm}: {'✓' if has_permission else '✗'}")
    
    async def _demo_performance(self):
        """演示性能优化"""
        print("\n--- Performance Demo ---")
        
        if not self.performance_manager:
            print("Performance manager not initialized")
            return
        
        # 获取连接池统计
        stats = self.performance_manager.get_global_stats()
        print(f"Connection pools: {stats['total_pools']}")
        
        for pool_name, pool_stats in stats['pools'].items():
            print(f"  {pool_name}:")
            print(f"    Active connections: {pool_stats['active_connections']}")
            print(f"    Total connections: {pool_stats['total_connections']}")
            print(f"    Pool utilization: {pool_stats['utilization']:.2%}")
    
    async def _demo_observability(self):
        """演示可观测性"""
        print("\n--- Observability Demo ---")
        
        if not self.observability_manager:
            print("Observability manager not initialized")
            return
        
        # 创建追踪
        tracing_manager = self.observability_manager['tracing']
        metrics_manager = self.observability_manager['metrics']
        
        # 开始一个追踪
        with tracing_manager.start_trace("demo_operation") as trace:
            trace.set_attribute("user_id", "user1")
            trace.set_attribute("operation", "data_processing")
            
            # 模拟一些工作
            await asyncio.sleep(0.1)
            
            # 记录指标
            metrics_manager.increment_counter(
                "http_requests_total",
                {"method": "GET", "endpoint": "/api/v1/users", "status": "200"}
            )
            
            metrics_manager.record_histogram(
                "http_request_duration_seconds",
                0.1,
                {"method": "GET", "endpoint": "/api/v1/users"}
            )
            
            trace.add_event("Processing completed")
        
        print("Created trace and recorded metrics")
        
        # 获取指标统计
        metrics_stats = metrics_manager.get_stats()
        print(f"Total metrics: {metrics_stats['total_metrics']}")
        print(f"Total samples: {metrics_stats['total_samples']}")
    
    async def _demo_data_governance(self):
        """演示数据治理"""
        print("\n--- Data Governance Demo ---")
        
        if not self.data_governance_manager:
            print("Data governance manager not initialized")
            return
        
        data_quality_manager = self.data_governance_manager['data_quality']
        
        # 模拟数据质量检查
        sample_data = {
            'user_id': [1, 2, 3, 4, 5],
            'email': ['user1@example.com', 'user2@example.com', None, 'invalid-email', 'user5@example.com'],
            'age': [25, 30, 35, 40, 45]
        }
        
        print("Running data quality checks on sample data...")
        
        # 执行数据质量检查
        results = data_quality_manager.run_checks('users_table', sample_data)
        
        print(f"Data quality check results:")
        for result in results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {result.rule_name}: {status} (Score: {result.score:.2f})")
            if result.issues:
                for issue in result.issues[:3]:  # 显示前3个问题
                    print(f"    - {issue.description}")
    
    async def _demo_high_availability(self):
        """演示高可用性"""
        print("\n--- High Availability Demo ---")
        
        if not self.load_balancer_manager or not self.failover_manager:
            print("High availability managers not initialized")
            return
        
        # 负载均衡演示
        balancer = self.load_balancer_manager.get_balancer("web_servers")
        if balancer:
            print("Load balancer server selection:")
            for i in range(5):
                server = balancer.select_server({'client_ip': f'192.168.1.{100+i}'})
                if server:
                    print(f"  Request {i+1}: {server.host}:{server.port}")
        
        # 故障转移演示
        print("\nFailover manager status:")
        failover_stats = self.failover_manager.get_stats()
        print(f"  Total services: {failover_stats['total_services']}")
        print(f"  Active services: {failover_stats['active_services']}")
        print(f"  Standby services: {failover_stats['standby_services']}")
        print(f"  Failed services: {failover_stats['failed_services']}")
    
    async def _demo_enterprise_integration(self):
        """演示企业集成"""
        print("\n--- Enterprise Integration Demo ---")
        
        if not self.api_gateway:
            print("API gateway not initialized")
            return
        
        # 模拟API请求
        sample_requests = [
            {
                'method': 'GET',
                'path': '/api/v1/users',
                'headers': {'X-API-Key': 'demo-api-key'},
                'client_ip': '192.168.1.100'
            },
            {
                'method': 'POST',
                'path': '/api/v1/admin',
                'headers': {'Authorization': 'Bearer demo-jwt-token'},
                'client_ip': '192.168.1.101'
            },
            {
                'method': 'GET',
                'path': '/health',
                'headers': {},
                'client_ip': '192.168.1.102'
            }
        ]
        
        print("Processing sample API requests:")
        for i, request in enumerate(sample_requests):
            print(f"  Request {i+1}: {request['method']} {request['path']}")
            try:
                response = await self.api_gateway.handle_request(request)
                print(f"    Response: {response['status_code']}")
            except Exception as e:
                print(f"    Error: {str(e)}")
        
        # 获取网关统计
        gateway_stats = self.api_gateway.get_stats()
        print(f"\nAPI Gateway Statistics:")
        print(f"  Total requests: {gateway_stats['total_requests']}")
        print(f"  Success rate: {gateway_stats['success_rate']:.2%}")
        print(f"  Average response time: {gateway_stats['average_response_time']:.3f}s")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # 安全组件状态
        if self.security_manager:
            api_key_stats = self.security_manager['api_key'].get_stats()
            rbac_stats = self.security_manager['rbac'].get_stats()
            status['components']['security'] = {
                'api_keys': api_key_stats,
                'rbac': rbac_stats
            }
        
        # 性能组件状态
        if self.performance_manager:
            perf_stats = self.performance_manager.get_global_stats()
            status['components']['performance'] = perf_stats
        
        # 可观测性组件状态
        if self.observability_manager:
            metrics_stats = self.observability_manager['metrics'].get_stats()
            status['components']['observability'] = {
                'metrics': metrics_stats
            }
        
        # 数据治理组件状态
        if self.data_governance_manager:
            dq_stats = self.data_governance_manager['data_quality'].get_stats()
            status['components']['data_governance'] = {
                'data_quality': dq_stats
            }
        
        # 高可用性组件状态
        if self.load_balancer_manager and self.failover_manager:
            lb_stats = self.load_balancer_manager.get_global_stats()
            fo_stats = self.failover_manager.get_stats()
            status['components']['high_availability'] = {
                'load_balancer': lb_stats,
                'failover': fo_stats
            }
        
        # 企业集成组件状态
        if self.api_gateway:
            gateway_stats = self.api_gateway.get_stats()
            status['components']['enterprise_integration'] = {
                'api_gateway': gateway_stats
            }
        
        return status


async def main():
    """主函数"""
    print("=== Enterprise Application Demo ===")
    
    # 创建企业级应用
    app = EnterpriseApplication()
    
    # 演示所有功能
    await app.demonstrate_features()
    
    # 显示系统状态
    print("\n=== System Status ===")
    status = app.get_system_status()
    
    print(f"System Status at {status['timestamp']}:")
    for component, stats in status['components'].items():
        print(f"  {component.title()}: ✓ Active")
    
    print("\n=== Demo Completed ===")
    print("All enterprise features have been successfully demonstrated!")
    print("\nKey Features Showcased:")
    print("  ✓ Security Management (API Keys, RBAC)")
    print("  ✓ Performance Optimization (Connection Pools)")
    print("  ✓ Observability (Tracing, Metrics, Alerting)")
    print("  ✓ Data Governance (Data Quality)")
    print("  ✓ High Availability (Load Balancing, Failover)")
    print("  ✓ Enterprise Integration (API Gateway)")


if __name__ == "__main__":
    asyncio.run(main())