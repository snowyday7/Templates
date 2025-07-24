"""高可用性模块

提供完整的高可用性功能，包括：
- 负载均衡
- 故障转移
- 服务发现
- 健康检查
- 熔断器
- 限流器
"""

from .load_balancer import (
    LoadBalancer,
    LoadBalancingStrategy,
    RoundRobinBalancer,
    WeightedRoundRobinBalancer,
    LeastConnectionsBalancer,
    ConsistentHashBalancer,
    HealthCheckBalancer,
    initialize_load_balancer,
    get_load_balancer
)

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    initialize_circuit_breaker,
    get_circuit_breaker_manager
)

from .rate_limiter import (
    RateLimiter,
    TokenBucketLimiter,
    SlidingWindowLimiter,
    FixedWindowLimiter,
    RateLimiterConfig,
    RateLimiterManager,
    initialize_rate_limiter,
    get_rate_limiter_manager
)

from .service_discovery import (
    ServiceRegistry,
    ServiceInstance,
    ServiceDiscovery,
    HealthChecker,
    ServiceStatus,
    initialize_service_discovery,
    get_service_discovery
)

from .failover import (
    FailoverManager,
    FailoverStrategy,
    FailoverConfig,
    BackupService,
    initialize_failover,
    get_failover_manager
)

__all__ = [
    # Load Balancer
    "LoadBalancer",
    "LoadBalancingStrategy",
    "RoundRobinBalancer",
    "WeightedRoundRobinBalancer",
    "LeastConnectionsBalancer",
    "ConsistentHashBalancer",
    "HealthCheckBalancer",
    "initialize_load_balancer",
    "get_load_balancer",
    
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerState",
    "CircuitBreakerConfig",
    "CircuitBreakerManager",
    "initialize_circuit_breaker",
    "get_circuit_breaker_manager",
    
    # Rate Limiter
    "RateLimiter",
    "TokenBucketLimiter",
    "SlidingWindowLimiter",
    "FixedWindowLimiter",
    "RateLimiterConfig",
    "RateLimiterManager",
    "initialize_rate_limiter",
    "get_rate_limiter_manager",
    
    # Service Discovery
    "ServiceRegistry",
    "ServiceInstance",
    "ServiceDiscovery",
    "HealthChecker",
    "ServiceStatus",
    "initialize_service_discovery",
    "get_service_discovery",
    
    # Failover
    "FailoverManager",
    "FailoverStrategy",
    "FailoverConfig",
    "BackupService",
    "initialize_failover",
    "get_failover_manager",
]