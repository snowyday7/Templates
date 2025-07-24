"""安全模块

提供企业级安全功能，包括：
- API密钥管理
- RBAC权限系统
- 安全审计
- 数据加密
- 安全策略
"""

# API密钥管理
from .api_key_manager import (
    APIKeyManager,
    APIKey,
    APIKeyScope,
    generate_api_key,
    validate_api_key,
    revoke_api_key,
)

# RBAC权限系统
from .rbac import (
    Role,
    Permission,
    RBACManager,
    require_permission,
    require_role,
    check_permission,
    get_user_permissions,
)

# 安全审计
from .audit import (
    AuditEvent,
    AuditLogger,
    SecurityAuditor,
    log_security_event,
    track_access,
    monitor_suspicious_activity,
)

# 数据加密
from .encryption import (
    EncryptionManager,
    encrypt_data,
    decrypt_data,
    hash_password,
    verify_password,
    generate_salt,
)

# 安全策略
from .security_policy import (
    SecurityPolicy,
    PasswordPolicy,
    SessionPolicy,
    SecurityPolicyManager,
    enforce_password_policy,
    validate_session,
)

__all__ = [
    # API密钥管理
    "APIKeyManager",
    "APIKey",
    "APIKeyScope",
    "generate_api_key",
    "validate_api_key",
    "revoke_api_key",
    
    # RBAC权限系统
    "Role",
    "Permission",
    "RBACManager",
    "require_permission",
    "require_role",
    "check_permission",
    "get_user_permissions",
    
    # 安全审计
    "AuditEvent",
    "AuditLogger",
    "SecurityAuditor",
    "log_security_event",
    "track_access",
    "monitor_suspicious_activity",
    
    # 数据加密
    "EncryptionManager",
    "encrypt_data",
    "decrypt_data",
    "hash_password",
    "verify_password",
    "generate_salt",
    
    # 安全策略
    "SecurityPolicy",
    "PasswordPolicy",
    "SessionPolicy",
    "SecurityPolicyManager",
    "enforce_password_policy",
    "validate_session",
]