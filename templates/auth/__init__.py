"""认证授权模块模板

提供各种认证授权机制的模板代码，包括：
- JWT认证机制
- OAuth2.0集成
- RBAC权限控制
- 安全中间件
"""

from .jwt_auth import (
    JWTManager,
    AuthService,
    PasswordManager,
    TokenType,
    AuthSettings,
)
# from .oauth_template import OAuthManager, GoogleOAuth, GitHubOAuth  # TODO: 实现OAuth模板
# from .rbac_template import (  # TODO: 实现RBAC模板
#     RBACManager,
#     Permission,
#     Role,
#     User,
#     require_permission,
#     require_role,
# )
# from .password_utils import (  # TODO: 实现密码工具模板
#     PasswordManager,
#     hash_password,
#     verify_password,
#     generate_salt,
# )

__all__ = [
    "JWTManager",
    "create_access_token",
    "verify_token",
    "get_current_user",
    "AuthSettings",
    # "OAuthManager",  # TODO: 实现OAuth模板
    # "GoogleOAuth",  # TODO: 实现OAuth模板
    # "GitHubOAuth",  # TODO: 实现OAuth模板
    # "RBACManager",  # TODO: 实现RBAC模板
    # "Permission",  # TODO: 实现RBAC模板
    # "Role",  # TODO: 实现RBAC模板
    # "require_permission",  # TODO: 实现RBAC模板
    # "require_role",  # TODO: 实现RBAC模板
]
