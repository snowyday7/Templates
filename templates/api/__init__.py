"""API开发模块模板

提供各种API开发框架的模板代码，包括：
- FastAPI RESTful API模板
- Flask API模板
- GraphQL服务模板
- API文档自动生成
"""

from .fastapi_template import (
    FastAPIApp,
    APISettings,
    create_fastapi_app,
    setup_cors,
    setup_middleware,
    get_current_user,
)
from fastapi import APIRouter
# from .flask_template import FlaskApp, create_flask_app, Blueprint  # TODO: 实现Flask模板
# from .graphql_template import GraphQLApp, create_graphql_app  # TODO: 实现GraphQL模板
# from .middleware import (  # TODO: 实现中间件模板
#     CORSMiddleware,
#     AuthMiddleware,
#     LoggingMiddleware,
#     RateLimitMiddleware,
# )

__all__ = [
    "FastAPIApp",
    "APISettings",
    "create_fastapi_app",
    "setup_cors",
    "setup_middleware",
    "APIRouter",
    "get_current_user",
]