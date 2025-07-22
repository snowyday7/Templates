# 贡献指南

🎉 感谢您对Python后端开发功能组件模板库的关注！我们欢迎所有形式的贡献。

## 📋 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [开发环境设置](#开发环境设置)
- [提交指南](#提交指南)
- [代码规范](#代码规范)
- [测试指南](#测试指南)
- [文档贡献](#文档贡献)
- [问题报告](#问题报告)
- [功能请求](#功能请求)
- [发布流程](#发布流程)

## 🤝 行为准则

### 我们的承诺

为了营造一个开放和友好的环境，我们作为贡献者和维护者承诺，无论年龄、体型、残疾、种族、性别认同和表达、经验水平、国籍、个人形象、种族、宗教或性取向如何，参与我们项目和社区的每个人都能享受无骚扰的体验。

### 我们的标准

**积极行为包括：**
- 使用友好和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同理心

**不可接受的行为包括：**
- 使用性化的语言或图像
- 恶意评论、人身攻击或政治攻击
- 公开或私下骚扰
- 未经明确许可发布他人的私人信息
- 其他在专业环境中可能被认为不当的行为

## 🚀 如何贡献

### 贡献类型

我们欢迎以下类型的贡献：

1. **🐛 Bug修复**：修复现有功能中的问题
2. **✨ 新功能**：添加新的模板或功能
3. **📚 文档改进**：改善文档质量和完整性
4. **🧪 测试增强**：添加或改进测试用例
5. **🎨 代码优化**：提升代码质量和性能
6. **🔧 工具改进**：改善开发工具和流程

### 贡献流程

1. **Fork 项目**
   ```bash
   # 在GitHub上Fork项目
   # 然后克隆到本地
   git clone https://github.com/your-username/python-backend-templates.git
   cd python-backend-templates
   ```

2. **创建分支**
   ```bash
   # 创建并切换到新分支
   git checkout -b feature/your-feature-name
   # 或者修复bug
   git checkout -b fix/your-bug-fix
   ```

3. **进行更改**
   - 编写代码
   - 添加测试
   - 更新文档

4. **提交更改**
   ```bash
   git add .
   git commit -m "feat: add new template for GraphQL API"
   ```

5. **推送分支**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **创建Pull Request**
   - 在GitHub上创建Pull Request
   - 填写详细的描述
   - 等待代码审查

## 🛠️ 开发环境设置

### 系统要求

- Python 3.8+
- Git
- Docker (可选，用于测试容器化功能)
- Redis (可选，用于测试缓存功能)
- PostgreSQL/MySQL (可选，用于测试数据库功能)

### 环境配置

1. **克隆项目**
   ```bash
   git clone https://github.com/your-username/python-backend-templates.git
   cd python-backend-templates
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   
   # Linux/Mac
   source venv/bin/activate
   
   # Windows
   venv\Scripts\activate
   ```

3. **安装依赖**
   ```bash
   # 安装基础依赖
   pip install -r requirements.txt
   
   # 安装开发依赖
   pip install -r requirements-dev.txt
   
   # 或者使用pip安装开发模式
   pip install -e ".[dev,testing,docs]"
   ```

4. **安装pre-commit钩子**
   ```bash
   pre-commit install
   ```

5. **验证安装**
   ```bash
   # 运行测试
   pytest
   
   # 检查代码格式
   black --check templates/
   flake8 templates/
   isort --check-only templates/
   mypy templates/
   ```

### IDE配置

推荐使用以下IDE配置：

**VS Code**
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

**PyCharm**
- 配置Black作为代码格式化工具
- 启用Flake8和MyPy检查
- 配置isort进行导入排序

## 📝 提交指南

### 提交消息格式

我们使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### 提交类型

- **feat**: 新功能
- **fix**: Bug修复
- **docs**: 文档更新
- **style**: 代码格式化（不影响功能）
- **refactor**: 代码重构
- **test**: 测试相关
- **chore**: 构建过程或辅助工具的变动
- **perf**: 性能优化
- **ci**: CI/CD相关

### 提交示例

```bash
# 新功能
git commit -m "feat(api): add GraphQL API template"

# Bug修复
git commit -m "fix(database): resolve connection pool leak"

# 文档更新
git commit -m "docs: update installation guide"

# 重构
git commit -m "refactor(auth): simplify JWT token validation"

# 测试
git commit -m "test(cache): add Redis integration tests"
```

## 🎨 代码规范

### Python代码规范

我们遵循以下代码规范：

1. **PEP 8**: Python代码风格指南
2. **Black**: 代码格式化
3. **isort**: 导入排序
4. **Flake8**: 代码检查
5. **MyPy**: 类型检查

### 代码质量检查

```bash
# 格式化代码
black templates/

# 排序导入
isort templates/

# 代码检查
flake8 templates/

# 类型检查
mypy templates/

# 安全检查
bandit -r templates/

# 一键检查所有
make lint
```

### 代码风格要求

1. **函数和变量命名**：使用snake_case
2. **类命名**：使用PascalCase
3. **常量命名**：使用UPPER_CASE
4. **文档字符串**：使用Google风格
5. **类型注解**：所有公共函数都应有类型注解

### 示例代码

```python
from typing import Optional, Dict, Any
from pydantic import BaseModel


class UserConfig(BaseModel):
    """用户配置模型。
    
    Args:
        username: 用户名
        email: 邮箱地址
        is_active: 是否激活
        metadata: 额外元数据
    """
    username: str
    email: str
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None


def create_user_config(
    username: str,
    email: str,
    is_active: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> UserConfig:
    """创建用户配置。
    
    Args:
        username: 用户名
        email: 邮箱地址
        is_active: 是否激活，默认为True
        metadata: 额外元数据
        
    Returns:
        UserConfig: 用户配置对象
        
    Raises:
        ValueError: 当用户名或邮箱为空时
    """
    if not username or not email:
        raise ValueError("用户名和邮箱不能为空")
        
    return UserConfig(
        username=username,
        email=email,
        is_active=is_active,
        metadata=metadata or {}
    )
```

## 🧪 测试指南

### 测试要求

1. **测试覆盖率**：新代码的测试覆盖率应达到90%以上
2. **测试类型**：包括单元测试、集成测试和端到端测试
3. **测试命名**：使用描述性的测试名称
4. **测试隔离**：每个测试应该独立运行

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/test_database.py

# 运行带覆盖率的测试
pytest --cov=templates --cov-report=html

# 运行特定标记的测试
pytest -m "not slow"

# 并行运行测试
pytest -n auto
```

### 测试分类

使用pytest标记对测试进行分类：

```python
import pytest

@pytest.mark.unit
def test_user_creation():
    """测试用户创建功能。"""
    pass

@pytest.mark.integration
def test_database_connection():
    """测试数据库连接。"""
    pass

@pytest.mark.slow
def test_large_data_processing():
    """测试大数据处理（耗时较长）。"""
    pass
```

### 测试示例

```python
import pytest
from unittest.mock import Mock, patch
from templates.database import DatabaseManager


class TestSQLAlchemyTemplate:
    """SQLAlchemy模板测试类。"""
    
    def test_create_engine_with_valid_url(self):
        """测试使用有效URL创建数据库引擎。"""
        template = SQLAlchemyTemplate()
        engine = template.create_engine("sqlite:///test.db")
        assert engine is not None
        
    def test_create_engine_with_invalid_url(self):
        """测试使用无效URL创建数据库引擎应抛出异常。"""
        template = SQLAlchemyTemplate()
        with pytest.raises(ValueError):
            template.create_engine("invalid-url")
            
    @patch('templates.database.create_engine')
    def test_create_session(self, mock_create_engine):
        """测试创建数据库会话。"""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        template = SQLAlchemyTemplate()
        session = template.create_session("sqlite:///test.db")
        
        assert session is not None
        mock_create_engine.assert_called_once()
```

## 📚 文档贡献

### 文档类型

1. **API文档**：自动生成的代码文档
2. **用户指南**：使用说明和教程
3. **开发者文档**：架构设计和开发指南
4. **示例代码**：完整的使用示例

### 文档编写规范

1. **语言**：使用简洁明了的中文
2. **格式**：使用Markdown格式
3. **结构**：逻辑清晰，层次分明
4. **示例**：提供完整的代码示例
5. **更新**：保持文档与代码同步

### 构建文档

```bash
# 构建Sphinx文档
cd docs/
make html

# 构建MkDocs文档
mkdocs serve

# 检查文档链接
mkdocs build --strict
```

## 🐛 问题报告

### 报告Bug

在报告Bug时，请提供以下信息：

1. **Bug描述**：清晰描述问题
2. **重现步骤**：详细的重现步骤
3. **期望行为**：期望的正确行为
4. **实际行为**：实际发生的行为
5. **环境信息**：
   - Python版本
   - 操作系统
   - 相关依赖版本
6. **错误日志**：完整的错误堆栈
7. **最小示例**：能重现问题的最小代码

### Bug报告模板

```markdown
## Bug描述
简要描述遇到的问题。

## 重现步骤
1. 执行 '...'
2. 点击 '....'
3. 滚动到 '....'
4. 看到错误

## 期望行为
清晰简洁地描述您期望发生的事情。

## 实际行为
清晰简洁地描述实际发生的事情。

## 环境信息
- OS: [e.g. macOS 12.0]
- Python版本: [e.g. 3.9.7]
- 项目版本: [e.g. 1.0.0]

## 错误日志
```
粘贴完整的错误日志
```

## 最小重现示例
```python
# 提供能重现问题的最小代码示例
```

## 额外信息
添加任何其他有助于解决问题的信息。
```

## ✨ 功能请求

### 请求新功能

在请求新功能时，请提供：

1. **功能描述**：详细描述所需功能
2. **使用场景**：说明使用场景和需求
3. **解决方案**：建议的实现方案
4. **替代方案**：考虑过的其他方案
5. **优先级**：功能的重要性和紧急程度

### 功能请求模板

```markdown
## 功能描述
清晰简洁地描述您想要的功能。

## 使用场景
描述这个功能解决了什么问题，在什么场景下使用。

## 建议的解决方案
清晰简洁地描述您希望如何实现这个功能。

## 替代方案
清晰简洁地描述您考虑过的任何替代解决方案或功能。

## 额外信息
添加任何其他有关功能请求的信息或截图。
```

## 🚀 发布流程

### 版本发布

1. **更新版本号**
   ```bash
   # 使用bump2version更新版本
   bump2version patch  # 修订版本
   bump2version minor  # 次版本
   bump2version major  # 主版本
   ```

2. **更新CHANGELOG**
   - 记录所有重要变更
   - 按照Keep a Changelog格式

3. **创建发布标签**
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

4. **构建和发布**
   ```bash
   # 构建分发包
   python setup.py sdist bdist_wheel
   
   # 检查包
   twine check dist/*
   
   # 发布到PyPI
   twine upload dist/*
   ```

### 发布检查清单

- [ ] 所有测试通过
- [ ] 文档已更新
- [ ] CHANGELOG已更新
- [ ] 版本号已更新
- [ ] 创建了Git标签
- [ ] 构建包成功
- [ ] 发布到PyPI成功

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者！

### 贡献者

- [贡献者列表](https://github.com/your-username/python-backend-templates/contributors)

### 特别感谢

- 所有提供反馈和建议的用户
- 开源社区的支持和启发
- 相关开源项目的贡献

---

## 📞 联系我们

如果您有任何问题或建议，请通过以下方式联系我们：

- 📧 **邮件**：support@python-templates.com
- 💬 **讨论区**：[GitHub Discussions](https://github.com/your-username/python-backend-templates/discussions)
- 🐛 **问题反馈**：[GitHub Issues](https://github.com/your-username/python-backend-templates/issues)

再次感谢您的贡献！🎉