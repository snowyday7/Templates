# Python后端开发功能组件模板库 - Makefile
# 提供便捷的开发、测试和部署命令

.PHONY: help install install-dev test lint format clean build publish docs serve-docs

# 默认目标
help:
	@echo "Python后端开发功能组件模板库 - 可用命令:"
	@echo ""
	@echo "开发环境:"
	@echo "  install      - 安装基础依赖"
	@echo "  install-dev  - 安装开发依赖"
	@echo "  setup        - 设置开发环境（包括pre-commit）"
	@echo ""
	@echo "代码质量:"
	@echo "  lint         - 运行所有代码检查"
	@echo "  format       - 格式化代码"
	@echo "  type-check   - 运行类型检查"
	@echo "  security     - 运行安全检查"
	@echo ""
	@echo "测试:"
	@echo "  test         - 运行所有测试"
	@echo "  test-unit    - 运行单元测试"
	@echo "  test-integration - 运行集成测试"
	@echo "  test-cov     - 运行测试并生成覆盖率报告"
	@echo ""
	@echo "构建和发布:"
	@echo "  build        - 构建分发包"
	@echo "  publish      - 发布到PyPI"
	@echo "  publish-test - 发布到测试PyPI"
	@echo ""
	@echo "文档:"
	@echo "  docs         - 构建文档"
	@echo "  serve-docs   - 启动文档服务器"
	@echo ""
	@echo "清理:"
	@echo "  clean        - 清理构建文件"
	@echo "  clean-all    - 清理所有生成文件"

# 安装依赖
install:
	@echo "📦 安装基础依赖..."
	pip install -r requirements.txt

install-dev:
	@echo "🛠️ 安装开发依赖..."
	pip install -r requirements-dev.txt
	pip install -e ".[dev,testing,docs]"

# 设置开发环境
setup: install-dev
	@echo "⚙️ 设置开发环境..."
	pre-commit install
	@echo "✅ 开发环境设置完成！"

# 代码格式化
format:
	@echo "🎨 格式化代码..."
	black templates/ tests/
	isort templates/ tests/
	@echo "✅ 代码格式化完成！"

# 代码检查
lint:
	@echo "🔍 运行代码检查..."
	black --check templates/ tests/
	flake8 templates/ tests/
	isort --check-only templates/ tests/
	mypy templates/
	@echo "✅ 代码检查完成！"

# 类型检查
type-check:
	@echo "🔍 运行类型检查..."
	mypy templates/

# 安全检查
security:
	@echo "🔒 运行安全检查..."
	bandit -r templates/
	safety check

# 测试
test:
	@echo "🧪 运行所有测试..."
	pytest tests/ -v

test-unit:
	@echo "🧪 运行单元测试..."
	pytest tests/ -v -m "unit"

test-integration:
	@echo "🧪 运行集成测试..."
	pytest tests/ -v -m "integration"

test-cov:
	@echo "📊 运行测试并生成覆盖率报告..."
	pytest tests/ --cov=templates --cov-report=html --cov-report=term
	@echo "📊 覆盖率报告已生成到 htmlcov/ 目录"

test-fast:
	@echo "⚡ 运行快速测试（跳过慢速测试）..."
	pytest tests/ -v -m "not slow"

# 构建
build: clean
	@echo "🏗️ 构建分发包..."
	python setup.py sdist bdist_wheel
	twine check dist/*
	@echo "✅ 构建完成！包文件在 dist/ 目录"

# 发布
publish: build
	@echo "🚀 发布到PyPI..."
	twine upload dist/*

publish-test: build
	@echo "🧪 发布到测试PyPI..."
	twine upload --repository testpypi dist/*

# 文档
docs:
	@echo "📚 构建文档..."
	cd docs && make html
	@echo "📚 文档已构建到 docs/_build/html/"

serve-docs:
	@echo "🌐 启动文档服务器..."
	mkdocs serve

# 清理
clean:
	@echo "🧹 清理构建文件..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-all: clean
	@echo "🧹 清理所有生成文件..."
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	rm -rf docs/_build/
	rm -rf site/

# 开发工具
check-deps:
	@echo "🔍 检查依赖更新..."
	pip list --outdated

update-deps:
	@echo "⬆️ 更新依赖..."
	pip-review --auto

# 项目信息
info:
	@echo "📋 项目信息:"
	@echo "  名称: Python Backend Templates"
	@echo "  版本: $(shell python -c 'from templates import __version__; print(__version__)')"
	@echo "  Python版本: $(shell python --version)"
	@echo "  虚拟环境: $(VIRTUAL_ENV)"

# 快速开发命令
dev: format lint test
	@echo "✅ 开发检查完成！"

# CI/CD命令
ci: install-dev lint test-cov security
	@echo "✅ CI检查完成！"

# 发布前检查
pre-release: clean ci build
	@echo "✅ 发布前检查完成！"

# 创建示例项目
example:
	@echo "📝 创建示例项目..."
	python -m templates.cli create example-project --module api --module database --module auth --output examples/

# Docker相关命令
docker-build:
	@echo "🐳 构建Docker镜像..."
	docker build -t python-backend-templates .

docker-run:
	@echo "🐳 运行Docker容器..."
	docker run -p 8000:8000 python-backend-templates

# 数据库相关（用于测试）
db-start:
	@echo "🗄️ 启动测试数据库..."
	docker run -d --name test-postgres -e POSTGRES_PASSWORD=test -e POSTGRES_DB=test -p 5432:5432 postgres:13

db-stop:
	@echo "🗄️ 停止测试数据库..."
	docker stop test-postgres && docker rm test-postgres

# Redis相关（用于测试）
redis-start:
	@echo "🔴 启动测试Redis..."
	docker run -d --name test-redis -p 6379:6379 redis:7-alpine

redis-stop:
	@echo "🔴 停止测试Redis..."
	docker stop test-redis && docker rm test-redis

# 测试环境
test-env-start: db-start redis-start
	@echo "🧪 测试环境已启动"

test-env-stop: db-stop redis-stop
	@echo "🧪 测试环境已停止"

# 性能测试
perf-test:
	@echo "⚡ 运行性能测试..."
	pytest tests/ -v -m "performance" --benchmark-only

# 生成需求文件
freeze:
	@echo "❄️ 生成当前环境的需求文件..."
	pip freeze > requirements.lock

# 检查许可证
check-licenses:
	@echo "📄 检查依赖许可证..."
	pip-licenses

# 代码统计
stats:
	@echo "📊 代码统计:"
	@echo "Python文件数量:"
	@find templates/ -name "*.py" | wc -l
	@echo "代码行数:"
	@find templates/ -name "*.py" -exec wc -l {} + | tail -1
	@echo "测试文件数量:"
	@find tests/ -name "*.py" | wc -l

# Git相关
git-clean:
	@echo "🧹 清理Git仓库..."
	git clean -fd
	git reset --hard HEAD

tag:
	@echo "🏷️ 创建版本标签..."
	@read -p "输入版本号: " version; \
	git tag -a v$$version -m "Release version $$version"; \
	echo "标签 v$$version 已创建"

# 帮助信息
help-dev:
	@echo "🛠️ 开发者常用命令:"
	@echo "  make setup       - 初始化开发环境"
	@echo "  make dev         - 运行开发检查（格式化+检查+测试）"
	@echo "  make test-cov    - 运行测试并查看覆盖率"
	@echo "  make serve-docs  - 启动文档服务器"
	@echo "  make example     - 创建示例项目"

help-ci:
	@echo "🤖 CI/CD常用命令:"
	@echo "  make ci          - 运行CI检查"
	@echo "  make pre-release - 发布前检查"
	@echo "  make publish     - 发布到PyPI"

# 版本管理
bump-patch:
	@echo "⬆️ 升级补丁版本..."
	bump2version patch

bump-minor:
	@echo "⬆️ 升级次版本..."
	bump2version minor

bump-major:
	@echo "⬆️ 升级主版本..."
	bump2version major