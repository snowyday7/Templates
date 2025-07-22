#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python后端开发功能组件模板库

一个全面、强大的Python后端开发功能组件模板库，
为开发者提供开箱即用的高质量代码模板。
"""

from setuptools import setup, find_packages
import os

# 读取README文件
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# 版本信息
VERSION = "1.0.0"

setup(
    name="python-backend-templates",
    version=VERSION,
    author="Python Backend Templates Team",
    author_email="support@python-templates.com",
    description="全面、强大的Python后端开发功能组件模板库",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/python-backend-templates",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/python-backend-templates/issues",
        "Documentation": "https://python-templates.readthedocs.io",
        "Source Code": "https://github.com/your-username/python-backend-templates",
        "Changelog": "https://github.com/your-username/python-backend-templates/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: Database",
        "Topic :: System :: Monitoring",
        "Framework :: FastAPI",
        "Framework :: AsyncIO",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.4.0",
        ],
        "testing": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "factory-boy>=3.3.0",
            "faker>=20.1.0",
        ],
        "monitoring": [
            "sentry-sdk[fastapi]>=1.38.0",
            "prometheus-client>=0.19.0",
            "structlog>=23.2.0",
        ],
        "deployment": [
            "docker>=6.1.0",
            "kubernetes>=28.1.0",
            "pyyaml>=6.0.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "python-templates=templates.cli:main",
            "pt-create=templates.cli:create_project",
            "pt-init=templates.cli:init_template",
        ],
    },
    include_package_data=True,
    package_data={
        "templates": [
            "**/*.py",
            "**/*.yaml",
            "**/*.yml",
            "**/*.json",
            "**/*.toml",
            "**/*.cfg",
            "**/*.ini",
            "**/*.conf",
            "**/*.md",
            "**/*.txt",
        ],
    },
    keywords=[
        "python",
        "backend",
        "templates",
        "fastapi",
        "sqlalchemy",
        "redis",
        "celery",
        "docker",
        "kubernetes",
        "microservices",
        "api",
        "orm",
        "authentication",
        "authorization",
        "monitoring",
        "logging",
        "deployment",
        "devops",
    ],
    zip_safe=False,
)