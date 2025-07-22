"""Docker配置模板
提供完整的Docker功能，包括：
- Dockerfile生成
- Docker Compose配置
- 容器管理
- 镜像构建
- 多阶段构建
"""
import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
class PythonVersion(str, Enum):
    """Python版本枚举"""
    PYTHON_38 = "3.8"
    PYTHON_39 = "3.9"
    PYTHON_310 = "3.10"
    PYTHON_311 = "3.11"
    PYTHON_312 = "3.12"
class BaseImage(str, Enum):
    """基础镜像枚举"""
    PYTHON_SLIM = "python:{version}-slim"
    PYTHON_ALPINE = "python:{version}-alpine"
    UBUNTU = "ubuntu:22.04"
    DEBIAN = "debian:bullseye-slim"
class DockerConfig(BaseModel):
    """Docker配置"""
    # 基础配置
    app_name: str = "python-app"
    python_version: PythonVersion = PythonVersion.PYTHON_311
    base_image: BaseImage = BaseImage.PYTHON_SLIM
    # 应用配置
    working_dir: str = "/app"
    app_port: int = 8000
    requirements_file: str = "requirements.txt"
    app_module: str = "main:app"
    # 用户配置
    create_user: bool = True
    user_name: str = "appuser"
    user_id: int = 1000
    group_id: int = 1000
    # 环境变量
    environment_vars: Dict[str, str] = Field(default_factory=dict)
    # 卷挂载
    volumes: List[str] = Field(default_factory=list)
    # 健康检查
    health_check: bool = True
    health_check_path: str = "/health"
    health_check_interval: str = "30s"
    health_check_timeout: str = "10s"
    health_check_retries: int = 3
    # 多阶段构建
    multi_stage: bool = False
    build_stage_packages: List[str] = Field(default_factory=list)
    runtime_packages: List[str] = Field(default_factory=list)
class DockerfileGenerator:
    """Dockerfile生成器"""
    def __init__(self, config: DockerConfig):
        self.config = config
    def generate(self) -> str:
        """生成Dockerfile内容"""
        if self.config.multi_stage:
            return self._generate_multi_stage_dockerfile()
        else:
            return self._generate_single_stage_dockerfile()
    def _generate_single_stage_dockerfile(self) -> str:
        """生成单阶段Dockerfile"""
        base_image = self.config.base_image.value.format(version=self.config.python_version.value)
        dockerfile_content = f"""# Python应用Dockerfile
FROM {base_image}
# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1
"""
        # 添加自定义环境变量
        if self.config.environment_vars:
            dockerfile_content += "# 自定义环境变量\n"
            for key, value in self.config.environment_vars.items():
                dockerfile_content += f"ENV {key}={value}\n"
            dockerfile_content += "\n"
        # 安装系统依赖
        if "alpine" in base_image:
            dockerfile_content += """# 安装系统依赖
RUN apk update && apk add --no-cache \\
    gcc \\
    musl-dev \\
    libffi-dev \\
    openssl-dev \\
    && rm -rf /var/cache/apk/*
"""
        else:
            dockerfile_content += """# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    libc6-dev \\
    libffi-dev \\
    libssl-dev \\
    && rm -rf /var/lib/apt/lists/*
"""
        # 创建用户
        if self.config.create_user:
            dockerfile_content += f"""# 创建应用用户
RUN groupadd -g {self.config.group_id} {self.config.user_name} && \\
    useradd -r -u {self.config.user_id} -g {self.config.user_name} {self.config.user_name}
"""
        # 设置工作目录
        dockerfile_content += f"""# 设置工作目录
WORKDIR {self.config.working_dir}
"""
        # 复制并安装Python依赖
        dockerfile_content += f"""# 复制并安装Python依赖
COPY {self.config.requirements_file} .
RUN pip install --no-cache-dir -r {self.config.requirements_file}
"""
        # 复制应用代码
        dockerfile_content += """# 复制应用代码
COPY . .
"""
        # 更改所有权
        if self.config.create_user:
            dockerfile_content += f"""# 更改文件所有权
RUN chown -R {self.config.user_name}:{self.config.user_name} {self.config.working_dir}
"""
        # 切换用户
        if self.config.create_user:
            dockerfile_content += f"""# 切换到应用用户
USER {self.config.user_name}
"""
        # 暴露端口
        dockerfile_content += f"""# 暴露端口
EXPOSE {self.config.app_port}
"""
        # 健康检查
        if self.config.health_check:
            dockerfile_content += f"""# 健康检查
HEALTHCHECK --interval={self.config.health_check_interval} \\
            --timeout={self.config.health_check_timeout} \\
            --retries={self.config.health_check_retries} \\
    CMD curl -f http://localhost:{self.config.app_port}{self.config.health_check_path} || exit 1
"""
        # 启动命令
        dockerfile_content += f"""# 启动应用
CMD ["python", "-m", "uvicorn", "{self.config.app_module}", "--host", "0.0.0.0", "--port", "{self.config.app_port}"]
"""
        return dockerfile_content
    def _generate_multi_stage_dockerfile(self) -> str:
        """生成多阶段Dockerfile"""
        base_image = self.config.base_image.value.format(version=self.config.python_version.value)
        dockerfile_content = f"""# 多阶段构建Dockerfile
# 构建阶段
FROM {base_image} AS builder
# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1
"""
        # 安装构建依赖
        if "alpine" in base_image:
            dockerfile_content += """# 安装构建依赖
RUN apk update && apk add --no-cache \\
    gcc \\
    musl-dev \\
    libffi-dev \\
    openssl-dev \\
    build-base
"""
        else:
            dockerfile_content += """# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    libc6-dev \\
    libffi-dev \\
    libssl-dev \\
    build-essential
"""
        # 添加构建阶段包
        if self.config.build_stage_packages:
            packages = " \\".join(self.config.build_stage_packages)
            if "alpine" in base_image:
                dockerfile_content += f"""# 安装额外构建包
RUN apk add --no-cache \\
    {packages}
"""
            else:
                dockerfile_content += f"""# 安装额外构建包
RUN apt-get install -y --no-install-recommends \\
    {packages}
"""
        dockerfile_content += f"""# 设置工作目录
WORKDIR {self.config.working_dir}
# 复制并安装Python依赖
COPY {self.config.requirements_file} .
RUN pip install --user --no-cache-dir -r {self.config.requirements_file}
# 运行阶段
FROM {base_image} AS runtime
# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PATH=/home/{self.config.user_name}/.local/bin:$PATH
"""
        # 添加自定义环境变量
        if self.config.environment_vars:
            dockerfile_content += "# 自定义环境变量\n"
            for key, value in self.config.environment_vars.items():
                dockerfile_content += f"ENV {key}={value}\n"
            dockerfile_content += "\n"
        # 安装运行时依赖
        if self.config.runtime_packages:
            packages = " \\".join(self.config.runtime_packages)
            if "alpine" in base_image:
                dockerfile_content += f"""# 安装运行时依赖
RUN apk update && apk add --no-cache \\
    {packages} \\
    && rm -rf /var/cache/apk/*
"""
            else:
                dockerfile_content += f"""# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \\
    {packages} \\
    && rm -rf /var/lib/apt/lists/*
"""
        # 创建用户
        if self.config.create_user:
            dockerfile_content += f"""# 创建应用用户
RUN groupadd -g {self.config.group_id} {self.config.user_name} && \\
    useradd -r -u {self.config.user_id} -g {self.config.user_name} {self.config.user_name}
"""
        # 从构建阶段复制依赖
        dockerfile_content += f"""# 从构建阶段复制Python包
COPY --from=builder /root/.local /home/{self.config.user_name}/.local
# 设置工作目录
WORKDIR {self.config.working_dir}
# 复制应用代码
COPY . .
"""
        # 更改所有权
        if self.config.create_user:
            dockerfile_content += f"""# 更改文件所有权
RUN chown -R {self.config.user_name}:{self.config.user_name} {self.config.working_dir} /home/{self.config.user_name}/.local
"""
        # 切换用户
        if self.config.create_user:
            dockerfile_content += f"""# 切换到应用用户
USER {self.config.user_name}
"""
        # 暴露端口
        dockerfile_content += f"""# 暴露端口
EXPOSE {self.config.app_port}
"""
        # 健康检查
        if self.config.health_check:
            dockerfile_content += f"""# 健康检查
HEALTHCHECK --interval={self.config.health_check_interval} \\
            --timeout={self.config.health_check_timeout} \\
            --retries={self.config.health_check_retries} \\
    CMD curl -f http://localhost:{self.config.app_port}{self.config.health_check_path} || exit 1
"""
        # 启动命令
        dockerfile_content += f"""# 启动应用
CMD ["python", "-m", "uvicorn", "{self.config.app_module}", "--host", "0.0.0.0", "--port", "{self.config.app_port}"]
"""
        return dockerfile_content
    def save_to_file(self, filepath: Union[str, Path] = "Dockerfile") -> None:
        """保存Dockerfile到文件"""
        content = self.generate()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
class DockerComposeGenerator:
    """Docker Compose生成器"""
    def __init__(self, config: DockerConfig):
        self.config = config
    def generate(self, include_database: bool = True, include_redis: bool = True,
                include_nginx: bool = False) -> str:
        """生成docker-compose.yml内容"""
        compose_content = """version: '3.8'
services:
"""
        # 应用服务
        compose_content += self._generate_app_service()
        # 数据库服务
        if include_database:
            compose_content += self._generate_database_service()
        # Redis服务
        if include_redis:
            compose_content += self._generate_redis_service()
        # Nginx服务
        if include_nginx:
            compose_content += self._generate_nginx_service()
        # 网络配置
        compose_content += self._generate_networks()
        # 卷配置
        compose_content += self._generate_volumes(include_database, include_redis)
        return compose_content
    def _generate_app_service(self) -> str:
        """生成应用服务配置"""
        service_config = f"""  {self.config.app_name}:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "{self.config.app_port}:{self.config.app_port}"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/app_db
      - REDIS_URL=redis://redis:6379/0
"""
        # 添加自定义环境变量
        for key, value in self.config.environment_vars.items():
            service_config += f"      - {key}={value}\n"
        service_config += """    depends_on:
      - db
      - redis
    networks:
      - app-network
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs
"""
        # 添加自定义卷
        for volume in self.config.volumes:
            service_config += f"      - {volume}\n"
        service_config += "\n"
        return service_config
    def _generate_database_service(self) -> str:
        """生成数据库服务配置"""
        return """  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=app_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - app-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 30s
      timeout: 10s
      retries: 3
"""
    def _generate_redis_service(self) -> str:
        """生成Redis服务配置"""
        return """  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - app-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
"""
    def _generate_nginx_service(self) -> str:
        """生成Nginx服务配置"""
        return """  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    networks:
      - app-network
    restart: unless-stopped
"""
    def _generate_networks(self) -> str:
        """生成网络配置"""
        return """networks:
  app-network:
    driver: bridge
"""
    def _generate_volumes(self, include_database: bool, include_redis: bool) -> str:
        """生成卷配置"""
        volumes_config = "volumes:\n"
        if include_database:
            volumes_config += "  postgres_data:\n"
        if include_redis:
            volumes_config += "  redis_data:\n"
        return volumes_config
    def save_to_file(self, filepath: Union[str, Path] = "docker-compose.yml",
                    include_database: bool = True, include_redis: bool = True,
                    include_nginx: bool = False) -> None:
        """保存docker-compose.yml到文件"""
        content = self.generate(include_database, include_redis, include_nginx)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
class ContainerManager:
    """容器管理器"""
    def __init__(self, config: DockerConfig):
        self.config = config
    def build_image(self, tag: Optional[str] = None, dockerfile: str = "Dockerfile",
                   context: str = ".", no_cache: bool = False) -> bool:
        """构建Docker镜像"""
        if not tag:
            tag = self.config.app_name
        cmd = ["docker", "build", "-t", tag, "-f", dockerfile]
        if no_cache:
            cmd.append("--no-cache")
        cmd.append(context)
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Image built successfully: {tag}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to build image: {e.stderr}")
            return False
    def run_container(self, image: Optional[str] = None, name: Optional[str] = None,
                     ports: Optional[Dict[int, int]] = None, 
                     environment: Optional[Dict[str, str]] = None,
                     volumes: Optional[Dict[str, str]] = None,
                     detach: bool = True) -> bool:
        """运行Docker容器"""
        if not image:
            image = self.config.app_name
        if not name:
            name = f"{self.config.app_name}-container"
        cmd = ["docker", "run"]
        if detach:
            cmd.append("-d")
        cmd.extend(["--name", name])
        # 端口映射
        if ports:
            for host_port, container_port in ports.items():
                cmd.extend(["-p", f"{host_port}:{container_port}"])
        else:
            cmd.extend(["-p", f"{self.config.app_port}:{self.config.app_port}"])
        # 环境变量
        if environment:
            for key, value in environment.items():
                cmd.extend(["-e", f"{key}={value}"])
        # 卷挂载
        if volumes:
            for host_path, container_path in volumes.items():
                cmd.extend(["-v", f"{host_path}:{container_path}"])
        cmd.append(image)
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Container started successfully: {name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to start container: {e.stderr}")
            return False
    def stop_container(self, name: Optional[str] = None) -> bool:
        """停止Docker容器"""
        if not name:
            name = f"{self.config.app_name}-container"
        try:
            subprocess.run(["docker", "stop", name], check=True)
            print(f"Container stopped successfully: {name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to stop container: {e}")
            return False
    def remove_container(self, name: Optional[str] = None) -> bool:
        """删除Docker容器"""
        if not name:
            name = f"{self.config.app_name}-container"
        try:
            subprocess.run(["docker", "rm", name], check=True)
            print(f"Container removed successfully: {name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to remove container: {e}")
            return False
    def get_container_logs(self, name: Optional[str] = None, tail: int = 100) -> str:
        """获取容器日志"""
        if not name:
            name = f"{self.config.app_name}-container"
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", str(tail), name],
                check=True, capture_output=True, text=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Failed to get logs: {e.stderr}"
class ImageBuilder:
    """镜像构建器"""
    def __init__(self, config: DockerConfig):
        self.config = config
        self.dockerfile_generator = DockerfileGenerator(config)
        self.container_manager = ContainerManager(config)
    def build_and_push(self, registry: str, tag: Optional[str] = None,
                      dockerfile: str = "Dockerfile", context: str = ".") -> bool:
        """构建并推送镜像"""
        if not tag:
            tag = self.config.app_name
        full_tag = f"{registry}/{tag}"
        # 构建镜像
        if not self.container_manager.build_image(full_tag, dockerfile, context):
            return False
        # 推送镜像
        try:
            subprocess.run(["docker", "push", full_tag], check=True)
            print(f"Image pushed successfully: {full_tag}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to push image: {e}")
            return False
    def generate_and_build(self, tag: Optional[str] = None) -> bool:
        """生成Dockerfile并构建镜像"""
        # 生成Dockerfile
        self.dockerfile_generator.save_to_file()
        # 构建镜像
        return self.container_manager.build_image(tag)
# 全局配置和实例
docker_config = DockerConfig()
dockerfile_generator = DockerfileGenerator(docker_config)
docker_compose_generator = DockerComposeGenerator(docker_config)
container_manager = ContainerManager(docker_config)
image_builder = ImageBuilder(docker_config)
# 便捷函数
def generate_dockerfile(config: Optional[DockerConfig] = None) -> str:
    """生成Dockerfile"""
    generator = DockerfileGenerator(config or docker_config)
    return generator.generate()
def generate_docker_compose(config: Optional[DockerConfig] = None,
                           include_database: bool = True,
                           include_redis: bool = True,
                           include_nginx: bool = False) -> str:
    """生成docker-compose.yml"""
    generator = DockerComposeGenerator(config or docker_config)
    return generator.generate(include_database, include_redis, include_nginx)
def build_image(tag: Optional[str] = None, config: Optional[DockerConfig] = None) -> bool:
    """构建Docker镜像"""
    manager = ContainerManager(config or docker_config)
    return manager.build_image(tag)
def run_container(image: Optional[str] = None, name: Optional[str] = None,
                 config: Optional[DockerConfig] = None) -> bool:
    """运行Docker容器"""
    manager = ContainerManager(config or docker_config)
    return manager.run_container(image, name)
def stop_container(name: Optional[str] = None, config: Optional[DockerConfig] = None) -> bool:
    """停止Docker容器"""
    manager = ContainerManager(config or docker_config)
    return manager.stop_container(name)
# 使用示例
if __name__ == "__main__":
    # 创建配置
    config = DockerConfig(
        app_name="my-python-app",
        python_version=PythonVersion.PYTHON_311,
        app_port=8000,
        environment_vars={
            "ENV": "production",
            "DEBUG": "false"
        },
        multi_stage=True
    )
    # 生成Dockerfile
    dockerfile_gen = DockerfileGenerator(config)
    dockerfile_content = dockerfile_gen.generate()
    print("Generated Dockerfile:")
    print(dockerfile_content)
    # 生成docker-compose.yml
    compose_gen = DockerComposeGenerator(config)
    compose_content = compose_gen.generate()
    print("\nGenerated docker-compose.yml:")
    print(compose_content)
    # 保存文件
    dockerfile_gen.save_to_file()
    compose_gen.save_to_file()
    print("\nFiles saved successfully!")