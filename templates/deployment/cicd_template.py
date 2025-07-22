"""CI/CD配置模板

提供完整的CI/CD配置功能，包括：
- GitHub Actions配置生成
- GitLab CI配置生成
- Jenkins Pipeline配置
- Docker构建和推送
- 自动化测试
- 部署流水线
"""

import json
import yaml
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class CIPlatform(str, Enum):
    """CI平台类型"""

    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    AZURE_DEVOPS = "azure_devops"


class TriggerEvent(str, Enum):
    """触发事件"""

    PUSH = "push"
    PULL_REQUEST = "pull_request"
    SCHEDULE = "schedule"
    MANUAL = "workflow_dispatch"
    TAG = "tag"


class Environment(str, Enum):
    """部署环境"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class StepConfig(BaseModel):
    """步骤配置"""

    name: str
    command: Optional[str] = None
    script: Optional[List[str]] = None
    uses: Optional[str] = None  # GitHub Actions action
    with_params: Optional[Dict[str, Any]] = None
    env: Optional[Dict[str, str]] = None
    condition: Optional[str] = None
    timeout: Optional[int] = None
    retry: Optional[int] = None


class JobConfig(BaseModel):
    """作业配置"""

    name: str
    runs_on: str = "ubuntu-latest"
    steps: List[StepConfig]
    env: Optional[Dict[str, str]] = None
    needs: Optional[List[str]] = None
    condition: Optional[str] = None
    timeout: Optional[int] = None
    strategy: Optional[Dict[str, Any]] = None
    services: Optional[Dict[str, Any]] = None


class PipelineConfig(BaseModel):
    """流水线配置"""

    name: str
    platform: CIPlatform

    # 触发配置
    triggers: List[TriggerEvent] = Field(default_factory=lambda: [TriggerEvent.PUSH])
    branches: List[str] = Field(default_factory=lambda: ["main", "develop"])
    paths: Optional[List[str]] = None
    schedule: Optional[str] = None  # cron表达式

    # 作业配置
    jobs: List[JobConfig]

    # 全局环境变量
    env: Optional[Dict[str, str]] = None

    # 权限配置
    permissions: Optional[Dict[str, str]] = None

    # 并发控制
    concurrency: Optional[Dict[str, Any]] = None


class DockerConfig(BaseModel):
    """Docker配置"""

    registry: str = "docker.io"
    repository: str
    tag_strategy: str = "commit"  # commit, branch, semantic
    dockerfile: str = "Dockerfile"
    context: str = "."
    build_args: Optional[Dict[str, str]] = None
    platforms: List[str] = Field(default_factory=lambda: ["linux/amd64"])


class TestConfig(BaseModel):
    """测试配置"""

    enabled: bool = True

    # 单元测试
    unit_tests: bool = True
    unit_test_command: str = "pytest tests/unit"

    # 集成测试
    integration_tests: bool = True
    integration_test_command: str = "pytest tests/integration"

    # 端到端测试
    e2e_tests: bool = False
    e2e_test_command: str = "pytest tests/e2e"

    # 代码覆盖率
    coverage: bool = True
    coverage_threshold: int = 80
    coverage_command: str = "pytest --cov=src --cov-report=xml"

    # 代码质量检查
    linting: bool = True
    lint_command: str = "flake8 src tests"

    # 类型检查
    type_checking: bool = True
    type_check_command: str = "mypy src"

    # 安全扫描
    security_scan: bool = True
    security_scan_command: str = "bandit -r src"


class DeploymentConfig(BaseModel):
    """部署配置"""

    environment: Environment
    enabled: bool = True

    # 部署策略
    strategy: str = "rolling"  # rolling, blue_green, canary

    # Kubernetes部署
    kubernetes: bool = False
    k8s_namespace: Optional[str] = None
    k8s_manifest_path: str = "k8s"

    # Docker部署
    docker: bool = True
    docker_compose_file: str = "docker-compose.yml"

    # 服务器部署
    servers: List[str] = Field(default_factory=list)
    deploy_script: Optional[str] = None

    # 健康检查
    health_check: bool = True
    health_check_url: Optional[str] = None
    health_check_timeout: int = 300

    # 回滚配置
    rollback_enabled: bool = True
    rollback_on_failure: bool = True


class CICDGenerator:
    """CI/CD配置生成器"""

    def __init__(self, output_dir: str = ".github/workflows"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_github_actions(self, config: PipelineConfig) -> Dict[str, Any]:
        """生成GitHub Actions配置"""
        workflow = {"name": config.name, "on": self._build_github_triggers(config)}

        if config.env:
            workflow["env"] = config.env

        if config.permissions:
            workflow["permissions"] = config.permissions

        if config.concurrency:
            workflow["concurrency"] = config.concurrency

        # 构建作业
        jobs = {}
        for job in config.jobs:
            job_config = {
                "runs-on": job.runs_on,
                "steps": self._build_github_steps(job.steps),
            }

            if job.env:
                job_config["env"] = job.env

            if job.needs:
                job_config["needs"] = job.needs

            if job.condition:
                job_config["if"] = job.condition

            if job.timeout:
                job_config["timeout-minutes"] = job.timeout

            if job.strategy:
                job_config["strategy"] = job.strategy

            if job.services:
                job_config["services"] = job.services

            jobs[job.name.lower().replace(" ", "-")] = job_config

        workflow["jobs"] = jobs
        return workflow

    def generate_gitlab_ci(self, config: PipelineConfig) -> Dict[str, Any]:
        """生成GitLab CI配置"""
        gitlab_ci = {
            "stages": [job.name.lower().replace(" ", "-") for job in config.jobs]
        }

        if config.env:
            gitlab_ci["variables"] = config.env

        # 构建作业
        for job in config.jobs:
            job_name = job.name.lower().replace(" ", "-")
            job_config = {
                "stage": job_name,
                "image": (
                    job.runs_on if job.runs_on != "ubuntu-latest" else "ubuntu:latest"
                ),
                "script": self._build_gitlab_script(job.steps),
            }

            if job.env:
                job_config["variables"] = job.env

            if job.needs:
                job_config["needs"] = job.needs

            if job.condition:
                job_config["rules"] = [{"if": job.condition}]

            if job.timeout:
                job_config["timeout"] = f"{job.timeout}m"

            if job.services:
                job_config["services"] = list(job.services.values())

            gitlab_ci[job_name] = job_config

        return gitlab_ci

    def generate_jenkins_pipeline(self, config: PipelineConfig) -> str:
        """生成Jenkins Pipeline配置"""
        pipeline = f"""pipeline {{
    agent any
    
    environment {{
"""

        if config.env:
            for key, value in config.env.items():
                pipeline += f"        {key} = '{value}'\n"

        pipeline += "    }\n\n"

        # 触发器
        pipeline += "    triggers {\n"
        if TriggerEvent.SCHEDULE in config.triggers and config.schedule:
            pipeline += f"        cron('{config.schedule}')\n"
        pipeline += "    }\n\n"

        # 阶段
        pipeline += "    stages {\n"
        for job in config.jobs:
            pipeline += f"        stage('{job.name}') {{\n"
            pipeline += "            steps {\n"

            for step in job.steps:
                if step.command:
                    pipeline += f"                sh '{step.command}'\n"
                elif step.script:
                    for cmd in step.script:
                        pipeline += f"                sh '{cmd}'\n"

            pipeline += "            }\n"
            pipeline += "        }\n"

        pipeline += "    }\n"

        # 后处理
        pipeline += "\n    post {\n"
        pipeline += "        always {\n"
        pipeline += "            cleanWs()\n"
        pipeline += "        }\n"
        pipeline += "    }\n"
        pipeline += "}"

        return pipeline

    def generate_build_pipeline(
        self,
        app_name: str,
        docker_config: DockerConfig,
        test_config: TestConfig,
        deploy_configs: List[DeploymentConfig],
    ) -> PipelineConfig:
        """生成标准构建流水线"""
        jobs = []

        # 测试作业
        if test_config.enabled:
            test_steps = [
                StepConfig(name="Checkout code", uses="actions/checkout@v4"),
                StepConfig(
                    name="Set up Python",
                    uses="actions/setup-python@v4",
                    with_params={"python-version": "3.11"},
                ),
                StepConfig(
                    name="Install dependencies",
                    command="pip install -r requirements.txt",
                ),
            ]

            if test_config.linting:
                test_steps.append(
                    StepConfig(name="Lint code", command=test_config.lint_command)
                )

            if test_config.type_checking:
                test_steps.append(
                    StepConfig(
                        name="Type check", command=test_config.type_check_command
                    )
                )

            if test_config.security_scan:
                test_steps.append(
                    StepConfig(
                        name="Security scan", command=test_config.security_scan_command
                    )
                )

            if test_config.unit_tests:
                test_steps.append(
                    StepConfig(
                        name="Run unit tests", command=test_config.unit_test_command
                    )
                )

            if test_config.integration_tests:
                test_steps.append(
                    StepConfig(
                        name="Run integration tests",
                        command=test_config.integration_test_command,
                    )
                )

            if test_config.coverage:
                test_steps.append(
                    StepConfig(name="Upload coverage", uses="codecov/codecov-action@v3")
                )

            jobs.append(
                JobConfig(name="test", runs_on="ubuntu-latest", steps=test_steps)
            )

        # 构建作业
        build_steps = [
            StepConfig(name="Checkout code", uses="actions/checkout@v4"),
            StepConfig(
                name="Set up Docker Buildx", uses="docker/setup-buildx-action@v3"
            ),
            StepConfig(
                name="Login to registry",
                uses="docker/login-action@v3",
                with_params={
                    "registry": docker_config.registry,
                    "username": "${{ secrets.DOCKER_USERNAME }}",
                    "password": "${{ secrets.DOCKER_PASSWORD }}",
                },
            ),
            StepConfig(
                name="Build and push",
                uses="docker/build-push-action@v5",
                with_params={
                    "context": docker_config.context,
                    "file": docker_config.dockerfile,
                    "push": True,
                    "tags": f"{docker_config.repository}:${{{{ github.sha }}}}",
                    "platforms": ",".join(docker_config.platforms),
                },
            ),
        ]

        build_job = JobConfig(name="build", runs_on="ubuntu-latest", steps=build_steps)

        if test_config.enabled:
            build_job.needs = ["test"]

        jobs.append(build_job)

        # 部署作业
        for deploy_config in deploy_configs:
            if not deploy_config.enabled:
                continue

            deploy_steps = [
                StepConfig(name="Checkout code", uses="actions/checkout@v4")
            ]

            if deploy_config.kubernetes:
                deploy_steps.extend(
                    [
                        StepConfig(
                            name="Set up kubectl", uses="azure/setup-kubectl@v3"
                        ),
                        StepConfig(
                            name="Deploy to Kubernetes",
                            command=f"kubectl apply -f {deploy_config.k8s_manifest_path}/",
                        ),
                    ]
                )

            if deploy_config.docker:
                deploy_steps.append(
                    StepConfig(
                        name="Deploy with Docker Compose",
                        command=f"docker-compose -f {deploy_config.docker_compose_file} up -d",
                    )
                )

            if deploy_config.health_check and deploy_config.health_check_url:
                deploy_steps.append(
                    StepConfig(
                        name="Health check",
                        command=f"curl -f {deploy_config.health_check_url} || exit 1",
                    )
                )

            deploy_job = JobConfig(
                name=f"deploy-{deploy_config.environment.value}",
                runs_on="ubuntu-latest",
                steps=deploy_steps,
                needs=["build"],
                condition=f"github.ref == 'refs/heads/{deploy_config.environment.value}'",
            )

            jobs.append(deploy_job)

        return PipelineConfig(
            name=f"{app_name} CI/CD",
            platform=CIPlatform.GITHUB_ACTIONS,
            triggers=[TriggerEvent.PUSH, TriggerEvent.PULL_REQUEST],
            branches=["main", "develop"],
            jobs=jobs,
        )

    def _build_github_triggers(self, config: PipelineConfig) -> Dict[str, Any]:
        """构建GitHub Actions触发器"""
        triggers = {}

        for trigger in config.triggers:
            if trigger == TriggerEvent.PUSH:
                triggers["push"] = {"branches": config.branches}
                if config.paths:
                    triggers["push"]["paths"] = config.paths

            elif trigger == TriggerEvent.PULL_REQUEST:
                triggers["pull_request"] = {"branches": config.branches}
                if config.paths:
                    triggers["pull_request"]["paths"] = config.paths

            elif trigger == TriggerEvent.SCHEDULE and config.schedule:
                triggers["schedule"] = [{"cron": config.schedule}]

            elif trigger == TriggerEvent.MANUAL:
                triggers["workflow_dispatch"] = {}

        return triggers

    def _build_github_steps(self, steps: List[StepConfig]) -> List[Dict[str, Any]]:
        """构建GitHub Actions步骤"""
        github_steps = []

        for step in steps:
            github_step = {"name": step.name}

            if step.uses:
                github_step["uses"] = step.uses
                if step.with_params:
                    github_step["with"] = step.with_params

            elif step.command:
                github_step["run"] = step.command

            elif step.script:
                github_step["run"] = "\n".join(step.script)

            if step.env:
                github_step["env"] = step.env

            if step.condition:
                github_step["if"] = step.condition

            if step.timeout:
                github_step["timeout-minutes"] = step.timeout

            github_steps.append(github_step)

        return github_steps

    def _build_gitlab_script(self, steps: List[StepConfig]) -> List[str]:
        """构建GitLab CI脚本"""
        script = []

        for step in steps:
            if step.command:
                script.append(step.command)
            elif step.script:
                script.extend(step.script)

        return script

    def save_github_workflow(
        self, config: PipelineConfig, filename: Optional[str] = None
    ) -> None:
        """保存GitHub Actions工作流"""
        if not filename:
            filename = f"{config.name.lower().replace(' ', '-')}.yml"

        workflow = self.generate_github_actions(config)
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(workflow, f, default_flow_style=False, allow_unicode=True)

    def save_gitlab_ci(
        self, config: PipelineConfig, filename: str = ".gitlab-ci.yml"
    ) -> None:
        """保存GitLab CI配置"""
        gitlab_ci = self.generate_gitlab_ci(config)
        filepath = Path(filename)

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(gitlab_ci, f, default_flow_style=False, allow_unicode=True)

    def save_jenkins_pipeline(
        self, config: PipelineConfig, filename: str = "Jenkinsfile"
    ) -> None:
        """保存Jenkins Pipeline配置"""
        pipeline = self.generate_jenkins_pipeline(config)
        filepath = Path(filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(pipeline)


# 全局生成器
cicd_generator = CICDGenerator()


# 便捷函数
def generate_github_actions(config: PipelineConfig) -> Dict[str, Any]:
    """生成GitHub Actions配置"""
    return cicd_generator.generate_github_actions(config)


def generate_gitlab_ci(config: PipelineConfig) -> Dict[str, Any]:
    """生成GitLab CI配置"""
    return cicd_generator.generate_gitlab_ci(config)


def generate_build_pipeline(
    app_name: str,
    docker_config: DockerConfig,
    test_config: TestConfig,
    deploy_configs: List[DeploymentConfig],
) -> PipelineConfig:
    """生成标准构建流水线"""
    return cicd_generator.generate_build_pipeline(
        app_name, docker_config, test_config, deploy_configs
    )


# 使用示例
if __name__ == "__main__":
    # 创建生成器
    generator = CICDGenerator()

    # Docker配置
    docker_config = DockerConfig(repository="my-python-app", tag_strategy="commit")

    # 测试配置
    test_config = TestConfig(
        unit_tests=True,
        integration_tests=True,
        coverage=True,
        linting=True,
        type_checking=True,
    )

    # 部署配置
    deploy_configs = [
        DeploymentConfig(
            environment=Environment.STAGING, kubernetes=True, k8s_namespace="staging"
        ),
        DeploymentConfig(
            environment=Environment.PRODUCTION,
            kubernetes=True,
            k8s_namespace="production",
        ),
    ]

    # 生成流水线配置
    pipeline = generator.generate_build_pipeline(
        "my-python-app", docker_config, test_config, deploy_configs
    )

    # 保存GitHub Actions工作流
    generator.save_github_workflow(pipeline)

    # 保存GitLab CI配置
    generator.save_gitlab_ci(pipeline)

    # 保存Jenkins Pipeline
    generator.save_jenkins_pipeline(pipeline)

    print("CI/CD configurations generated successfully!")
