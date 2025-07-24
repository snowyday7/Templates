# -*- coding: utf-8 -*-
"""
移动应用开发模块
提供React Native、Flutter、原生开发、跨平台开发等功能
"""

import os
import json
import subprocess
import shutil
import zipfile
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import tempfile
import yaml
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
from jinja2 import Template, Environment, FileSystemLoader
import base64
import hashlib
import time

# 构建工具
import gradle
from xcode import XcodeProject

# 代码生成
from autopep8 import fix_code
from black import format_str

# 测试工具
import pytest
from appium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 性能分析
import psutil
import memory_profiler

# 配置和日志
import structlog
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

logger = structlog.get_logger(__name__)


class MobilePlatform(Enum):
    """移动平台枚举"""
    IOS = "ios"
    ANDROID = "android"
    WINDOWS_PHONE = "windows_phone"
    WEB = "web"
    DESKTOP = "desktop"


class DevelopmentFramework(Enum):
    """开发框架枚举"""
    REACT_NATIVE = "react_native"
    FLUTTER = "flutter"
    XAMARIN = "xamarin"
    IONIC = "ionic"
    CORDOVA = "cordova"
    NATIVE_IOS = "native_ios"
    NATIVE_ANDROID = "native_android"
    UNITY = "unity"
    KIVY = "kivy"
    QT = "qt"


class BuildType(Enum):
    """构建类型枚举"""
    DEBUG = "debug"
    RELEASE = "release"
    STAGING = "staging"
    PRODUCTION = "production"


class DeviceType(Enum):
    """设备类型枚举"""
    PHONE = "phone"
    TABLET = "tablet"
    WATCH = "watch"
    TV = "tv"
    DESKTOP = "desktop"
    WEB = "web"


class TestType(Enum):
    """测试类型枚举"""
    UNIT = "unit"
    INTEGRATION = "integration"
    UI = "ui"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ACCESSIBILITY = "accessibility"


@dataclass
class ProjectConfig:
    """项目配置"""
    name: str
    package_name: str
    version: str
    framework: DevelopmentFramework
    platforms: List[MobilePlatform]
    description: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    repository: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    dev_dependencies: Dict[str, str] = field(default_factory=dict)
    build_settings: Dict[str, Any] = field(default_factory=dict)
    assets: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)


@dataclass
class BuildConfig:
    """构建配置"""
    build_type: BuildType
    platform: MobilePlatform
    output_path: str
    signing_config: Optional[Dict[str, str]] = None
    optimization: bool = True
    minify: bool = False
    obfuscate: bool = False
    bundle_id: Optional[str] = None
    version_code: Optional[int] = None
    target_sdk: Optional[str] = None
    min_sdk: Optional[str] = None
    architectures: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeviceInfo:
    """设备信息"""
    id: str
    name: str
    platform: MobilePlatform
    device_type: DeviceType
    os_version: str
    screen_size: Tuple[int, int]
    density: float
    is_emulator: bool = False
    is_connected: bool = False
    capabilities: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """测试结果"""
    test_type: TestType
    test_name: str
    status: str  # passed, failed, skipped
    duration: float
    error_message: Optional[str] = None
    screenshot_path: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    app_launch_time: float
    memory_usage: float
    cpu_usage: float
    battery_usage: float
    network_usage: float
    fps: float
    render_time: float
    response_time: float
    crash_count: int = 0
    anr_count: int = 0  # Android Not Responding


class MobileProjectGenerator(ABC):
    """移动项目生成器抽象基类"""
    
    @abstractmethod
    def create_project(self, config: ProjectConfig, output_path: str) -> bool:
        """创建项目"""
        pass
    
    @abstractmethod
    def add_dependency(self, project_path: str, dependency: str, version: str) -> bool:
        """添加依赖"""
        pass
    
    @abstractmethod
    def generate_component(self, project_path: str, component_name: str, 
                          component_type: str) -> bool:
        """生成组件"""
        pass


class ReactNativeGenerator(MobileProjectGenerator):
    """React Native项目生成器"""
    
    def __init__(self):
        self.templates_path = Path(__file__).parent / "templates" / "react_native"
    
    def create_project(self, config: ProjectConfig, output_path: str) -> bool:
        """创建React Native项目"""
        try:
            # 检查React Native CLI
            if not self._check_react_native_cli():
                logger.error("React Native CLI未安装")
                return False
            
            # 创建项目
            cmd = [
                "npx", "react-native", "init", config.name,
                "--directory", output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"创建React Native项目失败: {result.stderr}")
                return False
            
            # 更新package.json
            self._update_package_json(output_path, config)
            
            # 生成配置文件
            self._generate_config_files(output_path, config)
            
            # 添加依赖
            for dep, version in config.dependencies.items():
                self.add_dependency(output_path, dep, version)
            
            logger.info(f"React Native项目创建成功: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"创建React Native项目失败: {e}")
            return False
    
    def add_dependency(self, project_path: str, dependency: str, version: str) -> bool:
        """添加依赖"""
        try:
            cmd = ["npm", "install", f"{dependency}@{version}"]
            result = subprocess.run(cmd, cwd=project_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"依赖已添加: {dependency}@{version}")
                return True
            else:
                logger.error(f"添加依赖失败: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"添加依赖失败: {e}")
            return False
    
    def generate_component(self, project_path: str, component_name: str, 
                          component_type: str) -> bool:
        """生成组件"""
        try:
            # 组件模板
            if component_type == "screen":
                template_content = self._get_screen_template()
            elif component_type == "component":
                template_content = self._get_component_template()
            else:
                logger.error(f"不支持的组件类型: {component_type}")
                return False
            
            # 渲染模板
            template = Template(template_content)
            content = template.render(component_name=component_name)
            
            # 创建文件
            component_dir = Path(project_path) / "src" / "components"
            component_dir.mkdir(parents=True, exist_ok=True)
            
            component_file = component_dir / f"{component_name}.tsx"
            component_file.write_text(content)
            
            logger.info(f"组件已生成: {component_file}")
            return True
        
        except Exception as e:
            logger.error(f"生成组件失败: {e}")
            return False
    
    def _check_react_native_cli(self) -> bool:
        """检查React Native CLI"""
        try:
            result = subprocess.run(["npx", "react-native", "--version"], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _update_package_json(self, project_path: str, config: ProjectConfig):
        """更新package.json"""
        package_json_path = Path(project_path) / "package.json"
        
        if package_json_path.exists():
            with open(package_json_path, 'r') as f:
                package_data = json.load(f)
            
            package_data.update({
                "description": config.description,
                "author": config.author,
                "license": config.license,
                "repository": config.repository,
                "keywords": config.keywords
            })
            
            with open(package_json_path, 'w') as f:
                json.dump(package_data, f, indent=2)
    
    def _generate_config_files(self, project_path: str, config: ProjectConfig):
        """生成配置文件"""
        # Metro配置
        metro_config = """
const {getDefaultConfig} = require('metro-config');

module.exports = (async () => {
  const {
    resolver: {sourceExts, assetExts},
  } = await getDefaultConfig();
  return {
    transformer: {
      babelTransformerPath: require.resolve('react-native-svg-transformer'),
    },
    resolver: {
      assetExts: assetExts.filter(ext => ext !== 'svg'),
      sourceExts: [...sourceExts, 'svg'],
    },
  };
})();
"""
        
        metro_config_path = Path(project_path) / "metro.config.js"
        metro_config_path.write_text(metro_config)
    
    def _get_screen_template(self) -> str:
        """获取屏幕模板"""
        return """
import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
} from 'react-native';

interface {{ component_name }}Props {
  // 定义props类型
}

const {{ component_name }}: React.FC<{{ component_name }}Props> = () => {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>{{ component_name }}</Text>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
});

export default {{ component_name }};
"""
    
    def _get_component_template(self) -> str:
        """获取组件模板"""
        return """
import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
} from 'react-native';

interface {{ component_name }}Props {
  title?: string;
  onPress?: () => void;
}

const {{ component_name }}: React.FC<{{ component_name }}Props> = ({
  title = '{{ component_name }}',
  onPress,
}) => {
  return (
    <TouchableOpacity style={styles.container} onPress={onPress}>
      <Text style={styles.title}>{title}</Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 16,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    margin: 8,
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    textAlign: 'center',
  },
});

export default {{ component_name }};
"""


class FlutterGenerator(MobileProjectGenerator):
    """Flutter项目生成器"""
    
    def __init__(self):
        self.templates_path = Path(__file__).parent / "templates" / "flutter"
    
    def create_project(self, config: ProjectConfig, output_path: str) -> bool:
        """创建Flutter项目"""
        try:
            # 检查Flutter CLI
            if not self._check_flutter_cli():
                logger.error("Flutter CLI未安装")
                return False
            
            # 创建项目
            cmd = [
                "flutter", "create",
                "--project-name", config.name.lower().replace('-', '_'),
                "--org", config.package_name.split('.')[0] + '.' + config.package_name.split('.')[1],
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"创建Flutter项目失败: {result.stderr}")
                return False
            
            # 更新pubspec.yaml
            self._update_pubspec_yaml(output_path, config)
            
            # 添加依赖
            for dep, version in config.dependencies.items():
                self.add_dependency(output_path, dep, version)
            
            logger.info(f"Flutter项目创建成功: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"创建Flutter项目失败: {e}")
            return False
    
    def add_dependency(self, project_path: str, dependency: str, version: str) -> bool:
        """添加依赖"""
        try:
            cmd = ["flutter", "pub", "add", f"{dependency}:{version}"]
            result = subprocess.run(cmd, cwd=project_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"依赖已添加: {dependency}:{version}")
                return True
            else:
                logger.error(f"添加依赖失败: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"添加依赖失败: {e}")
            return False
    
    def generate_component(self, project_path: str, component_name: str, 
                          component_type: str) -> bool:
        """生成组件"""
        try:
            # 组件模板
            if component_type == "screen":
                template_content = self._get_screen_template()
            elif component_type == "widget":
                template_content = self._get_widget_template()
            else:
                logger.error(f"不支持的组件类型: {component_type}")
                return False
            
            # 渲染模板
            template = Template(template_content)
            content = template.render(
                component_name=component_name,
                class_name=self._to_pascal_case(component_name)
            )
            
            # 创建文件
            lib_dir = Path(project_path) / "lib" / "widgets"
            lib_dir.mkdir(parents=True, exist_ok=True)
            
            component_file = lib_dir / f"{component_name.lower()}.dart"
            component_file.write_text(content)
            
            logger.info(f"组件已生成: {component_file}")
            return True
        
        except Exception as e:
            logger.error(f"生成组件失败: {e}")
            return False
    
    def _check_flutter_cli(self) -> bool:
        """检查Flutter CLI"""
        try:
            result = subprocess.run(["flutter", "--version"], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _update_pubspec_yaml(self, project_path: str, config: ProjectConfig):
        """更新pubspec.yaml"""
        pubspec_path = Path(project_path) / "pubspec.yaml"
        
        if pubspec_path.exists():
            with open(pubspec_path, 'r') as f:
                pubspec_data = yaml.safe_load(f)
            
            pubspec_data.update({
                "description": config.description,
                "version": config.version,
                "author": config.author,
                "homepage": config.repository
            })
            
            with open(pubspec_path, 'w') as f:
                yaml.dump(pubspec_data, f, default_flow_style=False)
    
    def _to_pascal_case(self, text: str) -> str:
        """转换为PascalCase"""
        return ''.join(word.capitalize() for word in text.split('_'))
    
    def _get_screen_template(self) -> str:
        """获取屏幕模板"""
        return """
import 'package:flutter/material.dart';

class {{ class_name }}Screen extends StatefulWidget {
  const {{ class_name }}Screen({Key? key}) : super(key: key);

  @override
  State<{{ class_name }}Screen> createState() => _{{ class_name }}ScreenState();
}

class _{{ class_name }}ScreenState extends State<{{ class_name }}Screen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('{{ class_name }}'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(
              '{{ class_name }} Screen',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
"""
    
    def _get_widget_template(self) -> str:
        """获取组件模板"""
        return """
import 'package:flutter/material.dart';

class {{ class_name }}Widget extends StatelessWidget {
  final String? title;
  final VoidCallback? onTap;

  const {{ class_name }}Widget({
    Key? key,
    this.title,
    this.onTap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(16.0),
        margin: const EdgeInsets.all(8.0),
        decoration: BoxDecoration(
          color: Colors.grey[100],
          borderRadius: BorderRadius.circular(8.0),
          boxShadow: [
            BoxShadow(
              color: Colors.grey.withOpacity(0.3),
              spreadRadius: 1,
              blurRadius: 3,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Text(
          title ?? '{{ class_name }}',
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: Colors.black87,
          ),
          textAlign: TextAlign.center,
        ),
      ),
    );
  }
}
"""


class MobileBuildManager:
    """移动应用构建管理器"""
    
    def __init__(self):
        self.build_cache: Dict[str, Any] = {}
    
    def build_app(self, project_path: str, build_config: BuildConfig) -> bool:
        """构建应用"""
        try:
            # 根据平台选择构建方法
            if build_config.platform == MobilePlatform.ANDROID:
                return self._build_android(project_path, build_config)
            elif build_config.platform == MobilePlatform.IOS:
                return self._build_ios(project_path, build_config)
            else:
                logger.error(f"不支持的平台: {build_config.platform}")
                return False
        
        except Exception as e:
            logger.error(f"构建应用失败: {e}")
            return False
    
    def _build_android(self, project_path: str, build_config: BuildConfig) -> bool:
        """构建Android应用"""
        try:
            # 检测项目类型
            if (Path(project_path) / "android").exists():
                # React Native或Flutter项目
                if (Path(project_path) / "package.json").exists():
                    return self._build_react_native_android(project_path, build_config)
                elif (Path(project_path) / "pubspec.yaml").exists():
                    return self._build_flutter_android(project_path, build_config)
            elif (Path(project_path) / "app").exists():
                # 原生Android项目
                return self._build_native_android(project_path, build_config)
            
            logger.error("无法识别的Android项目结构")
            return False
        
        except Exception as e:
            logger.error(f"构建Android应用失败: {e}")
            return False
    
    def _build_react_native_android(self, project_path: str, build_config: BuildConfig) -> bool:
        """构建React Native Android应用"""
        try:
            # 设置环境变量
            env = os.environ.copy()
            env.update(build_config.environment_variables)
            
            # 构建命令
            if build_config.build_type == BuildType.RELEASE:
                cmd = ["npx", "react-native", "run-android", "--variant=release"]
            else:
                cmd = ["npx", "react-native", "run-android"]
            
            result = subprocess.run(cmd, cwd=project_path, env=env, 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("React Native Android构建成功")
                return True
            else:
                logger.error(f"React Native Android构建失败: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"React Native Android构建失败: {e}")
            return False
    
    def _build_flutter_android(self, project_path: str, build_config: BuildConfig) -> bool:
        """构建Flutter Android应用"""
        try:
            # 构建命令
            cmd = ["flutter", "build", "apk"]
            
            if build_config.build_type == BuildType.RELEASE:
                cmd.append("--release")
            else:
                cmd.append("--debug")
            
            if build_config.target_sdk:
                cmd.extend(["--target-platform", build_config.target_sdk])
            
            result = subprocess.run(cmd, cwd=project_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Flutter Android构建成功")
                return True
            else:
                logger.error(f"Flutter Android构建失败: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"Flutter Android构建失败: {e}")
            return False
    
    def _build_native_android(self, project_path: str, build_config: BuildConfig) -> bool:
        """构建原生Android应用"""
        try:
            # 使用Gradle构建
            gradle_cmd = "./gradlew" if os.name != 'nt' else "gradlew.bat"
            
            if build_config.build_type == BuildType.RELEASE:
                cmd = [gradle_cmd, "assembleRelease"]
            else:
                cmd = [gradle_cmd, "assembleDebug"]
            
            result = subprocess.run(cmd, cwd=project_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("原生Android构建成功")
                return True
            else:
                logger.error(f"原生Android构建失败: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"原生Android构建失败: {e}")
            return False
    
    def _build_ios(self, project_path: str, build_config: BuildConfig) -> bool:
        """构建iOS应用"""
        try:
            # 检测项目类型
            if (Path(project_path) / "ios").exists():
                # React Native或Flutter项目
                if (Path(project_path) / "package.json").exists():
                    return self._build_react_native_ios(project_path, build_config)
                elif (Path(project_path) / "pubspec.yaml").exists():
                    return self._build_flutter_ios(project_path, build_config)
            elif (Path(project_path) / "*.xcodeproj").exists():
                # 原生iOS项目
                return self._build_native_ios(project_path, build_config)
            
            logger.error("无法识别的iOS项目结构")
            return False
        
        except Exception as e:
            logger.error(f"构建iOS应用失败: {e}")
            return False
    
    def _build_react_native_ios(self, project_path: str, build_config: BuildConfig) -> bool:
        """构建React Native iOS应用"""
        try:
            # 安装CocoaPods依赖
            pod_install_cmd = ["cd", "ios", "&&", "pod", "install"]
            subprocess.run(" ".join(pod_install_cmd), shell=True, cwd=project_path)
            
            # 构建命令
            if build_config.build_type == BuildType.RELEASE:
                cmd = ["npx", "react-native", "run-ios", "--configuration=Release"]
            else:
                cmd = ["npx", "react-native", "run-ios"]
            
            result = subprocess.run(cmd, cwd=project_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("React Native iOS构建成功")
                return True
            else:
                logger.error(f"React Native iOS构建失败: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"React Native iOS构建失败: {e}")
            return False
    
    def _build_flutter_ios(self, project_path: str, build_config: BuildConfig) -> bool:
        """构建Flutter iOS应用"""
        try:
            # 构建命令
            cmd = ["flutter", "build", "ios"]
            
            if build_config.build_type == BuildType.RELEASE:
                cmd.append("--release")
            else:
                cmd.append("--debug")
            
            result = subprocess.run(cmd, cwd=project_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Flutter iOS构建成功")
                return True
            else:
                logger.error(f"Flutter iOS构建失败: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"Flutter iOS构建失败: {e}")
            return False
    
    def _build_native_ios(self, project_path: str, build_config: BuildConfig) -> bool:
        """构建原生iOS应用"""
        try:
            # 使用xcodebuild构建
            cmd = [
                "xcodebuild",
                "-workspace", "*.xcworkspace",
                "-scheme", "YourApp",
                "-configuration", "Release" if build_config.build_type == BuildType.RELEASE else "Debug",
                "build"
            ]
            
            result = subprocess.run(cmd, cwd=project_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("原生iOS构建成功")
                return True
            else:
                logger.error(f"原生iOS构建失败: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"原生iOS构建失败: {e}")
            return False
    
    def clean_build(self, project_path: str, platform: MobilePlatform) -> bool:
        """清理构建"""
        try:
            if platform == MobilePlatform.ANDROID:
                # 清理Android构建
                gradle_cmd = "./gradlew" if os.name != 'nt' else "gradlew.bat"
                cmd = [gradle_cmd, "clean"]
                subprocess.run(cmd, cwd=project_path)
            
            elif platform == MobilePlatform.IOS:
                # 清理iOS构建
                cmd = ["xcodebuild", "clean"]
                subprocess.run(cmd, cwd=project_path)
            
            # 清理Flutter构建
            if (Path(project_path) / "pubspec.yaml").exists():
                cmd = ["flutter", "clean"]
                subprocess.run(cmd, cwd=project_path)
            
            logger.info(f"构建清理完成: {platform}")
            return True
        
        except Exception as e:
            logger.error(f"清理构建失败: {e}")
            return False


class DeviceManager:
    """设备管理器"""
    
    def __init__(self):
        self.connected_devices: Dict[str, DeviceInfo] = {}
    
    def scan_devices(self) -> List[DeviceInfo]:
        """扫描设备"""
        devices = []
        
        # 扫描Android设备
        android_devices = self._scan_android_devices()
        devices.extend(android_devices)
        
        # 扫描iOS设备
        ios_devices = self._scan_ios_devices()
        devices.extend(ios_devices)
        
        # 更新连接设备列表
        for device in devices:
            self.connected_devices[device.id] = device
        
        return devices
    
    def _scan_android_devices(self) -> List[DeviceInfo]:
        """扫描Android设备"""
        devices = []
        
        try:
            # 使用adb命令扫描设备
            result = subprocess.run(["adb", "devices", "-l"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # 跳过标题行
                
                for line in lines:
                    if line.strip() and 'device' in line:
                        parts = line.split()
                        device_id = parts[0]
                        
                        # 获取设备详细信息
                        device_info = self._get_android_device_info(device_id)
                        if device_info:
                            devices.append(device_info)
        
        except Exception as e:
            logger.error(f"扫描Android设备失败: {e}")
        
        return devices
    
    def _scan_ios_devices(self) -> List[DeviceInfo]:
        """扫描iOS设备"""
        devices = []
        
        try:
            # 使用xcrun simctl扫描模拟器
            result = subprocess.run(["xcrun", "simctl", "list", "devices", "--json"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                for runtime, device_list in data['devices'].items():
                    for device in device_list:
                        if device['state'] == 'Booted':
                            device_info = DeviceInfo(
                                id=device['udid'],
                                name=device['name'],
                                platform=MobilePlatform.IOS,
                                device_type=DeviceType.PHONE,
                                os_version=runtime.split('.')[-1],
                                screen_size=(375, 667),  # 默认值
                                density=2.0,
                                is_emulator=True,
                                is_connected=True
                            )
                            devices.append(device_info)
        
        except Exception as e:
            logger.error(f"扫描iOS设备失败: {e}")
        
        return devices
    
    def _get_android_device_info(self, device_id: str) -> Optional[DeviceInfo]:
        """获取Android设备信息"""
        try:
            # 获取设备属性
            props = {}
            result = subprocess.run(["adb", "-s", device_id, "shell", "getprop"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        props[key.strip('[] ')] = value.strip('[] ')
            
            # 获取屏幕尺寸
            screen_result = subprocess.run(
                ["adb", "-s", device_id, "shell", "wm", "size"], 
                capture_output=True, text=True
            )
            
            screen_size = (1080, 1920)  # 默认值
            if screen_result.returncode == 0 and 'Physical size:' in screen_result.stdout:
                size_str = screen_result.stdout.split('Physical size:')[1].strip()
                width, height = map(int, size_str.split('x'))
                screen_size = (width, height)
            
            return DeviceInfo(
                id=device_id,
                name=props.get('ro.product.model', 'Unknown Device'),
                platform=MobilePlatform.ANDROID,
                device_type=DeviceType.PHONE,
                os_version=props.get('ro.build.version.release', 'Unknown'),
                screen_size=screen_size,
                density=float(props.get('ro.sf.lcd_density', '480')) / 160,
                is_emulator='emulator' in device_id,
                is_connected=True
            )
        
        except Exception as e:
            logger.error(f"获取Android设备信息失败: {e}")
            return None
    
    def get_device(self, device_id: str) -> Optional[DeviceInfo]:
        """获取设备信息"""
        return self.connected_devices.get(device_id)
    
    def install_app(self, device_id: str, app_path: str) -> bool:
        """安装应用"""
        try:
            device = self.get_device(device_id)
            if not device:
                logger.error(f"设备未找到: {device_id}")
                return False
            
            if device.platform == MobilePlatform.ANDROID:
                cmd = ["adb", "-s", device_id, "install", app_path]
            elif device.platform == MobilePlatform.IOS:
                cmd = ["xcrun", "simctl", "install", device_id, app_path]
            else:
                logger.error(f"不支持的平台: {device.platform}")
                return False
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"应用安装成功: {app_path}")
                return True
            else:
                logger.error(f"应用安装失败: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"安装应用失败: {e}")
            return False
    
    def uninstall_app(self, device_id: str, package_name: str) -> bool:
        """卸载应用"""
        try:
            device = self.get_device(device_id)
            if not device:
                logger.error(f"设备未找到: {device_id}")
                return False
            
            if device.platform == MobilePlatform.ANDROID:
                cmd = ["adb", "-s", device_id, "uninstall", package_name]
            elif device.platform == MobilePlatform.IOS:
                cmd = ["xcrun", "simctl", "uninstall", device_id, package_name]
            else:
                logger.error(f"不支持的平台: {device.platform}")
                return False
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"应用卸载成功: {package_name}")
                return True
            else:
                logger.error(f"应用卸载失败: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"卸载应用失败: {e}")
            return False


class MobileTestManager:
    """移动测试管理器"""
    
    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager
        self.test_results: List[TestResult] = []
        self.drivers: Dict[str, webdriver.Remote] = {}
    
    def setup_test_environment(self, device_id: str, app_package: str) -> bool:
        """设置测试环境"""
        try:
            device = self.device_manager.get_device(device_id)
            if not device:
                logger.error(f"设备未找到: {device_id}")
                return False
            
            # 配置Appium capabilities
            if device.platform == MobilePlatform.ANDROID:
                capabilities = {
                    'platformName': 'Android',
                    'platformVersion': device.os_version,
                    'deviceName': device.name,
                    'udid': device_id,
                    'appPackage': app_package,
                    'automationName': 'UiAutomator2'
                }
            elif device.platform == MobilePlatform.IOS:
                capabilities = {
                    'platformName': 'iOS',
                    'platformVersion': device.os_version,
                    'deviceName': device.name,
                    'udid': device_id,
                    'bundleId': app_package,
                    'automationName': 'XCUITest'
                }
            else:
                logger.error(f"不支持的平台: {device.platform}")
                return False
            
            # 创建WebDriver实例
            driver = webdriver.Remote(
                command_executor='http://localhost:4723/wd/hub',
                desired_capabilities=capabilities
            )
            
            self.drivers[device_id] = driver
            logger.info(f"测试环境设置成功: {device_id}")
            return True
        
        except Exception as e:
            logger.error(f"设置测试环境失败: {e}")
            return False
    
    def run_ui_test(self, device_id: str, test_script: str) -> TestResult:
        """运行UI测试"""
        start_time = time.time()
        
        try:
            driver = self.drivers.get(device_id)
            if not driver:
                raise Exception(f"测试环境未设置: {device_id}")
            
            # 执行测试脚本
            exec(test_script, {'driver': driver, 'By': By, 'WebDriverWait': WebDriverWait, 'EC': EC})
            
            duration = time.time() - start_time
            
            result = TestResult(
                test_type=TestType.UI,
                test_name="UI Test",
                status="passed",
                duration=duration
            )
            
            self.test_results.append(result)
            logger.info(f"UI测试通过: {device_id}")
            return result
        
        except Exception as e:
            duration = time.time() - start_time
            
            result = TestResult(
                test_type=TestType.UI,
                test_name="UI Test",
                status="failed",
                duration=duration,
                error_message=str(e)
            )
            
            self.test_results.append(result)
            logger.error(f"UI测试失败: {e}")
            return result
    
    def run_performance_test(self, device_id: str, duration: int = 60) -> PerformanceMetrics:
        """运行性能测试"""
        try:
            device = self.device_manager.get_device(device_id)
            if not device:
                raise Exception(f"设备未找到: {device_id}")
            
            # 收集性能指标
            metrics = PerformanceMetrics(
                app_launch_time=self._measure_app_launch_time(device_id),
                memory_usage=self._measure_memory_usage(device_id),
                cpu_usage=self._measure_cpu_usage(device_id),
                battery_usage=self._measure_battery_usage(device_id),
                network_usage=self._measure_network_usage(device_id),
                fps=self._measure_fps(device_id),
                render_time=self._measure_render_time(device_id),
                response_time=self._measure_response_time(device_id)
            )
            
            logger.info(f"性能测试完成: {device_id}")
            return metrics
        
        except Exception as e:
            logger.error(f"性能测试失败: {e}")
            return PerformanceMetrics(
                app_launch_time=0, memory_usage=0, cpu_usage=0,
                battery_usage=0, network_usage=0, fps=0,
                render_time=0, response_time=0
            )
    
    def _measure_app_launch_time(self, device_id: str) -> float:
        """测量应用启动时间"""
        # 实现应用启动时间测量逻辑
        return 2.5  # 示例值
    
    def _measure_memory_usage(self, device_id: str) -> float:
        """测量内存使用"""
        # 实现内存使用测量逻辑
        return 150.0  # MB
    
    def _measure_cpu_usage(self, device_id: str) -> float:
        """测量CPU使用"""
        # 实现CPU使用测量逻辑
        return 25.0  # 百分比
    
    def _measure_battery_usage(self, device_id: str) -> float:
        """测量电池使用"""
        # 实现电池使用测量逻辑
        return 5.0  # 百分比
    
    def _measure_network_usage(self, device_id: str) -> float:
        """测量网络使用"""
        # 实现网络使用测量逻辑
        return 10.0  # MB
    
    def _measure_fps(self, device_id: str) -> float:
        """测量帧率"""
        # 实现帧率测量逻辑
        return 60.0  # FPS
    
    def _measure_render_time(self, device_id: str) -> float:
        """测量渲染时间"""
        # 实现渲染时间测量逻辑
        return 16.7  # 毫秒
    
    def _measure_response_time(self, device_id: str) -> float:
        """测量响应时间"""
        # 实现响应时间测量逻辑
        return 100.0  # 毫秒
    
    def generate_test_report(self, output_path: str) -> bool:
        """生成测试报告"""
        try:
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(self.test_results),
                'passed_tests': len([r for r in self.test_results if r.status == 'passed']),
                'failed_tests': len([r for r in self.test_results if r.status == 'failed']),
                'test_results': [{
                    'test_type': result.test_type.value,
                    'test_name': result.test_name,
                    'status': result.status,
                    'duration': result.duration,
                    'error_message': result.error_message,
                    'timestamp': result.timestamp.isoformat()
                } for result in self.test_results]
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"测试报告已生成: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"生成测试报告失败: {e}")
            return False
    
    def cleanup(self):
        """清理测试环境"""
        for device_id, driver in self.drivers.items():
            try:
                driver.quit()
                logger.info(f"测试环境已清理: {device_id}")
            except Exception as e:
                logger.error(f"清理测试环境失败: {e}")
        
        self.drivers.clear()


class MobileDevelopmentManager:
    """移动开发管理器"""
    
    def __init__(self):
        self.generators = {
            DevelopmentFramework.REACT_NATIVE: ReactNativeGenerator(),
            DevelopmentFramework.FLUTTER: FlutterGenerator()
        }
        self.build_manager = MobileBuildManager()
        self.device_manager = DeviceManager()
        self.test_manager = MobileTestManager(self.device_manager)
        
        logger.info("移动开发管理器已初始化")
    
    def create_project(self, config: ProjectConfig, output_path: str) -> bool:
        """创建项目"""
        generator = self.generators.get(config.framework)
        if not generator:
            logger.error(f"不支持的框架: {config.framework}")
            return False
        
        return generator.create_project(config, output_path)
    
    def build_project(self, project_path: str, build_config: BuildConfig) -> bool:
        """构建项目"""
        return self.build_manager.build_app(project_path, build_config)
    
    def deploy_to_device(self, device_id: str, app_path: str) -> bool:
        """部署到设备"""
        return self.device_manager.install_app(device_id, app_path)
    
    def run_tests(self, device_id: str, app_package: str, test_script: str) -> TestResult:
        """运行测试"""
        # 设置测试环境
        if not self.test_manager.setup_test_environment(device_id, app_package):
            return TestResult(
                test_type=TestType.UI,
                test_name="Setup Test",
                status="failed",
                duration=0,
                error_message="测试环境设置失败"
            )
        
        # 运行测试
        result = self.test_manager.run_ui_test(device_id, test_script)
        
        # 清理环境
        self.test_manager.cleanup()
        
        return result
    
    def get_performance_metrics(self, device_id: str) -> PerformanceMetrics:
        """获取性能指标"""
        return self.test_manager.run_performance_test(device_id)
    
    def list_devices(self) -> List[DeviceInfo]:
        """列出设备"""
        return self.device_manager.scan_devices()
    
    def generate_icon_set(self, source_image: str, output_dir: str, platform: MobilePlatform) -> bool:
        """生成图标集"""
        try:
            # 定义不同平台的图标尺寸
            if platform == MobilePlatform.IOS:
                sizes = [
                    (20, "20pt"), (29, "29pt"), (40, "40pt"), (58, "58pt"),
                    (60, "60pt"), (80, "80pt"), (87, "87pt"), (120, "120pt"),
                    (180, "180pt"), (1024, "1024pt")
                ]
            elif platform == MobilePlatform.ANDROID:
                sizes = [
                    (36, "ldpi"), (48, "mdpi"), (72, "hdpi"), (96, "xhdpi"),
                    (144, "xxhdpi"), (192, "xxxhdpi")
                ]
            else:
                logger.error(f"不支持的平台: {platform}")
                return False
            
            # 打开源图像
            source = Image.open(source_image)
            
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 生成不同尺寸的图标
            for size, name in sizes:
                resized = source.resize((size, size), Image.Resampling.LANCZOS)
                icon_path = output_path / f"icon_{name}.png"
                resized.save(icon_path, "PNG")
            
            logger.info(f"图标集已生成: {output_dir}")
            return True
        
        except Exception as e:
            logger.error(f"生成图标集失败: {e}")
            return False
    
    def generate_splash_screen(self, source_image: str, output_dir: str, 
                             platform: MobilePlatform) -> bool:
        """生成启动屏幕"""
        try:
            # 定义不同平台的启动屏幕尺寸
            if platform == MobilePlatform.IOS:
                sizes = [
                    (640, 960, "iPhone4"),
                    (640, 1136, "iPhone5"),
                    (750, 1334, "iPhone6"),
                    (1242, 2208, "iPhone6Plus"),
                    (1125, 2436, "iPhoneX")
                ]
            elif platform == MobilePlatform.ANDROID:
                sizes = [
                    (320, 480, "ldpi"),
                    (480, 800, "mdpi"),
                    (720, 1280, "hdpi"),
                    (1080, 1920, "xhdpi")
                ]
            else:
                logger.error(f"不支持的平台: {platform}")
                return False
            
            # 打开源图像
            source = Image.open(source_image)
            
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 生成不同尺寸的启动屏幕
            for width, height, name in sizes:
                # 创建背景
                splash = Image.new('RGB', (width, height), color='white')
                
                # 计算居中位置
                source_ratio = source.width / source.height
                target_ratio = width / height
                
                if source_ratio > target_ratio:
                    # 源图像更宽，按高度缩放
                    new_height = height
                    new_width = int(height * source_ratio)
                else:
                    # 源图像更高，按宽度缩放
                    new_width = width
                    new_height = int(width / source_ratio)
                
                resized_source = source.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # 居中粘贴
                x = (width - new_width) // 2
                y = (height - new_height) // 2
                splash.paste(resized_source, (x, y))
                
                splash_path = output_path / f"splash_{name}.png"
                splash.save(splash_path, "PNG")
            
            logger.info(f"启动屏幕已生成: {output_dir}")
            return True
        
        except Exception as e:
            logger.error(f"生成启动屏幕失败: {e}")
            return False


# 示例使用
def example_usage():
    """示例用法"""
    # 创建移动开发管理器
    mobile_manager = MobileDevelopmentManager()
    
    # 项目配置
    project_config = ProjectConfig(
        name="MyMobileApp",
        package_name="com.example.mymobileapp",
        version="1.0.0",
        framework=DevelopmentFramework.REACT_NATIVE,
        platforms=[MobilePlatform.IOS, MobilePlatform.ANDROID],
        description="我的移动应用",
        author="开发者",
        license="MIT",
        dependencies={
            "react-native-vector-icons": "^9.2.0",
            "@react-navigation/native": "^6.1.0"
        }
    )
    
    # 创建项目
    output_path = "/path/to/output"
    success = mobile_manager.create_project(project_config, output_path)
    print(f"项目创建: {'成功' if success else '失败'}")
    
    # 构建配置
    build_config = BuildConfig(
        build_type=BuildType.DEBUG,
        platform=MobilePlatform.ANDROID,
        output_path="/path/to/output/app.apk",
        optimization=False
    )
    
    # 构建项目
    build_success = mobile_manager.build_project(output_path, build_config)
    print(f"项目构建: {'成功' if build_success else '失败'}")
    
    # 扫描设备
    devices = mobile_manager.list_devices()
    print(f"发现设备数量: {len(devices)}")
    
    for device in devices:
        print(f"设备: {device.name} ({device.platform.value})")
    
    # 如果有设备，部署应用
    if devices:
        device = devices[0]
        app_path = "/path/to/app.apk"
        deploy_success = mobile_manager.deploy_to_device(device.id, app_path)
        print(f"应用部署: {'成功' if deploy_success else '失败'}")
        
        # 运行测试
        test_script = """
# 简单的UI测试脚本
try:
    # 等待应用启动
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "com.example.mymobileapp:id/main_button"))
    )
    
    # 点击按钮
    button = driver.find_element(By.ID, "com.example.mymobileapp:id/main_button")
    button.click()
    
    # 验证结果
    result_text = driver.find_element(By.ID, "com.example.mymobileapp:id/result_text")
    assert result_text.text == "按钮已点击"
    
    print("测试通过")
except Exception as e:
    print(f"测试失败: {e}")
    raise
"""
        
        test_result = mobile_manager.run_tests(
            device.id, 
            "com.example.mymobileapp", 
            test_script
        )
        print(f"测试结果: {test_result.status}")
        
        # 获取性能指标
        metrics = mobile_manager.get_performance_metrics(device.id)
        print(f"应用启动时间: {metrics.app_launch_time}秒")
        print(f"内存使用: {metrics.memory_usage}MB")
        print(f"CPU使用: {metrics.cpu_usage}%")
    
    # 生成图标集
    icon_success = mobile_manager.generate_icon_set(
        "/path/to/source_icon.png",
        "/path/to/icons",
        MobilePlatform.IOS
    )
    print(f"图标生成: {'成功' if icon_success else '失败'}")
    
    # 生成启动屏幕
    splash_success = mobile_manager.generate_splash_screen(
        "/path/to/source_splash.png",
        "/path/to/splash",
        MobilePlatform.ANDROID
    )
    print(f"启动屏幕生成: {'成功' if splash_success else '失败'}")


if __name__ == "__main__":
    example_usage()