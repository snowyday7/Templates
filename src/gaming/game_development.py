# -*- coding: utf-8 -*-
"""
游戏开发模块
提供Unity、Pygame、Godot、Unreal Engine等游戏引擎集成和游戏开发功能
"""

import os
import json
import subprocess
import shutil
import zipfile
import math
import random
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 游戏开发库
import pygame
import pymunk
import pymunk.pygame_util
from panda3d.core import *
from panda3d.direct.showbase.ShowBase import ShowBase
from panda3d.direct.task import Task

# 音频处理
import pygame.mixer
from pydub import AudioSegment
from pydub.playback import play

# 图形和动画
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 网络
import socket
import websockets
import asyncio
from aiohttp import web, ClientSession

# 数据库
import sqlite3
import aiosqlite
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 配置和日志
import structlog
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

# AI和机器学习
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

logger = structlog.get_logger(__name__)


class GameEngine(Enum):
    """游戏引擎枚举"""
    UNITY = "unity"
    UNREAL_ENGINE = "unreal_engine"
    GODOT = "godot"
    PYGAME = "pygame"
    PANDA3D = "panda3d"
    COCOS2D = "cocos2d"
    CONSTRUCT3 = "construct3"
    GAMEMAKER = "gamemaker"
    DEFOLD = "defold"
    CUSTOM = "custom"


class GameGenre(Enum):
    """游戏类型枚举"""
    ACTION = "action"
    ADVENTURE = "adventure"
    RPG = "rpg"
    STRATEGY = "strategy"
    SIMULATION = "simulation"
    PUZZLE = "puzzle"
    RACING = "racing"
    SPORTS = "sports"
    FIGHTING = "fighting"
    SHOOTER = "shooter"
    PLATFORMER = "platformer"
    SURVIVAL = "survival"
    HORROR = "horror"
    EDUCATIONAL = "educational"
    CASUAL = "casual"
    MMO = "mmo"
    MOBILE = "mobile"
    VR = "vr"
    AR = "ar"


class GamePlatform(Enum):
    """游戏平台枚举"""
    PC_WINDOWS = "pc_windows"
    PC_MAC = "pc_mac"
    PC_LINUX = "pc_linux"
    MOBILE_IOS = "mobile_ios"
    MOBILE_ANDROID = "mobile_android"
    CONSOLE_PS5 = "console_ps5"
    CONSOLE_XBOX = "console_xbox"
    CONSOLE_SWITCH = "console_switch"
    WEB_BROWSER = "web_browser"
    VR_OCULUS = "vr_oculus"
    VR_STEAMVR = "vr_steamvr"
    STEAM = "steam"
    EPIC_GAMES = "epic_games"
    GOG = "gog"


class AssetType(Enum):
    """资源类型枚举"""
    TEXTURE = "texture"
    MODEL = "model"
    ANIMATION = "animation"
    AUDIO = "audio"
    MUSIC = "music"
    SCRIPT = "script"
    SHADER = "shader"
    MATERIAL = "material"
    PREFAB = "prefab"
    SCENE = "scene"
    FONT = "font"
    VIDEO = "video"
    PARTICLE = "particle"
    UI = "ui"


class GameState(Enum):
    """游戏状态枚举"""
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    LOADING = "loading"
    SETTINGS = "settings"
    INVENTORY = "inventory"
    DIALOGUE = "dialogue"
    CUTSCENE = "cutscene"


@dataclass
class Vector2D:
    """2D向量"""
    x: float = 0.0
    y: float = 0.0
    
    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def normalize(self) -> 'Vector2D':
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)
    
    def distance_to(self, other: 'Vector2D') -> float:
        return (self - other).magnitude()


@dataclass
class Vector3D:
    """3D向量"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
    
    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def dot(self, other: 'Vector3D') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z


@dataclass
class Transform:
    """变换组件"""
    position: Vector3D = field(default_factory=Vector3D)
    rotation: Vector3D = field(default_factory=Vector3D)  # 欧拉角
    scale: Vector3D = field(default_factory=lambda: Vector3D(1, 1, 1))


@dataclass
class GameConfig:
    """游戏配置"""
    title: str
    version: str
    engine: GameEngine
    genre: GameGenre
    platforms: List[GamePlatform]
    screen_width: int = 1920
    screen_height: int = 1080
    fps: int = 60
    fullscreen: bool = False
    vsync: bool = True
    audio_enabled: bool = True
    music_volume: float = 0.7
    sfx_volume: float = 0.8
    language: str = "zh-CN"
    debug_mode: bool = False
    auto_save: bool = True
    save_interval: int = 300  # 秒
    max_save_files: int = 10
    graphics_quality: str = "high"  # low, medium, high, ultra
    anti_aliasing: bool = True
    shadows: bool = True
    post_processing: bool = True


@dataclass
class GameAsset:
    """游戏资源"""
    id: str
    name: str
    asset_type: AssetType
    file_path: str
    size: int  # 字节
    format: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    is_loaded: bool = False
    reference_count: int = 0


@dataclass
class GameObject:
    """游戏对象"""
    id: str
    name: str
    transform: Transform = field(default_factory=Transform)
    components: Dict[str, Any] = field(default_factory=dict)
    children: List['GameObject'] = field(default_factory=list)
    parent: Optional['GameObject'] = None
    active: bool = True
    layer: int = 0
    tags: List[str] = field(default_factory=list)
    
    def add_component(self, component_type: str, component: Any):
        """添加组件"""
        self.components[component_type] = component
    
    def get_component(self, component_type: str) -> Optional[Any]:
        """获取组件"""
        return self.components.get(component_type)
    
    def remove_component(self, component_type: str) -> bool:
        """移除组件"""
        if component_type in self.components:
            del self.components[component_type]
            return True
        return False
    
    def add_child(self, child: 'GameObject'):
        """添加子对象"""
        child.parent = self
        self.children.append(child)
    
    def remove_child(self, child: 'GameObject') -> bool:
        """移除子对象"""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
            return True
        return False


@dataclass
class Scene:
    """场景"""
    name: str
    game_objects: List[GameObject] = field(default_factory=list)
    cameras: List[GameObject] = field(default_factory=list)
    lights: List[GameObject] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    physics_settings: Dict[str, Any] = field(default_factory=dict)
    audio_settings: Dict[str, Any] = field(default_factory=dict)
    
    def add_game_object(self, game_object: GameObject):
        """添加游戏对象"""
        self.game_objects.append(game_object)
    
    def remove_game_object(self, game_object: GameObject) -> bool:
        """移除游戏对象"""
        if game_object in self.game_objects:
            self.game_objects.remove(game_object)
            return True
        return False
    
    def find_game_object(self, name: str) -> Optional[GameObject]:
        """查找游戏对象"""
        for obj in self.game_objects:
            if obj.name == name:
                return obj
        return None


@dataclass
class PlayerStats:
    """玩家统计"""
    level: int = 1
    experience: int = 0
    health: float = 100.0
    max_health: float = 100.0
    mana: float = 50.0
    max_mana: float = 50.0
    strength: int = 10
    defense: int = 10
    agility: int = 10
    intelligence: int = 10
    score: int = 0
    coins: int = 0
    play_time: float = 0.0  # 秒
    deaths: int = 0
    achievements: List[str] = field(default_factory=list)
    inventory: Dict[str, int] = field(default_factory=dict)


class Component(ABC):
    """组件抽象基类"""
    
    def __init__(self, game_object: GameObject):
        self.game_object = game_object
        self.enabled = True
    
    @abstractmethod
    def update(self, delta_time: float):
        """更新组件"""
        pass
    
    def start(self):
        """组件启动"""
        pass
    
    def destroy(self):
        """销毁组件"""
        pass


class RigidBodyComponent(Component):
    """刚体组件"""
    
    def __init__(self, game_object: GameObject, mass: float = 1.0):
        super().__init__(game_object)
        self.mass = mass
        self.velocity = Vector3D()
        self.acceleration = Vector3D()
        self.drag = 0.1
        self.gravity_scale = 1.0
        self.is_kinematic = False
        self.freeze_rotation = False
    
    def update(self, delta_time: float):
        if not self.is_kinematic:
            # 应用重力
            gravity = Vector3D(0, -9.81 * self.gravity_scale, 0)
            self.acceleration = self.acceleration + gravity
            
            # 应用阻力
            drag_force = self.velocity * (-self.drag)
            self.acceleration = self.acceleration + drag_force
            
            # 更新速度和位置
            self.velocity = self.velocity + self.acceleration * delta_time
            self.game_object.transform.position = self.game_object.transform.position + self.velocity * delta_time
            
            # 重置加速度
            self.acceleration = Vector3D()
    
    def add_force(self, force: Vector3D):
        """添加力"""
        self.acceleration = self.acceleration + force / self.mass
    
    def add_impulse(self, impulse: Vector3D):
        """添加冲量"""
        self.velocity = self.velocity + impulse / self.mass


class ColliderComponent(Component):
    """碰撞器组件"""
    
    def __init__(self, game_object: GameObject, size: Vector3D):
        super().__init__(game_object)
        self.size = size
        self.is_trigger = False
        self.collision_callbacks: List[Callable] = []
    
    def update(self, delta_time: float):
        pass
    
    def add_collision_callback(self, callback: Callable):
        """添加碰撞回调"""
        self.collision_callbacks.append(callback)
    
    def on_collision(self, other: 'ColliderComponent'):
        """碰撞事件"""
        for callback in self.collision_callbacks:
            callback(other)
    
    def check_collision(self, other: 'ColliderComponent') -> bool:
        """检查碰撞（简单的AABB碰撞检测）"""
        pos1 = self.game_object.transform.position
        pos2 = other.game_object.transform.position
        
        return (
            abs(pos1.x - pos2.x) < (self.size.x + other.size.x) / 2 and
            abs(pos1.y - pos2.y) < (self.size.y + other.size.y) / 2 and
            abs(pos1.z - pos2.z) < (self.size.z + other.size.z) / 2
        )


class SpriteRendererComponent(Component):
    """精灵渲染器组件"""
    
    def __init__(self, game_object: GameObject, texture_path: str):
        super().__init__(game_object)
        self.texture_path = texture_path
        self.color = (255, 255, 255, 255)  # RGBA
        self.flip_x = False
        self.flip_y = False
        self.sorting_layer = 0
        self.order_in_layer = 0
        self.texture = None
    
    def update(self, delta_time: float):
        pass
    
    def load_texture(self):
        """加载纹理"""
        try:
            self.texture = pygame.image.load(self.texture_path)
            if self.flip_x or self.flip_y:
                self.texture = pygame.transform.flip(self.texture, self.flip_x, self.flip_y)
        except Exception as e:
            logger.error(f"加载纹理失败: {e}")


class AudioSourceComponent(Component):
    """音频源组件"""
    
    def __init__(self, game_object: GameObject, audio_path: str):
        super().__init__(game_object)
        self.audio_path = audio_path
        self.volume = 1.0
        self.pitch = 1.0
        self.loop = False
        self.is_playing = False
        self.is_3d = False
        self.max_distance = 100.0
        self.sound = None
    
    def update(self, delta_time: float):
        pass
    
    def load_audio(self):
        """加载音频"""
        try:
            self.sound = pygame.mixer.Sound(self.audio_path)
        except Exception as e:
            logger.error(f"加载音频失败: {e}")
    
    def play(self):
        """播放音频"""
        if self.sound:
            if self.loop:
                pygame.mixer.Sound.play(self.sound, loops=-1)
            else:
                pygame.mixer.Sound.play(self.sound)
            self.is_playing = True
    
    def stop(self):
        """停止音频"""
        if self.sound:
            pygame.mixer.Sound.stop(self.sound)
            self.is_playing = False
    
    def pause(self):
        """暂停音频"""
        if self.sound:
            pygame.mixer.pause()
    
    def resume(self):
        """恢复音频"""
        if self.sound:
            pygame.mixer.unpause()


class AnimatorComponent(Component):
    """动画器组件"""
    
    def __init__(self, game_object: GameObject):
        super().__init__(game_object)
        self.animations: Dict[str, List[str]] = {}  # 动画名 -> 帧列表
        self.current_animation = ""
        self.current_frame = 0
        self.frame_time = 0.1  # 每帧时间
        self.elapsed_time = 0.0
        self.loop = True
        self.is_playing = False
    
    def update(self, delta_time: float):
        if self.is_playing and self.current_animation:
            self.elapsed_time += delta_time
            
            if self.elapsed_time >= self.frame_time:
                self.elapsed_time = 0.0
                self.current_frame += 1
                
                animation_frames = self.animations.get(self.current_animation, [])
                if self.current_frame >= len(animation_frames):
                    if self.loop:
                        self.current_frame = 0
                    else:
                        self.current_frame = len(animation_frames) - 1
                        self.is_playing = False
    
    def add_animation(self, name: str, frames: List[str]):
        """添加动画"""
        self.animations[name] = frames
    
    def play_animation(self, name: str, loop: bool = True):
        """播放动画"""
        if name in self.animations:
            self.current_animation = name
            self.current_frame = 0
            self.elapsed_time = 0.0
            self.loop = loop
            self.is_playing = True
    
    def stop_animation(self):
        """停止动画"""
        self.is_playing = False
        self.current_frame = 0
        self.elapsed_time = 0.0
    
    def get_current_frame(self) -> Optional[str]:
        """获取当前帧"""
        if self.current_animation and self.current_animation in self.animations:
            frames = self.animations[self.current_animation]
            if 0 <= self.current_frame < len(frames):
                return frames[self.current_frame]
        return None


class AssetManager:
    """资源管理器"""
    
    def __init__(self):
        self.assets: Dict[str, GameAsset] = {}
        self.loaded_assets: Dict[str, Any] = {}
        self.asset_cache_size = 100  # MB
        self.current_cache_size = 0
    
    def register_asset(self, asset: GameAsset):
        """注册资源"""
        self.assets[asset.id] = asset
        logger.info(f"资源已注册: {asset.name} ({asset.id})")
    
    def load_asset(self, asset_id: str) -> Optional[Any]:
        """加载资源"""
        if asset_id in self.loaded_assets:
            asset = self.assets[asset_id]
            asset.reference_count += 1
            return self.loaded_assets[asset_id]
        
        asset = self.assets.get(asset_id)
        if not asset:
            logger.error(f"资源未找到: {asset_id}")
            return None
        
        try:
            # 根据资源类型加载
            if asset.asset_type == AssetType.TEXTURE:
                loaded_asset = pygame.image.load(asset.file_path)
            elif asset.asset_type == AssetType.AUDIO:
                loaded_asset = pygame.mixer.Sound(asset.file_path)
            elif asset.asset_type == AssetType.MUSIC:
                loaded_asset = asset.file_path  # 音乐文件路径
            elif asset.asset_type == AssetType.FONT:
                size = asset.metadata.get('size', 24)
                loaded_asset = pygame.font.Font(asset.file_path, size)
            else:
                # 其他类型的资源
                with open(asset.file_path, 'rb') as f:
                    loaded_asset = f.read()
            
            self.loaded_assets[asset_id] = loaded_asset
            asset.is_loaded = True
            asset.reference_count = 1
            self.current_cache_size += asset.size / (1024 * 1024)  # 转换为MB
            
            logger.info(f"资源已加载: {asset.name}")
            return loaded_asset
        
        except Exception as e:
            logger.error(f"加载资源失败: {e}")
            return None
    
    def unload_asset(self, asset_id: str):
        """卸载资源"""
        if asset_id in self.loaded_assets:
            asset = self.assets[asset_id]
            asset.reference_count -= 1
            
            if asset.reference_count <= 0:
                del self.loaded_assets[asset_id]
                asset.is_loaded = False
                asset.reference_count = 0
                self.current_cache_size -= asset.size / (1024 * 1024)
                logger.info(f"资源已卸载: {asset.name}")
    
    def get_asset(self, asset_id: str) -> Optional[GameAsset]:
        """获取资源信息"""
        return self.assets.get(asset_id)
    
    def list_assets(self, asset_type: Optional[AssetType] = None) -> List[GameAsset]:
        """列出资源"""
        if asset_type:
            return [asset for asset in self.assets.values() if asset.asset_type == asset_type]
        return list(self.assets.values())
    
    def cleanup_cache(self):
        """清理缓存"""
        if self.current_cache_size > self.asset_cache_size:
            # 按引用计数和最后使用时间排序，卸载最少使用的资源
            sorted_assets = sorted(
                [(aid, asset) for aid, asset in self.assets.items() if asset.is_loaded],
                key=lambda x: (x[1].reference_count, x[1].modified_at)
            )
            
            for asset_id, asset in sorted_assets:
                if self.current_cache_size <= self.asset_cache_size * 0.8:
                    break
                if asset.reference_count == 0:
                    self.unload_asset(asset_id)


class InputManager:
    """输入管理器"""
    
    def __init__(self):
        self.keys_pressed = set()
        self.keys_just_pressed = set()
        self.keys_just_released = set()
        self.mouse_position = Vector2D()
        self.mouse_buttons = set()
        self.mouse_just_pressed = set()
        self.mouse_just_released = set()
        self.mouse_wheel_delta = 0
        self.input_bindings: Dict[str, List[int]] = {}
    
    def update(self):
        """更新输入状态"""
        # 清除上一帧的即时输入
        self.keys_just_pressed.clear()
        self.keys_just_released.clear()
        self.mouse_just_pressed.clear()
        self.mouse_just_released.clear()
        self.mouse_wheel_delta = 0
        
        # 处理pygame事件
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                self.keys_pressed.add(event.key)
                self.keys_just_pressed.add(event.key)
            elif event.type == pygame.KEYUP:
                self.keys_pressed.discard(event.key)
                self.keys_just_released.add(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_buttons.add(event.button)
                self.mouse_just_pressed.add(event.button)
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_buttons.discard(event.button)
                self.mouse_just_released.add(event.button)
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_position = Vector2D(event.pos[0], event.pos[1])
            elif event.type == pygame.MOUSEWHEEL:
                self.mouse_wheel_delta = event.y
    
    def is_key_pressed(self, key: int) -> bool:
        """检查按键是否按下"""
        return key in self.keys_pressed
    
    def is_key_just_pressed(self, key: int) -> bool:
        """检查按键是否刚刚按下"""
        return key in self.keys_just_pressed
    
    def is_key_just_released(self, key: int) -> bool:
        """检查按键是否刚刚释放"""
        return key in self.keys_just_released
    
    def is_mouse_button_pressed(self, button: int) -> bool:
        """检查鼠标按键是否按下"""
        return button in self.mouse_buttons
    
    def is_mouse_button_just_pressed(self, button: int) -> bool:
        """检查鼠标按键是否刚刚按下"""
        return button in self.mouse_just_pressed
    
    def is_mouse_button_just_released(self, button: int) -> bool:
        """检查鼠标按键是否刚刚释放"""
        return button in self.mouse_just_released
    
    def bind_input(self, action: str, keys: List[int]):
        """绑定输入"""
        self.input_bindings[action] = keys
    
    def is_action_pressed(self, action: str) -> bool:
        """检查动作是否按下"""
        keys = self.input_bindings.get(action, [])
        return any(self.is_key_pressed(key) for key in keys)
    
    def is_action_just_pressed(self, action: str) -> bool:
        """检查动作是否刚刚按下"""
        keys = self.input_bindings.get(action, [])
        return any(self.is_key_just_pressed(key) for key in keys)


class PhysicsEngine:
    """物理引擎"""
    
    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, -981)  # 重力
        self.bodies: Dict[str, pymunk.Body] = {}
        self.shapes: Dict[str, pymunk.Shape] = {}
        self.collision_handlers: Dict[Tuple[int, int], Callable] = {}
    
    def create_static_body(self, game_object_id: str, position: Vector2D, 
                          vertices: List[Tuple[float, float]]) -> pymunk.Body:
        """创建静态刚体"""
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = position.x, position.y
        
        shape = pymunk.Poly(body, vertices)
        shape.friction = 0.7
        
        self.space.add(body, shape)
        self.bodies[game_object_id] = body
        self.shapes[game_object_id] = shape
        
        return body
    
    def create_dynamic_body(self, game_object_id: str, position: Vector2D, 
                           mass: float, size: Vector2D) -> pymunk.Body:
        """创建动态刚体"""
        moment = pymunk.moment_for_box(mass, (size.x, size.y))
        body = pymunk.Body(mass, moment)
        body.position = position.x, position.y
        
        shape = pymunk.Poly.create_box(body, (size.x, size.y))
        shape.friction = 0.7
        
        self.space.add(body, shape)
        self.bodies[game_object_id] = body
        self.shapes[game_object_id] = shape
        
        return body
    
    def add_collision_handler(self, type_a: int, type_b: int, handler: Callable):
        """添加碰撞处理器"""
        self.collision_handlers[(type_a, type_b)] = handler
        
        def collision_func(arbiter, space, data):
            return handler(arbiter, space, data)
        
        collision_handler = self.space.add_collision_handler(type_a, type_b)
        collision_handler.begin = collision_func
    
    def update(self, delta_time: float):
        """更新物理模拟"""
        self.space.step(delta_time)
    
    def get_body(self, game_object_id: str) -> Optional[pymunk.Body]:
        """获取刚体"""
        return self.bodies.get(game_object_id)
    
    def remove_body(self, game_object_id: str):
        """移除刚体"""
        if game_object_id in self.bodies:
            body = self.bodies[game_object_id]
            shape = self.shapes[game_object_id]
            self.space.remove(body, shape)
            del self.bodies[game_object_id]
            del self.shapes[game_object_id]


class AudioManager:
    """音频管理器"""
    
    def __init__(self):
        pygame.mixer.init()
        self.music_volume = 0.7
        self.sfx_volume = 0.8
        self.current_music = None
        self.sound_effects: Dict[str, pygame.mixer.Sound] = {}
        self.music_playlist: List[str] = []
        self.current_music_index = 0
        self.is_music_playing = False
    
    def load_music(self, file_path: str):
        """加载背景音乐"""
        try:
            pygame.mixer.music.load(file_path)
            self.current_music = file_path
            logger.info(f"音乐已加载: {file_path}")
        except Exception as e:
            logger.error(f"加载音乐失败: {e}")
    
    def play_music(self, loops: int = -1, start: float = 0.0):
        """播放背景音乐"""
        try:
            pygame.mixer.music.play(loops, start)
            pygame.mixer.music.set_volume(self.music_volume)
            self.is_music_playing = True
            logger.info("音乐开始播放")
        except Exception as e:
            logger.error(f"播放音乐失败: {e}")
    
    def stop_music(self):
        """停止背景音乐"""
        pygame.mixer.music.stop()
        self.is_music_playing = False
    
    def pause_music(self):
        """暂停背景音乐"""
        pygame.mixer.music.pause()
    
    def resume_music(self):
        """恢复背景音乐"""
        pygame.mixer.music.unpause()
    
    def set_music_volume(self, volume: float):
        """设置音乐音量"""
        self.music_volume = max(0.0, min(1.0, volume))
        pygame.mixer.music.set_volume(self.music_volume)
    
    def load_sound_effect(self, name: str, file_path: str):
        """加载音效"""
        try:
            sound = pygame.mixer.Sound(file_path)
            self.sound_effects[name] = sound
            logger.info(f"音效已加载: {name}")
        except Exception as e:
            logger.error(f"加载音效失败: {e}")
    
    def play_sound_effect(self, name: str, volume: Optional[float] = None):
        """播放音效"""
        if name in self.sound_effects:
            sound = self.sound_effects[name]
            if volume is not None:
                sound.set_volume(volume)
            else:
                sound.set_volume(self.sfx_volume)
            sound.play()
    
    def set_sfx_volume(self, volume: float):
        """设置音效音量"""
        self.sfx_volume = max(0.0, min(1.0, volume))
    
    def add_to_playlist(self, file_path: str):
        """添加到播放列表"""
        self.music_playlist.append(file_path)
    
    def play_next_in_playlist(self):
        """播放播放列表中的下一首"""
        if self.music_playlist:
            self.current_music_index = (self.current_music_index + 1) % len(self.music_playlist)
            next_music = self.music_playlist[self.current_music_index]
            self.load_music(next_music)
            self.play_music()


class GameStateManager:
    """游戏状态管理器"""
    
    def __init__(self):
        self.current_state = GameState.MENU
        self.previous_state = None
        self.state_stack: List[GameState] = []
        self.state_handlers: Dict[GameState, Callable] = {}
        self.transition_callbacks: Dict[Tuple[GameState, GameState], Callable] = {}
    
    def register_state_handler(self, state: GameState, handler: Callable):
        """注册状态处理器"""
        self.state_handlers[state] = handler
    
    def register_transition_callback(self, from_state: GameState, to_state: GameState, callback: Callable):
        """注册状态转换回调"""
        self.transition_callbacks[(from_state, to_state)] = callback
    
    def change_state(self, new_state: GameState):
        """改变游戏状态"""
        if new_state != self.current_state:
            self.previous_state = self.current_state
            
            # 执行转换回调
            transition_key = (self.current_state, new_state)
            if transition_key in self.transition_callbacks:
                self.transition_callbacks[transition_key]()
            
            self.current_state = new_state
            logger.info(f"游戏状态改变: {self.previous_state} -> {self.current_state}")
    
    def push_state(self, state: GameState):
        """推入状态到栈"""
        self.state_stack.append(self.current_state)
        self.change_state(state)
    
    def pop_state(self):
        """从栈中弹出状态"""
        if self.state_stack:
            previous_state = self.state_stack.pop()
            self.change_state(previous_state)
    
    def update(self, delta_time: float):
        """更新当前状态"""
        if self.current_state in self.state_handlers:
            self.state_handlers[self.current_state](delta_time)
    
    def get_current_state(self) -> GameState:
        """获取当前状态"""
        return self.current_state
    
    def is_state(self, state: GameState) -> bool:
        """检查是否为指定状态"""
        return self.current_state == state


class SaveGameManager:
    """存档管理器"""
    
    def __init__(self, save_directory: str = "saves"):
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(exist_ok=True)
        self.current_save_slot = 0
        self.auto_save_enabled = True
        self.auto_save_interval = 300  # 5分钟
        self.last_auto_save = time.time()
    
    def save_game(self, save_data: Dict[str, Any], slot: int = 0, name: str = "") -> bool:
        """保存游戏"""
        try:
            save_file_name = f"save_{slot:03d}.json" if not name else f"{name}.json"
            save_file_path = self.save_directory / save_file_name
            
            # 添加元数据
            save_data_with_meta = {
                "metadata": {
                    "save_time": datetime.now().isoformat(),
                    "slot": slot,
                    "name": name,
                    "version": "1.0.0"
                },
                "game_data": save_data
            }
            
            with open(save_file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data_with_meta, f, indent=2, ensure_ascii=False)
            
            logger.info(f"游戏已保存: {save_file_path}")
            return True
        
        except Exception as e:
            logger.error(f"保存游戏失败: {e}")
            return False
    
    def load_game(self, slot: int = 0, name: str = "") -> Optional[Dict[str, Any]]:
        """加载游戏"""
        try:
            save_file_name = f"save_{slot:03d}.json" if not name else f"{name}.json"
            save_file_path = self.save_directory / save_file_name
            
            if not save_file_path.exists():
                logger.warning(f"存档文件不存在: {save_file_path}")
                return None
            
            with open(save_file_path, 'r', encoding='utf-8') as f:
                save_data_with_meta = json.load(f)
            
            logger.info(f"游戏已加载: {save_file_path}")
            return save_data_with_meta.get("game_data", {})
        
        except Exception as e:
            logger.error(f"加载游戏失败: {e}")
            return None
    
    def delete_save(self, slot: int = 0, name: str = "") -> bool:
        """删除存档"""
        try:
            save_file_name = f"save_{slot:03d}.json" if not name else f"{name}.json"
            save_file_path = self.save_directory / save_file_name
            
            if save_file_path.exists():
                save_file_path.unlink()
                logger.info(f"存档已删除: {save_file_path}")
                return True
            else:
                logger.warning(f"存档文件不存在: {save_file_path}")
                return False
        
        except Exception as e:
            logger.error(f"删除存档失败: {e}")
            return False
    
    def list_saves(self) -> List[Dict[str, Any]]:
        """列出所有存档"""
        saves = []
        
        for save_file in self.save_directory.glob("*.json"):
            try:
                with open(save_file, 'r', encoding='utf-8') as f:
                    save_data = json.load(f)
                
                metadata = save_data.get("metadata", {})
                saves.append({
                    "file_name": save_file.name,
                    "slot": metadata.get("slot", 0),
                    "name": metadata.get("name", ""),
                    "save_time": metadata.get("save_time", ""),
                    "version": metadata.get("version", "")
                })
            
            except Exception as e:
                logger.error(f"读取存档信息失败: {e}")
        
        return sorted(saves, key=lambda x: x["save_time"], reverse=True)
    
    def auto_save(self, save_data: Dict[str, Any]):
        """自动保存"""
        if self.auto_save_enabled:
            current_time = time.time()
            if current_time - self.last_auto_save >= self.auto_save_interval:
                self.save_game(save_data, slot=999, name="autosave")
                self.last_auto_save = current_time


class GameEngine2D:
    """2D游戏引擎"""
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.running = False
        self.clock = pygame.time.Clock()
        self.screen = None
        self.delta_time = 0.0
        self.total_time = 0.0
        
        # 管理器
        self.asset_manager = AssetManager()
        self.input_manager = InputManager()
        self.audio_manager = AudioManager()
        self.physics_engine = PhysicsEngine()
        self.state_manager = GameStateManager()
        self.save_manager = SaveGameManager()
        
        # 场景和对象
        self.current_scene: Optional[Scene] = None
        self.scenes: Dict[str, Scene] = {}
        
        # 渲染
        self.camera_position = Vector2D()
        self.camera_zoom = 1.0
        
        # 初始化pygame
        self._initialize_pygame()
    
    def _initialize_pygame(self):
        """初始化pygame"""
        pygame.init()
        
        # 设置显示模式
        if self.config.fullscreen:
            self.screen = pygame.display.set_mode(
                (self.config.screen_width, self.config.screen_height),
                pygame.FULLSCREEN
            )
        else:
            self.screen = pygame.display.set_mode(
                (self.config.screen_width, self.config.screen_height)
            )
        
        pygame.display.set_caption(self.config.title)
        
        # 设置音频
        if self.config.audio_enabled:
            self.audio_manager.set_music_volume(self.config.music_volume)
            self.audio_manager.set_sfx_volume(self.config.sfx_volume)
    
    def add_scene(self, name: str, scene: Scene):
        """添加场景"""
        self.scenes[name] = scene
    
    def load_scene(self, name: str):
        """加载场景"""
        if name in self.scenes:
            self.current_scene = self.scenes[name]
            logger.info(f"场景已加载: {name}")
        else:
            logger.error(f"场景未找到: {name}")
    
    def run(self):
        """运行游戏主循环"""
        self.running = True
        last_time = time.time()
        
        logger.info(f"游戏开始运行: {self.config.title}")
        
        while self.running:
            current_time = time.