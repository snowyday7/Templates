# -*- coding: utf-8 -*-
"""
OpenAI客户端模块

提供与OpenAI API的集成功能，支持流式响应和上下文管理。
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
import json
import asyncio
from datetime import datetime

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import openai
import tiktoken
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

# 导入模板库组件
from src.core.logging import get_logger
from src.utils.exceptions import ValidationException

# 导入应用配置
from .config import get_settings

settings = get_settings()
logger = get_logger(__name__)


class OpenAIClient:
    """OpenAI客户端类"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        timeout: int = 60
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # 初始化异步客户端
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout
        )
        
        # 初始化tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # 如果模型不支持，使用默认编码
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"OpenAI客户端初始化完成，模型: {model}")
    
    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.error(f"计算token数量失败: {e}")
            # 粗略估算：1个token约等于4个字符
            return len(text) // 4
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """计算消息列表的总token数量"""
        total_tokens = 0
        for message in messages:
            # 每条消息的基础token开销
            total_tokens += 4
            for key, value in message.items():
                total_tokens += self.count_tokens(str(value))
                if key == "name":
                    total_tokens -= 1
        
        # 对话的基础token开销
        total_tokens += 2
        return total_tokens
    
    def prepare_messages(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]] = None,
        system_prompt: str = None
    ) -> List[Dict[str, str]]:
        """准备发送给OpenAI的消息格式"""
        messages = []
        
        # 添加系统提示
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 添加对话历史
        if conversation_history:
            messages.extend(conversation_history)
        
        # 添加用户消息
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # 检查token限制
        total_tokens = self.count_messages_tokens(messages)
        if total_tokens > self.max_tokens * 0.8:  # 留20%空间给回复
            # 如果超出限制，截断历史消息
            messages = self._truncate_messages(messages, int(self.max_tokens * 0.6))
        
        return messages
    
    def _truncate_messages(self, messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """截断消息以适应token限制"""
        # 保留系统消息和最新的用户消息
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        user_message = messages[-1] if messages and messages[-1]["role"] == "user" else None
        
        # 从最新的消息开始，逐步添加历史消息
        truncated = system_messages.copy()
        if user_message:
            truncated.append(user_message)
        
        # 添加历史消息（从最新开始）
        history_messages = [msg for msg in messages[len(system_messages):-1]]
        history_messages.reverse()
        
        for msg in history_messages:
            temp_messages = truncated[:-1] + [msg] + [truncated[-1]]
            if self.count_messages_tokens(temp_messages) <= max_tokens:
                truncated.insert(-1, msg)
            else:
                break
        
        return truncated
    
    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> ChatCompletion:
        """创建聊天完成"""
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": stream,
                **kwargs
            }
            
            logger.debug(f"发送OpenAI请求: {json.dumps(params, ensure_ascii=False, indent=2)}")
            
            response = await self.client.chat.completions.create(**params)
            
            if not stream:
                logger.info(f"OpenAI响应完成，使用token: {response.usage.total_tokens if response.usage else 'unknown'}")
            
            return response
            
        except openai.RateLimitError as e:
            logger.error(f"OpenAI API限流: {e}")
            raise ValidationException("API请求过于频繁，请稍后再试")
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI API认证失败: {e}")
            raise ValidationException("API密钥无效")
        except openai.APIError as e:
            logger.error(f"OpenAI API错误: {e}")
            raise ValidationException(f"AI服务暂时不可用: {str(e)}")
        except Exception as e:
            logger.error(f"OpenAI请求失败: {e}")
            raise ValidationException("AI服务请求失败")
    
    async def create_stream_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """创建流式聊天完成"""
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": True,
                **kwargs
            }
            
            logger.debug(f"发送OpenAI流式请求: {json.dumps(params, ensure_ascii=False, indent=2)}")
            
            stream = await self.client.chat.completions.create(**params)
            
            async for chunk in stream:
                yield chunk
                
        except Exception as e:
            logger.error(f"OpenAI流式请求失败: {e}")
            raise ValidationException("AI服务流式请求失败")
    
    async def chat(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]] = None,
        system_prompt: str = None,
        stream: bool = False
    ) -> ChatCompletion:
        """简化的聊天接口"""
        messages = self.prepare_messages(
            user_message=user_message,
            conversation_history=conversation_history,
            system_prompt=system_prompt
        )
        
        return await self.create_completion(messages, stream=stream)
    
    async def chat_stream(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]] = None,
        system_prompt: str = None
    ) -> AsyncGenerator[str, None]:
        """流式聊天接口"""
        messages = self.prepare_messages(
            user_message=user_message,
            conversation_history=conversation_history,
            system_prompt=system_prompt
        )
        
        full_response = ""
        async for chunk in self.create_stream_completion(messages):
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content
        
        logger.info(f"流式响应完成，总长度: {len(full_response)}")
    
    async def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        try:
            models = await self.client.models.list()
            return [model.id for model in models.data if "gpt" in model.id]
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 发送一个简单的测试请求
            response = await self.create_completion(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return response is not None
        except Exception as e:
            logger.error(f"OpenAI健康检查失败: {e}")
            return False


# 全局OpenAI客户端实例
_openai_client: Optional[OpenAIClient] = None


def get_openai_client() -> OpenAIClient:
    """获取OpenAI客户端实例"""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            max_tokens=settings.openai_max_tokens,
            temperature=settings.openai_temperature
        )
    return _openai_client


# 默认系统提示
DEFAULT_SYSTEM_PROMPT = """
你是一个有用的AI助手。请以友好、专业的方式回答用户的问题。
如果你不确定答案，请诚实地说明。
请用中文回答，除非用户明确要求使用其他语言。
"""