# ChatGPT Backend API 文档

## 📋 目录

- [API 概览](#api-概览)
- [认证机制](#认证机制)
- [错误处理](#错误处理)
- [认证接口](#认证接口)
- [用户管理](#用户管理)
- [对话管理](#对话管理)
- [消息处理](#消息处理)
- [WebSocket 接口](#websocket-接口)
- [系统接口](#系统接口)
- [SDK 示例](#sdk-示例)

## 🌐 API 概览

### 基础信息

- **Base URL**: `http://localhost:8000/api/v1`
- **协议**: HTTP/HTTPS
- **数据格式**: JSON
- **认证方式**: JWT Bearer Token
- **API 版本**: v1

### 通用响应格式

#### 成功响应
```json
{
  "success": true,
  "data": {},
  "message": "操作成功",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### 错误响应
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "输入数据验证失败",
    "details": {
      "field": "username",
      "issue": "用户名已存在"
    }
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 请求成功 |
| 201 | 创建成功 |
| 400 | 请求参数错误 |
| 401 | 未认证 |
| 403 | 权限不足 |
| 404 | 资源不存在 |
| 422 | 数据验证失败 |
| 429 | 请求频率超限 |
| 500 | 服务器内部错误 |

## 🔐 认证机制

### JWT Token 认证

大部分 API 需要在请求头中包含 JWT Token：

```http
Authorization: Bearer <your_jwt_token>
```

### Token 生命周期

- **Access Token**: 30分钟有效期
- **Refresh Token**: 7天有效期
- **Password Reset Token**: 1小时有效期

## ❌ 错误处理

### 错误代码列表

| 错误代码 | 说明 |
|----------|------|
| `VALIDATION_ERROR` | 数据验证失败 |
| `AUTHENTICATION_ERROR` | 认证失败 |
| `AUTHORIZATION_ERROR` | 权限不足 |
| `RESOURCE_NOT_FOUND` | 资源不存在 |
| `QUOTA_EXCEEDED` | 配额超限 |
| `OPENAI_API_ERROR` | OpenAI API 调用失败 |
| `DATABASE_ERROR` | 数据库操作失败 |
| `RATE_LIMIT_EXCEEDED` | 请求频率超限 |
| `INTERNAL_SERVER_ERROR` | 服务器内部错误 |

## 🔑 认证接口

### 用户注册

**POST** `/auth/register`

注册新用户账户。

#### 请求参数

```json
{
  "username": "testuser",
  "email": "test@example.com",
  "password": "password123",
  "confirm_password": "password123"
}
```

#### 响应示例

```json
{
  "success": true,
  "data": {
    "user": {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "username": "testuser",
      "email": "test@example.com",
      "is_active": true,
      "is_vip": false,
      "created_at": "2024-01-01T12:00:00Z"
    },
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer",
    "expires_in": 1800
  },
  "message": "用户注册成功"
}
```

### 用户登录

**POST** `/auth/login`

用户登录获取访问令牌。

#### 请求参数

```json
{
  "username": "testuser",
  "password": "password123"
}
```

#### 响应示例

```json
{
  "success": true,
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer",
    "expires_in": 1800,
    "user": {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "username": "testuser",
      "email": "test@example.com"
    }
  },
  "message": "登录成功"
}
```

### 刷新令牌

**POST** `/auth/refresh`

使用刷新令牌获取新的访问令牌。

#### 请求参数

```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

### 用户登出

**POST** `/auth/logout`

用户登出，使令牌失效。

**Headers**: `Authorization: Bearer <token>`

### 密码重置请求

**POST** `/auth/forgot-password`

请求密码重置邮件。

#### 请求参数

```json
{
  "email": "test@example.com"
}
```

### 重置密码

**POST** `/auth/reset-password`

使用重置令牌设置新密码。

#### 请求参数

```json
{
  "token": "reset_token_here",
  "new_password": "newpassword123",
  "confirm_password": "newpassword123"
}
```

## 👤 用户管理

### 获取当前用户信息

**GET** `/users/me`

获取当前登录用户的详细信息。

**Headers**: `Authorization: Bearer <token>`

#### 响应示例

```json
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "username": "testuser",
    "email": "test@example.com",
    "is_active": true,
    "is_vip": false,
    "created_at": "2024-01-01T12:00:00Z",
    "updated_at": "2024-01-01T12:00:00Z",
    "quota": {
      "daily_requests": 50,
      "monthly_requests": 1000,
      "daily_tokens": 10000,
      "monthly_tokens": 100000,
      "requests_used_today": 5,
      "tokens_used_today": 1500
    }
  }
}
```

### 更新用户信息

**PUT** `/users/me`

更新当前用户的信息。

**Headers**: `Authorization: Bearer <token>`

#### 请求参数

```json
{
  "email": "newemail@example.com",
  "username": "newusername"
}
```

### 修改密码

**POST** `/users/me/change-password`

修改当前用户密码。

**Headers**: `Authorization: Bearer <token>`

#### 请求参数

```json
{
  "current_password": "oldpassword123",
  "new_password": "newpassword123",
  "confirm_password": "newpassword123"
}
```

### 获取用户配额

**GET** `/users/me/quota`

获取当前用户的配额使用情况。

**Headers**: `Authorization: Bearer <token>`

#### 响应示例

```json
{
  "success": true,
  "data": {
    "daily_limit": {
      "requests": 50,
      "tokens": 10000
    },
    "monthly_limit": {
      "requests": 1000,
      "tokens": 100000
    },
    "usage_today": {
      "requests": 5,
      "tokens": 1500
    },
    "usage_this_month": {
      "requests": 120,
      "tokens": 25000
    },
    "remaining_today": {
      "requests": 45,
      "tokens": 8500
    }
  }
}
```

## 💬 对话管理

### 获取对话列表

**GET** `/conversations`

获取当前用户的对话列表。

**Headers**: `Authorization: Bearer <token>`

#### 查询参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `page` | int | 1 | 页码 |
| `size` | int | 20 | 每页数量 |
| `archived` | bool | false | 是否包含已归档 |
| `search` | str | - | 搜索关键词 |

#### 响应示例

```json
{
  "success": true,
  "data": {
    "conversations": [
      {
        "id": "550e8400-e29b-41d4-a716-446655440001",
        "title": "关于 Python 编程的讨论",
        "model": "gpt-3.5-turbo",
        "message_count": 15,
        "is_archived": false,
        "created_at": "2024-01-01T12:00:00Z",
        "updated_at": "2024-01-01T14:30:00Z",
        "last_message": {
          "content": "谢谢你的解释，我明白了！",
          "role": "user",
          "created_at": "2024-01-01T14:30:00Z"
        }
      }
    ],
    "pagination": {
      "page": 1,
      "size": 20,
      "total": 5,
      "pages": 1
    }
  }
}
```

### 创建新对话

**POST** `/conversations`

创建一个新的对话。

**Headers**: `Authorization: Bearer <token>`

#### 请求参数

```json
{
  "title": "新的对话",
  "model": "gpt-3.5-turbo",
  "system_message": "你是一个有用的助手。" // 可选
}
```

#### 响应示例

```json
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440002",
    "title": "新的对话",
    "model": "gpt-3.5-turbo",
    "message_count": 0,
    "is_archived": false,
    "created_at": "2024-01-01T15:00:00Z",
    "updated_at": "2024-01-01T15:00:00Z"
  },
  "message": "对话创建成功"
}
```

### 获取对话详情

**GET** `/conversations/{conversation_id}`

获取指定对话的详细信息。

**Headers**: `Authorization: Bearer <token>`

#### 响应示例

```json
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440001",
    "title": "关于 Python 编程的讨论",
    "model": "gpt-3.5-turbo",
    "message_count": 15,
    "is_archived": false,
    "created_at": "2024-01-01T12:00:00Z",
    "updated_at": "2024-01-01T14:30:00Z",
    "messages": [
      {
        "id": "550e8400-e29b-41d4-a716-446655440010",
        "role": "user",
        "content": "请解释一下 Python 的装饰器",
        "tokens_used": 12,
        "created_at": "2024-01-01T12:00:00Z"
      },
      {
        "id": "550e8400-e29b-41d4-a716-446655440011",
        "role": "assistant",
        "content": "Python 装饰器是一种设计模式...",
        "tokens_used": 150,
        "created_at": "2024-01-01T12:00:30Z"
      }
    ]
  }
}
```

### 更新对话

**PUT** `/conversations/{conversation_id}`

更新对话信息。

**Headers**: `Authorization: Bearer <token>`

#### 请求参数

```json
{
  "title": "更新后的对话标题",
  "is_archived": false
}
```

### 删除对话

**DELETE** `/conversations/{conversation_id}`

删除指定对话及其所有消息。

**Headers**: `Authorization: Bearer <token>`

### 归档对话

**POST** `/conversations/{conversation_id}/archive`

归档指定对话。

**Headers**: `Authorization: Bearer <token>`

### 取消归档对话

**POST** `/conversations/{conversation_id}/unarchive`

取消归档指定对话。

**Headers**: `Authorization: Bearer <token>`

## 📝 消息处理

### 获取对话消息

**GET** `/conversations/{conversation_id}/messages`

获取指定对话的消息列表。

**Headers**: `Authorization: Bearer <token>`

#### 查询参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `page` | int | 1 | 页码 |
| `size` | int | 50 | 每页数量 |
| `order` | str | desc | 排序方式 (asc/desc) |

### 发送消息

**POST** `/conversations/{conversation_id}/messages`

向指定对话发送消息并获取 AI 回复。

**Headers**: `Authorization: Bearer <token>`

#### 请求参数

```json
{
  "content": "请解释一下机器学习的基本概念",
  "role": "user",
  "model": "gpt-3.5-turbo", // 可选，覆盖对话默认模型
  "temperature": 0.7, // 可选，0-2之间
  "max_tokens": 1000, // 可选，最大回复长度
  "stream": false // 可选，是否流式响应
}
```

#### 响应示例

```json
{
  "success": true,
  "data": {
    "user_message": {
      "id": "550e8400-e29b-41d4-a716-446655440020",
      "role": "user",
      "content": "请解释一下机器学习的基本概念",
      "tokens_used": 15,
      "created_at": "2024-01-01T15:30:00Z"
    },
    "assistant_message": {
      "id": "550e8400-e29b-41d4-a716-446655440021",
      "role": "assistant",
      "content": "机器学习是人工智能的一个分支...",
      "tokens_used": 200,
      "created_at": "2024-01-01T15:30:15Z"
    },
    "usage": {
      "prompt_tokens": 15,
      "completion_tokens": 200,
      "total_tokens": 215
    }
  },
  "message": "消息发送成功"
}
```

### 流式发送消息

**POST** `/conversations/{conversation_id}/messages/stream`

流式发送消息，实时接收 AI 回复。

**Headers**: `Authorization: Bearer <token>`

#### 请求参数

```json
{
  "content": "写一首关于春天的诗",
  "role": "user",
  "model": "gpt-3.5-turbo",
  "temperature": 0.8
}
```

#### 响应格式 (Server-Sent Events)

```
data: {"type": "start", "message_id": "550e8400-e29b-41d4-a716-446655440022"}

data: {"type": "content", "content": "春"}

data: {"type": "content", "content": "天"}

data: {"type": "content", "content": "来"}

data: {"type": "end", "usage": {"total_tokens": 150}}
```

### 删除消息

**DELETE** `/messages/{message_id}`

删除指定消息。

**Headers**: `Authorization: Bearer <token>`

### 编辑消息

**PUT** `/messages/{message_id}`

编辑指定消息内容。

**Headers**: `Authorization: Bearer <token>`

#### 请求参数

```json
{
  "content": "修改后的消息内容"
}
```

## 🔌 WebSocket 接口

### 连接 WebSocket

**WebSocket** `/ws/chat`

建立 WebSocket 连接进行实时通信。

#### 连接参数

```
ws://localhost:8000/api/v1/ws/chat?token=<your_jwt_token>
```

### 消息格式

#### 发送消息

```json
{
  "type": "chat_message",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440001",
  "content": "你好，世界！",
  "model": "gpt-3.5-turbo",
  "temperature": 0.7
}
```

#### 接收消息

```json
{
  "type": "message_start",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440001",
  "message_id": "550e8400-e29b-41d4-a716-446655440030",
  "timestamp": "2024-01-01T16:00:00Z"
}

{
  "type": "message_content",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440001",
  "message_id": "550e8400-e29b-41d4-a716-446655440030",
  "content": "你好！",
  "timestamp": "2024-01-01T16:00:01Z"
}

{
  "type": "message_end",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440001",
  "message_id": "550e8400-e29b-41d4-a716-446655440030",
  "usage": {
    "total_tokens": 25
  },
  "timestamp": "2024-01-01T16:00:05Z"
}
```

#### 错误消息

```json
{
  "type": "error",
  "error": {
    "code": "QUOTA_EXCEEDED",
    "message": "今日配额已用完"
  },
  "timestamp": "2024-01-01T16:00:00Z"
}
```

### WebSocket 事件类型

| 事件类型 | 说明 |
|----------|------|
| `chat_message` | 发送聊天消息 |
| `message_start` | 消息开始 |
| `message_content` | 消息内容片段 |
| `message_end` | 消息结束 |
| `error` | 错误信息 |
| `ping` | 心跳检测 |
| `pong` | 心跳响应 |

## 🔧 系统接口

### 健康检查

**GET** `/health`

检查系统健康状态。

#### 响应示例

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T16:00:00Z",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "openai": "healthy"
  },
  "uptime": 86400
}
```

### 应用信息

**GET** `/info`

获取应用基本信息。

#### 响应示例

```json
{
  "name": "ChatGPT Backend API",
  "version": "1.0.0",
  "environment": "development",
  "supported_models": [
    "gpt-4",
    "gpt-4-turbo",
    "gpt-3.5-turbo"
  ],
  "features": {
    "websocket": true,
    "streaming": true,
    "file_upload": true
  }
}
```

### 获取支持的模型

**GET** `/models`

获取支持的 AI 模型列表。

#### 响应示例

```json
{
  "success": true,
  "data": {
    "models": [
      {
        "id": "gpt-4",
        "name": "GPT-4",
        "description": "最先进的 GPT 模型",
        "max_tokens": 8192,
        "cost_per_1k_tokens": {
          "input": 0.03,
          "output": 0.06
        },
        "capabilities": ["text", "code", "reasoning"]
      },
      {
        "id": "gpt-3.5-turbo",
        "name": "GPT-3.5 Turbo",
        "description": "快速且经济的 GPT 模型",
        "max_tokens": 4096,
        "cost_per_1k_tokens": {
          "input": 0.001,
          "output": 0.002
        },
        "capabilities": ["text", "code"]
      }
    ]
  }
}
```

### 系统指标

**GET** `/metrics`

获取系统运行指标（Prometheus 格式）。

```
# HELP http_requests_total Total number of HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",endpoint="/health"} 1234

# HELP openai_requests_total Total number of OpenAI API requests
# TYPE openai_requests_total counter
openai_requests_total{model="gpt-3.5-turbo",status="success"} 567
```

## 📚 SDK 示例

### Python SDK 示例

```python
import requests
import json
from typing import Optional

class ChatGPTBackendClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}'
            })
    
    def login(self, username: str, password: str) -> dict:
        """用户登录"""
        response = self.session.post(
            f'{self.base_url}/api/v1/auth/login',
            json={'username': username, 'password': password}
        )
        data = response.json()
        if data['success']:
            token = data['data']['access_token']
            self.session.headers.update({
                'Authorization': f'Bearer {token}'
            })
        return data
    
    def create_conversation(self, title: str, model: str = 'gpt-3.5-turbo') -> dict:
        """创建对话"""
        response = self.session.post(
            f'{self.base_url}/api/v1/conversations',
            json={'title': title, 'model': model}
        )
        return response.json()
    
    def send_message(self, conversation_id: str, content: str, **kwargs) -> dict:
        """发送消息"""
        data = {'content': content, 'role': 'user', **kwargs}
        response = self.session.post(
            f'{self.base_url}/api/v1/conversations/{conversation_id}/messages',
            json=data
        )
        return response.json()
    
    def get_conversations(self, page: int = 1, size: int = 20) -> dict:
        """获取对话列表"""
        response = self.session.get(
            f'{self.base_url}/api/v1/conversations',
            params={'page': page, 'size': size}
        )
        return response.json()

# 使用示例
client = ChatGPTBackendClient('http://localhost:8000')

# 登录
login_result = client.login('testuser', 'password123')
print(f"登录结果: {login_result['message']}")

# 创建对话
conversation = client.create_conversation('Python 学习讨论')
conv_id = conversation['data']['id']

# 发送消息
message_result = client.send_message(
    conv_id, 
    '请解释一下 Python 的列表推导式',
    temperature=0.7
)
print(f"AI 回复: {message_result['data']['assistant_message']['content']}")
```

### JavaScript SDK 示例

```javascript
class ChatGPTBackendClient {
    constructor(baseUrl, apiKey = null) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.apiKey = apiKey;
    }

    async request(method, endpoint, data = null) {
        const url = `${this.baseUrl}/api/v1${endpoint}`;
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
        };

        if (this.apiKey) {
            options.headers['Authorization'] = `Bearer ${this.apiKey}`;
        }

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(url, options);
        return await response.json();
    }

    async login(username, password) {
        const result = await this.request('POST', '/auth/login', {
            username,
            password
        });
        
        if (result.success) {
            this.apiKey = result.data.access_token;
        }
        
        return result;
    }

    async createConversation(title, model = 'gpt-3.5-turbo') {
        return await this.request('POST', '/conversations', {
            title,
            model
        });
    }

    async sendMessage(conversationId, content, options = {}) {
        return await this.request('POST', `/conversations/${conversationId}/messages`, {
            content,
            role: 'user',
            ...options
        });
    }

    // WebSocket 连接
    connectWebSocket() {
        const wsUrl = `ws://localhost:8000/api/v1/ws/chat?token=${this.apiKey}`;
        const ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log('WebSocket 连接已建立');
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('收到消息:', data);
        };
        
        return ws;
    }
}

// 使用示例
const client = new ChatGPTBackendClient('http://localhost:8000');

// 登录
client.login('testuser', 'password123').then(result => {
    console.log('登录结果:', result.message);
    
    // 创建对话
    return client.createConversation('JavaScript 学习');
}).then(conversation => {
    const convId = conversation.data.id;
    
    // 发送消息
    return client.sendMessage(convId, '请解释一下 JavaScript 的闭包');
}).then(result => {
    console.log('AI 回复:', result.data.assistant_message.content);
});
```

## 📋 请求限制

### 频率限制

| 端点类型 | 限制 |
|----------|------|
| 认证接口 | 5次/分钟 |
| 消息发送 | 20次/分钟 |
| 其他接口 | 100次/分钟 |

### 数据限制

| 项目 | 限制 |
|------|------|
| 消息长度 | 最大 10,000 字符 |
| 对话标题 | 最大 200 字符 |
| 用户名 | 3-50 字符 |
| 密码 | 最少 8 字符 |

### 配额限制

| 用户类型 | 每日请求 | 每月请求 | 每日 Token | 每月 Token |
|----------|----------|----------|------------|------------|
| 普通用户 | 50 | 1,000 | 10,000 | 100,000 |
| VIP 用户 | 200 | 5,000 | 50,000 |