#!/usr/bin/env python3
"""
验证器工具模块

提供各种数据验证功能
"""

import re
import mimetypes
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
from email_validator import validate_email as email_validate, EmailNotValidError
import phonenumbers
from phonenumbers import NumberParseException
import bleach
from datetime import datetime
import ipaddress

from .logger import get_logger


# =============================================================================
# 配置和常量
# =============================================================================

logger = get_logger(__name__)

# 允许的HTML标签和属性（用于内容清理）
ALLOWED_TAGS = [
    'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'blockquote', 'code', 'pre',
    'a', 'img'
]

ALLOWED_ATTRIBUTES = {
    'a': ['href', 'title'],
    'img': ['src', 'alt', 'title', 'width', 'height'],
    '*': ['class']
}

# 文件类型配置
ALLOWED_IMAGE_TYPES = {
    'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml'
}

ALLOWED_DOCUMENT_TYPES = {
    'application/pdf', 'application/msword',
    'application/vnd.openxmlformats-officedocument.' +
        'wordprocessingml.document',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.' +
        'spreadsheetml.sheet',
    'text/plain', 'text/csv'
}

ALLOWED_ARCHIVE_TYPES = {
    'application/zip', 'application/x-rar-compressed',
    'application/x-7z-compressed', 'application/x-tar',
    'application/gzip'
}

# 文件大小限制（字节）
MAX_FILE_SIZES = {
    'image': 5 * 1024 * 1024,      # 5MB
    'document': 10 * 1024 * 1024,  # 10MB
    'archive': 50 * 1024 * 1024,   # 50MB
    'default': 2 * 1024 * 1024     # 2MB
}


# =============================================================================
# 基础验证函数
# =============================================================================

def validate_email(email: str) -> tuple[bool, Optional[str]]:
    """
    验证邮箱地址

    Args:
        email: 邮箱地址

    Returns:
        (是否有效, 错误信息)
    """
    try:
        if not email or not isinstance(email, str):
            return False, "邮箱地址不能为空"

        # 基础格式检查
        if len(email) > 254:
            return False, "邮箱地址过长"

        # 使用email-validator库进行验证
        email_validate(email)
        return True, None

    except EmailNotValidError as e:
        return False, f"邮箱格式无效: {str(e)}"
    except Exception as e:
        logger.error(f"Error validating email: {e}")
        return False, "邮箱验证失败"


def validate_phone(phone: str, region: str = "CN") -> tuple[bool, Optional[str]]:
    """
    验证手机号码

    Args:
        phone: 手机号码
        region: 地区代码

    Returns:
        (是否有效, 错误信息)
    """
    try:
        if not phone or not isinstance(phone, str):
            return False, "手机号码不能为空"

        # 解析手机号码
        parsed_number = phonenumbers.parse(phone, region)

        # 验证号码
        if not phonenumbers.is_valid_number(parsed_number):
            return False, "手机号码格式无效"

        # 检查是否为手机号码（非固定电话）
        number_type = phonenumbers.number_type(parsed_number)
        valid_types = [
            phonenumbers.PhoneNumberType.MOBILE,
            phonenumbers.PhoneNumberType.FIXED_LINE_OR_MOBILE
        ]
        if number_type not in valid_types:
            return False, "请输入手机号码"

        return True, None

    except NumberParseException as e:
        return False, f"手机号码解析失败: {str(e)}"
    except Exception as e:
        logger.error(f"Error validating phone: {e}")
        return False, "手机号码验证失败"


def validate_password(password: str, min_length: int = 8, max_length: int = 128) -> tuple[bool, List[str]]:
    """
    验证密码强度

    Args:
        password: 密码
        min_length: 最小长度
        max_length: 最大长度

    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []

    try:
        if not password or not isinstance(password, str):
            errors.append("密码不能为空")
            return False, errors

        # 长度检查
        if len(password) < min_length:
            errors.append(f"密码长度至少{min_length}位")

        if len(password) > max_length:
            errors.append(f"密码长度不能超过{max_length}位")

        # 复杂度检查
        if not re.search(r'[a-z]', password):
            errors.append("密码必须包含小写字母")

        if not re.search(r'[A-Z]', password):
            errors.append("密码必须包含大写字母")

        if not re.search(r'\d', password):
            errors.append("密码必须包含数字")

        special_chars = r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]'
        if not re.search(special_chars, password):
            errors.append("密码必须包含特殊字符")

        # 检查常见弱密码模式
        weak_patterns = [
            r'(.)\1{2,}',  # 连续相同字符
            r'123456',     # 连续数字
            r'abcdef',     # 连续字母
            r'qwerty',     # 键盘序列
        ]

        for pattern in weak_patterns:
            if re.search(pattern, password.lower()):
                errors.append("密码不能包含简单的重复或连续字符")
                break

        return len(errors) == 0, errors

    except Exception as e:
        logger.error(f"Error validating password: {e}")
        return False, ["密码验证失败"]


def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> tuple[bool, Optional[str]]:
    """
    验证URL

    Args:
        url: URL地址
        allowed_schemes: 允许的协议列表

    Returns:
        (是否有效, 错误信息)
    """
    try:
        if not url or not isinstance(url, str):
            return False, "URL不能为空"

        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']

        # 解析URL
        parsed = urlparse(url)

        # 检查协议
        if parsed.scheme not in allowed_schemes:
            return False, f"不支持的协议: {parsed.scheme}"

        # 检查域名
        if not parsed.netloc:
            return False, "URL缺少域名"

        # 检查URL长度
        if len(url) > 2048:
            return False, "URL过长"

        return True, None

    except Exception as e:
        logger.error(f"Error validating URL: {e}")
        return False, "URL验证失败"


def validate_ip_address(ip: str) -> tuple[bool, Optional[str]]:
    """
    验证IP地址

    Args:
        ip: IP地址

    Returns:
        (是否有效, 错误信息)
    """
    try:
        if not ip or not isinstance(ip, str):
            return False, "IP地址不能为空"

        # 尝试解析IPv4或IPv6地址
        ipaddress.ip_address(ip)
        return True, None

    except ValueError:
        return False, "IP地址格式无效"
    except Exception as e:
        logger.error(f"Error validating IP address: {e}")
        return False, "IP地址验证失败"


def validate_domain(domain: str) -> tuple[bool, Optional[str]]:
    """
    验证域名

    Args:
        domain: 域名

    Returns:
        (是否有效, 错误信息)
    """
    try:
        if not domain or not isinstance(domain, str):
            return False, "域名不能为空"

        # 域名长度检查
        if len(domain) > 253:
            return False, "域名过长"

        # 域名格式检查
        domain_pattern = (r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+'
                         r'(?:xn--[a-zA-Z0-9]+|[a-zA-Z]{2,})$')
        if not re.match(domain_pattern, domain):
            return False, "域名格式无效"

        return True, None

    except Exception as e:
        logger.error(f"Error validating domain: {e}")
        return False, "域名验证失败"


# =============================================================================
# 文件验证
# =============================================================================

def validate_file_type(filename: str, file_content: bytes, allowed_types: Optional[set] = None) -> tuple[bool, Optional[str]]:
    """
    验证文件类型

    Args:
        filename: 文件名
        file_content: 文件内容
        allowed_types: 允许的MIME类型集合

    Returns:
        (是否有效, 错误信息)
    """
    try:
        if not filename:
            return False, "文件名不能为空"

        # 获取文件扩展名
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ''

        # 根据文件名猜测MIME类型
        mime_type, _ = mimetypes.guess_type(filename)

        if not mime_type:
            return False, "无法识别文件类型"

        # 如果没有指定允许的类型，使用默认配置
        if allowed_types is None:
            allowed_types = ALLOWED_IMAGE_TYPES | ALLOWED_DOCUMENT_TYPES | ALLOWED_ARCHIVE_TYPES

        if mime_type not in allowed_types:
            return False, f"不支持的文件类型: {mime_type}"

        # 文件头验证（防止文件伪造）
        if not _validate_file_signature(file_content, mime_type):
            return False, "文件内容与扩展名不匹配"

        return True, None

    except Exception as e:
        logger.error(f"Error validating file type: {e}")
        return False, "文件类型验证失败"


def validate_file_size(file_size: int, file_type: str = 'default') -> tuple[bool, Optional[str]]:
    """
    验证文件大小

    Args:
        file_size: 文件大小（字节）
        file_type: 文件类型

    Returns:
        (是否有效, 错误信息)
    """
    try:
        if file_size <= 0:
            return False, "文件不能为空"

        max_size = MAX_FILE_SIZES.get(file_type, MAX_FILE_SIZES['default'])

        if file_size > max_size:
            max_size_mb = max_size / (1024 * 1024)
            return False, f"文件大小不能超过{max_size_mb:.1f}MB"

        return True, None

    except Exception as e:
        logger.error(f"Error validating file size: {e}")
        return False, "文件大小验证失败"


def _validate_file_signature(file_content: bytes, mime_type: str) -> bool:
    """
    验证文件签名（文件头）

    Args:
        file_content: 文件内容
        mime_type: MIME类型

    Returns:
        是否匹配
    """
    if len(file_content) < 4:
        return False

    # 文件签名映射
    signatures = {
        'image/jpeg': [b'\xff\xd8\xff'],
        'image/png': [b'\x89PNG\r\n\x1a\n'],
        'image/gif': [b'GIF87a', b'GIF89a'],
        'image/webp': [b'RIFF'],
        'application/pdf': [b'%PDF'],
        'application/zip': [b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08'],
        'application/x-rar-compressed': [
            b'Rar!\x1a\x07\x00',
            b'Rar!\x1a\x07\x01\x00'
        ],
    }

    expected_signatures = signatures.get(mime_type, [])
    if not expected_signatures:
        return True  # 如果没有定义签名，则跳过验证

    for signature in expected_signatures:
        if file_content.startswith(signature):
            return True

    return False


# =============================================================================
# 内容验证和清理
# =============================================================================

def sanitize_input(text: str, allow_html: bool = False) -> str:
    """
    清理用户输入

    Args:
        text: 输入文本
        allow_html: 是否允许HTML

    Returns:
        清理后的文本
    """
    try:
        if not text or not isinstance(text, str):
            return ""

        # 移除控制字符
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')

        if allow_html:
            # 清理HTML，只保留安全的标签和属性
            text = bleach.clean(
                text,
                tags=ALLOWED_TAGS,
                attributes=ALLOWED_ATTRIBUTES,
                strip=True
            )
        else:
            # 转义HTML字符
            text = bleach.clean(text, tags=[], strip=True)

        return text.strip()

    except Exception as e:
        logger.error(f"Error sanitizing input: {e}")
        return ""


def validate_json_structure(data: Any, schema: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    验证JSON数据结构

    Args:
        data: 要验证的数据
        schema: 期望的结构模式

    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []

    try:
        def _validate_field(value: Any, field_schema: Dict[str, Any], field_path: str = ""):
            field_type = field_schema.get('type')
            required = field_schema.get('required', False)

            # 检查必填字段
            if required and (value is None or value == ""):
                errors.append(f"{field_path}: 字段是必填的")
                return

            # 如果字段为空且非必填，跳过验证
            if value is None or value == "":
                return

            # 类型验证
            if field_type == 'string' and not isinstance(value, str):
                errors.append(f"{field_path}: 期望字符串类型")
            elif field_type == 'integer' and not isinstance(value, int):
                errors.append(f"{field_path}: 期望整数类型")
            elif field_type == 'number' and not isinstance(value, (int, float)):
                errors.append(f"{field_path}: 期望数字类型")
            elif field_type == 'boolean' and not isinstance(value, bool):
                errors.append(f"{field_path}: 期望布尔类型")
            elif field_type == 'array' and not isinstance(value, list):
                errors.append(f"{field_path}: 期望数组类型")
            elif field_type == 'object' and not isinstance(value, dict):
                errors.append(f"{field_path}: 期望对象类型")

            # 长度验证
            if isinstance(value, str):
                min_length = field_schema.get('min_length')
                max_length = field_schema.get('max_length')

                if min_length and len(value) < min_length:
                    errors.append(f"{field_path}: 长度不能少于{min_length}")

                if max_length and len(value) > max_length:
                    errors.append(f"{field_path}: 长度不能超过{max_length}")

            # 数值范围验证
            if isinstance(value, (int, float)):
                min_value = field_schema.get('min_value')
                max_value = field_schema.get('max_value')

                if min_value is not None and value < min_value:
                    errors.append(f"{field_path}: 值不能小于{min_value}")

                if max_value is not None and value > max_value:
                    errors.append(f"{field_path}: 值不能大于{max_value}")

            # 枚举值验证
            enum_values = field_schema.get('enum')
            if enum_values and value not in enum_values:
                errors.append(f"{field_path}: 值必须是{enum_values}中的一个")

            # 正则表达式验证
            pattern = field_schema.get('pattern')
            if pattern and isinstance(value, str) and not re.match(pattern, value):
                errors.append(f"{field_path}: 格式不正确")

            # 递归验证对象属性
            if field_type == 'object' and isinstance(value, dict):
                properties = field_schema.get('properties', {})
                for prop_name, prop_schema in properties.items():
                    prop_value = value.get(prop_name)
                    prop_path = f"{field_path}.{prop_name}" if field_path else prop_name
                    _validate_field(prop_value, prop_schema, prop_path)

            # 递归验证数组元素
            if field_type == 'array' and isinstance(value, list):
                items_schema = field_schema.get('items')
                if items_schema:
                    for i, item in enumerate(value):
                        item_path = f"{field_path}[{i}]" if field_path else f"[{i}]"
                        _validate_field(item, items_schema, item_path)

        # 验证根级别的属性
        if isinstance(schema, dict) and 'properties' in schema:
            for field_name, field_schema in schema['properties'].items():
                field_value = data.get(field_name) if isinstance(data, dict) else None
                _validate_field(field_value, field_schema, field_name)

        return len(errors) == 0, errors

    except Exception as e:
        logger.error(f"Error validating JSON structure: {e}")
        return False, ["JSON结构验证失败"]


# =============================================================================
# 日期时间验证
# =============================================================================

def validate_date_format(date_str: str, format_str: str = "%Y-%m-%d") -> tuple[bool, Optional[str]]:
    """
    验证日期格式

    Args:
        date_str: 日期字符串
        format_str: 期望的日期格式

    Returns:
        (是否有效, 错误信息)
    """
    try:
        if not date_str or not isinstance(date_str, str):
            return False, "日期不能为空"

        datetime.strptime(date_str, format_str)
        return True, None

    except ValueError:
        return False, f"日期格式无效，期望格式: {format_str}"
    except Exception as e:
        logger.error(f"Error validating date format: {e}")
        return False, "日期验证失败"


def validate_datetime_range(
    datetime_str: str,
    min_datetime: Optional[datetime] = None,
    max_datetime: Optional[datetime] = None,
    format_str: str = "%Y-%m-%d %H:%M:%S"
) -> tuple[bool, Optional[str]]:
    """
    验证日期时间范围

    Args:
        datetime_str: 日期时间字符串
        min_datetime: 最小日期时间
        max_datetime: 最大日期时间
        format_str: 日期时间格式

    Returns:
        (是否有效, 错误信息)
    """
    try:
        if not datetime_str or not isinstance(datetime_str, str):
            return False, "日期时间不能为空"

        dt = datetime.strptime(datetime_str, format_str)

        if min_datetime and dt < min_datetime:
            return False, f"日期时间不能早于{min_datetime.strftime(format_str)}"

        if max_datetime and dt > max_datetime:
            return False, f"日期时间不能晚于{max_datetime.strftime(format_str)}"

        return True, None

    except ValueError:
        return False, f"日期时间格式无效，期望格式: {format_str}"
    except Exception as e:
        logger.error(f"Error validating datetime range: {e}")
        return False, "日期时间验证失败"


# =============================================================================
# 业务逻辑验证
# =============================================================================

def validate_username(
    username: str,
    min_length: int = 3,
    max_length: int = 30
) -> tuple[bool, Optional[str]]:
    """
    验证用户名

    Args:
        username: 用户名
        min_length: 最小长度
        max_length: 最大长度

    Returns:
        (是否有效, 错误信息)
    """
    try:
        if not username or not isinstance(username, str):
            return False, "用户名不能为空"

        # 长度检查
        if len(username) < min_length:
            return False, f"用户名长度不能少于{min_length}位"

        if len(username) > max_length:
            return False, f"用户名长度不能超过{max_length}位"

        # 格式检查：只允许字母、数字、下划线、连字符
        if not re.match(r'^[a-zA-Z0-9_-]+$', username):
            return False, "用户名只能包含字母、数字、下划线和连字符"

        # 不能以数字开头
        if username[0].isdigit():
            return False, "用户名不能以数字开头"

        # 不能全是数字
        if username.isdigit():
            return False, "用户名不能全是数字"

        # 检查保留词
        reserved_words = {
            'admin', 'administrator', 'root', 'system', 'user',
            'api', 'www', 'mail', 'ftp', 'test', 'guest',
            'null', 'undefined', 'true', 'false'
        }

        if username.lower() in reserved_words:
            return False, "用户名不能使用保留词"

        return True, None

    except Exception as e:
        logger.error(f"Error validating username: {e}")
        return False, "用户名验证失败"


def validate_slug(slug: str, max_length: int = 100) -> tuple[bool, Optional[str]]:
    """
    验证URL别名（slug）

    Args:
        slug: URL别名
        max_length: 最大长度

    Returns:
        (是否有效, 错误信息)
    """
    try:
        if not slug or not isinstance(slug, str):
            return False, "URL别名不能为空"

        # 长度检查
        if len(slug) > max_length:
            return False, f"URL别名长度不能超过{max_length}位"

        # 格式检查：只允许小写字母、数字、连字符
        if not re.match(r'^[a-z0-9-]+$', slug):
            return False, "URL别名只能包含小写字母、数字和连字符"

        # 不能以连字符开头或结尾
        if slug.startswith('-') or slug.endswith('-'):
            return False, "URL别名不能以连字符开头或结尾"

        # 不能包含连续的连字符
        if '--' in slug:
            return False, "URL别名不能包含连续的连字符"

        return True, None

    except Exception as e:
        logger.error(f"Error validating slug: {e}")
        return False, "URL别名验证失败"


# =============================================================================
# 批量验证
# =============================================================================

def validate_multiple_fields(
    data: Dict[str, Any],
    validators: Dict[str, callable]
) -> Dict[str, List[str]]:
    """
    批量验证多个字段

    Args:
        data: 要验证的数据
        validators: 验证器映射 {字段名: 验证函数}

    Returns:
        验证错误映射 {字段名: 错误列表}
    """
    errors = {}

    try:
        for field_name, validator in validators.items():
            field_value = data.get(field_name)

            try:
                is_valid, error_message = validator(field_value)
                if not is_valid:
                    if field_name not in errors:
                        errors[field_name] = []

                    if isinstance(error_message, list):
                        errors[field_name].extend(error_message)
                    else:
                        errors[field_name].append(error_message)

            except Exception as e:
                logger.error(f"Error validating field {field_name}: {e}")
                if field_name not in errors:
                    errors[field_name] = []
                errors[field_name].append(f"验证失败: {str(e)}")

        return errors

    except Exception as e:
        logger.error(f"Error in batch validation: {e}")
        return {'_general': ['批量验证失败']}
