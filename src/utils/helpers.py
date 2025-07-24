#!/usr/bin/env python3
"""
辅助工具模块

提供各种常用的辅助函数
"""

import uuid
import secrets
import string
import re
import unicodedata
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
from urllib.parse import urlparse
from user_agents import parse as parse_user_agent_string
import math
from decimal import Decimal, ROUND_HALF_UP

from .logger import get_logger


# =============================================================================
# 配置和常量
# =============================================================================

logger = get_logger(__name__)

# 文件大小单位
FILE_SIZE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']

# 时间格式
DATETIME_FORMATS = {
    'iso': '%Y-%m-%dT%H:%M:%S',
    'date': '%Y-%m-%d',
    'time': '%H:%M:%S',
    'datetime': '%Y-%m-%d %H:%M:%S',
    'timestamp': '%Y%m%d%H%M%S',
    'human': '%Y年%m月%d日 %H:%M:%S'
}


# =============================================================================
# UUID和随机字符串生成
# =============================================================================

def generate_uuid(version: int = 4) -> str:
    """
    生成UUID
    
    Args:
        version: UUID版本（1或4）
        
    Returns:
        UUID字符串
    """
    try:
        if version == 1:
            return str(uuid.uuid1())
        elif version == 4:
            return str(uuid.uuid4())
        else:
            raise ValueError("UUID版本只支持1或4")
    except Exception as e:
        logger.error(f"Error generating UUID: {e}")
        return str(uuid.uuid4())  # 默认返回UUID4


def generate_short_uuid(length: int = 8) -> str:
    """
    生成短UUID
    
    Args:
        length: 长度
        
    Returns:
        短UUID字符串
    """
    try:
        full_uuid = str(uuid.uuid4()).replace('-', '')
        return full_uuid[:length]
    except Exception as e:
        logger.error(f"Error generating short UUID: {e}")
        return generate_random_string(length)


def generate_random_string(
    length: int = 32,
    include_uppercase: bool = True,
    include_lowercase: bool = True,
    include_digits: bool = True,
    include_symbols: bool = False,
    exclude_ambiguous: bool = True
) -> str:
    """
    生成随机字符串
    
    Args:
        length: 字符串长度
        include_uppercase: 包含大写字母
        include_lowercase: 包含小写字母
        include_digits: 包含数字
        include_symbols: 包含符号
        exclude_ambiguous: 排除易混淆字符
        
    Returns:
        随机字符串
    """
    try:
        characters = ""
        
        if include_lowercase:
            chars = string.ascii_lowercase
            if exclude_ambiguous:
                chars = chars.replace('l', '').replace('o', '')
            characters += chars
        
        if include_uppercase:
            chars = string.ascii_uppercase
            if exclude_ambiguous:
                chars = chars.replace('I', '').replace('O', '')
            characters += chars
        
        if include_digits:
            chars = string.digits
            if exclude_ambiguous:
                chars = chars.replace('0', '').replace('1', '')
            characters += chars
        
        if include_symbols:
            characters += "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        if not characters:
            raise ValueError("至少需要包含一种字符类型")
        
        return ''.join(secrets.choice(characters) for _ in range(length))
        
    except Exception as e:
        logger.error(f"Error generating random string: {e}")
        return secrets.token_urlsafe(length)[:length]


def generate_verification_code(length: int = 6, digits_only: bool = True) -> str:
    """
    生成验证码
    
    Args:
        length: 验证码长度
        digits_only: 只包含数字
        
    Returns:
        验证码
    """
    try:
        if digits_only:
            return ''.join(secrets.choice(string.digits) for _ in range(length))
        else:
            return generate_random_string(
                length=length,
                include_uppercase=True,
                include_lowercase=False,
                include_digits=True,
                include_symbols=False,
                exclude_ambiguous=True
            )
    except Exception as e:
        logger.error(f"Error generating verification code: {e}")
        return ''.join(secrets.choice(string.digits) for _ in range(length))


# =============================================================================
# 日期时间处理
# =============================================================================

def format_datetime(
    dt: datetime,
    format_type: str = 'datetime',
    custom_format: Optional[str] = None
) -> str:
    """
    格式化日期时间
    
    Args:
        dt: 日期时间对象
        format_type: 格式类型
        custom_format: 自定义格式
        
    Returns:
        格式化后的字符串
    """
    try:
        if custom_format:
            return dt.strftime(custom_format)
        
        format_str = DATETIME_FORMATS.get(format_type, DATETIME_FORMATS['datetime'])
        return dt.strftime(format_str)
        
    except Exception as e:
        logger.error(f"Error formatting datetime: {e}")
        return str(dt)


def parse_datetime(
    date_str: str,
    format_type: str = 'datetime',
    custom_format: Optional[str] = None
) -> Optional[datetime]:
    """
    解析日期时间字符串
    
    Args:
        date_str: 日期时间字符串
        format_type: 格式类型
        custom_format: 自定义格式
        
    Returns:
        日期时间对象
    """
    try:
        if custom_format:
            return datetime.strptime(date_str, custom_format)
        
        # 尝试多种格式
        formats_to_try = [
            DATETIME_FORMATS.get(format_type, DATETIME_FORMATS['datetime']),
            '%Y-%m-%dT%H:%M:%S.%fZ',  # ISO格式带毫秒
            '%Y-%m-%dT%H:%M:%SZ',     # ISO格式
            '%Y-%m-%d %H:%M:%S.%f',   # 带毫秒
            '%Y-%m-%d %H:%M:%S',      # 标准格式
            '%Y-%m-%d',               # 只有日期
        ]
        
        for fmt in formats_to_try:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
        
    except Exception as e:
        logger.error(f"Error parsing datetime: {e}")
        return None


def calculate_age(birth_date: datetime, reference_date: Optional[datetime] = None) -> int:
    """
    计算年龄
    
    Args:
        birth_date: 出生日期
        reference_date: 参考日期（默认为当前日期）
        
    Returns:
        年龄
    """
    try:
        if reference_date is None:
            reference_date = datetime.now()
        
        age = reference_date.year - birth_date.year
        
        # 检查是否还没到生日
        if (reference_date.month, reference_date.day) < (birth_date.month, birth_date.day):
            age -= 1
        
        return max(0, age)
        
    except Exception as e:
        logger.error(f"Error calculating age: {e}")
        return 0


def time_ago(dt: datetime, reference_time: Optional[datetime] = None) -> str:
    """
    计算相对时间（多久以前）
    
    Args:
        dt: 目标时间
        reference_time: 参考时间（默认为当前时间）
        
    Returns:
        相对时间描述
    """
    try:
        if reference_time is None:
            reference_time = datetime.now()
        
        diff = reference_time - dt
        
        if diff.days > 365:
            years = diff.days // 365
            return f"{years}年前"
        elif diff.days > 30:
            months = diff.days // 30
            return f"{months}个月前"
        elif diff.days > 0:
            return f"{diff.days}天前"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours}小时前"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes}分钟前"
        else:
            return "刚刚"
            
    except Exception as e:
        logger.error(f"Error calculating time ago: {e}")
        return "未知时间"


# =============================================================================
# 字符串处理
# =============================================================================

def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断字符串
    
    Args:
        text: 原始字符串
        max_length: 最大长度
        suffix: 后缀
        
    Returns:
        截断后的字符串
    """
    try:
        if not text or len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
        
    except Exception as e:
        logger.error(f"Error truncating string: {e}")
        return text


def slugify(text: str, max_length: int = 100) -> str:
    """
    将文本转换为URL友好的slug
    
    Args:
        text: 原始文本
        max_length: 最大长度
        
    Returns:
        slug字符串
    """
    try:
        # 转换为小写
        text = text.lower()
        
        # 标准化Unicode字符
        text = unicodedata.normalize('NFKD', text)
        
        # 移除非ASCII字符
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # 替换空格和特殊字符为连字符
        text = re.sub(r'[^a-z0-9]+', '-', text)
        
        # 移除开头和结尾的连字符
        text = text.strip('-')
        
        # 移除连续的连字符
        text = re.sub(r'-+', '-', text)
        
        # 截断到指定长度
        if len(text) > max_length:
            text = text[:max_length].rstrip('-')
        
        return text
        
    except Exception as e:
        logger.error(f"Error creating slug: {e}")
        return generate_random_string(8, include_uppercase=False, include_symbols=False)


def camel_to_snake(name: str) -> str:
    """
    驼峰命名转下划线命名
    
    Args:
        name: 驼峰命名字符串
        
    Returns:
        下划线命名字符串
    """
    try:
        # 在大写字母前插入下划线
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    except Exception as e:
        logger.error(f"Error converting camel to snake: {e}")
        return name


def snake_to_camel(name: str, capitalize_first: bool = False) -> str:
    """
    下划线命名转驼峰命名
    
    Args:
        name: 下划线命名字符串
        capitalize_first: 是否首字母大写
        
    Returns:
        驼峰命名字符串
    """
    try:
        components = name.split('_')
        if capitalize_first:
            return ''.join(word.capitalize() for word in components)
        else:
            return components[0] + ''.join(word.capitalize() for word in components[1:])
    except Exception as e:
        logger.error(f"Error converting snake to camel: {e}")
        return name


def extract_domain(url: str) -> Optional[str]:
    """
    从URL中提取域名
    
    Args:
        url: URL地址
        
    Returns:
        域名
    """
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        parsed = urlparse(url)
        return parsed.netloc
        
    except Exception as e:
        logger.error(f"Error extracting domain: {e}")
        return None


def mask_sensitive_data(data: str, mask_char: str = '*', visible_chars: int = 4) -> str:
    """
    遮蔽敏感数据
    
    Args:
        data: 敏感数据
        mask_char: 遮蔽字符
        visible_chars: 可见字符数
        
    Returns:
        遮蔽后的数据
    """
    try:
        if len(data) <= visible_chars * 2:
            return mask_char * len(data)
        
        start = data[:visible_chars]
        end = data[-visible_chars:]
        middle = mask_char * (len(data) - visible_chars * 2)
        
        return start + middle + end
        
    except Exception as e:
        logger.error(f"Error masking sensitive data: {e}")
        return mask_char * len(data)


# =============================================================================
# 数值处理
# =============================================================================

def format_file_size(size_bytes: int, decimal_places: int = 2) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 文件大小（字节）
        decimal_places: 小数位数
        
    Returns:
        格式化后的文件大小
    """
    try:
        if size_bytes == 0:
            return "0 B"
        
        size_bytes = abs(size_bytes)
        unit_index = int(math.floor(math.log(size_bytes, 1024)))
        
        if unit_index >= len(FILE_SIZE_UNITS):
            unit_index = len(FILE_SIZE_UNITS) - 1
        
        size = size_bytes / (1024 ** unit_index)
        unit = FILE_SIZE_UNITS[unit_index]
        
        if decimal_places == 0:
            return f"{int(size)} {unit}"
        else:
            return f"{size:.{decimal_places}f} {unit}"
            
    except Exception as e:
        logger.error(f"Error formatting file size: {e}")
        return f"{size_bytes} B"


def round_decimal(value: Union[float, Decimal], decimal_places: int = 2) -> Decimal:
    """
    精确的小数四舍五入
    
    Args:
        value: 数值
        decimal_places: 小数位数
        
    Returns:
        四舍五入后的Decimal
    """
    try:
        if isinstance(value, float):
            value = Decimal(str(value))
        elif not isinstance(value, Decimal):
            value = Decimal(value)
        
        quantizer = Decimal('0.1') ** decimal_places
        return value.quantize(quantizer, rounding=ROUND_HALF_UP)
        
    except Exception as e:
        logger.error(f"Error rounding decimal: {e}")
        return Decimal('0')


def format_number(number: Union[int, float], thousands_separator: str = ',') -> str:
    """
    格式化数字（添加千位分隔符）
    
    Args:
        number: 数字
        thousands_separator: 千位分隔符
        
    Returns:
        格式化后的数字字符串
    """
    try:
        return f"{number:,}".replace(',', thousands_separator)
    except Exception as e:
        logger.error(f"Error formatting number: {e}")
        return str(number)


def percentage(part: Union[int, float], total: Union[int, float], decimal_places: int = 2) -> float:
    """
    计算百分比
    
    Args:
        part: 部分值
        total: 总值
        decimal_places: 小数位数
        
    Returns:
        百分比
    """
    try:
        if total == 0:
            return 0.0
        
        result = (part / total) * 100
        return round(result, decimal_places)
        
    except Exception as e:
        logger.error(f"Error calculating percentage: {e}")
        return 0.0


# =============================================================================
# 网络和请求处理
# =============================================================================

def get_client_ip(request_headers: Dict[str, str]) -> str:
    """
    获取客户端IP地址
    
    Args:
        request_headers: 请求头字典
        
    Returns:
        客户端IP地址
    """
    try:
        # 检查代理头部
        proxy_headers = [
            'X-Forwarded-For',
            'X-Real-IP',
            'X-Client-IP',
            'CF-Connecting-IP',  # Cloudflare
            'True-Client-IP',    # Akamai
        ]
        
        for header in proxy_headers:
            ip = request_headers.get(header)
            if ip:
                # X-Forwarded-For可能包含多个IP，取第一个
                if ',' in ip:
                    ip = ip.split(',')[0].strip()
                
                # 验证IP格式
                try:
                    import ipaddress
                    ipaddress.ip_address(ip)
                    return ip
                except ValueError:
                    continue
        
        # 如果没有代理头部，返回默认值
        return request_headers.get('Remote-Addr', '127.0.0.1')
        
    except Exception as e:
        logger.error(f"Error getting client IP: {e}")
        return '127.0.0.1'


def parse_user_agent(user_agent_string: str) -> Dict[str, Any]:
    """
    解析User-Agent字符串
    
    Args:
        user_agent_string: User-Agent字符串
        
    Returns:
        解析结果字典
    """
    try:
        if not user_agent_string:
            return {
                'browser': 'Unknown',
                'browser_version': 'Unknown',
                'os': 'Unknown',
                'os_version': 'Unknown',
                'device': 'Unknown',
                'is_mobile': False,
                'is_tablet': False,
                'is_pc': True
            }
        
        user_agent = parse_user_agent_string(user_agent_string)
        
        return {
            'browser': user_agent.browser.family,
            'browser_version': user_agent.browser.version_string,
            'os': user_agent.os.family,
            'os_version': user_agent.os.version_string,
            'device': user_agent.device.family,
            'is_mobile': user_agent.is_mobile,
            'is_tablet': user_agent.is_tablet,
            'is_pc': user_agent.is_pc
        }
        
    except Exception as e:
        logger.error(f"Error parsing user agent: {e}")
        return {
            'browser': 'Unknown',
            'browser_version': 'Unknown',
            'os': 'Unknown',
            'os_version': 'Unknown',
            'device': 'Unknown',
            'is_mobile': False,
            'is_tablet': False,
            'is_pc': True
        }


# =============================================================================
# 数据结构处理
# =============================================================================

def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并字典
    
    Args:
        dict1: 字典1
        dict2: 字典2
        
    Returns:
        合并后的字典
    """
    try:
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
        
    except Exception as e:
        logger.error(f"Error deep merging dictionaries: {e}")
        return dict1


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    扁平化嵌套字典
    
    Args:
        d: 嵌套字典
        parent_key: 父键名
        sep: 分隔符
        
    Returns:
        扁平化后的字典
    """
    try:
        items = []
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        
        return dict(items)
        
    except Exception as e:
        logger.error(f"Error flattening dictionary: {e}")
        return d


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    将列表分块
    
    Args:
        lst: 原始列表
        chunk_size: 块大小
        
    Returns:
        分块后的列表
    """
    try:
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    except Exception as e:
        logger.error(f"Error chunking list: {e}")
        return [lst]


def remove_duplicates(lst: List[Any], key_func: Optional[callable] = None) -> List[Any]:
    """
    移除列表中的重复项
    
    Args:
        lst: 原始列表
        key_func: 用于比较的键函数
        
    Returns:
        去重后的列表
    """
    try:
        if key_func is None:
            return list(dict.fromkeys(lst))  # 保持顺序的去重
        else:
            seen = set()
            result = []
            for item in lst:
                key = key_func(item)
                if key not in seen:
                    seen.add(key)
                    result.append(item)
            return result
            
    except Exception as e:
        logger.error(f"Error removing duplicates: {e}")
        return lst


# =============================================================================
# 其他工具函数
# =============================================================================

def safe_int(value: Any, default: int = 0) -> int:
    """
    安全转换为整数
    
    Args:
        value: 要转换的值
        default: 默认值
        
    Returns:
        整数值
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    安全转换为浮点数
    
    Args:
        value: 要转换的值
        default: 默认值
        
    Returns:
        浮点数值
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    """
    安全转换为布尔值
    
    Args:
        value: 要转换的值
        default: 默认值
        
    Returns:
        布尔值
    """
    try:
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
        elif isinstance(value, (int, float)):
            return bool(value)
        else:
            return default
    except Exception:
        return default


def is_valid_json(json_string: str) -> bool:
    """
    检查字符串是否为有效的JSON
    
    Args:
        json_string: JSON字符串
        
    Returns:
        是否有效
    """
    try:
        import json
        json.loads(json_string)
        return True
    except (ValueError, TypeError):
        return False


def get_nested_value(data: Dict[str, Any], key_path: str, default: Any = None, separator: str = '.') -> Any:
    """
    获取嵌套字典中的值
    
    Args:
        data: 数据字典
        key_path: 键路径（如 'user.profile.name'）
        default: 默认值
        separator: 分隔符
        
    Returns:
        值
    """
    try:
        keys = key_path.split(separator)
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
        
    except Exception as e:
        logger.error(f"Error getting nested value: {e}")
        return default


def set_nested_value(data: Dict[str, Any], key_path: str, value: Any, separator: str = '.') -> Dict[str, Any]:
    """
    设置嵌套字典中的值
    
    Args:
        data: 数据字典
        key_path: 键路径（如 'user.profile.name'）
        value: 要设置的值
        separator: 分隔符
        
    Returns:
        更新后的字典
    """
    try:
        keys = key_path.split(separator)
        current = data
        
        # 创建嵌套结构
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        # 设置最终值
        current[keys[-1]] = value
        
        return data
        
    except Exception as e:
        logger.error(f"Error setting nested value: {e}")
        return data