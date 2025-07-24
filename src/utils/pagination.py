#!/usr/bin/env python3
"""
分页工具模块

提供分页相关的功能
"""

import math
import functools
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from dataclasses import dataclass
from urllib.parse import urlencode, urlparse, parse_qs

from .logger import get_logger


# =============================================================================
# 类型定义
# =============================================================================

T = TypeVar('T')
logger = get_logger(__name__)


# =============================================================================
# 分页配置
# =============================================================================

@dataclass
class PaginationConfig:
    """分页配置"""
    default_page_size: int = 20
    max_page_size: int = 100
    min_page_size: int = 1
    page_param: str = 'page'
    size_param: str = 'size'
    sort_param: str = 'sort'
    order_param: str = 'order'
    search_param: str = 'search'


# 默认配置
DEFAULT_CONFIG = PaginationConfig()


# =============================================================================
# 分页信息类
# =============================================================================

@dataclass
class PageInfo:
    """分页信息"""
    page: int  # 当前页码（从1开始）
    size: int  # 每页大小
    total: int  # 总记录数
    total_pages: int  # 总页数
    has_previous: bool  # 是否有上一页
    has_next: bool  # 是否有下一页
    previous_page: Optional[int]  # 上一页页码
    next_page: Optional[int]  # 下一页页码
    start_index: int  # 当前页第一条记录的索引（从0开始）
    end_index: int  # 当前页最后一条记录的索引（从0开始）
    
    @classmethod
    def create(
        cls,
        page: int,
        size: int,
        total: int
    ) -> 'PageInfo':
        """创建分页信息"""
        # 确保页码至少为1
        page = max(1, page)
        
        # 计算总页数
        total_pages = math.ceil(total / size) if total > 0 else 1
        
        # 确保页码不超过总页数
        page = min(page, total_pages)
        
        # 计算索引
        start_index = (page - 1) * size
        end_index = min(start_index + size - 1, total - 1)
        
        # 计算上一页和下一页
        has_previous = page > 1
        has_next = page < total_pages
        previous_page = page - 1 if has_previous else None
        next_page = page + 1 if has_next else None
        
        return cls(
            page=page,
            size=size,
            total=total,
            total_pages=total_pages,
            has_previous=has_previous,
            has_next=has_next,
            previous_page=previous_page,
            next_page=next_page,
            start_index=start_index,
            end_index=end_index
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'page': self.page,
            'size': self.size,
            'total': self.total,
            'total_pages': self.total_pages,
            'has_previous': self.has_previous,
            'has_next': self.has_next,
            'previous_page': self.previous_page,
            'next_page': self.next_page,
            'start_index': self.start_index,
            'end_index': self.end_index
        }


# =============================================================================
# 分页结果类
# =============================================================================

@dataclass
class PageResult(Generic[T]):
    """分页结果"""
    items: List[T]  # 当前页的数据
    page_info: PageInfo  # 分页信息
    
    def to_dict(self, item_serializer: Optional[callable] = None) -> Dict[str, Any]:
        """转换为字典"""
        if item_serializer:
            items = [item_serializer(item) for item in self.items]
        else:
            items = self.items
        
        return {
            'items': items,
            'pagination': self.page_info.to_dict()
        }
    
    @property
    def is_empty(self) -> bool:
        """是否为空"""
        return len(self.items) == 0
    
    @property
    def count(self) -> int:
        """当前页记录数"""
        return len(self.items)


# =============================================================================
# 分页参数类
# =============================================================================

@dataclass
class PaginationParams:
    """分页参数"""
    page: int = 1
    size: int = 20
    sort: Optional[str] = None
    order: str = 'asc'  # asc 或 desc
    search: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """参数验证和标准化"""
        # 验证页码
        self.page = max(1, self.page)
        
        # 验证页面大小
        self.size = max(DEFAULT_CONFIG.min_page_size, 
                       min(DEFAULT_CONFIG.max_page_size, self.size))
        
        # 标准化排序顺序
        self.order = self.order.lower()
        if self.order not in ['asc', 'desc']:
            self.order = 'asc'
        
        # 初始化过滤器
        if self.filters is None:
            self.filters = {}
    
    @classmethod
    def from_dict(
        cls,
        params: Dict[str, Any],
        config: Optional[PaginationConfig] = None
    ) -> 'PaginationParams':
        """从字典创建分页参数"""
        if config is None:
            config = DEFAULT_CONFIG
        
        # 提取分页参数
        page = int(params.get(config.page_param, 1))
        size = int(params.get(config.size_param, config.default_page_size))
        sort = params.get(config.sort_param)
        order = params.get(config.order_param, 'asc')
        search = params.get(config.search_param)
        
        # 提取过滤器（排除分页相关参数）
        excluded_params = {
            config.page_param,
            config.size_param,
            config.sort_param,
            config.order_param,
            config.search_param
        }
        
        filters = {
            k: v for k, v in params.items()
            if k not in excluded_params and v is not None and v != ''
        }
        
        return cls(
            page=page,
            size=size,
            sort=sort,
            order=order,
            search=search,
            filters=filters
        )
    
    @classmethod
    def from_url(
        cls,
        url: str,
        config: Optional[PaginationConfig] = None
    ) -> 'PaginationParams':
        """从URL创建分页参数"""
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        
        # 将查询参数转换为单值字典
        params = {
            k: v[0] if v else None
            for k, v in query_params.items()
        }
        
        return cls.from_dict(params, config)
    
    def to_dict(self, config: Optional[PaginationConfig] = None) -> Dict[str, Any]:
        """转换为字典"""
        if config is None:
            config = DEFAULT_CONFIG
        
        result = {
            config.page_param: self.page,
            config.size_param: self.size,
            config.order_param: self.order
        }
        
        if self.sort:
            result[config.sort_param] = self.sort
        
        if self.search:
            result[config.search_param] = self.search
        
        # 添加过滤器
        if self.filters:
            result.update(self.filters)
        
        return result
    
    def to_url_params(self, config: Optional[PaginationConfig] = None) -> str:
        """转换为URL参数字符串"""
        params = self.to_dict(config)
        # 过滤空值
        params = {k: v for k, v in params.items() if v is not None and v != ''}
        return urlencode(params)
    
    def get_offset(self) -> int:
        """获取偏移量（用于数据库查询）"""
        return (self.page - 1) * self.size
    
    def get_limit(self) -> int:
        """获取限制数量（用于数据库查询）"""
        return self.size


# =============================================================================
# 分页器类
# =============================================================================

class Paginator:
    """分页器"""
    
    def __init__(self, config: Optional[PaginationConfig] = None):
        self.config = config or DEFAULT_CONFIG
    
    def paginate(
        self,
        items: List[T],
        params: PaginationParams
    ) -> PageResult[T]:
        """对列表进行分页"""
        total = len(items)
        page_info = PageInfo.create(params.page, params.size, total)
        
        # 计算切片范围
        start = page_info.start_index
        end = min(start + params.size, total)
        
        # 获取当前页数据
        page_items = items[start:end] if start < total else []
        
        return PageResult(items=page_items, page_info=page_info)
    
    def create_page_result(
        self,
        items: List[T],
        total: int,
        params: PaginationParams
    ) -> PageResult[T]:
        """创建分页结果（用于数据库查询结果）"""
        page_info = PageInfo.create(params.page, params.size, total)
        return PageResult(items=items, page_info=page_info)
    
    def get_page_numbers(
        self,
        page_info: PageInfo,
        max_pages: int = 10
    ) -> List[int]:
        """获取页码列表（用于分页导航）"""
        if page_info.total_pages <= max_pages:
            return list(range(1, page_info.total_pages + 1))
        
        # 计算显示范围
        half_max = max_pages // 2
        start = max(1, page_info.page - half_max)
        end = min(page_info.total_pages, start + max_pages - 1)
        
        # 调整起始位置
        if end - start + 1 < max_pages:
            start = max(1, end - max_pages + 1)
        
        return list(range(start, end + 1))
    
    def get_navigation_urls(
        self,
        base_url: str,
        params: PaginationParams,
        page_info: PageInfo
    ) -> Dict[str, Optional[str]]:
        """获取导航URL"""
        def build_url(page: int) -> str:
            new_params = PaginationParams(
                page=page,
                size=params.size,
                sort=params.sort,
                order=params.order,
                search=params.search,
                filters=params.filters
            )
            query_string = new_params.to_url_params(self.config)
            separator = '&' if '?' in base_url else '?'
            return f"{base_url}{separator}{query_string}" if query_string else base_url
        
        return {
            'first': build_url(1) if page_info.page > 1 else None,
            'previous': build_url(page_info.previous_page) if page_info.has_previous else None,
            'next': build_url(page_info.next_page) if page_info.has_next else None,
            'last': build_url(page_info.total_pages) if page_info.page < page_info.total_pages else None
        }


# =============================================================================
# 工具函数
# =============================================================================

def paginate_list(
    items: List[T],
    page: int = 1,
    size: int = 20,
    sort_key: Optional[str] = None,
    reverse: bool = False
) -> PageResult[T]:
    """简单的列表分页函数"""
    # 排序
    if sort_key:
        try:
            items = sorted(items, key=lambda x: getattr(x, sort_key), reverse=reverse)
        except AttributeError:
            logger.warning(f"Sort key '{sort_key}' not found in items")
    
    # 创建分页参数
    params = PaginationParams(page=page, size=size)
    
    # 分页
    paginator = Paginator()
    return paginator.paginate(items, params)


def calculate_page_from_offset(offset: int, size: int) -> int:
    """从偏移量计算页码"""
    return (offset // size) + 1


def calculate_offset_from_page(page: int, size: int) -> int:
    """从页码计算偏移量"""
    return (page - 1) * size


def validate_pagination_params(
    page: Optional[int] = None,
    size: Optional[int] = None,
    config: Optional[PaginationConfig] = None
) -> tuple[int, int]:
    """验证分页参数"""
    if config is None:
        config = DEFAULT_CONFIG
    
    # 验证页码
    if page is None or page < 1:
        page = 1
    
    # 验证页面大小
    if size is None:
        size = config.default_page_size
    else:
        size = max(config.min_page_size, min(config.max_page_size, size))
    
    return page, size


def create_pagination_response(
    items: List[T],
    total: int,
    page: int,
    size: int,
    base_url: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    item_serializer: Optional[callable] = None
) -> Dict[str, Any]:
    """创建标准的分页响应"""
    # 创建分页参数
    params = PaginationParams(
        page=page,
        size=size,
        filters=extra_params or {}
    )
    
    # 创建分页器
    paginator = Paginator()
    
    # 创建分页结果
    page_result = paginator.create_page_result(items, total, params)
    
    # 基础响应
    response = page_result.to_dict(item_serializer)
    
    # 添加导航链接
    if base_url:
        navigation = paginator.get_navigation_urls(base_url, params, page_result.page_info)
        response['links'] = navigation
    
    # 添加页码列表
    page_numbers = paginator.get_page_numbers(page_result.page_info)
    response['pagination']['pages'] = page_numbers
    
    return response


def merge_pagination_params(
    base_params: Dict[str, Any],
    new_params: Dict[str, Any],
    config: Optional[PaginationConfig] = None
) -> Dict[str, Any]:
    """合并分页参数"""
    if config is None:
        config = DEFAULT_CONFIG
    
    # 合并参数
    merged = base_params.copy()
    merged.update(new_params)
    
    # 如果页码或大小改变，重置页码为1
    if (config.size_param in new_params and 
        new_params[config.size_param] != base_params.get(config.size_param)):
        merged[config.page_param] = 1
    
    return merged


# =============================================================================
# 分页装饰器
# =============================================================================

def paginated(
    default_size: int = 20,
    max_size: int = 100,
    config: Optional[PaginationConfig] = None
):
    """分页装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 从kwargs中提取分页参数
            page = kwargs.pop('page', 1)
            size = kwargs.pop('size', default_size)
            
            # 验证参数
            page, size = validate_pagination_params(page, size, config)
            
            # 添加分页参数到kwargs
            kwargs['page'] = page
            kwargs['size'] = size
            kwargs['offset'] = calculate_offset_from_page(page, size)
            kwargs['limit'] = size
            
            # 执行原函数
            return func(*args, **kwargs)
        
        return wrapper
    return decorator