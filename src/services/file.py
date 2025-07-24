#!/usr/bin/env python3
"""
文件服务

提供文件管理相关的业务逻辑，包括：
1. 文件上传和下载
2. 文件存储管理
3. 图片处理
4. 文件安全检查
5. 文件元数据管理
6. 文件清理和归档
"""

import os
import hashlib
import mimetypes
import aiofiles
from typing import Optional, List, Dict, Any, BinaryIO, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image, ImageOps
from io import BytesIO
import magic

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from .base import BaseService
from ..models.file import FileRecord
from ..core.exceptions import FileException, ValidationException
from ..utils.logger import get_logger
from ..core.config import get_settings


# =============================================================================
# 常量定义
# =============================================================================

# 允许的文件类型
ALLOWED_IMAGE_TYPES = {
    'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'
}

ALLOWED_DOCUMENT_TYPES = {
    'application/pdf', 'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'text/plain', 'text/csv'
}

ALLOWED_ARCHIVE_TYPES = {
    'application/zip', 'application/x-rar-compressed',
    'application/x-7z-compressed', 'application/x-tar',
    'application/gzip'
}

# 文件大小限制（字节）
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50MB
MAX_ARCHIVE_SIZE = 100 * 1024 * 1024  # 100MB

# 图片尺寸限制
MAX_IMAGE_DIMENSION = 4096  # 4K

# 缩略图尺寸
THUMBNAIL_SIZES = {
    'small': (150, 150),
    'medium': (300, 300),
    'large': (600, 600)
}


# =============================================================================
# 文件服务类
# =============================================================================

class FileService(BaseService[FileRecord]):
    """
    文件服务类
    
    提供文件管理相关的业务逻辑操作
    """
    
    def __init__(self, db: AsyncSession):
        """
        初始化文件服务
        
        Args:
            db: 数据库会话
        """
        super().__init__(db, FileRecord)
        self.logger = get_logger(__name__)
        self.settings = get_settings()
        
        # 文件存储路径
        self.upload_dir = Path(self.settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.image_dir = self.upload_dir / "images"
        self.document_dir = self.upload_dir / "documents"
        self.archive_dir = self.upload_dir / "archives"
        self.thumbnail_dir = self.upload_dir / "thumbnails"
        
        for directory in [self.image_dir, self.document_dir, self.archive_dir, self.thumbnail_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 文件上传
    # =========================================================================
    
    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: Optional[str] = None,
        user_id: Optional[int] = None,
        category: str = "general",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> FileRecord:
        """
        上传文件
        
        Args:
            file_content: 文件内容
            filename: 文件名
            content_type: 内容类型
            user_id: 用户ID
            category: 文件分类
            description: 文件描述
            tags: 文件标签
            
        Returns:
            文件记录
        """
        try:
            # 验证文件
            await self._validate_file(file_content, filename, content_type)
            
            # 检测文件类型
            if not content_type:
                content_type = self._detect_content_type(file_content, filename)
            
            # 生成文件信息
            file_hash = self._calculate_file_hash(file_content)
            file_size = len(file_content)
            
            # 检查是否已存在相同文件
            existing_file = await self._get_file_by_hash(file_hash)
            if existing_file:
                self.logger.info(f"File already exists: {file_hash}")
                return existing_file
            
            # 确定存储路径
            storage_path = await self._get_storage_path(content_type, filename, file_hash)
            
            # 保存文件
            await self._save_file_to_disk(file_content, storage_path)
            
            # 生成缩略图（如果是图片）
            thumbnail_paths = {}
            if self._is_image(content_type):
                thumbnail_paths = await self._generate_thumbnails(file_content, file_hash)
            
            # 创建文件记录
            file_record = FileRecord(
                filename=filename,
                original_filename=filename,
                file_path=str(storage_path.relative_to(self.upload_dir)),
                file_size=file_size,
                content_type=content_type,
                file_hash=file_hash,
                user_id=user_id,
                category=category,
                description=description,
                tags=tags or [],
                thumbnail_paths=thumbnail_paths,
                metadata=await self._extract_metadata(file_content, content_type)
            )
            
            # 保存到数据库
            created_file = await self.create(file_record)
            
            self.logger.info(f"File uploaded successfully: {filename} ({file_hash})")
            
            return created_file
            
        except Exception as e:
            self.logger.error(f"Error uploading file {filename}: {e}")
            raise FileException(f"文件上传失败: {str(e)}")
    
    async def upload_image(
        self,
        image_content: bytes,
        filename: str,
        user_id: Optional[int] = None,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        quality: int = 85
    ) -> FileRecord:
        """
        上传图片（带压缩和优化）
        
        Args:
            image_content: 图片内容
            filename: 文件名
            user_id: 用户ID
            max_width: 最大宽度
            max_height: 最大高度
            quality: 压缩质量
            
        Returns:
            文件记录
        """
        try:
            # 验证是否为图片
            content_type = self._detect_content_type(image_content, filename)
            if not self._is_image(content_type):
                raise ValidationException("文件不是有效的图片格式")
            
            # 处理图片
            processed_image = await self._process_image(
                image_content, max_width, max_height, quality
            )
            
            # 上传处理后的图片
            return await self.upload_file(
                processed_image,
                filename,
                content_type,
                user_id,
                category="image"
            )
            
        except Exception as e:
            self.logger.error(f"Error uploading image {filename}: {e}")
            raise FileException(f"图片上传失败: {str(e)}")
    
    # =========================================================================
    # 文件下载和访问
    # =========================================================================
    
    async def get_file_content(self, file_id: int) -> Tuple[bytes, str, str]:
        """
        获取文件内容
        
        Args:
            file_id: 文件ID
            
        Returns:
            (文件内容, 文件名, 内容类型)
        """
        try:
            # 获取文件记录
            file_record = await self.get_by_id(file_id)
            if not file_record:
                raise FileException("文件不存在")
            
            # 检查文件是否存在
            file_path = self.upload_dir / file_record.file_path
            if not file_path.exists():
                raise FileException("文件已被删除")
            
            # 读取文件内容
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            
            # 更新下载次数
            await self._update_download_count(file_record)
            
            return content, file_record.filename, file_record.content_type
            
        except Exception as e:
            self.logger.error(f"Error getting file content {file_id}: {e}")
            raise FileException(f"获取文件内容失败: {str(e)}")
    
    async def get_thumbnail(
        self,
        file_id: int,
        size: str = "medium"
    ) -> Tuple[bytes, str]:
        """
        获取缩略图
        
        Args:
            file_id: 文件ID
            size: 缩略图尺寸（small, medium, large）
            
        Returns:
            (缩略图内容, 内容类型)
        """
        try:
            # 获取文件记录
            file_record = await self.get_by_id(file_id)
            if not file_record:
                raise FileException("文件不存在")
            
            # 检查是否有缩略图
            if not file_record.thumbnail_paths or size not in file_record.thumbnail_paths:
                raise FileException("缩略图不存在")
            
            # 读取缩略图
            thumbnail_path = self.upload_dir / file_record.thumbnail_paths[size]
            if not thumbnail_path.exists():
                raise FileException("缩略图文件已被删除")
            
            async with aiofiles.open(thumbnail_path, 'rb') as f:
                content = await f.read()
            
            return content, "image/jpeg"
            
        except Exception as e:
            self.logger.error(f"Error getting thumbnail {file_id}: {e}")
            raise FileException(f"获取缩略图失败: {str(e)}")
    
    async def get_file_url(self, file_id: int) -> str:
        """
        获取文件访问URL
        
        Args:
            file_id: 文件ID
            
        Returns:
            文件URL
        """
        try:
            file_record = await self.get_by_id(file_id)
            if not file_record:
                raise FileException("文件不存在")
            
            # 生成访问URL
            base_url = self.settings.BASE_URL.rstrip('/')
            return f"{base_url}/api/v1/files/{file_id}/download"
            
        except Exception as e:
            self.logger.error(f"Error getting file URL {file_id}: {e}")
            raise FileException(f"获取文件URL失败: {str(e)}")
    
    # =========================================================================
    # 文件管理
    # =========================================================================
    
    async def get_user_files(
        self,
        user_id: int,
        category: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        获取用户文件列表
        
        Args:
            user_id: 用户ID
            category: 文件分类
            page: 页码
            page_size: 每页大小
            
        Returns:
            文件列表和分页信息
        """
        try:
            # 构建查询条件
            conditions = [FileRecord.user_id == user_id, FileRecord.is_deleted == False]
            
            if category:
                conditions.append(FileRecord.category == category)
            
            # 获取文件列表
            result = await self.get_with_pagination(
                conditions=and_(*conditions),
                page=page,
                page_size=page_size,
                order_by=[FileRecord.created_at.desc()]
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting user files {user_id}: {e}")
            raise FileException(f"获取用户文件失败: {str(e)}")
    
    async def search_files(
        self,
        query: str,
        user_id: Optional[int] = None,
        category: Optional[str] = None,
        content_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        搜索文件
        
        Args:
            query: 搜索关键词
            user_id: 用户ID
            category: 文件分类
            content_type: 内容类型
            page: 页码
            page_size: 每页大小
            
        Returns:
            搜索结果和分页信息
        """
        try:
            # 构建查询条件
            conditions = [FileRecord.is_deleted == False]
            
            # 关键词搜索
            if query:
                search_conditions = [
                    FileRecord.filename.ilike(f"%{query}%"),
                    FileRecord.description.ilike(f"%{query}%"),
                    FileRecord.tags.op('@>')([query])  # PostgreSQL数组包含操作
                ]
                conditions.append(or_(*search_conditions))
            
            # 其他过滤条件
            if user_id:
                conditions.append(FileRecord.user_id == user_id)
            if category:
                conditions.append(FileRecord.category == category)
            if content_type:
                conditions.append(FileRecord.content_type == content_type)
            
            # 执行搜索
            result = await self.get_with_pagination(
                conditions=and_(*conditions),
                page=page,
                page_size=page_size,
                order_by=[FileRecord.created_at.desc()]
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error searching files: {e}")
            raise FileException(f"搜索文件失败: {str(e)}")
    
    async def update_file_info(
        self,
        file_id: int,
        filename: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> FileRecord:
        """
        更新文件信息
        
        Args:
            file_id: 文件ID
            filename: 新文件名
            description: 新描述
            tags: 新标签
            category: 新分类
            
        Returns:
            更新后的文件记录
        """
        try:
            # 获取文件记录
            file_record = await self.get_by_id(file_id)
            if not file_record:
                raise FileException("文件不存在")
            
            # 更新字段
            update_data = {}
            if filename is not None:
                update_data['filename'] = filename
            if description is not None:
                update_data['description'] = description
            if tags is not None:
                update_data['tags'] = tags
            if category is not None:
                update_data['category'] = category
            
            if update_data:
                updated_file = await self.update(file_id, update_data)
                self.logger.info(f"File info updated: {file_id}")
                return updated_file
            
            return file_record
            
        except Exception as e:
            self.logger.error(f"Error updating file info {file_id}: {e}")
            raise FileException(f"更新文件信息失败: {str(e)}")
    
    async def delete_file(self, file_id: int, permanent: bool = False) -> bool:
        """
        删除文件
        
        Args:
            file_id: 文件ID
            permanent: 是否永久删除
            
        Returns:
            是否删除成功
        """
        try:
            # 获取文件记录
            file_record = await self.get_by_id(file_id)
            if not file_record:
                raise FileException("文件不存在")
            
            if permanent:
                # 永久删除：删除物理文件和数据库记录
                await self._delete_physical_file(file_record)
                await self.delete(file_id)
                self.logger.info(f"File permanently deleted: {file_id}")
            else:
                # 软删除：只标记为已删除
                await self.update(file_id, {'is_deleted': True})
                self.logger.info(f"File soft deleted: {file_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting file {file_id}: {e}")
            raise FileException(f"删除文件失败: {str(e)}")
    
    # =========================================================================
    # 文件统计
    # =========================================================================
    
    async def get_file_statistics(
        self,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        获取文件统计信息
        
        Args:
            user_id: 用户ID（可选）
            
        Returns:
            统计信息
        """
        try:
            # 构建基础查询
            base_query = select(FileRecord).where(FileRecord.is_deleted == False)
            
            if user_id:
                base_query = base_query.where(FileRecord.user_id == user_id)
            
            # 总文件数
            total_count_query = select(func.count(FileRecord.id)).select_from(base_query.subquery())
            total_count_result = await self.db.execute(total_count_query)
            total_count = total_count_result.scalar()
            
            # 总文件大小
            total_size_query = select(func.sum(FileRecord.file_size)).select_from(base_query.subquery())
            total_size_result = await self.db.execute(total_size_query)
            total_size = total_size_result.scalar() or 0
            
            # 按分类统计
            category_query = select(
                FileRecord.category,
                func.count(FileRecord.id).label('count'),
                func.sum(FileRecord.file_size).label('size')
            ).where(FileRecord.is_deleted == False)
            
            if user_id:
                category_query = category_query.where(FileRecord.user_id == user_id)
            
            category_query = category_query.group_by(FileRecord.category)
            category_result = await self.db.execute(category_query)
            category_stats = [
                {
                    'category': row.category,
                    'count': row.count,
                    'size': row.size or 0
                }
                for row in category_result.fetchall()
            ]
            
            # 按内容类型统计
            content_type_query = select(
                FileRecord.content_type,
                func.count(FileRecord.id).label('count'),
                func.sum(FileRecord.file_size).label('size')
            ).where(FileRecord.is_deleted == False)
            
            if user_id:
                content_type_query = content_type_query.where(FileRecord.user_id == user_id)
            
            content_type_query = content_type_query.group_by(FileRecord.content_type)
            content_type_result = await self.db.execute(content_type_query)
            content_type_stats = [
                {
                    'content_type': row.content_type,
                    'count': row.count,
                    'size': row.size or 0
                }
                for row in content_type_result.fetchall()
            ]
            
            return {
                'total_files': total_count,
                'total_size': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'by_category': category_stats,
                'by_content_type': content_type_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting file statistics: {e}")
            raise FileException(f"获取文件统计失败: {str(e)}")
    
    # =========================================================================
    # 文件清理
    # =========================================================================
    
    async def cleanup_orphaned_files(self) -> Dict[str, Any]:
        """
        清理孤立文件（数据库中不存在但磁盘上存在的文件）
        
        Returns:
            清理结果
        """
        try:
            orphaned_files = []
            cleaned_size = 0
            
            # 获取所有数据库中的文件路径
            db_files_query = select(FileRecord.file_path).where(FileRecord.is_deleted == False)
            db_files_result = await self.db.execute(db_files_query)
            db_file_paths = {row.file_path for row in db_files_result.fetchall()}
            
            # 扫描磁盘文件
            for root, dirs, files in os.walk(self.upload_dir):
                for file in files:
                    file_path = Path(root) / file
                    relative_path = str(file_path.relative_to(self.upload_dir))
                    
                    # 跳过缩略图目录
                    if 'thumbnails' in relative_path:
                        continue
                    
                    # 检查是否为孤立文件
                    if relative_path not in db_file_paths:
                        file_size = file_path.stat().st_size
                        orphaned_files.append({
                            'path': relative_path,
                            'size': file_size
                        })
                        
                        # 删除孤立文件
                        file_path.unlink()
                        cleaned_size += file_size
            
            self.logger.info(f"Cleaned {len(orphaned_files)} orphaned files, {cleaned_size} bytes")
            
            return {
                'orphaned_files_count': len(orphaned_files),
                'cleaned_size': cleaned_size,
                'cleaned_size_mb': round(cleaned_size / (1024 * 1024), 2),
                'orphaned_files': orphaned_files[:10]  # 只返回前10个
            }
            
        except Exception as e:
            self.logger.error(f"Error cleaning orphaned files: {e}")
            raise FileException(f"清理孤立文件失败: {str(e)}")
    
    async def cleanup_deleted_files(self, days_old: int = 30) -> Dict[str, Any]:
        """
        清理已删除的文件（永久删除超过指定天数的软删除文件）
        
        Args:
            days_old: 删除多少天前的文件
            
        Returns:
            清理结果
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # 查询需要清理的文件
            cleanup_query = select(FileRecord).where(
                and_(
                    FileRecord.is_deleted == True,
                    FileRecord.updated_at < cutoff_date
                )
            )
            cleanup_result = await self.db.execute(cleanup_query)
            files_to_cleanup = cleanup_result.fetchall()
            
            cleaned_count = 0
            cleaned_size = 0
            
            for file_record in files_to_cleanup:
                try:
                    # 删除物理文件
                    await self._delete_physical_file(file_record)
                    
                    # 删除数据库记录
                    await self.delete(file_record.id)
                    
                    cleaned_count += 1
                    cleaned_size += file_record.file_size
                    
                except Exception as e:
                    self.logger.error(f"Error cleaning file {file_record.id}: {e}")
                    continue
            
            self.logger.info(f"Cleaned {cleaned_count} deleted files, {cleaned_size} bytes")
            
            return {
                'cleaned_files_count': cleaned_count,
                'cleaned_size': cleaned_size,
                'cleaned_size_mb': round(cleaned_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error cleaning deleted files: {e}")
            raise FileException(f"清理已删除文件失败: {str(e)}")
    
    # =========================================================================
    # 私有辅助方法
    # =========================================================================
    
    async def _validate_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: Optional[str] = None
    ):
        """验证文件"""
        # 检查文件大小
        file_size = len(file_content)
        if file_size == 0:
            raise ValidationException("文件不能为空")
        
        # 检测内容类型
        if not content_type:
            content_type = self._detect_content_type(file_content, filename)
        
        # 检查文件类型
        allowed_types = ALLOWED_IMAGE_TYPES | ALLOWED_DOCUMENT_TYPES | ALLOWED_ARCHIVE_TYPES
        if content_type not in allowed_types:
            raise ValidationException(f"不支持的文件类型: {content_type}")
        
        # 检查文件大小限制
        if content_type in ALLOWED_IMAGE_TYPES and file_size > MAX_IMAGE_SIZE:
            raise ValidationException(f"图片文件过大，最大允许 {MAX_IMAGE_SIZE // (1024*1024)}MB")
        elif content_type in ALLOWED_DOCUMENT_TYPES and file_size > MAX_DOCUMENT_SIZE:
            raise ValidationException(f"文档文件过大，最大允许 {MAX_DOCUMENT_SIZE // (1024*1024)}MB")
        elif content_type in ALLOWED_ARCHIVE_TYPES and file_size > MAX_ARCHIVE_SIZE:
            raise ValidationException(f"压缩文件过大，最大允许 {MAX_ARCHIVE_SIZE // (1024*1024)}MB")
        
        # 检查文件名
        if not filename or len(filename) > 255:
            raise ValidationException("文件名无效")
        
        # 检查图片尺寸
        if self._is_image(content_type):
            await self._validate_image_dimensions(file_content)
    
    def _detect_content_type(self, file_content: bytes, filename: str) -> str:
        """检测文件内容类型"""
        try:
            # 使用python-magic检测
            mime_type = magic.from_buffer(file_content, mime=True)
            if mime_type:
                return mime_type
        except Exception:
            pass
        
        # 回退到基于文件扩展名的检测
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or 'application/octet-stream'
    
    def _calculate_file_hash(self, file_content: bytes) -> str:
        """计算文件哈希"""
        return hashlib.sha256(file_content).hexdigest()
    
    async def _get_file_by_hash(self, file_hash: str) -> Optional[FileRecord]:
        """根据哈希获取文件"""
        query = select(FileRecord).where(
            and_(
                FileRecord.file_hash == file_hash,
                FileRecord.is_deleted == False
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def _get_storage_path(self, content_type: str, filename: str, file_hash: str) -> Path:
        """获取存储路径"""
        # 根据内容类型确定目录
        if self._is_image(content_type):
            base_dir = self.image_dir
        elif content_type in ALLOWED_DOCUMENT_TYPES:
            base_dir = self.document_dir
        elif content_type in ALLOWED_ARCHIVE_TYPES:
            base_dir = self.archive_dir
        else:
            base_dir = self.upload_dir
        
        # 使用哈希的前两位作为子目录
        sub_dir = base_dir / file_hash[:2]
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名（使用哈希 + 原始扩展名）
        file_ext = Path(filename).suffix
        storage_filename = f"{file_hash}{file_ext}"
        
        return sub_dir / storage_filename
    
    async def _save_file_to_disk(self, file_content: bytes, file_path: Path):
        """保存文件到磁盘"""
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
    
    def _is_image(self, content_type: str) -> bool:
        """检查是否为图片"""
        return content_type in ALLOWED_IMAGE_TYPES
    
    async def _validate_image_dimensions(self, image_content: bytes):
        """验证图片尺寸"""
        try:
            with Image.open(BytesIO(image_content)) as img:
                width, height = img.size
                if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                    raise ValidationException(
                        f"图片尺寸过大，最大允许 {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}"
                    )
        except Exception as e:
            if isinstance(e, ValidationException):
                raise
            raise ValidationException("无效的图片文件")
    
    async def _process_image(
        self,
        image_content: bytes,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        quality: int = 85
    ) -> bytes:
        """处理图片（压缩、调整大小）"""
        try:
            with Image.open(BytesIO(image_content)) as img:
                # 转换为RGB（如果需要）
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # 调整尺寸
                if max_width or max_height:
                    img.thumbnail((max_width or img.width, max_height or img.height), Image.Resampling.LANCZOS)
                
                # 自动旋转（基于EXIF信息）
                img = ImageOps.exif_transpose(img)
                
                # 保存处理后的图片
                output = BytesIO()
                img.save(output, format='JPEG', quality=quality, optimize=True)
                return output.getvalue()
                
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            raise FileException(f"图片处理失败: {str(e)}")
    
    async def _generate_thumbnails(self, image_content: bytes, file_hash: str) -> Dict[str, str]:
        """生成缩略图"""
        thumbnail_paths = {}
        
        try:
            with Image.open(BytesIO(image_content)) as img:
                # 转换为RGB
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # 自动旋转
                img = ImageOps.exif_transpose(img)
                
                # 生成不同尺寸的缩略图
                for size_name, (width, height) in THUMBNAIL_SIZES.items():
                    # 创建缩略图
                    thumbnail = img.copy()
                    thumbnail.thumbnail((width, height), Image.Resampling.LANCZOS)
                    
                    # 保存缩略图
                    thumbnail_filename = f"{file_hash}_{size_name}.jpg"
                    thumbnail_path = self.thumbnail_dir / thumbnail_filename
                    
                    thumbnail.save(thumbnail_path, format='JPEG', quality=85, optimize=True)
                    
                    # 记录相对路径
                    relative_path = str(thumbnail_path.relative_to(self.upload_dir))
                    thumbnail_paths[size_name] = relative_path
                
        except Exception as e:
            self.logger.error(f"Error generating thumbnails: {e}")
            # 缩略图生成失败不影响文件上传
        
        return thumbnail_paths
    
    async def _extract_metadata(self, file_content: bytes, content_type: str) -> Dict[str, Any]:
        """提取文件元数据"""
        metadata = {}
        
        try:
            if self._is_image(content_type):
                # 提取图片元数据
                with Image.open(BytesIO(file_content)) as img:
                    metadata.update({
                        'width': img.width,
                        'height': img.height,
                        'format': img.format,
                        'mode': img.mode
                    })
                    
                    # 提取EXIF信息
                    if hasattr(img, '_getexif') and img._getexif():
                        exif = img._getexif()
                        if exif:
                            metadata['exif'] = {k: v for k, v in exif.items() if isinstance(v, (str, int, float))}
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")
        
        return metadata
    
    async def _update_download_count(self, file_record: FileRecord):
        """更新下载次数"""
        try:
            await self.update(file_record.id, {
                'download_count': (file_record.download_count or 0) + 1,
                'last_accessed_at': datetime.utcnow()
            })
        except Exception as e:
            self.logger.error(f"Error updating download count: {e}")
    
    async def _delete_physical_file(self, file_record: FileRecord):
        """删除物理文件"""
        try:
            # 删除主文件
            main_file_path = self.upload_dir / file_record.file_path
            if main_file_path.exists():
                main_file_path.unlink()
            
            # 删除缩略图
            if file_record.thumbnail_paths:
                for thumbnail_path in file_record.thumbnail_paths.values():
                    thumb_file_path = self.upload_dir / thumbnail_path
                    if thumb_file_path.exists():
                        thumb_file_path.unlink()
                        
        except Exception as e:
            self.logger.error(f"Error deleting physical file {file_record.id}: {e}")
            # 不抛出异常，避免影响数据库删除操作