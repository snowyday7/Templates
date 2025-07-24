# -*- coding: utf-8 -*-
"""
高级工具模块
提供数据处理、文件操作、网络工具、加密解密等高级实用功能
"""

import asyncio
import aiofiles
import aiohttp
import json
import csv
import xml.etree.ElementTree as ET
import yaml
import zipfile
import tarfile
import gzip
import shutil
import tempfile
import hashlib
import hmac
import base64
import uuid
import re
import math
import statistics
from typing import (
    Dict, List, Any, Optional, Union, Callable, Iterator,
    AsyncIterator, Tuple, Set
)
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urljoin, quote, unquote
from collections import defaultdict, Counter
import logging
from contextlib import asynccontextmanager, contextmanager
import io
import mimetypes
from PIL import Image
import qrcode
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class DataFormat(Enum):
    """数据格式枚举"""
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    EXCEL = "excel"
    PARQUET = "parquet"


class CompressionType(Enum):
    """压缩类型枚举"""
    ZIP = "zip"
    TAR = "tar"
    TAR_GZ = "tar.gz"
    TAR_BZ2 = "tar.bz2"
    GZIP = "gzip"


class ImageFormat(Enum):
    """图片格式枚举"""
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"
    WEBP = "webp"


@dataclass
class ProcessingResult:
    """处理结果"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


class DataProcessor:
    """数据处理器"""

    @staticmethod
    def convert_format(
        data: Any,
        from_format: DataFormat,
        to_format: DataFormat
    ) -> ProcessingResult:
        """转换数据格式"""
        try:
            start_time = datetime.now()
            
            # 首先解析源格式
            if from_format == DataFormat.JSON:
                if isinstance(data, str):
                    parsed_data = json.loads(data)
                else:
                    parsed_data = data
            elif from_format == DataFormat.XML:
                if isinstance(data, str):
                    root = ET.fromstring(data)
                    parsed_data = DataProcessor._xml_to_dict(root)
                else:
                    parsed_data = data
            elif from_format == DataFormat.CSV:
                if isinstance(data, str):
                    reader = csv.DictReader(io.StringIO(data))
                    parsed_data = list(reader)
                else:
                    parsed_data = data
            elif from_format == DataFormat.YAML:
                if isinstance(data, str):
                    parsed_data = yaml.safe_load(data)
                else:
                    parsed_data = data
            else:
                parsed_data = data

            # 转换为目标格式
            if to_format == DataFormat.JSON:
                result_data = json.dumps(parsed_data, indent=2, ensure_ascii=False)
            elif to_format == DataFormat.XML:
                result_data = DataProcessor._dict_to_xml(parsed_data)
            elif to_format == DataFormat.CSV:
                if isinstance(parsed_data, list) and parsed_data:
                    output = io.StringIO()
                    writer = csv.DictWriter(output, fieldnames=parsed_data[0].keys())
                    writer.writeheader()
                    writer.writerows(parsed_data)
                    result_data = output.getvalue()
                else:
                    result_data = ""
            elif to_format == DataFormat.YAML:
                result_data = yaml.dump(parsed_data, default_flow_style=False)
            else:
                result_data = parsed_data

            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                data=result_data,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Data format conversion error: {e}")
            return ProcessingResult(
                success=False,
                error=str(e)
            )

    @staticmethod
    def _xml_to_dict(element: ET.Element) -> Dict[str, Any]:
        """XML转字典"""
        result = {}
        
        # 处理属性
        if element.attrib:
            result["@attributes"] = element.attrib
        
        # 处理子元素
        children = list(element)
        if children:
            child_dict = defaultdict(list)
            for child in children:
                child_data = DataProcessor._xml_to_dict(child)
                child_dict[child.tag].append(child_data)
            
            for tag, values in child_dict.items():
                if len(values) == 1:
                    result[tag] = values[0]
                else:
                    result[tag] = values
        
        # 处理文本内容
        if element.text and element.text.strip():
            if result:
                result["#text"] = element.text.strip()
            else:
                return element.text.strip()
        
        return result

    @staticmethod
    def _dict_to_xml(data: Dict[str, Any], root_name: str = "root") -> str:
        """字典转XML"""
        def build_element(name: str, value: Any) -> ET.Element:
            element = ET.Element(name)
            
            if isinstance(value, dict):
                for k, v in value.items():
                    if k == "@attributes":
                        element.attrib.update(v)
                    elif k == "#text":
                        element.text = str(v)
                    else:
                        child = build_element(k, v)
                        element.append(child)
            elif isinstance(value, list):
                for item in value:
                    child = build_element(name[:-1] if name.endswith('s') else 'item', item)
                    element.append(child)
            else:
                element.text = str(value)
            
            return element
        
        root = build_element(root_name, data)
        return ET.tostring(root, encoding='unicode')

    @staticmethod
    def validate_data(
        data: Any,
        schema: Dict[str, Any],
        format_type: DataFormat = DataFormat.JSON
    ) -> ProcessingResult:
        """验证数据格式"""
        try:
            # 这里可以集成jsonschema等验证库
            # 简化实现
            if format_type == DataFormat.JSON:
                if not isinstance(data, (dict, list)):
                    return ProcessingResult(
                        success=False,
                        error="Invalid JSON data type"
                    )
            
            return ProcessingResult(
                success=True,
                data=data,
                metadata={"validation": "passed"}
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )

    @staticmethod
    def clean_data(data: List[Dict[str, Any]]) -> ProcessingResult:
        """清洗数据"""
        try:
            cleaned_data = []
            
            for item in data:
                cleaned_item = {}
                for key, value in item.items():
                    # 清理空值
                    if value is not None and value != "":
                        # 清理字符串
                        if isinstance(value, str):
                            value = value.strip()
                            # 移除多余空格
                            value = re.sub(r'\s+', ' ', value)
                        
                        cleaned_item[key] = value
                
                if cleaned_item:  # 只添加非空记录
                    cleaned_data.append(cleaned_item)
            
            return ProcessingResult(
                success=True,
                data=cleaned_data,
                metadata={
                    "original_count": len(data),
                    "cleaned_count": len(cleaned_data)
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )


class FileManager:
    """文件管理器"""

    @staticmethod
    async def read_file(
        file_path: Union[str, Path],
        encoding: str = "utf-8"
    ) -> ProcessingResult:
        """异步读取文件"""
        try:
            async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                content = await f.read()
            
            return ProcessingResult(
                success=True,
                data=content,
                metadata={
                    "file_path": str(file_path),
                    "size": len(content)
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )

    @staticmethod
    async def write_file(
        file_path: Union[str, Path],
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True
    ) -> ProcessingResult:
        """异步写入文件"""
        try:
            file_path = Path(file_path)
            
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(file_path, 'w', encoding=encoding) as f:
                await f.write(content)
            
            return ProcessingResult(
                success=True,
                metadata={
                    "file_path": str(file_path),
                    "size": len(content)
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )

    @staticmethod
    def compress_files(
        file_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        compression_type: CompressionType = CompressionType.ZIP
    ) -> ProcessingResult:
        """压缩文件"""
        try:
            output_path = Path(output_path)
            
            if compression_type == CompressionType.ZIP:
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for file_path in file_paths:
                        file_path = Path(file_path)
                        if file_path.exists():
                            zf.write(file_path, file_path.name)
            
            elif compression_type in [CompressionType.TAR, CompressionType.TAR_GZ, CompressionType.TAR_BZ2]:
                mode_map = {
                    CompressionType.TAR: 'w',
                    CompressionType.TAR_GZ: 'w:gz',
                    CompressionType.TAR_BZ2: 'w:bz2'
                }
                
                with tarfile.open(output_path, mode_map[compression_type]) as tf:
                    for file_path in file_paths:
                        file_path = Path(file_path)
                        if file_path.exists():
                            tf.add(file_path, file_path.name)
            
            return ProcessingResult(
                success=True,
                metadata={
                    "output_path": str(output_path),
                    "file_count": len(file_paths),
                    "compression_type": compression_type.value
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )

    @staticmethod
    def extract_archive(
        archive_path: Union[str, Path],
        extract_to: Union[str, Path]
    ) -> ProcessingResult:
        """解压文件"""
        try:
            archive_path = Path(archive_path)
            extract_to = Path(extract_to)
            extract_to.mkdir(parents=True, exist_ok=True)
            
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(extract_to)
                    extracted_files = zf.namelist()
            
            elif archive_path.suffix.lower() in ['.tar', '.gz', '.bz2']:
                with tarfile.open(archive_path, 'r:*') as tf:
                    tf.extractall(extract_to)
                    extracted_files = tf.getnames()
            
            else:
                return ProcessingResult(
                    success=False,
                    error=f"Unsupported archive format: {archive_path.suffix}"
                )
            
            return ProcessingResult(
                success=True,
                metadata={
                    "archive_path": str(archive_path),
                    "extract_to": str(extract_to),
                    "extracted_files": extracted_files
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )

    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> ProcessingResult:
        """获取文件信息"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return ProcessingResult(
                    success=False,
                    error="File does not exist"
                )
            
            stat = file_path.stat()
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            file_info = {
                "name": file_path.name,
                "path": str(file_path.absolute()),
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "mime_type": mime_type,
                "extension": file_path.suffix,
                "is_file": file_path.is_file(),
                "is_dir": file_path.is_dir()
            }
            
            return ProcessingResult(
                success=True,
                data=file_info
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )


class NetworkTools:
    """网络工具"""

    @staticmethod
    async def download_file(
        url: str,
        output_path: Union[str, Path],
        chunk_size: int = 8192,
        headers: Optional[Dict[str, str]] = None
    ) -> ProcessingResult:
        """下载文件"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        total_size = int(response.headers.get('content-length', 0))
                        downloaded = 0
                        
                        async with aiofiles.open(output_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(chunk_size):
                                await f.write(chunk)
                                downloaded += len(chunk)
                        
                        return ProcessingResult(
                            success=True,
                            metadata={
                                "url": url,
                                "output_path": str(output_path),
                                "size": downloaded,
                                "total_size": total_size
                            }
                        )
                    else:
                        return ProcessingResult(
                            success=False,
                            error=f"HTTP {response.status}: {response.reason}"
                        )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )

    @staticmethod
    async def make_request(
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        timeout: int = 30
    ) -> ProcessingResult:
        """发送HTTP请求"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.request(method, url, headers=headers, json=data) as response:
                    content_type = response.headers.get('content-type', '')
                    
                    if 'application/json' in content_type:
                        response_data = await response.json()
                    else:
                        response_data = await response.text()
                    
                    return ProcessingResult(
                        success=response.status < 400,
                        data=response_data,
                        metadata={
                            "status_code": response.status,
                            "headers": dict(response.headers),
                            "url": str(response.url)
                        }
                    )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )

    @staticmethod
    def validate_url(url: str) -> ProcessingResult:
        """验证URL"""
        try:
            parsed = urlparse(url)
            
            if not parsed.scheme or not parsed.netloc:
                return ProcessingResult(
                    success=False,
                    error="Invalid URL format"
                )
            
            return ProcessingResult(
                success=True,
                data={
                    "scheme": parsed.scheme,
                    "netloc": parsed.netloc,
                    "path": parsed.path,
                    "params": parsed.params,
                    "query": parsed.query,
                    "fragment": parsed.fragment
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )


class ImageProcessor:
    """图片处理器"""

    @staticmethod
    def resize_image(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        size: Tuple[int, int],
        maintain_aspect: bool = True
    ) -> ProcessingResult:
        """调整图片大小"""
        try:
            with Image.open(input_path) as img:
                if maintain_aspect:
                    img.thumbnail(size, Image.Resampling.LANCZOS)
                else:
                    img = img.resize(size, Image.Resampling.LANCZOS)
                
                img.save(output_path)
            
            return ProcessingResult(
                success=True,
                metadata={
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "new_size": img.size
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )

    @staticmethod
    def convert_format(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        format_type: ImageFormat
    ) -> ProcessingResult:
        """转换图片格式"""
        try:
            with Image.open(input_path) as img:
                # 处理透明度
                if format_type == ImageFormat.JPEG and img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                
                img.save(output_path, format=format_type.value.upper())
            
            return ProcessingResult(
                success=True,
                metadata={
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "format": format_type.value
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )

    @staticmethod
    def generate_qr_code(
        data: str,
        output_path: Union[str, Path],
        size: int = 10,
        border: int = 4
    ) -> ProcessingResult:
        """生成二维码"""
        try:
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=size,
                border=border,
            )
            qr.add_data(data)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            img.save(output_path)
            
            return ProcessingResult(
                success=True,
                metadata={
                    "data": data,
                    "output_path": str(output_path),
                    "size": img.size
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )


class PDFGenerator:
    """PDF生成器"""

    @staticmethod
    def create_simple_pdf(
        content: str,
        output_path: Union[str, Path],
        title: str = "Document",
        page_size: Tuple[float, float] = letter
    ) -> ProcessingResult:
        """创建简单PDF"""
        try:
            c = canvas.Canvas(str(output_path), pagesize=page_size)
            width, height = page_size
            
            # 设置标题
            c.setFont("Helvetica-Bold", 16)
            c.drawString(72, height - 72, title)
            
            # 设置内容
            c.setFont("Helvetica", 12)
            y_position = height - 120
            
            lines = content.split('\n')
            for line in lines:
                if y_position < 72:  # 新页面
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y_position = height - 72
                
                c.drawString(72, y_position, line)
                y_position -= 14
            
            c.save()
            
            return ProcessingResult(
                success=True,
                metadata={
                    "output_path": str(output_path),
                    "title": title,
                    "page_count": c.getPageNumber()
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )


class CryptoTools:
    """加密工具"""

    @staticmethod
    def generate_key() -> str:
        """生成加密密钥"""
        return Fernet.generate_key().decode()

    @staticmethod
    def encrypt_data(data: str, key: str) -> ProcessingResult:
        """加密数据"""
        try:
            f = Fernet(key.encode())
            encrypted_data = f.encrypt(data.encode())
            
            return ProcessingResult(
                success=True,
                data=encrypted_data.decode()
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )

    @staticmethod
    def decrypt_data(encrypted_data: str, key: str) -> ProcessingResult:
        """解密数据"""
        try:
            f = Fernet(key.encode())
            decrypted_data = f.decrypt(encrypted_data.encode())
            
            return ProcessingResult(
                success=True,
                data=decrypted_data.decode()
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )

    @staticmethod
    def hash_data(
        data: str,
        algorithm: str = "sha256",
        salt: Optional[str] = None
    ) -> ProcessingResult:
        """哈希数据"""
        try:
            if salt:
                data = data + salt
            
            hash_obj = hashlib.new(algorithm)
            hash_obj.update(data.encode())
            hash_value = hash_obj.hexdigest()
            
            return ProcessingResult(
                success=True,
                data=hash_value,
                metadata={
                    "algorithm": algorithm,
                    "salt_used": bool(salt)
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )


class StatisticsCalculator:
    """统计计算器"""

    @staticmethod
    def calculate_basic_stats(data: List[Union[int, float]]) -> ProcessingResult:
        """计算基础统计信息"""
        try:
            if not data:
                return ProcessingResult(
                    success=False,
                    error="Empty data list"
                )
            
            stats = {
                "count": len(data),
                "sum": sum(data),
                "mean": statistics.mean(data),
                "median": statistics.median(data),
                "mode": statistics.mode(data) if len(set(data)) < len(data) else None,
                "min": min(data),
                "max": max(data),
                "range": max(data) - min(data),
                "variance": statistics.variance(data) if len(data) > 1 else 0,
                "std_dev": statistics.stdev(data) if len(data) > 1 else 0
            }
            
            return ProcessingResult(
                success=True,
                data=stats
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )

    @staticmethod
    def calculate_percentiles(
        data: List[Union[int, float]],
        percentiles: List[float] = [25, 50, 75, 90, 95, 99]
    ) -> ProcessingResult:
        """计算百分位数"""
        try:
            if not data:
                return ProcessingResult(
                    success=False,
                    error="Empty data list"
                )
            
            sorted_data = sorted(data)
            result = {}
            
            for p in percentiles:
                index = (len(sorted_data) - 1) * (p / 100)
                lower = int(index)
                upper = min(lower + 1, len(sorted_data) - 1)
                weight = index - lower
                
                value = sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
                result[f"p{p}"] = value
            
            return ProcessingResult(
                success=True,
                data=result
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )


# 使用示例
if __name__ == "__main__":
    async def example_usage():
        # 数据格式转换
        data = {"name": "John", "age": 30, "city": "New York"}
        result = DataProcessor.convert_format(
            data, DataFormat.JSON, DataFormat.XML
        )
        print(f"JSON to XML: {result.success}")
        
        # 文件操作
        await FileManager.write_file(
            "test.txt", "Hello, World!"
        )
        file_result = await FileManager.read_file("test.txt")
        print(f"File content: {file_result.data}")
        
        # 网络请求
        url_result = NetworkTools.validate_url("https://example.com")
        print(f"URL validation: {url_result.success}")
        
        # 加密解密
        key = CryptoTools.generate_key()
        encrypt_result = CryptoTools.encrypt_data("Secret message", key)
        if encrypt_result.success:
            decrypt_result = CryptoTools.decrypt_data(encrypt_result.data, key)
            print(f"Decrypted: {decrypt_result.data}")
        
        # 统计计算
        numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        stats_result = StatisticsCalculator.calculate_basic_stats(numbers)
        print(f"Statistics: {stats_result.data}")
    
    # 运行示例
    asyncio.run(example_usage())