#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品管理API

提供商品的CRUD操作、分类管理、库存管理等功能。
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..core.database import get_db
from ..models.product import Product, Category
from ..services.product_service import ProductService
from ..core.auth import get_current_user

router = APIRouter(prefix="/products", tags=["商品管理"])


class ProductCreate(BaseModel):
    """创建商品请求模型"""
    name: str
    description: Optional[str] = None
    price: float
    category_id: int
    stock_quantity: int = 0
    sku: Optional[str] = None
    images: Optional[List[str]] = []
    is_active: bool = True


class ProductUpdate(BaseModel):
    """更新商品请求模型"""
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    category_id: Optional[int] = None
    stock_quantity: Optional[int] = None
    sku: Optional[str] = None
    images: Optional[List[str]] = None
    is_active: Optional[bool] = None


class ProductResponse(BaseModel):
    """商品响应模型"""
    id: int
    name: str
    description: Optional[str]
    price: float
    category_id: int
    category_name: str
    stock_quantity: int
    sku: Optional[str]
    images: List[str]
    is_active: bool
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


@router.get("/", response_model=List[ProductResponse])
async def get_products(
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(20, ge=1, le=100, description="返回的记录数"),
    category_id: Optional[int] = Query(None, description="分类ID筛选"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    is_active: Optional[bool] = Query(None, description="是否激活"),
    db: Session = Depends(get_db)
):
    """
    获取商品列表
    
    支持分页、分类筛选、关键词搜索等功能。
    """
    service = ProductService(db)
    products = service.get_products(
        skip=skip,
        limit=limit,
        category_id=category_id,
        search=search,
        is_active=is_active
    )
    return products


@router.get("/{product_id}", response_model=ProductResponse)
async def get_product(
    product_id: int,
    db: Session = Depends(get_db)
):
    """
    获取单个商品详情
    """
    service = ProductService(db)
    product = service.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="商品不存在")
    return product


@router.post("/", response_model=ProductResponse)
async def create_product(
    product_data: ProductCreate,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    创建新商品
    
    需要管理员权限。
    """
    # 检查权限
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="权限不足")
    
    service = ProductService(db)
    
    # 检查分类是否存在
    if not service.category_exists(product_data.category_id):
        raise HTTPException(status_code=400, detail="分类不存在")
    
    # 检查SKU是否重复
    if product_data.sku and service.sku_exists(product_data.sku):
        raise HTTPException(status_code=400, detail="SKU已存在")
    
    product = service.create_product(product_data.dict())
    return product


@router.put("/{product_id}", response_model=ProductResponse)
async def update_product(
    product_id: int,
    product_data: ProductUpdate,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    更新商品信息
    
    需要管理员权限。
    """
    # 检查权限
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="权限不足")
    
    service = ProductService(db)
    
    # 检查商品是否存在
    if not service.product_exists(product_id):
        raise HTTPException(status_code=404, detail="商品不存在")
    
    # 检查分类是否存在
    if product_data.category_id and not service.category_exists(product_data.category_id):
        raise HTTPException(status_code=400, detail="分类不存在")
    
    # 检查SKU是否重复
    if product_data.sku and service.sku_exists(product_data.sku, exclude_id=product_id):
        raise HTTPException(status_code=400, detail="SKU已存在")
    
    product = service.update_product(product_id, product_data.dict(exclude_unset=True))
    return product


@router.delete("/{product_id}")
async def delete_product(
    product_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    删除商品
    
    需要管理员权限。实际执行软删除。
    """
    # 检查权限
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="权限不足")
    
    service = ProductService(db)
    
    # 检查商品是否存在
    if not service.product_exists(product_id):
        raise HTTPException(status_code=404, detail="商品不存在")
    
    service.delete_product(product_id)
    return {"message": "商品删除成功"}


@router.post("/{product_id}/stock")
async def update_stock(
    product_id: int,
    quantity: int,
    operation: str = Query(..., regex="^(add|subtract|set)$", description="操作类型：add增加，subtract减少，set设置"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    更新商品库存
    
    支持增加、减少、设置库存数量。
    """
    # 检查权限
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="权限不足")
    
    service = ProductService(db)
    
    # 检查商品是否存在
    if not service.product_exists(product_id):
        raise HTTPException(status_code=404, detail="商品不存在")
    
    try:
        new_stock = service.update_stock(product_id, quantity, operation)
        return {
            "message": "库存更新成功",
            "new_stock": new_stock
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/categories/", response_model=List[dict])
async def get_categories(
    db: Session = Depends(get_db)
):
    """
    获取商品分类列表
    """
    service = ProductService(db)
    categories = service.get_categories()
    return categories


@router.get("/search/", response_model=List[ProductResponse])
async def search_products(
    q: str = Query(..., min_length=1, description="搜索关键词"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    搜索商品
    
    支持商品名称、描述、SKU等字段的模糊搜索。
    """
    service = ProductService(db)
    products = service.search_products(q, skip=skip, limit=limit)
    return products