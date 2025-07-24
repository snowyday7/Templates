# -*- coding: utf-8 -*-
"""
使用量服务

提供使用量统计和配额管理相关的业务逻辑。
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
from decimal import Decimal

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, extract

# 导入模板库组件
from src.utils.exceptions import ValidationException, AuthorizationException
from src.core.logging import get_logger

# 导入应用组件
from app.models.usage import (
    Usage, UsageCreate, UserQuota, UsageStats, 
    UsageReport, UsageType
)
from app.models.user import User

logger = get_logger(__name__)


class UsageService:
    """使用量服务类"""
    
    def record_usage(self, db: Session, user_id: int, usage_data: UsageCreate) -> Usage:
        """
        记录使用量
        """
        try:
            # 创建使用记录
            db_usage = Usage(
                user_id=user_id,
                usage_type=usage_data.usage_type,
                usage_date=usage_data.usage_date or date.today(),
                count=usage_data.count,
                prompt_tokens=usage_data.prompt_tokens or 0,
                completion_tokens=usage_data.completion_tokens or 0,
                total_tokens=usage_data.total_tokens or 0,
                cost=usage_data.cost or Decimal('0.00'),
                model=usage_data.model,
                conversation_id=usage_data.conversation_id,
                message_id=usage_data.message_id,
                metadata=usage_data.metadata or {},
                created_at=datetime.utcnow()
            )
            
            db.add(db_usage)
            
            # 更新用户配额
            self._update_user_quota(db, user_id, usage_data)
            
            db.commit()
            db.refresh(db_usage)
            
            logger.info(f"记录使用量: 用户 {user_id}, 类型 {usage_data.usage_type.value}, 数量 {usage_data.count}")
            
            return db_usage
            
        except Exception as e:
            db.rollback()
            logger.error(f"记录使用量失败: {e}")
            raise ValidationException("记录使用量失败")
    
    def _update_user_quota(self, db: Session, user_id: int, usage_data: UsageCreate):
        """
        更新用户配额使用量
        """
        quota = self.get_user_quota(db, user_id)
        if not quota:
            # 如果没有配额记录，创建默认配额
            quota = self.create_user_quota(db, user_id)
        
        today = date.today()
        
        # 更新每日使用量
        if quota.daily_reset_date != today:
            # 重置每日计数
            quota.daily_messages_used = 0
            quota.daily_tokens_used = 0
            quota.daily_reset_date = today
        
        # 更新每月使用量
        if quota.monthly_reset_date.month != today.month or quota.monthly_reset_date.year != today.year:
            # 重置每月计数
            quota.monthly_messages_used = 0
            quota.monthly_tokens_used = 0
            quota.monthly_reset_date = today.replace(day=1)
        
        # 增加使用量
        if usage_data.usage_type in [UsageType.chat_message, UsageType.api_call]:
            quota.daily_messages_used += usage_data.count
            quota.monthly_messages_used += usage_data.count
        
        quota.daily_tokens_used += usage_data.total_tokens or 0
        quota.monthly_tokens_used += usage_data.total_tokens or 0
        quota.updated_at = datetime.utcnow()
    
    def get_user_quota(self, db: Session, user_id: int) -> Optional[UserQuota]:
        """
        获取用户配额
        """
        return db.query(UserQuota).filter(UserQuota.user_id == user_id).first()
    
    def create_user_quota(self, db: Session, user_id: int, daily_message_limit: int = 100, daily_token_limit: int = 50000, monthly_message_limit: int = 3000, monthly_token_limit: int = 1500000) -> UserQuota:
        """
        创建用户配额
        """
        try:
            today = date.today()
            
            quota = UserQuota(
                user_id=user_id,
                daily_message_limit=daily_message_limit,
                daily_token_limit=daily_token_limit,
                monthly_message_limit=monthly_message_limit,
                monthly_token_limit=monthly_token_limit,
                daily_messages_used=0,
                daily_tokens_used=0,
                monthly_messages_used=0,
                monthly_tokens_used=0,
                daily_reset_date=today,
                monthly_reset_date=today.replace(day=1),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            db.add(quota)
            db.commit()
            db.refresh(quota)
            
            return quota
            
        except Exception as e:
            db.rollback()
            logger.error(f"创建用户配额失败: {e}")
            raise ValidationException("创建用户配额失败")
    
    def update_user_quota(self, db: Session, user_id: int, quota_data: dict) -> Optional[UserQuota]:
        """
        更新用户配额
        """
        try:
            quota = self.get_user_quota(db, user_id)
            if not quota:
                return None
            
            # 更新字段
            for field, value in quota_data.items():
                if hasattr(quota, field) and field.endswith('_limit'):
                    setattr(quota, field, value)
            
            quota.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(quota)
            
            return quota
            
        except Exception as e:
            db.rollback()
            logger.error(f"更新用户配额失败: {e}")
            return None
    
    def check_quota_limit(self, db: Session, user_id: int, usage_type: UsageType, count: int = 1, tokens: int = 0) -> bool:
        """
        检查配额限制
        """
        quota = self.get_user_quota(db, user_id)
        if not quota:
            # 如果没有配额记录，创建默认配额
            quota = self.create_user_quota(db, user_id)
        
        today = date.today()
        
        # 检查每日限制
        if usage_type in [UsageType.chat_message, UsageType.api_call]:
            if quota.daily_messages_used + count > quota.daily_message_limit:
                return False
        
        if quota.daily_tokens_used + tokens > quota.daily_token_limit:
            return False
        
        # 检查每月限制
        if usage_type in [UsageType.chat_message, UsageType.api_call]:
            if quota.monthly_messages_used + count > quota.monthly_message_limit:
                return False
        
        if quota.monthly_tokens_used + tokens > quota.monthly_token_limit:
            return False
        
        return True
    
    def get_usage_stats(self, db: Session, user_id: int, start_date: Optional[date] = None, end_date: Optional[date] = None) -> UsageStats:
        """
        获取使用量统计
        """
        if not start_date:
            start_date = date.today() - timedelta(days=30)
        if not end_date:
            end_date = date.today()
        
        base_query = db.query(Usage).filter(
            Usage.user_id == user_id,
            Usage.usage_date >= start_date,
            Usage.usage_date <= end_date
        )
        
        # 总使用量
        total_count = base_query.with_entities(func.sum(Usage.count)).scalar() or 0
        total_tokens = base_query.with_entities(func.sum(Usage.total_tokens)).scalar() or 0
        total_cost = base_query.with_entities(func.sum(Usage.cost)).scalar() or Decimal('0.00')
        
        # 按类型统计
        usage_by_type = {}
        for usage_type in UsageType:
            type_query = base_query.filter(Usage.usage_type == usage_type)
            count = type_query.with_entities(func.sum(Usage.count)).scalar() or 0
            tokens = type_query.with_entities(func.sum(Usage.total_tokens)).scalar() or 0
            cost = type_query.with_entities(func.sum(Usage.cost)).scalar() or Decimal('0.00')
            
            usage_by_type[usage_type.value] = {
                "count": count,
                "tokens": tokens,
                "cost": float(cost)
            }
        
        # 按模型统计
        usage_by_model = {}
        model_stats = base_query.with_entities(
            Usage.model,
            func.sum(Usage.count).label('count'),
            func.sum(Usage.total_tokens).label('tokens'),
            func.sum(Usage.cost).label('cost')
        ).group_by(Usage.model).all()
        
        for model, count, tokens, cost in model_stats:
            if model:
                usage_by_model[model] = {
                    "count": count or 0,
                    "tokens": tokens or 0,
                    "cost": float(cost or 0)
                }
        
        # 每日统计
        daily_stats = {}
        daily_query = base_query.with_entities(
            Usage.usage_date,
            func.sum(Usage.count).label('count'),
            func.sum(Usage.total_tokens).label('tokens'),
            func.sum(Usage.cost).label('cost')
        ).group_by(Usage.usage_date).order_by(Usage.usage_date).all()
        
        for usage_date, count, tokens, cost in daily_query:
            daily_stats[usage_date.isoformat()] = {
                "count": count or 0,
                "tokens": tokens or 0,
                "cost": float(cost or 0)
            }
        
        return UsageStats(
            total_count=total_count,
            total_tokens=total_tokens,
            total_cost=float(total_cost),
            usage_by_type=usage_by_type,
            usage_by_model=usage_by_model,
            daily_stats=daily_stats,
            period_start=start_date,
            period_end=end_date
        )
    
    def get_usage_report(self, db: Session, user_id: int, report_type: str = "monthly") -> UsageReport:
        """
        获取使用量报告
        """
        today = date.today()
        
        if report_type == "daily":
            start_date = today
            end_date = today
        elif report_type == "weekly":
            start_date = today - timedelta(days=today.weekday())
            end_date = start_date + timedelta(days=6)
        elif report_type == "monthly":
            start_date = today.replace(day=1)
            end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        else:
            # 年度报告
            start_date = today.replace(month=1, day=1)
            end_date = today.replace(month=12, day=31)
        
        # 获取统计数据
        stats = self.get_usage_stats(db, user_id, start_date, end_date)
        
        # 获取配额信息
        quota = self.get_user_quota(db, user_id)
        
        # 计算使用率
        quota_usage = {}
        if quota:
            if report_type == "daily":
                quota_usage = {
                    "message_usage_rate": (quota.daily_messages_used / quota.daily_message_limit * 100) if quota.daily_message_limit > 0 else 0,
                    "token_usage_rate": (quota.daily_tokens_used / quota.daily_token_limit * 100) if quota.daily_token_limit > 0 else 0,
                    "messages_remaining": max(0, quota.daily_message_limit - quota.daily_messages_used),
                    "tokens_remaining": max(0, quota.daily_token_limit - quota.daily_tokens_used)
                }
            else:
                quota_usage = {
                    "message_usage_rate": (quota.monthly_messages_used / quota.monthly_message_limit * 100) if quota.monthly_message_limit > 0 else 0,
                    "token_usage_rate": (quota.monthly_tokens_used / quota.monthly_token_limit * 100) if quota.monthly_token_limit > 0 else 0,
                    "messages_remaining": max(0, quota.monthly_message_limit - quota.monthly_messages_used),
                    "tokens_remaining": max(0, quota.monthly_token_limit - quota.monthly_tokens_used)
                }
        
        # 获取趋势数据（与上一周期比较）
        if report_type == "daily":
            prev_start = start_date - timedelta(days=1)
            prev_end = end_date - timedelta(days=1)
        elif report_type == "weekly":
            prev_start = start_date - timedelta(days=7)
            prev_end = end_date - timedelta(days=7)
        elif report_type == "monthly":
            prev_month = start_date - timedelta(days=1)
            prev_start = prev_month.replace(day=1)
            prev_end = start_date - timedelta(days=1)
        else:
            prev_start = start_date.replace(year=start_date.year - 1)
            prev_end = end_date.replace(year=end_date.year - 1)
        
        prev_stats = self.get_usage_stats(db, user_id, prev_start, prev_end)
        
        # 计算变化率
        trends = {
            "count_change": ((stats.total_count - prev_stats.total_count) / prev_stats.total_count * 100) if prev_stats.total_count > 0 else 0,
            "tokens_change": ((stats.total_tokens - prev_stats.total_tokens) / prev_stats.total_tokens * 100) if prev_stats.total_tokens > 0 else 0,
            "cost_change": ((stats.total_cost - prev_stats.total_cost) / prev_stats.total_cost * 100) if prev_stats.total_cost > 0 else 0
        }
        
        return UsageReport(
            report_type=report_type,
            period_start=start_date,
            period_end=end_date,
            stats=stats,
            quota_usage=quota_usage,
            trends=trends,
            generated_at=datetime.utcnow()
        )
    
    def get_user_usage_list(self, db: Session, user_id: int, page: int = 1, size: int = 50, filters: Optional[dict] = None) -> tuple[List[Usage], int]:
        """
        获取用户使用记录列表
        """
        query = db.query(Usage).filter(Usage.user_id == user_id)
        
        # 应用过滤器
        if filters:
            if filters.get("usage_type"):
                query = query.filter(Usage.usage_type == filters["usage_type"])
            
            if filters.get("model"):
                query = query.filter(Usage.model == filters["model"])
            
            if filters.get("date_from"):
                query = query.filter(Usage.usage_date >= filters["date_from"])
            
            if filters.get("date_to"):
                query = query.filter(Usage.usage_date <= filters["date_to"])
            
            if filters.get("conversation_id"):
                query = query.filter(Usage.conversation_id == filters["conversation_id"])
        
        # 获取总数
        total = query.count()
        
        # 分页
        usage_records = query.order_by(
            desc(Usage.created_at)
        ).offset((page - 1) * size).limit(size).all()
        
        return usage_records, total
    
    def get_top_models(self, db: Session, user_id: int, limit: int = 10, start_date: Optional[date] = None, end_date: Optional[date] = None) -> List[dict]:
        """
        获取最常用的模型
        """
        if not start_date:
            start_date = date.today() - timedelta(days=30)
        if not end_date:
            end_date = date.today()
        
        query = db.query(
            Usage.model,
            func.sum(Usage.count).label('total_count'),
            func.sum(Usage.total_tokens).label('total_tokens'),
            func.sum(Usage.cost).label('total_cost')
        ).filter(
            Usage.user_id == user_id,
            Usage.usage_date >= start_date,
            Usage.usage_date <= end_date,
            Usage.model.isnot(None)
        ).group_by(Usage.model).order_by(
            desc('total_count')
        ).limit(limit)
        
        results = []
        for model, count, tokens, cost in query.all():
            results.append({
                "model": model,
                "count": count or 0,
                "tokens": tokens or 0,
                "cost": float(cost or 0)
            })
        
        return results
    
    def get_usage_summary(self, db: Session, user_id: int) -> dict:
        """
        获取使用量摘要
        """
        today = date.today()
        
        # 今日使用量
        today_stats = self.get_usage_stats(db, user_id, today, today)
        
        # 本月使用量
        month_start = today.replace(day=1)
        month_stats = self.get_usage_stats(db, user_id, month_start, today)
        
        # 总使用量
        total_stats = self.get_usage_stats(db, user_id, date(2020, 1, 1), today)
        
        # 配额信息
        quota = self.get_user_quota(db, user_id)
        
        return {
            "today": {
                "count": today_stats.total_count,
                "tokens": today_stats.total_tokens,
                "cost": today_stats.total_cost
            },
            "this_month": {
                "count": month_stats.total_count,
                "tokens": month_stats.total_tokens,
                "cost": month_stats.total_cost
            },
            "total": {
                "count": total_stats.total_count,
                "tokens": total_stats.total_tokens,
                "cost": total_stats.total_cost
            },
            "quota": {
                "daily_messages": {
                    "used": quota.daily_messages_used if quota else 0,
                    "limit": quota.daily_message_limit if quota else 0,
                    "remaining": (quota.daily_message_limit - quota.daily_messages_used) if quota else 0
                },
                "daily_tokens": {
                    "used": quota.daily_tokens_used if quota else 0,
                    "limit": quota.daily_token_limit if quota else 0,
                    "remaining": (quota.daily_token_limit - quota.daily_tokens_used) if quota else 0
                },
                "monthly_messages": {
                    "used": quota.monthly_messages_used if quota else 0,
                    "limit": quota.monthly_message_limit if quota else 0,
                    "remaining": (quota.monthly_message_limit - quota.monthly_messages_used) if quota else 0
                },
                "monthly_tokens": {
                    "used": quota.monthly_tokens_used if quota else 0,
                    "limit": quota.monthly_token_limit if quota else 0,
                    "remaining": (quota.monthly_token_limit - quota.monthly_tokens_used) if quota else 0
                }
            } if quota else None
        }
    
    def reset_user_quota(self, db: Session, user_id: int, reset_type: str = "both") -> bool:
        """
        重置用户配额
        """
        try:
            quota = self.get_user_quota(db, user_id)
            if not quota:
                return False
            
            today = date.today()
            
            if reset_type in ["daily", "both"]:
                quota.daily_messages_used = 0
                quota.daily_tokens_used = 0
                quota.daily_reset_date = today
            
            if reset_type in ["monthly", "both"]:
                quota.monthly_messages_used = 0
                quota.monthly_tokens_used = 0
                quota.monthly_reset_date = today.replace(day=1)
            
            quota.updated_at = datetime.utcnow()
            
            db.commit()
            
            logger.info(f"重置用户配额: 用户 {user_id}, 类型 {reset_type}")
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"重置用户配额失败: {e}")
            return False
    
    def export_usage_data(self, db: Session, user_id: int, start_date: Optional[date] = None, end_date: Optional[date] = None, format: str = "json") -> dict:
        """
        导出使用量数据
        """
        if not start_date:
            start_date = date.today() - timedelta(days=30)
        if not end_date:
            end_date = date.today()
        
        # 获取使用记录
        usage_records, _ = self.get_user_usage_list(
            db, user_id, page=1, size=10000,
            filters={"date_from": start_date, "date_to": end_date}
        )
        
        # 获取统计数据
        stats = self.get_usage_stats(db, user_id, start_date, end_date)
        
        # 获取配额信息
        quota = self.get_user_quota(db, user_id)
        
        export_data = {
            "export_info": {
                "user_id": user_id,
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat(),
                "exported_at": datetime.utcnow().isoformat(),
                "format": format,
                "total_records": len(usage_records)
            },
            "statistics": {
                "total_count": stats.total_count,
                "total_tokens": stats.total_tokens,
                "total_cost": stats.total_cost,
                "usage_by_type": stats.usage_by_type,
                "usage_by_model": stats.usage_by_model
            },
            "quota": {
                "daily_message_limit": quota.daily_message_limit if quota else None,
                "daily_token_limit": quota.daily_token_limit if quota else None,
                "monthly_message_limit": quota.monthly_message_limit if quota else None,
                "monthly_token_limit": quota.monthly_token_limit if quota else None,
                "daily_messages_used": quota.daily_messages_used if quota else None,
                "daily_tokens_used": quota.daily_tokens_used if quota else None,
                "monthly_messages_used": quota.monthly_messages_used if quota else None,
                "monthly_tokens_used": quota.monthly_tokens_used if quota else None
            } if quota else None,
            "records": [
                {
                    "id": record.id,
                    "usage_type": record.usage_type.value,
                    "usage_date": record.usage_date.isoformat(),
                    "count": record.count,
                    "prompt_tokens": record.prompt_tokens,
                    "completion_tokens": record.completion_tokens,
                    "total_tokens": record.total_tokens,
                    "cost": float(record.cost),
                    "model": record.model,
                    "conversation_id": record.conversation_id,
                    "message_id": record.message_id,
                    "created_at": record.created_at.isoformat()
                }
                for record in usage_records
            ]
        }
        
        return export_data