"""数据质量管理系统

提供完整的数据质量管理功能，包括：
- 数据质量规则定义
- 数据质量检查执行
- 质量报告生成
- 质量趋势分析
- 自动修复建议
"""

import time
import re
import statistics
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from collections import defaultdict, Counter

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sqlalchemy import text, inspect
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class QualityRuleType(str, Enum):
    """数据质量规则类型"""
    COMPLETENESS = "completeness"  # 完整性
    ACCURACY = "accuracy"  # 准确性
    CONSISTENCY = "consistency"  # 一致性
    VALIDITY = "validity"  # 有效性
    UNIQUENESS = "uniqueness"  # 唯一性
    TIMELINESS = "timeliness"  # 及时性
    CONFORMITY = "conformity"  # 符合性
    INTEGRITY = "integrity"  # 完整性约束


class QualityCheckStatus(str, Enum):
    """质量检查状态"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


class QualitySeverity(str, Enum):
    """质量问题严重程度"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class QualityMetric:
    """质量指标"""
    name: str
    value: float
    threshold: Optional[float] = None
    unit: str = ""
    description: str = ""
    
    def is_passing(self) -> bool:
        """检查是否通过阈值"""
        if self.threshold is None:
            return True
        return self.value >= self.threshold


@dataclass
class QualityIssue:
    """质量问题"""
    rule_name: str
    description: str
    severity: QualitySeverity
    affected_records: int = 0
    sample_values: List[Any] = field(default_factory=list)
    suggested_fix: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityCheckResult:
    """质量检查结果"""
    rule_name: str
    rule_type: QualityRuleType
    status: QualityCheckStatus
    score: float  # 0-100
    metrics: List[QualityMetric] = field(default_factory=list)
    issues: List[QualityIssue] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "rule_type": self.rule_type.value,
            "status": self.status.value,
            "score": self.score,
            "metrics": [{
                "name": m.name,
                "value": m.value,
                "threshold": m.threshold,
                "unit": m.unit,
                "description": m.description
            } for m in self.metrics],
            "issues": [{
                "rule_name": i.rule_name,
                "description": i.description,
                "severity": i.severity.value,
                "affected_records": i.affected_records,
                "sample_values": i.sample_values,
                "suggested_fix": i.suggested_fix,
                "metadata": i.metadata
            } for i in self.issues],
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class DataQualityRule(ABC):
    """数据质量规则基类"""
    
    def __init__(self, name: str, description: str, rule_type: QualityRuleType,
                 severity: QualitySeverity = QualitySeverity.MEDIUM,
                 enabled: bool = True, **kwargs):
        self.name = name
        self.description = description
        self.rule_type = rule_type
        self.severity = severity
        self.enabled = enabled
        self.config = kwargs
    
    @abstractmethod
    def check(self, data: Any, context: Optional[Dict[str, Any]] = None) -> QualityCheckResult:
        """执行质量检查"""
        pass
    
    def _create_result(self, status: QualityCheckStatus, score: float,
                      metrics: List[QualityMetric] = None,
                      issues: List[QualityIssue] = None,
                      execution_time: float = 0.0,
                      metadata: Dict[str, Any] = None) -> QualityCheckResult:
        """创建检查结果"""
        return QualityCheckResult(
            rule_name=self.name,
            rule_type=self.rule_type,
            status=status,
            score=score,
            metrics=metrics or [],
            issues=issues or [],
            execution_time=execution_time,
            metadata=metadata or {}
        )


class CompletenessRule(DataQualityRule):
    """完整性规则"""
    
    def __init__(self, name: str, columns: List[str], threshold: float = 0.95, **kwargs):
        super().__init__(name, f"检查列 {columns} 的完整性", QualityRuleType.COMPLETENESS, **kwargs)
        self.columns = columns
        self.threshold = threshold
    
    def check(self, data: Any, context: Optional[Dict[str, Any]] = None) -> QualityCheckResult:
        start_time = time.time()
        
        try:
            if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                return self._check_dataframe(data)
            elif isinstance(data, list):
                return self._check_list(data)
            else:
                return self._create_result(
                    QualityCheckStatus.ERROR, 0.0,
                    metadata={"error": "Unsupported data type"}
                )
        except Exception as e:
            return self._create_result(
                QualityCheckStatus.ERROR, 0.0,
                execution_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _check_dataframe(self, df: pd.DataFrame) -> QualityCheckResult:
        start_time = time.time()
        metrics = []
        issues = []
        total_score = 0.0
        
        for column in self.columns:
            if column not in df.columns:
                issues.append(QualityIssue(
                    rule_name=self.name,
                    description=f"列 '{column}' 不存在",
                    severity=QualitySeverity.HIGH
                ))
                continue
            
            # 计算完整性
            total_count = len(df)
            null_count = df[column].isnull().sum()
            completeness = (total_count - null_count) / total_count if total_count > 0 else 0.0
            
            metrics.append(QualityMetric(
                name=f"{column}_completeness",
                value=completeness,
                threshold=self.threshold,
                unit="ratio",
                description=f"列 {column} 的完整性比例"
            ))
            
            total_score += completeness
            
            # 检查是否低于阈值
            if completeness < self.threshold:
                issues.append(QualityIssue(
                    rule_name=self.name,
                    description=f"列 '{column}' 完整性 {completeness:.2%} 低于阈值 {self.threshold:.2%}",
                    severity=self.severity,
                    affected_records=null_count,
                    suggested_fix=f"检查数据源，补充缺失的 {column} 值"
                ))
        
        # 计算总分
        avg_score = (total_score / len(self.columns)) * 100 if self.columns else 0.0
        status = QualityCheckStatus.PASSED if not issues else QualityCheckStatus.FAILED
        
        return self._create_result(
            status, avg_score, metrics, issues,
            execution_time=time.time() - start_time
        )
    
    def _check_list(self, data: List[Dict]) -> QualityCheckResult:
        start_time = time.time()
        metrics = []
        issues = []
        total_score = 0.0
        
        if not data:
            return self._create_result(
                QualityCheckStatus.PASSED, 100.0,
                execution_time=time.time() - start_time
            )
        
        for column in self.columns:
            total_count = len(data)
            null_count = sum(1 for item in data if item.get(column) is None)
            completeness = (total_count - null_count) / total_count
            
            metrics.append(QualityMetric(
                name=f"{column}_completeness",
                value=completeness,
                threshold=self.threshold,
                unit="ratio",
                description=f"字段 {column} 的完整性比例"
            ))
            
            total_score += completeness
            
            if completeness < self.threshold:
                issues.append(QualityIssue(
                    rule_name=self.name,
                    description=f"字段 '{column}' 完整性 {completeness:.2%} 低于阈值 {self.threshold:.2%}",
                    severity=self.severity,
                    affected_records=null_count
                ))
        
        avg_score = (total_score / len(self.columns)) * 100 if self.columns else 0.0
        status = QualityCheckStatus.PASSED if not issues else QualityCheckStatus.FAILED
        
        return self._create_result(
            status, avg_score, metrics, issues,
            execution_time=time.time() - start_time
        )


class UniquenessRule(DataQualityRule):
    """唯一性规则"""
    
    def __init__(self, name: str, columns: List[str], threshold: float = 1.0, **kwargs):
        super().__init__(name, f"检查列 {columns} 的唯一性", QualityRuleType.UNIQUENESS, **kwargs)
        self.columns = columns
        self.threshold = threshold
    
    def check(self, data: Any, context: Optional[Dict[str, Any]] = None) -> QualityCheckResult:
        start_time = time.time()
        
        try:
            if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                return self._check_dataframe(data)
            elif isinstance(data, list):
                return self._check_list(data)
            else:
                return self._create_result(
                    QualityCheckStatus.ERROR, 0.0,
                    metadata={"error": "Unsupported data type"}
                )
        except Exception as e:
            return self._create_result(
                QualityCheckStatus.ERROR, 0.0,
                execution_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _check_dataframe(self, df: pd.DataFrame) -> QualityCheckResult:
        start_time = time.time()
        metrics = []
        issues = []
        
        # 检查组合唯一性
        if len(self.columns) > 1:
            subset_df = df[self.columns].dropna()
            total_count = len(subset_df)
            unique_count = len(subset_df.drop_duplicates())
            uniqueness = unique_count / total_count if total_count > 0 else 1.0
            
            metrics.append(QualityMetric(
                name="combined_uniqueness",
                value=uniqueness,
                threshold=self.threshold,
                unit="ratio",
                description=f"列组合 {self.columns} 的唯一性比例"
            ))
            
            if uniqueness < self.threshold:
                duplicate_count = total_count - unique_count
                duplicates = subset_df[subset_df.duplicated(keep=False)]
                sample_duplicates = duplicates.head(5).to_dict('records') if not duplicates.empty else []
                
                issues.append(QualityIssue(
                    rule_name=self.name,
                    description=f"列组合 {self.columns} 唯一性 {uniqueness:.2%} 低于阈值 {self.threshold:.2%}",
                    severity=self.severity,
                    affected_records=duplicate_count,
                    sample_values=sample_duplicates,
                    suggested_fix="检查并移除重复记录"
                ))
        else:
            # 检查单列唯一性
            column = self.columns[0]
            if column in df.columns:
                series = df[column].dropna()
                total_count = len(series)
                unique_count = series.nunique()
                uniqueness = unique_count / total_count if total_count > 0 else 1.0
                
                metrics.append(QualityMetric(
                    name=f"{column}_uniqueness",
                    value=uniqueness,
                    threshold=self.threshold,
                    unit="ratio",
                    description=f"列 {column} 的唯一性比例"
                ))
                
                if uniqueness < self.threshold:
                    duplicate_count = total_count - unique_count
                    duplicates = series[series.duplicated()].unique()[:5].tolist()
                    
                    issues.append(QualityIssue(
                        rule_name=self.name,
                        description=f"列 '{column}' 唯一性 {uniqueness:.2%} 低于阈值 {self.threshold:.2%}",
                        severity=self.severity,
                        affected_records=duplicate_count,
                        sample_values=duplicates
                    ))
        
        # 计算总分
        avg_score = (sum(m.value for m in metrics) / len(metrics)) * 100 if metrics else 0.0
        status = QualityCheckStatus.PASSED if not issues else QualityCheckStatus.FAILED
        
        return self._create_result(
            status, avg_score, metrics, issues,
            execution_time=time.time() - start_time
        )
    
    def _check_list(self, data: List[Dict]) -> QualityCheckResult:
        start_time = time.time()
        metrics = []
        issues = []
        
        if not data:
            return self._create_result(
                QualityCheckStatus.PASSED, 100.0,
                execution_time=time.time() - start_time
            )
        
        # 提取指定列的值
        values = []
        for item in data:
            row_values = tuple(item.get(col) for col in self.columns)
            if all(v is not None for v in row_values):
                values.append(row_values)
        
        total_count = len(values)
        unique_count = len(set(values))
        uniqueness = unique_count / total_count if total_count > 0 else 1.0
        
        metrics.append(QualityMetric(
            name="uniqueness",
            value=uniqueness,
            threshold=self.threshold,
            unit="ratio",
            description=f"字段 {self.columns} 的唯一性比例"
        ))
        
        if uniqueness < self.threshold:
            duplicate_count = total_count - unique_count
            counter = Counter(values)
            duplicates = [v for v, count in counter.items() if count > 1][:5]
            
            issues.append(QualityIssue(
                rule_name=self.name,
                description=f"字段 {self.columns} 唯一性 {uniqueness:.2%} 低于阈值 {self.threshold:.2%}",
                severity=self.severity,
                affected_records=duplicate_count,
                sample_values=duplicates
            ))
        
        score = uniqueness * 100
        status = QualityCheckStatus.PASSED if not issues else QualityCheckStatus.FAILED
        
        return self._create_result(
            status, score, metrics, issues,
            execution_time=time.time() - start_time
        )


class ValidityRule(DataQualityRule):
    """有效性规则"""
    
    def __init__(self, name: str, column: str, pattern: Optional[str] = None,
                 data_type: Optional[str] = None, value_range: Optional[Tuple] = None,
                 allowed_values: Optional[List] = None, threshold: float = 0.95, **kwargs):
        super().__init__(name, f"检查列 {column} 的有效性", QualityRuleType.VALIDITY, **kwargs)
        self.column = column
        self.pattern = pattern
        self.data_type = data_type
        self.value_range = value_range
        self.allowed_values = allowed_values
        self.threshold = threshold
    
    def check(self, data: Any, context: Optional[Dict[str, Any]] = None) -> QualityCheckResult:
        start_time = time.time()
        
        try:
            if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                return self._check_dataframe(data)
            elif isinstance(data, list):
                return self._check_list(data)
            else:
                return self._create_result(
                    QualityCheckStatus.ERROR, 0.0,
                    metadata={"error": "Unsupported data type"}
                )
        except Exception as e:
            return self._create_result(
                QualityCheckStatus.ERROR, 0.0,
                execution_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _check_dataframe(self, df: pd.DataFrame) -> QualityCheckResult:
        start_time = time.time()
        metrics = []
        issues = []
        
        if self.column not in df.columns:
            return self._create_result(
                QualityCheckStatus.ERROR, 0.0,
                execution_time=time.time() - start_time,
                metadata={"error": f"Column '{self.column}' not found"}
            )
        
        series = df[self.column].dropna()
        total_count = len(series)
        valid_count = 0
        invalid_samples = []
        
        if total_count == 0:
            return self._create_result(
                QualityCheckStatus.PASSED, 100.0,
                execution_time=time.time() - start_time
            )
        
        for value in series:
            is_valid = True
            
            # 检查正则表达式模式
            if self.pattern and isinstance(value, str):
                if not re.match(self.pattern, value):
                    is_valid = False
            
            # 检查数据类型
            if self.data_type:
                if self.data_type == 'int' and not isinstance(value, (int, np.integer)):
                    is_valid = False
                elif self.data_type == 'float' and not isinstance(value, (float, np.floating)):
                    is_valid = False
                elif self.data_type == 'str' and not isinstance(value, str):
                    is_valid = False
            
            # 检查值范围
            if self.value_range and isinstance(value, (int, float)):
                min_val, max_val = self.value_range
                if not (min_val <= value <= max_val):
                    is_valid = False
            
            # 检查允许的值
            if self.allowed_values and value not in self.allowed_values:
                is_valid = False
            
            if is_valid:
                valid_count += 1
            elif len(invalid_samples) < 5:
                invalid_samples.append(value)
        
        validity = valid_count / total_count
        
        metrics.append(QualityMetric(
            name=f"{self.column}_validity",
            value=validity,
            threshold=self.threshold,
            unit="ratio",
            description=f"列 {self.column} 的有效性比例"
        ))
        
        if validity < self.threshold:
            invalid_count = total_count - valid_count
            issues.append(QualityIssue(
                rule_name=self.name,
                description=f"列 '{self.column}' 有效性 {validity:.2%} 低于阈值 {self.threshold:.2%}",
                severity=self.severity,
                affected_records=invalid_count,
                sample_values=invalid_samples,
                suggested_fix="检查数据格式和值的有效性"
            ))
        
        score = validity * 100
        status = QualityCheckStatus.PASSED if not issues else QualityCheckStatus.FAILED
        
        return self._create_result(
            status, score, metrics, issues,
            execution_time=time.time() - start_time
        )
    
    def _check_list(self, data: List[Dict]) -> QualityCheckResult:
        start_time = time.time()
        metrics = []
        issues = []
        
        if not data:
            return self._create_result(
                QualityCheckStatus.PASSED, 100.0,
                execution_time=time.time() - start_time
            )
        
        values = [item.get(self.column) for item in data if item.get(self.column) is not None]
        total_count = len(values)
        valid_count = 0
        invalid_samples = []
        
        for value in values:
            is_valid = True
            
            # 检查正则表达式模式
            if self.pattern and isinstance(value, str):
                if not re.match(self.pattern, value):
                    is_valid = False
            
            # 检查数据类型
            if self.data_type:
                if self.data_type == 'int' and not isinstance(value, int):
                    is_valid = False
                elif self.data_type == 'float' and not isinstance(value, (int, float)):
                    is_valid = False
                elif self.data_type == 'str' and not isinstance(value, str):
                    is_valid = False
            
            # 检查值范围
            if self.value_range and isinstance(value, (int, float)):
                min_val, max_val = self.value_range
                if not (min_val <= value <= max_val):
                    is_valid = False
            
            # 检查允许的值
            if self.allowed_values and value not in self.allowed_values:
                is_valid = False
            
            if is_valid:
                valid_count += 1
            elif len(invalid_samples) < 5:
                invalid_samples.append(value)
        
        validity = valid_count / total_count if total_count > 0 else 1.0
        
        metrics.append(QualityMetric(
            name=f"{self.column}_validity",
            value=validity,
            threshold=self.threshold,
            unit="ratio",
            description=f"字段 {self.column} 的有效性比例"
        ))
        
        if validity < self.threshold:
            invalid_count = total_count - valid_count
            issues.append(QualityIssue(
                rule_name=self.name,
                description=f"字段 '{self.column}' 有效性 {validity:.2%} 低于阈值 {self.threshold:.2%}",
                severity=self.severity,
                affected_records=invalid_count,
                sample_values=invalid_samples
            ))
        
        score = validity * 100
        status = QualityCheckStatus.PASSED if not issues else QualityCheckStatus.FAILED
        
        return self._create_result(
            status, score, metrics, issues,
            execution_time=time.time() - start_time
        )


class DataQualityConfig(BaseSettings):
    """数据质量配置"""
    enabled: bool = True
    
    # 检查配置
    parallel_execution: bool = True
    max_workers: int = 4
    timeout: int = 300  # seconds
    
    # 报告配置
    generate_reports: bool = True
    report_format: str = "json"  # json, html, csv
    report_retention_days: int = 30
    
    # 阈值配置
    default_completeness_threshold: float = 0.95
    default_uniqueness_threshold: float = 1.0
    default_validity_threshold: float = 0.95
    
    # 存储配置
    store_results: bool = True
    max_results_per_rule: int = 100
    
    class Config:
        env_prefix = "DATA_QUALITY_"


class DataQualityManager:
    """数据质量管理器"""
    
    def __init__(self, config: DataQualityConfig):
        self.config = config
        self.rules: Dict[str, DataQualityRule] = {}
        self.results_history: Dict[str, List[QualityCheckResult]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def add_rule(self, rule: DataQualityRule):
        """添加质量规则"""
        with self._lock:
            self.rules[rule.name] = rule
    
    def remove_rule(self, rule_name: str):
        """移除质量规则"""
        with self._lock:
            self.rules.pop(rule_name, None)
            self.results_history.pop(rule_name, None)
    
    def get_rule(self, rule_name: str) -> Optional[DataQualityRule]:
        """获取质量规则"""
        return self.rules.get(rule_name)
    
    def get_all_rules(self) -> Dict[str, DataQualityRule]:
        """获取所有质量规则"""
        return self.rules.copy()
    
    def check_data(self, data: Any, rule_names: Optional[List[str]] = None,
                   context: Optional[Dict[str, Any]] = None) -> List[QualityCheckResult]:
        """检查数据质量"""
        if not self.config.enabled:
            return []
        
        # 确定要执行的规则
        rules_to_check = []
        if rule_names:
            rules_to_check = [self.rules[name] for name in rule_names if name in self.rules]
        else:
            rules_to_check = [rule for rule in self.rules.values() if rule.enabled]
        
        results = []
        
        if self.config.parallel_execution and len(rules_to_check) > 1:
            # 并行执行
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_rule = {
                    executor.submit(rule.check, data, context): rule
                    for rule in rules_to_check
                }
                
                for future in concurrent.futures.as_completed(future_to_rule, timeout=self.config.timeout):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        rule = future_to_rule[future]
                        error_result = QualityCheckResult(
                            rule_name=rule.name,
                            rule_type=rule.rule_type,
                            status=QualityCheckStatus.ERROR,
                            score=0.0,
                            metadata={"error": str(e)}
                        )
                        results.append(error_result)
        else:
            # 串行执行
            for rule in rules_to_check:
                try:
                    result = rule.check(data, context)
                    results.append(result)
                except Exception as e:
                    error_result = QualityCheckResult(
                        rule_name=rule.name,
                        rule_type=rule.rule_type,
                        status=QualityCheckStatus.ERROR,
                        score=0.0,
                        metadata={"error": str(e)}
                    )
                    results.append(error_result)
        
        # 存储结果
        if self.config.store_results:
            self._store_results(results)
        
        return results
    
    def _store_results(self, results: List[QualityCheckResult]):
        """存储检查结果"""
        with self._lock:
            for result in results:
                history = self.results_history[result.rule_name]
                history.append(result)
                
                # 限制历史记录数量
                if len(history) > self.config.max_results_per_rule:
                    history.pop(0)
    
    def get_results_history(self, rule_name: str) -> List[QualityCheckResult]:
        """获取规则的历史结果"""
        return self.results_history.get(rule_name, [])
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """获取质量摘要"""
        summary = {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for rule in self.rules.values() if rule.enabled),
            "rule_types": {},
            "recent_scores": {},
            "trend_analysis": {}
        }
        
        # 统计规则类型
        for rule in self.rules.values():
            rule_type = rule.rule_type.value
            summary["rule_types"][rule_type] = summary["rule_types"].get(rule_type, 0) + 1
        
        # 获取最近的分数
        for rule_name, history in self.results_history.items():
            if history:
                latest_result = history[-1]
                summary["recent_scores"][rule_name] = {
                    "score": latest_result.score,
                    "status": latest_result.status.value,
                    "timestamp": latest_result.timestamp
                }
                
                # 趋势分析
                if len(history) >= 2:
                    scores = [r.score for r in history[-10:]]  # 最近10次
                    if len(scores) >= 2:
                        trend = "improving" if scores[-1] > scores[0] else "declining" if scores[-1] < scores[0] else "stable"
                        summary["trend_analysis"][rule_name] = {
                            "trend": trend,
                            "avg_score": statistics.mean(scores),
                            "score_variance": statistics.variance(scores) if len(scores) > 1 else 0
                        }
        
        return summary
    
    def generate_report(self, results: List[QualityCheckResult], format: str = "json") -> str:
        """生成质量报告"""
        if format == "json":
            import json
            report_data = {
                "timestamp": time.time(),
                "summary": {
                    "total_checks": len(results),
                    "passed": sum(1 for r in results if r.status == QualityCheckStatus.PASSED),
                    "failed": sum(1 for r in results if r.status == QualityCheckStatus.FAILED),
                    "errors": sum(1 for r in results if r.status == QualityCheckStatus.ERROR),
                    "avg_score": statistics.mean([r.score for r in results]) if results else 0
                },
                "results": [r.to_dict() for r in results]
            }
            return json.dumps(report_data, indent=2, ensure_ascii=False)
        
        elif format == "html":
            # 简单的HTML报告
            html = "<html><head><title>Data Quality Report</title></head><body>"
            html += f"<h1>数据质量报告</h1>"
            html += f"<p>生成时间: {datetime.now()}</p>"
            html += f"<h2>摘要</h2>"
            html += f"<ul>"
            html += f"<li>总检查数: {len(results)}</li>"
            html += f"<li>通过: {sum(1 for r in results if r.status == QualityCheckStatus.PASSED)}</li>"
            html += f"<li>失败: {sum(1 for r in results if r.status == QualityCheckStatus.FAILED)}</li>"
            html += f"<li>错误: {sum(1 for r in results if r.status == QualityCheckStatus.ERROR)}</li>"
            html += f"</ul>"
            
            html += f"<h2>详细结果</h2>"
            html += f"<table border='1'>"
            html += f"<tr><th>规则名称</th><th>类型</th><th>状态</th><th>分数</th><th>问题数</th></tr>"
            
            for result in results:
                html += f"<tr>"
                html += f"<td>{result.rule_name}</td>"
                html += f"<td>{result.rule_type.value}</td>"
                html += f"<td>{result.status.value}</td>"
                html += f"<td>{result.score:.2f}</td>"
                html += f"<td>{len(result.issues)}</td>"
                html += f"</tr>"
            
            html += f"</table>"
            html += "</body></html>"
            return html
        
        else:
            raise ValueError(f"Unsupported report format: {format}")


# 全局数据质量管理器
_data_quality_manager: Optional[DataQualityManager] = None


def initialize_data_quality(config: DataQualityConfig) -> DataQualityManager:
    """初始化数据质量管理"""
    global _data_quality_manager
    _data_quality_manager = DataQualityManager(config)
    return _data_quality_manager


def get_data_quality_manager() -> Optional[DataQualityManager]:
    """获取全局数据质量管理器"""
    return _data_quality_manager


# 便捷函数
def add_quality_rule(rule: DataQualityRule):
    """添加质量规则的便捷函数"""
    manager = get_data_quality_manager()
    if manager:
        manager.add_rule(rule)


def check_data_quality(data: Any, rule_names: Optional[List[str]] = None) -> List[QualityCheckResult]:
    """检查数据质量的便捷函数"""
    manager = get_data_quality_manager()
    return manager.check_data(data, rule_names) if manager else []