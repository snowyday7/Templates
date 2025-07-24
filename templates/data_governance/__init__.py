"""数据治理模块

提供完整的数据治理功能，包括：
- 数据质量管理
- 数据血缘追踪
- 数据合规性检查
- 数据分类和标记
- 数据生命周期管理
"""

from .data_quality import (
    DataQualityRule,
    DataQualityCheck,
    DataQualityManager,
    QualityRuleType,
    QualityCheckResult,
    initialize_data_quality,
    get_data_quality_manager
)

from .data_lineage import (
    DataLineageNode,
    DataLineageEdge,
    DataLineageGraph,
    DataLineageTracker,
    LineageNodeType,
    LineageRelationType,
    initialize_data_lineage,
    get_lineage_tracker
)

from .data_compliance import (
    ComplianceRule,
    ComplianceCheck,
    ComplianceManager,
    ComplianceFramework,
    ComplianceStatus,
    DataClassification,
    initialize_compliance,
    get_compliance_manager
)

from .data_catalog import (
    DataAsset,
    DataSchema,
    DataCatalog,
    AssetType,
    DataTag,
    initialize_data_catalog,
    get_data_catalog
)

__all__ = [
    # Data Quality
    "DataQualityRule",
    "DataQualityCheck",
    "DataQualityManager",
    "QualityRuleType",
    "QualityCheckResult",
    "initialize_data_quality",
    "get_data_quality_manager",
    
    # Data Lineage
    "DataLineageNode",
    "DataLineageEdge",
    "DataLineageGraph",
    "DataLineageTracker",
    "LineageNodeType",
    "LineageRelationType",
    "initialize_data_lineage",
    "get_lineage_tracker",
    
    # Data Compliance
    "ComplianceRule",
    "ComplianceCheck",
    "ComplianceManager",
    "ComplianceFramework",
    "ComplianceStatus",
    "DataClassification",
    "initialize_compliance",
    "get_compliance_manager",
    
    # Data Catalog
    "DataAsset",
    "DataSchema",
    "DataCatalog",
    "AssetType",
    "DataTag",
    "initialize_data_catalog",
    "get_data_catalog",
]