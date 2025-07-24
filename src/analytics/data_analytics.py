# -*- coding: utf-8 -*-
"""
数据分析模块
提供数据挖掘、机器学习、统计分析、预测建模等高级分析功能
"""

import asyncio
import numpy as np
import pandas as pd
from typing import (
    Dict, List, Any, Optional, Union, Tuple, Callable,
    Iterator, AsyncIterator
)
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import pickle
import joblib
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import warnings
from collections import defaultdict, Counter
import math
import statistics
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso
)
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class AnalysisType(Enum):
    """分析类型枚举"""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"


class ModelType(Enum):
    """模型类型枚举"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    TIME_SERIES = "time_series"
    ANOMALY_DETECTION = "anomaly_detection"


class ScalingMethod(Enum):
    """数据缩放方法枚举"""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"


class VisualizationType(Enum):
    """可视化类型枚举"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    PIE = "pie"
    VIOLIN = "violin"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"


@dataclass
class AnalysisConfig:
    """分析配置"""
    analysis_type: AnalysisType
    model_type: Optional[ModelType] = None
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    test_size: float = 0.2
    random_state: int = 42
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    cross_validation: bool = True
    cv_folds: int = 5
    hyperparameter_tuning: bool = False
    save_model: bool = True
    model_path: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """分析结果"""
    success: bool
    analysis_type: AnalysisType
    data: Any = None
    model: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    visualizations: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据"""
        # 移除重复行
        df = df.drop_duplicates()
        
        # 处理缺失值
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                # 数值列用中位数填充
                df[column].fillna(df[column].median(), inplace=True)
            else:
                # 分类列用众数填充
                mode_value = df[column].mode()
                if not mode_value.empty:
                    df[column].fillna(mode_value[0], inplace=True)
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """编码分类变量"""
        df_encoded = df.copy()
        
        for column in columns:
            if column in df.columns:
                if df[column].nunique() <= 10:  # 使用独热编码
                    encoder = OneHotEncoder(sparse=False, drop='first')
                    encoded_data = encoder.fit_transform(df[[column]])
                    encoded_columns = [f"{column}_{cat}" for cat in encoder.categories_[0][1:]]
                    
                    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=df.index)
                    df_encoded = pd.concat([df_encoded.drop(column, axis=1), encoded_df], axis=1)
                    
                    self.encoders[column] = encoder
                else:  # 使用标签编码
                    encoder = LabelEncoder()
                    df_encoded[column] = encoder.fit_transform(df[column])
                    self.encoders[column] = encoder
        
        return df_encoded
    
    def scale_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: ScalingMethod = ScalingMethod.STANDARD
    ) -> pd.DataFrame:
        """特征缩放"""
        df_scaled = df.copy()
        
        if method == ScalingMethod.STANDARD:
            scaler = StandardScaler()
        elif method == ScalingMethod.MINMAX:
            scaler = MinMaxScaler()
        else:
            return df_scaled
        
        df_scaled[columns] = scaler.fit_transform(df[columns])
        self.scalers[method.value] = scaler
        
        return df_scaled
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 10,
        score_func: Callable = f_classif
    ) -> pd.DataFrame:
        """特征选择"""
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()]
        self.feature_selectors['selectkbest'] = selector
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def reduce_dimensions(
        self,
        X: pd.DataFrame,
        method: str = "pca",
        n_components: int = 2
    ) -> Tuple[pd.DataFrame, Any]:
        """降维"""
        if method.lower() == "pca":
            reducer = PCA(n_components=n_components)
        elif method.lower() == "svd":
            reducer = TruncatedSVD(n_components=n_components)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
        X_reduced = reducer.fit_transform(X)
        
        columns = [f"{method.upper()}_{i+1}" for i in range(n_components)]
        df_reduced = pd.DataFrame(X_reduced, columns=columns, index=X.index)
        
        return df_reduced, reducer


class StatisticalAnalyzer:
    """统计分析器"""
    
    @staticmethod
    def descriptive_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """描述性统计"""
        stats_dict = {
            "basic_stats": df.describe().to_dict(),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "unique_values": df.nunique().to_dict(),
            "correlation_matrix": df.corr().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 1 else {},
            "skewness": df.skew().to_dict(),
            "kurtosis": df.kurtosis().to_dict()
        }
        
        return stats_dict
    
    @staticmethod
    def hypothesis_testing(
        data1: Union[pd.Series, List],
        data2: Union[pd.Series, List] = None,
        test_type: str = "ttest",
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """假设检验"""
        results = {}
        
        if test_type == "ttest":
            if data2 is not None:
                # 双样本t检验
                statistic, p_value = stats.ttest_ind(data1, data2)
                test_name = "Independent t-test"
            else:
                # 单样本t检验
                statistic, p_value = stats.ttest_1samp(data1, 0)
                test_name = "One-sample t-test"
        
        elif test_type == "chi2":
            # 卡方检验
            statistic, p_value, dof, expected = stats.chi2_contingency(data1)
            test_name = "Chi-square test"
            results["degrees_of_freedom"] = dof
            results["expected_frequencies"] = expected.tolist()
        
        elif test_type == "anova":
            # 方差分析
            statistic, p_value = stats.f_oneway(*data1)
            test_name = "One-way ANOVA"
        
        elif test_type == "mannwhitney":
            # Mann-Whitney U检验
            statistic, p_value = stats.mannwhitneyu(data1, data2)
            test_name = "Mann-Whitney U test"
        
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        results.update({
            "test_name": test_name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "alpha": alpha,
            "significant": p_value < alpha,
            "conclusion": "Reject null hypothesis" if p_value < alpha else "Fail to reject null hypothesis"
        })
        
        return results
    
    @staticmethod
    def correlation_analysis(
        df: pd.DataFrame,
        method: str = "pearson"
    ) -> Dict[str, Any]:
        """相关性分析"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if method == "pearson":
            corr_matrix = numeric_df.corr(method='pearson')
        elif method == "spearman":
            corr_matrix = numeric_df.corr(method='spearman')
        elif method == "kendall":
            corr_matrix = numeric_df.corr(method='kendall')
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
        
        # 找出强相关性
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        "variable1": corr_matrix.columns[i],
                        "variable2": corr_matrix.columns[j],
                        "correlation": float(corr_value)
                    })
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "method": method,
            "strong_correlations": strong_correlations
        }
    
    @staticmethod
    def outlier_detection(
        data: Union[pd.Series, pd.DataFrame],
        method: str = "iqr",
        threshold: float = 1.5
    ) -> Dict[str, Any]:
        """异常值检测"""
        if isinstance(data, pd.DataFrame):
            numeric_data = data.select_dtypes(include=[np.number])
        else:
            numeric_data = data
        
        outliers = {}
        
        if method == "iqr":
            for column in numeric_data.columns if isinstance(numeric_data, pd.DataFrame) else [numeric_data.name or 'data']:
                series = numeric_data[column] if isinstance(numeric_data, pd.DataFrame) else numeric_data
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_indices = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
                outliers[column] = {
                    "indices": outlier_indices,
                    "count": len(outlier_indices),
                    "percentage": len(outlier_indices) / len(series) * 100,
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound)
                }
        
        elif method == "zscore":
            for column in numeric_data.columns if isinstance(numeric_data, pd.DataFrame) else [numeric_data.name or 'data']:
                series = numeric_data[column] if isinstance(numeric_data, pd.DataFrame) else numeric_data
                z_scores = np.abs(stats.zscore(series))
                outlier_indices = series[z_scores > threshold].index.tolist()
                
                outliers[column] = {
                    "indices": outlier_indices,
                    "count": len(outlier_indices),
                    "percentage": len(outlier_indices) / len(series) * 100,
                    "threshold": threshold
                }
        
        return outliers


class MachineLearningAnalyzer:
    """机器学习分析器"""
    
    def __init__(self):
        self.models = {}
        self.preprocessor = DataPreprocessor()
    
    def train_classification_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = "random_forest",
        test_size: float = 0.2,
        random_state: int = 42,
        cross_validation: bool = True,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """训练分类模型"""
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 选择模型
        if model_type == "random_forest":
            model = RandomForestClassifier(random_state=random_state)
        elif model_type == "logistic_regression":
            model = LogisticRegression(random_state=random_state, max_iter=1000)
        elif model_type == "svm":
            model = SVC(random_state=random_state, probability=True)
        elif model_type == "naive_bayes":
            model = GaussianNB()
        elif model_type == "decision_tree":
            model = DecisionTreeClassifier(random_state=random_state)
        elif model_type == "knn":
            model = KNeighborsClassifier()
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(random_state=random_state)
        else:
            raise ValueError(f"Unsupported classification model: {model_type}")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # 评估指标
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted')),
            "recall": float(recall_score(y_test, y_pred, average='weighted')),
            "f1_score": float(f1_score(y_test, y_pred, average='weighted')),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        
        # 交叉验证
        if cross_validation:
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            metrics["cv_scores"] = cv_scores.tolist()
            metrics["cv_mean"] = float(cv_scores.mean())
            metrics["cv_std"] = float(cv_scores.std())
        
        # 特征重要性
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            metrics["feature_importance"] = feature_importance
        
        self.models[f"classification_{model_type}"] = model
        
        return {
            "model": model,
            "metrics": metrics,
            "predictions": {
                "y_test": y_test.tolist(),
                "y_pred": y_pred.tolist(),
                "y_pred_proba": y_pred_proba.tolist() if y_pred_proba is not None else None
            }
        }
    
    def train_regression_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = "random_forest",
        test_size: float = 0.2,
        random_state: int = 42,
        cross_validation: bool = True,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """训练回归模型"""
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 选择模型
        if model_type == "random_forest":
            model = RandomForestRegressor(random_state=random_state)
        elif model_type == "linear_regression":
            model = LinearRegression()
        elif model_type == "ridge":
            model = Ridge(random_state=random_state)
        elif model_type == "lasso":
            model = Lasso(random_state=random_state)
        elif model_type == "svr":
            model = SVR()
        elif model_type == "decision_tree":
            model = DecisionTreeRegressor(random_state=random_state)
        elif model_type == "knn":
            model = KNeighborsRegressor()
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(random_state=random_state)
        else:
            raise ValueError(f"Unsupported regression model: {model_type}")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估指标
        metrics = {
            "mse": float(mean_squared_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2_score": float(r2_score(y_test, y_pred))
        }
        
        # 交叉验证
        if cross_validation:
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
            metrics["cv_scores"] = cv_scores.tolist()
            metrics["cv_mean"] = float(cv_scores.mean())
            metrics["cv_std"] = float(cv_scores.std())
        
        # 特征重要性
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            metrics["feature_importance"] = feature_importance
        
        self.models[f"regression_{model_type}"] = model
        
        return {
            "model": model,
            "metrics": metrics,
            "predictions": {
                "y_test": y_test.tolist(),
                "y_pred": y_pred.tolist()
            }
        }
    
    def perform_clustering(
        self,
        X: pd.DataFrame,
        algorithm: str = "kmeans",
        n_clusters: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """执行聚类分析"""
        if algorithm == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        elif algorithm == "dbscan":
            model = DBSCAN(**kwargs)
        elif algorithm == "agglomerative":
            model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
        
        # 执行聚类
        cluster_labels = model.fit_predict(X)
        
        # 计算聚类指标
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        metrics = {
            "n_clusters": len(np.unique(cluster_labels)),
            "cluster_counts": dict(zip(*np.unique(cluster_labels, return_counts=True)))
        }
        
        if len(np.unique(cluster_labels)) > 1:
            metrics["silhouette_score"] = float(silhouette_score(X, cluster_labels))
            metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X, cluster_labels))
        
        # 聚类中心（如果适用）
        if hasattr(model, 'cluster_centers_'):
            metrics["cluster_centers"] = model.cluster_centers_.tolist()
        
        self.models[f"clustering_{algorithm}"] = model
        
        return {
            "model": model,
            "cluster_labels": cluster_labels.tolist(),
            "metrics": metrics
        }


class DataVisualizer:
    """数据可视化器"""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_distribution_plot(
        self,
        data: pd.Series,
        title: str = "Distribution Plot",
        save_path: str = None
    ) -> str:
        """创建分布图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 直方图
        ax1.hist(data, bins=30, alpha=0.7, edgecolor='black')
        ax1.set_title(f"{title} - Histogram")
        ax1.set_xlabel(data.name or "Value")
        ax1.set_ylabel("Frequency")
        
        # 箱线图
        ax2.boxplot(data.dropna())
        ax2.set_title(f"{title} - Box Plot")
        ax2.set_ylabel(data.name or "Value")
        
        plt.tight_layout()
        
        if not save_path:
            save_path = self.output_dir / f"distribution_{data.name or 'data'}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_correlation_heatmap(
        self,
        df: pd.DataFrame,
        title: str = "Correlation Heatmap",
        save_path: str = None
    ) -> str:
        """创建相关性热力图"""
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f'
        )
        plt.title(title)
        plt.tight_layout()
        
        if not save_path:
            save_path = self.output_dir / "correlation_heatmap.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_scatter_plot(
        self,
        x: pd.Series,
        y: pd.Series,
        hue: pd.Series = None,
        title: str = "Scatter Plot",
        save_path: str = None
    ) -> str:
        """创建散点图"""
        plt.figure(figsize=(10, 6))
        
        if hue is not None:
            sns.scatterplot(x=x, y=y, hue=hue, alpha=0.7)
        else:
            plt.scatter(x, y, alpha=0.7)
        
        plt.title(title)
        plt.xlabel(x.name or "X")
        plt.ylabel(y.name or "Y")
        
        if not save_path:
            save_path = self.output_dir / f"scatter_{x.name}_{y.name}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_feature_importance_plot(
        self,
        feature_importance: Dict[str, float],
        title: str = "Feature Importance",
        save_path: str = None,
        top_n: int = 20
    ) -> str:
        """创建特征重要性图"""
        # 排序并取前N个
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importance = zip(*sorted_features)
        
        plt.figure(figsize=(10, max(6, len(features) * 0.3)))
        plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel("Importance")
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if not save_path:
            save_path = self.output_dir / "feature_importance.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_confusion_matrix_plot(
        self,
        confusion_matrix: List[List[int]],
        class_names: List[str] = None,
        title: str = "Confusion Matrix",
        save_path: str = None
    ) -> str:
        """创建混淆矩阵图"""
        cm = np.array(confusion_matrix)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        
        if not save_path:
            save_path = self.output_dir / "confusion_matrix.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_interactive_plot(
        self,
        df: pd.DataFrame,
        plot_type: VisualizationType,
        x_column: str = None,
        y_column: str = None,
        color_column: str = None,
        title: str = "Interactive Plot",
        save_path: str = None
    ) -> str:
        """创建交互式图表"""
        if plot_type == VisualizationType.SCATTER:
            fig = px.scatter(
                df, x=x_column, y=y_column, color=color_column,
                title=title, hover_data=df.columns.tolist()
            )
        elif plot_type == VisualizationType.LINE:
            fig = px.line(
                df, x=x_column, y=y_column, color=color_column,
                title=title
            )
        elif plot_type == VisualizationType.BAR:
            fig = px.bar(
                df, x=x_column, y=y_column, color=color_column,
                title=title
            )
        elif plot_type == VisualizationType.HISTOGRAM:
            fig = px.histogram(
                df, x=x_column, color=color_column,
                title=title
            )
        elif plot_type == VisualizationType.BOX:
            fig = px.box(
                df, x=x_column, y=y_column, color=color_column,
                title=title
            )
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        if not save_path:
            save_path = self.output_dir / f"interactive_{plot_type.value}.html"
        
        fig.write_html(save_path)
        
        return str(save_path)


class DataAnalyticsEngine:
    """数据分析引擎"""
    
    def __init__(self, output_dir: str = "analytics_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.preprocessor = DataPreprocessor()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.ml_analyzer = MachineLearningAnalyzer()
        self.visualizer = DataVisualizer(str(self.output_dir / "visualizations"))
        
        self.analysis_history = []
    
    async def analyze_data(
        self,
        data: Union[pd.DataFrame, str, Path],
        config: AnalysisConfig
    ) -> AnalysisResult:
        """执行数据分析"""
        start_time = datetime.now()
        
        try:
            # 加载数据
            if isinstance(data, (str, Path)):
                df = pd.read_csv(data)
            else:
                df = data.copy()
            
            # 数据预处理
            df = self.preprocessor.clean_data(df)
            
            # 根据分析类型执行分析
            if config.analysis_type == AnalysisType.DESCRIPTIVE:
                result = await self._descriptive_analysis(df, config)
            elif config.analysis_type == AnalysisType.PREDICTIVE:
                result = await self._predictive_analysis(df, config)
            elif config.analysis_type == AnalysisType.DIAGNOSTIC:
                result = await self._diagnostic_analysis(df, config)
            else:
                raise ValueError(f"Unsupported analysis type: {config.analysis_type}")
            
            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            
            # 保存分析历史
            self.analysis_history.append({
                "timestamp": start_time,
                "config": config,
                "result": result
            })
            
            return result
        
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return AnalysisResult(
                success=False,
                analysis_type=config.analysis_type,
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _descriptive_analysis(
        self,
        df: pd.DataFrame,
        config: AnalysisConfig
    ) -> AnalysisResult:
        """描述性分析"""
        insights = []
        recommendations = []
        visualizations = []
        
        # 基础统计
        stats = self.statistical_analyzer.descriptive_statistics(df)
        
        # 相关性分析
        correlation_analysis = self.statistical_analyzer.correlation_analysis(df)
        
        # 异常值检测
        outliers = self.statistical_analyzer.outlier_detection(df)
        
        # 生成可视化
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 1:
            # 相关性热力图
            heatmap_path = self.visualizer.create_correlation_heatmap(df)
            visualizations.append(heatmap_path)
        
        # 分布图
        for column in numeric_columns[:5]:  # 限制前5个数值列
            dist_path = self.visualizer.create_distribution_plot(
                df[column], title=f"{column} Distribution"
            )
            visualizations.append(dist_path)
        
        # 生成洞察
        insights.append(f"数据集包含 {len(df)} 行和 {len(df.columns)} 列")
        insights.append(f"缺失值总数: {df.isnull().sum().sum()}")
        
        if correlation_analysis["strong_correlations"]:
            insights.append(f"发现 {len(correlation_analysis['strong_correlations'])} 对强相关变量")
        
        # 生成建议
        if df.isnull().sum().sum() > 0:
            recommendations.append("建议处理缺失值以提高数据质量")
        
        if any(info["count"] > 0 for info in outliers.values()):
            recommendations.append("检测到异常值，建议进一步调查")
        
        return AnalysisResult(
            success=True,
            analysis_type=config.analysis_type,
            data={
                "descriptive_stats": stats,
                "correlation_analysis": correlation_analysis,
                "outliers": outliers
            },
            visualizations=visualizations,
            insights=insights,
            recommendations=recommendations
        )
    
    async def _predictive_analysis(
        self,
        df: pd.DataFrame,
        config: AnalysisConfig
    ) -> AnalysisResult:
        """预测性分析"""
        if not config.target_column or config.target_column not in df.columns:
            raise ValueError("Target column is required for predictive analysis")
        
        insights = []
        recommendations = []
        visualizations = []
        
        # 准备特征和目标变量
        if config.feature_columns:
            feature_columns = [col for col in config.feature_columns if col in df.columns]
        else:
            feature_columns = [col for col in df.columns if col != config.target_column]
        
        X = df[feature_columns]
        y = df[config.target_column]
        
        # 编码分类变量
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            X = self.preprocessor.encode_categorical(X, categorical_columns)
        
        # 特征缩放
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns and config.scaling_method != ScalingMethod.NONE:
            X = self.preprocessor.scale_features(X, numeric_columns, config.scaling_method)
        
        # 确定模型类型
        if config.model_type == ModelType.CLASSIFICATION or y.dtype == 'object' or y.nunique() < 10:
            # 分类任务
            model_result = self.ml_analyzer.train_classification_model(
                X, y,
                model_type="random_forest",
                test_size=config.test_size,
                random_state=config.random_state,
                cross_validation=config.cross_validation,
                cv_folds=config.cv_folds
            )
            
            # 混淆矩阵可视化
            cm_path = self.visualizer.create_confusion_matrix_plot(
                model_result["metrics"]["confusion_matrix"],
                class_names=[str(cls) for cls in sorted(y.unique())]
            )
            visualizations.append(cm_path)
            
            insights.append(f"分类模型准确率: {model_result['metrics']['accuracy']:.3f}")
            
        else:
            # 回归任务
            model_result = self.ml_analyzer.train_regression_model(
                X, y,
                model_type="random_forest",
                test_size=config.test_size,
                random_state=config.random_state,
                cross_validation=config.cross_validation,
                cv_folds=config.cv_folds
            )
            
            insights.append(f"回归模型R²得分: {model_result['metrics']['r2_score']:.3f}")
        
        # 特征重要性可视化
        if "feature_importance" in model_result["metrics"]:
            fi_path = self.visualizer.create_feature_importance_plot(
                model_result["metrics"]["feature_importance"]
            )
            visualizations.append(fi_path)
        
        # 保存模型
        if config.save_model:
            model_path = config.model_path or str(self.output_dir / "model.joblib")
            joblib.dump(model_result["model"], model_path)
            model_result["model_path"] = model_path
        
        return AnalysisResult(
            success=True,
            analysis_type=config.analysis_type,
            data=model_result,
            model=model_result["model"],
            metrics=model_result["metrics"],
            visualizations=visualizations,
            insights=insights,
            recommendations=recommendations
        )
    
    async def _diagnostic_analysis(
        self,
        df: pd.DataFrame,
        config: AnalysisConfig
    ) -> AnalysisResult:
        """诊断性分析"""
        insights = []
        recommendations = []
        visualizations = []
        
        # 数据质量检查
        quality_issues = []
        
        # 检查缺失值
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            quality_issues.append({
                "issue": "missing_values",
                "description": "数据集中存在缺失值",
                "affected_columns": missing_data[missing_data > 0].to_dict()
            })
        
        # 检查重复行
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            quality_issues.append({
                "issue": "duplicate_rows",
                "description": f"发现 {duplicate_count} 行重复数据",
                "count": int(duplicate_count)
            })
        
        # 检查数据类型一致性
        type_issues = []
        for column in df.columns:
            if df[column].dtype == 'object':
                # 检查是否应该是数值类型
                try:
                    pd.to_numeric(df[column], errors='raise')
                    type_issues.append(column)
                except:
                    pass
        
        if type_issues:
            quality_issues.append({
                "issue": "data_type_inconsistency",
                "description": "某些列可能需要类型转换",
                "columns": type_issues
            })
        
        # 异常值分析
        outlier_analysis = self.statistical_analyzer.outlier_detection(df)
        
        # 相关性分析
        correlation_analysis = self.statistical_analyzer.correlation_analysis(df)
        
        # 生成洞察和建议
        insights.append(f"数据质量检查发现 {len(quality_issues)} 个问题")
        
        for issue in quality_issues:
            insights.append(f"- {issue['description']}")
            
            if issue["issue"] == "missing_values":
                recommendations.append("建议使用适当的插值方法处理缺失值")
            elif issue["issue"] == "duplicate_rows":
                recommendations.append("建议移除重复行以避免数据偏差")
            elif issue["issue"] == "data_type_inconsistency":
                recommendations.append("建议检查并转换数据类型以确保一致性")
        
        return AnalysisResult(
            success=True,
            analysis_type=config.analysis_type,
            data={
                "quality_issues": quality_issues,
                "outlier_analysis": outlier_analysis,
                "correlation_analysis": correlation_analysis
            },
            visualizations=visualizations,
            insights=insights,
            recommendations=recommendations
        )
    
    def generate_report(
        self,
        analysis_result: AnalysisResult,
        format_type: str = "html"
    ) -> str:
        """生成分析报告"""
        report_path = self.output_dir / f"analysis_report.{format_type}"
        
        if format_type == "html":
            html_content = self._generate_html_report(analysis_result)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        elif format_type == "json":
            # 序列化结果
            report_data = {
                "success": analysis_result.success,
                "analysis_type": analysis_result.analysis_type.value,
                "metrics": analysis_result.metrics,
                "insights": analysis_result.insights,
                "recommendations": analysis_result.recommendations,
                "visualizations": analysis_result.visualizations,
                "processing_time": analysis_result.processing_time,
                "metadata": analysis_result.metadata
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return str(report_path)
    
    def _generate_html_report(self, result: AnalysisResult) -> str:
        """生成HTML报告"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>数据分析报告</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .metric { background-color: #e8f4fd; padding: 10px; margin: 5px 0; border-radius: 3px; }
                .insight { background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }
                .recommendation { background-color: #d1ecf1; padding: 10px; margin: 5px 0; border-radius: 3px; }
                .visualization { margin: 10px 0; }
                img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>数据分析报告</h1>
                <p>分析类型: {analysis_type}</p>
                <p>处理时间: {processing_time:.2f} 秒</p>
                <p>生成时间: {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>关键指标</h2>
                {metrics_html}
            </div>
            
            <div class="section">
                <h2>分析洞察</h2>
                {insights_html}
            </div>
            
            <div class="section">
                <h2>建议</h2>
                {recommendations_html}
            </div>
            
            <div class="section">
                <h2>可视化图表</h2>
                {visualizations_html}
            </div>
        </body>
        </html>
        """
        
        # 生成指标HTML
        metrics_html = ""
        for key, value in result.metrics.items():
            if isinstance(value, (int, float)):
                metrics_html += f'<div class="metric"><strong>{key}:</strong> {value:.4f}</div>'
            elif isinstance(value, str):
                metrics_html += f'<div class="metric"><strong>{key}:</strong> {value}</div>'
        
        # 生成洞察HTML
        insights_html = ""
        for insight in result.insights:
            insights_html += f'<div class="insight">{insight}</div>'
        
        # 生成建议HTML
        recommendations_html = ""
        for recommendation in result.recommendations:
            recommendations_html += f'<div class="recommendation">{recommendation}</div>'
        
        # 生成可视化HTML
        visualizations_html = ""
        for viz_path in result.visualizations:
            if Path(viz_path).exists():
                visualizations_html += f'<div class="visualization"><img src="{viz_path}" alt="Visualization"></div>'
        
        return html_template.format(
            analysis_type=result.analysis_type.value,
            processing_time=result.processing_time,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metrics_html=metrics_html,
            insights_html=insights_html,
            recommendations_html=recommendations_html,
            visualizations_html=visualizations_html
        )


# 使用示例
if __name__ == "__main__":
    async def example_usage():
        # 创建分析引擎
        engine = DataAnalyticsEngine()
        
        # 生成示例数据
        np.random.seed(42)
        data = {
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(2, 1.5, 1000),
            'feature3': np.random.exponential(1, 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000),
            'target': np.random.choice([0, 1], 1000)
        }
        df = pd.DataFrame(data)
        
        # 描述性分析
        desc_config = AnalysisConfig(
            analysis_type=AnalysisType.DESCRIPTIVE
        )
        desc_result = await engine.analyze_data(df, desc_config)
        print(f"描述性分析完成: {desc_result.success}")
        
        # 预测性分析
        pred_config = AnalysisConfig(
            analysis_type=AnalysisType.PREDICTIVE,
            model_type=ModelType.CLASSIFICATION,
            target_column='target',
            feature_columns=['feature1', 'feature2', 'feature3', 'category']
        )
        pred_result = await engine.analyze_data(df, pred_config)
        print(f"预测性分析完成: {pred_result.success}")
        
        # 生成报告
        if pred_result.success:
            report_path = engine.generate_report(pred_result, "html")
            print(f"报告已生成: {report_path}")
    
    # 运行示例
    asyncio.run(example_usage())