# -*- coding: utf-8 -*-
"""
机器学习和AI模块
提供模型训练、预测、自然语言处理、计算机视觉等功能
"""

import asyncio
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import (
    Dict, List, Any, Optional, Union, Tuple, Callable,
    TypeVar, Generic, Iterator, AsyncIterator
)
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import io
import base64
import hashlib
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from contextlib import asynccontextmanager
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, KFold, TimeSeriesSplit
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder,
    OneHotEncoder, PolynomialFeatures, PowerTransformer
)
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFE, RFECV,
    f_classif, f_regression, mutual_info_classif, mutual_info_regression
)
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, NMF
from sklearn.manifold import TSNE, UMAP
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering,
    GaussianMixture
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    VotingClassifier, VotingRegressor, BaggingClassifier, BaggingRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor, PassiveAggressiveClassifier
)
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import (
    KNeighborsClassifier, KNeighborsRegressor,
    NearestNeighbors, LocalOutlierFactor
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin

# 深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# 自然语言处理库
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        pipeline, Trainer, TrainingArguments
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# 计算机视觉库
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# 时间序列库
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# 可视化库
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 其他工具库
from scipy import stats
from scipy.optimize import minimize
from scipy.spatial.distance import cosine, euclidean
import requests
import aiohttp
from jinja2 import Template
import redis
import aioredis
from motor.motor_asyncio import AsyncIOMotorClient
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Histogram, Gauge
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class ModelType(Enum):
    """模型类型枚举"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    DEEP_LEARNING = "deep_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class TaskType(Enum):
    """任务类型枚举"""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    LINEAR_REGRESSION = "linear_regression"
    POLYNOMIAL_REGRESSION = "polynomial_regression"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TEXT_CLASSIFICATION = "text_classification"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    MACHINE_TRANSLATION = "machine_translation"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    IMAGE_SEGMENTATION = "image_segmentation"
    FACE_RECOGNITION = "face_recognition"


class ModelStatus(Enum):
    """模型状态枚举"""
    CREATED = "created"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    DEPLOYED = "deployed"
    FAILED = "failed"
    DEPRECATED = "deprecated"


@dataclass
class ModelMetrics:
    """模型评估指标"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    silhouette_score: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    model_type: ModelType
    task_type: TaskType
    algorithm: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    preprocessing_steps: List[str] = field(default_factory=list)
    feature_selection: Optional[str] = None
    cross_validation: Dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: List[str] = field(default_factory=list)
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingResult:
    """训练结果"""
    model_id: str
    status: ModelStatus
    metrics: ModelMetrics
    model_path: Optional[str] = None
    training_history: Dict[str, List[float]] = field(default_factory=dict)
    validation_history: Dict[str, List[float]] = field(default_factory=dict)
    best_parameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)
    preprocessing_pipeline: Optional[Any] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class PredictionResult:
    """预测结果"""
    predictions: Union[np.ndarray, List[Any]]
    probabilities: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    prediction_time: Optional[float] = None
    model_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMLModel(ABC):
    """机器学习模型抽象基类"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.id = str(uuid.uuid4())
        self.status = ModelStatus.CREATED
        self.model = None
        self.preprocessing_pipeline = None
        self.feature_names = []
        self.target_names = []
        self.metrics = ModelMetrics()
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    @abstractmethod
    async def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> TrainingResult:
        """训练模型"""
        pass
    
    @abstractmethod
    async def predict(self, X: np.ndarray, **kwargs) -> PredictionResult:
        """预测"""
        pass
    
    @abstractmethod
    def save_model(self, file_path: str) -> bool:
        """保存模型"""
        pass
    
    @abstractmethod
    def load_model(self, file_path: str) -> bool:
        """加载模型"""
        pass
    
    def preprocess_data(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """数据预处理"""
        if self.preprocessing_pipeline is None:
            return X
        
        if fit:
            return self.preprocessing_pipeline.fit_transform(X)
        else:
            return self.preprocessing_pipeline.transform(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance))
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_).flatten()
            return dict(zip(self.feature_names, importance))
        return None


class SklearnModel(BaseMLModel):
    """Scikit-learn模型包装器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = self._create_model()
        self.preprocessing_pipeline = self._create_preprocessing_pipeline()
    
    def _create_model(self):
        """创建模型实例"""
        algorithm = self.config.algorithm
        params = self.config.hyperparameters
        
        model_map = {
            # 分类器
            'random_forest_classifier': RandomForestClassifier,
            'logistic_regression': LogisticRegression,
            'svm_classifier': SVC,
            'decision_tree_classifier': DecisionTreeClassifier,
            'gradient_boosting_classifier': GradientBoostingClassifier,
            'naive_bayes': GaussianNB,
            'knn_classifier': KNeighborsClassifier,
            'mlp_classifier': MLPClassifier,
            
            # 回归器
            'random_forest_regressor': RandomForestRegressor,
            'linear_regression': LinearRegression,
            'ridge_regression': Ridge,
            'lasso_regression': Lasso,
            'elastic_net': ElasticNet,
            'svm_regressor': SVR,
            'decision_tree_regressor': DecisionTreeRegressor,
            'gradient_boosting_regressor': GradientBoostingRegressor,
            'knn_regressor': KNeighborsRegressor,
            'mlp_regressor': MLPRegressor,
            
            # 聚类
            'kmeans': KMeans,
            'dbscan': DBSCAN,
            'gaussian_mixture': GaussianMixture,
            
            # 降维
            'pca': PCA,
            'tsne': TSNE,
            'umap': UMAP if 'umap' in globals() else None
        }
        
        model_class = model_map.get(algorithm)
        if model_class is None:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return model_class(**params)
    
    def _create_preprocessing_pipeline(self):
        """创建预处理管道"""
        steps = []
        
        for step in self.config.preprocessing_steps:
            if step == 'standard_scaler':
                steps.append(('scaler', StandardScaler()))
            elif step == 'minmax_scaler':
                steps.append(('scaler', MinMaxScaler()))
            elif step == 'robust_scaler':
                steps.append(('scaler', RobustScaler()))
            elif step == 'polynomial_features':
                steps.append(('poly', PolynomialFeatures()))
            elif step == 'pca':
                steps.append(('pca', PCA()))
        
        if steps:
            return Pipeline(steps)
        return None
    
    async def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> TrainingResult:
        """训练模型"""
        self.status = ModelStatus.TRAINING
        start_time = datetime.now()
        
        try:
            # 保存特征名称
            if hasattr(X, 'columns'):
                self.feature_names = list(X.columns)
            else:
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            # 数据预处理
            if self.preprocessing_pipeline:
                X_processed = self.preprocessing_pipeline.fit_transform(X)
            else:
                X_processed = X
            
            # 特征选择
            if self.config.feature_selection:
                X_processed = await self._apply_feature_selection(X_processed, y)
            
            # 交叉验证或简单训练
            cv_config = self.config.cross_validation
            if cv_config.get('enabled', False):
                result = await self._train_with_cv(X_processed, y, cv_config)
            else:
                result = await self._simple_train(X_processed, y)
            
            self.status = ModelStatus.TRAINED
            end_time = datetime.now()
            
            result.model_id = self.id
            result.status = self.status
            result.start_time = start_time
            result.end_time = end_time
            result.duration = (end_time - start_time).total_seconds()
            result.feature_names = self.feature_names
            result.preprocessing_pipeline = self.preprocessing_pipeline
            
            return result
        
        except Exception as e:
            self.status = ModelStatus.FAILED
            logger.error(f"模型训练失败: {e}", model_id=self.id)
            
            return TrainingResult(
                model_id=self.id,
                status=ModelStatus.FAILED,
                metrics=ModelMetrics(),
                error_message=str(e),
                start_time=start_time,
                end_time=datetime.now()
            )
    
    async def _simple_train(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """简单训练"""
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        training_start = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - training_start
        
        # 预测
        prediction_start = time.time()
        y_pred = self.model.predict(X_test)
        prediction_time = time.time() - prediction_start
        
        # 计算指标
        metrics = self._calculate_metrics(y_test, y_pred, X_test)
        metrics.training_time = training_time
        metrics.prediction_time = prediction_time
        
        # 特征重要性
        feature_importance = self.get_feature_importance()
        if feature_importance:
            metrics.feature_importance = feature_importance
        
        return TrainingResult(
            model_id=self.id,
            status=ModelStatus.TRAINED,
            metrics=metrics
        )
    
    async def _train_with_cv(self, X: np.ndarray, y: np.ndarray, cv_config: Dict[str, Any]) -> TrainingResult:
        """交叉验证训练"""
        cv_folds = cv_config.get('folds', 5)
        cv_strategy = cv_config.get('strategy', 'kfold')
        
        # 选择交叉验证策略
        if cv_strategy == 'kfold':
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        elif cv_strategy == 'stratified':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        elif cv_strategy == 'timeseries':
            cv = TimeSeriesSplit(n_splits=cv_folds)
        else:
            cv = cv_folds
        
        # 执行交叉验证
        if self.config.model_type == ModelType.CLASSIFICATION:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        else:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        
        # 训练最终模型
        training_start = time.time()
        self.model.fit(X, y)
        training_time = time.time() - training_start
        
        # 创建指标
        metrics = ModelMetrics(
            training_time=training_time,
            custom_metrics={
                'cv_mean_score': scores.mean(),
                'cv_std_score': scores.std(),
                'cv_scores': scores.tolist()
            }
        )
        
        return TrainingResult(
            model_id=self.id,
            status=ModelStatus.TRAINED,
            metrics=metrics
        )
    
    async def _apply_feature_selection(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """应用特征选择"""
        method = self.config.feature_selection
        
        if method == 'selectkbest':
            if self.config.model_type == ModelType.CLASSIFICATION:
                selector = SelectKBest(score_func=f_classif, k=10)
            else:
                selector = SelectKBest(score_func=f_regression, k=10)
        elif method == 'rfe':
            selector = RFE(estimator=self.model, n_features_to_select=10)
        elif method == 'from_model':
            selector = SelectFromModel(estimator=self.model)
        else:
            return X
        
        X_selected = selector.fit_transform(X, y)
        
        # 更新特征名称
        if hasattr(selector, 'get_support'):
            selected_features = selector.get_support()
            self.feature_names = [name for name, selected in zip(self.feature_names, selected_features) if selected]
        
        return X_selected
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, X_test: np.ndarray = None) -> ModelMetrics:
        """计算评估指标"""
        metrics = ModelMetrics()
        
        if self.config.model_type == ModelType.CLASSIFICATION:
            metrics.accuracy = accuracy_score(y_true, y_pred)
            
            # 多分类或二分类
            if len(np.unique(y_true)) > 2:
                metrics.precision = precision_score(y_true, y_pred, average='weighted')
                metrics.recall = recall_score(y_true, y_pred, average='weighted')
                metrics.f1_score = f1_score(y_true, y_pred, average='weighted')
            else:
                metrics.precision = precision_score(y_true, y_pred)
                metrics.recall = recall_score(y_true, y_pred)
                metrics.f1_score = f1_score(y_true, y_pred)
                
                # ROC AUC (仅二分类)
                if hasattr(self.model, 'predict_proba'):
                    y_proba = self.model.predict_proba(X_test)[:, 1]
                    metrics.roc_auc = roc_auc_score(y_true, y_proba)
            
            metrics.confusion_matrix = confusion_matrix(y_true, y_pred)
        
        elif self.config.model_type == ModelType.REGRESSION:
            metrics.mse = mean_squared_error(y_true, y_pred)
            metrics.mae = mean_absolute_error(y_true, y_pred)
            metrics.r2_score = r2_score(y_true, y_pred)
        
        elif self.config.model_type == ModelType.CLUSTERING:
            if X_test is not None:
                metrics.silhouette_score = silhouette_score(X_test, y_pred)
        
        return metrics
    
    async def predict(self, X: np.ndarray, **kwargs) -> PredictionResult:
        """预测"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        start_time = time.time()
        
        # 数据预处理
        if self.preprocessing_pipeline:
            X_processed = self.preprocessing_pipeline.transform(X)
        else:
            X_processed = X
        
        # 预测
        predictions = self.model.predict(X_processed)
        
        # 预测概率（如果支持）
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_processed)
        
        # 置信度分数（如果支持）
        confidence_scores = None
        if hasattr(self.model, 'decision_function'):
            confidence_scores = self.model.decision_function(X_processed)
        
        prediction_time = time.time() - start_time
        
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            confidence_scores=confidence_scores,
            feature_importance=self.get_feature_importance(),
            prediction_time=prediction_time,
            model_version=self.config.version
        )
    
    def save_model(self, file_path: str) -> bool:
        """保存模型"""
        try:
            model_data = {
                'model': self.model,
                'preprocessing_pipeline': self.preprocessing_pipeline,
                'config': self.config,
                'feature_names': self.feature_names,
                'target_names': self.target_names,
                'metrics': self.metrics,
                'created_at': self.created_at,
                'updated_at': self.updated_at
            }
            
            joblib.dump(model_data, file_path)
            logger.info(f"模型已保存: {file_path}", model_id=self.id)
            return True
        
        except Exception as e:
            logger.error(f"模型保存失败: {e}", model_id=self.id)
            return False
    
    def load_model(self, file_path: str) -> bool:
        """加载模型"""
        try:
            model_data = joblib.load(file_path)
            
            self.model = model_data['model']
            self.preprocessing_pipeline = model_data['preprocessing_pipeline']
            self.config = model_data['config']
            self.feature_names = model_data['feature_names']
            self.target_names = model_data['target_names']
            self.metrics = model_data['metrics']
            self.created_at = model_data['created_at']
            self.updated_at = model_data['updated_at']
            
            self.status = ModelStatus.TRAINED
            logger.info(f"模型已加载: {file_path}", model_id=self.id)
            return True
        
        except Exception as e:
            logger.error(f"模型加载失败: {e}", model_id=self.id)
            return False


class AutoMLModel(BaseMLModel):
    """自动机器学习模型"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.best_model = None
        self.search_results = []
    
    async def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> TrainingResult:
        """自动训练最佳模型"""
        self.status = ModelStatus.TRAINING
        start_time = datetime.now()
        
        try:
            # 保存特征名称
            if hasattr(X, 'columns'):
                self.feature_names = list(X.columns)
            else:
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            # 数据预处理
            X_processed, y_processed = await self._auto_preprocess(X, y)
            
            # 自动模型选择和超参数优化
            best_model, best_score, search_results = await self._auto_model_selection(X_processed, y_processed)
            
            self.model = best_model
            self.search_results = search_results
            
            # 最终训练
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.2, random_state=42
            )
            
            training_start = time.time()
            self.model.fit(X_train, y_train)
            training_time = time.time() - training_start
            
            # 评估
            y_pred = self.model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred, X_test)
            metrics.training_time = training_time
            
            self.status = ModelStatus.TRAINED
            end_time = datetime.now()
            
            return TrainingResult(
                model_id=self.id,
                status=self.status,
                metrics=metrics,
                best_parameters=self.model.get_params(),
                feature_names=self.feature_names,
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds()
            )
        
        except Exception as e:
            self.status = ModelStatus.FAILED
            logger.error(f"AutoML训练失败: {e}", model_id=self.id)
            
            return TrainingResult(
                model_id=self.id,
                status=ModelStatus.FAILED,
                metrics=ModelMetrics(),
                error_message=str(e),
                start_time=start_time,
                end_time=datetime.now()
            )
    
    async def _auto_preprocess(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """自动数据预处理"""
        # 处理缺失值
        if pd.isna(X).any().any():
            # 数值列用中位数填充，分类列用众数填充
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            categorical_columns = X.select_dtypes(include=['object', 'category']).columns
            
            for col in numeric_columns:
                X[col].fillna(X[col].median(), inplace=True)
            
            for col in categorical_columns:
                X[col].fillna(X[col].mode()[0], inplace=True)
        
        # 编码分类变量
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            # 使用独热编码
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
        
        # 标准化数值特征
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            scaler = StandardScaler()
            X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
            self.preprocessing_pipeline = scaler
        
        return X.values, y
    
    async def _auto_model_selection(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, float, List[Dict]]:
        """自动模型选择和超参数优化"""
        models_to_try = []
        
        if self.config.model_type == ModelType.CLASSIFICATION:
            models_to_try = [
                ('RandomForest', RandomForestClassifier(random_state=42), {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }),
                ('LogisticRegression', LogisticRegression(random_state=42), {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }),
                ('SVM', SVC(random_state=42), {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }),
                ('GradientBoosting', GradientBoostingClassifier(random_state=42), {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                })
            ]
            scoring = 'accuracy'
        
        elif self.config.model_type == ModelType.REGRESSION:
            models_to_try = [
                ('RandomForest', RandomForestRegressor(random_state=42), {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }),
                ('LinearRegression', LinearRegression(), {}),
                ('Ridge', Ridge(random_state=42), {
                    'alpha': [0.1, 1, 10, 100]
                }),
                ('SVR', SVR(), {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }),
                ('GradientBoosting', GradientBoostingRegressor(random_state=42), {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                })
            ]
            scoring = 'r2'
        
        best_model = None
        best_score = -np.inf
        search_results = []
        
        for name, model, param_grid in models_to_try:
            try:
                if param_grid:
                    # 网格搜索
                    grid_search = GridSearchCV(
                        model, param_grid, cv=5, scoring=scoring, n_jobs=-1
                    )
                    grid_search.fit(X, y)
                    
                    current_model = grid_search.best_estimator_
                    current_score = grid_search.best_score_
                    best_params = grid_search.best_params_
                else:
                    # 简单交叉验证
                    scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
                    current_score = scores.mean()
                    current_model = model
                    best_params = {}
                
                search_results.append({
                    'model_name': name,
                    'score': current_score,
                    'parameters': best_params
                })
                
                if current_score > best_score:
                    best_score = current_score
                    best_model = current_model
                
                logger.info(f"模型 {name} 评分: {current_score:.4f}")
            
            except Exception as e:
                logger.warning(f"模型 {name} 训练失败: {e}")
        
        return best_model, best_score, search_results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, X_test: np.ndarray = None) -> ModelMetrics:
        """计算评估指标"""
        metrics = ModelMetrics()
        
        if self.config.model_type == ModelType.CLASSIFICATION:
            metrics.accuracy = accuracy_score(y_true, y_pred)
            
            if len(np.unique(y_true)) > 2:
                metrics.precision = precision_score(y_true, y_pred, average='weighted')
                metrics.recall = recall_score(y_true, y_pred, average='weighted')
                metrics.f1_score = f1_score(y_true, y_pred, average='weighted')
            else:
                metrics.precision = precision_score(y_true, y_pred)
                metrics.recall = recall_score(y_true, y_pred)
                metrics.f1_score = f1_score(y_true, y_pred)
                
                if hasattr(self.model, 'predict_proba'):
                    y_proba = self.model.predict_proba(X_test)[:, 1]
                    metrics.roc_auc = roc_auc_score(y_true, y_proba)
            
            metrics.confusion_matrix = confusion_matrix(y_true, y_pred)
        
        elif self.config.model_type == ModelType.REGRESSION:
            metrics.mse = mean_squared_error(y_true, y_pred)
            metrics.mae = mean_absolute_error(y_true, y_pred)
            metrics.r2_score = r2_score(y_true, y_pred)
        
        return metrics
    
    async def predict(self, X: np.ndarray, **kwargs) -> PredictionResult:
        """预测"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        start_time = time.time()
        
        # 数据预处理
        if self.preprocessing_pipeline:
            X_processed = self.preprocessing_pipeline.transform(X)
        else:
            X_processed = X
        
        # 预测
        predictions = self.model.predict(X_processed)
        
        # 预测概率
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_processed)
        
        prediction_time = time.time() - start_time
        
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            prediction_time=prediction_time,
            model_version=self.config.version,
            metadata={'search_results': self.search_results}
        )
    
    def save_model(self, file_path: str) -> bool:
        """保存模型"""
        try:
            model_data = {
                'model': self.model,
                'preprocessing_pipeline': self.preprocessing_pipeline,
                'config': self.config,
                'feature_names': self.feature_names,
                'search_results': self.search_results,
                'created_at': self.created_at,
                'updated_at': self.updated_at
            }
            
            joblib.dump(model_data, file_path)
            logger.info(f"AutoML模型已保存: {file_path}", model_id=self.id)
            return True
        
        except Exception as e:
            logger.error(f"AutoML模型保存失败: {e}", model_id=self.id)
            return False
    
    def load_model(self, file_path: str) -> bool:
        """加载模型"""
        try:
            model_data = joblib.load(file_path)
            
            self.model = model_data['model']
            self.preprocessing_pipeline = model_data['preprocessing_pipeline']
            self.config = model_data['config']
            self.feature_names = model_data['feature_names']
            self.search_results = model_data['search_results']
            self.created_at = model_data['created_at']
            self.updated_at = model_data['updated_at']
            
            self.status = ModelStatus.TRAINED
            logger.info(f"AutoML模型已加载: {file_path}", model_id=self.id)
            return True
        
        except Exception as e:
            logger.error(f"AutoML模型加载失败: {e}", model_id=self.id)
            return False


class NLPModel(BaseMLModel):
    """自然语言处理模型"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tokenizer = None
        self.vectorizer = None
        self.nlp_pipeline = None
        
        if NLTK_AVAILABLE:
            # 下载必要的NLTK数据
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
            except:
                pass
    
    async def train(self, texts: List[str], labels: List[Any], **kwargs) -> TrainingResult:
        """训练NLP模型"""
        self.status = ModelStatus.TRAINING
        start_time = datetime.now()
        
        try:
            # 文本预处理
            processed_texts = await self._preprocess_texts(texts)
            
            # 特征提取
            X = await self._extract_features(processed_texts, fit=True)
            y = np.array(labels)
            
            # 创建和训练模型
            if self.config.task_type == TaskType.SENTIMENT_ANALYSIS:
                self.model = self._create_sentiment_model()
            elif self.config.task_type == TaskType.TEXT_CLASSIFICATION:
                self.model = self._create_text_classifier()
            else:
                raise ValueError(f"Unsupported NLP task: {self.config.task_type}")
            
            # 训练
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            training_start = time.time()
            self.model.fit(X_train, y_train)
            training_time = time.time() - training_start
            
            # 评估
            y_pred = self.model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred, X_test)
            metrics.training_time = training_time
            
            self.status = ModelStatus.TRAINED
            end_time = datetime.now()
            
            return TrainingResult(
                model_id=self.id,
                status=self.status,
                metrics=metrics,
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds()
            )
        
        except Exception as e:
            self.status = ModelStatus.FAILED
            logger.error(f"NLP模型训练失败: {e}", model_id=self.id)
            
            return TrainingResult(
                model_id=self.id,
                status=ModelStatus.FAILED,
                metrics=ModelMetrics(),
                error_message=str(e),
                start_time=start_time,
                end_time=datetime.now()
            )
    
    async def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """文本预处理"""
        processed_texts = []
        
        for text in texts:
            # 转换为小写
            text = text.lower()
            
            # 移除特殊字符（保留字母、数字、空格）
            import re
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            
            # 分词和去停用词
            if NLTK_AVAILABLE:
                tokens = word_tokenize(text)
                stop_words = set(stopwords.words('english'))
                tokens = [token for token in tokens if token not in stop_words]
                
                # 词干提取
                stemmer = PorterStemmer()
                tokens = [stemmer.stem(token) for token in tokens]
                
                text = ' '.join(tokens)
            
            processed_texts.append(text)
        
        return processed_texts
    
    async def _extract_features(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """特征提取"""
        if self.vectorizer is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        
        if fit:
            X = self.vectorizer.fit_transform(texts)
        else:
            X = self.vectorizer.transform(texts)
        
        return X.toarray()
    
    def _create_sentiment_model(self):
        """创建情感分析模型"""
        return Pipeline([
            ('classifier', LogisticRegression(random_state=42))
        ])
    
    def _create_text_classifier(self):
        """创建文本分类模型"""
        return Pipeline([
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, X_test: np.ndarray = None) -> ModelMetrics:
        """计算评估指标"""
        metrics = ModelMetrics()
        
        metrics.accuracy = accuracy_score(y_true, y_pred)
        
        if len(np.unique(y_true)) > 2:
            metrics.precision = precision_score(y_true, y_pred, average='weighted')
            metrics.recall = recall_score(y_true, y_pred, average='weighted')
            metrics.f1_score = f1_score(y_true, y_pred, average='weighted')
        else:
            metrics.precision = precision_score(y_true, y_pred)
            metrics.recall = recall_score(y_true, y_pred)
            metrics.f1_score = f1_score(y_true, y_pred)
        
        metrics.confusion_matrix = confusion_matrix(y_true, y_pred)
        
        return metrics
    
    async def predict(self, texts: List[str], **kwargs) -> PredictionResult:
        """预测"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        start_time = time.time()
        
        # 文本预处理
        processed_texts = await self._preprocess_texts(texts)
        
        # 特征提取
        X = await self._extract_features(processed_texts, fit=False)
        
        # 预测
        predictions = self.model.predict(X)
        
        # 预测概率
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
        
        prediction_time = time.time() - start_time
        
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            prediction_time=prediction_time,
            model_version=self.config.version
        )
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """情感分析"""
        if NLTK_AVAILABLE:
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'sentiment': 'positive' if scores['compound'] > 0.05 else 'negative' if scores['compound'] < -0.05 else 'neutral'
            }
        else:
            # 使用训练的模型
            result = await self.predict([text])
            return {
                'prediction': result.predictions[0],
                'probability': result.probabilities[0] if result.probabilities is not None else None
            }
    
    def save_model(self, file_path: str) -> bool:
        """保存模型"""
        try:
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'config': self.config,
                'created_at': self.created_at,
                'updated_at': self.updated_at
            }
            
            joblib.dump(model_data, file_path)
            logger.info(f"NLP模型已保存: {file_path}", model_id=self.id)
            return True
        
        except Exception as e:
            logger.error(f"NLP模型保存失败: {e}", model_id=self.id)
            return False
    
    def load_model(self, file_path: str) -> bool:
        """加载模型"""
        try:
            model_data = joblib.load(file_path)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.config = model_data['config']
            self.created_at = model_data['created_at']
            self.updated_at = model_data['updated_at']
            
            self.status = ModelStatus.TRAINED
            logger.info(f"NLP模型已加载: {file_path}", model_id=self.id)
            return True
        
        except Exception as e:
            logger.error(f"NLP模型加载失败: {e}", model_id=self.id)
            return False


class TimeSeriesModel(BaseMLModel):
    """时间序列模型"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.scaler = None
        self.lookback_window = config.hyperparameters.get('lookback_window', 30)
        self.forecast_horizon = config.hyperparameters.get('forecast_horizon', 7)
    
    async def train(self, data: pd.DataFrame, target_column: str, **kwargs) -> TrainingResult:
        """训练时间序列模型"""
        self.status = ModelStatus.TRAINING
        start_time = datetime.now()
        
        try:
            # 数据预处理
            processed_data = await self._preprocess_timeseries(data, target_column)
            
            # 创建训练数据
            X, y = self._create_sequences(processed_data[target_column].values)
            
            # 分割数据
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # 创建和训练模型
            algorithm = self.config.algorithm
            
            if algorithm == 'arima' and STATSMODELS_AVAILABLE:
                self.model = await self._train_arima(processed_data[target_column])
            elif algorithm == 'prophet' and PROPHET_AVAILABLE:
                self.model = await self._train_prophet(processed_data, target_column)
            elif algorithm == 'lstm' and TORCH_AVAILABLE:
                self.model = await self._train_lstm(X_train, y_train, X_test, y_test)
            else:
                # 使用传统机器学习方法
                self.model = await self._train_ml_timeseries(X_train, y_train)
            
            # 评估
            y_pred = await self._predict_timeseries(X_test)
            metrics = self._calculate_timeseries_metrics(y_test, y_pred)
            
            self.status = ModelStatus.TRAINED
            end_time = datetime.now()
            
            return TrainingResult(
                model_id=self.id,
                status=self.status,
                metrics=metrics,
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds()
            )
        
        except Exception as e:
            self.status = ModelStatus.FAILED
            logger.error(f"时间序列模型训练失败: {e}", model_id=self.id)
            
            return TrainingResult(
                model_id=self.id,
                status=ModelStatus.FAILED,
                metrics=ModelMetrics(),
                error_message=str(e),
                start_time=start_time,
                end_time=datetime.now()
            )
    
    async def _preprocess_timeseries(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """时间序列数据预处理"""
        # 确保数据按时间排序
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date')
        
        # 处理缺失值
        data[target_column] = data[target_column].interpolate()
        
        # 标准化
        self.scaler = StandardScaler()
        data[target_column] = self.scaler.fit_transform(data[[target_column]])
        
        return data
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """创建序列数据"""
        X, y = [], []
        
        for i in range(self.lookback_window, len(data) - self.forecast_horizon + 1):
            X.append(data[i-self.lookback_window:i])
            y.append(data[i:i+self.forecast_horizon])
        
        return np.array(X), np.array(y)
    
    async def _train_arima(self, series: pd.Series):
        """训练ARIMA模型"""
        # 自动选择ARIMA参数
        model = ARIMA(series, order=(1, 1, 1))
        fitted_model = model.fit()
        return fitted_model
    
    async def _train_prophet(self, data: pd.DataFrame, target_column: str):
        """训练Prophet模型"""
        # 准备Prophet数据格式
        prophet_data = data[['date', target_column]].copy()
        prophet_data.columns = ['ds', 'y']
        
        model = Prophet()
        model.fit(prophet_data)
        return model
    
    async def _train_lstm(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        """训练LSTM模型"""
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)
        y_train_tensor = torch.FloatTensor(y_train)
        
        # 创建模型
        model = LSTMModel(
            input_size=1,
            hidden_size=50,
            num_layers=2,
            output_size=self.forecast_horizon
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练
        epochs = 100
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        return model
    
    async def _train_ml_timeseries(self, X_train: np.ndarray, y_train: np.ndarray):
        """使用传统机器学习训练时间序列模型"""
        # 展平y_train用于单步预测
        y_train_flat = y_train[:, 0] if len(y_train.shape) > 1 else y_train
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train_flat)
        return model
    
    async def _predict_timeseries(self, X_test: np.ndarray) -> np.ndarray:
        """时间序列预测"""
        if hasattr(self.model, 'forecast'):
            # ARIMA模型
            forecast = self.model.forecast(steps=len(X_test))
            return forecast
        elif hasattr(self.model, 'predict') and hasattr(self.model, 'make_future_dataframe'):
            # Prophet模型
            future = self.model.make_future_dataframe(periods=len(X_test))
            forecast = self.model.predict(future)
            return forecast['yhat'].tail(len(X_test)).values
        elif TORCH_AVAILABLE and isinstance(self.model, nn.Module):
            # LSTM模型
            self.model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).unsqueeze(-1)
                predictions = self.model(X_test_tensor)
                return predictions.numpy()
        else:
            # 传统机器学习模型
            return self.model.predict(X_test)
    
    def _calculate_timeseries_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """计算时间序列指标"""
        metrics = ModelMetrics()
        
        # 确保维度一致
        if len(y_true.shape) > 1:
            y_true = y_true[:, 0]
        if len(y_pred.shape) > 1:
            y_pred = y_pred[:, 0]
        
        metrics.mse = mean_squared_error(y_true, y_pred)
        metrics.mae = mean_absolute_error(y_true, y_pred)
        metrics.r2_score = r2_score(y_true, y_pred)
        
        # 计算MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics.custom_metrics['mape'] = mape
        
        return metrics
    
    async def predict(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> PredictionResult:
        """预测"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        start_time = time.time()
        
        # 数据预处理
        if isinstance(data, pd.DataFrame):
            if self.scaler:
                data_scaled = self.scaler.transform(data)
            else:
                data_scaled = data.values
        else:
            if self.scaler:
                data_scaled = self.scaler.transform(data.reshape(-1, 1)).flatten()
            else:
                data_scaled = data
        
        # 创建序列
        if len(data_scaled) >= self.lookback_window:
            X = data_scaled[-self.lookback_window:].reshape(1, -1)
        else:
            raise ValueError(f"Need at least {self.lookback_window} data points")
        
        # 预测
        predictions = await self._predict_timeseries(X)
        
        # 反标准化
        if self.scaler:
            predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        prediction_time = time.time() - start_time
        
        return PredictionResult(
            predictions=predictions,
            prediction_time=prediction_time,
            model_version=self.config.version
        )
    
    def save_model(self, file_path: str) -> bool:
        """保存模型"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'config': self.config,
                'lookback_window': self.lookback_window,
                'forecast_horizon': self.forecast_horizon,
                'created_at': self.created_at,
                'updated_at': self.updated_at
            }
            
            if TORCH_AVAILABLE and isinstance(self.model, nn.Module):
                # 保存PyTorch模型
                torch.save(self.model.state_dict(), file_path + '.pth')
                model_data['model'] = None  # 不用pickle保存PyTorch模型
            
            joblib.dump(model_data, file_path)
            logger.info(f"时间序列模型已保存: {file_path}", model_id=self.id)
            return True
        
        except Exception as e:
            logger.error(f"时间序列模型保存失败: {e}", model_id=self.id)
            return False
    
    def load_model(self, file_path: str) -> bool:
        """加载模型"""
        try:
            model_data = joblib.load(file_path)
            
            self.scaler = model_data['scaler']
            self.config = model_data['config']
            self.lookback_window = model_data['lookback_window']
            self.forecast_horizon = model_data['forecast_horizon']
            self.created_at = model_data['created_at']
            self.updated_at = model_data['updated_at']
            
            if model_data['model'] is None and TORCH_AVAILABLE:
                # 加载PyTorch模型
                from torch import nn
                class LSTMModel(nn.Module):
                    def __init__(self, input_size, hidden_size, num_layers, output_size):
                        super(LSTMModel, self).__init__()
                        self.hidden_size = hidden_size
                        self.num_layers = num_layers
                        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                        self.fc = nn.Linear(hidden_size, output_size)
                    
                    def forward(self, x):
                        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                        out, _ = self.lstm(x, (h0, c0))
                        out = self.fc(out[:, -1, :])
                        return out
                
                self.model = LSTMModel(1, 50, 2, self.forecast_horizon)
                self.model.load_state_dict(torch.load(file_path + '.pth'))
            else:
                self.model = model_data['model']
            
            self.status = ModelStatus.TRAINED
            logger.info(f"时间序列模型已加载: {file_path}", model_id=self.id)
            return True
        
        except Exception as e:
            logger.error(f"时间序列模型加载失败: {e}", model_id=self.id)
            return False


class ModelManager:
    """模型管理器"""
    
    def __init__(self, storage_path: str = "./models"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.models: Dict[str, BaseMLModel] = {}
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self.metrics_collector = Counter('ml_model_operations_total')
        self.training_time_histogram = Histogram('ml_model_training_duration_seconds')
        self.prediction_time_histogram = Histogram('ml_model_prediction_duration_seconds')
    
    def create_model(self, config: ModelConfig) -> BaseMLModel:
        """创建模型"""
        try:
            if config.model_type == ModelType.NLP:
                model = NLPModel(config)
            elif config.model_type == ModelType.TIME_SERIES:
                model = TimeSeriesModel(config)
            elif config.algorithm == 'auto':
                model = AutoMLModel(config)
            else:
                model = SklearnModel(config)
            
            self.models[model.id] = model
            self.model_registry[model.id] = {
                'config': config,
                'created_at': datetime.now(),
                'status': ModelStatus.CREATED
            }
            
            self.metrics_collector.labels(operation='create').inc()
            logger.info(f"模型已创建: {model.id}", model_type=config.model_type.value)
            
            return model
        
        except Exception as e:
            logger.error(f"模型创建失败: {e}", config=config.name)
            raise
    
    async def train_model(self, model_id: str, X: np.ndarray, y: np.ndarray, **kwargs) -> TrainingResult:
        """训练模型"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        with self.training_time_histogram.time():
            result = await model.train(X, y, **kwargs)
        
        # 更新注册表
        self.model_registry[model_id].update({
            'status': result.status,
            'trained_at': datetime.now(),
            'metrics': result.metrics
        })
        
        self.metrics_collector.labels(operation='train').inc()
        
        # 自动保存模型
        if result.status == ModelStatus.TRAINED:
            model_path = self.storage_path / f"{model_id}.joblib"
            model.save_model(str(model_path))
            result.model_path = str(model_path)
        
        return result
    
    async def predict(self, model_id: str, X: np.ndarray, **kwargs) -> PredictionResult:
        """预测"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        if model.status != ModelStatus.TRAINED:
            raise ValueError(f"Model {model_id} is not trained")
        
        with self.prediction_time_histogram.time():
            result = await model.predict(X, **kwargs)
        
        self.metrics_collector.labels(operation='predict').inc()
        
        return result
    
    def get_model(self, model_id: str) -> Optional[BaseMLModel]:
        """获取模型"""
        return self.models.get(model_id)
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """列出所有模型"""
        return self.model_registry.copy()
    
    def delete_model(self, model_id: str) -> bool:
        """删除模型"""
        try:
            if model_id in self.models:
                del self.models[model_id]
            
            if model_id in self.model_registry:
                del self.model_registry[model_id]
            
            # 删除模型文件
            model_path = self.storage_path / f"{model_id}.joblib"
            if model_path.exists():
                model_path.unlink()
            
            # 删除PyTorch模型文件
            pth_path = self.storage_path / f"{model_id}.joblib.pth"
            if pth_path.exists():
                pth_path.unlink()
            
            self.metrics_collector.labels(operation='delete').inc()
            logger.info(f"模型已删除: {model_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"模型删除失败: {e}", model_id=model_id)
            return False
    
    def load_model(self, model_id: str, file_path: str) -> bool:
        """加载模型"""
        try:
            # 首先加载模型数据以获取配置
            model_data = joblib.load(file_path)
            config = model_data['config']
            
            # 创建模型实例
            model = self.create_model(config)
            model.id = model_id
            
            # 加载模型
            success = model.load_model(file_path)
            
            if success:
                self.models[model_id] = model
                self.model_registry[model_id] = {
                    'config': config,
                    'status': ModelStatus.TRAINED,
                    'loaded_at': datetime.now()
                }
                
                self.metrics_collector.labels(operation='load').inc()
                logger.info(f"模型已加载: {model_id}")
            
            return success
        
        except Exception as e:
            logger.error(f"模型加载失败: {e}", model_id=model_id, file_path=file_path)
            return False
    
    async def evaluate_model(self, model_id: str, X_test: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
        """评估模型"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        if model.status != ModelStatus.TRAINED:
            raise ValueError(f"Model {model_id} is not trained")
        
        # 预测
        result = await model.predict(X_test)
        y_pred = result.predictions
        
        # 计算指标
        if model.config.model_type == ModelType.CLASSIFICATION:
            metrics = ModelMetrics(
                accuracy=accuracy_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred, average='weighted'),
                recall=recall_score(y_test, y_pred, average='weighted'),
                f1_score=f1_score(y_test, y_pred, average='weighted'),
                confusion_matrix=confusion_matrix(y_test, y_pred)
            )
            
            if len(np.unique(y_test)) == 2 and result.probabilities is not None:
                metrics.roc_auc = roc_auc_score(y_test, result.probabilities[:, 1])
        
        elif model.config.model_type == ModelType.REGRESSION:
            metrics = ModelMetrics(
                mse=mean_squared_error(y_test, y_pred),
                mae=mean_absolute_error(y_test, y_pred),
                r2_score=r2_score(y_test, y_pred)
            )
        
        else:
            metrics = ModelMetrics()
        
        # 更新模型指标
        model.metrics = metrics
        self.model_registry[model_id]['metrics'] = metrics
        
        self.metrics_collector.labels(operation='evaluate').inc()
        
        return metrics
    
    def get_model_metrics(self, model_id: str) -> Optional[ModelMetrics]:
        """获取模型指标"""
        if model_id in self.models:
            return self.models[model_id].metrics
        return None
    
    def compare_models(self, model_ids: List[str], metric: str = 'accuracy') -> Dict[str, float]:
        """比较模型性能"""
        comparison = {}
        
        for model_id in model_ids:
            if model_id in self.models:
                metrics = self.models[model_id].metrics
                if hasattr(metrics, metric):
                    comparison[model_id] = getattr(metrics, metric)
                elif metric in metrics.custom_metrics:
                    comparison[model_id] = metrics.custom_metrics[metric]
        
        return comparison
    
    async def batch_predict(self, model_id: str, X_batch: List[np.ndarray], **kwargs) -> List[PredictionResult]:
        """批量预测"""
        results = []
        
        for X in X_batch:
            result = await self.predict(model_id, X, **kwargs)
            results.append(result)
        
        return results
    
    def export_model_info(self, model_id: str) -> Dict[str, Any]:
        """导出模型信息"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.model_registry[model_id].copy()
        
        if model_id in self.models:
            model = self.models[model_id]
            model_info.update({
                'feature_names': model.feature_names,
                'target_names': model.target_names,
                'feature_importance': model.get_feature_importance()
            })
        
        return model_info
    
    def get_training_history(self, model_id: str) -> Dict[str, Any]:
        """获取训练历史"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found")
        
        return {
            'model_id': model_id,
            'registry_info': self.model_registry[model_id],
            'current_status': self.models[model_id].status if model_id in self.models else None
        }


# 工厂函数
def create_classification_model(name: str, algorithm: str = 'random_forest_classifier', **kwargs) -> BaseMLModel:
    """创建分类模型"""
    config = ModelConfig(
        name=name,
        model_type=ModelType.CLASSIFICATION,
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        algorithm=algorithm,
        hyperparameters=kwargs
    )
    return SklearnModel(config)


def create_regression_model(name: str, algorithm: str = 'random_forest_regressor', **kwargs) -> BaseMLModel:
    """创建回归模型"""
    config = ModelConfig(
        name=name,
        model_type=ModelType.REGRESSION,
        task_type=TaskType.LINEAR_REGRESSION,
        algorithm=algorithm,
        hyperparameters=kwargs
    )
    return SklearnModel(config)


def create_nlp_model(name: str, task_type: TaskType = TaskType.TEXT_CLASSIFICATION, **kwargs) -> NLPModel:
    """创建NLP模型"""
    config = ModelConfig(
        name=name,
        model_type=ModelType.NLP,
        task_type=task_type,
        algorithm='text_classifier',
        hyperparameters=kwargs
    )
    return NLPModel(config)


def create_timeseries_model(name: str, algorithm: str = 'lstm', **kwargs) -> TimeSeriesModel:
    """创建时间序列模型"""
    config = ModelConfig(
        name=name,
        model_type=ModelType.TIME_SERIES,
        task_type=TaskType.TIME_SERIES_FORECASTING,
        algorithm=algorithm,
        hyperparameters=kwargs
    )
    return TimeSeriesModel(config)


def create_automl_model(name: str, model_type: ModelType, **kwargs) -> AutoMLModel:
    """创建AutoML模型"""
    task_type_map = {
        ModelType.CLASSIFICATION: TaskType.MULTICLASS_CLASSIFICATION,
        ModelType.REGRESSION: TaskType.LINEAR_REGRESSION
    }
    
    config = ModelConfig(
        name=name,
        model_type=model_type,
        task_type=task_type_map.get(model_type, TaskType.MULTICLASS_CLASSIFICATION),
        algorithm='auto',
        hyperparameters=kwargs
    )
    return AutoMLModel(config)


# 示例使用
async def example_usage():
    """示例用法"""
    # 创建模型管理器
    manager = ModelManager()
    
    # 创建分类模型
    config = ModelConfig(
        name="iris_classifier",
        model_type=ModelType.CLASSIFICATION,
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        algorithm="random_forest_classifier",
        hyperparameters={'n_estimators': 100, 'random_state': 42},
        preprocessing_steps=['standard_scaler'],
        cross_validation={'enabled': True, 'folds': 5}
    )
    
    model = manager.create_model(config)
    
    # 生成示例数据
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 训练模型
    result = await manager.train_model(model.id, X, y)
    print(f"训练结果: {result.status}, 准确率: {result.metrics.accuracy:.4f}")
    
    # 预测
    prediction_result = await manager.predict(model.id, X[:5])
    print(f"预测结果: {prediction_result.predictions}")
    
    # 评估模型
    metrics = await manager.evaluate_model(model.id, X[100:], y[100:])
    print(f"评估指标: 准确率={metrics.accuracy:.4f}, F1={metrics.f1_score:.4f}")
    
    # 创建NLP模型示例
    nlp_config = ModelConfig(
        name="sentiment_analyzer",
        model_type=ModelType.NLP,
        task_type=TaskType.SENTIMENT_ANALYSIS,
        algorithm="text_classifier"
    )
    
    nlp_model = manager.create_model(nlp_config)
    
    # 示例文本数据
    texts = ["I love this product!", "This is terrible", "It's okay"]
    labels = [1, 0, 1]  # 1: positive, 0: negative
    
    # 训练NLP模型
    nlp_result = await manager.train_model(nlp_model.id, texts, labels)
    print(f"NLP训练结果: {nlp_result.status}")
    
    # NLP预测
    nlp_prediction = await manager.predict(nlp_model.id, ["This is amazing!"])
    print(f"情感预测: {nlp_prediction.predictions}")


if __name__ == "__main__":
    asyncio.run(example_usage())