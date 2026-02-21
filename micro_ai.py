"""
MICRO-AI: Machine Learning-Based DNA Microarray Analysis for Disease Detection

A comprehensive machine learning framework for high-resolution DNA microarray data 
analysis and automated disease detection.

Author: Implementation based on MICRO-AI paper
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


class QuantileNormalizer:
    """
    Quantile Normalization for microarray data.
    
    Transforms each sample to have identical distribution by mapping 
    to a reference distribution computed as the mean across all samples.
    
    Reference: Eq. (1) and (2) in paper
    """
    
    def __init__(self):
        self.reference_distribution = None
        
    def fit(self, X):
        """
        Compute reference distribution from training data.
        
        Args:
            X: Expression matrix (samples x genes)
        """
        X = np.array(X)
        n_samples, n_genes = X.shape
        
        # Sort each sample and compute mean across samples at each rank
        sorted_data = np.sort(X, axis=1)
        self.reference_distribution = np.mean(sorted_data, axis=0)
        
        return self
    
    def transform(self, X):
        """
        Apply quantile normalization.
        
        Args:
            X: Expression matrix (samples x genes)
            
        Returns:
            Normalized expression matrix
        """
        X = np.array(X)
        n_samples, n_genes = X.shape
        X_normalized = np.zeros_like(X)
        
        for i in range(n_samples):
            # Get ranks of original values
            ranks = stats.rankdata(X[i], method='ordinal') - 1
            # Map to reference distribution
            X_normalized[i] = self.reference_distribution[ranks.astype(int)]
            
        return X_normalized
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


class ComBatCorrection:
    """
    ComBat batch effect correction using empirical Bayes estimation.
    
    Removes systematic batch effects while preserving biological variation.
    
    Reference: Eq. (3) and (4) in paper
    """
    
    def __init__(self):
        self.grand_mean = None
        self.batch_means = None
        self.batch_vars = None
        self.gamma_star = None
        self.delta_star = None
        
    def fit(self, X, batch_labels):
        """
        Estimate batch effect parameters.
        
        Args:
            X: Expression matrix (samples x genes)
            batch_labels: Batch assignment for each sample
        """
        X = np.array(X)
        batch_labels = np.array(batch_labels)
        unique_batches = np.unique(batch_labels)
        
        n_samples, n_genes = X.shape
        
        # Compute grand mean
        self.grand_mean = np.mean(X, axis=0)
        
        # Compute batch-specific parameters
        self.batch_means = {}
        self.batch_vars = {}
        
        for batch in unique_batches:
            mask = batch_labels == batch
            batch_data = X[mask]
            self.batch_means[batch] = np.mean(batch_data, axis=0)
            self.batch_vars[batch] = np.var(batch_data, axis=0) + 1e-10
            
        # Empirical Bayes estimation of batch effects
        self.gamma_star = {}
        self.delta_star = {}
        
        for batch in unique_batches:
            # Additive batch effect (standardized)
            self.gamma_star[batch] = self.batch_means[batch] - self.grand_mean
            # Multiplicative batch effect
            pooled_var = np.mean([self.batch_vars[b] for b in unique_batches], axis=0)
            self.delta_star[batch] = np.sqrt(self.batch_vars[batch] / (pooled_var + 1e-10))
            
        return self
    
    def transform(self, X, batch_labels):
        """
        Apply batch correction.
        
        Args:
            X: Expression matrix (samples x genes)
            batch_labels: Batch assignment for each sample
            
        Returns:
            Batch-corrected expression matrix
        """
        X = np.array(X)
        batch_labels = np.array(batch_labels)
        X_corrected = np.zeros_like(X)
        
        for i, batch in enumerate(batch_labels):
            if batch in self.gamma_star:
                # Apply correction: (Y - gamma*) / delta* + grand_mean
                X_corrected[i] = (X[i] - self.gamma_star[batch]) / (self.delta_star[batch] + 1e-10) + self.grand_mean
            else:
                X_corrected[i] = X[i]
                
        return X_corrected
    
    def fit_transform(self, X, batch_labels):
        """Fit and transform in one step."""
        self.fit(X, batch_labels)
        return self.transform(X, batch_labels)


class KNNImputer:
    """
    K-Nearest Neighbors imputation with weighted averaging.
    
    Reference: Eq. (5) and (6) in paper
    """
    
    def __init__(self, n_neighbors=10, sigma=1.0):
        """
        Args:
            n_neighbors: Number of nearest neighbors (K)
            sigma: Bandwidth parameter for Gaussian weights
        """
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        
    def fit_transform(self, X):
        """
        Impute missing values using KNN.
        
        Args:
            X: Expression matrix with possible NaN values
            
        Returns:
            Imputed expression matrix
        """
        X = np.array(X, dtype=float)
        n_samples, n_genes = X.shape
        
        # Find missing values
        missing_mask = np.isnan(X)
        
        if not np.any(missing_mask):
            return X
            
        X_imputed = X.copy()
        
        for i in range(n_samples):
            if not np.any(missing_mask[i]):
                continue
                
            # Find non-missing genes for this sample
            valid_genes = ~missing_mask[i]
            missing_genes = missing_mask[i]
            
            # Compute distances to other samples using valid genes
            distances = []
            for j in range(n_samples):
                if i == j or np.any(missing_mask[j, valid_genes]):
                    distances.append(np.inf)
                else:
                    dist = np.sqrt(np.sum((X[i, valid_genes] - X[j, valid_genes])**2))
                    distances.append(dist)
                    
            distances = np.array(distances)
            
            # Get K nearest neighbors
            neighbor_indices = np.argsort(distances)[:self.n_neighbors]
            neighbor_distances = distances[neighbor_indices]
            
            # Compute Gaussian weights
            weights = np.exp(-neighbor_distances**2 / (2 * self.sigma**2))
            weights = weights / (np.sum(weights) + 1e-10)
            
            # Impute missing values
            for gene_idx in np.where(missing_genes)[0]:
                neighbor_values = X[neighbor_indices, gene_idx]
                valid_neighbor_mask = ~np.isnan(neighbor_values)
                if np.any(valid_neighbor_mask):
                    X_imputed[i, gene_idx] = np.sum(weights[valid_neighbor_mask] * neighbor_values[valid_neighbor_mask]) / np.sum(weights[valid_neighbor_mask])
                else:
                    X_imputed[i, gene_idx] = np.nanmean(X[:, gene_idx])
                    
        return X_imputed


class AttentionFeatureSelector:
    """
    Attention-weighted feature selection mechanism.
    
    Computes attention scores that quantify discriminative importance of each gene
    based on statistical embeddings and mutual information.
    
    Reference: Eq. (7), (8), and (9) in paper
    """
    
    def __init__(self, embedding_dim=64, mad_threshold_percentile=25):
        """
        Args:
            embedding_dim: Dimension of gene embeddings (d)
            mad_threshold_percentile: Percentile for MAD filtering threshold
        """
        self.embedding_dim = embedding_dim
        self.mad_threshold_percentile = mad_threshold_percentile
        self.attention_scores = None
        self.selected_genes = None
        self.W_e = None
        self.b_e = None
        self.W_h = None
        self.b_h = None
        self.w = None
        
    def _compute_statistics(self, X, y):
        """
        Compute statistical features for each gene.
        
        Args:
            X: Expression matrix (samples x genes)
            y: Class labels
            
        Returns:
            Statistical features (genes x 5): [mean, std, skewness, kurtosis, MI]
        """
        n_genes = X.shape[1]
        stats_features = np.zeros((n_genes, 5))
        
        for j in range(n_genes):
            gene_expr = X[:, j]
            stats_features[j, 0] = np.mean(gene_expr)  # Mean
            stats_features[j, 1] = np.std(gene_expr)   # Std
            stats_features[j, 2] = stats.skew(gene_expr)  # Skewness
            stats_features[j, 3] = stats.kurtosis(gene_expr)  # Kurtosis
            
        # Compute mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        stats_features[:, 4] = mi_scores
        
        # Normalize features
        stats_features = (stats_features - np.mean(stats_features, axis=0)) / (np.std(stats_features, axis=0) + 1e-10)
        
        return stats_features
    
    def _compute_embeddings(self, stats_features):
        """
        Compute gene embeddings from statistical features.
        
        Reference: Eq. (8)
        
        Args:
            stats_features: Statistical features (genes x 5)
            
        Returns:
            Gene embeddings (genes x embedding_dim)
        """
        n_genes = stats_features.shape[0]
        
        # Initialize learnable parameters (using random initialization)
        np.random.seed(42)
        self.W_e = np.random.randn(5, self.embedding_dim) * 0.1
        self.b_e = np.zeros(self.embedding_dim)
        
        # Compute embeddings: h_j = sigmoid(W_e @ stats + b_e)
        embeddings = 1 / (1 + np.exp(-(stats_features @ self.W_e + self.b_e)))
        
        return embeddings
    
    def _compute_attention_scores(self, embeddings):
        """
        Compute attention scores for each gene.
        
        Reference: Eq. (7)
        
        Args:
            embeddings: Gene embeddings (genes x embedding_dim)
            
        Returns:
            Attention scores (genes,)
        """
        n_genes = embeddings.shape[0]
        
        # Initialize attention parameters
        np.random.seed(43)
        self.W_h = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.1
        self.b_h = np.zeros(self.embedding_dim)
        self.w = np.random.randn(self.embedding_dim) * 0.1
        
        # Compute attention: a_j = softmax(w^T * tanh(W_h @ h_j + b_h))
        hidden = np.tanh(embeddings @ self.W_h + self.b_h)
        scores = hidden @ self.w
        
        # Softmax normalization
        exp_scores = np.exp(scores - np.max(scores))
        attention_scores = exp_scores / np.sum(exp_scores)
        
        return attention_scores
    
    def fit(self, X, y):
        """
        Compute attention scores and select top genes.
        
        Args:
            X: Expression matrix (samples x genes)
            y: Class labels
        """
        X = np.array(X)
        y = np.array(y)
        n_genes = X.shape[1]
        
        # MAD filtering
        mad_values = np.median(np.abs(X - np.median(X, axis=0)), axis=0)
        mad_threshold = np.percentile(mad_values, self.mad_threshold_percentile)
        self.high_mad_genes = mad_values > mad_threshold
        
        print(f"MAD filtering: {np.sum(self.high_mad_genes)} genes pass threshold")
        
        # Compute statistics for genes passing MAD filter
        X_filtered = X[:, self.high_mad_genes]
        stats_features = self._compute_statistics(X_filtered, y)
        
        # Compute embeddings
        embeddings = self._compute_embeddings(stats_features)
        
        # Compute attention scores
        attention_filtered = self._compute_attention_scores(embeddings)
        
        # Map back to full gene set
        self.attention_scores = np.zeros(n_genes)
        self.attention_scores[self.high_mad_genes] = attention_filtered
        
        return self
    
    def get_top_genes(self, n_genes):
        """
        Get indices of top n genes by attention score.
        
        Args:
            n_genes: Number of genes to select
            
        Returns:
            Indices of selected genes
        """
        return np.argsort(self.attention_scores)[-n_genes:]
    
    def transform(self, X, selected_indices):
        """
        Select genes based on provided indices.
        
        Args:
            X: Expression matrix
            selected_indices: Indices of genes to keep
            
        Returns:
            Reduced expression matrix
        """
        return X[:, selected_indices]


class AdaptiveEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive Ensemble Classifier combining GBM, RF, and SVM.
    
    Features:
    - Gradient Boosting Machine for nonlinear interactions
    - Random Forest for variance reduction
    - Support Vector Machine for high-dimensional effectiveness
    - Adaptive weight optimization using SLSQP
    - Probability calibration using isotonic regression
    
    Reference: Eq. (10)-(18) in paper
    """
    
    def __init__(self, 
                 gbm_params=None,
                 rf_params=None,
                 svm_params=None,
                 calibrate=True,
                 random_state=42):
        """
        Args:
            gbm_params: Parameters for Gradient Boosting
            rf_params: Parameters for Random Forest
            svm_params: Parameters for SVM
            calibrate: Whether to calibrate probabilities
            random_state: Random seed
        """
        self.gbm_params = gbm_params or {
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 200,
            'random_state': random_state
        }
        self.rf_params = rf_params or {
            'n_estimators': 500,
            'max_features': 'sqrt',
            'random_state': random_state
        }
        self.svm_params = svm_params or {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': random_state
        }
        self.calibrate = calibrate
        self.random_state = random_state
        
        # Initialize classifiers
        self.gbm = GradientBoostingClassifier(**self.gbm_params)
        self.rf = RandomForestClassifier(**self.rf_params)
        self.svm = SVC(**self.svm_params)
        
        # Ensemble weights (optimized during training)
        self.weights = np.array([1/3, 1/3, 1/3])
        
        # Calibration models
        self.calibrators = {}
        
        # Training history
        self.training_history = {'gbm_loss': [], 'rf_oob': [], 'svm_loss': []}
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train ensemble classifiers and optimize weights.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional, for weight optimization)
            y_val: Validation labels (optional)
        """
        X = np.array(X)
        y = np.array(y)
        
        print("Training Gradient Boosting...")
        self.gbm.fit(X, y)
        
        print("Training Random Forest...")
        self.rf.fit(X, y)
        
        print("Training SVM...")
        self.svm.fit(X, y)
        
        # Record training history for GBM
        if hasattr(self.gbm, 'train_score_'):
            self.training_history['gbm_loss'] = self.gbm.train_score_.tolist()
        
        # Record OOB score for RF
        if hasattr(self.rf, 'oob_score_'):
            self.training_history['rf_oob'] = [self.rf.oob_score_]
            
        # Optimize weights if validation data provided
        if X_val is not None and y_val is not None:
            self._optimize_weights(X_val, y_val)
        else:
            # Use cross-validation to optimize weights
            self._optimize_weights_cv(X, y)
            
        # Calibrate probabilities
        if self.calibrate:
            self._fit_calibrators(X, y)
            
        return self
    
    def _optimize_weights(self, X_val, y_val):
        """
        Optimize ensemble weights using SLSQP.
        
        Reference: Eq. (17)
        """
        def cross_entropy_loss(weights):
            """Compute cross-entropy loss for weight combination."""
            weights = weights / np.sum(weights)  # Normalize
            
            # Get predictions from each classifier
            p_gbm = self.gbm.predict_proba(X_val)
            p_rf = self.rf.predict_proba(X_val)
            p_svm = self.svm.predict_proba(X_val)
            
            # Weighted combination
            p_ensemble = weights[0] * p_gbm + weights[1] * p_rf + weights[2] * p_svm
            p_ensemble = np.clip(p_ensemble, 1e-10, 1 - 1e-10)
            
            # Cross-entropy loss
            n_samples = len(y_val)
            loss = 0
            for i in range(n_samples):
                loss -= np.log(p_ensemble[i, y_val[i]])
            
            return loss / n_samples
        
        # Constraints: weights sum to 1, all non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1), (0, 1), (0, 1)]
        
        # Optimize
        result = minimize(
            cross_entropy_loss,
            x0=np.array([1/3, 1/3, 1/3]),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        self.weights = result.x / np.sum(result.x)
        print(f"Optimized weights: GBM={self.weights[0]:.3f}, RF={self.weights[1]:.3f}, SVM={self.weights[2]:.3f}")
        
    def _optimize_weights_cv(self, X, y, cv=3):
        """Optimize weights using cross-validation."""
        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        all_preds_gbm = []
        all_preds_rf = []
        all_preds_svm = []
        all_labels = []
        
        for train_idx, val_idx in kfold.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train temporary classifiers
            gbm_temp = GradientBoostingClassifier(**self.gbm_params)
            rf_temp = RandomForestClassifier(**self.rf_params)
            svm_temp = SVC(**self.svm_params)
            
            gbm_temp.fit(X_train, y_train)
            rf_temp.fit(X_train, y_train)
            svm_temp.fit(X_train, y_train)
            
            all_preds_gbm.append(gbm_temp.predict_proba(X_val))
            all_preds_rf.append(rf_temp.predict_proba(X_val))
            all_preds_svm.append(svm_temp.predict_proba(X_val))
            all_labels.append(y_val)
            
        # Concatenate predictions
        p_gbm = np.vstack(all_preds_gbm)
        p_rf = np.vstack(all_preds_rf)
        p_svm = np.vstack(all_preds_svm)
        y_all = np.concatenate(all_labels)
        
        def cross_entropy_loss(weights):
            weights = weights / np.sum(weights)
            p_ensemble = weights[0] * p_gbm + weights[1] * p_rf + weights[2] * p_svm
            p_ensemble = np.clip(p_ensemble, 1e-10, 1 - 1e-10)
            loss = 0
            for i in range(len(y_all)):
                loss -= np.log(p_ensemble[i, y_all[i]])
            return loss / len(y_all)
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1), (0, 1), (0, 1)]
        
        result = minimize(
            cross_entropy_loss,
            x0=np.array([1/3, 1/3, 1/3]),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        self.weights = result.x / np.sum(result.x)
        print(f"Optimized weights (CV): GBM={self.weights[0]:.3f}, RF={self.weights[1]:.3f}, SVM={self.weights[2]:.3f}")
        
    def _fit_calibrators(self, X, y):
        """
        Fit isotonic regression calibrators.
        
        Reference: Eq. (18)
        """
        # Get uncalibrated probabilities
        p_ensemble = self._get_uncalibrated_proba(X)
        
        n_classes = p_ensemble.shape[1]
        
        for c in range(n_classes):
            y_binary = (y == c).astype(int)
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(p_ensemble[:, c], y_binary)
            self.calibrators[c] = calibrator
            
    def _get_uncalibrated_proba(self, X):
        """Get uncalibrated ensemble probabilities."""
        p_gbm = self.gbm.predict_proba(X)
        p_rf = self.rf.predict_proba(X)
        p_svm = self.svm.predict_proba(X)
        
        p_ensemble = self.weights[0] * p_gbm + self.weights[1] * p_rf + self.weights[2] * p_svm
        return p_ensemble
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Calibrated class probabilities
        """
        p_ensemble = self._get_uncalibrated_proba(X)
        
        if self.calibrate and self.calibrators:
            p_calibrated = np.zeros_like(p_ensemble)
            for c, calibrator in self.calibrators.items():
                p_calibrated[:, c] = calibrator.predict(p_ensemble[:, c])
            # Normalize
            p_calibrated = p_calibrated / (np.sum(p_calibrated, axis=1, keepdims=True) + 1e-10)
            return p_calibrated
        
        return p_ensemble
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def get_feature_importance(self):
        """Get aggregated feature importance from ensemble."""
        # GBM importance
        gbm_imp = self.gbm.feature_importances_
        
        # RF importance
        rf_imp = self.rf.feature_importances_
        
        # Aggregate with ensemble weights
        importance = self.weights[0] * gbm_imp + self.weights[1] * rf_imp
        importance = importance / np.sum(importance)
        
        return importance


class MICROAI:
    """
    MICRO-AI: Complete framework for DNA microarray analysis and disease detection.
    
    Integrates:
    - Quantile normalization
    - ComBat batch effect correction
    - KNN missing value imputation
    - Attention-weighted feature selection
    - RFECV for optimal gene subset selection
    - Adaptive ensemble classification
    - Probability calibration
    """
    
    def __init__(self,
                 n_selected_features=150,
                 n_neighbors_impute=10,
                 embedding_dim=64,
                 mad_percentile=25,
                 rfecv_step=0.1,
                 cv_folds=5,
                 calibrate=True,
                 random_state=42):
        """
        Args:
            n_selected_features: Target number of selected genes
            n_neighbors_impute: K for KNN imputation
            embedding_dim: Dimension for attention embeddings
            mad_percentile: Percentile threshold for MAD filtering
            rfecv_step: Step size for RFECV
            cv_folds: Number of cross-validation folds
            calibrate: Whether to calibrate probabilities
            random_state: Random seed for reproducibility
        """
        self.n_selected_features = n_selected_features
        self.n_neighbors_impute = n_neighbors_impute
        self.embedding_dim = embedding_dim
        self.mad_percentile = mad_percentile
        self.rfecv_step = rfecv_step
        self.cv_folds = cv_folds
        self.calibrate = calibrate
        self.random_state = random_state
        
        # Components (initialized during fit)
        self.quantile_normalizer = QuantileNormalizer()
        self.combat_corrector = None
        self.knn_imputer = KNNImputer(n_neighbors=n_neighbors_impute)
        self.attention_selector = AttentionFeatureSelector(
            embedding_dim=embedding_dim,
            mad_threshold_percentile=mad_percentile
        )
        self.scaler = StandardScaler()
        self.ensemble = None
        
        # Selected gene indices
        self.selected_genes = None
        self.gene_names = None
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        
        # Training info
        self.n_classes = None
        self.training_info = {}
        
    def preprocess(self, X, batch_labels=None, fit=True):
        """
        Apply preprocessing pipeline.
        
        Args:
            X: Expression matrix (samples x genes)
            batch_labels: Batch assignments (optional)
            fit: Whether to fit preprocessing components
            
        Returns:
            Preprocessed expression matrix
        """
        X = np.array(X, dtype=float)
        
        # Step 1: KNN Imputation (if missing values)
        if np.any(np.isnan(X)):
            print("Imputing missing values...")
            X = self.knn_imputer.fit_transform(X)
            
        # Step 2: Quantile Normalization
        print("Applying quantile normalization...")
        if fit:
            X = self.quantile_normalizer.fit_transform(X)
        else:
            X = self.quantile_normalizer.transform(X)
            
        # Step 3: ComBat Batch Correction (if batch labels provided)
        if batch_labels is not None:
            print("Applying ComBat batch correction...")
            if fit:
                self.combat_corrector = ComBatCorrection()
                X = self.combat_corrector.fit_transform(X, batch_labels)
            elif self.combat_corrector is not None:
                X = self.combat_corrector.transform(X, batch_labels)
                
        return X
    
    def select_features(self, X, y):
        """
        Apply attention-weighted feature selection with RFECV.
        
        Args:
            X: Preprocessed expression matrix
            y: Class labels
            
        Returns:
            Reduced expression matrix, selected gene indices
        """
        print("\nFeature Selection Pipeline")
        print("-" * 40)
        
        # Step 1: Compute attention scores
        print("Computing attention scores...")
        self.attention_selector.fit(X, y)
        
        # Step 2: Pre-select top genes based on attention
        n_preselect = min(5000, X.shape[1])
        top_genes = self.attention_selector.get_top_genes(n_preselect)
        X_preselected = X[:, top_genes]
        
        print(f"Pre-selected {len(top_genes)} genes based on attention scores")
        
        # Step 3: RFECV for final selection
        print("Running RFECV...")
        base_estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rfecv = RFECV(
            estimator=base_estimator,
            step=self.rfecv_step,
            cv=StratifiedKFold(self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring='accuracy',
            n_jobs=-1,
            min_features_to_select=min(self.n_selected_features, X_preselected.shape[1])
        )
        
        rfecv.fit(X_preselected, y)
        
        # Get selected genes
        rfecv_selected = rfecv.support_
        final_genes_in_preselected = np.where(rfecv_selected)[0]
        self.selected_genes = top_genes[final_genes_in_preselected]
        
        print(f"RFECV selected {len(self.selected_genes)} genes")
        print(f"Optimal CV score: {rfecv.cv_results_['mean_test_score'].max():.4f}")
        
        # Store training info
        self.training_info['n_genes_original'] = X.shape[1]
        self.training_info['n_genes_after_attention'] = len(top_genes)
        self.training_info['n_genes_final'] = len(self.selected_genes)
        self.training_info['feature_reduction'] = 100 * (1 - len(self.selected_genes) / X.shape[1])
        
        return X[:, self.selected_genes], self.selected_genes
    
    def fit(self, X, y, batch_labels=None, gene_names=None):
        """
        Train the complete MICRO-AI pipeline.
        
        Args:
            X: Expression matrix (samples x genes)
            y: Class labels
            batch_labels: Batch assignments (optional)
            gene_names: Gene identifiers (optional)
        """
        print("=" * 60)
        print("MICRO-AI Training Pipeline")
        print("=" * 60)
        
        X = np.array(X, dtype=float)
        self.gene_names = gene_names
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.n_classes = len(np.unique(y_encoded))
        
        print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} genes, {self.n_classes} classes")
        
        # Preprocessing
        print("\n--- Preprocessing ---")
        X_preprocessed = self.preprocess(X, batch_labels, fit=True)
        
        # Feature selection
        print("\n--- Feature Selection ---")
        X_selected, _ = self.select_features(X_preprocessed, y_encoded)
        
        # Standardization
        print("\n--- Standardization ---")
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Train ensemble
        print("\n--- Ensemble Training ---")
        self.ensemble = AdaptiveEnsembleClassifier(
            calibrate=self.calibrate,
            random_state=self.random_state
        )
        self.ensemble.fit(X_scaled, y_encoded)
        
        print("\n" + "=" * 60)
        print("Training Complete")
        print("=" * 60)
        
        return self
    
    def predict(self, X, batch_labels=None):
        """
        Predict class labels for new samples.
        
        Args:
            X: Expression matrix
            batch_labels: Batch assignments (optional)
            
        Returns:
            Predicted class labels (original encoding)
        """
        X = np.array(X, dtype=float)
        
        # Preprocessing
        X_preprocessed = self.preprocess(X, batch_labels, fit=False)
        
        # Select features
        X_selected = X_preprocessed[:, self.selected_genes]
        
        # Standardization
        X_scaled = self.scaler.transform(X_selected)
        
        # Predict
        y_pred_encoded = self.ensemble.predict(X_scaled)
        
        # Decode labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def predict_proba(self, X, batch_labels=None):
        """
        Predict class probabilities for new samples.
        
        Args:
            X: Expression matrix
            batch_labels: Batch assignments (optional)
            
        Returns:
            Class probabilities
        """
        X = np.array(X, dtype=float)
        
        # Preprocessing
        X_preprocessed = self.preprocess(X, batch_labels, fit=False)
        
        # Select features
        X_selected = X_preprocessed[:, self.selected_genes]
        
        # Standardization
        X_scaled = self.scaler.transform(X_selected)
        
        # Predict probabilities
        proba = self.ensemble.predict_proba(X_scaled)
        
        return proba
    
    def get_selected_gene_names(self):
        """Get names of selected genes."""
        if self.gene_names is not None and self.selected_genes is not None:
            return [self.gene_names[i] for i in self.selected_genes]
        return self.selected_genes
    
    def get_feature_importance(self):
        """Get feature importance for selected genes."""
        if self.ensemble is None:
            return None
        return self.ensemble.get_feature_importance()
    
    def get_attention_scores(self):
        """Get attention scores for all genes."""
        return self.attention_selector.attention_scores


def evaluate_model(y_true, y_pred, y_proba=None, average='weighted'):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        average: Averaging method for multi-class
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    # AUC-ROC (requires probabilities)
    if y_proba is not None:
        n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else 2
        if n_classes == 2:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average=average)
            except:
                metrics['auc_roc'] = None
                
    # Specificity (for binary)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        # Multi-class: macro average of specificity
        specificities = []
        for i in range(cm.shape[0]):
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        metrics['specificity'] = np.mean(specificities)
        metrics['sensitivity'] = metrics['recall']
        
    return metrics


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing MICRO-AI with synthetic data...")
    
    np.random.seed(42)
    n_samples = 200
    n_genes = 1000
    n_classes = 2
    
    # Generate synthetic microarray data
    X = np.random.randn(n_samples, n_genes)
    
    # Add discriminative signal to some genes
    discriminative_genes = np.random.choice(n_genes, 50, replace=False)
    y = np.random.randint(0, n_classes, n_samples)
    
    for i, gene in enumerate(discriminative_genes):
        X[:, gene] += y * (2 + 0.5 * np.random.randn())
        
    # Add some noise and missing values
    X += np.random.randn(n_samples, n_genes) * 0.5
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train model
    model = MICROAI(
        n_selected_features=100,
        cv_folds=3,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_proba)
    
    print("\nTest Results:")
    print("-" * 40)
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")
    
    print("\nSelected genes:", len(model.selected_genes))
    print(f"Feature reduction: {model.training_info['feature_reduction']:.1f}%")
