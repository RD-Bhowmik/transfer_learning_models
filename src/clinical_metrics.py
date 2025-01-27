import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from scipy import stats
import json
import os
import traceback

class ClinicalMetrics:
    """Medical-specific metrics for cervical cancer detection"""
    
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        
    def calculate_clinical_metrics(self, y_true, y_pred_prob):
        """Calculate comprehensive clinical metrics"""
        metrics = {
            'diagnostic_metrics': self._calculate_diagnostic_metrics(y_true, y_pred_prob),
            'risk_metrics': self._calculate_risk_metrics(y_true, y_pred_prob),
            'clinical_thresholds': self._calculate_optimal_thresholds(y_true, y_pred_prob),
            'uncertainty_metrics': self._calculate_uncertainty_metrics(y_pred_prob)
        }
        return metrics
    
    def _calculate_diagnostic_metrics(self, y_true, y_pred_prob):
        """Calculate diagnostic performance metrics"""
        y_pred = (y_pred_prob >= self.confidence_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate clinical metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Calculate likelihood ratios
        plr = sensitivity / (1 - specificity) if (1 - specificity) > 0 else float('inf')  # Positive Likelihood Ratio
        nlr = (1 - sensitivity) / specificity if specificity > 0 else float('inf')  # Negative Likelihood Ratio
        
        return {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(ppv),
            'npv': float(npv),
            'plr': float(plr),
            'nlr': float(nlr),
            'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
            'f1_score': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0
        }
    
    def _calculate_risk_metrics(self, y_true, y_pred_prob):
        """Calculate risk-related metrics"""
        # Calculate risk ratios
        risk_ratio = np.mean(y_pred_prob[y_true == 1]) / np.mean(y_pred_prob[y_true == 0])
        
        # Calculate odds ratio
        odds_pred = y_pred_prob / (1 - y_pred_prob)
        odds_ratio = np.mean(odds_pred[y_true == 1]) / np.mean(odds_pred[y_true == 0])
        
        # Calculate confidence intervals
        ci_level = 0.95
        z_score = stats.norm.ppf((1 + ci_level) / 2)
        
        return {
            'risk_ratio': float(risk_ratio),
            'odds_ratio': float(odds_ratio),
            'confidence_intervals': {
                'level': ci_level,
                'lower': float(risk_ratio - z_score * np.std(y_pred_prob)),
                'upper': float(risk_ratio + z_score * np.std(y_pred_prob))
            }
        }
    
    def _calculate_optimal_thresholds(self, y_true, y_pred_prob):
        """Calculate optimal clinical thresholds"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        
        # Youden's J statistic (maximizing sensitivity + specificity - 1)
        j_scores = tpr - fpr
        j_optimal_idx = np.argmax(j_scores)
        
        # High sensitivity threshold (for screening)
        high_sens_idx = np.argmin(np.abs(tpr - 0.95))  # 95% sensitivity
        
        # High specificity threshold (for confirmation)
        high_spec_idx = np.argmin(np.abs(1 - fpr - 0.95))  # 95% specificity
        
        return {
            'optimal_threshold': float(thresholds[j_optimal_idx]),
            'screening_threshold': float(thresholds[high_sens_idx]),
            'confirmation_threshold': float(thresholds[high_spec_idx])
        }
    
    def _calculate_uncertainty_metrics(self, y_pred_prob):
        """Calculate uncertainty and confidence metrics"""
        # Calculate prediction confidence
        confidence_scores = np.abs(y_pred_prob - 0.5) * 2
        
        # Calculate entropy
        entropy = -y_pred_prob * np.log2(y_pred_prob + 1e-10) - \
                 (1 - y_pred_prob) * np.log2(1 - y_pred_prob + 1e-10)
        
        return {
            'mean_confidence': float(np.mean(confidence_scores)),
            'confidence_std': float(np.std(confidence_scores)),
            'mean_entropy': float(np.mean(entropy)),
            'high_uncertainty_ratio': float(np.mean(confidence_scores < 0.5))
        }

    def _analyze_errors(self, y_true, y_pred_prob):
        """Analyze prediction errors in detail"""
        try:
            # Ensure y_pred_prob is the right shape (flatten if needed)
            y_pred_prob = y_pred_prob.ravel()
            
            # Calculate binary predictions
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            # Calculate error masks
            fp_mask = (y_pred == 1) & (y_true == 0)
            fn_mask = (y_pred == 0) & (y_true == 1)
            
            return {
                'false_positives': {
                    'count': int(np.sum(fp_mask)),
                    'mean_confidence': float(np.mean(y_pred_prob[fp_mask])) if np.any(fp_mask) else 0.0,
                    'indices': [int(i) for i, x in enumerate(fp_mask) if x]
                },
                'false_negatives': {
                    'count': int(np.sum(fn_mask)),
                    'mean_confidence': float(np.mean(1 - y_pred_prob[fn_mask])) if np.any(fn_mask) else 0.0,
                    'indices': [int(i) for i, x in enumerate(fn_mask) if x]
                },
                'error_rate': float(np.mean(y_pred != y_true)),
                'confidence_threshold_analysis': self._analyze_confidence_thresholds(y_true, y_pred_prob)
            }
        except Exception as e:
            print(f"Error in error analysis: {str(e)}")
            traceback.print_exc()
            return {
                'false_positives': {'count': 0, 'mean_confidence': 0.0, 'indices': []},
                'false_negatives': {'count': 0, 'mean_confidence': 0.0, 'indices': []},
                'error_rate': 0.0,
                'confidence_threshold_analysis': {}
            }

class ClinicalValidator:
    """Validate model performance for clinical use"""
    
    def __init__(self, safety_threshold=0.95):
        self.safety_threshold = safety_threshold
        
    def validate_model_safety(self, y_true, y_pred_prob, clinical_features=None):
        """Perform comprehensive clinical validation"""
        validation_results = {
            'safety_metrics': self._assess_safety_metrics(y_true, y_pred_prob),
            'subgroup_analysis': self._perform_subgroup_analysis(y_true, y_pred_prob, clinical_features),
            'error_analysis': self._analyze_errors(y_true, y_pred_prob),
            'validation_status': None
        }
        
        # Determine overall validation status
        validation_results['validation_status'] = self._determine_validation_status(validation_results)
        
        return validation_results
    
    def _assess_safety_metrics(self, y_true, y_pred_prob):
        """Assess safety-critical metrics"""
        # Calculate false negative rate for high-risk cases
        high_risk_mask = y_true == 1
        fn_rate = np.mean((y_pred_prob < self.safety_threshold) & high_risk_mask)
        
        # Calculate risk of missed diagnoses
        missed_diagnosis_risk = np.mean(y_pred_prob[y_true == 1] < 0.5)
        
        return {
            'false_negative_rate': float(fn_rate),
            'missed_diagnosis_risk': float(missed_diagnosis_risk),
            'safety_threshold_compliance': float(np.mean(y_pred_prob >= self.safety_threshold))
        }
    
    def _perform_subgroup_analysis(self, y_true, y_pred_prob, clinical_features):
        """Analyze performance across different patient subgroups"""
        if clinical_features is None:
            return None
            
        subgroup_results = {}
        
        # Analyze by age groups if available
        if 'age' in clinical_features:
            age_groups = pd.qcut(clinical_features['age'], q=4)
            for group in age_groups.unique():
                mask = age_groups == group
                subgroup_results[f'age_group_{group}'] = {
                    'accuracy': float(np.mean((y_pred_prob[mask] >= 0.5) == y_true[mask])),
                    'sample_size': int(np.sum(mask))
                }
        
        # Add more subgroup analyses based on available clinical features
        
        return subgroup_results
    
    def _analyze_errors(self, y_true, y_pred_prob):
        """Analyze prediction errors in detail"""
        try:
            # Ensure y_pred_prob is the right shape (flatten if needed)
            y_pred_prob = y_pred_prob.ravel()
            y_true = y_true.ravel()
            
            # Calculate binary predictions
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            # Calculate error masks
            fp_mask = (y_pred == 1) & (y_true == 0)
            fn_mask = (y_pred == 0) & (y_true == 1)
            
            # Calculate error metrics
            error_metrics = {
                'false_positives': {
                    'count': int(np.sum(fp_mask)),
                    'mean_confidence': float(np.mean(y_pred_prob[fp_mask])) if np.any(fp_mask) else 0.0,
                    'indices': [int(i) for i, x in enumerate(fp_mask) if x]
                },
                'false_negatives': {
                    'count': int(np.sum(fn_mask)),
                    'mean_confidence': float(np.mean(1 - y_pred_prob[fn_mask])) if np.any(fn_mask) else 0.0,
                    'indices': [int(i) for i, x in enumerate(fn_mask) if x]
                },
                'error_rate': float(np.mean(y_pred != y_true))
            }
            
            # Add confidence threshold analysis
            thresholds = np.linspace(0.1, 0.9, 9)
            threshold_metrics = {}
            for threshold in thresholds:
                y_pred_t = (y_pred_prob > threshold).astype(int)
                fp = np.sum((y_pred_t == 1) & (y_true == 0))
                fn = np.sum((y_pred_t == 0) & (y_true == 1))
                threshold_metrics[f'threshold_{threshold:.1f}'] = {
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'total_errors': int(fp + fn)
                }
            
            error_metrics['threshold_analysis'] = threshold_metrics
            
            return error_metrics
            
        except Exception as e:
            print(f"Error in clinical error analysis: {str(e)}")
            traceback.print_exc()
            return {
                'false_positives': {'count': 0, 'mean_confidence': 0.0, 'indices': []},
                'false_negatives': {'count': 0, 'mean_confidence': 0.0, 'indices': []},
                'error_rate': 0.0,
                'threshold_analysis': {}
            }
    
    def _determine_validation_status(self, validation_results):
        """Determine if the model meets clinical validation criteria"""
        safety_metrics = validation_results['safety_metrics']
        
        # Define validation criteria
        criteria = {
            'false_negative_rate': safety_metrics['false_negative_rate'] < 0.05,
            'missed_diagnosis_risk': safety_metrics['missed_diagnosis_risk'] < 0.01,
            'safety_compliance': safety_metrics['safety_threshold_compliance'] > 0.95
        }
        
        # Check if all criteria are met
        validation_passed = all(criteria.values())
        
        return {
            'status': 'PASSED' if validation_passed else 'FAILED',
            'criteria_results': criteria
        }

def save_clinical_validation(validation_results, save_path):
    """Save clinical validation results"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(validation_results, f, indent=4) 