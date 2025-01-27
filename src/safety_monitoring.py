import numpy as np
from datetime import datetime
import json
import os
import logging
from collections import deque
import traceback
from scipy import stats

class SafetyMonitor:
    """Monitor model safety and trigger alerts"""
    
    def __init__(self, safety_thresholds=None):
        self.safety_thresholds = safety_thresholds or {
            'false_negative_rate': 0.05,
            'uncertainty_threshold': 0.8,
            'confidence_threshold': 0.95
        }
        self.alert_history = deque(maxlen=1000)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for safety monitoring"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('safety_monitoring.log'),
                logging.StreamHandler()
            ]
        )
        
    def monitor_prediction(self, prediction, uncertainty, clinical_features=None):
        """Monitor a single prediction for safety concerns"""
        try:
            alerts = []
            
            # Check prediction confidence
            if uncertainty > self.safety_thresholds['uncertainty_threshold']:
                alerts.append({
                    'level': 'WARNING',
                    'type': 'HIGH_UNCERTAINTY',
                    'message': f'High uncertainty in prediction: {uncertainty:.3f}'
                })
                
            # Check for high-risk cases
            if prediction > 0.7 and uncertainty > 0.2:
                alerts.append({
                    'level': 'CRITICAL',
                    'type': 'HIGH_RISK_UNCERTAIN',
                    'message': 'High risk prediction with significant uncertainty'
                })
                
            # Additional clinical checks
            if clinical_features is not None and not isinstance(clinical_features, (float, int)):
                clinical_alerts = self._check_clinical_features(
                    prediction, clinical_features
                )
                alerts.extend(clinical_alerts)
                
            # Log and store alerts
            self._handle_alerts(alerts)
            return alerts
            
        except Exception as e:
            print(f"Error in prediction monitoring: {str(e)}")
            traceback.print_exc()
            return []
    
    def monitor_batch_predictions(self, predictions, uncertainties, clinical_features=None):
        """Monitor a batch of predictions"""
        try:
            # Convert single values to arrays if needed
            if isinstance(predictions, (float, int)):
                predictions = np.array([predictions])
            if isinstance(uncertainties, (float, int)):
                uncertainties = np.array([uncertainties])
            
            # Ensure inputs are numpy arrays
            predictions = np.asarray(predictions).flatten()
            uncertainties = np.asarray(uncertainties).flatten()
            
            batch_alerts = []
            
            # Monitor each prediction
            for i, (pred, unc) in enumerate(zip(predictions, uncertainties)):
                # Handle clinical features properly
                features = None
                if clinical_features is not None:
                    if hasattr(clinical_features, 'iloc'):
                        features = clinical_features.iloc[i] if i < len(clinical_features) else None
                    elif isinstance(clinical_features, (list, np.ndarray)):
                        features = clinical_features[i] if i < len(clinical_features) else None
                
                alerts = self.monitor_prediction(pred, unc, features)
                if alerts:
                    batch_alerts.append({
                        'index': i,
                        'alerts': alerts
                    })
            
            # Check batch-level statistics
            batch_stats = self._calculate_batch_statistics(predictions, uncertainties)
            if batch_stats['high_uncertainty_rate'] > 0.1:  # More than 10% uncertain
                batch_alerts.append({
                    'level': 'WARNING',
                    'type': 'BATCH_UNCERTAINTY',
                    'message': f'High uncertainty rate in batch: {batch_stats["high_uncertainty_rate"]:.3f}'
                })
            
            return batch_alerts, batch_stats
        
        except Exception as e:
            print(f"Error in batch prediction monitoring: {str(e)}")
            traceback.print_exc()
            return [], {
                'mean_prediction': 0.0,
                'mean_uncertainty': 0.0,
                'high_uncertainty_rate': 0.0,
                'high_risk_rate': 0.0
            }
    
    def _check_clinical_features(self, prediction, features):
        """Check clinical features for safety concerns"""
        try:
            alerts = []
            
            # Convert features to dictionary if it's a pandas Series or numpy array
            if hasattr(features, 'to_dict'):
                features = features.to_dict()
            elif isinstance(features, np.ndarray):
                features = {i: v for i, v in enumerate(features)}
            
            # Example clinical checks (modify based on your feature names)
            for key, value in features.items():
                if isinstance(value, (int, float)) and value > 3:  # Example threshold
                    alerts.append({
                        'level': 'WARNING',
                        'type': f'HIGH_VALUE_{key}',
                        'message': f'High value detected for feature {key}: {value}'
                    })
            
            return alerts
            
        except Exception as e:
            print(f"Error in clinical feature check: {str(e)}")
            traceback.print_exc()
            return []
    
    def _calculate_batch_statistics(self, predictions, uncertainties):
        """Calculate safety statistics for a batch"""
        return {
            'mean_prediction': float(np.mean(predictions)),
            'mean_uncertainty': float(np.mean(uncertainties)),
            'high_uncertainty_rate': float(np.mean(uncertainties > self.safety_thresholds['uncertainty_threshold'])),
            'high_risk_rate': float(np.mean(predictions > 0.7))
        }
    
    def _handle_alerts(self, alerts):
        """Handle and log safety alerts"""
        for alert in alerts:
            self.alert_history.append({
                'timestamp': datetime.now().isoformat(),
                **alert
            })
            
            if alert['level'] == 'CRITICAL':
                logging.critical(alert['message'])
            elif alert['level'] == 'WARNING':
                logging.warning(alert['message'])
                
    def get_alert_summary(self):
        """Get summary of recent alerts"""
        return {
            'total_alerts': len(self.alert_history),
            'critical_alerts': sum(1 for a in self.alert_history if a['level'] == 'CRITICAL'),
            'warning_alerts': sum(1 for a in self.alert_history if a['level'] == 'WARNING'),
            'recent_alerts': list(self.alert_history)[-10:]  # Last 10 alerts
        }

    def _analyze_prediction_trends(self, predictions_history):
        """Analyze trends in predictions over time"""
        predictions_array = np.array(predictions_history)
        
        return {
            'mean_trend': float(np.mean(predictions_array)),
            'std_trend': float(np.std(predictions_array)),
            'trend_direction': 'increasing' if len(predictions_array) > 1 and 
                              predictions_array[-1] > predictions_array[0] else 'decreasing',
            'volatility': float(np.std(np.diff(predictions_array))) if len(predictions_array) > 1 else 0.0
        }

    def _assess_prediction_stability(self, predictions_history):
        """Assess stability of predictions"""
        if len(predictions_history) < 2:
            return {'stability_score': 1.0}
        
        differences = np.diff(predictions_history)
        stability_score = 1.0 / (1.0 + np.std(differences))
        
        return {
            'stability_score': float(stability_score),
            'max_change': float(np.max(np.abs(differences))),
            'change_frequency': float(np.mean(np.abs(differences) > 0.1))
        }

    def _detect_model_drift(self, predictions_history):
        """Detect potential model drift"""
        if len(predictions_history) < 10:
            return {'drift_detected': False}
        
        # Split history into two halves
        mid_point = len(predictions_history) // 2
        first_half = predictions_history[:mid_point]
        second_half = predictions_history[mid_point:]
        
        # Compare distributions
        stat, p_value = stats.ks_2samp(first_half, second_half)
        
        return {
            'drift_detected': p_value < 0.05,
            'drift_statistic': float(stat),
            'p_value': float(p_value)
        }

    def _check_high_risk_conditions(self, prediction, patient_data):
        """Check for high-risk conditions"""
        risk_factors = []
        
        # Check prediction threshold
        if prediction > 0.7:
            risk_factors.append('high_prediction_score')
        
        # Check clinical factors if available
        if patient_data is not None:
            if 'age' in patient_data and patient_data['age'] > 50:
                risk_factors.append('age_risk')
            if 'family_history' in patient_data and patient_data['family_history']:
                risk_factors.append('family_history_risk')
            
        return len(risk_factors) > 0, risk_factors

    def _generate_critical_alert(self, risk_factors=None):
        """Generate a critical alert with detailed information"""
        return {
            'level': 'CRITICAL',
            'timestamp': datetime.now().isoformat(),
            'risk_factors': risk_factors or [],
            'recommended_actions': [
                'Immediate clinical review required',
                'Schedule follow-up examination',
                'Review patient history'
            ]
        } 