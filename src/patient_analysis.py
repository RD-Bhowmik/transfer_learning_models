import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import os

class PatientAnalyzer:
    def __init__(self, model, metadata_features, clinical_features):
        self.model = model
        self.metadata_features = metadata_features
        self.clinical_features = clinical_features
        
    def generate_patient_profile(self, patient_idx, metadata, predictions):
        """Generate a profile for a single patient"""
        try:
            patient_metadata = metadata[patient_idx]
            feature_means = np.nanmean(metadata, axis=0)
            feature_stds = np.nanstd(metadata, axis=0)
            
            # Avoid division by zero and handle NaN values
            feature_scores = []
            for i in range(len(patient_metadata)):
                if feature_stds[i] == 0 or np.isnan(feature_stds[i]):
                    z_score = 0.0
                else:
                    z_score = (patient_metadata[i] - feature_means[i]) / feature_stds[i]
                feature_scores.append(float(z_score))
            
            return {
                'patient_id': int(patient_idx),
                'prediction': float(predictions[patient_idx]),
                'feature_scores': dict(zip(self.clinical_features, feature_scores)),
                'anomalous_features': [
                    self.clinical_features[i] for i, score in enumerate(feature_scores)
                    if abs(score) > 2.0  # More than 2 standard deviations from mean
                ]
            }
        except Exception as e:
            print(f"Error generating patient profile: {str(e)}")
            return None
    
    def _identify_risk_factors(self, patient_metadata):
        """Identify significant risk factors for the patient"""
        risk_factors = []
        feature_means = np.mean(self.metadata_features, axis=0)
        feature_stds = np.std(self.metadata_features, axis=0)
        
        for i, feature in enumerate(self.clinical_features):
            z_score = (patient_metadata[i] - feature_means[i]) / feature_stds[i]
            if abs(z_score) > 1.5:  # Significant deviation
                risk_factors.append({
                    'feature': feature,
                    'value': float(patient_metadata[i]),
                    'z_score': float(z_score)
                })
                
        return risk_factors
    
    def _find_similar_cases(self, metadata, patient_idx, n_similar=5):
        """Find similar cases based on clinical features"""
        similarities = cosine_similarity([metadata[patient_idx]], metadata)[0]
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        
        similar_cases = []
        for idx in similar_indices:
            similar_cases.append({
                'index': int(idx),
                'similarity_score': float(similarities[idx]),
                'features': {f: float(metadata[idx, i]) 
                           for i, f in enumerate(self.clinical_features)}
            })
            
        return similar_cases 

    def analyze_risk_factors(self, patient_data):
        """Analyze patient-specific risk factors"""
        risk_factors = {
            'genetic_factors': self._analyze_genetic_risk(patient_data),
            'environmental_factors': self._analyze_environmental_risk(patient_data),
            'medical_history': self._analyze_medical_history(patient_data),
            'lifestyle_factors': self._analyze_lifestyle_factors(patient_data)
        }
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(risk_factors)
        return {**risk_factors, 'overall_risk_score': risk_score}
    
    def _analyze_genetic_risk(self, patient_data):
        """Analyze genetic risk factors"""
        genetic_risk = 0.0
        risk_factors = []
        
        if 'family_history' in patient_data:
            if patient_data['family_history']:
                genetic_risk += 0.3
                risk_factors.append('family_history_positive')
                
        return {
            'risk_level': genetic_risk,
            'identified_factors': risk_factors
        }
    
    def _analyze_environmental_risk(self, patient_data):
        """Analyze environmental risk factors"""
        env_risk = 0.0
        risk_factors = []
        
        # Add environmental risk analysis
        if 'smoking' in patient_data and patient_data['smoking']:
            env_risk += 0.2
            risk_factors.append('smoking')
            
        return {
            'risk_level': env_risk,
            'identified_factors': risk_factors
        }
    
    def generate_longitudinal_analysis(self, patient_history):
        """Generate longitudinal analysis of patient data"""
        try:
            # Sort history by date
            sorted_history = sorted(patient_history, key=lambda x: x['date'])
            
            # Analyze disease progression
            progression = self._analyze_disease_progression(sorted_history)
            
            # Analyze treatment effectiveness
            treatment_response = self._analyze_treatment_response(sorted_history)
            
            # Predict risk trajectory
            risk_trajectory = self._predict_risk_trajectory(sorted_history)
            
            return {
                'progression_analysis': progression,
                'treatment_response': treatment_response,
                'risk_trajectory': risk_trajectory,
                'recommendations': self._generate_recommendations(
                    progression, treatment_response, risk_trajectory
                )
            }
            
        except Exception as e:
            print(f"Error in longitudinal analysis: {str(e)}")
            return None
    
    def _analyze_disease_progression(self, history):
        """Analyze disease progression over time"""
        progression_rates = []
        for i in range(1, len(history)):
            current = history[i]
            previous = history[i-1]
            
            # Calculate progression rate
            time_diff = (current['date'] - previous['date']).days
            severity_diff = current['severity'] - previous['severity']
            progression_rates.append(severity_diff / time_diff if time_diff > 0 else 0)
        
        return {
            'avg_progression_rate': float(np.mean(progression_rates)) if progression_rates else 0,
            'progression_trend': 'increasing' if np.mean(progression_rates) > 0 else 'stable',
            'progression_rates': progression_rates
        } 