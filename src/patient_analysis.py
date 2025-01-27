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