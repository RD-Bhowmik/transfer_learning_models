import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, auc
from statsmodels.stats.outliers_influence import variance_inflation_factor
import json
import os
import traceback

def perform_statistical_tests(metadata, clinical_features):
    """Perform statistical analysis on metadata"""
    try:
        results = {}
        
        # 1. Chi-square tests for categorical variables
        chi_square_tests = {}
        for feature in clinical_features:
            if feature in metadata.columns and feature != 'HPV':
                try:
                    # Convert to categorical if needed
                    feature_data = pd.Categorical(metadata[feature])
                    target_data = pd.Categorical(metadata['HPV'])
                    
                    # Create contingency table
                    contingency = pd.crosstab(feature_data, target_data)
                    
                    # Only perform test if we have valid data
                    if contingency.size > 0 and not contingency.empty:
                        chi2, p_value = stats.chi2_contingency(contingency)[:2]
                        chi_square_tests[feature] = {
                            'chi2': float(chi2),
                            'p_value': float(p_value)
                        }
                except Exception as e:
                    print(f"Warning: Could not perform chi-square test for {feature}: {str(e)}")
                    continue
        
        # 2. Correlation analysis
        numeric_features = metadata[clinical_features].apply(pd.to_numeric, errors='coerce')
        # Fill NaN values with mean before correlation
        numeric_features = numeric_features.fillna(numeric_features.mean())
        correlations = numeric_features.corr().to_dict()
        
        # 3. VIF analysis
        try:
            # Remove any remaining NaN or inf values
            clean_features = numeric_features.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
            
            if len(clean_features.columns) >= 2:  # Need at least 2 features
                vif_data = pd.DataFrame()
                vif_data["Feature"] = clean_features.columns
                vif_data["VIF"] = [variance_inflation_factor(clean_features.values, i)
                                 for i in range(clean_features.shape[1])]
                vif_factors = vif_data.set_index("Feature")["VIF"].to_dict()
                vif_factors = {k: float(v) for k, v in vif_factors.items() if not np.isnan(v)}
            else:
                vif_factors = {}
        except Exception as e:
            print(f"Warning: Could not compute VIF: {str(e)}")
            vif_factors = {}
        
        return {
            'chi_square_tests': chi_square_tests,
            'correlations': correlations,
            'vif_factors': vif_factors
        }
        
    except Exception as e:
        print(f"Error in statistical analysis: {str(e)}")
        traceback.print_exc()
        return {
            'chi_square_tests': {},
            'correlations': {},
            'vif_factors': {}
        }

def calculate_risk_ratio(data, feature):
    """Calculate risk ratio for a feature relative to HPV status"""
    contingency = pd.crosstab(data[feature], data['HPV'])
    if contingency.shape[1] != 2 or contingency.shape[0] < 2:
        return None
    
    risk_ratio = (contingency[1] / contingency.sum(axis=1)) / \
                 (contingency[0] / contingency.sum(axis=1))
    return risk_ratio.to_dict()

def calculate_confidence_intervals(y_pred_prob, confidence_level=0.95):
    """Calculate confidence intervals for predictions"""
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z_score * np.sqrt((y_pred_prob * (1 - y_pred_prob)) / len(y_pred_prob))
    
    return {
        'lower_bound': y_pred_prob - margin_of_error,
        'upper_bound': y_pred_prob + margin_of_error,
        'confidence_level': confidence_level
    }

def save_statistical_analysis(results, viz_folder):
    """Save statistical analysis results"""
    stats_folder = os.path.join(viz_folder, 'statistical_analysis')
    os.makedirs(stats_folder, exist_ok=True)
    
    with open(os.path.join(stats_folder, 'statistical_results.json'), 'w') as f:
        json.dump(results, f, indent=4) 