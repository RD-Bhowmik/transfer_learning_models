import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os
from scipy import stats
import traceback

def visualize_statistical_results(statistical_results, viz_folder):
    """Visualize statistical analysis results"""
    try:
        # Create visualization folder
        os.makedirs(os.path.join(viz_folder, 'statistical_analysis'), exist_ok=True)
        
        # Chi-square test visualization
        if statistical_results.get('chi_square_tests'):
            chi_square_data = pd.DataFrame([
                {
                    'feature': feature,
                    'chi2': result['chi2'],
                    'p_value': result['p_value']
                }
                for feature, result in statistical_results['chi_square_tests'].items()
            ])
            
            if not chi_square_data.empty:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=chi_square_data, x='chi2', y='p_value')
                plt.axhline(y=0.05, color='r', linestyle='--', label='p=0.05')
                plt.title('Chi-Square Test Results')
                plt.xlabel('Chi-Square Statistic')
                plt.ylabel('P-Value')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(viz_folder, 'statistical_analysis', 'chi_square_results.png'))
                plt.close()
        
        # Correlation visualization
        if statistical_results.get('correlations'):
            corr_data = pd.DataFrame(statistical_results['correlations'])
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlations')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_folder, 'statistical_analysis', 'correlations.png'))
            plt.close()
        
        # VIF factors visualization
        if statistical_results.get('vif_factors'):
            vif_data = pd.DataFrame([
                {'Feature': k, 'VIF': v}
                for k, v in statistical_results['vif_factors'].items()
            ])
            if not vif_data.empty:
                plt.figure(figsize=(10, 6))
                sns.barplot(data=vif_data, x='VIF', y='Feature')
                plt.title('Variance Inflation Factors')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_folder, 'statistical_analysis', 'vif_factors.png'))
                plt.close()
                
    except Exception as e:
        print(f"Error in statistical visualization: {str(e)}")
        traceback.print_exc()

def visualize_model_interpretations(interpretations_folder, viz_folder):
    """Visualize model interpretation results"""
    # Load all interpretation files
    interpretation_files = [f for f in os.listdir(interpretations_folder) 
                          if f.startswith('interpretation_sample_')]
    
    all_importances = []
    all_uncertainties = []
    
    for file in interpretation_files:
        with open(os.path.join(interpretations_folder, file), 'r') as f:
            data = json.load(f)
            all_importances.append(data['feature_importance'])
            all_uncertainties.append(data['uncertainty'])
    
    fig = plt.figure(figsize=(20, 15))
    
    # Average feature importance
    plt.subplot(2, 2, 1)
    avg_importance = pd.DataFrame(all_importances).mean()
    plt.barh(range(len(avg_importance)), avg_importance.values)
    plt.yticks(range(len(avg_importance)), avg_importance.index)
    plt.xlabel('Average Importance')
    plt.title('Feature Importance Across Samples')
    
    # Uncertainty distribution
    plt.subplot(2, 2, 2)
    uncertainties = [u['uncertainty'] for u in all_uncertainties]
    plt.hist(uncertainties, bins=20)
    plt.xlabel('Prediction Uncertainty')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Uncertainties')
    
    # Confidence intervals
    plt.subplot(2, 2, 3)
    for i, u in enumerate(all_uncertainties):
        plt.errorbar(u['mean_prediction'], i, 
                    xerr=[[u['mean_prediction'] - u['confidence_interval'][0]], 
                          [u['confidence_interval'][1] - u['mean_prediction']]], 
                    fmt='o')
    plt.xlabel('Prediction')
    plt.ylabel('Sample Index')
    plt.title('Predictions with Confidence Intervals')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, 'model_interpretation', 'interpretation_summary.png'))
    plt.close()

def visualize_patient_profiles(profiles_folder, viz_folder):
    """Visualize aggregated patient profile information"""
    try:
        # Create visualization folder
        os.makedirs(os.path.join(viz_folder, 'patient_analysis'), exist_ok=True)
        
        # Load all profile files
        profile_files = [f for f in os.listdir(profiles_folder) 
                        if f.startswith('patient_profile_')]
        
        all_profiles = []
        for file in profile_files:
            with open(os.path.join(profiles_folder, file), 'r') as f:
                all_profiles.append(json.load(f))
        
        if not all_profiles:
            print("No patient profiles found to visualize")
            return
            
        # Collect data for visualization
        predictions = []
        anomalous_features = []
        feature_scores = {}
        
        for profile in all_profiles:
            if profile and isinstance(profile, dict):
                # Collect predictions
                if 'prediction' in profile:
                    predictions.append(profile['prediction'])
                
                # Collect anomalous features
                if 'anomalous_features' in profile:
                    anomalous_features.extend(profile['anomalous_features'])
                
                # Collect feature scores
                if 'feature_scores' in profile:
                    for feature, score in profile['feature_scores'].items():
                        if feature not in feature_scores:
                            feature_scores[feature] = []
                        feature_scores[feature].append(score)
        
        # Create visualizations
        plt.figure(figsize=(20, 15))
        
        # Prediction distribution
        plt.subplot(2, 2, 1)
        if predictions:
            plt.hist(predictions, bins=20)
            plt.xlabel('Prediction Score')
            plt.ylabel('Count')
            plt.title('Distribution of Predictions')
        
        # Most common anomalous features
        plt.subplot(2, 2, 2)
        if anomalous_features:
            feature_counts = pd.Series(anomalous_features).value_counts()
            plt.barh(range(len(feature_counts)), feature_counts.values)
            plt.yticks(range(len(feature_counts)), feature_counts.index)
            plt.xlabel('Count')
            plt.title('Most Common Anomalous Features')
        
        # Feature score distributions
        plt.subplot(2, 2, 3)
        if feature_scores:
            plt.boxplot(feature_scores.values(), labels=feature_scores.keys())
            plt.xticks(rotation=45)
            plt.ylabel('Z-Score')
            plt.title('Feature Score Distributions')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_folder, 'patient_analysis', 'profile_summary.png'))
        plt.close()
        
        print("Patient profile visualizations saved successfully")
        
    except Exception as e:
        print(f"Error in patient profile visualization: {str(e)}")
        traceback.print_exc()

def visualize_combined_analysis(viz_folder):
    """Create a comprehensive visualization combining all analyses"""
    try:
        # Load all results
        stats_path = os.path.join(viz_folder, 'statistical_analysis', 'statistical_results.json')
        if not os.path.exists(stats_path):
            print("Statistical results not found")
            return
            
        with open(stats_path, 'r') as f:
            stats_results = json.load(f)
        
        # Get interpretation files
        interpretation_files = [f for f in os.listdir(viz_folder) 
                              if f.startswith('interpretation_sample_')]
        interpretations = []
        for file in interpretation_files:
            with open(os.path.join(viz_folder, file), 'r') as f:
                interpretations.append(json.load(f))
        
        # Create comprehensive visualization
        plt.figure(figsize=(20, 15))
        
        # Statistical significance vs Model importance
        if stats_results.get('chi_square_tests') and interpretations:
            plt.subplot(2, 2, 1)
            
            # Get common features between statistical tests and model importance
            common_features = set(stats_results['chi_square_tests'].keys()) & \
                            set(interpretations[0]['feature_importance'].keys())
            
            if common_features:
                feature_significance = [stats_results['chi_square_tests'][f]['p_value'] 
                                     for f in common_features]
                feature_importance = [np.mean([interp['feature_importance'][f] 
                                    for interp in interpretations]) 
                                    for f in common_features]
                
                plt.scatter(feature_significance, feature_importance)
                plt.xlabel('Statistical Significance (p-value)')
                plt.ylabel('Model Feature Importance')
                plt.title('Statistical vs Model-based Feature Importance')
                
                # Add feature labels
                for i, feature in enumerate(common_features):
                    plt.annotate(feature, (feature_significance[i], feature_importance[i]))
        
        # Correlation heatmap
        if stats_results.get('correlations'):
            plt.subplot(2, 2, 2)
            corr_data = pd.DataFrame(stats_results['correlations'])
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlations')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_folder, 'combined_analysis.png'))
        plt.close()
        
        print("Combined analysis visualization saved successfully")
        
    except Exception as e:
        print(f"Error in combined analysis visualization: {str(e)}")
        traceback.print_exc() 