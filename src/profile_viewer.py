import json
import os
import pandas as pd
import numpy as np

def display_patient_profile(profile_path):
    """Display patient profile in a readable format"""
    try:
        with open(profile_path, 'r') as f:
            profile = json.load(f)
            
        print("\nPatient Profile Analysis")
        print("=" * 50)
        print(f"Patient ID: {profile['patient_id']}")
        print(f"Risk Prediction: {profile['prediction']:.3f}")
        
        print("\nFeature Analysis:")
        print("-" * 30)
        
        # Convert feature scores to DataFrame for better formatting
        scores_df = pd.DataFrame([
            {
                'Feature': feature,
                'Z-Score': score,
                'Status': '🚨 Anomalous' if abs(score) > 2.0 else 'Normal'
            }
            for feature, score in profile['feature_scores'].items()
        ])
        
        # Sort by absolute z-score
        scores_df['Abs_Score'] = abs(scores_df['Z-Score'])
        scores_df = scores_df.sort_values('Abs_Score', ascending=False)
        
        # Print feature scores
        for _, row in scores_df.iterrows():
            print(f"{row['Feature']:<35} {row['Z-Score']:>6.2f}  {row['Status']}")
            
        print("\nAnomalous Features:")
        print("-" * 30)
        if profile['anomalous_features']:
            for feature in profile['anomalous_features']:
                score = profile['feature_scores'][feature]
                print(f"• {feature:<35} (z-score: {score:>6.2f})")
        else:
            print("No anomalous features detected")
            
    except Exception as e:
        print(f"Error reading profile: {str(e)}")

if __name__ == "__main__":
    # Example usage
    experiment_folder = "path/to/experiment/folder"
    profile_path = os.path.join(experiment_folder, "visualizations", "patient_profile_11.json")
    display_patient_profile(profile_path) 