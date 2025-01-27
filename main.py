import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import traceback
import json
import datetime

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import SAVE_FOLDER
from src.data_processing import (
    load_and_preprocess_data, align_data, preprocess_metadata,
    prepare_data_splits, setup_data_augmentation
)
from src.model import (
    create_model, setup_callbacks, train_model, evaluate_model,
    make_predictions, save_intermediate_model, load_previous_best_model
)
from src.visualization import (
    visualize_results, visualize_metadata, visualize_model_performance,
    visualize_learning_dynamics, visualize_metadata_distributions,
    visualize_weight_updates, visualize_hyperparameter_effects,
    visualize_data_augmentation, visualize_clinical_patterns,
    visualize_model_interpretation
)
from src.metrics import (
    track_hyperparameter_performance, save_detailed_metrics,
    generate_performance_report, analyze_feature_importance
)
from src.utils import (
    setup_visualization_folder, save_json, create_experiment_folder,
    log_training_info
)
from src.statistical_analysis import perform_statistical_tests, save_statistical_analysis
from src.model_interpretation import (
    ModelInterpreter, 
    safe_compare_model_weights
)
from src.patient_analysis import PatientAnalyzer
from src.advanced_visualization import (
    visualize_statistical_results, visualize_model_interpretations,
    visualize_patient_profiles, visualize_combined_analysis
)
from src.medical_preprocessing import ColposcopyPreprocessor, LesionAnalyzer
from src.clinical_metrics import ClinicalMetrics, ClinicalValidator
from src.clinical_reporting import ClinicalReport
from src.safety_monitoring import SafetyMonitor

def setup_hyperparameters():
    """Define hyperparameter combinations for training"""
    return {
        'learning_rates': [1e-3],
        'batch_sizes': [16],
        'dropout_rates': [0.2]
    }

def train_and_evaluate():
    """Main training and evaluation pipeline"""
    try:
        # Define clinical features
        clinical_features = [
            'Adequacy',
            'Squamocolumnar junction visibility',
            'Transformation zone',
            'Original squamous epithelium',
            'Columnar epithelium',
            'Metaplastic squamous epithelium',
            'Location of the lesion',
            'Grade 1',
            'Grade 2',
            'Suspicious for invasion',
            'Aceto uptake',
            'Margins',
            'Vessels',
            'Lesion size',
            'Iodine uptake',
            'SwedeFinal'
        ]
        
        # Initialize components
        preprocessor = ColposcopyPreprocessor()
        lesion_analyzer = LesionAnalyzer()
        clinical_metrics = ClinicalMetrics()
        clinical_validator = ClinicalValidator()
        safety_monitor = SafetyMonitor()
        
        # Create experiment folder
        experiment_folder = create_experiment_folder(SAVE_FOLDER)
        print(f"Created experiment folder: {experiment_folder}")
        
        # Setup visualization folder
        viz_folder = setup_visualization_folder(experiment_folder)
        clinical_report = ClinicalReport(viz_folder)
        
        # Load and preprocess data with medical-specific preprocessing
        print("\nLoading and preprocessing data...")
        aligned_images, aligned_labels, aligned_metadata = load_and_preprocess_data()
        
        # Apply medical preprocessing to images
        processed_images = []
        lesion_features = []
        for image in aligned_images:
            # Medical-specific preprocessing
            processed_image, roi_mask = preprocessor.preprocess_image(image)
            processed_images.append(processed_image)
            
            # Detect and analyze lesions
            lesions = preprocessor.detect_lesions(processed_image)
            if lesions:
                lesion_features.append(lesion_analyzer.analyze_lesion(processed_image, lesions[0]))
            else:
                lesion_features.append(None)
        
        # Add lesion features to metadata
        lesion_df = pd.DataFrame(lesion_features)
        aligned_metadata = pd.concat([aligned_metadata, lesion_df], axis=1)
        
        # Continue with existing pipeline...
        visualize_metadata(aligned_metadata, viz_folder)
        previous_model, previous_metrics = load_previous_best_model(viz_folder)
        
        metadata_features_processed, metadata_labels = preprocess_metadata(aligned_metadata)
        train_data, val_data, test_data = prepare_data_splits(
            processed_images, aligned_labels, metadata_features_processed
        )
        
        # Unpack the data splits
        (X_train, meta_X_train, y_train) = train_data
        (X_val, meta_X_val, y_val) = val_data
        (X_test, meta_X_test, y_test) = test_data
        
        # Training setup and execution
        hyperparameters = setup_hyperparameters()
        callbacks = setup_callbacks(experiment_folder)
        
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = dict(enumerate(class_weights))
        
        # Initialize tracking
        tracking_dict = track_hyperparameter_performance()
        best_val_accuracy = 0 if previous_metrics is None else previous_metrics['metrics']['val_accuracy']
        best_model = None
        best_history = None
        best_params = None
        
        # Add iteration counter
        iteration = 0
        total_iterations = (len(hyperparameters['learning_rates']) * 
                          len(hyperparameters['batch_sizes']) * 
                          len(hyperparameters['dropout_rates']))
        
        # Training loop
        print("\nStarting training loop...")
        for lr in hyperparameters['learning_rates']:
            for batch_size in hyperparameters['batch_sizes']:
                for dropout_rate in hyperparameters['dropout_rates']:
                    iteration += 1
                    print(f"\nTraining iteration {iteration}/{total_iterations}")
                    print(f"Parameters: lr={lr}, batch_size={batch_size}, dropout_rate={dropout_rate}")
                    
                    # Create and compile model
                    model = create_model(
                        dropout_rate=dropout_rate, 
                        metadata_input_shape=meta_X_train.shape[1]
                    )

                    # Compile model with metrics
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                        loss="binary_crossentropy",
                        metrics=[
                            "accuracy",
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall'),
                            tf.keras.metrics.AUC(name='auc')
                        ]
                    )

                    # Load weights from previous best model if available
                    if previous_model is not None:
                        try:
                            model.set_weights(previous_model.get_weights())
                            print("Loaded weights from previous best model")
                        except:
                            print("Could not load weights from previous model, starting fresh")

                    # Train model
                    history = train_model(
                        model,
                        train_data,
                        val_data,
                        batch_size,
                        callbacks,
                        class_weights
                    )

                    # Update tracking
                    val_accuracy = max(history.history['val_accuracy'])
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        best_model = model
                        best_history = history
                        best_params = {
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'dropout_rate': dropout_rate
                        }

                    # Save intermediate results
                    current_metrics = {
                        'val_accuracy': val_accuracy,
                        'val_loss': min(history.history['val_loss'])
                    }
                    
                    save_intermediate_model(
                        model, 
                        current_metrics, 
                        best_params, 
                        iteration,
                        viz_folder
                    )

        # Evaluate best model
        print("\nEvaluating best model...")
        test_metrics = evaluate_model(best_model, test_data)
        
        # Compare with previous model if it exists
        if previous_model is not None:
            print("\nComparing with previous best model...")
            prev_test_metrics = evaluate_model(previous_model, test_data)
            improvement = {
                metric: test_metrics[metric] - prev_test_metrics[metric]
                for metric in test_metrics.keys()
            }
            print("\nImprovement over previous model:")
            for metric, value in improvement.items():
                print(f"{metric}: {value:+.4f}")
        
        # Save final results
        final_results = {
            'test_metrics': test_metrics,
            'best_hyperparameters': best_params,
            'training_history': {
                'accuracy': best_history.history['accuracy'],
                'val_accuracy': best_history.history['val_accuracy'],
                'loss': best_history.history['loss'],
                'val_loss': best_history.history['val_loss']
            }
        }
        
        save_json(final_results, os.path.join(experiment_folder, 'final_results.json'))
        
        # Save the best model with .keras extension
        model_save_path = os.path.join(experiment_folder, 'best_model.keras')
        best_model.save(model_save_path)
        print(f"Saved best model to: {model_save_path}")
        
        # Generate predictions
        y_pred_prob, y_pred = make_predictions(best_model, X_test, meta_X_test)
        
        # Save and visualize results
        save_detailed_metrics(y_test, y_pred, y_pred_prob, experiment_folder)
        visualize_results(best_history, viz_folder)
        visualize_model_performance(y_test, y_pred, y_pred_prob, viz_folder)
        visualize_learning_dynamics(best_history, viz_folder)

        # Add metadata visualizations
        visualize_metadata_distributions(aligned_metadata, viz_folder)
        
        # Add weight evolution visualization
        weight_changes = safe_compare_model_weights(previous_model, best_model, viz_folder)
        if weight_changes:
            visualize_weight_updates(weight_changes, viz_folder)
        
        # Add hyperparameter visualization
        visualize_hyperparameter_effects(tracking_dict, viz_folder)
        
        # Add augmentation visualization
        sample_image = X_train[0]
        visualize_data_augmentation(sample_image, viz_folder)

        # Add clinical analysis
        visualize_clinical_patterns(aligned_metadata, viz_folder)
        
        # Analyze feature importance
        importance_dict = analyze_feature_importance(
            best_model, X_test, meta_X_test, clinical_features, viz_folder
        )
        
        # Model interpretation
        sample_idx = np.random.randint(len(X_test))
        visualize_model_interpretation(
            best_model,
            X_test[sample_idx],
            meta_X_test[sample_idx],
            clinical_features,
            viz_folder
        )

        # Generate final report
        performance_report = generate_performance_report(tracking_dict, viz_folder)
        save_json(performance_report, os.path.join(experiment_folder, 'performance_report.json'))

        # Statistical Analysis
        print("\nPerforming statistical analysis...")
        statistical_results = perform_statistical_tests(aligned_metadata, clinical_features)
        save_statistical_analysis(statistical_results, viz_folder)
        
        # Model Interpretation
        print("\nGenerating model interpretations...")
        interpreter = ModelInterpreter(best_model)
        
        # Analyze sample cases
        n_samples = 5
        sample_indices = np.random.choice(len(X_test), n_samples)
        for idx in sample_indices:
            # Generate GradCAM
            image = np.expand_dims(X_test[idx], 0)
            heatmap = interpreter.generate_gradcam(image)
            
            # Feature importance
            importance = interpreter.analyze_feature_importance(
                image, 
                np.expand_dims(meta_X_test[idx], 0),
                clinical_features
            )
            
            # Uncertainty estimation
            uncertainty = interpreter.generate_uncertainty(
                image,
                np.expand_dims(meta_X_test[idx], 0)
            )
            
            # Save interpretations
            interpretation_results = {
                'sample_id': int(idx),
                'feature_importance': importance,
                'uncertainty': uncertainty
            }
            save_json(interpretation_results, 
                     os.path.join(viz_folder, f'interpretation_sample_{idx}.json'))
        
        # Patient Analysis
        print("\nGenerating patient profiles...")
        analyzer = PatientAnalyzer(
            best_model,
            meta_X_test,
            clinical_features
        )
        
        # Generate profiles for test cases
        for idx in range(len(X_test)):
            profile = analyzer.generate_patient_profile(
                idx,
                meta_X_test,
                y_pred_prob
            )
            save_json(profile, 
                     os.path.join(viz_folder, f'patient_profile_{idx}.json'))

        # Visualize advanced analyses
        print("\nGenerating advanced visualizations...")
        visualize_statistical_results(statistical_results, viz_folder)
        visualize_model_interpretations(viz_folder, viz_folder)
        visualize_patient_profiles(viz_folder, viz_folder)
        visualize_combined_analysis(viz_folder)

        # After training, add clinical validation and safety monitoring
        print("\nPerforming clinical validation...")
        
        # Make predictions with uncertainty estimates
        y_pred_prob, y_pred = make_predictions(best_model, X_test, meta_X_test)
        uncertainty_results = interpreter.generate_uncertainty(X_test, meta_X_test)
        uncertainties = np.array([uncertainty_results['uncertainty']] if isinstance(uncertainty_results['uncertainty'], float) 
                               else uncertainty_results['uncertainty'])
        
        # Clinical metrics calculation
        clinical_results = clinical_metrics.calculate_clinical_metrics(
            y_test, y_pred_prob
        )
        
        # Clinical validation
        validation_results = clinical_validator.validate_model_safety(
            y_test, y_pred_prob, clinical_features=meta_X_test
        )
        
        # Safety monitoring
        safety_alerts, batch_stats = safety_monitor.monitor_batch_predictions(
            y_pred_prob, uncertainties, clinical_features=meta_X_test
        )
        
        # Generate clinical report
        print("\nGenerating clinical report...")
        prediction_results = {
            'prediction': float(np.mean(y_pred_prob)),
            'uncertainties': uncertainties.tolist() if hasattr(uncertainties, 'tolist') else [float(uncertainties)],
            'batch_statistics': batch_stats
        }
        
        report_path = clinical_report.generate_clinical_report(
            prediction_results,
            clinical_results,
            validation_results,
            patient_data=aligned_metadata
        )
        
        # Save all results
        results = {
            'clinical_metrics': clinical_results,
            'validation_results': validation_results,
            'safety_alerts': safety_alerts,
            'batch_statistics': batch_stats
        }
        
        save_json(results, os.path.join(experiment_folder, 'clinical_results.json'))
        
        print("\nTraining and validation completed successfully!")
        print(f"Results saved in: {experiment_folder}")
        print(f"Clinical report generated: {report_path}")
        
        # If there are critical safety alerts, print them
        critical_alerts = [a for a in safety_alerts if any(
            alert['level'] == 'CRITICAL' for alert in a.get('alerts', [])
        )]
        if critical_alerts:
            print("\nWARNING: Critical safety alerts detected!")
            for alert in critical_alerts:
                print(f"Sample {alert['index']}: {alert['alerts']}")
        
        return best_model, best_history, test_metrics, results

    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def generate_clinical_report(predictions, uncertainties, validation_results, safety_alerts, viz_folder):
    """Generate comprehensive clinical report"""
    try:
        # Convert inputs to lists if they're single values
        if isinstance(predictions, (float, int)):
            predictions = [predictions]
        if isinstance(uncertainties, (float, int)):
            uncertainties = [uncertainties]
            
        # Ensure all inputs are lists
        predictions = np.asarray(predictions).flatten().tolist()
        uncertainties = np.asarray(uncertainties).flatten().tolist()
        
        report_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'predictions': predictions,
            'uncertainties': uncertainties,
            'validation_results': validation_results,
            'safety_alerts': safety_alerts
        }
        
        # Save report
        report_path = os.path.join(viz_folder, 'clinical_report.json')
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=4)
            
        print("Clinical report generated successfully")
        return True
        
    except Exception as e:
        print(f"Error generating clinical report: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting transfer learning pipeline...")
    best_model, history, metrics, results = train_and_evaluate()
    
    if best_model is not None:
        print("\nFinal Test Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}") 