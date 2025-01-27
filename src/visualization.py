import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import traceback
from tensorflow.keras.models import Model

def save_plot(fig, folder, filename):
    """Save matplotlib figure to specified folder"""
    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    filepath = os.path.join(folder, f"{filename}.png")
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)

def visualize_results(history, viz_folder):
    """Visualize and save training results"""
    fig = plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    
    save_plot(fig, os.path.join(viz_folder, "training_progress"), "training_history")

def visualize_metadata(metadata, output_folder):
    """Visualize metadata distributions and relationships"""
    try:
        # Create metadata visualization folder
        viz_folder = os.path.join(output_folder, 'metadata_analysis')
        os.makedirs(viz_folder, exist_ok=True)
        
        # Convert column names to strings
        metadata = metadata.copy()
        metadata.columns = [str(col) for col in metadata.columns]
        
        # Print debug info
        print("\nMetadata Visualization Debug:")
        print("Columns after conversion:", metadata.columns.tolist())
        
        # Create visualizations for categorical variables
        categorical_cols = ['HPV', 'Label', 'Adequacy', 'Grade 1', 'Grade 2']
        plt.figure(figsize=(15, 10))
        
        for i, col in enumerate(categorical_cols, 1):
            if col in metadata.columns:
                plt.subplot(2, 3, i)
                sns.countplot(data=metadata, x=col)
                plt.title(f'{col} Distribution')
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_folder, 'categorical_distributions.png'))
        plt.close()
        
        # Create correlation heatmap for numerical variables
        numerical_cols = metadata.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = metadata[numerical_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Feature Correlations')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_folder, 'correlations.png'))
            plt.close()
        
        # Save summary statistics
        with open(os.path.join(viz_folder, 'metadata_summary.txt'), 'w') as f:
            f.write("Metadata Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Dataset Statistics:\n")
            f.write(f"Total samples: {len(metadata)}\n")
            f.write(f"Total features: {len(metadata.columns)}\n\n")
            
            f.write("Categorical Variables:\n")
            for col in metadata.select_dtypes(include=['object']).columns:
                f.write(f"\n{col}:\n")
                f.write(metadata[col].value_counts().to_string())
                f.write(f"\nMissing values: {metadata[col].isnull().sum()}\n")
            
            f.write("\nNumerical Variables:\n")
            f.write(metadata.select_dtypes(include=[np.number]).describe().to_string())
        
        print(f"\nMetadata visualizations saved to: {viz_folder}")
        
    except Exception as e:
        print(f"\nError in visualize_metadata: {str(e)}")
        print("\nDebug Information:")
        print("Metadata shape:", metadata.shape)
        print("Column names:", metadata.columns.tolist())
        print("Column types:")
        for col in metadata.columns:
            print(f"{col}: {type(col)}")
        traceback.print_exc()

def visualize_model_performance(y_true, y_pred, y_pred_prob, viz_folder):
    """Visualize and save model performance metrics"""
    # Confusion Matrix
    fig = plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    save_plot(fig, os.path.join(viz_folder, "model_performance"), "confusion_matrix")
    
    # ROC and Precision-Recall Curves
    fig = plt.figure(figsize=(12, 5))
    
    # ROC Curve
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    
    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    avg_precision = average_precision_score(y_true, y_pred_prob)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AP={avg_precision:.3f})')
    
    plt.tight_layout()
    save_plot(fig, os.path.join(viz_folder, "model_performance"), "performance_curves")

def visualize_learning_dynamics(history, viz_folder):
    """Visualize detailed learning dynamics"""
    metrics = {
        'loss': 'loss',
        'accuracy': 'accuracy',
        'precision': 'precision_1' if 'precision_1' in history.history else 'precision',
        'recall': 'recall_1' if 'recall_1' in history.history else 'recall',
        'auc': 'auc_1' if 'auc_1' in history.history else 'auc'
    }
    
    fig = plt.figure(figsize=(15, 10))
    
    for i, (display_name, metric_name) in enumerate(metrics.items(), 1):
        plt.subplot(2, 3, i)
        if metric_name in history.history:
            plt.plot(history.history[metric_name], label=f'Training {display_name}')
            plt.plot(history.history[f'val_{metric_name}'], label=f'Validation {display_name}')
            plt.title(f'Model {display_name.capitalize()}')
            plt.xlabel('Epoch')
            plt.ylabel(display_name.capitalize())
            plt.legend()
    
    plt.tight_layout()
    save_plot(fig, os.path.join(viz_folder, "training_progress"), "learning_dynamics")

def visualize_prediction_distribution(y_pred_prob, y_true, viz_folder):
    """Analyze and visualize prediction distribution"""
    fig = plt.figure(figsize=(12, 4))
    
    # Prediction distribution for each class
    plt.subplot(1, 2, 1)
    for label in [0, 1]:
        mask = y_true == label
        plt.hist(y_pred_prob[mask], bins=20, alpha=0.5, 
                label=f'Class {label}', density=True)
    plt.xlabel('Prediction Probability')
    plt.ylabel('Density')
    plt.title('Prediction Distribution by Class')
    plt.legend()
    
    # Prediction confidence analysis
    plt.subplot(1, 2, 2)
    confidence = np.abs(y_pred_prob - 0.5) * 2
    correct = (y_pred_prob > 0.5) == y_true
    plt.scatter(confidence, y_pred_prob, c=correct, 
               cmap='coolwarm', alpha=0.5)
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Prediction Probability')
    plt.title('Prediction Confidence vs Accuracy')
    
    plt.tight_layout()
    save_plot(fig, os.path.join(viz_folder, "model_performance"), "prediction_analysis")

def visualize_learning_progression(viz_folder):
    """Visualize the learning progression across iterations"""
    try:
        model_folder = os.path.join(viz_folder, "model_evolution", "intermediate_models")
        if not os.path.exists(model_folder):
            return None
            
        # Load all metrics files
        metrics_files = [f for f in os.listdir(model_folder) if f.startswith("metrics_iteration_") and f.endswith(".json")]
        if not metrics_files:
            return None
            
        # Extract data
        progression_data = {
            'iteration': [],
            'val_accuracy': [],
            'val_loss': [],
            'learning_rate': [],
            'batch_size': [],
            'dropout_rate': [],
            'training_time': []
        }
        
        for metrics_file in sorted(metrics_files, key=lambda x: int(x.split('_')[-1].split('.')[0])):
            with open(os.path.join(model_folder, metrics_file), 'r') as f:
                data = json.load(f)
                progression_data['iteration'].append(data['iteration'])
                progression_data['val_accuracy'].append(data['metrics']['val_accuracy'])
                progression_data['val_loss'].append(data['metrics']['val_loss'])
                progression_data['learning_rate'].append(data['parameters']['learning_rate'])
                progression_data['batch_size'].append(data['parameters']['batch_size'])
                progression_data['dropout_rate'].append(data['parameters']['dropout_rate'])
                progression_data['training_time'].append(data.get('training_time', 0))
        
        # Create visualization
        fig = plt.figure(figsize=(15, 10))
        
        # Plot accuracy and loss
        plt.subplot(2, 2, 1)
        plt.plot(progression_data['iteration'], progression_data['val_accuracy'], 'b-', label='Validation Accuracy')
        plt.title('Validation Accuracy Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(progression_data['iteration'], progression_data['val_loss'], 'r-', label='Validation Loss')
        plt.title('Validation Loss Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Plot hyperparameters
        plt.subplot(2, 2, 3)
        plt.scatter(progression_data['learning_rate'], progression_data['val_accuracy'])
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Validation Accuracy')
        plt.title('Accuracy vs Learning Rate')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.scatter(progression_data['dropout_rate'], progression_data['val_accuracy'])
        plt.xlabel('Dropout Rate')
        plt.ylabel('Validation Accuracy')
        plt.title('Accuracy vs Dropout Rate')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_folder, 'model_evolution', 'learning_progression.png'))
        plt.close()
        
        return progression_data
        
    except Exception as e:
        print(f"Error visualizing learning progression: {str(e)}")
        return None

def visualize_metadata_distributions(metadata, viz_folder):
    """Visualize metadata distributions and save to metadata_analysis/distributions"""
    # HPV Distribution
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    sns.countplot(data=metadata, x='HPV')
    plt.title('HPV Status Distribution')
    
    # Swede Score Distribution (if available)
    if 'SwedeFinal' in metadata.columns:
        plt.subplot(2, 2, 2)
        sns.histplot(data=metadata, x='SwedeFinal', bins=20)
        plt.title('Swede Score Distribution')
    
    # Location Distribution
    if 'Location of the lesion' in metadata.columns:
        plt.subplot(2, 2, 3)
        location_counts = metadata['Location of the lesion'].value_counts()
        plt.pie(location_counts, labels=location_counts.index, autopct='%1.1f%%')
        plt.title('Lesion Location Distribution')
    
    # Grade Distribution
    if 'Grade 1' in metadata.columns and 'Grade 2' in metadata.columns:
        plt.subplot(2, 2, 4)
        grade_data = pd.melt(metadata[['Grade 1', 'Grade 2']])
        sns.boxplot(data=grade_data, x='variable', y='value')
        plt.title('Grade Distribution')
    
    plt.tight_layout()
    save_plot(fig, os.path.join(viz_folder, "metadata_analysis", "distributions"), "metadata_distributions")

def visualize_weight_updates(weight_changes, viz_folder):
    """Visualize model weight evolution"""
    if not weight_changes:
        return
        
    fig = plt.figure(figsize=(12, 6))
    
    # Plot weight changes by layer
    layers = [change['layer_name'] for change in weight_changes]
    diffs = [change['weight_diff'] for change in weight_changes]
    
    plt.barh(layers, diffs)
    plt.xlabel('Average Weight Change')
    plt.ylabel('Layer Name')
    plt.title('Model Weight Updates by Layer')
    
    plt.tight_layout()
    save_plot(fig, os.path.join(viz_folder, "model_evolution", "weight_updates"), "weight_changes")

def visualize_hyperparameter_effects(tracking_dict, save_folder):
    """Visualize the effects of hyperparameters on model performance"""
    plt.figure(figsize=(20, 15))
    
    # Learning Rate Effect
    plt.subplot(2, 2, 1)
    lr_df = pd.DataFrame({
        'learning_rate': tracking_dict['learning_rates'],
        'accuracy': tracking_dict['val_accuracies']
    })
    lr_mean = lr_df.groupby('learning_rate')['accuracy'].mean()
    plt.semilogx(lr_mean.index, lr_mean.values, 'o-')
    plt.grid(True)
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy')
    plt.title('Learning Rate Effect')
    
    # Batch Size Effect
    plt.subplot(2, 2, 2)
    batch_df = pd.DataFrame({
        'batch_size': tracking_dict['batch_sizes'],
        'accuracy': tracking_dict['val_accuracies']
    })
    batch_mean = batch_df.groupby('batch_size')['accuracy'].mean()
    plt.plot(batch_mean.index, batch_mean.values, 'o-')
    plt.grid(True)
    plt.xlabel('Batch Size')
    plt.ylabel('Validation Accuracy')
    plt.title('Batch Size Effect')
    
    # Dropout Rate Effect
    plt.subplot(2, 2, 3)
    dropout_df = pd.DataFrame({
        'dropout_rate': tracking_dict['dropout_rates'],
        'accuracy': tracking_dict['val_accuracies']
    })
    dropout_mean = dropout_df.groupby('dropout_rate')['accuracy'].mean()
    plt.plot(dropout_mean.index, dropout_mean.values, 'o-')
    plt.grid(True)
    plt.xlabel('Dropout Rate')
    plt.ylabel('Validation Accuracy')
    plt.title('Dropout Rate Effect')
    
    # Training Time vs Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(tracking_dict['training_times'], tracking_dict['val_accuracies'], 'o-')
    plt.grid(True)
    plt.xlabel('Training Time (s)')
    plt.ylabel('Validation Accuracy')
    plt.title('Training Time vs Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'hyperparameter_effects.png'))
    plt.close()
    
    # Save numerical results
    results = {
        'learning_rate_effect': lr_mean.to_dict(),
        'batch_size_effect': batch_mean.to_dict(),
        'dropout_effect': dropout_mean.to_dict(),
        'training_time_correlation': np.corrcoef(tracking_dict['training_times'], 
                                               tracking_dict['val_accuracies'])[0,1]
    }
    
    with open(os.path.join(save_folder, 'hyperparameter_analysis.json'), 'w') as f:
        json.dump(results, f, indent=4)

def visualize_data_augmentation(sample_image, viz_folder):
    """Visualize data augmentation effects on a sample image"""
    try:
        # Create augmentation folder
        aug_folder = os.path.join(viz_folder, 'augmentation_examples')
        os.makedirs(aug_folder, exist_ok=True)
        
        # Undo EfficientNet preprocessing
        # EfficientNet uses: x = (x - mean) / std
        # So to reverse: x = (x * std) + mean
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        
        # Denormalize image
        sample_image = (sample_image * std) + mean
        sample_image = np.clip(sample_image, 0, 255).astype('uint8')
        
        print(f"Debug - Image stats after denormalization:")
        print(f"Shape: {sample_image.shape}, Min: {sample_image.min()}, Max: {sample_image.max()}")
        
        # Setup data generator with rescaling turned off
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            rescale=None
        )
        
        # Create iterator
        img_arr = np.expand_dims(sample_image, 0)
        it = datagen.flow(img_arr, batch_size=1)
        
        # Generate augmented images
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 4, 1)
        plt.imshow(sample_image)
        plt.title('Original')
        plt.axis('off')
        
        # Three augmented versions
        for i in range(3):
            plt.subplot(1, 4, i+2)
            aug_img = next(it)[0].astype('uint8')
            plt.imshow(aug_img)
            plt.title(f'Augmented {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(aug_folder, 'augmentation_examples.png'))
        plt.close()
        
        print(f"Augmentation examples saved to: {aug_folder}")
        
    except Exception as e:
        print(f"Error in data augmentation visualization: {str(e)}")
        print(f"Image shape: {sample_image.shape}, dtype: {sample_image.dtype}")
        print(f"Image min: {sample_image.min()}, max: {sample_image.max()}")
        traceback.print_exc()

def visualize_clinical_correlations(metadata, viz_folder):
    """Visualize correlations between clinical features and HPV status"""
    clinical_features = [
        'SwedeFinal', 'Aceto uptake', 'Margins', 'Vessels', 
        'Lesion size', 'Iodine uptake'
    ]
    
    fig = plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(clinical_features, 1):
        if feature in metadata.columns:
            plt.subplot(2, 3, i)
            if pd.api.types.is_numeric_dtype(metadata[feature]):
                sns.boxplot(data=metadata, x='HPV', y=feature)
            else:
                sns.countplot(data=metadata, x=feature, hue='HPV')
            plt.title(f'{feature} vs HPV Status')
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    save_plot(fig, os.path.join(viz_folder, "metadata_analysis", "clinical_correlations"), 
             "clinical_features_vs_hpv")

def visualize_model_interpretation(model, sample_image, sample_metadata, clinical_features, viz_folder):
    """Visualize how model interprets both image and clinical features"""
    fig = plt.figure(figsize=(20, 10))
    
    # Original prediction
    base_pred = model.predict([np.expand_dims(sample_image, 0), 
                             np.expand_dims(sample_metadata, 0)])[0][0]
    
    # Image interpretation using Grad-CAM
    last_conv_layer = next(layer for layer in reversed(model.layers) 
                          if isinstance(layer, tf.keras.layers.Conv2D))
    
    with tf.GradientTape() as tape:
        # Get the convolutional output and predictions
        conv_output = model.get_layer(last_conv_layer.name).output
        grad_model = Model(inputs=model.inputs, outputs=[conv_output, model.output])
        conv_output, predictions = grad_model([np.expand_dims(sample_image, 0), 
                                               np.expand_dims(sample_metadata, 0)])
        loss = predictions[:, 0]
        
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    
    # Plot original image
    plt.subplot(2, 2, 1)
    plt.imshow(sample_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot heatmap
    plt.subplot(2, 2, 2)
    plt.imshow(sample_image)
    plt.imshow(heatmap[0], alpha=0.6, cmap='jet')
    plt.title('Feature Activation Heatmap')
    plt.axis('off')
    
    # Clinical feature contribution
    plt.subplot(2, 2, 3)
    feature_impacts = []
    for i, feature in enumerate(clinical_features):
        modified = sample_metadata.copy()
        modified[i] = 0  # Zero out feature
        new_pred = model.predict([np.expand_dims(sample_image, 0), 
                                np.expand_dims(modified, 0)])[0][0]
        impact = base_pred - new_pred
        feature_impacts.append((feature, impact))
    
    # Plot feature impacts
    features, impacts = zip(*sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True))
    plt.barh(range(len(features)), impacts)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Impact on Prediction')
    plt.title('Clinical Feature Contributions')
    
    # Combined prediction explanation
    plt.subplot(2, 2, 4)
    plt.axis('off')
    explanation_text = f"""
    Model Prediction: {base_pred:.3f}
    
    Top Contributing Features:
    {chr(10).join(f'• {f}: {i:.3f}' for f, i in feature_impacts[:5])}
    
    Image Contribution:
    • Activation strength: {np.max(heatmap):.3f}
    • Active regions: {np.sum(heatmap > np.mean(heatmap)):.0f} pixels
    """
    plt.text(0.1, 0.9, explanation_text, fontsize=10, 
             verticalalignment='top', wrap=True)
    
    plt.tight_layout()
    save_plot(fig, os.path.join(viz_folder, "model_interpretation"), "feature_interpretation")

def visualize_clinical_patterns(metadata, viz_folder):
    """Visualize patterns and relationships in clinical features"""
    # Create correlation matrix
    clinical_features = [
        'SwedeFinal', 'Aceto uptake', 'Margins', 'Vessels', 
        'Lesion size', 'Iodine uptake', 'Grade 1', 'Grade 2'
    ]
    
    numeric_features = metadata[clinical_features].select_dtypes(include=['float64', 'int64'])
    
    fig = plt.figure(figsize=(20, 15))
    
    # Correlation heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(numeric_features.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')
    
    # Feature combinations
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=metadata, x='SwedeFinal', y='Lesion size', 
                   hue='HPV', style='HPV', s=100)
    plt.title('Swede Score vs Lesion Size by HPV Status')
    
    # Clinical patterns
    plt.subplot(2, 2, 3)
    pattern_data = metadata.groupby(['Margins', 'Vessels'])['HPV'].value_counts(normalize=True)
    pattern_data = pattern_data.unstack()
    sns.heatmap(pattern_data, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('HPV Patterns by Margins and Vessels')
    
    # Grade progression
    plt.subplot(2, 2, 4)
    grade_data = metadata[['Grade 1', 'Grade 2', 'HPV']].melt(id_vars=['HPV'])
    sns.boxplot(data=grade_data, x='variable', y='value', hue='HPV')
    plt.title('Grade Distribution by HPV Status')
    
    plt.tight_layout()
    save_plot(fig, os.path.join(viz_folder, "clinical_analysis"), "clinical_patterns")

if __name__ == "__main__":
    print("Testing visualization module...")
    # Create a test visualization
    test_folder = "test_visualizations"
    os.makedirs(test_folder, exist_ok=True)
    
    # Create sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Create and save a test plot
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title("Test Visualization")
    save_plot(fig, test_folder, "test_plot")
    
    print(f"Test plot saved to {test_folder}/test_plot.png") 