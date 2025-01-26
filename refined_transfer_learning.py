import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from collections import Counter
import json
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_images_from_folder, load_metadata, visualize_metadata, display_dataset_info
from tensorflow.keras import regularizers
from itertools import product
import scipy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from scipy import stats
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score, matthews_corrcoef, cohen_kappa_score
import time

# Paths to the dataset
positive_folder = "D:\\categorization\\new\\hpv\\positive"
negative_folder = "D:\\categorization\\new\\hpv\\negative"
xlsx_file_path = "D:\\categorization\\IARCImageBankColpo\\Cases Meta data.xlsx"
save_folder = "D:\\transfer_learning"

def extract_case_id_from_filename(filename):
    """Extract CaseID from filename"""
    case_id = filename.split('.')[0]  # Remove file extension
    case_id = ''.join([char for char in case_id if not char.isdigit()])  # Remove digits
    return case_id.strip().lower()  # Ensure lowercase and clean up

def align_data(all_images, all_labels, all_filenames, metadata):
    """Align images with metadata"""
    metadata["CaseID"] = metadata["CaseID"].str.strip().str.lower()
    
    # Extract CaseIDs from image filenames
    image_case_ids = [extract_case_id_from_filename(fname) for fname in all_filenames]
    
    # Debug alignment
    print(f"Number of CaseIDs in metadata: {len(metadata['CaseID'].unique())}")
    print(f"Number of CaseIDs in image dataset: {len(image_case_ids)}")
    
    # Filter metadata to include only matching CaseIDs
    metadata = metadata[metadata["CaseID"].isin(image_case_ids)].reset_index(drop=True)
    
    # Align images and labels
    aligned_indices = [
        image_case_ids.index(case_id) for case_id in metadata["CaseID"] if case_id in image_case_ids
    ]
    all_images_aligned = all_images[aligned_indices]
    all_labels_aligned = all_labels[aligned_indices]
    
    return all_images_aligned, all_labels_aligned, metadata

def preprocess_metadata(metadata):
    """Preprocess metadata features"""
    # Create a copy of metadata to avoid warnings
    metadata = metadata.copy()
    
    # Handle missing values
    for column in metadata.columns:
        if pd.api.types.is_numeric_dtype(metadata[column]):
            if metadata[column].isna().all():
                metadata[column] = metadata[column].fillna(0)
            else:
                metadata[column] = metadata[column].fillna(metadata[column].mean())
        elif pd.api.types.is_object_dtype(metadata[column]):
            if metadata[column].mode().empty:
                metadata[column] = metadata[column].fillna('Unknown')
            else:
                metadata[column] = metadata[column].fillna(metadata[column].mode().iloc[0])

    # Separate metadata features and labels
    metadata_features = metadata.drop(columns=["Label", "CaseID"], errors="ignore")
    metadata_labels = metadata["Label"]

    try:
        # Create preprocessor with sparse_output=False for newer scikit-learn versions
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), 
                 metadata_features.select_dtypes(include=["float64", "int64"]).columns),
                ('cat', OneHotEncoder(handle_unknown="ignore", sparse_output=False), 
                 metadata_features.select_dtypes(include=["object"]).columns)
            ],
            remainder='passthrough'
        )

        # Process metadata features
        metadata_features_processed = preprocessor.fit_transform(metadata_features)
        
        print(f"Processed metadata features shape: {metadata_features_processed.shape}")
        
    except TypeError as e:
        # Fallback for older scikit-learn versions
        if "sparse_output" in str(e):
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), 
                     metadata_features.select_dtypes(include=["float64", "int64"]).columns),
                    ('cat', OneHotEncoder(handle_unknown="ignore", sparse=False), 
                     metadata_features.select_dtypes(include=["object"]).columns)
                ],
                remainder='passthrough'
            )
            metadata_features_processed = preprocessor.fit_transform(metadata_features)

    return metadata_features_processed, metadata_labels

def load_and_preprocess_data():
    """Load and preprocess all data using data_loader functions"""
    print("Starting data loading process...")
    
    # Load images using data_loader functions
    positive_images, positive_labels, positive_filenames = load_images_from_folder(positive_folder, label=1)
    negative_images, negative_labels, negative_filenames = load_images_from_folder(negative_folder, label=0)
    
    # Combine datasets
    all_images = np.array(positive_images + negative_images)
    all_labels = np.array(positive_labels + negative_labels)
    all_filenames = positive_filenames + negative_filenames
    
    # Load metadata
    metadata = load_metadata(xlsx_file_path)
    
    # Display dataset information
    display_dataset_info(all_images, all_labels, all_filenames, metadata)
    
    return all_images, all_labels, all_filenames, metadata

def setup_hyperparameters():
    """Setup hyperparameter combinations for model tuning"""
    learning_rates = [1e-3, 1e-4, 1e-5]
    batch_sizes = [16, 32, 64]
    dropout_rates = [0.2, 0.3, 0.5]
    return list(product(learning_rates, batch_sizes, dropout_rates))

def create_model(dropout_rate, image_input_shape=(224, 224, 3), metadata_input_shape=None):
    """Create the hybrid model combining image and metadata"""
    # Image input branch with EfficientNetB0
    image_input = tf.keras.Input(shape=image_input_shape, name="image_input")
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=image_input
    )
    base_model.trainable = True
    for layer in base_model.layers[:-20]:  # Freeze all layers except the last 20
        layer.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Metadata input branch
    # Convert single number to tuple for metadata shape
    if isinstance(metadata_input_shape, (int, float)):
        metadata_input_shape = (metadata_input_shape,)
        
    meta_input = tf.keras.Input(shape=metadata_input_shape, name="meta_input")
    meta_x = tf.keras.layers.Dense(128, activation="relu")(meta_input)
    meta_x = tf.keras.layers.Dropout(dropout_rate)(meta_x)

    # Combine branches
    combined = tf.keras.layers.concatenate([x, meta_x])
    combined_x = tf.keras.layers.Dense(256, activation="relu")(combined)
    combined_x = tf.keras.layers.Dropout(0.5)(combined_x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(combined_x)

    model = tf.keras.Model(inputs=[image_input, meta_input], outputs=output)
    return model

def setup_callbacks(save_folder):
    """Setup training callbacks"""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=5, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.2, 
            patience=3, 
            min_lr=1e-7
        ),
    ]
    return callbacks

def save_training_history(history, save_path):
    """Save training history with proper error handling"""
    try:
        history_dict = {}
        for key, value in history.history.items():
            history_dict[key] = [float(v) for v in value]
        
        with open(save_path, 'w') as f:
            json.dump(history_dict, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving training history: {str(e)}")
        return False

def train_model(model, train_data, val_data, batch_size, callbacks, class_weights):
    """Train the model with given parameters"""
    X_train, meta_X_train, y_train = train_data
    X_val, meta_X_val, y_val = val_data
    
    history = model.fit(
        [X_train, meta_X_train],
        y_train,
        validation_data=([X_val, meta_X_val], y_val),
        epochs=1,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    return history

def evaluate_model(model, test_data):
    """Evaluate model performance"""
    X_test, meta_X_test, y_test = test_data
    metrics = model.evaluate(
        [X_test, meta_X_test], 
        y_test, 
        verbose=1
    )
    return dict(zip(model.metrics_names, metrics))

def setup_visualization_folder():
    """Create organized folders for saving visualizations"""
    viz_folder = os.path.join(save_folder, "visualizations")
    subfolders = {
        "model_evolution": [
            "intermediate_models",
            "learning_curves",
            "weight_updates"
        ],
        "hyperparameter_tuning": [
            "learning_rates",
            "batch_sizes",
            "dropout_rates",
            "combinations"
        ],
        "performance_metrics": [
            "confusion_matrices",
            "roc_curves",
            "precision_recall",
            "gradcam"
        ],
        "data_analysis": [
            "distributions",
            "metadata_correlations",
            "augmentations",
            "normalizations"
        ],
        "training_progress": [
            "epoch_metrics",
            "validation_results",
            "learning_dynamics"
        ]
    }
    
    # Create main folder and all subfolders
    for main_folder, sub_folders in subfolders.items():
        main_path = os.path.join(viz_folder, main_folder)
        if not os.path.exists(main_path):
            os.makedirs(main_path)
        for sub_folder in sub_folders:
            sub_path = os.path.join(main_path, sub_folder)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
    
    return viz_folder

def save_plot(fig, folder, filename):
    """Save matplotlib figure to specified folder"""
    filepath = os.path.join(folder, f"{filename}.png")
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)

def visualize_results(history, viz_folder):
    """Visualize and save training results"""
    # Accuracy plot
    fig = plt.figure(figsize=(12, 4))
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
    
    # Save the plot
    save_plot(fig, os.path.join(viz_folder, "training_progress"), "training_history")

def visualize_metadata(metadata, viz_folder):
    """Visualize and save metadata analysis"""
    # HPV Status Distribution
    fig = plt.figure(figsize=(10, 8))
    hpv_counts = metadata['Label'].value_counts()
    plt.pie(hpv_counts, labels=['Negative', 'Positive'], 
            autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
    plt.title('HPV Status Distribution')
    save_plot(fig, os.path.join(viz_folder, "metadata_analysis"), "hpv_distribution")

    # Age Distribution
    age_col = next((col for col in metadata.columns if 'age' in col.lower()), None)
    if age_col:
        fig = plt.figure(figsize=(10, 6))
        metadata[age_col].hist(bins=20, color='skyblue')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Count')
        save_plot(fig, os.path.join(viz_folder, "metadata_analysis"), "age_distribution")
    
    # Correlation Heatmap
    numerical_cols = metadata.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_cols) > 1:
        fig = plt.figure(figsize=(12, 10))
        correlation = metadata[numerical_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        save_plot(fig, os.path.join(viz_folder, "metadata_analysis"), "correlation_heatmap")

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
    
    # Save classification report
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])
    report_path = os.path.join(viz_folder, "model_performance", "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

def save_predictions(y_true, y_pred, y_pred_prob, filenames, save_folder):
    """Save model predictions to CSV with proper error handling"""
    try:
        # Ensure all arrays are 1-dimensional
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        y_pred_prob = np.asarray(y_pred_prob).ravel()
        
        # Verify lengths match
        if not (len(filenames) == len(y_true) == len(y_pred) == len(y_pred_prob)):
            raise ValueError(f"Length mismatch: filenames({len(filenames)}), y_true({len(y_true)}), "
                           f"y_pred({len(y_pred)}), y_pred_prob({len(y_pred_prob)})")
        
        # Create DataFrame with verified 1D arrays
        predictions_df = pd.DataFrame({
            'Filename': filenames,
            'True_Label': y_true,
            'Predicted_Label': y_pred,
            'Prediction_Probability': y_pred_prob
        })
        
        # Save to CSV
        predictions_path = os.path.join(save_folder, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to: {predictions_path}")
        return True
        
    except Exception as e:
        print(f"Error saving predictions: {str(e)}")
        print(f"Shapes: y_true: {np.asarray(y_true).shape}, "
              f"y_pred: {np.asarray(y_pred).shape}, "
              f"y_pred_prob: {np.asarray(y_pred_prob).shape}")
        return False

def visualize_normalization(original_image, normalized_image, viz_folder, index):
    """Visualize and save normalization comparison"""
    fig = plt.figure(figsize=(12, 4))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image.astype('uint8'))
    plt.title('Original Image')
    plt.axis('off')
    
    # Normalized Image
    plt.subplot(1, 2, 2)
    plt.imshow(normalized_image)
    plt.title('Normalized Image')
    plt.axis('off')
    
    save_plot(fig, os.path.join(viz_folder, "normalization"), f"normalization_sample_{index}")

def setup_data_augmentation():
    """Setup data augmentation for training"""
    return ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=preprocess_input
    )

def visualize_augmentation(image, viz_folder):
    """Visualize data augmentation examples"""
    data_augmentation = setup_data_augmentation()
    
    fig = plt.figure(figsize=(15, 3))
    plt.subplot(1, 5, 1)
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    
    for i in range(4):
        augmented = data_augmentation.random_transform(image)
        plt.subplot(1, 5, i + 2)
        plt.imshow(augmented)
        plt.title(f'Augmented {i+1}')
        plt.axis('off')
    
    save_plot(fig, os.path.join(viz_folder, "data_distribution"), "augmentation_examples")

def generate_gradcam(model, image, meta_data, layer_name='top_conv'):
    """Generate Grad-CAM heatmap"""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([
            tf.expand_dims(image, 0),
            tf.expand_dims(meta_data, 0)
        ])
        loss = predictions[:, 0]
        
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(output, weights), axis=-1)
    
    cam = tf.maximum(cam, 0) / tf.math.reduce_max(cam)
    return cam.numpy()

def visualize_gradcam(image, heatmap, prediction, true_label, viz_folder, index):
    """Visualize and save Grad-CAM results"""
    # Resize heatmap to match image size
    heatmap = tf.image.resize(
        tf.expand_dims(heatmap, -1),
        (image.shape[0], image.shape[1])
    ).numpy()
    
    fig = plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f'Original (True: {true_label})')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap[..., 0], cmap='jet')
    plt.title(f'Grad-CAM (Pred: {prediction:.2f})')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(heatmap[..., 0], cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')
    
    save_plot(fig, os.path.join(viz_folder, "model_performance"), f"gradcam_sample_{index}")

def track_hyperparameter_performance():
    """Track performance of each hyperparameter combination"""
    return {
        'learning_rates': [],
        'batch_sizes': [],
        'dropout_rates': [],
        'val_accuracies': [],
        'val_losses': [],
        'training_times': [],
        'epochs_trained': [],
        'precision': [],
        'recall': [],
        'auc': []
    }

def visualize_hyperparameter_comparison(tracking_dict, viz_folder):
    """Visualize hyperparameter comparison"""
    results_df = pd.DataFrame({
        'Learning Rate': tracking_dict['learning_rates'],
        'Batch Size': tracking_dict['batch_sizes'],
        'Dropout Rate': tracking_dict['dropout_rates'],
        'Validation Accuracy': tracking_dict['val_accuracies'],
        'Validation Loss': tracking_dict['val_losses'],
        'Training Time (s)': tracking_dict['training_times'],
        'Epochs': tracking_dict['epochs_trained'],
        'Precision': tracking_dict['precision'],
        'Recall': tracking_dict['recall'],
        'AUC': tracking_dict['auc']
    })
    
    # Save detailed results
    results_path = os.path.join(viz_folder, "hyperparameter_tuning", "detailed_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Create visualization plots
    # 1. Learning Rate Analysis
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    sns.boxplot(x='Learning Rate', y='Validation Accuracy', data=results_df)
    plt.title('Learning Rate vs Accuracy')
    plt.xticks(rotation=45)
    
    # 2. Batch Size Analysis
    plt.subplot(2, 2, 2)
    sns.boxplot(x='Batch Size', y='Validation Accuracy', data=results_df)
    plt.title('Batch Size vs Accuracy')
    
    # 3. Dropout Rate Analysis
    plt.subplot(2, 2, 3)
    sns.boxplot(x='Dropout Rate', y='Validation Accuracy', data=results_df)
    plt.title('Dropout Rate vs Accuracy')
    
    # 4. Training Time Analysis
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='Training Time (s)', y='Validation Accuracy', 
                   size='Batch Size', hue='Learning Rate', data=results_df)
    plt.title('Training Time vs Accuracy')
    
    plt.tight_layout()
    save_plot(fig, os.path.join(viz_folder, "hyperparameter_tuning"), "hyperparameter_comparison")
    
    # Performance Metrics Heatmap
    plt.figure(figsize=(12, 8))
    metrics_heatmap = results_df.pivot_table(
        values='Validation Accuracy',
        index='Learning Rate',
        columns='Batch Size',
        aggfunc='mean'
    )
    sns.heatmap(metrics_heatmap, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Average Validation Accuracy by Learning Rate and Batch Size')
    save_plot(plt.gcf(), os.path.join(viz_folder, "hyperparameter_tuning"), "accuracy_heatmap")
    
    # Find best combinations
    best_accuracy_idx = results_df['Validation Accuracy'].idxmax()
    best_combo = results_df.iloc[best_accuracy_idx]
    
    # Save best parameters summary
    summary = f"""
    Best Hyperparameter Combination:
    ------------------------------
    Learning Rate: {best_combo['Learning Rate']}
    Batch Size: {best_combo['Batch Size']}
    Dropout Rate: {best_combo['Dropout Rate']}
    
    Performance:
    -----------
    Validation Accuracy: {best_combo['Validation Accuracy']:.4f}
    Precision: {best_combo['Precision']:.4f}
    Recall: {best_combo['Recall']:.4f}
    AUC: {best_combo['AUC']:.4f}
    Training Time: {best_combo['Training Time (s)']:.2f} seconds
    Epochs Trained: {best_combo['Epochs']}
    """
    
    with open(os.path.join(viz_folder, "hyperparameter_tuning", "best_parameters_summary.txt"), 'w') as f:
        f.write(summary)

def analyze_hyperparameter_statistics(tracking_dict, viz_folder):
    """Perform statistical analysis on hyperparameter performance"""
    results_df = pd.DataFrame({
        'Learning Rate': tracking_dict['learning_rates'],
        'Batch Size': tracking_dict['batch_sizes'],
        'Dropout Rate': tracking_dict['dropout_rates'],
        'Validation Accuracy': tracking_dict['val_accuracies']
    })
    
    # Statistical Analysis
    stats_summary = {
        'learning_rate': {},
        'batch_size': {},
        'dropout_rate': {}
    }
    
    # ANOVA test for each hyperparameter
    for param in ['Learning Rate', 'Batch Size', 'Dropout Rate']:
        groups = [group['Validation Accuracy'].values 
                 for name, group in results_df.groupby(param)]
        f_stat, p_val = stats.f_oneway(*groups)
        
        stats_summary[param.lower().replace(' ', '_')] = {
            'f_statistic': float(f_stat),
            'p_value': float(p_val),
            'mean_accuracies': results_df.groupby(param)['Validation Accuracy'].mean().to_dict(),
            'std_accuracies': results_df.groupby(param)['Validation Accuracy'].std().to_dict()
        }
    
    # Save statistical analysis
    with open(os.path.join(viz_folder, "hyperparameter_tuning", "statistical_analysis.json"), 'w') as f:
        json.dump(stats_summary, f, indent=4)
    
    return stats_summary

def visualize_advanced_metrics(y_true, y_pred, y_pred_prob, viz_folder):
    """Generate and save advanced performance metric visualizations"""
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    avg_precision = average_precision_score(y_true, y_pred_prob)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AP={avg_precision:.3f})')
    
    plt.tight_layout()
    save_plot(fig, os.path.join(viz_folder, "model_performance"), "advanced_metrics")
    
    # Calculate additional metrics
    additional_metrics = {
        'f1_score': f1_score(y_true, y_pred),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'average_precision': avg_precision
    }
    
    # Save additional metrics
    with open(os.path.join(viz_folder, "model_performance", "additional_metrics.json"), 'w') as f:
        json.dump(additional_metrics, f, indent=4)
    
    return additional_metrics

def visualize_learning_dynamics(history, viz_folder):
    """Visualize detailed learning dynamics"""
    # Update metric names with fallbacks
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

def analyze_prediction_distribution(y_pred_prob, y_true, viz_folder):
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

def save_intermediate_model(model, metrics, params, iteration, viz_folder):
    """Save intermediate model and its performance metrics"""
    model_folder = os.path.join(viz_folder, "model_evolution", "intermediate_models")
    # Add .keras extension to model path
    model_path = os.path.join(model_folder, f"model_iteration_{iteration}.keras")  # Added .keras extension
    
    # Save model
    model.save(model_path)
    
    # Save metrics and parameters
    metrics_file = os.path.join(model_folder, f"metrics_iteration_{iteration}.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            'metrics': metrics,
            'parameters': params,
            'iteration': iteration
        }, f, indent=4)

def load_previous_best_model(viz_folder):
    """Load the best model from previous training if available"""
    model_folder = os.path.join(viz_folder, "model_evolution", "intermediate_models")
    if not os.path.exists(model_folder):
        return None, None
    
    # Find the latest model (now looking for .keras extension)
    model_files = [f for f in os.listdir(model_folder) if f.startswith("model_iteration_") and f.endswith(".keras")]
    if not model_files:
        return None, None
    
    latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_metrics = f"metrics_iteration_{latest_model.split('_')[-1].split('.')[0]}.json"
    
    # Load model and metrics
    model_path = os.path.join(model_folder, latest_model)
    metrics_path = os.path.join(model_folder, latest_metrics)
    
    model = tf.keras.models.load_model(model_path)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return model, metrics

def get_test_filenames(all_filenames, all_labels_aligned, X_train, X_val):
    """Get correct test filenames based on dataset splits"""
    test_indices = np.arange(len(all_labels_aligned))[len(X_train) + len(X_val):]
    return [all_filenames[i] for i in test_indices]

def make_predictions(model, X_test, meta_X_test):
    """Make predictions with error handling"""
    try:
        y_pred_prob = model.predict([X_test, meta_X_test])
        y_pred = (y_pred_prob > 0.5).astype(int)
        return y_pred_prob, y_pred
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return None, None

def visualize_learning_progression(viz_folder):
    """Visualize learning progression with error handling"""
    model_folder = os.path.join(viz_folder, "model_evolution", "intermediate_models")
    
    if not os.path.exists(model_folder):
        print(f"Model folder not found: {model_folder}")
        return None
    
    metrics_files = sorted([f for f in os.listdir(model_folder) 
                          if f.startswith("metrics_iteration_")],
                         key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not metrics_files:
        print("No metrics files found for visualization")
        return None
    
    progression_data = {
        'iteration': [],
        'val_accuracy': [],
        'val_loss': [],
        'training_time': [],
        'learning_rate': [],
        'batch_size': [],
        'dropout_rate': []
    }
    
    for metrics_file in metrics_files:
        with open(os.path.join(model_folder, metrics_file), 'r') as f:
            data = json.load(f)
            progression_data['iteration'].append(data['iteration'])
            progression_data['val_accuracy'].append(data['metrics']['val_accuracy'])
            progression_data['val_loss'].append(data['metrics']['val_loss'])
            progression_data['training_time'].append(data['metrics']['training_time'])
            progression_data['learning_rate'].append(data['parameters']['learning_rate'])
            progression_data['batch_size'].append(data['parameters']['batch_size'])
            progression_data['dropout_rate'].append(data['parameters']['dropout_rate'])
    
    # Create visualization of learning progression
    fig = plt.figure(figsize=(15, 10))
    
    # Plot accuracy progression
    plt.subplot(2, 2, 1)
    plt.plot(progression_data['iteration'], progression_data['val_accuracy'], 'b-o')
    plt.title('Validation Accuracy Progression')
    plt.xlabel('Iteration')
    plt.ylabel('Validation Accuracy')
    
    # Plot loss progression
    plt.subplot(2, 2, 2)
    plt.plot(progression_data['iteration'], progression_data['val_loss'], 'r-o')
    plt.title('Validation Loss Progression')
    plt.xlabel('Iteration')
    plt.ylabel('Validation Loss')
    
    # Plot training time
    plt.subplot(2, 2, 3)
    plt.plot(progression_data['iteration'], progression_data['training_time'], 'g-o')
    plt.title('Training Time Progression')
    plt.xlabel('Iteration')
    plt.ylabel('Training Time (s)')
    
    # Plot hyperparameter influence
    plt.subplot(2, 2, 4)
    scatter = plt.scatter(progression_data['learning_rate'], 
                         progression_data['val_accuracy'],
                         c=progression_data['iteration'],
                         s=np.array(progression_data['batch_size'])/2,
                         alpha=0.6)
    plt.colorbar(scatter, label='Iteration')
    plt.title('Hyperparameter Influence')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy')
    
    plt.tight_layout()
    save_plot(fig, os.path.join(viz_folder, "model_evolution", "learning_curves"), 
              "learning_progression")
    
    return progression_data

def compare_model_weights(previous_model, current_model, viz_folder):
    """Compare weight changes between model iterations"""
    previous_weights = previous_model.get_weights()
    current_weights = current_model.get_weights()
    
    weight_changes = []
    for pw, cw in zip(previous_weights, current_weights):
        if pw.shape == cw.shape:
            weight_changes.append(np.mean(np.abs(cw - pw)))
    
    fig = plt.figure(figsize=(10, 5))
    plt.bar(range(len(weight_changes)), weight_changes)
    plt.title('Weight Changes Across Layers')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Absolute Weight Change')
    save_plot(fig, os.path.join(viz_folder, "model_evolution", "weight_updates"), 
              "weight_changes")
    
    return weight_changes

def safe_compare_model_weights(previous_model, current_model, viz_folder):
    """Compare weights with proper None handling"""
    if previous_model is None or current_model is None:
        print("Cannot compare weights: one or both models are None")
        return None
    
    try:
        return compare_model_weights(previous_model, current_model, viz_folder)
    except Exception as e:
        print(f"Error comparing model weights: {str(e)}")
        return None

def generate_performance_report(progression_data, viz_folder):
    """Generate comprehensive performance report across iterations"""
    report = {
        'overall_improvement': {
            'initial_accuracy': progression_data['val_accuracy'][0],
            'final_accuracy': progression_data['val_accuracy'][-1],
            'accuracy_improvement': progression_data['val_accuracy'][-1] - progression_data['val_accuracy'][0],
            'total_training_time': sum(progression_data['training_time']),
            'iterations': len(progression_data['iteration'])
        },
        'best_performance': {
            'best_accuracy': max(progression_data['val_accuracy']),
            'best_iteration': progression_data['iteration'][np.argmax(progression_data['val_accuracy'])],
            'corresponding_hyperparameters': {
                'learning_rate': progression_data['learning_rate'][np.argmax(progression_data['val_accuracy'])],
                'batch_size': progression_data['batch_size'][np.argmax(progression_data['val_accuracy'])],
                'dropout_rate': progression_data['dropout_rate'][np.argmax(progression_data['val_accuracy'])]
            }
        }
    }
    
    # Save report
    report_path = os.path.join(viz_folder, "model_evolution", "performance_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    return report

def main():
    # Create save directory if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setup visualization folder at the start
    viz_folder = setup_visualization_folder()

    # Load and preprocess data
    all_images, all_labels, all_filenames, metadata = load_and_preprocess_data()
    
    # Now we can visualize metadata with the viz_folder
    visualize_metadata(metadata, viz_folder)
    
    # Load previous best model if available
    previous_model, previous_metrics = load_previous_best_model(viz_folder)
    iteration_start = 0 if previous_metrics is None else previous_metrics['iteration'] + 1

    # Align data
    all_images_aligned, all_labels_aligned, metadata_aligned = align_data(
        all_images, all_labels, all_filenames, metadata
    )
    
    # Preprocess metadata
    metadata_features_processed, metadata_labels = preprocess_metadata(metadata_aligned)
    
    # Split dataset
    test_size = 0.3
    val_size = 0.5
    
    # Split images and metadata
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_images_aligned, all_labels_aligned, 
        test_size=test_size, random_state=42, stratify=all_labels_aligned
    )
    
    meta_X_train, meta_X_temp = train_test_split(
        metadata_features_processed, 
        test_size=test_size, random_state=42, stratify=all_labels_aligned
    )
    
    # Further split temp set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    
    meta_X_val, meta_X_test = train_test_split(
        meta_X_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    
    # Print dataset information
    print(f"\nDataset Splits:")
    print(f"Training set: {X_train.shape}, {meta_X_train.shape}, {len(y_train)}")
    print(f"Validation set: {X_val.shape}, {meta_X_val.shape}, {len(y_val)}")
    print(f"Test set: {X_test.shape}, {meta_X_test.shape}, {len(y_test)}")
    
    print("\nClass Distribution:")
    print("Training set:", Counter(y_train))
    print("Validation set:", Counter(y_val))
    print("Test set:", Counter(y_test))

    # Setup training
    hyperparameters = setup_hyperparameters()
    callbacks = setup_callbacks(save_folder)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    # Train and evaluate models
    best_val_accuracy = 0 if previous_metrics is None else previous_metrics['metrics']['val_accuracy']
    best_model = None
    best_history = None
    best_params = None

    tracking_dict = track_hyperparameter_performance()

    for i, (lr, batch_size, dropout_rate) in enumerate(hyperparameters):
        print(f"\nTraining with lr={lr}, batch_size={batch_size}, dropout_rate={dropout_rate}")
        
        # Create and compile model
        model = create_model(
            dropout_rate=dropout_rate, 
            metadata_input_shape=meta_X_train.shape[1]
        )

        # Load weights from previous best model if available
        if previous_model is not None:
            try:
                model.set_weights(previous_model.get_weights())
                print("Loaded weights from previous best model")
            except:
                print("Could not load weights from previous model, starting fresh")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name='precision'),  # Add name parameter
                tf.keras.metrics.Recall(name='recall'),       # Add name parameter
                tf.keras.metrics.AUC(name='auc')             # Add name parameter
            ]
        )

        # Train model
        start_time = time.time()
        history = train_model(
            model,
            (X_train, meta_X_train, y_train),
            (X_val, meta_X_val, y_val),
            batch_size,
            callbacks,
            class_weights
        )
        end_time = time.time()
        training_time = end_time - start_time

        # Check if this is the best model
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

        # Track metrics with consistent names
        tracking_dict['learning_rates'].append(lr)
        tracking_dict['batch_sizes'].append(batch_size)
        tracking_dict['dropout_rates'].append(dropout_rate)
        tracking_dict['val_accuracies'].append(val_accuracy)
        tracking_dict['val_losses'].append(min(history.history['val_loss']))
        tracking_dict['training_times'].append(training_time)
        tracking_dict['epochs_trained'].append(len(history.history['loss']))
        
        # Use consistent metric names
        tracking_dict['precision'].append(max(history.history['precision_1' if 'precision_1' in history.history else 'precision']))
        tracking_dict['recall'].append(max(history.history['recall_1' if 'recall_1' in history.history else 'recall']))
        tracking_dict['auc'].append(max(history.history['auc_1' if 'auc_1' in history.history else 'auc']))

        # After training, save intermediate model
        current_metrics = {
            'val_accuracy': val_accuracy,
            'val_loss': min(history.history['val_loss']),
            'training_time': training_time,
            'epochs': len(history.history['accuracy'])
        }
        
        current_params = {
            'learning_rate': lr,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate
        }
        
        save_intermediate_model(
            model, 
            current_metrics, 
            current_params, 
            iteration_start + i,
            viz_folder
        )

        # Update previous model if this one is better
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model
            best_history = history
            best_params = current_params
            previous_model = model

    # Save best hyperparameters
    with open(os.path.join(save_folder, 'best_hyperparameters.json'), 'w') as f:
        json.dump(best_params, f)

    # Evaluate best model
    test_metrics = evaluate_model(
        best_model, 
        (X_test, meta_X_test, y_test)
    )
    
    # Save test metrics
    with open(os.path.join(save_folder, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f)

    # Visualize results
    visualize_results(best_history, viz_folder)
    visualize_metadata(metadata, viz_folder)

    # Get correct test filenames
    test_filenames = get_test_filenames(all_filenames, all_labels_aligned, X_train, X_val)

    # Make predictions once
    y_pred_prob, y_pred = make_predictions(best_model, X_test, meta_X_test)
    if y_pred_prob is not None and y_pred is not None:
        # Visualize model performance
        visualize_model_performance(y_test, y_pred, y_pred_prob, viz_folder)
        
        # Save predictions
        save_predictions(y_test, y_pred, y_pred_prob, test_filenames, save_folder)

    # Save training history
    history_file = os.path.join(save_folder, 'training_history.json')
    save_training_history(best_history, history_file)

    # Compare weights safely
    weight_changes = safe_compare_model_weights(previous_model, best_model, viz_folder)

    # Visualize learning progression with error handling
    progression_data = visualize_learning_progression(viz_folder)
    if progression_data is not None:
        performance_report = generate_performance_report(progression_data, viz_folder)
    else:
        performance_report = None

    return best_model, best_history, test_metrics, performance_report

if __name__ == "__main__":
    main() 