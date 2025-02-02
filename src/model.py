import tensorflow as tf
from tensorflow.keras import regularizers, layers, Model, Input
import numpy as np
import json
import os
import time
from sklearn.utils.class_weight import compute_class_weight
import datetime
import traceback
from tqdm import tqdm  # Add at the top of the file

class CervicalCancerModel:
    """Enhanced model for cervical cancer detection"""
    
    def __init__(self, image_size=(224, 224, 3), n_clinical_features=None):
        self.image_size = image_size
        self.n_clinical_features = n_clinical_features
        
    def build_model(self):
        """Build the enhanced model architecture"""
        # Image pathway
        image_input = layers.Input(shape=self.image_size, name='image_input')
        image_features = self._build_image_pathway(image_input)
        
        # Clinical pathway
        if self.n_clinical_features:
            clinical_input = layers.Input(shape=(self.n_clinical_features,), 
                                        name='clinical_input')
            clinical_features = self._build_clinical_pathway(clinical_input)
            
            # Combine pathways
            combined_features = self._combine_features(image_features, clinical_features)
            outputs = self._build_classification_head(combined_features)
            
            model = Model(inputs=[image_input, clinical_input], outputs=outputs)
        else:
            outputs = self._build_classification_head(image_features)
            model = Model(inputs=image_input, outputs=outputs)
        
        return model
    
    def _build_image_pathway(self, inputs):
        """Enhanced image processing pathway"""
        # Base model with pre-trained weights
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        
        # Add attention mechanism
        x = base_model.output
        attention = layers.Conv2D(1, 1)(x)
        attention = layers.Activation('sigmoid')(attention)
        x = layers.Multiply()([x, attention])
        
        # Multi-scale analysis
        scales = [1, 2, 4]
        multi_scale_features = []
        
        for scale in scales:
            if scale == 1:
                scaled_features = x
            else:
                scaled_features = layers.AveragePooling2D(scale)(x)
            
            features = layers.Conv2D(256//scale, 3, padding='same')(scaled_features)
            features = layers.BatchNormalization()(features)
            features = layers.ReLU()(features)
            
            if scale != 1:
                features = layers.UpSampling2D(scale)(features)
            
            multi_scale_features.append(features)
        
        # Combine multi-scale features
        x = layers.Concatenate()(multi_scale_features)
        x = layers.Conv2D(256, 1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Global features
        x = layers.GlobalAveragePooling2D()(x)
        
        return x
    
    def _build_clinical_pathway(self, inputs):
        """Enhanced clinical feature processing"""
        # Initial dense layers
        x = layers.Dense(128)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.3)(x)
        
        # Multiple pathways for different feature types
        dense_features = layers.Dense(64, activation='relu')(x)
        
        # Categorical features pathway
        categorical = layers.Dense(32, activation='relu')(x)
        categorical = layers.Dense(16, activation='relu')(categorical)
        
        # Numerical features pathway
        numerical = layers.Dense(32, activation='relu')(x)
        numerical = layers.Dense(16, activation='relu')(numerical)
        
        # Combine all pathways
        combined = layers.Concatenate()([dense_features, categorical, numerical])
        
        return combined
    
    def _combine_features(self, image_features, clinical_features):
        """Enhanced feature combination"""
        # Cross-attention mechanism
        attention_weights = layers.Dense(1, activation='sigmoid')(clinical_features)
        weighted_image_features = layers.Multiply()([image_features, attention_weights])
        
        # Combine features
        combined = layers.Concatenate()([weighted_image_features, clinical_features])
        
        # Additional processing
        x = layers.Dense(256)(combined)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.4)(x)
        
        return x
    
    def _build_classification_head(self, features):
        """Enhanced classification head"""
        # Multiple dense layers with residual connections
        x = features
        for units in [128, 64, 32]:
            residual = x
            x = layers.Dense(units)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(0.3)(x)
            
            # Add residual if shapes match
            if residual.shape[-1] == units:
                x = layers.Add()([x, residual])
        
        # Final classification
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        return outputs

    def _cross_attention(self, image_features, clinical_features):
        """Implement cross-attention between image and clinical features"""
        # Project clinical features to match image feature dimensions
        clinical_projection = layers.Dense(image_features.shape[-1])(clinical_features)
        clinical_projection = layers.Reshape((1, 1, -1))(clinical_projection)
        clinical_projection = layers.UpSampling2D(
            size=(image_features.shape[1], image_features.shape[2])
        )(clinical_projection)
        
        # Calculate attention weights
        attention = layers.Multiply()([image_features, clinical_projection])
        attention = layers.Conv2D(1, 1, activation='sigmoid')(attention)
        
        # Apply attention to image features
        attended_features = layers.Multiply()([image_features, attention])
        
        return attended_features

    def _gated_fusion(self, image_features, clinical_features):
        """Implement gated fusion of image and clinical features"""
        # Create gates for both modalities
        image_gate = layers.Dense(image_features.shape[-1], activation='sigmoid')(clinical_features)
        clinical_gate = layers.Dense(clinical_features.shape[-1], activation='sigmoid')(
            layers.GlobalAveragePooling2D()(image_features)
        )
        
        # Apply gates
        gated_image = layers.Multiply()([image_features, image_gate])
        gated_clinical = layers.Multiply()([clinical_features, clinical_gate])
        
        # Combine features
        fused_features = layers.Concatenate()([
            layers.GlobalAveragePooling2D()(gated_image),
            gated_clinical
        ])
        
        return fused_features

    def build_multi_task_heads(self, features):
        """Build multiple task-specific heads"""
        # Cancer classification head
        cancer_pred = layers.Dense(256, activation='relu')(features)
        cancer_pred = layers.Dropout(0.3)(cancer_pred)
        cancer_pred = layers.Dense(1, activation='sigmoid', name='cancer_prediction')(cancer_pred)
        
        # Lesion type classification
        lesion_type = layers.Dense(128, activation='relu')(features)
        lesion_type = layers.Dropout(0.3)(lesion_type)
        lesion_type = layers.Dense(4, activation='softmax', name='lesion_type')(lesion_type)
        
        # Severity score regression
        severity_score = layers.Dense(128, activation='relu')(features)
        severity_score = layers.Dropout(0.3)(severity_score)
        severity_score = layers.Dense(1, activation='linear', name='severity_score')(severity_score)
        
        return [cancer_pred, lesion_type, severity_score]

def create_model(dropout_rate, learning_rate, image_input_shape=(224, 224, 3), metadata_input_shape=None):
    """Create hybrid model combining image and clinical features
    
    Args:
        dropout_rate (float): Dropout rate for regularization
        learning_rate (float): Learning rate for optimizer
        image_input_shape (tuple): Shape of input images (height, width, channels)
        metadata_input_shape (int, optional): Number of clinical features
    """
    # Image input branch
    image_input = tf.keras.Input(shape=image_input_shape, name="image_input")
    x = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=image_input
    ).output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    image_features = tf.keras.layers.Dense(256, activation='relu')(x)

    # Clinical features branch
    if metadata_input_shape:
        metadata_input = tf.keras.Input(shape=(metadata_input_shape,), name="metadata_input")
        m = tf.keras.layers.Dense(128, activation='relu')(metadata_input)
        m = tf.keras.layers.Dropout(dropout_rate)(m)
        m = tf.keras.layers.Dense(64, activation='relu')(m)
        clinical_features = tf.keras.layers.Dense(32, activation='relu')(m)
        
        # Combine image and clinical features
        combined = tf.keras.layers.Concatenate()([image_features, clinical_features])
        combined = tf.keras.layers.Dense(128, activation='relu')(combined)
        combined = tf.keras.layers.Dropout(dropout_rate)(combined)
        combined = tf.keras.layers.Dense(64, activation='relu')(combined)
        
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(combined)
        model = tf.keras.Model(inputs=[image_input, metadata_input], outputs=outputs)
    else:
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(image_features)
        model = tf.keras.Model(inputs=image_input, outputs=outputs)
    
    # Compile model with provided learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
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

def train_model(model, train_data, val_data, batch_size, callbacks, class_weights):
    """Train the model with given parameters"""
    X_train, meta_X_train, y_train = train_data
    X_val, meta_X_val, y_val = val_data
    
    history = model.fit(
        [X_train, meta_X_train],
        y_train,
        validation_data=([X_val, meta_X_val], y_val),
        epochs=10,
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

def make_predictions(model, X_test, meta_X_test):
    """Make predictions with error handling"""
    try:
        y_pred_prob = model.predict([X_test, meta_X_test])
        y_pred = (y_pred_prob > 0.5).astype(int)
        return y_pred_prob, y_pred
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return None, None

def save_intermediate_model(model, metrics, parameters, iteration, viz_folder):
    """Save intermediate model and its metrics during training"""
    try:
        # Create directory structure
        model_folder = os.path.join(viz_folder, "model_evolution", "intermediate_models")
        os.makedirs(model_folder, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_folder, f"model_iteration_{iteration}.keras")
        model.save(model_path)
        
        # Prepare metrics for saving (ensure all values are JSON serializable)
        metrics_dict = {
            'iteration': iteration,
            'metrics': {
                k: float(v) for k, v in metrics.items()  # Convert numpy types to float
            },
            'parameters': parameters,
            'timestamp': str(datetime.datetime.now())
        }
        
        # Save metrics
        metrics_path = os.path.join(model_folder, f"metrics_iteration_{iteration}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
            
        print(f"Saved intermediate model and metrics for iteration {iteration}")
        
    except Exception as e:
        print(f"Error saving intermediate model: {str(e)}")
        traceback.print_exc()

def load_previous_best_model(viz_folder):
    """Load the best model from previous training if available"""
    model_folder = os.path.join(viz_folder, "model_evolution", "intermediate_models")
    if not os.path.exists(model_folder):
        return None, None
    
    model_files = [f for f in os.listdir(model_folder) if f.startswith("model_iteration_") and f.endswith(".keras")]
    if not model_files:
        return None, None
    
    latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_metrics = f"metrics_iteration_{latest_model.split('_')[-1].split('.')[0]}.json"
    
    model_path = os.path.join(model_folder, latest_model)
    metrics_path = os.path.join(model_folder, latest_metrics)
    
    model = tf.keras.models.load_model(model_path)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return model, metrics

def compute_class_weights(y_train):
    """Compute balanced class weights"""
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    return dict(enumerate(weights))

def safe_compare_model_weights(previous_model, current_model, viz_folder):
    """Safely compare weights between two models"""
    if previous_model is None or current_model is None:
        return None
        
    try:
        weight_changes = []
        for (prev_layer, curr_layer) in zip(previous_model.layers, current_model.layers):
            if len(prev_layer.get_weights()) > 0:
                prev_weights = prev_layer.get_weights()[0]
                curr_weights = curr_layer.get_weights()[0]
                diff = np.mean(np.abs(prev_weights - curr_weights))
                weight_changes.append({
                    'layer_name': prev_layer.name,
                    'weight_diff': float(diff)
                })
        
        # Save weight changes
        changes_path = os.path.join(viz_folder, 'model_evolution', 'weight_changes.json')
        with open(changes_path, 'w') as f:
            json.dump(weight_changes, f, indent=4)
            
        return weight_changes
    except Exception as e:
        print(f"Error comparing weights: {str(e)}")
        return None

def setup_hyperparameters():
    """Define hyperparameter combinations for training"""
    return {
        'learning_rates': [1e-3],        # Reduced to 2 values
        'batch_sizes': [16],               # Reduced to 2 values
        'dropout_rates': [0.2]            # Reduced to 2 values
    }

def train_and_evaluate():
    try:
        hyperparameters = setup_hyperparameters()
        
        # Calculate total combinations
        total_combinations = (
            len(hyperparameters['learning_rates']) * 
            len(hyperparameters['batch_sizes']) * 
            len(hyperparameters['dropout_rates'])
        )
        
        # Initialize tracking
        best_val_accuracy = 0
        best_model = None
        best_history = None
        best_params = None
        
        print(f"\nStarting hyperparameter search with {total_combinations} combinations...")
        
        # Create combinations list for tqdm
        combinations = [
            (lr, bs, dr) 
            for lr in hyperparameters['learning_rates']
            for bs in hyperparameters['batch_sizes']
            for dr in hyperparameters['dropout_rates']
        ]
        
        # Training loop with progress bar
        for lr, batch_size, dropout_rate in tqdm(combinations, 
                                               desc="Hyperparameter Search",
                                               total=total_combinations):
            print(f"\nTrying: lr={lr}, batch_size={batch_size}, dropout_rate={dropout_rate}")
            
            # Create and train model...
            model = create_model(
                dropout_rate=dropout_rate,
                learning_rate=lr,
                metadata_input_shape=meta_X_train.shape[1]
            )
            
            # Add early stopping with more aggressive parameters
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,                # Reduced from 5 to 3
                min_delta=0.01,           # Minimum improvement required
                restore_best_weights=True
            )
            
            # Add reduce learning rate on plateau
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.2,
                patience=2,
                min_lr=1e-6
            )
            
            callbacks = [early_stopping, reduce_lr]
            
            # Train and evaluate
            history = train_model(model, train_data, val_data, batch_size, callbacks, class_weights)
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
                print(f"\nNew best model found!")
                print(f"Validation accuracy: {val_accuracy:.4f}")
                print(f"Parameters: {best_params}")
            
            print(f"Best validation accuracy so far: {best_val_accuracy:.4f}")

        print("\nHyperparameter search completed!")
        print("\nBest hyperparameters found:")
        print(f"Learning rate: {best_params['learning_rate']}")
        print(f"Batch size: {best_params['batch_size']}")
        print(f"Dropout rate: {best_params['dropout_rate']}")
        
        return best_model, best_history, test_metrics, results
        
    except Exception as e:
        print(f"Error during hyperparameter search: {str(e)}")
        traceback.print_exc()

def create_stage_specific_model(dropout_rate=0.3, learning_rate=1e-4, metadata_input_shape=296):
    # Image pathway
    image_input = Input(shape=(224, 224, 3), name="image_input")
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=image_input
    )
    
    # Add attention mechanism for image features
    x = base_model.output
    attention = layers.Conv2D(1, 1)(x)
    attention = layers.Activation('sigmoid')(attention)
    x = layers.Multiply()([x, attention])
    x = layers.GlobalAveragePooling2D()(x)
    
    # Enhanced image feature processing
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    
    # Metadata pathway with feature importance
    meta_input = Input(shape=(metadata_input_shape,), name="meta_input")
    meta_x = layers.Dense(256, activation="relu")(meta_input)
    meta_x = layers.BatchNormalization()(meta_x)
    meta_x = layers.Dropout(dropout_rate)(meta_x)
    meta_x = layers.Dense(128, activation="relu")(meta_x)
    
    # Advanced feature fusion
    combined = layers.concatenate([x, meta_x])
    combined_x = layers.Dense(256, activation="relu")(combined)
    combined_x = layers.BatchNormalization()(combined_x)
    combined_x = layers.Dropout(dropout_rate)(combined_x)
    
    # Multi-head output for better stage discrimination
    shared = layers.Dense(128, activation="relu")(combined_x)
    
    # Stage-specific outputs
    output = layers.Dense(3, activation="softmax", name="stage_output")(shared)
    
    model = Model(inputs=[image_input, meta_input], outputs=output)
    
    # Add class weights to handle imbalance
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.F1Score(name='f1')
        ]
    )
    
    return model

if __name__ == "__main__":
    # Add some basic testing code
    print("Testing model module...")
    test_model = create_model(
        dropout_rate=0.5,
        learning_rate=1e-3,
        metadata_input_shape=10
    )
    print("Model created successfully!")
    print(f"Model input shapes: {[input.shape for input in test_model.inputs]}")
    print(f"Model output shape: {test_model.outputs[0].shape}") 