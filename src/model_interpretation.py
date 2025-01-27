import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import cv2
import traceback

class ModelInterpreter:
    def __init__(self, model):
        self.model = model
        self.last_conv_layer = self._find_last_conv_layer()
        
    def _find_last_conv_layer(self):
        """Find the last convolutional layer in the model"""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        return None
        
    def generate_gradcam(self, image, metadata=None):
        """Generate Grad-CAM visualization"""
        try:
            # If metadata is not provided, create a zero array with correct shape
            if metadata is None:
                # Get the metadata input shape from the model
                metadata_shape = self.model.inputs[1].shape[1:]
                metadata = np.zeros((1, *metadata_shape))
            
            # Create Grad-CAM model
            grad_model = Model(
                inputs=self.model.inputs,
                outputs=[
                    self.model.get_layer(self.last_conv_layer).output, 
                    self.model.output
                ]
            )
            
            # Get predictions and gradients
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model([image, metadata])
                class_idx = tf.argmax(predictions[0])
                class_output = predictions[:, class_idx]
                
            # Calculate gradients
            grads = tape.gradient(class_output, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Generate heatmap
            conv_output = conv_output[0]
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            return heatmap.numpy()
            
        except Exception as e:
            print(f"Error generating Grad-CAM: {str(e)}")
            traceback.print_exc()
            return None
    
    def analyze_feature_importance(self, image, metadata, clinical_features):
        """Analyze importance of each feature for a specific prediction"""
        base_pred = self.model.predict([image, metadata])[0][0]
        feature_importance = {}
        
        for i, feature in enumerate(clinical_features):
            modified_metadata = metadata.copy()
            modified_metadata[0, i] = 0  # Zero out feature
            new_pred = self.model.predict([image, modified_metadata])[0][0]
            importance = abs(base_pred - new_pred)
            feature_importance[feature] = float(importance)
            
        return feature_importance
    
    def generate_uncertainty(self, image, metadata, n_iterations=10):
        """Generate prediction uncertainty using Monte Carlo Dropout"""
        try:
            predictions = []
            for _ in range(n_iterations):
                # Use model in training mode for dropout
                self.model.trainable = True
                pred = self.model([image, metadata], training=True)
                pred = pred.numpy()[0][0]
                predictions.append(float(pred))  # Convert to Python float
                
            mean_pred = float(np.mean(predictions))
            std_pred = float(np.std(predictions))
            
            return {
                'mean_prediction': mean_pred,
                'uncertainty': std_pred,
                'confidence_interval': [
                    float(mean_pred - 2*std_pred),
                    float(mean_pred + 2*std_pred)
                ]
            }
            
        except Exception as e:
            print(f"Error generating uncertainty: {str(e)}")
            traceback.print_exc()
            return {
                'mean_prediction': 0.0,
                'uncertainty': 1.0,
                'confidence_interval': [-1.0, 1.0]
            }

def safe_compare_model_weights(previous_model, current_model, viz_folder):
    """Safely compare weights between two models and return changes.
    
    Args:
        previous_model: Previous trained model or None
        current_model: Current trained model
        viz_folder: Folder to save visualization results
        
    Returns:
        dict: Weight changes between models or None if comparison not possible
    """
    try:
        if previous_model is None:
            print("No previous model available for weight comparison")
            return None
            
        weight_changes = {}
        
        # Get weights from both models
        prev_weights = previous_model.get_weights()
        curr_weights = current_model.get_weights()
        
        # Check if models have same architecture
        if len(prev_weights) != len(curr_weights):
            print("Models have different architectures, skipping weight comparison")
            return None
            
        # Compare weights layer by layer
        for i, (prev_w, curr_w) in enumerate(zip(prev_weights, curr_weights)):
            if prev_w.shape != curr_w.shape:
                continue
                
            # Calculate changes
            abs_diff = np.abs(curr_w - prev_w)
            rel_diff = abs_diff / (np.abs(prev_w) + 1e-7)  # Avoid division by zero
            
            weight_changes[f'layer_{i}'] = {
                'mean_abs_change': float(np.mean(abs_diff)),
                'max_abs_change': float(np.max(abs_diff)),
                'mean_rel_change': float(np.mean(rel_diff)),
                'max_rel_change': float(np.max(rel_diff))
            }
            
        return weight_changes
        
    except Exception as e:
        print(f"Error in weight comparison: {str(e)}")
        traceback.print_exc()
        return None 