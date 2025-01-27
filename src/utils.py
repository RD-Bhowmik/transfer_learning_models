import os
import json
import numpy as np
from datetime import datetime

def setup_visualization_folder(save_folder):
    """Create organized folders for saving visualizations"""
    viz_folder = os.path.join(save_folder, "visualizations")
    
    # Only create folders we actually use
    folders = {
        "model_evolution": [
            "intermediate_models"  # For model checkpoints and metrics
        ],
        "training_progress": [
            "learning_curves"  # For accuracy/loss plots
        ],
        "performance_metrics": [
            "confusion_matrices",  # For confusion matrix plots
            "roc_curves",         # For ROC and PR curves
        ],
        "metadata_analysis": [
            "distributions"       # For data distribution plots
        ]
    }
    
    created_folders = {}
    # Create main folder and required subfolders
    for main_folder, sub_folders in folders.items():
        main_path = os.path.join(viz_folder, main_folder)
        if not os.path.exists(main_path):
            os.makedirs(main_path)
        created_folders[main_folder] = main_path
        
        for sub_folder in sub_folders:
            sub_path = os.path.join(main_path, sub_folder)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
            created_folders[f"{main_folder}/{sub_folder}"] = sub_path
    
    print("\nCreated output folders:")
    for folder_name, folder_path in created_folders.items():
        print(f"- {folder_name}")
    
    return viz_folder

def save_json(data, filepath):
    """Save data to JSON file with error handling"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)
        return True
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")
        return False

def load_json(filepath):
    """Load data from JSON file with error handling"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        return None

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def create_experiment_folder(base_folder):
    """Create a new experiment folder with a timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = os.path.join(base_folder, f"experiment_{timestamp}")
    os.makedirs(experiment_folder, exist_ok=True)
    return experiment_folder

def ensure_folder_exists(folder_path):
    """Ensure folder exists, create if it doesn't"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def get_latest_checkpoint(checkpoint_dir):
    """Get the latest checkpoint from directory"""
    try:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.keras')]
        if not checkpoints:
            return None
        latest = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
        return os.path.join(checkpoint_dir, latest)
    except Exception as e:
        print(f"Error getting latest checkpoint: {str(e)}")
        return None

def log_training_info(log_file, epoch, metrics):
    """Log training information to file"""
    try:
        with open(log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            f.write(f"{timestamp} - Epoch {epoch}: {metrics_str}\n")
        return True
    except Exception as e:
        print(f"Error logging training info: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing utils module...")
    
    # Test folder creation
    test_folder = "test_utils"
    viz_folder = setup_visualization_folder(test_folder)
    print(f"Created visualization folders in: {viz_folder}")
    
    # Test JSON saving/loading with correct numpy types
    test_data = {
        "numpy_int": np.int64(42),
        "numpy_float": np.float64(3.14),
        "numpy_array": np.array([1, 2, 3], dtype=np.int64),
        "regular_list": [4, 5, 6]
    }
    
    # Ensure test folder exists
    os.makedirs(test_folder, exist_ok=True)
    
    # Test JSON operations
    json_path = os.path.join(test_folder, "test_data.json")
    save_success = save_json(test_data, json_path)
    print(f"\nJSON save successful: {save_success}")
    
    if save_success:
        loaded_data = load_json(json_path)
        print(f"Loaded data: {loaded_data}")
        
        # Verify data integrity
        if loaded_data:
            print("\nData verification:")
            print(f"Integer value: {loaded_data['numpy_int']} (type: {type(loaded_data['numpy_int'])})")
            print(f"Float value: {loaded_data['numpy_float']} (type: {type(loaded_data['numpy_float'])})")
            print(f"Array: {loaded_data['numpy_array']} (type: {type(loaded_data['numpy_array'])})")
            print(f"List: {loaded_data['regular_list']} (type: {type(loaded_data['regular_list'])})")
    
    # Test experiment folder creation
    exp_folder = create_experiment_folder(test_folder)
    print(f"\nCreated experiment folder: {exp_folder}") 