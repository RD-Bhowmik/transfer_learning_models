# Create a new folder structure for the refined code
import os

# Base paths
REFINED_CODE_PATH = "D:\\transfer_learning_models"
DATA_PATH = "D:\\categorization"

# Create main project directory
if not os.path.exists(REFINED_CODE_PATH):
    os.makedirs(REFINED_CODE_PATH)

# Create source directory
SRC_PATH = os.path.join(REFINED_CODE_PATH, "src")
if not os.path.exists(SRC_PATH):
    os.makedirs(SRC_PATH)

# Create empty __init__.py in src directory
init_file = os.path.join(SRC_PATH, "__init__.py")
if not os.path.exists(init_file):
    open(init_file, 'a').close()

# Dataset paths
POSITIVE_FOLDER = os.path.join(DATA_PATH, "new", "hpv", "positive")
NEGATIVE_FOLDER = os.path.join(DATA_PATH, "new", "hpv", "negative")
METADATA_PATH = os.path.join(DATA_PATH, "IARCImageBankColpo", "Cases Meta data.xlsx")
SAVE_FOLDER = "path/to/save/folder"  # Update this path as needed

# Create output directory
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Print structure for verification
print("Created folder structure:")
print(f"├── {REFINED_CODE_PATH}")
print(f"│   ├── src/")
print(f"│   │   └── __init__.py")
print(f"│   └── outputs/")

# Create necessary directories
def create_directories():
    """Create all necessary directories"""
    directories = [
        REFINED_CODE_PATH,
        SAVE_FOLDER
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

# Create directories when config is imported
create_directories()