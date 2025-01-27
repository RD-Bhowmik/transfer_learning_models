import os
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import re

# Paths to the dataset
positive_folder = "D:\\categorization\\new\\hpv\\positive"
negative_folder = "D:\\categorization\\new\\hpv\\negative"
xlsx_file_path = "D:\\categorization\\IARCImageBankColpo\\Cases Meta data.xlsx"

# Parameters
image_size = (224, 224)  # Resize dimensions

def visualize_normalization(original_image, normalized_image):
    """Visualize original and normalized images side by side with detailed changes."""
    fig = plt.figure(figsize=(15, 8))
    
    # Original Image and its details
    plt.subplot(2, 2, 1)
    plt.imshow(original_image.astype('uint8'))
    plt.title('Original Image\n(Pixel values: 0-255)')
    plt.axis('off')
    
    # Add text box with original image statistics
    stats_orig = f'Min: {original_image.min():.1f}\nMax: {original_image.max():.1f}\nMean: {original_image.mean():.1f}\nStd: {original_image.std():.1f}'
    plt.text(1.1, 0.5, stats_orig, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    # Normalized Image and its details
    plt.subplot(2, 2, 2)
    plt.imshow(normalized_image)
    plt.title('Normalized Image\n(Pixel values: 0-1)')
    plt.axis('off')
    
    # Add text box with normalized image statistics
    stats_norm = f'Min: {normalized_image.min():.3f}\nMax: {normalized_image.max():.3f}\nMean: {normalized_image.mean():.3f}\nStd: {normalized_image.std():.3f}'
    plt.text(1.1, 0.5, stats_norm, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    # Original Image Histogram
    plt.subplot(2, 2, 3)
    plt.hist(original_image.ravel(), bins=50, color='blue', alpha=0.7)
    plt.title('Original Pixel Distribution')
    plt.xlabel('Pixel Value (0-255)')
    plt.ylabel('Frequency')
    
    # Normalized Image Histogram
    plt.subplot(2, 2, 4)
    plt.hist(normalized_image.ravel(), bins=50, color='green', alpha=0.7)
    plt.title('Normalized Pixel Distribution')
    plt.xlabel('Pixel Value (0-1)')
    plt.ylabel('Frequency')
    
    plt.suptitle('Image Normalization Analysis\nDividing all pixel values by 255 to scale to [0,1] range', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Print detailed explanation
    print("\nNormalization Process Explanation:")
    print("1. Original Image:")
    print(f"   - Pixel value range: [{original_image.min():.1f}, {original_image.max():.1f}]")
    print(f"   - Mean pixel value: {original_image.mean():.1f}")
    print(f"   - Standard deviation: {original_image.std():.1f}")
    print("\n2. Normalized Image:")
    print(f"   - Pixel value range: [{normalized_image.min():.3f}, {normalized_image.max():.3f}]")
    print(f"   - Mean pixel value: {normalized_image.mean():.3f}")
    print(f"   - Standard deviation: {normalized_image.std():.3f}")
    print("\n3. Changes Made:")
    print("   - All pixel values divided by 255")
    print("   - This scales the values from [0-255] to [0-1]")
    print("   - Helps in model training by making all pixel values smaller and more manageable")

def load_images_from_folder(folder_path, target_size=(224, 224), label=None):
    """Load and preprocess images from a folder"""
    images = []
    filenames = []
    labels = []
    
    print(f"Loading images from {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            try:
                img_path = os.path.join(folder_path, filename)
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img)
                img_array = preprocess_input(img_array)
                
                images.append(img_array)
                filenames.append(filename)
                if label is not None:
                    labels.append(label)
            except Exception as e:
                print(f"Error loading image {filename}: {str(e)}")
    
    if label is not None:
        return np.array(images), np.array(labels), filenames
    return np.array(images), filenames

def load_metadata(xlsx_path):
    """Load and preprocess metadata from Excel file"""
    try:
        # Read the first row to get column names
        header_df = pd.read_excel(xlsx_path, nrows=0)
        
        # Read the actual data, skipping the header row
        metadata = pd.read_excel(xlsx_path, skiprows=[0])
        print(f"Loaded metadata with shape: {metadata.shape}")
        
        # Create clean metadata DataFrame
        clean_metadata = pd.DataFrame()
        
        # Map the first three columns correctly
        clean_metadata['CaseID'] = metadata.iloc[:, 1].astype(str).str.strip()  # Second column is CaseID
        clean_metadata['HPV'] = metadata.iloc[:, 2].astype(str).str.strip()  # Third column is HPV
        
        # Clean HPV status and create Label
        clean_metadata['HPV'] = clean_metadata['HPV'].str.lower()
        clean_metadata['Label'] = clean_metadata['HPV'].map({'positive': 1, 'negative': 0})
        
        # Map clinical features
        column_mapping = {
            3: 'Adequacy',  # Column index 3 is Adequacy
            5: 'Squamocolumnar junction visibility',
            6: 'Transformation zone',
            7: 'Original squamous epithelium',
            8: 'Columnar epithelium',
            9: 'Metaplastic squamous epithelium',
            11: 'Location of the lesion',
            18: 'Grade 1',
            22: 'Grade 2',
            29: 'Suspicious for invasion',
            34: 'Aceto uptake',
            35: 'Margins',
            36: 'Vessels',
            37: 'Lesion size',
            38: 'Iodine uptake',
            39: 'SwedeFinal'
        }
        
        # Add clinical features using column indices
        for col_idx, feature_name in column_mapping.items():
            if col_idx < len(metadata.columns):
                clean_metadata[feature_name] = metadata.iloc[:, col_idx]
        
        # Remove rows with missing CaseID or Label
        clean_metadata = clean_metadata.dropna(subset=['CaseID', 'Label']).reset_index(drop=True)
        
        # Convert Label to int explicitly
        clean_metadata['Label'] = clean_metadata['Label'].astype('Int64')
        
        # Handle missing values in other columns
        for col in clean_metadata.columns:
            if col not in ['CaseID', 'HPV', 'Label']:
                if pd.api.types.is_numeric_dtype(clean_metadata[col]):
                    clean_metadata[col] = clean_metadata[col].fillna(clean_metadata[col].mean())
                else:
                    clean_metadata[col] = clean_metadata[col].fillna(clean_metadata[col].mode().iloc[0])
        
        print(f"\nProcessed metadata with shape: {clean_metadata.shape}")
        print(f"Label distribution:\n{clean_metadata['Label'].value_counts()}")
        
        # Add debug information
        print("\nSample of processed metadata:")
        print(clean_metadata[['CaseID', 'HPV', 'Label']].head())
        print("\nAvailable clinical features:")
        print([col for col in clean_metadata.columns if col not in ['CaseID', 'HPV', 'Label']])
        
        # Verify data types
        print("\nColumn data types:")
        print(clean_metadata.dtypes)
        
        return clean_metadata
        
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        print(f"Excel file path: {xlsx_path}")
        if 'metadata' in locals():
            print("\nFirst few rows of raw metadata:")
            print(metadata.head())
            print("\nColumn indices and names:")
            for i, col in enumerate(metadata.columns):
                print(f"{i}: {col}")
        raise

def display_dataset_info(all_images, all_labels, metadata):
    """Display information about the dataset"""
    print("\nDataset Information:")
    print(f"Number of images: {len(all_images)}")
    print(f"Image shape: {all_images[0].shape}")
    print(f"Number of labels: {len(all_labels)}")
    print(f"Label distribution: {np.bincount(all_labels)}")
    print(f"Metadata shape: {metadata.shape}")
    print("\nMetadata columns:")
    for col in metadata.columns:
        print(f"- {col}")

def visualize_metadata(metadata):
    """Visualize key information from the metadata."""
    plt.figure(figsize=(15, 10))
    
    # 1. HPV Status Distribution
    plt.subplot(2, 2, 1)
    hpv_counts = metadata['Label'].value_counts()
    plt.pie(hpv_counts, labels=['Negative', 'Positive'], autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
    plt.title('HPV Status Distribution')
    
    # 2. Age Distribution (if available)
    age_col = next((col for col in metadata.columns if 'age' in col.lower()), None)
    if age_col:
        plt.subplot(2, 2, 2)
        metadata[age_col].hist(bins=20, color='skyblue')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Count')
    
    # 3. Other numerical columns distribution
    numerical_cols = metadata.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'Label']
    
    if len(numerical_cols) > 0:
        plt.subplot(2, 2, 3)
        metadata[numerical_cols[:3]].boxplot()  # Show first 3 numerical columns
        plt.xticks(rotation=45)
        plt.title('Numerical Features Distribution')
    
    # 4. Categorical data distribution (if available)
    categorical_cols = metadata.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        plt.subplot(2, 2, 4)
        cat_col = categorical_cols[0]  # Take first categorical column
        cat_counts = metadata[cat_col].value_counts()[:5]  # Top 5 categories
        cat_counts.plot(kind='bar')
        plt.title(f'Top 5 Categories in {cat_col}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Additional correlation heatmap for numerical columns
    numerical_cols = metadata.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation = metadata[numerical_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()

def debug_case_summary(text):
    """Debug function to understand Case Summary content"""
    print("\nAnalyzing Case Summary:")
    print(f"Raw text: {text}")
    case_match = re.search(r'(?i)case\s*(\d+)', str(text))
    print(f"Case match: {case_match.group(0) if case_match else 'None'}")
    print(f"Contains 'positive': {'positive' in str(text).lower()}")
    print(f"Contains 'negative': {'negative' in str(text).lower()}")

def main():
    # Load images
    positive_images, positive_labels, positive_filenames = load_images_from_folder(positive_folder, label=1)
    negative_images, negative_labels, negative_filenames = load_images_from_folder(negative_folder, label=0)
    
    # Combine datasets
    all_images = np.concatenate([positive_images, negative_images])
    all_labels = np.concatenate([positive_labels, negative_labels])
    all_filenames = positive_filenames + negative_filenames
    
    # Load metadata
    metadata = load_metadata(xlsx_file_path)
    
    # Display information about the dataset
    display_dataset_info(all_images, all_labels, metadata)
    
    # Add visualization of metadata
    print("\nGenerating metadata visualizations...")
    visualize_metadata(metadata)
    
    return all_images, all_labels, all_filenames, metadata

if __name__ == "__main__":
    all_images, all_labels, all_filenames, metadata = main() 