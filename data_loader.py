import os
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to the dataset
positive_folder = "D:\\categorization\\new\\hpv\\positive"
negative_folder = "D:\\categorization\\new\\hpv\\negative"
xlsx_file_path = "D:\\categorization\\IARCImageBankColpo\\Cases Meta data.xlsx"

# Parameters
image_size = (224, 224)  # Resize dimensions

def load_images_from_folder(folder, label):
    """Load images from a folder and assign labels."""
    images, labels, filenames = [], [], []
    print(f"\nLoading images from {folder}")
    
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            img = tf.keras.utils.load_img(filepath, target_size=image_size)
            img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normalize pixel values
            images.append(img_array)
            labels.append(label)
            filenames.append(filename)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    
    return images, labels, filenames

def load_metadata(xlsx_path):
    """Load and preprocess metadata from Excel file."""
    print(f"\nLoading metadata from {xlsx_path}")
    
    # Load metadata with header at row 1
    metadata = pd.read_excel(xlsx_path, header=1)
    
    # Find important columns
    caseid_col = next((col for col in metadata.columns if 'caseid' in col.lower()), None)
    hpv_col = next((col for col in metadata.columns if 'hpv' in col.lower()), None)
    
    if not caseid_col or not hpv_col:
        raise KeyError("Required columns (CaseID, HPV status) not found in metadata")
    
    # Rename columns for consistency
    metadata = metadata.rename(columns={caseid_col: "CaseID", hpv_col: "Label"})
    
    # Convert HPV status to binary
    metadata["Label"] = metadata["Label"].apply(lambda x: 1 if str(x).strip().lower() == "positive" else 0)
    
    return metadata

def display_dataset_info(images, labels, filenames, metadata):
    """Display information about the loaded dataset."""
    print("\nDataset Summary:")
    print(f"Total images loaded: {len(images)}")
    print(f"Label distribution: {Counter(labels)}")
    print(f"\nMetadata shape: {metadata.shape}")
    print("\nMetadata columns:", metadata.columns.tolist())
    
    # Find indices for positive and negative samples
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]
    
    # Display sample images (3 positive and 3 negative)
    plt.figure(figsize=(15, 5))
    
    # Show positive samples
    for i in range(3):
        plt.subplot(2, 3, i + 1)
        idx = positive_indices[i] if i < len(positive_indices) else 0
        plt.imshow(images[idx])
        plt.title(f"Positive Sample {i+1}")
        plt.axis('off')
    
    # Show negative samples
    for i in range(3):
        plt.subplot(2, 3, i + 4)
        idx = negative_indices[i] if i < len(negative_indices) else 0
        plt.imshow(images[idx])
        plt.title(f"Negative Sample {i+1}")
        plt.axis('off')
    
    plt.suptitle("Sample Images from Dataset (Top: Positive, Bottom: Negative)")
    plt.tight_layout()
    plt.show()

    # Print more detailed information
    print("\nDetailed Dataset Information:")
    print(f"Number of positive samples: {len(positive_indices)}")
    print(f"Number of negative samples: {len(negative_indices)}")

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

def main():
    # Load images
    positive_images, positive_labels, positive_filenames = load_images_from_folder(positive_folder, label=1)
    negative_images, negative_labels, negative_filenames = load_images_from_folder(negative_folder, label=0)
    
    # Combine datasets
    all_images = positive_images + negative_images
    all_labels = positive_labels + negative_labels
    all_filenames = positive_filenames + negative_filenames
    
    # Load metadata
    metadata = load_metadata(xlsx_file_path)
    
    # Convert lists to numpy arrays
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    
    # Display information about the dataset
    display_dataset_info(all_images, all_labels, all_filenames, metadata)
    
    # Add visualization of metadata
    print("\nGenerating metadata visualizations...")
    visualize_metadata(metadata)
    
    return all_images, all_labels, all_filenames, metadata

if __name__ == "__main__":
    all_images, all_labels, all_filenames, metadata = main() 