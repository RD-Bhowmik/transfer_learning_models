import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from collections import Counter
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to the dataset
positive_folder = "D:\\categorization\\new\\hpv\\positive"
negative_folder = "D:\\categorization\\new\\hpv\\negative"
xlsx_file_path = "D:\\categorization\\IARCImageBankColpo\\Cases Meta data.xlsx"

# Parameters
image_size = (224, 224)  # Resize dimensions


def load_images_from_folder(folder, label):
    images, labels, filenames = [], [], []
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

# Load positive and negative images
positive_images, positive_labels, positive_filenames = load_images_from_folder(positive_folder, label=1)
negative_images, negative_labels, negative_filenames = load_images_from_folder(negative_folder, label=0)

# Combine all images and labels
all_images = np.array(positive_images + negative_images)
all_labels = np.array(positive_labels + negative_labels)
all_filenames = positive_filenames + negative_filenames

print(f"Loaded {len(all_images)} images: {Counter(all_labels)}")

# Load and preprocess metadata
metadata = pd.read_excel(xlsx_file_path, header=1)

# Identify columns in metadata
def find_column(possible_names):
    for name in possible_names:
        matching_cols = [col for col in metadata.columns if name.lower() in col.lower()]
        if matching_cols:
            return matching_cols[0]
    return None

caseid_col = find_column(['CaseID', 'Case ID', 'Case_ID'])
hpv_col = find_column(['HPV', 'HPV Status', 'HPVStatus'])

if not hpv_col or not caseid_col:
    raise KeyError("Required columns (CaseID, HPV status) are missing from the metadata.")

# Rename columns for consistency
metadata = metadata.rename(columns={caseid_col: "CaseID", hpv_col: "Label"})

# Convert labels to binary
metadata["Label"] = metadata["Label"].apply(lambda x: 1 if str(x).strip().lower() == "positive" else 0)

# Handle missing values
for column in metadata.columns:
    if pd.api.types.is_numeric_dtype(metadata[column]):
        metadata[column].fillna(metadata[column].mean(), inplace=True)
    elif pd.api.types.is_object_dtype(metadata[column]):
        metadata[column].fillna(metadata[column].mode().iloc[0], inplace=True)

# Separate metadata features and labels
metadata_features = metadata.drop(columns=["Label", "CaseID"], errors="ignore")
metadata_labels = metadata["Label"]

# Preprocess metadata features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), metadata_features.select_dtypes(include=["float64", "int64"]).columns),
        ('cat', OneHotEncoder(handle_unknown="ignore"), metadata_features.select_dtypes(include=["object"]).columns)
    ]
)
metadata_features_processed = preprocessor.fit_transform(metadata_features).toarray()

print(f"Processed metadata shape: {metadata_features_processed.shape}")

metadata["CaseID"] = metadata["CaseID"].str.strip().str.lower()

# Function to extract CaseID from filenames
def extract_case_id_from_filename(filename):
    # Remove the file extension and trailing digits
    case_id = filename.split('.')[0]  # Remove file extension
    case_id = ''.join([char for char in case_id if not char.isdigit()])  # Remove digits
    return case_id.strip().lower()  # Ensure lowercase and clean up

# Extract CaseIDs from image filenames
image_case_ids = [extract_case_id_from_filename(fname) for fname in all_filenames]

# Debugging Metadata and Image Alignment
print(f"Number of CaseIDs in metadata: {len(metadata['CaseID'].unique())}")
print(f"Number of CaseIDs in image dataset: {len(image_case_ids)}")
print(f"CaseIDs in metadata not found in image dataset: {set(metadata['CaseID']) - set(image_case_ids)}")
print(f"CaseIDs in image dataset not found in metadata: {set(image_case_ids) - set(metadata['CaseID'])}")

# Filter metadata to include only matching CaseIDs
metadata = metadata[metadata["CaseID"].isin(image_case_ids)].reset_index(drop=True)

# Align images, labels, and metadata
aligned_indices = [
    image_case_ids.index(case_id) for case_id in metadata["CaseID"] if case_id in image_case_ids
]
all_images_aligned = all_images[aligned_indices]
all_labels_aligned = all_labels[aligned_indices]
metadata_features_aligned = metadata_features_processed[:len(metadata)]  # Match filtered metadata size

# Debugging Alignment Results
print(f"Aligned dataset shape: {all_images_aligned.shape}, {metadata_features_aligned.shape}")
print(f"Number of aligned labels: {len(all_labels_aligned)}")

from sklearn.model_selection import train_test_split
from collections import Counter

# Parameters for splitting
test_size = 0.3  # 30% for testing and validation
val_size = 0.5   # Half of the test set for validation

# Split the dataset into training and temp (test + validation) sets
X_train, X_temp, y_train, y_temp = train_test_split(
    all_images_aligned, all_labels_aligned, test_size=test_size, random_state=42, stratify=all_labels_aligned
)

# Split the metadata in alignment with the image splits
meta_X_train, meta_X_temp = train_test_split(
    metadata_features_aligned, test_size=test_size, random_state=42, stratify=all_labels_aligned
)

# Further split temp set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
)

# Split metadata for validation and test sets
meta_X_val, meta_X_test = train_test_split(
    meta_X_temp, test_size=val_size, random_state=42, stratify=y_temp
)

# Print dataset shapes
print(f"Training set: {X_train.shape}, {meta_X_train.shape}, {len(y_train)}")
print(f"Validation set: {X_val.shape}, {meta_X_val.shape}, {len(y_val)}")
print(f"Test set: {X_test.shape}, {meta_X_test.shape}, {len(y_test)}")

# Check class distribution in each set
print("Class distribution in training set:", Counter(y_train))
print("Class distribution in validation set:", Counter(y_val))
print("Class distribution in test set:", Counter(y_test))

from itertools import product

# Define hyperparameter values
learning_rates = [1e-3, 1e-4, 1e-5]
batch_sizes = [16, 32, 64]
dropout_rates = [0.2, 0.3, 0.5]

# Create a grid of hyperparameter combinations
hyperparameter_combinations = list(product(learning_rates, batch_sizes, dropout_rates))

# Function to create the model with different dropout rates
from tensorflow.keras import regularizers

def create_model(dropout_rate, image_input_shape=(224, 224, 3), metadata_input_shape=(296,)):
    # Image Input
    image_input = tf.keras.Input(shape=image_input_shape, name="image_input")
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.01))(image_input)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Metadata Input
    meta_input = tf.keras.Input(shape=metadata_input_shape, name="meta_input")
    meta_x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(meta_input)
    meta_x = tf.keras.layers.Dropout(dropout_rate)(meta_x)

    # Concatenate Image and Metadata Features
    combined = tf.keras.layers.Concatenate()([x, meta_x])
    combined = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(combined)
    combined = tf.keras.layers.Dropout(dropout_rate)(combined)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(combined)

    # Define the Model
    model = tf.keras.Model(inputs=[image_input, meta_input], outputs=output)
    return model

# Load or initialize best hyperparameters
def load_best_hyperparameters():
    if os.path.exists("best_hyperparameters.json") and os.path.getsize("best_hyperparameters.json") > 0:
        with open("best_hyperparameters.json", "r") as f:
            try:
                hyperparams = json.load(f)
                print(f"Loaded hyperparameters: {hyperparams}")
                return hyperparams
            except json.JSONDecodeError:
                print("Error: The file is corrupted or empty.")
                return None
    else:
        print("No previous hyperparameters found or file is empty.")
        return None

save_folder = "D:\\transfer_learning"
hyperparams_file = os.path.join(save_folder, "best_hyperparameters.json")
def save_best_hyperparameters(hyperparameters):
    with open(hyperparams_file, 'w') as file:
        json.dump(hyperparameters, file)

# Check for existing best hyperparameters
best_hyperparams = load_best_hyperparameters()
if best_hyperparams:
    print("Using previously saved best hyperparameters:", best_hyperparams)
    learning_rates = [best_hyperparams["learning_rate"]]
    batch_sizes = [best_hyperparams["batch_size"]]
    dropout_rates = [best_hyperparams["dropout_rate"]]
else:
    # Fallback to testing all combinations
    learning_rates = [1e-3, 1e-4, 1e-5]
    batch_sizes = [16, 32, 64]
    dropout_rates = [0.2, 0.3, 0.5]
    
# Define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7),
]
from sklearn.utils.class_weight import compute_class_weight
# Calculate class weights to address class imbalance
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print(f"Class weights: {class_weights}")
# Iterate through combinations
results = []
for lr, batch_size, dropout_rate in hyperparameter_combinations:
    print(f"Testing: Learning Rate={lr}, Batch Size={batch_size}, Dropout Rate={dropout_rate}")

    # Create and compile the model
    model = create_model(dropout_rate=dropout_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

    # Train the model
    history = model.fit(
        [X_train, meta_X_train],
        y_train,
        validation_data=([X_val, meta_X_val], y_val),
        epochs=15,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1 
    )

    # Evaluate on the test set
    test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate([X_test, meta_X_test], y_test)

    results.append((lr, batch_size, dropout_rate, test_loss, test_accuracy, test_precision, test_recall, test_auc))

# Sort and identify best hyperparameters
results.sort(key=lambda x: x[4], reverse=True)  # Sort by accuracy
best_hyperparams = results[0]


# Save best hyperparameters
best_params = {
    "learning_rate": best_hyperparams[0],  # Learning rate
    "batch_size": best_hyperparams[1],     # Batch size
    "dropout_rate": best_hyperparams[2]    # Dropout rate
}
print("Best Hyperparameters:", best_params)

# Save best hyperparameters
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
with open(hyperparams_file, "w") as f:
    json.dump(best_params, f)
    print(f"Best hyperparameters saved to {hyperparams_file}")
    
if os.path.exists(hyperparams_file):
    with open(hyperparams_file, "r") as file:
        best_params = json.load(file)
    print("Using saved best hyperparameters:", best_params)
else:
    print("No hyperparameter file found, using default.")
    best_params = {
        'learning_rate': 1e-4, 
        'batch_size': 32, 
        'dropout_rate': 0.3
    }
    
lr = best_params['learning_rate']
batch_size = best_params['batch_size']
dropout_rate = best_params['dropout_rate']

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting accuracy vs. epoch
# plt.figure(figsize=(10, 5))
# plt.plot(history.history['accuracy'], label="Train Accuracy")
# plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.title("Accuracy over Epochs")
# plt.show()
# Check if history object exists
if 'accuracy' in history.history:
    # Plotting accuracy vs. epoch
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.show()

    # Plotting loss vs. epoch
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()


# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame(results, columns=["Learning Rate", "Batch Size", "Dropout Rate", "Loss", "Accuracy", "Precision", "Recall", "AUC"])

# Heatmap of Accuracy vs Hyperparameters
plt.figure(figsize=(12, 8))
sns.heatmap(results_df.pivot_table(index="Dropout Rate", columns="Batch Size", values="Accuracy", aggfunc="mean"),
            annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Accuracy vs Dropout Rate & Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Dropout Rate")
plt.show()

# Heatmap of AUC vs Hyperparameters
plt.figure(figsize=(12, 8))
sns.heatmap(results_df.pivot_table(index="Dropout Rate", columns="Batch Size", values="AUC", aggfunc="mean"),
            annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("AUC vs Dropout Rate & Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Dropout Rate")
plt.show()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Data Augmentation for the training set
data_augmentation = ImageDataGenerator(
    rotation_range=30,          # Rotate images by up to 30 degrees
    width_shift_range=0.2,      # Shift images horizontally by 20% of the width
    height_shift_range=0.2,     # Shift images vertically by 20% of the height
    zoom_range=0.3,             # Zoom in/out by up to 30%
    horizontal_flip=True,       # Randomly flip images horizontally
    fill_mode='nearest'         # Fill any gaps created by augmentation
)

# Visualize augmented images
augmented_images, augmented_labels = next(
    data_augmentation.flow(X_train, y_train, batch_size=5)  # Generate 5 augmented samples
)

# Display the augmented images
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(augmented_images[i])
    plt.axis("off")
plt.suptitle("Sample Augmented Images", fontsize=16)
plt.show()

import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# Image input branch
image_input = Input(shape=(224, 224, 3), name="image_input")
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_tensor=image_input
)
base_model.trainable = True  # Unfreeze EfficientNet for fine-tuning
for layer in base_model.layers[:-20]:  # Freeze all layers except the last 20
    layer.trainable = False

# Add pooling and dense layers for the image branch
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)

# Metadata input branch
meta_input = Input(shape=(meta_X_train.shape[1],), name="meta_input")
meta_x = layers.Dense(128, activation="relu")(meta_input)
meta_x = layers.Dropout(0.3)(meta_x)

# Combine the two branches
combined = layers.concatenate([x, meta_x])
combined_x = layers.Dense(256, activation="relu")(combined)
combined_x = layers.Dropout(0.5)(combined_x)
output = layers.Dense(1, activation="sigmoid")(combined_x)

# Create the model
model = Model(inputs=[image_input, meta_input], outputs=output)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
)

# Model summary
model.summary()







# Train the model
history = model.fit(
    [X_train, meta_X_train],
    y_train,
    validation_data=([X_val, meta_X_val], y_val),
    epochs=15,
    batch_size=batch_size,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1 
)

# Evaluate the model on the test set
test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(
    [X_test, meta_X_test], y_test
)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test AUC: {test_auc:.4f}")

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Generate predictions for the test set
y_pred_prob = model.predict([X_test, meta_X_test])
y_pred = (y_pred_prob > 0.5).astype(int)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix, annot=True, fmt="d", cmap="Blues", 
    xticklabels=["Negative", "Positive"], 
    yticklabels=["Negative", "Positive"]
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Visualize a few test images with predictions
plt.figure(figsize=(15, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i])
    plt.title(f"True: {y_test[i]}, Pred: {y_pred[i][0]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def grad_cam_multi_input(model, img_array, meta_array, last_conv_layer_name='conv2d', pred_index=None):
    # Create a model that outputs the activations of the last conv layer and predictions
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ],
    )
    
    # Get the gradient of the top predicted class for the last conv layer's output
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model([img_array, meta_array])
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute the gradient of the class score with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Pool gradients across the spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_grad_cam(image, heatmap, alpha=0.4):
    # Rescale heatmap to match the image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=-1)
    heatmap = tf.image.resize(heatmap, (image.shape[0], image.shape[1])).numpy()

    # Overlay heatmap on the image
    overlay = np.uint8(image * 255)
    overlay = tf.image.grayscale_to_rgb(tf.convert_to_tensor(heatmap)) * [255, 0, 0] + overlay

    plt.imshow(overlay / 255)
    plt.axis('off')
    plt.show()
    

# Select some test images
for i in range(3):  # Visualize Grad-CAM for 3 test images
    img_array = np.expand_dims(X_test[i], axis=0)  # Image input
    meta_array = np.expand_dims(meta_X_test[i], axis=0)  # Metadata input
    heatmap = grad_cam_multi_input(model, img_array, meta_array, last_conv_layer_name="top_conv")
    display_grad_cam(X_test[i], heatmap)
    
    