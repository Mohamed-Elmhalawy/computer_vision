import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.tree import DecisionTreeClassifier

# Parameters
image_size = (128, 128)
batch_size = 32
epochs = 10  # Number of epochs for CNN training
train_dir = 'Alzheimer_s Dataset/Alzheimer_s Dataset/train'
test_dir = 'Alzheimer_s Dataset/Alzheimer_s Dataset/test'

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,  # Increased rotation range
    width_shift_range=0.3,  # More significant width shift
    height_shift_range=0.3,  # More significant height shift
    shear_range=0.3,  # Increased shear range
    zoom_range=0.3,  # Increased zoom range
    horizontal_flip=True,
    fill_mode='nearest'  # Fill missing pixels after transformations
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_data_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Define file paths for the models
cnn_model_filepath = 'cnn_model.h5'
dt_model_filepath = 'dt_classifier_model.pkl'

# Check if the CNN model exists and load it, else train it
if os.path.exists(cnn_model_filepath):
    print("Loading pre-trained CNN model...")
    cnn_model = load_model(cnn_model_filepath)
else:
    print("Training CNN model...")
    # Define and train the CNN model
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),  # This layer will ensure the output is a 1D feature vector
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # Update with correct number of classes
    ])
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(train_data_gen, epochs=epochs, validation_data=test_data_gen)
    
    # Save the trained CNN model
    cnn_model.save(cnn_model_filepath)
    print("CNN model saved.")

# Create an intermediate model to extract features before the output layer
feature_extractor_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

# Function to extract features using the CNN model
def extract_features_from_data(model, data_gen):
    features = []
    labels = []
    for _ in range(len(data_gen)):
        images, labels_batch = next(data_gen)
        feature_vectors = model.predict(images)  # Get feature vectors from CNN layers
        features.extend(feature_vectors)
        labels.extend(labels_batch)
    return np.array(features), np.array(labels)

# Extract features from training data
train_features, train_labels = extract_features_from_data(feature_extractor_model, train_data_gen)

# Extract features from test data
test_features, test_labels = extract_features_from_data(feature_extractor_model, test_data_gen)

# Convert one-hot labels to class labels for test data
test_labels = np.argmax(test_labels, axis=1)

# Check if the Decision Tree model exists and load it, else train it
if os.path.exists(dt_model_filepath):
    print("Loading pre-trained Decision Tree classifier model...")
    decision_tree_model = joblib.load(dt_model_filepath)
else:
    print("Training Decision Tree classifier model...")
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(train_features, np.argmax(train_labels, axis=1))  # Ensure labels are in the right format
    
    # Save the trained Decision Tree model
    joblib.dump(decision_tree_model, dt_model_filepath)
    print("Decision Tree classifier model saved.")

# Evaluate the Decision Tree classifier model on test data
predicted_labels = decision_tree_model.predict(test_features)
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Decision Tree classifier model accuracy: {accuracy * 100:.2f}%")

def classify_new_image(image_path):
    # Load and preprocess the new image
    img = load_img(image_path, target_size=(128, 128))  # Resize to match the CNN input
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to match the training preprocessing

    # Extract features using the CNN model
    features = feature_extractor_model.predict(img_array)
    
    # Classify the features with the Decision Tree model
    predicted_class = decision_tree_model.predict(features)
    
    # Map the predicted class to a label
    class_labels = {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented', 3: 'VeryMildDemented'}
    result_label = class_labels[predicted_class[0]]
    return result_label

# Test the function with a new image
test_image_path = 'Alzheimer_s Dataset/Alzheimer_s Dataset/test/NonDemented/26 (66).jpg'
classification_result = classify_new_image(test_image_path)
print(f"The image is classified as: {classification_result}")
