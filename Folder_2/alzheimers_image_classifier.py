import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.linear_model import LogisticRegression

# Parameters
image_size = (128, 128)
batch_size = 32
epochs = 10  # Number of epochs for CNN training
train_data_path = 'Alzheimer_s Dataset/Alzheimer_s Dataset/train'
validation_data_path = 'Alzheimer_s Dataset/Alzheimer_s Dataset/test'

# Data augmentation for training data
train_data_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,  # Increased rotation range
    width_shift_range=0.3,  # More significant width shift
    height_shift_range=0.3,  # More significant height shift
    shear_range=0.3,  # Increased shear range
    zoom_range=0.3,  # Increased zoom range
    horizontal_flip=True,
    fill_mode='nearest'  # Fill missing pixels after transformations
)

validation_data_generator = ImageDataGenerator(rescale=1.0 / 255)

train_data_flow = train_data_generator.flow_from_directory(
    train_data_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

validation_data_flow = validation_data_generator.flow_from_directory(
    validation_data_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Define file paths for the models
cnn_model_filepath = 'cnn_model.h5'
lr_classifier_model_filepath = 'lr_classifier_model.pkl'

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
    cnn_model.fit(train_data_flow, epochs=epochs, validation_data=validation_data_flow)
    
    # Save the trained CNN model
    cnn_model.save(cnn_model_filepath)
    print("CNN model saved.")

# Create an intermediate model to extract features before the output layer
feature_extractor_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

# Function to extract features using the CNN model
def extract_features_from_data(model, data_flow):
    extracted_features = []
    extracted_labels = []
    for _ in range(len(data_flow)):
        images, labels = next(data_flow)
        feature_vectors = model.predict(images)  # Get feature vectors from CNN layers
        extracted_features.extend(feature_vectors)
        extracted_labels.extend(labels)
    return np.array(extracted_features), np.array(extracted_labels)

# Extract features from training data
train_features_data, train_labels_data = extract_features_from_data(feature_extractor_model, train_data_flow)

# Extract features from validation data
validation_features_data, validation_labels_data = extract_features_from_data(feature_extractor_model, validation_data_flow)

# Convert one-hot labels to class labels for validation data
validation_labels_data = np.argmax(validation_labels_data, axis=1)

# Check if the Logistic Regression model exists and load it, else train it
if os.path.exists(lr_classifier_model_filepath):
    print("Loading pre-trained Logistic Regression classifier model...")
    lr_classifier_model = joblib.load(lr_classifier_model_filepath)
else:
    print("Training Logistic Regression classifier model...")
    lr_classifier_model = LogisticRegression(max_iter=1000)
    lr_classifier_model.fit(train_features_data, np.argmax(train_labels_data, axis=1))  # Ensure labels are in the right format
    
    # Save the trained Logistic Regression model
    joblib.dump(lr_classifier_model, lr_classifier_model_filepath)
    print("Logistic Regression classifier model saved.")

# Evaluate the Logistic Regression classifier model on validation data
predicted_labels = lr_classifier_model.predict(validation_features_data)
model_accuracy = accuracy_score(validation_labels_data, predicted_labels)
print(f"Logistic Regression classifier model accuracy: {model_accuracy * 100:.2f}%")

def classify_new_image(image_filepath):
    # Load and preprocess the new image
    image = load_img(image_filepath, target_size=(128, 128))  # Resize to match the CNN input
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array /= 255.0  # Normalize to match the training preprocessing

    # Extract features using the CNN model
    features_from_image = feature_extractor_model.predict(image_array)
    
    # Classify the features with the Logistic Regression model
    prediction_result = lr_classifier_model.predict(features_from_image)
    
    # Interpret the prediction
    class_labels = {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented', 3: 'VeryMildDemented'}
    classification_result = class_labels[prediction_result[0]]
    return classification_result

# Test the function with a new image
new_image_path = 'Alzheimer_s Dataset/Alzheimer_s Dataset/test/NonDemented/26 (66).jpg'
classification_result = classify_new_image(new_image_path)
print(f"The image is classified as: {classification_result}")
