import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import xgboost as xgb

# Parameters
image_size = (128, 128)
batch_size = 32
epochs = 10  # Number of epochs for CNN training
train_data_path = 'Alzheimer_s Dataset/Alzheimer_s Dataset/train'
validation_data_path = 'Alzheimer_s Dataset/Alzheimer_s Dataset/test'

# Data augmentation for training data
train_data_augmentation = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,  # Increased rotation range
    width_shift_range=0.3,  # More significant width shift
    height_shift_range=0.3,  # More significant height shift
    shear_range=0.3,  # Increased shear range
    zoom_range=0.3,  # Increased zoom range
    horizontal_flip=True,
    fill_mode='nearest'  # Fill missing pixels after transformations
)

validation_data_augmentation = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_data_augmentation.flow_from_directory(
    train_data_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

validation_generator = validation_data_augmentation.flow_from_directory(
    validation_data_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Define file paths for the models
cnn_model_file_path = 'cnn_model.h5'
xgb_classifier_model_file_path = 'xgb_classifier_model.pkl'

# Check if the CNN model exists and load it, else train it
if os.path.exists(cnn_model_file_path):
    print("Loading pre-trained CNN model...")
    cnn_model = load_model(cnn_model_file_path)
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
    cnn_model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
    
    # Save the trained CNN model
    cnn_model.save(cnn_model_file_path)
    print("CNN model saved.")

# Create an intermediate model to extract features before the output layer
feature_extractor_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

# Function to extract features using the CNN model
def extract_cnn_features(model, data_generator):
    features = []
    labels = []
    for _ in range(len(data_generator)):
        images, labels_batch = next(data_generator)
        feature_vectors = model.predict(images)  # Get feature vectors from CNN layers
        features.extend(feature_vectors)
        labels.extend(labels_batch)
    return np.array(features), np.array(labels)

# Extract features from training data
train_features, train_labels = extract_cnn_features(feature_extractor_model, train_generator)

# Extract features from validation data
validation_features, validation_labels = extract_cnn_features(feature_extractor_model, validation_generator)

# Convert one-hot labels to class labels for validation data
validation_labels = np.argmax(validation_labels, axis=1)

# Check if the XGBoost model exists and load it, else train it
if os.path.exists(xgb_classifier_model_file_path):
    print("Loading pre-trained XGBoost classifier model...")
    xgb_classifier = joblib.load(xgb_classifier_model_file_path)
else:
    print("Training XGBoost classifier model...")
    xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_classifier.fit(train_features, np.argmax(train_labels, axis=1))  # Ensure labels are in the right format
    
    # Save the trained XGBoost model
    joblib.dump(xgb_classifier, xgb_classifier_model_file_path)
    print("XGBoost classifier model saved.")

# Evaluate the XGBoost classifier model on validation data
y_pred = xgb_classifier.predict(validation_features)
accuracy = accuracy_score(validation_labels, y_pred)
print(f"XGBoost classifier model accuracy: {accuracy * 100:.2f}%")

def classify_new_image(image_path):
    # Load and preprocess the new image
    img = load_img(image_path, target_size=(128, 128))  # Resize to match the CNN input
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to match the training preprocessing

    # Extract features using the CNN model
    features = feature_extractor_model.predict(img_array)
    
    # Classify the features with the XGBoost model
    prediction = xgb_classifier.predict(features)
    
    # Interpret the prediction
    class_labels = {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented', 3: 'VeryMildDemented'}
    result = class_labels[prediction[0]]
    return result

# Test the function with a new image
image_path = 'Alzheimer_s Dataset/Alzheimer_s Dataset/test/NonDemented/26 (66).jpg'
result = classify_new_image(image_path)
print(f"The image is classified as: {result}")
