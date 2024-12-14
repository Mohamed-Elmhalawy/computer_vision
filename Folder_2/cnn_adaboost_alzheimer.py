import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Parameters
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = 'Alzheimer_s Dataset/Alzheimer_s Dataset/train'
VALIDATION_DIR = 'Alzheimer_s Dataset/Alzheimer_s Dataset/test'
CNN_MODEL_PATH = 'cnn_model.h5'
ADABOOST_MODEL_PATH = 'adaboost_model.pkl'

# Data augmentation for training data
train_data_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_data_generator = ImageDataGenerator(rescale=1.0 / 255)

# Load data
train_data = train_data_generator.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_data = validation_data_generator.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Check if the CNN model exists and load it, else train it
if os.path.exists(CNN_MODEL_PATH):
    print("Loading pre-trained CNN model...")
    cnn_model = load_model(CNN_MODEL_PATH)
else:
    print("Training CNN model...")
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(train_data.class_indices), activation='softmax')
    ])
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(train_data, epochs=EPOCHS, validation_data=validation_data)
    cnn_model.save(CNN_MODEL_PATH)
    print("CNN model saved.")

# Create a feature extractor model
feature_extractor_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

# Function to extract features from data
def extract_features(model, data):
    features = model.predict(data, verbose=1)
    labels = data.classes
    return features, labels

# Extract features for training and validation datasets
train_features, train_labels = extract_features(feature_extractor_model, train_data)
validation_features, validation_labels = extract_features(feature_extractor_model, validation_data)

# Check if the AdaBoost model exists and load it, else train it
if os.path.exists(ADABOOST_MODEL_PATH):
    print("Loading pre-trained AdaBoost classifier model...")
    adaboost_model = joblib.load(ADABOOST_MODEL_PATH)
else:
    print("Training AdaBoost classifier model...")
    adaboost_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50
    )
    adaboost_model.fit(train_features, train_labels)
    joblib.dump(adaboost_model, ADABOOST_MODEL_PATH)
    print("AdaBoost classifier model saved.")

# Evaluate the AdaBoost model
y_pred = adaboost_model.predict(validation_features)
accuracy = accuracy_score(validation_labels, y_pred)
print(f"AdaBoost classifier model accuracy: {accuracy * 100:.2f}%")

# Function to classify a new image
def classify_image(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    features = feature_extractor_model.predict(img_array)
    prediction = adaboost_model.predict(features)

    class_labels = {v: k for k, v in train_data.class_indices.items()}
    result = class_labels[prediction[0]]
    return result

# Test the classifier with a new image
image_path = 'Alzheimer_s Dataset/Alzheimer_s Dataset/test/NonDemented/26 (66).jpg'
predicted_class = classify_image(image_path)
print(f"The image is classified as: {predicted_class}")
