from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLO model
yolo_model = YOLO("yolo11m.pt")

# Path to the image for object detection
image_path = "CR7.jpg"

# Perform YOLO inference on the image with a confidence threshold of 0.4
detection_results = yolo_model(source=image_path, conf=0.4)

# Get the image with annotations drawn by YOLO
annotated_image = detection_results[0].plot()[..., ::-1]  # Convert from BGR to RGB

# Display the annotated image
plt.imshow(annotated_image)
plt.axis('off')  # Hide axis
plt.show()
