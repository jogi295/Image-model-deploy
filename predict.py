import numpy as np
import tensorflow as tf
from PIL import Image

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="mobilenet_v2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the input image
def preprocess_image(image_path, target_size):
    # Load the image
    image = Image.open(image_path).resize(target_size)
    # Convert the image to a numpy array and normalize it
    image_array = np.array(image) / 255.0
    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
    return image_array

# Load labels
def load_labels(label_path):
    with open(label_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

# Predict the label of the input image
def predict_image_label(image_path):
    # Preprocess the input image
    input_image = preprocess_image(image_path, (224, 224))  # MobileNetV2 expects 224x224 input size
    
    # Set the tensor to the input data
    interpreter.set_tensor(input_details[0]['index'], input_image)
    
    # Run the interpreter
    interpreter.invoke()
    
    # Get the output data (predictions)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Find the index of the maximum probability
    predicted_label_index = np.argmax(output_data[0])
    
    # Load the labels
    labels = load_labels("labels.txt")
    
    # Get the corresponding label
    predicted_label = labels[predicted_label_index]
    
    print(f"Predicted label: {predicted_label}")

# Provide the path to your image
image_path = "photo.jpg"
predict_image_label(image_path)
