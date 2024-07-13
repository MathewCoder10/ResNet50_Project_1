import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="paddy_tf.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # ResNet-50 input size
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension
    return image

# Function to make predictions using the TFLite model
def predict(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Streamlit app layout
st.title("Paddy Leaf Disease Prediction")

st.write("Upload an image of a paddy leaf to predict the disease.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = predict(processed_image)

    # Display prediction
    disease_classes = ['Healthy', 'Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']  # Example disease classes
    predicted_class = disease_classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    st.write(f"Prediction: {predicted_class} with {confidence:.2f} confidence")
