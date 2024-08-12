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
    image = image.convert('RGB')  # Ensure the image is in RGB mode
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
st.title("Paddy Leaf Disease Classification and Prediction")


# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = predict(processed_image)

        # Display prediction
        disease_classes = ["Bacterial_leaf_blight", "Brown_spot", "Healthy", "Leaf_blast", "Leaf_scald", "Narrow_brown_spot"]
        predicted_class = disease_classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100  # Convert to percentage

        # Set a confidence threshold
        confidence_threshold = 60  # 60%

        if confidence < confidence_threshold:
            st.warning(f"The model is not confident about the prediction. Please ensure the uploaded image is a clear and proper paddy leaf image.")
        
        st.markdown(f"### **Prediction: {predicted_class} with {confidence:.2f}% confidence**")

        
    except Exception as e:
        st.error(f"Error processing the image: {e}")
