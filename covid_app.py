# importing required packages
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Create the title for the App
#st.title("Covid X-ray Prediction")
st.write("# Examine your X-ray image here to determine if it indicates the presence of COVID-19.")
st.write("###### A platform created by a researcher from the University of Hull allowing users to assess the condition of their chest X-rays, distinguishing between a normal chest, COVID-19, and pneumonia.")
# Create a file uploader
uploaded_file = st.file_uploader("Upload your x-ray image..", type=["jpg", "jpeg", "png"])

# Check if the image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    st.write("")

    # Preprocess the image
    img = np.array(image)
    # Resize the image using TensorFlow
    img = tf.image.resize(tf.convert_to_tensor(img), (64, 64))
    # Normalize the image
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Change the batch dimension to (1, 64, 64, 3)

    # Load the trained model
    model_path = "C:\\Users\\User\\covid_detection_model\\covid_vgg_model.h5"

    try:
        model = load_model(model_path)
        st.success("Image loaded successfully!")
    except Exception as e:
        st.error(f"Error loading the model: {e}")

    # Make predictions
    prediction = model.predict(img)

    # Determine the label based on numerical thresholds
    class_index = np.argmax(prediction)
    if class_index == 0:
        label = "Covid-19"
    elif class_index == 1:
        label = "Normal"
    else:
        label = "Viral Pneumonia"

    #st.write(f"Prediction: {label}")

    # Display the prediction
    st.write(f"## Predicted Chest Condition: {label}")
