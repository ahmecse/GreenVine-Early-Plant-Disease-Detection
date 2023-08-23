#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Function to load your model and make predictions
def load_model():
    model = tf.keras.models.load_model('my_model.h5')  # Load your trained model
    return model

def predict(image):
    # Preprocess the image (resize, normalize, etc.)
    image = cv2.resize(image, (128, 128))  # Adjust to your model's input size
    image = image.astype('float32') / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(image)

    return prediction[0][0]

# Streamlit setup
st.title("Plant Disease Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the model
    model = load_model()

    # Read the image file and perform prediction
    image = Image.open(uploaded_file)
    image = np.array(image)

    prediction = predict(image)

    # Display the result
    if prediction > 0.5:
        st.write("Prediction: Rust")
    else:
        st.write("Prediction: Healthy")


# In[ ]:




