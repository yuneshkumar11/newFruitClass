import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd


# Load the trained model (update the path accordingly)
model = tf.keras.models.load_model('./fruit_classificatin_model.keras')
label_df=pd.read_csv('./labels.csv')

# Define class names (update to your actual labels)

# <- Update this to your actual classes
class_names = sorted(label_df['Label'].unique())

# DataFrame to store image history
history_file = "image_history.csv"

# Function to load history from CSV if it exists, without using os
def load_history():
    try:
        return pd.read_csv(history_file)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Image", "Prediction", "Confidence"])
# Streamlit app title
st.title("Fruit & Vegetable Image Classification")
st.write("Upload an image to classify it as a fruit or vegetable.")

# Image upload functionality
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "jfif"])

# Preprocessing function for the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize if needed
    return img_array

# Image classification function
def predict(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(float(np.max(predictions[0]) * 100), 1)
    return predicted_class, confidence

# If an image is uploaded, run prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display uploaded image
    st.image(image, caption='Uploaded Image', width=200)
    
    # Prediction
    predicted_class, confidence = predict(image)
    
    # Show results
    st.header(f"Prediction: {predicted_class}")
    st.subheader(f"Confidence: {confidence}%")

# Load the history, append the new record, and save it back to CSV
    history_df = load_history()
    history_df = pd.concat([history_df, pd.DataFrame([{"Image": uploaded_file.name, "Prediction": predicted_class, "Confidence": confidence}])], ignore_index=True)
    history_df.to_csv(history_file, index=False)

    # Display history table
    st.write("### Image Classification History")
    st.dataframe(history_df)