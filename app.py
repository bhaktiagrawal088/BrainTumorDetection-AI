import streamlit as st
import tensorflow as tf
import numpy as np
import cv2git

file = st.file_uploader("Please upload a brain image", type=['jpg', 'png'])

def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Brain Tumor', 'Healthy']
    result = class_names[np.argmax(predictions)]
    st.success("The MRI image is of " + result)