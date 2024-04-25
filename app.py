# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import cv2
# from PIL import Image, ImageOps


# st.set_option('deprecation.showfileUploaderEncoding', False)
# @st.cache(allow_output_mutation=True)
# def load_model():
#   model = tf.keras.models.load_model('my_model2.hdf5')
#   return model

# model = load_model()

# st.write("""
#         # Brain Tumor Detection
#         """)

# file = st.file_uploader("Please uploaded a brain image", type=['jpg', 'png'])
# import cv2
# from PIL import Image, ImageOps

# # def import_and_predict(image_data, model):
# #   size = (180, 180)
# #   image = ImageOps.fit(image_data, size, 'ANTIALIAS')
# #   img = np.asarray(image)
# #   prediction = model.predict(img_reshape)

# #   return prediction
# def import_and_predict(image_data, model):
#     size = (180, 180)
#     try:
#         if image_data is not None:
#             image = ImageOps.fit(image_data, size, Image.BILINEAR)
#             img = np.asarray(image)
#             img_reshape = img[np.newaxis,...]
#             prediction = model.predict(img_reshape)
#             return prediction
#         else:
#             print("Image data is None")
#             return None
#     except Exception as e:
#         print("Error during image processing:", e)
#         return None


# if file is None:
#   st.text("Please upload an image file")
# else:
#   image = Image.open(file)
#   st.image(image, use_column_width = True)
#   predictions = import_and_predict(image, model)
#   class_names = ['Brain Tumor', 'Healthey']
#   # print("The MRI image is of " + class_names[np.argmax(predictions)])
#   # st.success(string)
#   result = class_names[np.argmax(predictions)]
#   st.success(f"The MRI image is {result}")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('my_model2.hdf5')
    return model

model = load_model()

st.write("# Brain Tumor Detection")

file = st.file_uploader("Please upload a brain image", type=['jpg', 'png'])

def import_and_predict(image_data, model):
    size = (180, 180)
    try:
        if image_data is not None:
            image = ImageOps.fit(image_data, size, Image.BILINEAR)
            img = np.asarray(image)
            img_reshape = img[np.newaxis,...]
            predicted_class_index = model.predict_classes(img_reshape)[0]
            return predicted_class_index
        else:
            print("Image data is None")
            return None
    except Exception as e:
        print("Error during image processing:", e)
        return None

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predicted_class_index = import_and_predict(image, model)
    if predicted_class_index is not None:
      class_names = ['Brain Tumor', 'Healthy']
      result = class_names[predicted_class_index]
      st.success(f"The MRI image is predicted as: {result}")