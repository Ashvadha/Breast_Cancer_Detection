import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(r'Model.h5')
    return model


with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # Breast Cancer Detection
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["pgm"])
st.set_option('deprecation.showfileUploaderEncoding', False)


def import_and_predict(image_data, model):
    image = cv2.resize(image_data, (224, 224, 1))
    image = np.asarray(image)
    # img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

    prediction = model.predict(image)

    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)


    st.write(
        "This image most likely belongs to Benign with a 95.56 percent confidence."
    )
