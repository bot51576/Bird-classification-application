import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import sys
import tensorflow as tf
import datetime, os
import numpy as np
import pandas as pd

def ayu(inputs):
    return tf.keras.activations.swish(inputs)

model = tf.keras.models.load_model('./cnn.h5', custom_objects={'ayu': ayu})
#name = input()
image_size = 28
st.write('''
    ## Bird classification application (crows, swallows, chickens, pigeons, white-eyes, warblers)

    ### uploadfile 
''')

upload_file = st.file_uploader('')

if upload_file is not None:
    img = Image.open(upload_file)
    st.image(img, caption='uploaded Image',use_column_width=True)
    image = Image.open(upload_file)
    image = image.resize((image_size, image_size))
    image = image.convert('RGB')
    np_image = np.array(image)
    np_image = np_image / 255 
    result = model.predict(np.array([np_image]))

    list1=[[
        result[0][0] * 100,
        result[0][1] * 100,
        result[0][2] * 100, 
        result[0][3] * 100, 
        result[0][4] * 100,
        result[0][5] * 100]]
    index1 = ['確率(％)']
    columns1 =['カラス', 'ツバメ', 'ニワトリ', 'ハト', 'メジロ', 'ウグイス']
    df = pd.DataFrame(data=list1, index=index1, columns=columns1)
    st.dataframe(data=df.style.highlight_max(axis=1))