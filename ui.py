import streamlit as st
import numpy as np
import cv2
from model_utils import *
import os
from PIL import Image

model = YOLO_MODEL("./yolo.pt")
ocr = PaddleOCR(lang='tr')

img_file_buffer = st.file_uploader('Upload a image')
if img_file_buffer is not None:
    
    file_name = img_file_buffer.name
    
    # Uzantıyı al (örneğin: '.jpg', '.png')
    file_extension = os.path.splitext(file_name)[1].lower()
    
    if file_extension in [".jpg",".JPG",".png",".PNG",".bmp",".BMP",".jpeg",".JPEG"]:
        #img=cv2.imread(img_file_buffer)
        img = Image.open(img_file_buffer)
        img_array = np.array(img)
        with st.spinner('Wait for it...'):
            det_img=det(img_array,ocr,model) 
        st.success("Done!")
        st.image(det_img, caption="Plate results")
    else:
        st.error('Unsupported file extension. Please upload a file with one of the following extensions: .jpg, .JPG, .png, .PNG, .bmp, .BMP ', icon="🚨")


else:
    st.error('This file could not be read', icon="🚨")      