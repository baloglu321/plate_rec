import streamlit as st
from model_utils_deneme import *

if 'processor_1' not in st.session_state:
    st.session_state.processor_1 = VideoProcessor.get_instance("processor_1", "./video.mp4", fps=20.0)

if 'processor_2' not in st.session_state:
    st.session_state.processor_2 = VideoProcessor.get_instance("processor_2", "./video-2.mp4", fps=15.0)

def main():
    st.sidebar.title("Yayınlar")
    page = st.sidebar.radio("Sayfa:", ["Kamera 1", "Kamera 2"])


    if page == "Kamera 1":
        st.title("Video Akışı")
        placeholder = st.empty()
        
        while True:
            frame = st.session_state.processor_1.get_current_frame()
            if frame is not None:
                placeholder.image(frame, caption="Video Akışı")
            time.sleep(0.033)  # ~30 FPS
            
    elif page == "Kamera 2":
        st.title("Video Akışı")
        placeholder = st.empty()
        
        while True:
            frame = st.session_state.processor_2.get_current_frame()
            if frame is not None:
                placeholder.image(frame, caption="Video Akışı")
            time.sleep(0.033)



pages = [
        st.Page(main,title="Yayınlar"),
        st.Page("tabs/plates.py", title="Plaka listesi"),
        st.Page("tabs/saved_videos.py", title="Kayıtlı videolar")
    ]



pg = st.navigation(pages)
pg.run()