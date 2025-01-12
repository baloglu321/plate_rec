import streamlit as st
import streamlit_authenticator as stauth
from model_utils import *
from base.auth import auth

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

authenticator=auth()

def log_out():
    authenticator.logout()

try:
    authenticator.login()
except Exception as e:
    st.error(e)

if st.session_state['authentication_status']:
    
    st.sidebar.title(f'Welcome *{st.session_state["name"]}*')

    
    pages = [
            st.Page(main,title="Yayınlar"),            
            st.Page("tabs/plates.py", title="Plaka listesi"),
            st.Page("tabs/saved_videos.py", title="Kayıtlı videolar"),
            st.Page(log_out,title="Çıkış Yap")
            
        ]
    
    pg = st.navigation(pages)
    pg.run()
    

elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')

