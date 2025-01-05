import cv2
import threading
import time
from PIL import Image
import numpy as np
from datetime import datetime
import os
import streamlit as st
from model_utils import *



class VideoProcessor:

    def __init__(self):
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.is_running = False
        self.thread = None
        self.frame_buffer = []
        self.max_buffer_size = 3
        self.skip_frames = 0  # Her N frame'den birini işle
        self.frame_count = 0
        self.model = None
        self.ocr = None

    def initialize_models(self):
        # Modelleri tek seferde yükle
        if self.model is None:
            self.model = YOLO_MODEL("./yolo.pt")
        if self.ocr is None:
            self.ocr = PaddleOCR(lang='tr', use_angle_cls=False, use_gpu=True)
    
    def start_processing(self, source, fps=30.0):
        if self.is_running:
            return
        self.initialize_models()    
        self.is_running = True
        self.thread = threading.Thread(target=self._process_and_record, args=(source, fps))
        self.thread.daemon = True
        self.thread.start()
        

    
    def process_frame(self, frame):
        # Frame boyutunu küçült
        scale_percent = 70  # orijinal boyutun %70'i
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        frame = cv2.resize(frame, (width, height))
        
        # Görüntü işleme
        det_img = det(frame, self.ocr, self.model)
        return det_img
    
    def stop_processing(self):
        self.is_running = False
        if self.thread is not None:
            self.thread.join()
        if self.video_writer is not None:
            self.video_writer.release()
    
    def get_current_frame(self):
        with self.frame_lock:
            return self.current_frame
    
    def _process_and_record(self, source, fps):
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"Hata: Video kaynağı açılamadı: {source}")
                return
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                self.frame_count += 1
                if self.frame_count % self.skip_frames != 0:
                    # Frame'i atla ama görüntüyü göster
                    with self.frame_lock:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.current_frame = Image.fromarray(rgb_frame)
                    continue
                
                # Frame'i işle
                processed_frame = self.process_frame(frame)
                
                with self.frame_lock:
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    self.current_frame = Image.fromarray(rgb_frame)
                
                time.sleep(1/fps)
                
        except Exception as e:
            print(f"Hata oluştu: {e}")
        finally:
            cap.release()

def main():
    st.sidebar.title("Sayfa Seçimi")
    page = st.sidebar.radio("Sayfa:", ["Ana Sayfa", "İkinci Sayfa"])

    if 'processor' not in st.session_state:
        st.session_state.processor = VideoProcessor()
        st.session_state.processor_2 = VideoProcessor()
        
    if 'initialized' not in st.session_state:
        st.session_state.processor.start_processing("./video.mp4", fps=30.0)
        st.session_state.processor_2.start_processing("./video-2.mp4", fps=30.0)
        st.session_state.initialized = True
    if page == "Ana Sayfa":
        st.title("Video Akışı")
        placeholder = st.empty()
        
        while True:
            frame = st.session_state.processor.get_current_frame()
            if frame is not None:
                placeholder.image(frame, caption="Video Akışı")
            time.sleep(0.033)  # ~30 FPS
            
    elif page == "İkinci Sayfa":
        st.title("Video Akışı")
        placeholder = st.empty()
        
        while True:
            frame = st.session_state.processor_2.get_current_frame()
            if frame is not None:
                placeholder.image(frame, caption="Video Akışı")
            time.sleep(0.033)

if __name__ == "__main__":
    main()

