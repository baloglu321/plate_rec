# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:27:02 2024

@author: Mehmet
"""

import re

from random import randint
from typing import List, Optional, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import threading
import time
from paddleocr import PaddleOCR # type: ignore
from ppocr.utils.logging import get_logger
import logging
from base.key import *


logger = get_logger()
logger.setLevel(logging.ERROR)
 
# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
THICKNESS = 1
# Get color.
color_red=(255,0,0)
color_green=(0,255,0)


def clean(img):
    start_time=time.time()
    """Preprocess image before OCR"""
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    stop_time=time.time()
    response_time=stop_time-start_time
    return img,response_time



def draw_label(im, label, points):
    x=points[0][0]
    y=points[0][1]
    start_time=time.time()
    ENC_FILE = "keywords.enc"
    if os.path.exists(ENC_FILE):
        keys = decrypt_and_load(ENC_FILE)
    else:
        keys = []
        encrypt_and_save(keys, ENC_FILE)
    
    """Draw text onto image at location."""
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]

    # Use text size to create a BLACK rectangle.
    if label in keys:
        cv2.rectangle(im, (x,y- dim[1] - baseline), (x + dim[0], y), color_green, cv2.FILLED);
        cv2.rectangle(im,points[0],points[1],color_green,2)
        cv2.putText(im, label, (x, y -baseline), FONT_FACE, FONT_SCALE, color_red, THICKNESS, cv2.LINE_AA)
    else:
        cv2.rectangle(im, (x,y- dim[1] - baseline), (x + dim[0], y), color_red, cv2.FILLED);
        cv2.rectangle(im,points[0],points[1],color_red,2)
        cv2.putText(im, label, (x, y -baseline), FONT_FACE, FONT_SCALE, color_green, THICKNESS, cv2.LINE_AA)
    # Display text inside the rectangle.
    
    stop_time=time.time()
    response_time=stop_time-start_time
    return response_time
    
    
def filter_text(region, ocr_result, region_threshold=0.2):
    start_time = time.time()
    
    if ocr_result[0] is None:
        stop_time = time.time()
        response_time = stop_time - start_time
        return '', 0, response_time  # Bu satırı güncelledim
        
    plate_texts = []
    scores = []

    if len(ocr_result[0]) <= 1:
        
        for result in ocr_result[0]:
            text = result[1][0]
            score = result[1][1]
            cleaned_text = text.replace(" ", "")  
            # Türkiye plaka formatına uygunluk kontrolü
            if re.match(r'^[0-9]{2}[A-Z]{1,3}[0-9]{2,5}$', cleaned_text):
                plate_texts.append(cleaned_text)
                scores.append(score)
                
        if not plate_texts:
            stop_time = time.time()
            response_time = stop_time - start_time
            return '', 0, response_time  # Bu satırı güncelledim
            
        # En yüksek skorlu plakayı döndür
        max_score_idx = np.argmax(scores)
        stop_time = time.time()
        response_time = stop_time - start_time
        return plate_texts[max_score_idx], scores[max_score_idx], response_time
    
    elif len(ocr_result[0]) > 1:
        if ocr_result[0] is None:
            stop_time = time.time()
            response_time = stop_time - start_time
            return '', 0, response_time  # Bu satırı güncelledim
            
        plate_texts = []
        scores = []
        combined_text = ""  # Birleştirilmiş metin
        
        for detection in ocr_result[0]:  # detection içindeki her bir elemanı gez
            for item in detection:  # Her bir detection içindeki elemanları gez
                if type(item) == tuple:
                    combined_text += item[0] + " "  # Metni ekle ve araya boşluk bırak
                    score = item[1]
                    
        combined_text = combined_text.strip()  # Fazladan boşlukları temizle
        
        # Türkiye plaka formatına uyan parçaları ara
        potential_plates = []
        for line in combined_text.splitlines():
            match = re.search(r'\d{2}\s+[A-Z]{1,3}\s+\d{2,5}\s*(TR)?', line)
            if match:
                text=match.group()
                text=text.split(" ")[:-1]
                com_text=""
                for t in text:
                    com_text=com_text+t
                potential_plates.append(com_text)

            match2 = re.search(r'\d{2}\s+[A-Z]{1,3}\s+\d{2,5}$', line)
            if match2:
                text=match.group()
                potential_plates.append(text)

        if potential_plates:
            potential_plates = ''.join(potential_plates[0])

            plate_texts.append(potential_plates)
            scores.append(score)  # Skoru ekle (son geçerli metnin skoru)
                
        if not plate_texts:
            stop_time = time.time()
            response_time = stop_time - start_time
            return '', 0, response_time  # Bu satırı güncelledim
            
        # En yüksek skorlu plakayı döndür
        max_score_idx = np.argmax(scores)
        stop_time = time.time()
        response_time = stop_time - start_time
        return plate_texts[max_score_idx], scores[max_score_idx], response_time


def recognize_plate_ocr(img, coords,reader,region_threshold=0.2):
    h, w = img.shape[:2]
    start_time=time.time()
    """recognize license plate numbers using paddle OCR"""
    # separate coordinates from box
    xmin, ymin = coords[0]
    xmax, ymax = coords[1]
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] ### cropping the number plate from the whole image
    try:
        nplate,clean_time = clean(nplate)
    except:
        return '',0,0,0,0
    

    ocr_result = reader.ocr(nplate)
    
    text, score,filter_time = filter_text(region=nplate, ocr_result=ocr_result, region_threshold= region_threshold)

    if len(text) ==1:
        text = text[0].upper()
    stop_time=time.time()
    response_time=stop_time-start_time
    return text, score,response_time,filter_time,clean_time




class YOLO_MODEL:
    def __init__(self, weights, device: Optional[str] = None):
        self.model = torch.hub.load('./yolov5', 'custom', 
                                source='local', 
                                path=weights,
                                force_reload=False)  # True yerine False
        self.model.eval()  # Inference modu
        if torch.cuda.is_available():
            self.model.cuda()

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None,
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections




def det(img,ocr,model):
    
    img=cv2.resize(img,(1280,720))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    start_time=time.time()
    yolo_detections = model(img,conf_threshold=0.55)
    stop_time=time.time()
    model_res_time=stop_time-start_time

    
    detections_as_xyxy = yolo_detections.xyxy[0]
    texts=[]
    total_ocr=0
    total_filter=0
    total_clean=0
    total_label=0
    for detection_as_xyxy in detections_as_xyxy:
        bbox = np.array(
            [
                [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
            ]
        )
        points = bbox.astype(int)
        #points = tuple(bbox)
        points = tuple(map(tuple, points))
        
        

        ocr_res, score,ocr_time,filter_time,clean_time= recognize_plate_ocr(img, points, ocr, 0.2)
        cv2.rectangle(img,points[0],points[1],(0,255,0),2)
        #text = get_best_ocr(ocr_res, score, obj.id)
        text=ocr_res


        texts.append(text)
        label_time=draw_label(img, text, points[0][0], points[0][1])

        #print(f"Ocr Time: {ocr_time} sn")
        #print(f"filter Time: {filter_time} sn")
        #print(f"clean Time: {clean_time} sn")
        #print(f"label Time: {label_time} sn")
        
        #total_ocr=total_ocr+ocr_time
        #total_filter=total_filter+filter_time
        #total_clean=total_clean+clean_time
        #total_label=total_label+label_time

    #cv2.imwrite("filename.jpg", img)
    #print(f"Model Time: {model_res_time} sn")
    #print(f"Total Ocr Time: {total_ocr} sn")
    #print(f"total filter Time: {total_filter} sn")
    #print(f"total clean Time: {total_clean} sn")
    #print(f"total label Time: {total_label} sn")

    return img

import threading
import time
import cv2
import numpy as np

class VideoProcessor:
    _instances = {}  # Tüm VideoProcessor nesnelerinin tutulduğu bir sınıf değişkeni

    def __init__(self, name):
        self.name = name
        self.video_cap = None
        self.current_frame = None
        self.is_running = False
        self.thread = None
        self.fps = None
        self.model = None
        self.ocr = None

    def load_model(self,model_path="./yolo.pt"):
        self.model=YOLO_MODEL(model_path)
        self.ocr=PaddleOCR(lang='en')

    def is_processing(self):
        return self.is_running

    def process_frame(self,frame):
        
        if self.model is None:
            self.load_model()
        
        if self.ocr is None:
            self.load_model()

        yolo_detections = self.model(frame,conf_threshold=0.55)    
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
                ]
            )
            points = bbox.astype(int)
            #points = tuple(bbox)
            points = tuple(map(tuple, points))
            ocr_res, score,ocr_time,filter_time,clean_time= recognize_plate_ocr(frame, points, self.ocr, region_threshold=0.2)
            label_time=draw_label(frame, ocr_res, points)
            #cv2.rectangle(frame,points[0],points[1],(0,255,0),2)
        return frame
    
    @classmethod
    def get_instance(cls, name, video_path=None, fps=30.0):
        """Mevcut bir VideoProcessor nesnesi varsa onu döndür, yoksa yeni bir tane oluştur."""
        if name in cls._instances:
            instance = cls._instances[name]
            print(f"Var olan VideoProcessor nesnesi döndürüldü: {name}")
        else:
            instance = cls(name)
            cls._instances[name] = instance
            print(f"Yeni bir VideoProcessor nesnesi oluşturuldu: {name}")
            if video_path:
                instance.start_processing(video_path, fps)
        return instance


    def process_video(self):
        
        while self.is_running:
            
            ret, frame = self.video_cap.read()
            
            if not ret:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Video bitince başa sar
                continue

            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)   
            det_img=self.process_frame(frame)
            self.current_frame = det_img
            
            if self.fps:
                time.sleep(1/self.fps)
    
    
    def start_processing(self, video_path, fps=30.0):
        if self.is_running:
            print(f"VideoProcessor zaten çalışıyor: {self.name}")
            return

        self.video_cap = cv2.VideoCapture(video_path)
        self.fps = fps
        self.is_running = True

        # Thread'i başlat ve adını ayarla
        self.thread = threading.Thread(target=self.process_video, name=self.name)
        self.thread.daemon = True
        self.thread.start()

    def get_current_frame(self):
        return self.current_frame

    def stop(self):
        self.is_running = False
        if self.video_cap:
            self.video_cap.release()
        if self.thread:
            self.thread.join()

