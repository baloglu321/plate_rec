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
from paddleocr import PaddleOCR # type: ignore
from ppocr.utils.logging import get_logger
import logging
logger = get_logger()
logger.setLevel(logging.ERROR)
 
# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
THICKNESS = 1
 
# Colors.
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)


def clean(img):
    """Preprocess image before OCR"""
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    return img



def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(im, (x,y- dim[1] - baseline), (x + dim[0], y), (0,0,0), cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(im, label, (x, y -baseline), FONT_FACE, FONT_SCALE, (0,255,0), THICKNESS, cv2.LINE_AA)
    
    
    
def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate, scores = [], []
    if ocr_result[0]!=None :
        for result in ocr_result[0]:
            length = np.sum(np.subtract(result[0][1], result[0][0]))
            height = np.sum(np.subtract(result[0][2], result[0][1]))
            
            if length*height / rectangle_size > region_threshold:
                plate.append(result[1][0])
                scores.append(result[1][1])

    plate = ''.join(plate)
    plate = re.sub(r'\W+', '', plate)
    if not scores:
        plate = ''
        scores.append(0)
    return plate.upper(), max(scores)


def recognize_plate_easyocr(img, coords,reader,region_threshold):
    """recognize license plate numbers using paddle OCR"""
    # separate coordinates from box
    xmin, ymin = coords[0]
    xmax, ymax = coords[1]
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    #nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] ### cropping the number plate from the whole image
    max_height = 80
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)]
    if nplate.shape[0] > max_height:
        scale = max_height / nplate.shape[0]
        nplate = cv2.resize(nplate, None, fx=scale, fy=scale)
    
    try:
        nplate = clean(nplate)
    except:
        return '',0

    ocr_result = reader.ocr(nplate)

    text, score = filter_text(region=nplate, ocr_result=ocr_result, region_threshold= region_threshold)

    if len(text) ==1:
        text = text[0].upper()
    return text, score

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
    h, w = img.shape[:2]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    yolo_detections = model(img,conf_threshold=0.55)
    detections_as_xyxy = yolo_detections.xyxy[0]
    texts=[]
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
        
        cv2.rectangle(img,points[0],points[1],(0,255,0),2)
        ocr_res, score = recognize_plate_easyocr(img, points, ocr, 0.2)
        #text = get_best_ocr(ocr_res, score, obj.id)
        text=ocr_res
        texts.append(text)
        draw_label(img, text, points[0][0], points[0][1])

    #cv2.imwrite("filename.jpg", img)

    return img


