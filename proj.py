#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import math
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class HandGestureApp(VideoProcessorBase):
    def __init__(self):
        self.detector = HandDetector(maxHands=1)
        self.classifier = Classifier("keras_model.h5", "labels.txt")

        self.offset = 20
        self.imgSize = 300
        self.labels = ["Dislike", "Goodjob", "Goodluck", "Hangout", "Hello", "Iloveyou", "Loser", "No", "Yes"]

    def on_recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        imgOutput = img.copy()
        hands, img = self.detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
            imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = self.imgSize / h
                wCal = np.ceil(k * w).astype(int)
                imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                wGap = np.ceil((self.imgSize - wCal) / 2).astype(int)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = self.imgSize / w
                hCal = np.ceil(k * h).astype(int)
                imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                hGap = np.ceil((self.imgSize - hCal) / 2).astype(int)
                imgWhite[hGap:hCal + hGap, :] = imgResize
            
            prediction, _ = self.classifier.getPrediction(imgWhite, draw=False)
            index = np.argmax(prediction)
            accuracy = round(prediction[index] * 100, 2)
            text = f"{self.labels[index]} - {accuracy}%"

            cv2.rectangle(imgOutput, (x - self.offset, y - self.offset - 50),
                          (x - self.offset + 90, y - self.offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, text, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - self.offset, y - self.offset),
                          (x + w + self.offset, y + h + self.offset), (255, 0, 255), 4)

            cv2.putText(imgOutput, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return imgOutput

def main():
    st.title("Hand Gesture Recognition")
    webrtc_ctx = webrtc_streamer(
        key="hand-gesture",
        video_processor_factory=HandGestureApp,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
    )

if __name__ == "__main__":
    main()


# In[ ]:




