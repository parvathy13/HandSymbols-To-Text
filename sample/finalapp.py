#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import math
from PIL import Image
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier


# In[4]:


class HandGestureApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1)
        self.classifier = Classifier("C:/Users/Parvathy/Downloads/Model-20230820T064858Z-001/Model/keras_model.h5", "C:/Users/Parvathy/Downloads/Model-20230820T064858Z-001/Model/labels.txt")

        self.offset = 20
        self.imgSize = 300
        self.labels = ["Dislike", "Goodjob", "Goodluck", "Hangout", "Hello", "Iloveyou", "Loser", "No", "Yes"]

        self.img_output = None  # Store the PhotoImage instance

    def update(self):
        success, img = self.cap.read()
        imgOutput = img.copy()
        hands, img = self.detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
            imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = self.imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((self.imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, _ = self.classifier.getPrediction(imgWhite, draw=False)
                index = np.argmax(prediction)
                accuracy = round(prediction[index] * 100, 2)
                text = f"{self.labels[index]} - {accuracy}%"
            else:
                k = self.imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((self.imgSize - hCal) / 2)
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
            img_output = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
            img_output_pil = Image.fromarray(img_output)
            self.img_output = ImageTk.PhotoImage(image=img_output_pil)

    def run(self):
        st.title("Hand Gesture Recognition")

        stframe = st.empty()  # Placeholder for displaying the image

        while True:
            self.update()  # Call the update method
            stframe.image(self.img_output, channels="RGB", use_column_width=True)

if __name__ == "__main__":
    app = HandGestureApp()
    app.run()


# In[ ]:




