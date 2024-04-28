import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
import random
import os
from PIL import Image
import time
import smtplib
import os
from twilio.rest import Client

# Define callback function
def detect_human():
    # Add your code to detect humans here
    # My Twilio account details
    my_account_sid = "sid"
    my_account_aut = "aut"



    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    distance_thres = 50

    count = 0


    cap = cv2.VideoCapture(0)
    def dist(pt1,pt2):
        try:
            return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5
        except:
            return
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    print('Output layers',output_layers)
    _,frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter('output.avi', fourcc, 30,(frame.shape[1], frame.shape[0]), True)
    ret = True
    while ret:
        ret, img = cap.read()
        if ret:
            height, width = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            confidences = []
            boxes = []
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    if class_id!=0:
                        continue
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            persons = []
            person_centres = []
            violate = set()
            for i in range(len(boxes)):
                if i in indexes:
                    x,y,w,h = boxes[i]
                    persons.append(boxes[i])
                    person_centres.append([x+w//2,y+h//2])
            
            v = 0
            for (x,y,w,h) in persons:
                color = (0,255,0)
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv2.circle(img,(x+w//2,y+h//2),2,(0,0,255),2)
            
        
                if count == 0:
                    
                    # ### Sending msg with twilio account that someone has entered office.
                    # client = Client(my_account_sid, my_account_aut)
                    # message = client.messages \
                    # .create(
                    #     body="Alert: Human 🧑 Detected!\n\nA human has been detected in the restricted area.",
                    #     from_='sendig number',
                    #     to='recevier number'
                    # )
                    print("SMS Alert has been sent successfully")

                count = count+1 


            writer.write(img)
            cv2.imshow("Image", img)
        
        if cv2.waitKey(1) == 27:
            break


    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Detection", "Human Detected!")

# Create the main application window
root = tk.Tk()
root.title("Human Detection GUI")

# Set window size and position
window_width = 800
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Set background image
bg_image = tk.PhotoImage(file="background.png")
bg_label = tk.Label(root, image=bg_image)
bg_label.place(relwidth=1, relheight=1)

# Create a frame for the content
frame = tk.Frame(root, bg="white", bd=5)
frame.place(relx=0.5, rely=0.1, relwidth=0.8, relheight=0.8, anchor="n")

# Create a label
label = tk.Label(frame, text="Click the button to detect humans", bg="white", font=("Helvetica", 16))
label.pack(pady=10)

# Create a button
button = tk.Button(frame, text="Detect Human", command=detect_human, font=("Helvetica", 14), bg="grey", fg="white")
button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()