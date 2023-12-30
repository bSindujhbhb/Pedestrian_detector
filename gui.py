import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model

pedestrian_model = load_model('modelPedestrianDetection.h5')

top = tk.Tk()
top.geometry('800x600')
top.title('Pedestrian Detector')
top.configure(background='#CDCDCD')

label = Label(top, background="#CDCDCD", font=('arial', 15, "bold"))
sign_image = Label(top)

def detect_pedestrians(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((120, 120))
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) 
    image_np = cv2.resize(image_np, (120, 120))  
    image_np = image_np / 255.0
    image_np = np.expand_dims(image_np, axis=0)

    prediction = pedestrian_model.predict(image_np)
    confidence = np.max(prediction)
    class_index = np.argmax(prediction)

    if class_index == 0:
        label.configure(foreground="#011638", text=f"Pedestrian Detected")
        draw_pedestrian_boxes(file_path)
    else:
        label.configure(foreground="#011638", text=f"No Pedestrian Detected")

def draw_pedestrian_boxes(file_path):
    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(image_rgb, winStride=(8, 8), padding=(4, 4), scale=1.05)
    for (x, y, w, h) in boxes:
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    image_pil = Image.fromarray(image_rgb)
    im = ImageTk.PhotoImage(image_pil)
    sign_image.configure(image=im)
    sign_image.image = im

def show_detect_button(file_path):
    detect_button = Button(top, text="Detect Pedestrian", command=lambda: detect_pedestrians(file_path), padx=10, pady=5)
    detect_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_button.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_detect_button(file_path)
    except Exception as e:
        print(e)

upload = Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand=True)
label.pack(side="bottom", expand=True)
heading = Label(top, text="Pedestrian Detection", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

top.mainloop()
