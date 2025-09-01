import mediapipe as mp
import cv2
import threading
import numpy as np
import tensorflow as tf
from tkinter import *
from PIL import Image, ImageTk
from tkinter import StringVar
import pygame
import os

# Load the model
model = tf.keras.models.load_model("sign_language_model.h5")

# Initialize GUI
win = Tk()
width = win.winfo_screenwidth()
height = win.winfo_screenheight()
win.geometry(f"{width}x{height}")
win.title('Sign Language Recognition')

# Translation dictionary for gestures
translations = {
    "Okay": {"en": "Okay", "hi": "ओके", "te": "అలాగే"},
    "Dislike": {"en": "Dislike", "hi": "नापसंद", "te": "నచ్చలేదు"},
    "Victory": {"en": "Victory", "hi": "जीत", "te": "విజయం"},
    "Stop": {"en": "Stop", "hi": "रुकें", "te": "ఆపు"},
    "Point": {"en": "Hey You", "hi": "अरे तुम!", "te": "హే నువ్వు!"}
}

# MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class SignLanguageConverter:
    def __init__(self):
        self.CountGesture = StringVar()
        self.gesture_key = None
        self.previous_gesture = None
        self.current_frame = None
        self.gesture_to_audio = {
            "Okay": "Okay",
            "Dislike": "Dislike",
            "Victory": "Victory",
            "Stop": "Stop",
            "Point": "Hey You"
        }
        pygame.mixer.init()  # Initialize audio mixer once

    def detect_gesture(self, image):
        with mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                            min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.gesture_key = self.get_gesture(hand_landmarks)

                if self.gesture_key:
                    self._update_display()
                    threading.Thread(target=self._handle_audio, daemon=True).start()  # Audio runs in background

    def _update_display(self):
        if self.gesture_key in translations:
            translated = translations[self.gesture_key].get(current_language.get(), "")
            self.CountGesture.set(translated)
        else:
            self.CountGesture.set("")

    def get_gesture(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[4]
        index_finger_tip = hand_landmarks.landmark[8]
        middle_finger_tip = hand_landmarks.landmark[12]
        ring_finger_tip = hand_landmarks.landmark[16]
        little_finger_tip = hand_landmarks.landmark[20]

        if thumb_tip.y < index_finger_tip.y < middle_finger_tip.y < ring_finger_tip.y < little_finger_tip.y:
            return "Okay"
        elif thumb_tip.y > index_finger_tip.y > middle_finger_tip.y > ring_finger_tip.y > little_finger_tip.y:
            return "Dislike"
        elif index_finger_tip.y < middle_finger_tip.y and abs(index_finger_tip.x - middle_finger_tip.x) < 0.2:
            return "Victory"
        elif thumb_tip.x < index_finger_tip.x < middle_finger_tip.x:
            return "Stop"
        else:
            return "Point"

    def _handle_audio(self):
        if self.gesture_key and self.gesture_key != self.previous_gesture:
            self.previous_gesture = self.gesture_key
            lang = current_language.get()
            filename = self.gesture_to_audio.get(self.gesture_key, "")
            filepath = f"audio/{lang}/{filename}.mp3"

            if os.path.exists(filepath):
                pygame.mixer.music.load(filepath)
                pygame.mixer.music.play()

# Create converter instance
sign_lang_conv = SignLanguageConverter()

# GUI Configuration
current_language = StringVar(value='en')
BG_COLOR = "#181823"
TEXT_COLOR = "#F5EAEA"
BUTTON_COLOR = "#20262E"

# Main frame
main_frame = Frame(win, bg=BG_COLOR)
main_frame.place(x=0, y=0, relwidth=1, relheight=1)

# Header
Label(main_frame, text='Sign Language Recognition', font=('Arial', 26, 'bold'), 
      bg=BUTTON_COLOR, fg=TEXT_COLOR).pack(pady=20, fill=X)

# Video feed
video_label = Label(main_frame, bg=BG_COLOR)
video_label.place(x=450, y=150)

# Language Selection Panel
lang_frame = Frame(main_frame, bg=BUTTON_COLOR)
lang_frame.place(x=1200, y=300)
Label(lang_frame, text="Choose Language:", font=('Arial', 12), bg=BUTTON_COLOR, fg=TEXT_COLOR).pack(pady=5)

languages = [('English', 'en'), ('हिन्दी', 'hi'), ('తెలుగు', 'te')]
for text, lang in languages:
    Radiobutton(lang_frame, text=text, variable=current_language, value=lang, 
                command=sign_lang_conv._update_display, bg=BUTTON_COLOR, fg=TEXT_COLOR, 
                selectcolor="#000000", font=('Arial', 12)).pack(anchor=W, padx=10)

# Gesture Display
Label(main_frame, text='Current Gesture:', font=('Arial', 18, 'bold'), bg=BUTTON_COLOR, fg=TEXT_COLOR).place(x=200, y=700)
gesture_display = Label(main_frame, textvariable=sign_lang_conv.CountGesture, font=('Arial', 18), bg=BUTTON_COLOR, fg=TEXT_COLOR)
gesture_display.place(x=400, y=700)

# Exit Button
Button(main_frame, text='Exit', command=win.destroy, bg=BUTTON_COLOR, fg=TEXT_COLOR, font=('Arial', 14)).place(x=1200, y=600)

# Video Processing
cap = cv2.VideoCapture(0)

def update_video_feed():
    if not cap.isOpened():
        print("Error: Camera not detected!")
        return

    ret, frame = cap.read()
    if ret:
        sign_lang_conv.current_frame = frame.copy()
        sign_lang_conv.detect_gesture(frame)

        with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(frame, sign_lang_conv.CountGesture.get(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    win.after(10, update_video_feed)

# Start application
update_video_feed()
win.mainloop()
