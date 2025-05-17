from time import sleep
import tkinter as tk
import cv2
import io
import pygame
import numpy as np
import pyttsx3
from tkinter import ttk, messagebox
from tensorflow.keras.models import load_model  # For gesture recognition
from gtts import gTTS
from translate import Translator  # For translation support

from segmentation import segment_hand

INPUT_SIZE = 128

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Load pre-trained gesture recognition model (you need a model trained for sign language)
DEFAULT_MODEL_PATH = 'asl_classifier.keras'
gesture_model = load_model(DEFAULT_MODEL_PATH)

# Gesture labels corresponding to the model's output - Add more labels as needed
gesture_labels = {0: "", 1: "Hello", 2: "Thanks", 3: "Yes", 4: "No", 5: "Help"}

# Pyttsx3 engine for text-to-speech support
tts_engine = pyttsx3.init()

# Function to capture webcam frames and process gestures
def recognize_sign_language():
    source = cv2.VideoCapture(0)
    visual_threshold = 70
    gaussian_window = 11
    fine_tune_c = 2
    recognized_text = ""

    if not source.isOpened():
        messagebox.showerror("Error", "Unable to access webcam.")
        return recognized_text
    
    ret, frame = source.read()
    img_h, img_w = frame.shape[:2]

    while True:
        ret, frame = source.read()
        if not ret:
            break

        # Preprocess frame for gesture model
        crop_img = segment_hand(frame)
        if crop_img is not None:
            # Clean up the image data for training
            crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(crop_gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, gaussian_window, fine_tune_c)
            ret, preprocessed = cv2.threshold(th3, visual_threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cv2.imshow("Preprocessed", preprocessed)

            resized = cv2.resize(preprocessed, (INPUT_SIZE, INPUT_SIZE))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, INPUT_SIZE, INPUT_SIZE, 1))

            # Predict gesture
            predictions = gesture_model.predict(reshaped)
            gesture_index = np.argmax(predictions)
            recognized_text = gesture_labels.get(gesture_index, "")

            if recognized_text:
                input_text.insert("end", recognized_text + " ")
                selected_language = language_var.get()
                language_code = languages[selected_language]
                
                # Translate the recognized text
                translated_text = translate_text(recognized_text, language_code)
                convert_text_to_audio(translated_text, language_code)

        # Display frame with the recognized gesture
        cv2.putText(frame, recognized_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Sign Language Recognition (Press 'q' to exit)", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    source.release()
    cv2.destroyAllWindows()
    return recognized_text

# Function to convert text to speech
def convert_text_to_audio(text, language):
    try:
        if language == "ml":  # Use pyttsx3 for Malayalam
            tts_engine.setProperty("voice", "com.apple.speech.synthesis.voice.malayalam")
            tts_engine.say(text)
            tts_engine.runAndWait()
        else:  # Use gTTS for other languages
            tts = gTTS(text, lang=language)
            speech_buffer = io.BytesIO()
            tts.write_to_fp(speech_buffer)
            speech_buffer.seek(0)
            pygame.mixer.music.load(speech_buffer, 'mp3')
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                continue
    except Exception as e:
        messagebox.showerror("Error", f"Audio generation failed: {e}")

# Function to translate text using the translate library
def translate_text(text, target_language):
    try:
        translator = Translator(to_lang=target_language)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        messagebox.showerror("Error", f"Translation failed: {e}")
        return text

# Function to handle sign language conversion
def handle_sign_language_conversion():
    selected_language = language_var.get()
    if not selected_language:
        messagebox.showwarning("Language Error", "Please select a language.")
        return
    recognize_sign_language()

# Main Tkinter window
root = tk.Tk()
root.title("Sign Language to Audio Converter")

# Language options
languages = {
    "Hindi": "hi",
    "Kannada": "kn",
    "French": "fr",
    "English": "en",
    "Arabic": "ar",
    "Greek": "el",
    "Gujarati": "gu",
    "Malayalam": "ml",
    "Punjabi": "pa"
}

# UI Elements
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky="nsew")

# Text input
ttk.Label(frame, text="Recognized Text:").grid(row=0, column=0, sticky="w")
input_text = tk.Text(frame, wrap="word", width=50, height=10)
input_text.grid(row=1, column=0, columnspan=2, pady=10)

# Language selection dropdown
ttk.Label(frame, text="Choose Language:").grid(row=2, column=0, sticky="w")
language_var = tk.StringVar()
language_dropdown = ttk.Combobox(frame, textvariable=language_var, state="readonly", width=30)
language_dropdown['values'] = list(languages.keys())
language_dropdown.grid(row=2, column=1, sticky="e")
language_dropdown.set("English")

# Buttons
sign_language_button = ttk.Button(frame, text="Convert Sign Language to Audio", command=handle_sign_language_conversion)
sign_language_button.grid(row=3, column=0, columnspan=2, pady=5)

# Run Tkinter event loop
root.mainloop()
