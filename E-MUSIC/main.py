import os
import cv2
import numpy as np
import pygame
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import glob
import random

# Define the base directory for songs
base_dir = "C:/Users/Guna/OneDrive/Music/E-MUSIC/Music_Recommendation_System"

# Load the face detector cascade
face_classifier = cv2.CascadeClassifier('C:/Users/Guna/OneDrive/Music/E-MUSIC/haarcascade_frontalface_default.xml')

# Load the emotion detection model
classifier = load_model('C:/Users/Guna/OneDrive/Music/E-MUSIC/model.h5')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize Pygame mixer
pygame.mixer.init()

# Define a function to play a song
def play_song(song_path):
    song_path = song_path.replace('\\', '/')
    pygame.mixer.music.load(song_path)
    pygame.mixer.music.play()

    # Wait for the song to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Open the video capture device
cap = cv2.VideoCapture('http://192.168.225.53:8080/video')

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Extract the region of interest (ROI) from the grayscale frame
        roi_gray = gray[y:y+h, x:x+w]

        # Resize the ROI to 48x48 pixels
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Check if the ROI is not empty
        if np.sum([roi_gray])!= 0:
            # Normalize the ROI pixel values to be between 0 and 1
            roi = roi_gray.astype('float')/255

            # Predict the emotion of the ROI
            preds = classifier.predict(np.expand_dims(roi, axis=0))
            emotion_index = np.argmax(preds)
            emotion = emotion_labels[emotion_index]

            # Get the path of the songs directory for the selected language
            language = input("Enter your preferred language (Kannada, Tamil, English): ")

            if language not in ['Kannada', 'Tamil', 'English']:
                print("Invalid language. Please enter one of the following: Kannada, Tamil, English")
                continue

            language_dir = os.path.join(base_dir, language)

            # Get the path of the song corresponding to the predicted emotion
            emotion_dir = os.path.join(language_dir, emotion)

            if not os.path.exists(emotion_dir):
                print(f"Emotion directory '{emotion}' not found")
                continue

            song_files = glob.glob(os.path.join(emotion_dir, '*.mp3'))

            if not song_files:
                print(f"No songs found in emotion directory '{emotion}'")
                continue

            # Play a random song from the list of songs
            song_file = random.choice(song_files)
            print(f"Playing {song_file}")
            os.startfile(song_file)

            # Draw the predicted emotion text above the face rectangle
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Display the frame with the detected faces and their predicted emotions
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close the windows
cap.release()
cv2.destroyAllWindows()