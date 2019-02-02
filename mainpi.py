import face_recognition # Relies on more
import picamera
import numpy as np
import pickle
import cv2
import os
import xml.etree.ElementTree as ET



database = open("database.frdb","rb")
tostore = pickle.load(database)
print("Loaded trained data")

known_face_encodings = tostore['encodings']
known_face_names = tostore['names']

camera = picamera.PiCamera()
camera.resolution = (320, 240)
output = np.empty((240, 320, 3), dtype=np.uint8)
camera.start_preview()

while True:
    print("Capturing")
    camera.capture(output, format="rgb")

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(output)
    print("Found {} faces in image.".format(len(face_locations)))
    face_encodings = face_recognition.face_encodings(output, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.45)
        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            person = known_face_names[first_match_index]
            print('Found '+person['name']+'. Level '+person['status']+' clearance')
        else:
            print('Unknown Person')