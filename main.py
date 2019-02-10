import dlibwrapper as face_recognition
# import picamera
import numpy as np
import pickle
import cv2
import os
import xml.etree.ElementTree as ET
import PySimpleGUI as sg
import psutil
import time
import base64
from PIL import Image


video_capture = cv2.VideoCapture(0)
known_face_encodings = [

]
known_face_names = [

]

print("Loading face image(s)")
for face in os.listdir("./faces"):
    if(face != ".DS_Store"):
        with open("./faces/"+face+"/info.xml", 'r') as f:
            xmlraw = f.read()
            root = ET.fromstring(xmlraw)
            name = root.find('name').text
        for file in os.listdir("./faces/"+face):
            if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")):
                print("Training "+name, end="", flush=True)
                tface = face_recognition.load_image_file("./faces/"+face+"/"+str(file))
                if(face_recognition.face_encodings(tface)):
                    tface_encoded = face_recognition.face_encodings(tface, num_jitters=100)[0]
                    known_face_encodings.append(tface_encoded)
                    im = Image.open("./faces/"+face+"/"+file)
                    im.save("./faces/"+face+"/"+file)
                    known_face_names.append(
                        {"name": name,
                        "dir":face+"/"+file,
                        "class": root.find('class').text,
                        "register": root.find('register').text,
                        "status": root.find('status').text})
                    print("[DONE]")
                else:
                    print("[FAIL]")
database = open("database.frdb","wb")
tostore = {"encodings": known_face_encodings,
            "names": known_face_names}
pickle.dump(tostore, database)
database.close()

layout = [[sg.Image('placeholder.png', size=(512,512), key="avatar")],
        [sg.Text('Name', font=('Helvetica', 50), justification='center', key='name')],
        [sg.Text('Clearance', text_color='red', font=('Helvetica', 50), justification='center', key='level')],
        [sg.Text('EAR', text_color='orange', font=('Helvetica', 50), justification='center', key='ear')]]
window = sg.Window('Innovation Security').Layout(layout)

event, values = window.Read(timeout=0)  
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the frame of video
    face_landmarks = face_recognition.face_landmarks(rgb_frame)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    print(face_locations)
    print(len(face_locations))
    if(len(face_locations) == 0):
        window.FindElement('avatar').Update("placeholder.png", size=(512, 512))
        window.FindElement('name').Update('Name')
        window.FindElement('level').Update('Clearance')
        window.FindElement('ear').Update('EAR')
    else:
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.38)

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                person = known_face_names[first_match_index]
                window.FindElement('avatar').Update("./faces/"+person['dir'], size=(512,512))
                window.FindElement('name').Update(person['name'])
                window.FindElement('level').Update('Level '+person['status'])
                #Get EAR Eye-Aspect-Ratio
                print((face_recognition.ear(face_landmarks[0]["right_eye"])))
            else:
                window.FindElement('avatar').Update("placeholder.png", size=(512, 512))
                window.FindElement('name').Update('Name')
                window.FindElement('level').Update('Clearance')
                window.FindElement('ear').Update('EAR')

    event, values = window.Read(timeout=0)


window.Close()
video_capture.release()