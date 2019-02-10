import face_recognition # Relies on more
# import picamera
import numpy as np
import pickle
import cv2
import os
import xml.etree.ElementTree as ET
import PySimpleGUI as sg
import psutil

sg.ChangeLookAndFeel('Black')
layout = [[sg.Text('')],
          [sg.Text('', size=(8, 2), font=('Helvetica', 20), justification='center', key='text')]]
window = sg.Window('Running Timer', no_titlebar=True, auto_size_buttons=False, keep_on_top=True,
                   grab_anywhere=True).Layout(layout)


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
                    tface_encoded = face_recognition.face_encodings(tface, num_jitters=200)[0]
                    known_face_encodings.append(tface_encoded)
                    known_face_names.append(
                        {"name": name,
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



while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.38)
        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            person = known_face_names[first_match_index]
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, person['name']+"/"+person['status'], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()