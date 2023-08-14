import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime



video_capture = cv2.VideoCapture(0)


aya_image =face_recognition.load_image_file("pics/aya.jpg")
aya_encoding = face_recognition.face_encodings(aya_image)[0]



sipa_image = face_recognition.load_image_file("pics/sipa.jpg")
sipa_encoding = face_recognition.face_encodings(sipa_image)[0]

known_face_encoding =[
    aya_encoding,
    sipa_encoding,
    
]

known_face_names = [
    "aya"
    "sipa"
]

students = known_face_names.copy()

face_location = []
face_encodings = []
face_names = []
s=True

now = datetime.now()
current_date = now.strftime("%d-%m-%y")

f =open(current_date+'.csv', 'w+' ,newline= '')
lnwriter = csv.writer(f)

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = small_frame[:, :, ::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_names.append(name)
                if name in known_face_names:
                    if name in students:
                        students.remove(name)
                        print(students)
                        current_time = now.strftime("%H-%M-%S")
                        lnwriter.writerow([name, current_time])

    cv2.imshow("attendence system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the file
video_capture.release()
cv2.destroyAllWindows()
f.close()