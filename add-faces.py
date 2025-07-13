import pickle
import cv2
import numpy as np
import os

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_datas = []
i = 0

name = input("Enter your name : ")

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
        crop_image = gray[y:y+h, x:x+w]
        resized_image = cv2.resize(crop_image, (50, 50))
        if len(face_datas) <= 100 and i % 10 == 0:
            face_datas.append(resized_image)
        i += 1
        cv2.putText(frame, str(len(face_datas)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cv2.imshow('Face Detection', frame)
    key = cv2.waitKey(1)
    if key == ord('q') or len(face_datas) == 100:
        break

video.release()
cv2.destroyAllWindows()

face_datas = np.asarray(face_datas)
# FIX: Actually apply the reshape
face_datas = face_datas.reshape(100, -1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

names_path = os.path.join(DATA_DIR, "names.pkl")
faces_path = os.path.join(DATA_DIR, "face_datas.pkl")

# Handle names
if not os.path.exists(names_path):
    names = [name] * 100
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 100
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)

# Handle face data
if not os.path.exists(faces_path):
    with open(faces_path, 'wb') as f:
        pickle.dump(face_datas, f)
else:
    with open(faces_path, 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, face_datas, axis=0)
    # FIX: Save the combined faces, not just face_datas
    with open(faces_path, 'wb') as f:
        pickle.dump(faces, f)