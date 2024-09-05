import cv2
import os
import numpy as np
from PIL import Image
import gradio as gr
import firebase_admin
from firebase_admin import credentials, firestore

# Inisialisasi Firebase
cred = credentials.Certificate('C:/Users/USER/Pictures/Face-Recognition-Haar-Cascade-master/fadlock-db-b6c009b12620.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Fungsi untuk memeriksa apakah pengguna sudah terdaftar
def check_user_registered(user_id):
    doc_ref = db.collection('users').document(user_id)
    doc = doc_ref.get()
    return doc.exists

# Membuat direktori 'dataset' jika belum ada
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Membuka webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Meminta input user_id
user_id = input('\n enter user id (string) and press <return> ==>  ')

if check_user_registered(user_id):
    print("\n [INFO] User already registered. Fetching data...")
else:
    print("\n [INFO] Initializing face capture. Look at the camera and wait ...")
    count = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg", gray[y:y + h, x:x + w])
            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff
        if k == 27 or count >= 10:
            break

    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

    # Simpan data pengguna ke Firebase
    db.collection('users').document(user_id).set({'registered': True})

print("\n [INFO] Proceeding with further processing...")
# Tambahkan logika lanjutan di sini jika diperlukan
