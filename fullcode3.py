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

# Fungsi untuk mendapatkan data pengguna dari Firebase
def get_user_data(user_id):
    doc_ref = db.collection('users').document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    else:
        return None

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
# Normalisasi dan Optimalisasi Data
def preprocess_data():
    dataset_path = 'dataset/'
    image_files = os.listdir(dataset_path)

    images = []
    labels = []

    for image_file in image_files:
        if image_file.startswith('User.') and image_file.endswith('.jpg'):
            try:
                user_id = int(image_file.split('.')[1])  # Ambil user_id dari nama file
                image_path = os.path.join(dataset_path, image_file)
                pil_image = Image.open(image_path).convert('L')  # Konversi ke grayscale
                image_np = np.array(pil_image, 'uint8')
                images.append(image_np)
                labels.append(user_id)
            except ValueError:
                print(f"Skipping invalid file format: {image_file}")
    
    labels = np.array(labels)

    # Normalisasi, contoh sederhana: resize gambar
    resized_images = [cv2.resize(image, (100, 100)) for image in images]

    return resized_images, labels

# Training Pengenalan Wajah
def train_face_recognition(images, labels):
    # Training recognizer (contoh menggunakan LBPH)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)
    recognizer.save('trainer.yml')

    print("\n [INFO] Training completed and model saved.")

    # Simpan informasi model yang dilatih ke Firebase
    db.collection('users').document(user_id).update({'trained': True})

# Panggil fungsi untuk preprocessing dan training
images, labels = preprocess_data()
train_face_recognition(images, labels)

# Fungsi untuk mendapatkan model yang dilatih
def load_trained_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    return recognizer

# Fungsi untuk mendapatkan akurasi dari model
def get_accuracy(recognizer):
    correct = 0
    total = 0

    for image_file in os.listdir('dataset/'):
        if image_file.startswith('User.'):
            user_id = int(image_file.split('.')[1])
            image_path = os.path.join('dataset', image_file)
            pil_image = Image.open(image_path).convert('L')
            image_np = np.array(pil_image, 'uint8')

            label, confidence = recognizer.predict(image_np)
            if label == user_id:
                correct += 1
            total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy

# Fungsi Gradio untuk menampilkan tampilan wajah dan akurasi
def recognize_face(image):
    # Konversi gambar Gradio menjadi gambar OpenCV
    img = np.array(image).astype(np.uint8)

    # Load model pengenalan wajah yang sudah dilatih
    recognizer = load_trained_model()

    # Deteksi wajah pada gambar
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        label, confidence = recognizer.predict(roi_gray)
        accuracy = round((1 - confidence / 400) * 100, 2)
        cv2.putText(img, f'{accuracy}% Accurate', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img[:, :, ::-1]  # Kembalikan gambar dalam format RGB untuk Gradio

# Menampilkan UI Gradio untuk memunculkan tampilan wajah dan nilai akurasi
iface = gr.Interface(recognize_face, gr.inputs.Image(shape=(480, 640)), "image")
iface.launch()
