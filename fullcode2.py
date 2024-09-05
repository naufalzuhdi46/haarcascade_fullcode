import cv2
import os
import numpy as np
from PIL import Image
import gradio as gr

# Membuat direktori 'dataset' jika belum ada
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Membuka webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n enter user id (string) and press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look at the camera and wait ...")
count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27 or count >= 10:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def normalize_images(images):
    return [cv2.equalizeHist(img) for img in images]

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples, ids, name_to_id = [], [], {}
    current_id = 0

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        name = os.path.split(imagePath)[-1].split(".")[1]
        if name not in name_to_id:
            name_to_id[name] = current_id
            current_id += 1
        id = name_to_id[name]
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    faceSamples = normalize_images(faceSamples)
    return faceSamples, ids, name_to_id

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids, name_to_id = getImagesAndLabels('dataset')
recognizer.train(faces, np.array(ids))
recognizer.write('trainer/trainer.yml')

if not os.path.exists('trainer'):
    os.makedirs('trainer')

with open('trainer/name_to_id.txt', 'w') as f:
    for name, id in name_to_id.items():
        f.write(f"{name},{id}\n")

print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

name_to_id, id_to_name = {}, {}
with open('trainer/name_to_id.txt', 'r') as f:
    for line in f:
        name, id = line.strip().split(',')
        name_to_id[name] = int(id)
        id_to_name[int(id)] = name

minW, minH = 0.1 * 640, 0.1 * 480

def detect_faces(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))
        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            name = id_to_name.get(id, "unknown")
            confidence_text = f"  {round(140 - confidence)}%" if confidence < 100 else f"  {round(100 - confidence)}%"
            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, name, (x + 5, y - 5), font, 1, color, 2)
            cv2.putText(image, confidence_text, (x + 5, y + h - 5), font, 1, color, 1)
        return image
    except Exception as e:
        print(f"Error in detect_faces: {e}")
        return image

iface = gr.Interface(fn=detect_faces, inputs="image", outputs="image")
iface.launch()