import cv2
import pickle
import numpy as np
import os
import sqlite3
from keras.models import load_model

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load model & histogram
model = load_model('cnn_model_keras2.h5')
with open("hist", "rb") as f:
    hist = pickle.load(f)

# ROI
x, y, w, h = 300, 100, 300, 300

# Input size
sample = cv2.imread('gestures/0/100.jpg', cv2.IMREAD_GRAYSCALE)
image_x, image_y = sample.shape

def keras_process_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = img.astype(np.float32) / 255.0
    return img.reshape((1, image_x, image_y, 1))

def get_text_from_db(g_id):
    conn = sqlite3.connect("gesture_db.db")
    row = conn.execute("SELECT g_name FROM gesture WHERE g_id=?", (g_id,)).fetchone()
    conn.close()
    return row[0] if row else ""

def get_frame_and_mask(frame):
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    back = cv2.calcBackProject([hsv], [0,1], hist, [0,180,0,256], 1)
    cv2.filter2D(back, -1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)), back)
    blur = cv2.GaussianBlur(back, (11,11), 0)
    blur = cv2.medianBlur(blur, 15)
    _, thresh_full = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    roi = thresh_full[y:y+h, x:x+w]
    contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return frame, contours, roi

def get_prediction(contour, frame):
    # 1) bounding box
    x1, y1, w1, h1 = cv2.boundingRect(contour)

    # 2) crop from grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop = gray[y1:y1+h1, x1:x1+w1]

    # 3) pad to square
    if w1 > h1:
        pad = (w1 - h1)//2
        crop = cv2.copyMakeBorder(crop, pad, pad, 0, 0, cv2.BORDER_CONSTANT, 0)
    else:
        pad = (h1 - w1)//2
        crop = cv2.copyMakeBorder(crop, 0, 0, pad, pad, cv2.BORDER_CONSTANT, 0)

    # 4) normalize & predict
    proc = cv2.resize(crop, (image_x, image_y)).astype(np.float32)/255.0
    proc = proc.reshape(1, image_x, image_y, 1)
    scores = model.predict(proc)[0]

    # DEBUG
    print("Raw network scores:", np.round(scores, 3))

    cls = int(np.argmax(scores))

    # 5) lookup label using exactly that index
    label = get_text_from_db(cls)

    # 6) confidence filter
    return label if scores[cls] > 0.7 else ""

def recognize():
    cam = cv2.VideoCapture(0)
    if not cam.read()[0]:
        cam = cv2.VideoCapture(1)

    while True:
        ret, raw = cam.read()
        if not ret:
            break

        raw = cv2.resize(raw, (640,480))
        frame, contours, mask = get_frame_and_mask(raw)

        pred = ""
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 10000:
                pred = get_prediction(cnt, raw)
                print("Predicted:", pred)

        # display
        black = np.zeros((480,640,3), np.uint8)
        cv2.putText(black, f"Prediction: {pred}", (30,240),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, (255,255,255), 2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        disp = np.hstack((frame, black))
        cv2.imshow("Sign Recognition", disp)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize()
