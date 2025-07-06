import os
import cv2
import pickle
import numpy as np

# CONFIGURE ▶
TARGET_DIR = "gestures"       # where we want 0/,1/,2/…
HIST_FILE  = "hist"           # your precomputed histogram

# Load histogram
with open(HIST_FILE, "rb") as f:
    hist = pickle.load(f)

def process(img):
    """Histogram backproject → clean → Otsu threshold → binary mask."""
    hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    back   = cv2.calcBackProject([hsv],[0,1],hist,[0,180,0,256],1)
    disc   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    cv2.filter2D(back,-1,disc,back)
    blur   = cv2.GaussianBlur(back,(11,11),0)
    blur   = cv2.medianBlur(blur,15)
    _,mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return mask

# Walk raw dataset
for class_id in os.listdir(RAW_DIR):
    class_path = os.path.join(RAW_DIR, class_id)
    if not os.path.isdir(class_path): 
        continue

    # Prepare target folder
    out_folder = os.path.join(TARGET_DIR, class_id)
    os.makedirs(out_folder, exist_ok=True)

    # Enumerate every image in this class
    idx = 1
    for fname in os.listdir(class_path):
        if not fname.lower().endswith((".jpg",".jpeg",".png",".bmp")):
            continue

        src = os.path.join(class_path, fname)
        img = cv2.imread(src)
        if img is None:
            print(f"⚠️  Skipping unreadable {src}")
            continue

        mask = process(img)
        # Save as e.g. gestures/5/1.jpg, 5/2.jpg, …
        dst  = os.path.join(out_folder, f"{idx}.jpg")
        cv2.imwrite(dst, mask)
        idx += 1

    print(f"Processed class {class_id}: saved {idx-1} images.")
