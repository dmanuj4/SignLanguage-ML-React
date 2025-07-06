import os
import cv2
import pickle
import numpy as np

# Paths (adjust as needed)
OUTPUT_DIR = "gestures"            # where processed silhouettes will go
HIST_FILE = "hist"                 # histogram from set_hand_histogram.py

# Load precomputed hand-color histogram
with open(HIST_FILE, "rb") as f:
    hist = pickle.load(f)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_image(img: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """
    Convert a color image to a binary hand silhouette using histogram backprojection.
    Returns a single-channel binary mask (0 or 255).
    """
    # Backprojection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    # Morphological filtering and blurring
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    cv2.filter2D(backproj, -1, disc, backproj)
    blur = cv2.GaussianBlur(backproj, (11, 11), 0)
    blur = cv2.medianBlur(blur, 15)
    # Otsu thresholding to get binary mask
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# Walk through raw dataset
for root, dirs, files in os.walk(RAW_DATASET_DIR):
    # Compute relative subfolder path
    rel_path = os.path.relpath(root, RAW_DATASET_DIR)
    out_dir = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(out_dir, exist_ok=True)

    for filename in files:
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        in_path = os.path.join(root, filename)
        out_path = os.path.join(out_dir, filename)

        # Read input image
        img = cv2.imread(in_path)
        if img is None:
            print(f"Warning: could not read {in_path}, skipping")
            continue

        # Process and save silhouette
        silhouette = process_image(img, hist)
        cv2.imwrite(out_path, silhouette)
        print(f"Saved silhouette: {out_path}")

print("Preprocessing complete. All gestures saved under:", OUTPUT_DIR)
