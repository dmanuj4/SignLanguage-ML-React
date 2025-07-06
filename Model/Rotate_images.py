import cv2
import os

def flip_images():
    gest_folder = "gestures"

    for g_id in os.listdir(gest_folder):
        class_dir = os.path.join(gest_folder, g_id)
        if not os.path.isdir(class_dir):
            continue

        # Gather and sort all .jpg files by their numeric name
        files = [f for f in os.listdir(class_dir) if f.lower().endswith('.jpg')]
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))

        orig_count = len(files)
        print(f"Class {g_id}: flipping {orig_count} images…")

        for idx, filename in enumerate(files, start=1):
            src_path = os.path.join(class_dir, filename)
            dst_name = f"{orig_count + idx}.jpg"
            dst_path = os.path.join(class_dir, dst_name)

            img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  ⚠️  Skipped unreadable {filename}")
                continue

            flipped = cv2.flip(img, 1)
            cv2.imwrite(dst_path, flipped)
            print(f"  Saved {dst_name}")

    print("All classes processed.")

if __name__ == "__main__":
    flip_images()
