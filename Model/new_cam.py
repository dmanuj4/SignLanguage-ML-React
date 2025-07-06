import cv2

# Try different indices (0, 1, 2…) and backends (CAP_DSHOW for Windows)
for idx in (0, 1, 2):
    cam = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # on Linux/Mac you can omit CAP_DSHOW
    ret, frame = cam.read()
    print(f"Index {idx} — frame OK? {ret}")
    if ret:
        cv2.imshow(f"Camera {idx}", frame)
        cv2.waitKey(10000)   # show for 2 seconds
        cv2.destroyAllWindows()
    cam.release()
