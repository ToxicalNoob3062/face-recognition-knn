import cv2

ip = '192.168.86.30'
port = '4747'
format = 'video'
resolution = '1920x1080'
url = f'http://{ip}:{port}/{format}?{resolution}'
cam = cv2.VideoCapture(url)

while True:
    success, img = cam.read()
    if not success:
        print("Failed to read from camera")
        break  # Avoid using exit() inside the loop

    # Apply screen mirroring (flip horizontally)
    mirrored_img = cv2.flip(img, 1)

    cv2.imshow("Mirrored Camera Feed", mirrored_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cam.release()
cv2.destroyAllWindows()
