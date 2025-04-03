import cv2
import numpy as np

ip = '192.168.86.30'
port = '4747'
format = 'video'
resolution = '1920x1080'
url = f'http://{ip}:{port}/{format}?{resolution}'
cam = cv2.VideoCapture(url)

file_name = input("Enter the name of the person: ")
dataset_path = "./data/"

model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
offset = 10

faceData = []
count = 0

while True:
    success, img = cam.read()
    if not success:
        print("Failed to read from camera")
        break  # Avoid using exit() inside the loop

    # Apply screen mirroring (flip horizontally)
    mirrored_img = cv2.flip(img, 1)

    faces = model.detectMultiScale(mirrored_img, 1.3, 5)

    # Largest face first
    faces = sorted(faces, key=lambda x: x[2] * x[3])

    # if face detected
    if (len(faces)>0):
        x, y, w, h = faces[-1]
        cv2.rectangle(mirrored_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # crop the face pixels from main image
        cropped_face = mirrored_img[y-offset:y+h+offset, x-offset:x+w+offset]
        # reshape the cropped image
        cropped_face = cv2.resize(cropped_face, (100, 100))
        count += 1
        # consider after every 10 frames
        if count % 10 ==0:
            faceData.append(cropped_face)
            print("Face Captured: ", len(faceData))

    cv2.imshow("Mirrored Camera Feed", mirrored_img)


    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# convert face data into an numpy array
faceData = np.asarray(faceData)
m = faceData.shape[0]

# from 4D make it 2D
faceData = faceData.reshape((m, -1))
print(faceData.shape)

# save the data as binary
file = dataset_path + file_name + ".npy"
np.save(file,faceData)
print("Data Successfully saved at "+ file)

# release resources
cam.release()
cv2.destroyAllWindows()
