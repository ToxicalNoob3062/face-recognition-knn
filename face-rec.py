import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
import os


dataset_path  = "./data/"
faceData = []
labels = []
nameMap = {}

classId =  0

for f in  os.listdir(dataset_path):
    if f.endswith(".npy"):

        nameMap[classId] = f.split(".")[0]

        # X Value
        dataItem = np.load(dataset_path + f)
        faceData.append(dataItem)
        m = dataItem.shape[0]

        # Y Value
        target = classId * np.ones((m,))
        classId +=1
        labels.append(target)

# concatinate all smaples into a large table
X = np.concatenate(faceData, axis=0)

# concantinate all targets into a large table
# reshaping the target from horizontal to vertical array
Y = np.concatenate(labels, axis=0).reshape(-1,)

print(X.shape)
print(Y.shape)
print(nameMap)

# use sk learn knn classifier
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X, Y)

# predict the label of a new face using camera;
ip = '192.168.86.30'
port = '4747'
format = 'video'
resolution = '1920x1080'
url = f'http://{ip}:{port}/{format}?{resolution}'
cam = cv2.VideoCapture(url)


model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
offset = 10
while True:
    success, img = cam.read()
    if not success:
        print("Failed to read from camera")
        break  # Avoid using exit() inside the loop

    # Apply screen mirroring (flip horizontally)
    mirrored_img = cv2.flip(img, 1)

    faces = model.detectMultiScale(mirrored_img, 1.3, 5)

    # if face detected
    for f in faces:
        x, y, w, h = f
        cv2.rectangle(mirrored_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = mirrored_img[y-offset:y+h+offset, x-offset:x+w+offset]
        cropped_face = cv2.resize(cropped_face, (100, 100))
        prediction = clf.predict([cropped_face.flatten()])
        name = nameMap[prediction[0]]
        cv2.putText(mirrored_img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the mirrored image with the name
    cv2.imshow('Mirrored Image', mirrored_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# release resources
cam.release()
cv2.destroyAllWindows()
