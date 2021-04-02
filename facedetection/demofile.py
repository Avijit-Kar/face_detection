import cv2
import os
path = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(path, 'pic.png')
cascade_path = os.path.join(path, 'haarcascade_frontalface_default.xml')

img = cv2.imread(image_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cascade_path)
faces = face_cascade.detectMultiScale(gray_img,  1.05,5)

for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y),(x+w, y+h), (1,200,2), 4)

#resized_img = cv2.resize(img, (1000, 500))
cv2.imshow("Galaxy", img)
cv2.waitKey(20000)
cv2.destroyAllWindows()