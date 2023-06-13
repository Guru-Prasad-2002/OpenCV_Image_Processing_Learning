import cv2
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(current_dir, "Resources/HaarCascade_frontal_face_default.xml")

# Load the cascade classifier
faceCascade = cv2.CascadeClassifier(cascade_path)


# Face detection using Haar cascades
# img=cv2.imread("Resources\lena.png")
img=cv2.imread("Resources\Faces.png")
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(imgGray,1.1,4)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow("Result",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
