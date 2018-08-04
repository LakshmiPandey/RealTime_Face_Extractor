import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

def face_detect(image, size = 0.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is():
        return (image)
    for x, y, w, h in faces():
       x = x - 50
       y = y - 50
       w = w + 50
       h = h + 50
       cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
       roi_gray = gray[y:y+h, x:x+w]
       roi_color = image[y:y+h, x:x+w]
       eyes  = eye_classifier.MultiScale(roi_gray)
       for ex, ey, ew, eh in eyes:
           cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (127, 255, 128), 2)
           
           
    roi_color = cv2.flip(roi_color, 1)
    return image      
 
    
cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    cv2.imshow("FACE EXTRACTOR", face_detect(frame))
    if(cv2.waitKey(1)== 13):      # 13 is for enter
        break

cap.release()
cv2.destroyAllWindows()   

