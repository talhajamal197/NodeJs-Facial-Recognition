import cv2
import os
import numpy as np 
import script1 as fr 



def run():
    test_img = cv2.imread('TestImages/f.jpg')
    faces_detected,gray_img = fr.faceDetection(test_img)


    faces,faceID = fr.labels_for_training('trainingImages')
    face_recognizer = fr.train_classifier(faces,faceID)
    face_recognizer.write('trainingData.yml')

  
    
    i=0
    for face in faces_detected:
        (x,y,w,h) = face
        roi_gray = gray_img[y:y+h,x:x+h]
        
        
        label,confidence = face_recognizer.predict(roi_gray)
        print("confidence: ",confidence)
        print(" Name :")
        if (label==0):
            print("fahad")
        else :
            print("qadri")
        break
    





run()