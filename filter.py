#import necessary modules

from classes import *
import cv2


#Uses Laptop Camera
camera = cv2.VideoCapture(0)

#Raises an error if unable to open camera
if not camera.isOpened():
    raise IOError("Cannot Open Camera")
if camera.isOpened():
    print("Camera Opening...", "\n")
    print("Press ESC to Close Camera", "\n")


while True:
    #reads evry frame
    ret, frame = camera.read()

    #creates an object with the class Frame and Face
    frameobject = Frame(frame)
    faceobject = Face(frame)

    #puts a christmas Hat over the head
    faceobject.putHat()

    #adds Christmas Lights, and Snow in the Background
    frameobject.putLights()

    frameobject.putSnow()

    #draws a beard, a mustache and colors the eyebrows and lips white
    frame =  faceobject.Santa()

    #displays every frame
    cv2.imshow("Snapchat Filter", frame)

    #breaks loop when ESC if pressed
    if cv2.waitKey(1) == 27:
        break

#closes camera
print("Closing Camera")
camera.release()
cv2.destroyAllWindows()
