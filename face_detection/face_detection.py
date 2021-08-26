from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import cv2 as cv

camera = PiCamera()
camera.resolution = (256, 256)
camera.framerate = 30
camera.rotation = 180
rawCapture = PiRGBArray(camera, size=camera.resolution)

face_cascade = cv.CascadeClassifier('./haar/haarcascade_frontalface.xml')

def detect(color_img, detector):
    gray = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(color_img,(x,y),(x+w,y+h),(255,0,0),2)
    return color_img

time.sleep(0.1)

prevTime = 0

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = frame.array
    
    # Insert FPS
    curTime = time.time()
    frame = detect(frame, face_cascade)
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1/(sec)
    str = "FPS : %0.1f" % fps
    cv.putText(frame, str, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    # Display frame
    cv.imshow("Frame", frame)
    
    rawCapture.truncate(0)
    key = cv.waitKey(1) & 0xff
    if key==27:
        # Stop using ESC
        break