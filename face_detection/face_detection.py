import time
import cv2 as cv

face_cascade = cv.CascadeClassifier('./haar/haarcascade_frontalface.xml')

def detect(color_img, detector):
    gray = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(color_img,(x,y),(x+w,y+h),(255,0,0),2)
    return color_img

time.sleep(0.1)

prevTime = 0
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 320)
while(True):
    ret, frame = cap.read()
    rows, cols = frame.shape[:2]

    # 이미지의 중심점을 기준으로 90도 회전 하면서 0.5배 Scale
    M= cv.getRotationMatrix2D((cols/2, rows/2),180, 1)

    frame = cv.warpAffine(frame, M, (cols, rows))
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
    
    key = cv.waitKey(1) & 0xff
    if key==27:
        # Stop using ESC
        break