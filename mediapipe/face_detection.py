# from picamera import PiCamera
# from picamera.array import PiRGBArray
import time
import cv2 as cv
import mediapipe as mp

# camera = PiCamera()
# camera.resolution = (256, 256)
# camera.framerate = 30
# camera.rotation = 180
# camera.hflip = True
# rawCapture = PiRGBArray(camera, size=camera.resolution)

mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(color=(128,128,128), thickness=1, circle_radius=1)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 320)

prevTime = 0
while(True):
    ret, frame = cap.read()
    rows, cols = frame.shape[:2]
    # 이미지의 중심점을 기준으로 90도 회전 하면서 0.5배 Scale
    M= cv.getRotationMatrix2D((cols/2, rows/2),180, 1)
    frame = cv.warpAffine(frame, M, (cols, rows))

    # Insert FPS
    curTime = time.time()
    results = face_detection.process(frame)
    if results.detections:
        # annotated_image = frame.copy()
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)
    
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1/(sec)
    str = "FPS : %0.1f" % fps
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.putText(frame, str, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    # Display frame
    cv.imshow('MediaPipe FaceDetection', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    key = cv.waitKey(1) & 0xff
    if key==27:
        # Stop using ESC
        break
face_detection.close()