from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import cv2 as cv
import mediapipe as mp

camera = PiCamera()
camera.resolution = (256, 256)
camera.framerate = 30
camera.rotation = 180
camera.hflip = True
rawCapture = PiRGBArray(camera, size=camera.resolution)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

idx = 0
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(color=(128,128,128), thickness=1, circle_radius=1)


prevTime = 0
for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
    frame = frame.array
    
    # Insert FPS
    curTime = time.time()
    results = pose.process(frame)
    mp_drawing.draw_landmarks(
        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1/(sec)
    str = "FPS : %0.1f" % fps
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.putText(frame, str, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    # Display frame
    cv.imshow('MediaPipe Pose', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    rawCapture.truncate(0)
    key = cv.waitKey(1) & 0xff
    if key==27:
        # Stop using ESC
        break
pose.close()