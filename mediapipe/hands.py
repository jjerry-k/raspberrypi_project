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
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
	min_detection_confidence=0.7, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(color=(128,128,128), thickness=1, circle_radius=1)


prevTime = 0
for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
    frame = frame.array
    
    # Insert FPS
    curTime = time.time()
    results = hands.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
				frame, hand_landmarks, 
                mp_hands.HAND_CONNECTIONS)
    
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1/(sec)
    str = "FPS : %0.1f" % fps
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.putText(frame, str, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    # Display frame
    cv.imshow('MediaPipe Hands', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    rawCapture.truncate(0)
    key = cv.waitKey(1) & 0xff
    if key==27:
        # Stop using ESC
        break
hands.close()