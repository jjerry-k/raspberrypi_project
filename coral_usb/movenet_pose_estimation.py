from picamera import PiCamera
from picamera.array import PiRGBArray
import time

import cv2 as cv
import numpy as np

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

# Camera setting
camera = PiCamera()
camera.resolution = (256, 256)
camera.framerate = 30
camera.rotation = 180
camera.hflip = True
rawCapture = PiRGBArray(camera, size=camera.resolution)

_NUM_KEYPOINTS = 17
_PLOT_KEYPOINTS = 11
time.sleep(0.1)



    
interpreter = make_interpreter("models/pose_estimation/movenet_single_pose_thunder_ptq_edgetpu.tflite")
interpreter.allocate_tensors()


prevTime = 0

for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
    frame = frame.array
    curTime = time.time()
    
    frame = Image.fromarray(frame)
    frame = frame.resize(common.input_size(interpreter), Image.ANTIALIAS)
    common.set_input(interpreter, frame)
    interpreter.invoke()
    
    pose = common.output_tensor(interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)
    draw = ImageDraw.Draw(frame)
    width, height = frame.size
    for i in range(0, _PLOT_KEYPOINTS):
        draw.ellipse(
            xy=[
                pose[i][1] * width - 2, pose[i][0] * height - 2,
                pose[i][1] * width + 2, pose[i][0] * height + 2
            ],
            fill=(255, 0, 0))
    frame = np.array(frame)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    sec = curTime - prevTime
    prevTime = curTime
    fps = 1/(sec)
    str = "FPS : %0.1f" % fps
    cv.putText(frame, str, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    cv.imshow("Frame", frame)
    rawCapture.truncate(0)
    key = cv.waitKey(1) & 0xff
    if key==27:
        # Stop using ESC
        break