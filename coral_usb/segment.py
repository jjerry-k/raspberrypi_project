from picamera import PiCamera
from picamera.array import PiRGBArray
import time

import cv2 as cv
import numpy as np
from PIL import Image

from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter

def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    indices = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((indices >> channel) & 1) << shift
        indices >>= 3

    return colormap


def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

# Camera setting
camera = PiCamera()
camera.resolution = (256, 256)
camera.framerate = 30
camera.rotation = 180
camera.hflip = True
rawCapture = PiRGBArray(camera, size=camera.resolution)

interpreter = make_interpreter("models/segmentation/deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite", device=':0')
interpreter.allocate_tensors()

prevTime = 0

for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
    origin_frame = frame.array
    curTime = time.time()

    frame = Image.fromarray(origin_frame)
    frame = frame.resize(common.input_size(interpreter), Image.ANTIALIAS)
    common.set_input(interpreter, frame)
    interpreter.invoke()
    
    result = segment.get_output(interpreter)

    if len(result.shape) == 3:
        result = np.argmax(result, axis=-1)

    mask = label_to_color_image(result).astype(np.uint8)
    mask = cv.resize(mask, dsize=camera.resolution)
    
    result = np.concatenate([cv.cvtColor(origin_frame, cv.COLOR_RGB2BGR), mask], axis=1)
    
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1/(sec)
    str = "FPS : %0.1f" % fps
    cv.putText(result, str, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    cv.imshow("Frame", result)
    rawCapture.truncate(0)
    key = cv.waitKey(1) & 0xff
    if key==27:
        # Stop using ESC
        break