# %%
import time
import cv2 as cv
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import platform

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]

from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import detect

# %%
# Load tflite model
interpreter = edgetpu.make_interpreter("./ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
interpreter.allocate_tensors()

labels = dataset.read_label_file('./face_labels.txt')

# Load image
# image = Image.open('./test.jpg')

# Load webcam
prevTime = 0
cap = cv.VideoCapture(0)
rows, cols = 320, 320
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 320)

# %%
# Run model
while(True):
    ret, image = cap.read()
    
    # 이미지의 중심점을 기준으로 90도 회전 하면서 0.5배 Scale
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    M= cv.getRotationMatrix2D((cols/2, rows/2),180, 1)
    image = cv.warpAffine(image, M, (cols, rows))
    image = Image.fromarray(image, "RGB")

    _, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
    
    # Insert FPS
    curTime = time.time()

    interpreter.invoke()

    # %%
    objs = detect.get_objects(interpreter, 0.3, scale)
    # %%
    if not objs:
        print('No objects detected')

    def draw_objects(draw, objs, labels):
        for obj in objs:
            bbox = obj.bbox
            draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                        outline='green')
            font = ImageFont.truetype("D2Coding.ttf", size=22)
            draw.text((bbox.xmin + 10, bbox.ymin + 10),
                    '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                    fill='green', font=font)

    draw_image = image.copy()
    draw_objects(ImageDraw.Draw(draw_image), objs, labels)
    draw_image = np.array(draw_image)
    draw_image = cv.cvtColor(draw_image, cv.COLOR_RGB2BGR)
    
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1/(sec)
    str = "FPS : %0.1f" % fps
    cv.putText(draw_image, str, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    # Display frame
    cv.imshow("Frame", draw_image)
    
    key = cv.waitKey(1) & 0xff
    if key==27:
        # Stop using ESC
        break