import os
import time
import argparse
import cv2 as cv
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import detect

def draw_text(draw, text, coordinate=(0,0)):
    draw = ImageDraw.Draw(draw)
    font = ImageFont.truetype("D2Coding.ttf", size=22)
    draw.text(coordinate, text, fill='green', font=font)

def draw_no_detect(draw):
    width, height = draw.size
    draw = ImageDraw.Draw(draw)
    font = ImageFont.truetype("D2Coding.ttf", size=22)
    draw.text((width//2-50, height//2),'Not Detect !' ,fill='red', font=font)

def draw_objects(draw, objs, labels):
    draw = ImageDraw.Draw(draw)
    font = ImageFont.truetype("D2Coding.ttf", size=22)
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                    outline='green')
        draw.text((bbox.xmin+5, bbox.ymin-45),
                '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                fill='green', font=font)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True,
                        help='Model directory')
    args = parser.parse_args()


    # Load tflite model
    detector = edgetpu.make_interpreter(os.path.join(args.model, "detector.tflite"))
    detector.allocate_tensors()

    labels = dataset.read_label_file(os.path.join(args.model, 'labels.txt'))

    # Load webcam
    prevTime = 0
    cap = cv.VideoCapture(0)
    rows, cols = 320, 320
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 320)

    # Run model
    while(True):
        ret, image = cap.read()
        
        # 이미지의 중심점을 기준으로 90도 회전 하면서 0.5배 Scale
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        M= cv.getRotationMatrix2D((cols/2, rows/2),180, 1)
        image = cv.warpAffine(image, M, (cols, rows))
        image = Image.fromarray(image, "RGB")

        _, scale = common.set_resized_input(detector, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
        
        # Insert FPS
        curTime = time.time()

        detector.invoke()

        objs = detect.get_objects(detector, 0.6, scale)

        draw_image = image.copy()

        if not objs:
            draw_no_detect(draw_image)
        else:
            draw_objects(draw_image, objs, labels)
        
        sec = curTime - prevTime
        prevTime = curTime
        fps = 1/(sec)
        str = "FPS : %0.1f" % fps
        draw_text(draw_image, str, (0, 0))

        draw_image = np.array(draw_image)
        draw_image = cv.cvtColor(draw_image, cv.COLOR_RGB2BGR)
        
        # Display frame
        cv.imshow("Frame", draw_image)
        
        key = cv.waitKey(1) & 0xff
        if key==27:
            # Stop using ESC
            break

if __name__ == '__main__':
    main()