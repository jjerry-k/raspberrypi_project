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
    font = ImageFont.truetype("/home/jerry/Projects/raspberrypi_project/coral_usb/models/detection/D2Coding.ttf", size=22)
    draw.text(coordinate, text, fill='green', font=font)

def draw_no_detect(draw):
    width, height = draw.size
    draw = ImageDraw.Draw(draw)
    font = ImageFont.truetype("/home/jerry/Projects/raspberrypi_project/coral_usb/models/detection/D2Coding.ttf", size=22)
    draw.text((width//2-50, height//2),'Not Detect !' ,fill='red', font=font)

def draw_objects(draw, objs, labels):
    draw = ImageDraw.Draw(draw)
    font = ImageFont.truetype("/home/jerry/Projects/raspberrypi_project/coral_usb/models/detection/D2Coding.ttf", size=22)
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
    parser.add_argument('-s', '--src', required=True,
                        help='Input image path')
    args = parser.parse_args()

    dst = "/".join(args.src.split("/")[:-1])
    dst = os.path.join(dst, "output.jpg")

    # Load tflite model
    detector = edgetpu.make_interpreter(os.path.join(args.model, "detector.tflite"))
    detector.allocate_tensors()

    labels = dataset.read_label_file(os.path.join(args.model, 'labels.txt'))

    # Load image
    image = cv.imread(args.src)
    rows, cols, _ = image.shape
    
    # Run model
    # 이미지의 중심점을 기준으로 90도 회전 하면서 0.5배 Scale
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # M= cv.getRotationMatrix2D((cols/2, rows/2), 180, 1)
    # image = cv.warpAffine(image, M, (cols, rows))
    image = Image.fromarray(image, "RGB")

    _, scale = common.set_resized_input(detector, image.size, lambda size: image.resize(size))
    
    # Insert FPS
    curTime = time.time()

    detector.invoke()

    objs = detect.get_objects(detector, 0.3, scale)

    draw_image = image.copy()

    if not objs:
        draw_no_detect(draw_image)
    else:
        draw_objects(draw_image, objs, labels)

    draw_image = np.array(draw_image)
    draw_image = cv.cvtColor(draw_image, cv.COLOR_RGB2BGR)

    cv.imwrite(dst, draw_image)

if __name__ == '__main__':
    main()