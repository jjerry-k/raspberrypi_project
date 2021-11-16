import time
import argparse
import cv2 as cv
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

def draw_text(draw, text, coordinate=(0,0)):
    draw = ImageDraw.Draw(draw)
    font = ImageFont.truetype("D2Coding.ttf", size=22)
    draw.text(coordinate, text, fill='green', font=font)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--model', required=True, help='File path of .tflite file.')
    parser.add_argument(
        '-r', '--roi', required=True, help='ROI [Face, Top, Whole]')
    args = parser.parse_args()

    if args.roi.lower() == 'top':
        _NUM_KEYPOINTS = 11
    elif args.roi.lower() == 'face':
        _NUM_KEYPOINTS = 5
    else: 
        _NUM_KEYPOINTS = 17

    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    # Load webcam
    prevTime = 0
    cap = cv.VideoCapture(0)
    rows, cols = 320, 320
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 320)

    # Run model
    while(True):
        _, image = cap.read()
        
        # 이미지의 중심점을 기준으로 90도 회전 하면서 0.5배 Scale
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        M= cv.getRotationMatrix2D((cols/2, rows/2),180, 1)
        image = cv.warpAffine(image, M, (cols, rows))
        image = Image.fromarray(image, "RGB")

        # resized_img = image.resize(common.input_size(interpreter), Image.ANTIALIAS)
        # common.set_input(interpreter, resized_img)
        common.set_resized_input(interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
        # Insert FPS
        curTime = time.time()
        
        interpreter.invoke()
        
        pose = common.output_tensor(interpreter, 0).copy().reshape(17, 3)
        
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for i in range(0, _NUM_KEYPOINTS):
            draw.ellipse(
                xy=[
                    pose[i][1] * width - 2, pose[i][0] * height - 2,
                    pose[i][1] * width + 2, pose[i][0] * height + 2
                ],
                fill=(255, 0, 0))

        sec = curTime - prevTime
        prevTime = curTime
        fps = 1/(sec)
        str = "FPS : %0.1f" % fps
        draw_text(image, str, (0, 0))

        image = np.array(image)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        
        # Display frame
        cv.imshow("Frame", image)
        
        key = cv.waitKey(1) & 0xff
        if key==27:
            # Stop using ESC
            break
if __name__ == '__main__':
    main()