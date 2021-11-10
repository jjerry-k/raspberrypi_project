# %%
import cv2 as cv

import time

from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import platform

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(EDGETPU_SHARED_LIB,
                                {'device': device[0]} if device else {})
        ])

def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, 3)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]

def set_input(interpreter, data):
    """Copies data to input tensor."""
    input_tensor(interpreter)[:, :] = data

def output_tensor(interpreter):
    """Returns dequantized output tensor."""
    output_details = interpreter.get_output_details()[0]
    output_data = np.squeeze(interpreter.tensor(output_details['index'])())
    scale, zero_point = output_details['quantization']
    return scale * (output_data - zero_point)
# %%
interpreter = make_interpreter("/home/pi/project/raspberrypi_project/coral_usb/models/detection/face-detector-quantized_edgetpu.tflite")
interpreter.allocate_tensors()

# %%
_, height, width, _ = interpreter.get_input_details()[0]['shape']

frame = cv.imread("/home/pi/project/raspberrypi_project/coral_usb/models/detection/test.jpg")
frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
frame = cv.resize(frame, (height, width))
set_input(interpreter, frame)
# %%
interpreter.invoke()

# %%
output_details = interpreter.get_output_details()
# %%
output_details[0]

# %%
input_index = interpreter.get_input_details()[0]['index']
input_shape = interpreter.get_input_details()[0]['shape']
bbox_index = interpreter.get_output_details()[0]['index']
score_index = interpreter.get_output_details()[1]['index']
# %%
raw_boxes = interpreter.get_tensor(bbox_index)
raw_scores = interpreter.get_tensor(score_index)
# %%
raw_boxes.shape, raw_scores.shape

# %%
def decode_boxes(raw_boxes: np.ndarray) -> np.ndarray:
    """Simplified version of
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    """
    # width == height so scale is the same across the board
    scale = input_shape[1]
    num_points = raw_boxes.shape[-1] // 2
    # scale all values (applies to positions, width, and height alike)
    boxes = raw_boxes.reshape(-1, num_points, 2) / scale
    # adjust center coordinates and key points to anchor positions
    boxes[:, 0] += self.anchors
    for i in range(2, num_points):
        boxes[:, i] += self.anchors
    # convert x_center, y_center, w, h to xmin, ymin, xmax, ymax
    center = np.array(boxes[:, 0])
    half_size = boxes[:, 1] / 2
    boxes[:, 0] = center - half_size
    boxes[:, 1] = center + half_size
    return boxes
