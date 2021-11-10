# %%
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
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

def ssd_generate_anchors(opts: dict) -> np.ndarray:
    """This is a trimmed down version of the C++ code; all irrelevant parts
    have been removed.
    (reference: mediapipe/calculators/tflite/ssd_anchors_calculator.cc)
    """
    layer_id = 0
    num_layers = opts['num_layers']
    strides = opts['strides']
    assert len(strides) == num_layers
    input_height = opts['input_size_height']
    input_width = opts['input_size_width']
    anchor_offset_x = opts['anchor_offset_x']
    anchor_offset_y = opts['anchor_offset_y']
    interpolated_scale_aspect_ratio = opts['interpolated_scale_aspect_ratio']
    anchors = []
    while layer_id < num_layers:
        last_same_stride_layer = layer_id
        repeats = 0
        while (last_same_stride_layer < num_layers and
               strides[last_same_stride_layer] == strides[layer_id]):
            last_same_stride_layer += 1
            # aspect_ratios are added twice per iteration
            repeats += 2 if interpolated_scale_aspect_ratio == 1.0 else 1
        stride = strides[layer_id]
        feature_map_height = input_height // stride
        feature_map_width = input_width // stride
        for y in range(feature_map_height):
            y_center = (y + anchor_offset_y) / feature_map_height
            for x in range(feature_map_width):
                x_center = (x + anchor_offset_x) / feature_map_width
                for _ in range(repeats):
                    anchors.append((x_center, y_center))
        layer_id = last_same_stride_layer
    return np.array(anchors, dtype=np.float32)

def decode_boxes(raw_boxes: np.ndarray, anchors: np.ndarray, input_shape: list) -> np.ndarray:
    """Simplified version of
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    """
    # width == height so scale is the same across the board
    scale = input_shape[1]
    num_points = raw_boxes.shape[-1] // 2
    # scale all values (applies to positions, width, and height alike)
    boxes = raw_boxes.reshape(-1, num_points, 2) / scale
    # adjust center coordinates and key points to anchor positions
    boxes[:, 0] += anchors
    for i in range(2, num_points):
        boxes[:, i] += anchors
    # convert x_center, y_center, w, h to xmin, ymin, xmax, ymax
    center = np.array(boxes[:, 0])
    half_size = boxes[:, 1] / 2
    boxes[:, 0] = center - half_size
    boxes[:, 1] = center + half_size
    return boxes

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_sigmoid_scores(raw_scores: np.ndarray) -> np.ndarray:
        """Extracted loop from ProcessCPU (line 327) in
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        """
        return sigmoid(raw_scores)

SSD_OPTIONS_FRONT = {
    'num_layers': 4,
    'input_size_height': 128,
    'input_size_width': 128,
    'anchor_offset_x': 0.5,
    'anchor_offset_y': 0.5,
    'strides': [8, 16, 16, 16],
    'interpolated_scale_aspect_ratio': 1.0
}
# %%
# Load tflite model
interpreter = make_interpreter("/home/pi/project/raspberrypi_project/coral_usb/models/detection/face-detector-quantized_edgetpu.tflite")
interpreter.allocate_tensors()

# %%
# Set input data
_, height, width, _ = interpreter.get_input_details()[0]['shape']

# Using webcam
# cap = cv.VideoCapture(0)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
# ret, frame = cap.read()
# M= cv.getRotationMatrix2D((width/2, height/2),180, 1)
# frame = cv.warpAffine(frame, M, (width, height))

frame = cv.imread('/home/pi/project/raspberrypi_project/coral_usb/models/detection/test.jpg')
frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
frame = cv.resize(frame, (height, width))
set_input(interpreter, frame)
# %%
# Run model 
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

anchors = ssd_generate_anchors(SSD_OPTIONS_FRONT)

boxes = decode_boxes(raw_boxes, anchors, [height, width])

# %%
scores = get_sigmoid_scores(raw_scores)
# %%
boxes[157].shape, scores[0].shape, anchors.shape
# %%
boxes[157]*128

plt.imshow(frame)

plt.plot((boxes[157]*128).astype(int))