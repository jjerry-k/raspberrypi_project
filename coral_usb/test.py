import os
import pathlib
import time
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite')
label_file = os.path.join(script_dir, 'models/inat_bird_labels.txt')
image_file = os.path.join(script_dir, 'images/parrot.jpg')

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

# Resize the image
size = common.input_size(interpreter)
image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)
common.set_input(interpreter, image)

# time check
for _ in range(5):
    start = time.perf_counter()
    # Run an inference    
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)

    # Print the result
    labels = dataset.read_label_file(label_file)
    for c in classes:
        print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
    inference_time = time.perf_counter() - start
    print('%.1fms' % (inference_time * 1000))
