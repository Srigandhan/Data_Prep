import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import cv2

# Constants
INPUT_SHAPE = (3, 544, 960)
NUM_CLASSES = 3

# Load class labels
with open("class_labels.txt", "r") as f:
    class_labels = f.read().splitlines()

# Load TensorRT engine
engine_path = "your_engine_file.engine"
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
with open(engine_path, "rb") as f:
    engine_data = f.read()
engine = runtime.deserialize_cuda_engine(engine_data)

# Create execution context
context = engine.create_execution_context()

# Allocate device memory
d_input = cuda.mem_alloc(np.prod(INPUT_SHAPE) * 4)  # 4 bytes per int8
d_output = cuda.mem_alloc(NUM_CLASSES * 4)  # 4 bytes per float32

# Create stream
stream = cuda.Stream()

# Load and preprocess image
image_path = "your_image.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (INPUT_SHAPE[2], INPUT_SHAPE[1]))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.transpose((2, 0, 1))  # Channels-first
image = image.astype(np.float32)
image /= 255.0
image = np.ascontiguousarray(image)

# Copy image to device
cuda.memcpy_htod_async(d_input, image.ravel(), stream)

# Run inference
context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
stream.synchronize()

# Copy output back to host
output = np.empty(NUM_CLASSES, dtype=np.float32)
cuda.memcpy_dtoh_async(output, d_output, stream)
stream.synchronize()

# Find the predicted class
predicted_class = np.argmax(output)
predicted_label = class_labels[predicted_class]

# Display the predicted class label
print("Predicted Class Label:", predicted_label)
