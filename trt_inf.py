import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import cv2

# Constants
INPUT_SIZE = (300, 300)
OUTPUT_LAYOUT = 7
NUM_CLASSES = 80

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
d_input = cuda.mem_alloc(INPUT_SIZE[0] * INPUT_SIZE[1] * 3 * 4)  # 4 bytes per float32
d_output = cuda.mem_alloc(OUTPUT_LAYOUT * OUTPUT_LAYOUT * NUM_CLASSES * 4)  # 4 bytes per float32

# Create stream
stream = cuda.Stream()

# Load and preprocess image
image_path = "your_image.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, INPUT_SIZE)
image = image.transpose((2, 0, 1))  # Channels-first
image = image.astype(np.float32) / 255.0
image = np.expand_dims(image, axis=0)

# Copy image to device
cuda.memcpy_htod_async(d_input, image, stream)

# Run inference
context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
stream.synchronize()

# Copy output back to host
output = np.empty((1, OUTPUT_LAYOUT, OUTPUT_LAYOUT, NUM_CLASSES), dtype=np.float32)
cuda.memcpy_dtoh_async(output, d_output, stream)
stream.synchronize()

# Process output
output = output.reshape((OUTPUT_LAYOUT, OUTPUT_LAYOUT, NUM_CLASSES))

# Draw bounding boxes on the image
for row in range(OUTPUT_LAYOUT):
    for col in range(OUTPUT_LAYOUT):
        for cls in range(NUM_CLASSES):
            confidence = output[row, col, cls]
            if confidence > 0.5:  # Adjust this threshold as needed
                class_label = class_labels[cls]
                x = int(col * image.shape[2] / OUTPUT_LAYOUT)
                y = int(row * image.shape[1] / OUTPUT_LAYOUT)
                cv2.rectangle(image, (x, y), (x + 50, y + 20), (0, 255, 0), -1)
                cv2.putText(image, class_label, (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Display the image
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
