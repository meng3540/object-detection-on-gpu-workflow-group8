# Workflow: Object Detection Inferencing on a GPU Edge Device

## Overview

This workflow explains how to set up and run a real-time object detection system on an NVIDIA Jetson Orin Nano using two implementations:

1. **CPU baseline implementation** using YOLOv8 through Ultralytics.
2. **GPU optimized implementation** using a TensorRT engine with PyCUDA.

The project goal is to capture a live camera feed, run object detection, draw bounding boxes and class labels, measure performance, and clearly demonstrate the use of GPU resources. The project requires the workflow to include system setup, model integration, results, references, and code.

---

# 1. System Block Diagram

```text
Live Webcam
    ↓
OpenCV Video Capture
    ↓
Frame Preprocessing
    ↓
Object Detection Model
    ↓
CPU Baseline OR GPU TensorRT Inference
    ↓
Postprocessing
    ↓
Bounding Boxes + Labels
    ↓
FPS Display
    ↓
Performance Monitoring
```

---

# 2. Repository Structure

The GitHub repository should be organized like this:

```text
ObjectDetectionWorkflow/
│
├── README.md
│
├── Workflow/
│   └── README.md
│
├── Reflection-Learning-Plan/
│   └── README.md
│
├── code/
│   ├── pyfileCPU.py
│   ├── pyfileGPU.py
│   ├── yolov8n.pt
│   └── yolov8n.engine
│
└── results/
    ├── CPU_IMAGE_OUTPUT.png
    ├── CPUSTATS_1.png
    ├── CPUSTATS_2.png
    ├── GPU_IMAGE_OUTPUT.png
    ├── GPUSTATS_1.png
    └── GPUSTATS_2.png
```

---

# 3. Hardware Used

## Selected Hardware

The selected embedded platform was:

```text
NVIDIA Jetson Orin Nano Engineering Reference Developer Kit
```

## Reason for Selection

The Jetson Orin Nano was selected because it has an embedded NVIDIA GPU and supports CUDA, TensorRT, Python, OpenCV, and real-time AI inference. This makes it suitable for demonstrating the difference between CPU-based inference and GPU-accelerated inference.

---

# 4. System and Environment Setup

This section explains how to prepare the Jetson system so someone else can run the project from the beginning.

---

## 4.1 Update the System

Run:

```bash
sudo apt update
sudo apt upgrade -y
```

Explanation:

- `sudo apt update` refreshes the package list.
- `sudo apt upgrade -y` updates installed packages.

---

## 4.2 Install Python and Pip

Run:

```bash
sudo apt install python3-pip -y
python3 --version
pip3 --version
```

Explanation:

- Python is used to run the object detection scripts.
- Pip is used to install Python libraries.

---

## 4.3 Install Git

Run:

```bash
sudo apt install git -y
```

Explanation:

Git is used for downloading code and managing the project repository.

---

## 4.4 Install Basic Python Libraries

Run:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install numpy==1.26.4 opencv-python
```

Explanation:

- `numpy` is used for image arrays and TensorRT preprocessing.
- `opencv-python` is used for webcam capture, drawing boxes, and displaying output.
- NumPy was set to `1.26.4` because NumPy 2.x caused compatibility issues with some compiled packages.

---

## 4.5 Install Ultralytics for the CPU Baseline

Run:

```bash
python3 -m pip install ultralytics
```

Explanation:

Ultralytics is used to load and run the YOLOv8 model in the CPU baseline implementation.

If you get a NumPy error like:

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

fix it with:

```bash
python3 -m pip uninstall -y numpy
python3 -m pip install numpy==1.26.4
```

---

## 4.6 Check Camera Connection

Run:

```bash
ls /dev/video*
```

Expected result:

```text
/dev/video0
```

Explanation:

This confirms that the webcam is detected by the Jetson.

The Python code opens the camera using:

```python
cap = cv2.VideoCapture(0)
```

If camera index `0` does not work, try:

```python
cap = cv2.VideoCapture(1)
```

---

# 5. CPU Baseline Implementation

## 5.1 Purpose of the CPU Baseline

The CPU implementation was created first to test the basic object detection workflow before using GPU acceleration.

The CPU version checks that:

- the webcam opens correctly,
- YOLOv8 loads successfully,
- objects are detected,
- bounding boxes and labels are drawn,
- FPS is displayed.

The CPU code uses the YOLOv8 `.pt` model and forces inference to run on the CPU using `device="cpu"`.

---

## 5.2 CPU Model File

The CPU implementation uses:

```text
yolov8n.pt
```

Place this file inside:

```text
code/yolov8n.pt
```

If the file is not already downloaded, Ultralytics can download it automatically when this line runs:

```python
model = YOLO("yolov8n.pt")
```

---

## 5.3 CPU Python File

Create this file:

```text
code/pyfileCPU.py
```

Paste this code:

```python
from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device="cpu")

    annotated_frame = results[0].plot()

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 5.4 Run the CPU Implementation

Go into the code folder:

```bash
cd code
```

Run:

```bash
python3 pyfileCPU.py
```

Press `q` to stop the program.

---

## 5.5 Expected CPU Output

The expected output is a live camera window with object detections.

In the CPU test result, the system detected:

```text
person 0.76
clock 0.48
bottle 0.36
chair 0.31
```

The displayed FPS was approximately:

```text
FPS: 1.47
```

---

## 5.6 CPU Observations

The CPU baseline successfully detected multiple objects and displayed bounding boxes. However, the performance was slow. The system only achieved about **1.47 FPS**, which is not smooth enough for real-time object detection.

This result showed that the object detection pipeline worked, but it also showed why GPU acceleration was needed.

---

# 6. Performance Monitoring Setup

Performance was monitored using Jetson system tools.

---

## 6.1 Install Jetson Monitoring Tool

Run:

```bash
sudo -H pip3 install -U jetson-stats
```

Then reboot:

```bash
sudo reboot
```

After reboot, run:

```bash
jtop
```

Explanation:

`jtop` shows CPU usage, GPU usage, memory usage, temperature, power draw, and running processes.

---

## 6.2 Alternative Monitoring Tool

You can also use:

```bash
tegrastats
```

Explanation:

`tegrastats` shows Jetson performance information directly in the terminal.

---

# 7. GPU TensorRT Setup

## 7.1 Why TensorRT Was Used

The CPU version was functional but too slow. To improve performance, the model was deployed using TensorRT.

TensorRT is designed for optimized inference on NVIDIA GPUs. It reduces runtime overhead and allows the model to run much faster on the Jetson GPU.

---

## 7.2 Check TensorRT Installation

Run:

```bash
dpkg -l | grep -i tensorrt
```

Expected result:

You should see TensorRT packages installed.

Example package names may include:

```text
tensorrt
libnvinfer
```

---

## 7.3 Check for `trtexec`

Run:

```bash
find /usr -name trtexec 2>/dev/null
```

If found, it may be located at:

```text
/usr/src/tensorrt/bin/trtexec
```

To use it directly:

```bash
/usr/src/tensorrt/bin/trtexec --help
```

If `trtexec` is not found, install the TensorRT binary tools:

```bash
sudo apt update
sudo apt install libnvinfer-bin
```

Optional: add TensorRT tools to PATH:

```bash
echo 'export PATH=/usr/src/tensorrt/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

Then test:

```bash
trtexec --help
```

---

## 7.4 Install PyCUDA

Run:

```bash
sudo apt update
sudo apt install python3-pycuda
```

Test PyCUDA:

```bash
python3 -c "import pycuda.driver as cuda; print('PyCUDA working')"
```

Explanation:

PyCUDA is used to allocate GPU memory and transfer data between the CPU and GPU for TensorRT inference.

---

## 7.5 Fix PyCUDA Permission Issue if Needed

If PyCUDA gives an error about not being able to create a CUDA context, run:

```bash
sudo usermod -aG video,render $USER
sudo reboot
```

After reboot, test again:

```bash
python3 -c "import pycuda.driver as cuda; cuda.init(); print(cuda.Device.count()); print(cuda.Device(0).name())"
```

Expected output:

```text
1
NVIDIA Jetson Orin Nano
```

---

# 8. Creating the TensorRT Engine

The final GPU implementation uses:

```text
yolov8n.engine
```

This file should be placed in:

```text
code/yolov8n.engine
```

If the TensorRT engine is already provided in the repository, this step can be skipped.

---

## 8.1 Optional: Export YOLOv8 to ONNX

Create this file:

```text
code/export_onnx.py
```

Paste:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", opset=12, imgsz=640)
```

Run:

```bash
cd code
python3 export_onnx.py
```

Expected result:

```text
yolov8n.onnx
```

---

## 8.2 Convert ONNX to TensorRT Engine

Run:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
```

Or, if `trtexec` is in PATH:

```bash
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
```

Explanation:

- `--onnx=yolov8n.onnx` selects the ONNX model.
- `--saveEngine=yolov8n.engine` saves the optimized TensorRT engine.
- `--fp16` enables half-precision inference, which improves speed on NVIDIA GPUs.

Expected result:

```text
yolov8n.engine
```

---

# 9. GPU Optimized Implementation

## 9.1 Purpose of the GPU Implementation

The GPU implementation was created to improve real-time performance.

Instead of using the YOLOv8 `.pt` model directly through PyTorch, the GPU version uses a TensorRT `.engine` file. It also avoids relying on PyTorch CUDA and uses TensorRT with PyCUDA directly.

The GPU code imports OpenCV, NumPy, TensorRT, and PyCUDA. It loads the TensorRT engine, allocates GPU memory, preprocesses frames, runs asynchronous inference, postprocesses YOLOv8 outputs, filters detections to the person class, draws bounding boxes, and displays FPS.

---

## 9.2 GPU Files Needed

The GPU implementation needs:

```text
code/pyfileGPU.py
code/yolov8n.engine
```

---

## 9.3 Important GPU Code Sections

The TensorRT engine path is defined as:

```python
ENGINE_PATH = "yolov8n.engine"
```

The confidence threshold is:

```python
CONF_THRESH = 0.5
```

The detection class is:

```python
PERSON_CLASS_ID = 0
```

This means the GPU version only draws detections for the **person** class.

---

## 9.4 Preprocessing in the GPU Version

The GPU version uses letterbox resizing so the image keeps its aspect ratio before inference.

The frame is converted as follows:

```text
BGR image
    ↓
RGB image
    ↓
float32
    ↓
normalized from 0 to 1
    ↓
HWC to CHW
    ↓
batch dimension added
```

This matches the expected YOLOv8 input format.

---

## 9.5 TensorRT Inference

The TensorRT engine is loaded using:

```python
with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
    self.engine = runtime.deserialize_cuda_engine(f.read())
```

The execution context is created using:

```python
self.context = self.engine.create_execution_context()
```

GPU memory is allocated using PyCUDA:

```python
cuda_mem = cuda.mem_alloc(host_mem.nbytes)
```

Inference is executed with:

```python
self.context.execute_async_v3(stream_handle=self.stream.handle)
```

---

## 9.6 Postprocessing

After inference, the YOLO output is postprocessed by:

1. extracting bounding boxes,
2. extracting class scores,
3. filtering only the person class,
4. applying confidence thresholding,
5. applying non-maximum suppression,
6. drawing the final bounding box.

The box is drawn with:

```python
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
```

The label is drawn with:

```python
cv2.putText(
    frame,
    f"Person {score:.2f}",
    (x1, max(y1 - 10, 20)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)
```

---

## 9.7 Run the GPU Implementation

Go into the code folder:

```bash
cd code
```

Run:

```bash
python3 pyfileGPU.py
```

Press `q` to stop the program.

---

## 9.8 Expected GPU Output

The expected output is a live camera feed with a green bounding box around the detected person.

In the GPU test result, the system detected:

```text
Person 0.85
```

The displayed FPS was approximately:

```text
FPS: 31.06
```

---

## 9.9 GPU Observations

The GPU version produced a much smoother real-time output than the CPU version. The TensorRT implementation reached about **31.06 FPS**, compared to about **1.47 FPS** for the CPU baseline.

The Jetson monitoring output also showed active GPU usage and GPU memory use while the Python process was running. This confirmed that the optimized implementation was using GPU resources.

---

# 10. CPU vs GPU Results

## 10.1 Performance Comparison

| Implementation | Model Format | Runtime | Output | FPS |
|---|---|---|---|---|
| CPU Baseline | `yolov8n.pt` | Ultralytics / CPU | Multiple objects detected | ~1.47 FPS |
| GPU Optimized | `yolov8n.engine` | TensorRT + PyCUDA | Person detected | ~31.06 FPS |

---

## 10.2 Speed Improvement

```text
Speedup = GPU FPS / CPU FPS
Speedup = 31.06 / 1.47
Speedup ≈ 21.13x
```

The GPU TensorRT implementation was approximately **21 times faster** than the CPU baseline.

---

# 11. Results Discussion

The CPU baseline confirmed that the object detection pipeline worked correctly. It successfully opened the webcam, loaded the YOLOv8 model, detected objects, drew bounding boxes, and displayed FPS. However, the CPU-only implementation was too slow for smooth real-time performance.

The GPU implementation improved the workflow by using TensorRT. The GPU version achieved around 31 FPS, making it suitable for real-time object detection. The output was also cleaner because it filtered detections to only the person class.

The performance comparison shows that the GPU-accelerated workflow is much more effective for real-time embedded object detection than CPU-only inference.

---

# 12. Common Issues and Fixes

## Issue 1: NumPy Compatibility Error

Error:

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

Fix:

```bash
python3 -m pip uninstall -y numpy
python3 -m pip install numpy==1.26.4
```

---

## Issue 2: Camera Does Not Open

Check camera:

```bash
ls /dev/video*
```

Try changing:

```python
cap = cv2.VideoCapture(0)
```

to:

```python
cap = cv2.VideoCapture(1)
```

---

## Issue 3: `trtexec` Not Found

Find it:

```bash
find /usr -name trtexec 2>/dev/null
```

Use full path:

```bash
/usr/src/tensorrt/bin/trtexec --help
```

Install if missing:

```bash
sudo apt install libnvinfer-bin
```

---

## Issue 4: PyCUDA Not Found

Error:

```text
ModuleNotFoundError: No module named 'pycuda'
```

Fix:

```bash
sudo apt install python3-pycuda
```

---

## Issue 5: PyCUDA Cannot Create CUDA Context

Fix permissions:

```bash
sudo usermod -aG video,render $USER
sudo reboot
```

Test after reboot:

```bash
python3 -c "import pycuda.driver as cuda; cuda.init(); print(cuda.Device.count()); print(cuda.Device(0).name())"
```

---

# 13. Final Workflow Summary

The final workflow was:

```text
1. Set up Jetson environment
2. Install Python, OpenCV, NumPy, Ultralytics, TensorRT, and PyCUDA
3. Test webcam access
4. Run YOLOv8 CPU baseline
5. Measure CPU FPS and system usage
6. Prepare TensorRT engine
7. Run TensorRT GPU implementation
8. Measure GPU FPS and GPU usage
9. Compare CPU and GPU performance
10. Document workflow, code, outputs, and observations
```

---

# 14. Final Observations

- The CPU baseline worked but was slow.
- The CPU implementation reached approximately **1.47 FPS**.
- The GPU TensorRT implementation was much faster.
- The GPU implementation reached approximately **31.06 FPS**.
- The GPU version was approximately **21 times faster**.
- TensorRT is more suitable for real-time object detection on Jetson.
- The workflow successfully demonstrated GPU acceleration.

---

# 15. References

- MENG3540 Object Detection Workflow project handout
- Ultralytics YOLOv8 documentation
- NVIDIA Jetson documentation
- NVIDIA TensorRT documentation
- OpenCV documentation
- PyCUDA documentation
- Jetson monitoring tools: jtop and tegrastats
