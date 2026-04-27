# Object Detection Inferencing on a GPU Edge Device

## Project Overview

This project focuses on developing a complete workflow for real-time object detection on an embedded GPU edge device. The system captures live camera input, runs object detection using a pre-trained YOLOv8 model, displays bounding boxes and class labels, and compares CPU-based inference with GPU-accelerated inference.

The project follows the MENG3540 workflow requirements, which include using accelerated embedded hardware, a pre-trained model, GPU acceleration tools, live camera annotation, performance measurement, and GitHub documentation. The project handout also states that the workflow and analysis must clearly demonstrate the use of GPU resources. :contentReference[oaicite:0]{index=0}

---

## Discussion on Problem Statement

Object detection is the process of identifying and locating objects inside an image or video frame. In this project, the system uses a live webcam feed as the input and runs object detection in real time. The detected objects are shown with bounding boxes, class labels, and confidence scores.

This problem is important because many real-world edge AI systems need to process visual data locally instead of sending every image to a cloud server. Examples include security cameras, autonomous robots, traffic monitoring systems, warehouse automation, and smart manufacturing systems.

AI inferencing is a parallel computation problem because a neural network performs many mathematical operations on image data. Each video frame contains thousands or millions of pixels, and the model processes the image through many layers of convolution, matrix multiplication, activation functions, and postprocessing. These operations can be performed in parallel, which makes GPUs much more suitable than CPUs for real-time object detection.

The main goal of this project was to show a complete workflow for object detection on an embedded edge device, beginning with a basic CPU implementation and then improving it using GPU acceleration through TensorRT.

---

## Possible Hardware Options

| Hardware | Description | Advantages | Limitations |
|---|---|---|---|
| Raspberry Pi | Low-cost single-board computer | Cheap, easy to use, good for simple vision tasks | No strong CUDA GPU acceleration |
| Standard Laptop CPU | General computer platform | Easy to develop and test Python code | Not an embedded GPU platform |
| NVIDIA Jetson Nano | Embedded NVIDIA GPU board | Supports CUDA and TensorRT | Older and less powerful |
| NVIDIA Jetson Xavier NX | Embedded AI platform | Good AI inference performance | More expensive |
| NVIDIA Jetson Orin Nano | Modern embedded GPU platform | Strong GPU support, CUDA, TensorRT, edge AI capability | Requires correct software setup |

---

## Selected Hardware

The selected hardware was the **NVIDIA Jetson Orin Nano Engineering Reference Developer Kit**.

### Rationale

The Jetson Orin Nano was selected because it is an embedded platform with accelerated NVIDIA GPU hardware. It supports CUDA, TensorRT, OpenCV, Python, and real-time AI inference workflows. This makes it a better choice than a CPU-only system because the project specifically requires the workflow to demonstrate GPU resources and acceleration.

---

## Possible Software Frameworks and Libraries

| Software / Library | Purpose |
|---|---|
| Python | Main programming language used for implementation |
| OpenCV | Capturing webcam frames, drawing bounding boxes, displaying output |
| PyTorch | Initial model testing and CPU inference using YOLOv8 |
| Ultralytics YOLOv8 | Loading the pre-trained YOLOv8 model |
| ONNX | Intermediate model format for deployment |
| TensorRT | NVIDIA inference optimization and GPU acceleration |
| PyCUDA | GPU memory allocation and TensorRT runtime execution |
| NumPy | Image array manipulation and preprocessing |
| jtop / tegrastats | Monitoring CPU, GPU, memory, temperature, and power |
| GitHub | Version control, documentation, and collaboration |

---

## Possible Pre-Trained Models

| Model | Description | Advantages | Limitations |
|---|---|---|---|
| YOLOv8n | Small YOLOv8 object detection model | Fast, lightweight, suitable for real-time embedded use | Less accurate than larger YOLO models |
| YOLOv8s | Slightly larger YOLOv8 model | Better accuracy than YOLOv8n | Slower than YOLOv8n |
| SSD MobileNet | Lightweight object detector | Good for embedded systems | Older and less accurate than YOLOv8 |
| Faster R-CNN | Two-stage object detector | High accuracy | Too slow for real-time edge inference |
| YOLOv5n | Earlier lightweight YOLO model | Fast and stable | Older than YOLOv8 |

---

## Selected Model

The selected model was **YOLOv8n**.

### Rationale

YOLOv8n was selected because it is lightweight, fast, and suitable for real-time object detection on embedded devices. Since it is pre-trained, it can detect common objects without needing to train a model from scratch. In the CPU version, the model detected multiple objects such as person, clock, bottle, and chair. In the optimized GPU version, the detection was filtered to focus on the person class.

---

## Final Implementation Summary

The project used two implementations:

1. **CPU baseline implementation**
   - Used Ultralytics YOLOv8 with `device="cpu"`
   - Ran object detection directly from `yolov8n.pt`
   - Displayed bounding boxes, labels, and FPS
   - Achieved around **1.47 FPS**

2. **GPU optimized implementation**
   - Used a TensorRT engine file, `yolov8n.engine`
   - Used TensorRT and PyCUDA directly instead of relying on PyTorch CUDA
   - Filtered detection to the person class
   - Displayed bounding box, confidence score, and FPS
   - Achieved around **31.06 FPS**

The GPU TensorRT implementation was significantly faster than the CPU implementation and better satisfied the project requirement of demonstrating GPU acceleration.

---

## Repository Structure

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
