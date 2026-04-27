
---

# `Reflection-Learning-Plan/README.md`

```markdown
# Reflection and Learning Plan

## A) Individual Reflection

### What new concepts, skills, or tools did I learn?

Through this project, I learned how to build a full object detection workflow for an embedded GPU edge device. Before this project, I understood the general idea of object detection, but this project helped me understand the complete deployment process from model selection to real-time inference.

The major concepts and tools I learned include:

- YOLOv8 object detection
- CPU vs GPU inference
- TensorRT engine deployment
- PyCUDA memory management
- OpenCV live camera processing
- Jetson system monitoring with jtop and tegrastats
- Model runtime issues involving PyTorch, CUDA, NumPy, and Torchvision
- How to compare performance using FPS and resource monitoring

I also improved my ability to debug real embedded AI problems. A major part of the project involved fixing software compatibility issues and understanding why a model might run on the CPU but fail on the GPU.

---

### Which existing skills did I improve or apply?

I applied and improved my Python programming skills, especially with OpenCV and real-time loops. I also used my existing knowledge of computer vision and embedded systems to understand how camera frames are captured, processed, and displayed.

I also improved my Linux terminal skills. During the setup, I used commands for installing packages, checking camera devices, monitoring performance, installing PyCUDA, and debugging Python package issues.

This project also connected well with my previous experience working with computer vision systems, especially because I have worked with YOLO-based detection before. This project helped me understand the deployment side more deeply.

---

### How will this apply to future study and career?

This project is directly useful for robotics, autonomous systems, smart cameras, industrial automation, and embedded AI. In many real engineering systems, it is not enough to simply train or load a model. The model must run efficiently on the target hardware.

The skills from this project will help me in future work involving:

- robotics perception
- autonomous navigation
- real-time camera systems
- edge AI deployment

---

### What aspects of the project went well?

The CPU baseline implementation went well because it confirmed that the camera, YOLOv8 model, bounding boxes, and FPS display were working. This gave a clear starting point for the workflow.

The final TensorRT GPU result also went well because it showed a major improvement in FPS. The CPU version reached about 1.47 FPS, while the GPU TensorRT version reached about 31.06 FPS. This clearly demonstrated the benefit of GPU acceleration.

The documentation also improved throughout the project because each error and solution became part of the final workflow.

---

### What challenges did I encounter?

The biggest challenge was software compatibility on the Jetson. At different stages, there were issues with:

- PyTorch not detecting CUDA
- NumPy 2.x causing compatibility errors
- Torchvision missing operators
- PyCUDA not being installed
- TensorRT requiring a different runtime approach than the basic Ultralytics code

These problems showed that embedded AI deployment is not always straightforward. The model itself may be correct, but the software environment must also match the hardware.

---

### What was my individual contribution?

My contribution was developing and testing the object detection workflow, debugging the CPU and GPU implementations, setting up the camera pipeline, testing YOLOv8 on the CPU, moving toward TensorRT for GPU inference, and comparing performance results.

I also helped identify why the CPU implementation was too slow and why a TensorRT GPU implementation was required. I tested the final output, collected screenshots, monitored system performance, and documented the workflow for GitHub.

---

## B) Individual Learning Plan

### What knowledge gaps became evident?

This project showed that I need to improve my knowledge in the following areas:

- Cuda software 
- The Linux terminal
- Version compatibility
- Memory management
- Debugging
- GPU profiling and benchmarking
- Managing Python package versions on embedded systems

The biggest knowledge gap was understanding how PyTorch, CUDA, TensorRT, and JetPack versions must match for GPU inference to work properly.

---

### If this solution were scaled for industry, what additional skills would be needed?

If this project were scaled for an industry application, additional skills would be required, including:

- More advanced TensorRT optimization
- Batch inference optimization
- Real-time camera pipeline optimization
- Multi-camera support
- Better error handling and logging
- Model accuracy testing with datasets

For an industrial system, the workflow would also need to be more reliable and repeatable. It would need automatic recovery if the camera disconnects, logs for debugging, and a clean installation script for deployment.

---

### What training and resources would I use?

To improve, I would use the following resources:

- NVIDIA Jetson AI tutorials
- NVIDIA TensorRT documentation
- Ultralytics YOLOv8 documentation
- OpenCV tutorials
- CUDA programming tutorials
- PyCUDA examples
- Jetson performance profiling guides
- Real-time computer vision deployment examples

I would also practice by building more small edge AI projects, such as:

- object detection with multiple cameras
- license plate detection
- robot obstacle detection
- gesture detection

---

## Final Reflection Summary

This project taught me that real-time AI deployment is not only about choosing a model. It is about creating a full workflow that includes hardware selection, software setup, model integration, runtime optimization, debugging, performance measurement, and documentation.

The final result showed a clear difference between CPU and GPU inference. The CPU implementation worked but only reached around 1.47 FPS. The TensorRT GPU implementation reached around 31.06 FPS, showing that GPU acceleration is essential for real-time embedded object detection.

Overall, this project improved my understanding of object detection, embedded AI, GPU acceleration, and practical engineering debugging.
