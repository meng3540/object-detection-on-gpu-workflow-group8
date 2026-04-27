import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


ENGINE_PATH = "yolov8n.engine"
CONF_THRESH = 0.5
NMS_THRESH = 0.45
PERSON_CLASS_ID = 0


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = image.shape[:2]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)

    resized_w = int(round(w * r))
    resized_h = int(round(h * r))

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    dw = new_w - resized_w
    dh = new_h - resized_h

    dw /= 2
    dh /= 2

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return padded, r, (dw, dh)


def preprocess_bgr(frame, input_w, input_h):
    img, ratio, dwdh = letterbox(frame, (input_w, input_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)    # NCHW
    img = np.ascontiguousarray(img)
    return img, ratio, dwdh


def xywh_to_xyxy(boxes):
    out = np.zeros_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def scale_boxes_to_original(boxes, ratio, dwdh, orig_shape):
    dw, dh = dwdh
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= ratio

    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
    return boxes


class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine.")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_name = None
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        if self.input_name is None:
            raise RuntimeError("No input tensor found.")

        input_shape = self.engine.get_tensor_shape(self.input_name)

        if -1 in input_shape:
            self.context.set_input_shape(self.input_name, (1, 3, 640, 640))
            input_shape = self.context.get_tensor_shape(self.input_name)

        self.input_shape = tuple(input_shape)
        self.batch_size, self.input_c, self.input_h, self.input_w = self.input_shape

        self.bindings = {}
        self.host_inputs = {}
        self.cuda_inputs = {}
        self.host_outputs = {}
        self.cuda_outputs = {}

        all_names = [self.input_name] + self.output_names

        for name in all_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(trt.volume(shape))

            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            self.context.set_tensor_address(name, int(cuda_mem))

            if name == self.input_name:
                self.host_inputs[name] = host_mem
                self.cuda_inputs[name] = cuda_mem
            else:
                self.host_outputs[name] = host_mem
                self.cuda_outputs[name] = cuda_mem

    def infer(self, input_tensor):
        input_name = self.input_name

        np.copyto(self.host_inputs[input_name], input_tensor.ravel())

        cuda.memcpy_htod_async(
            self.cuda_inputs[input_name],
            self.host_inputs[input_name],
            self.stream
        )

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        outputs = []

        for name in self.output_names:
            cuda.memcpy_dtoh_async(
                self.host_outputs[name],
                self.cuda_outputs[name],
                self.stream
            )

        self.stream.synchronize()

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            arr = np.array(self.host_outputs[name]).reshape(shape)
            outputs.append(arr)

        return outputs

def postprocess_yolov8(outputs, frame_shape, ratio, dwdh,
                       conf_thresh=0.5, nms_thresh=0.45, person_class_id=0):
    """
    Handles two common YOLOv8 TensorRT output formats:

    1) Raw output:
       [1, 84, 8400] or [1, 8400, 84]
       where 84 = 4 box coords + 80 class scores

    2) End-to-end/NMS output:
       [1, N, 6] or [N, 6]
       format assumed: x1, y1, x2, y2, conf, cls
    """
    if len(outputs) == 0:
        return []

    out = outputs[0]

    # Remove batch dim if needed
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]

    detections = []

    # Case A: [84, 8400] or [8400, 84]
    if out.ndim == 2 and (84 in out.shape):
        if out.shape[0] == 84:
            out = out.T  # -> [8400, 84]
        elif out.shape[1] != 84:
            raise RuntimeError(f"Unexpected raw output shape: {out.shape}")

        boxes_xywh = out[:, :4]
        class_scores = out[:, 4:]  # YOLOv8 raw export commonly has no separate objectness
        person_scores = class_scores[:, person_class_id]

        keep = person_scores > conf_thresh
        boxes_xywh = boxes_xywh[keep]
        scores = person_scores[keep]

        if len(boxes_xywh) == 0:
            return []

        boxes_xyxy = xywh_to_xyxy(boxes_xywh)
        boxes_xyxy = scale_boxes_to_original(boxes_xyxy, ratio, dwdh, frame_shape)

        boxes_for_nms = []
        for b in boxes_xyxy:
            x1, y1, x2, y2 = b
            boxes_for_nms.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

        indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores.tolist(), conf_thresh, nms_thresh)

        if len(indices) > 0:
            for idx in indices.flatten():
                x1, y1, w, h = boxes_for_nms[idx]
                detections.append({
                    "box": [x1, y1, x1 + w, y1 + h],
                    "score": float(scores[idx]),
                    "class_id": person_class_id
                })

        return detections

    # Case B: [N, 6] or [N, 7]
    if out.ndim == 2 and out.shape[1] in (6, 7):
        # If 7 columns, often batch index is first
        if out.shape[1] == 7:
            out = out[:, 1:]

        for det in out:
            x1, y1, x2, y2, score, cls_id = det[:6]
            cls_id = int(cls_id)

            if cls_id == person_class_id and score > conf_thresh:
                detections.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "score": float(score),
                    "class_id": cls_id
                })

        return detections

    raise RuntimeError(f"Unsupported output shape: {out.shape}")


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        score = det["score"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Person {score:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


def main():
    trt_model = TRTInference(ENGINE_PATH)

    print(f"Loaded engine: {ENGINE_PATH}")
    print(f"Input shape: {trt_model.input_shape}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    prev_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        input_tensor, ratio, dwdh = preprocess_bgr(frame, trt_model.input_w, trt_model.input_h)

        outputs = trt_model.infer(input_tensor)

        try:
            detections = postprocess_yolov8(
                outputs,
                frame.shape,
                ratio,
                dwdh,
                conf_thresh=CONF_THRESH,
                nms_thresh=NMS_THRESH,
                person_class_id=PERSON_CLASS_ID
            )
        except Exception as e:
            print("Postprocess error:", e)
            print("Output shapes:", [o.shape for o in outputs])
            break

        draw_detections(frame, detections)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time else 0.0
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Pure TensorRT Person Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
