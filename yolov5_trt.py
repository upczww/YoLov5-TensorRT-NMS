"""
An example that uses TensorRT Python api to make inferences.
"""
import ctypes
import os
import random
import threading
import time

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


INPUT_W = 608
INPUT_H = 608
CONF_THRESH = 0.1
IOU_THRESHOLD = 0.4

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.cfx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

    def infer(self, input_image_path):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()
        # Do image preprocess
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(input_image_path)
        # timing
        for i in range(100):
            start = time.time()
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(self.cuda_inputs[0], input_image.ravel(), self.stream)
            # Run inference.
            self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
            # Transfer predictions back from the GPU.
            for host_output, cuda_output in zip(self.host_outputs, self.cuda_outputs):
                cuda.memcpy_dtoh_async(host_output, cuda_output, self.stream)
            # Synchronize the stream
            self.stream.synchronize()
            # Remove any context from the top of the context stack, deactivating it.
            end = time.time()
            print("infer cost %.2fms" % ((end - start) * 1000))
        self.cfx.pop()
        output_counts = self.host_outputs[0]  # [b,1]
        output_boxes = self.host_outputs[1].reshape(1, -1, 4)  # [b,keep_topk,4]
        output_scores = self.host_outputs[2].reshape(1, -1)  # [b,keep_topk]
        output_classids = self.host_outputs[3].reshape(1, -1)  # [b,keep_topk]
        # Select index 0 in that batch size is 1.
        count = output_counts[0]
        boxes = output_boxes[0]
        scores = output_scores[0]
        classids = output_classids[0]

        # Do postprocess
        # Draw rectangles and labels on the original image
        for i in range(int(count)):
            box = self.get_rect(boxes[i], origin_h, origin_w, INPUT_H, INPUT_W)
            plot_one_box(
                box,
                image_raw,
                label="{}:{:.2f}".format(categories[int(classids[i])], scores[i]),
            )
        filename = os.path.basename(input_image_path)
        save_name = "output_" + filename
        # Save image
        cv2.imwrite(save_name, image_raw)

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

    def preprocess_image(self, input_image_path):
        """
        description: Read an image from image path, convert it to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = cv2.imread(input_image_path)
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = INPUT_W / w
        r_h = INPUT_H / h
        if r_h > r_w:
            tw = INPUT_W
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((INPUT_H - th) / 2)
            ty2 = INPUT_H - th - ty1
        else:
            tw = int(r_h * w)
            th = INPUT_H
            tx1 = int((INPUT_W - tw) / 2)
            tx2 = INPUT_W - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128))
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def get_rect(self, bbox, image_h, image_w, input_h, input_w):
        """
        description: postprocess the bbox
        param:
            bbox:     [x1,y1,x2,y2]
            image_h:   height of original image
            image_w:   width of original image
            input_h:   height of network input
            input_w:   width of network input
        return:
            result_bbox: finally box
        """

        result_bbox = [0, 0, 0, 0]
        r_w = input_w / (image_w * 1.0)
        r_h = input_h / (image_h * 1.0)
        if r_h > r_w:
            l = bbox[0] / r_w
            r = bbox[2] / r_w
            t = (bbox[1] - (input_h - r_w * image_h) / 2) / r_w
            b = (bbox[3] - (input_h - r_w * image_h) / 2) / r_w
        else:
            l = (bbox[0] - (input_w - r_h * image_w) / 2) / r_h
            r = (bbox[2] - (input_w - r_h * image_w) / 2) / r_h
            t = bbox[1] / r_h
            b = bbox[3] / r_h
        result_bbox[0] = l
        result_bbox[1] = t
        result_bbox[2] = r
        result_bbox[3] = b
        return result_bbox


class myThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)


if __name__ == "__main__":
    # load tensorrt plugins
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

    # load custom plugins
    PLUGIN_LIBRARY = "build/libyoloplugin.so"
    ctypes.CDLL(PLUGIN_LIBRARY)
    engine_file_path = "build/yolov5s.engine"

    # load coco labels
    categories = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    # a YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(engine_file_path)

    # from https://github.com/ultralytics/yolov5/tree/master/inference/images
    input_image_paths = ["samples/zidane.jpg", "samples/bus.jpg"]

    for input_image_path in input_image_paths:
        # create a new thread to do inference
        thread1 = myThread(yolov5_wrapper.infer, [input_image_path])
        thread1.start()
        thread1.join()

    # destroy the instance
    yolov5_wrapper.destroy()
