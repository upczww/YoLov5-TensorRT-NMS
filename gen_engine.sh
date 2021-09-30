#!/bin/bash

CURRENT_PATH=$(pwd)

MODEL_FILE=yolov5s.pt
MODEL_NAME=yolov5s  #s,m,l,x
WTS_FILE=${MODEL_NAME}.wts
ENGINE_FILE=${MODEL_NAME}.engine

sudo docker run --gpus all  \
    -v ${CURRENT_PATH}:/work  registry.cn-guangzhou.aliyuncs.com/nvidia-images/yolov5:4.0 \
    python3 gen_wts.py --model=/work/${MODEL_FILE} --wts=/work/${WTS_FILE}


# sudo docker run --gpus all -v /data/zww/tasks/suit-classification/v1/export:/work -it 657 bash


# sudo docker run --gpus all  \
#                 -v ${CURRENT_PATH}:/work \
#                 -w /work \
#                 -it registry.cn-guangzhou.aliyuncs.com/nvidia-images/tensorrt:21.06-py3-opencv \
#                 bash -c 'cd yolov5-4.0-nms-person && bash run.sh'