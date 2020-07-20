import sys
import os
from time import time

import cv2
import numpy as np

img_path = './data/root-test.jpg'
test_loops = 10

from openvino.inference_engine import IECore

model_xml = 'score_net.xml'
model_bin = os.path.splitext(model_xml)[0] + '.bin'
device = 'CPU'
# device = 'MYRIAD'

# create model
ie = IECore()

print(ie.available_devices)

model = ie.read_network(model_xml, model_bin)

input_layer_name = next(iter(model.inputs))
out_layer_name = next(iter(model.outputs))

exec_model = ie.load_network(model, device)

# prepare data
n, c, h, w = model.inputs[input_layer_name].shape

img = cv2.imread(img_path)
img = cv2.resize(img, (w, h))
img = img.transpose((2, 0, 1))

img = img / 255.

# infer
t0 = cv2.getTickCount()
for i in range(test_loops):
    output = exec_model.infer(inputs={input_layer_name: img})
infer_time = (cv2.getTickCount() - t0) / cv2.getTickFrequency()

print(infer_time)

print(output)