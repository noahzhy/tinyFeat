import os, sys, glob, time, random, time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.tools import *
from modules.xfeat import XFeat
from modules.mbv4 import mobilenetv4_conv_small
from modules.tinyFeat import TinyFeat

# disable cuda
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    model = XFeat().eval().to(device)
    # input_shape = (1, 3, 1280, 720) # 7.78 GFLOPs
    input_shape = (1, 3, 640, 480)  # 2.65 GFLOPs
    dummy_input = torch.randn(input_shape).to(device)
    out = model(dummy_input)
    for k, v in out[0].items():
        print(k, v.shape)

    input_shape = (1, 1, 640, 480)
    count_parameters(model, input_shape)

    model = TinyFeat().eval().to(device)
    output = model(torch.randn(input_shape).to(device))
    for output_item in output:
        print(output_item.shape)
    count_parameters(model, input_shape)
    
