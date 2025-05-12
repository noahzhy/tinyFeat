import os, sys, glob, time, random, time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.tools import *
from modules.xfeat import XFeat
from modules.mbv4 import mobilenetv4_conv_small

# disable cuda
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cpu
device = torch.device("cpu")


if __name__ == "__main__":
    model = XFeat().eval().to(device)
    # input_shape = (1, 3, 1280, 720) # 7.78 GFLOPs
    input_shape = (1, 3, 640, 480)  # 2.65 GFLOPs
    dummy_input = torch.randn(input_shape).to(device)
    out = model(dummy_input)

    for k, v in out[0].items():
        print(k, v.shape)

    count_parameters(model, input_shape)

    model = mobilenetv4_conv_small(num_classes=10).eval().to(device)
    count_parameters(model, input_shape)
    
