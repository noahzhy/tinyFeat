import os, sys, glob, time, random, time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.tools import *
from modules.xfeat import XFeat


if __name__ == "__main__":
    model = XFeat().eval()
    # input_shape = (1, 3, 1280, 720) # 7.78 GFLOPs
    input_shape = (1, 3, 640, 480)  # 2.65 GFLOPs
    dummy_input = torch.randn(input_shape).cpu()
    out = model(dummy_input)

    for k, v in out[0].items():
        print(k, v.shape)

    count_parameters(model, input_shape)
