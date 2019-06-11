#! py -2
# -*- coding: utf-8 -*-
"""
python2函数放在后面能不能调用
"""

import os
import cv2.cv2 as cv
import numpy as np



dash_line_path = "session1_center_white_dash_line.png"
dash_line = cv.imread(dash_line_path, cv.IMREAD_GRAYSCALE)
dash_line = ~dash_line

mask_path = "session_center_1channel_mask.png"
mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

mask &= dash_line


cv.imwrite("mask.png", mask)





