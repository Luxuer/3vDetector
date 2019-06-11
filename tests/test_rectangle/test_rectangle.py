#! py -2
# -*- coding: utf-8 -*-
"""
测试cv.rectangle函数
"""

import os
import cv2.cv2 as cv
import numpy as np

# 窗口名字
WINDOW_NAME_DISPLAY = "display"
# 窗口尺寸
WINDOW_WIDTH = 412*2
WINDOW_HEIGHT = 295*2
WINDOW_SHAPE = (WINDOW_HEIGHT, WINDOW_WIDTH)
WINDOW_SHAPE_RGB = (WINDOW_HEIGHT, WINDOW_WIDTH, 3)
# 正方形边长
SQUARE_WIDTH = 50
# 颜色
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (127, 127, 127)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (0, 0, 255) # 红色
COLOR_GREEN = (0, 255, 0) # 绿色
COLOR_BLUE = (255, 0, 0) # 蓝色

cv.namedWindow(WINDOW_NAME_DISPLAY, cv.WINDOW_NORMAL)
cv.resizeWindow(WINDOW_NAME_DISPLAY, WINDOW_WIDTH, WINDOW_HEIGHT)
cv.moveWindow(WINDOW_NAME_DISPLAY, 0, 0)


white_canvas = np.full(WINDOW_SHAPE_RGB, COLOR_GREEN, dtype=np.uint8)
# color = (0)

# cv.rectangle(white_canvas, (100, 100), (400, 300), color, thickness=-1)



cv.imshow(WINDOW_NAME_DISPLAY, white_canvas)

cv.waitKey()


