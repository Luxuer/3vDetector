#! py -2
# -*- coding: utf-8 -*-
"""
自定义的一些函数，给main函数用
"""

import os
import cv2.cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
# 导入一些需要用到的全局变量
from MyGlobalVariables import * 

# logging.disable(logging.CRITICAL) # 禁用日志
# 设置日志
# logging.basicConfig(
#     filename="log_file.txt",  # 打印到txt文件
#     level=logging.DEBUG, # 允许logging.DEBUG及以上级别的日志信息
#     format=" %(asctime)s - %(levelname)s - %(message)s", # 日志格式
#     datefmt="%Y-%m-%d %H:%M:%S" # 时间格式
#     ) 

# 设置日志系统
def SetLogger():
    logger = logging.getLogger()  # 不加名称设置root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: - %(message)s', # 日志格式
        datefmt='%Y-%m-%d %H:%M:%S' # 时间格式
        )

    # 使用FileHandler输出到文件
    fh = logging.FileHandler('log.txt')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

# 设置日志系统
my_logger = SetLogger()


# 在窗口中展示一张图片
def ShowImage(input_image):
    window_name = "WINDOW_NAME_SHOW_IMAGE"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv.moveWindow(window_name, 0, 0)
    cv.imshow(window_name, input_image)


# # 设置窗口的大小, 位置等
# def SetWindows():
#     # 设置显示窗口
#     cv.namedWindow(WINDOW_NAME_CURRENT_FRAME, cv.WINDOW_NORMAL)
#     cv.namedWindow(WINDOW_NAME_BACKGROUND, cv.WINDOW_NORMAL)
#     cv.namedWindow(WINDOW_NAME_FOREGROUND, cv.WINDOW_NORMAL)
#     cv.namedWindow(WINDOW_NAME_DISPLAY, cv.WINDOW_NORMAL)
    
#     # 调整窗口大小
#     cv.resizeWindow(WINDOW_NAME_CURRENT_FRAME, WINDOW_WIDTH, WINDOW_HEIGHT)
#     cv.resizeWindow(WINDOW_NAME_BACKGROUND, WINDOW_WIDTH, WINDOW_HEIGHT)
#     cv.resizeWindow(WINDOW_NAME_FOREGROUND, WINDOW_WIDTH, WINDOW_HEIGHT)
#     cv.resizeWindow(WINDOW_NAME_DISPLAY, WINDOW_WIDTH, WINDOW_HEIGHT)
#     # 将窗口移动到合适的位置
#     cv.moveWindow(WINDOW_NAME_CURRENT_FRAME, 0, 0)
#     cv.moveWindow(WINDOW_NAME_BACKGROUND, WINDOW_WIDTH + 10, 0)
#     cv.moveWindow(WINDOW_NAME_FOREGROUND, (WINDOW_WIDTH + 10) * 2, 0)
#     cv.moveWindow(WINDOW_NAME_DISPLAY, (WINDOW_WIDTH + 10) * 3, 0)

# # 显示若干设定好图像的窗口
# def ShowWindows():
#     SetWindows()  # 设置显示窗口大小, 位置等
#     cv.imshow(WINDOW_NAME_CURRENT_FRAME, current_frame)  # 显示当前帧
#     cv.imshow(WINDOW_NAME_BACKGROUND, background)  # 显示背景
#     cv.imshow(WINDOW_NAME_FOREGROUND, foreground)  # 显示前景


# 提炼前景
def RefineForeground(foreground):
    
    foreground = cv.threshold(foreground, 127, 255, cv.THRESH_BINARY)[1] # 去除阴影
    # cv.imshow("origin foreground", foreground)
    
    foreground = cv.medianBlur(foreground, 5) # 中值滤波去除噪声

    # cv.imshow("medianBlur", foreground)
    # 闭运算, 填补黑洞
    # foreground = cv.morphologyEx(foreground, cv.MORPH_CLOSE, (5, 5), iterations=5)
    # cv.imshow("MORPH_CLOSE", foreground)
    # 开运算, 填补白洞
    # foreground = cv.morphologyEx(foreground, cv.MORPH_OPEN, (5, 5))
    # cv.imshow("MORPH_OPEN", foreground)
    
    foreground = cv.erode(foreground, (3, 3), iterations=6) # 腐蚀, 去除小块区域
    # foreground = cv.dilate(foreground, (3, 3), iterations=3)
    contours = cv.findContours(foreground, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)[0]
    # print contours
    cars = []
    for contour in contours:
        if cv.contourArea(contour) > 1000:
            cars.append(contour)
    print len(cars)
    return foreground


# 根据前一帧判断当前帧是否可以作为新的背景
def IsBackground(current_frame, previous_frame):
    frame_diff = cv.absdiff(current_frame, previous_frame) # 当前帧与前一帧作差
    return np.max(frame_diff) < MAX_BACKGROUND_CHANGED
    

# # 返回两个数的绝对值, 适用于uint8等, 防止溢出
# def absdiffsingle(x, y):
#     if x > y:
#         return x - y
#     else:
#         return y - x

# # 更新背景
# def UpdateBackgroundAndForeground(current_frame, background, foreground):
#     frame_diff = cv.subtract(current_frame, background)
#     abs_frame_diff = abs(frame_diff)
#     print frame_diff, abs_frame_diff
    # abs_frame_diff < MAX_BACKGROUND_CHANGED and abs_frame_diff > MIN_BACKGROUND_CHANGED
    # rows, cols = current_frame.shape
    # count = 0
    # for i in range(rows):
    #     for j in range(cols):
    #         changed = absdiffsingle(current_frame[i, j], background[i, j])
    #         if changed < MAX_BACKGROUND_CHANGED:
    #             background[i, j] = current_frame[i, j]
    #             count += 1
    #         else:
    #             foreground[i, j] = 255
    # print "Background updated! Changed pixels number = %d" % (count) # 打印提示

# 用三维图像显示二维numpy数组
def PlotImage(input_image):
    # tmp = 
    # array([
    #       [0, ..., input_image.shape[0]], 
    #       [0, ..., input_image.shape[0]], 
    #        ...
    #       [0, ..., input_image.shape[0]]
    #       ])
    tmp = [range(input_image.shape[0])] * input_image.shape[1]
    tmp = np.array(tmp)
    # X = 
    # array([
    #        0, ..., input_image.shape[0], 
    #        0, ..., input_image.shape[0], 
    #        ...
    #        0, ..., input_image.shape[0]
    #       ])
    X = tmp.flatten()
    # Y = 
    # array([
    #        0, ..., 0, 
    #        1, ..., 1, 
    #        ...
    #        input_image.shape[0], ..., input_image.shape[0]
    #       ])
    Y = tmp.transpose().flatten()
    # 把input_image展成一维向量
    Z = input_image.flatten()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X, Y, Z)
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()

# 对输入的原始彩色图像进行预处理, 包括灰度化, 中值滤波, 利用图像掩码提取主要车道等
def PreprocessingImage(input_image, image_mask):
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY) # 转换成灰度
    input_image = cv.medianBlur(input_image, 3) # 中值滤波去除噪声
    input_image &= image_mask # 利用图像掩码提取主要车道
    return input_image
