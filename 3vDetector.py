#! py -2
# -*- coding: utf-8 -*-
"""
3vDetector - VideoVehicleVelocityDetector 根据监控视频检测车辆速度.
"""


import os
import cv2.cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
# logging.disable(logging.CRITICAL) # 禁用日志
# 设置日志
logging.basicConfig(
    filename="log_file.txt",  # 打印到txt文件
    level=logging.DEBUG, # 允许logging.DEBUG及以上级别的日志信息
    format=" %(asctime)s - %(levelname)s - %(message)s" # 日志格式
    ) 


# 全局变量, 预定义一些固定尺寸、名称等
# 阈值
MAX_BACKGROUND_CHANGED = 50 # 判断帧间像素是否过大, 如果过大, 就认为有运动物体, 不更新背景
MIN_BACKGROUND_CHANGED = 5 # 判断帧间像素是否过小, 如果过小, 就认为是图片噪声, 不改变
# 地面上车道分界线白色色块的尺寸参数, 单位是m
DASHEDLINE_CONTINOUS_LEN = 6.0 # 白色色块的长度
DASHEDLINE_SEPARATE_LEN = 12.0 # 两个白色色块之间的距离
# 窗口名称
WINDOW_NAME_CURRENT_FRAME = "current frame" # 显示当前帧的窗口名称
WINDOW_NAME_BACKGROUND = "background" # 显示背景的窗口名称
WINDOW_NAME_FOREGROUND = "foreground" # 显示前景的窗口名称
# 窗口尺寸
WINDOW_WIDTH = 413
WINDOW_HEIGHT = 295

# 设置窗口的大小, 位置等
def SetWindows():
    # 设置显示窗口
    cv.namedWindow(WINDOW_NAME_CURRENT_FRAME, cv.WINDOW_NORMAL)
    cv.namedWindow(WINDOW_NAME_BACKGROUND, cv.WINDOW_NORMAL)
    cv.namedWindow(WINDOW_NAME_FOREGROUND, cv.WINDOW_NORMAL)
    # 调整窗口大小
    cv.resizeWindow(WINDOW_NAME_CURRENT_FRAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv.resizeWindow(WINDOW_NAME_BACKGROUND, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv.resizeWindow(WINDOW_NAME_FOREGROUND, WINDOW_WIDTH, WINDOW_HEIGHT)
    # 将窗口移动到合适的位置
    cv.moveWindow(WINDOW_NAME_CURRENT_FRAME, 0, 0)
    cv.moveWindow(WINDOW_NAME_BACKGROUND, WINDOW_WIDTH + 10, 0)
    cv.moveWindow(WINDOW_NAME_FOREGROUND, (WINDOW_WIDTH + 10) * 2, 0)

# 提炼前景
def RefineForeground(foreground):
    
    ret_threshold, foreground = cv.threshold(foreground, 127, 255, cv.THRESH_BINARY) # 去除阴影
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
    contours, hierarchy = cv.findContours(foreground, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    # print contours
    cars = []
    for contour in contours:
        if cv.contourArea(contour) > 1000:
            cars.append(contour)
    # print len(cars)
    for i in range(0,len(cars)): 
        x, y, w, h = cv.boundingRect(cars[i])  
        cv.rectangle(foreground, (x,y), (x+w,y+h), (153,153,0), 5)
    return foreground


# # 根据前一帧判断当前帧是否可以作为新的背景
# def IsBackground(current_frame, previous_frame):
#     frame_diff = cv.absdiff(current_frame, previous_frame) # 当前帧与前一帧作差
#     if np.argmax(frame_diff) > MAX_BACKGROUND_CHANGED:
#         return False
#     else:
#         return True

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

# 主函数定义
def main():
    res_dir = os.path.join(".", "res") # 放置资源文件的目录
    images_dir = os.path.join(res_dir, "images") # 放置图片文件的目录
    videos_dir = os.path.join(res_dir, "videos") # 放置视频文件的目录
    # 获取原始背景帧
    origin_background_path = os.path.join(images_dir, "session1_center_no_car.png") # 原始背景帧的路径
    background = cv.imread(origin_background_path, cv.IMREAD_GRAYSCALE)

    # 设置视频捕获器
    video_path = os.path.join(videos_dir, "session1_center_10s-17s.avi")
    capture = cv.VideoCapture(video_path)
    # 计算每两帧之间隔多少秒
    fps = capture.get(cv.CAP_PROP_FPS) # frame per second, 帧每秒
    spf = 1.0 / fps # second per frame, 秒每帧
    SetWindows() # 设置显示窗口大小, 位置等
    # ret, previous_frame = capture.read() # 读取第一帧作为前一帧
    # if not ret: # 读取失败, 退出程序
    #     print "Error! Read video failed!"
    #     exit(1)
    # previous_frame = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY) # 转换成灰度
    # print previous_frame
    # previous_frame = cv.medianBlur(previous_frame, 5)
    # print previous_frame

    # fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
    fgbg = cv.createBackgroundSubtractorMOG2()
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, 3)

    while True:
        ret, current_frame = capture.read()
        if not ret: # 读取失败, 退出循环
            break
        # 对图像进行预处理
        current_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY) # 转换成灰度
        # current_frame = cv.medianBlur(current_frame, 5) # 中值滤波去除噪声
        # 背景减除, 提取前景
        foreground = fgbg.apply(current_frame)
        
        foreground = RefineForeground(foreground)
        
        
        # print type(current_frame[0, 0])
        # break
        # # 更新背景和前景
        # foreground = np.zeros(current_frame.shape)
        # UpdateBackgroundAndForeground(current_frame, background, foreground)
        # break
        # if IsBackground(current_frame, previous_frame): # 根据前一帧判断当前帧是否可以作为新的背景
        #     background = current_frame # 更新背景
        #     print "Background updated!\n" # 打印提示

        # foreground = cv.absdiff(current_frame, background) # 更新前景: 前景 = 当前帧 - 背景
        cv.imshow(WINDOW_NAME_CURRENT_FRAME, current_frame) # 显示当前帧
        cv.imshow(WINDOW_NAME_BACKGROUND, background) # 显示背景
        cv.imshow(WINDOW_NAME_FOREGROUND, foreground) # 显示前景

        # foreground = cv.blur(foreground, (5, 5))
        # PlotImage(foreground[::10, ::10])
        # break # debugging
        
        # previous_frame = current_frame # 当前帧在下一循环变为前一帧
        if cv.waitKey(1) & 0xFF == ord('q'): # 等待, 如果按键q, 就退出循环
            break

    print "Finished.\n"

    # 等待任意键输入
    cv.waitKey(0) 
    # 释放空间
    capture.release()
    cv.destroyAllWindows()









# 主函数的执行
if __name__ == '__main__':
    main()
