# #! py -2
# # -*- coding: utf-8 -*-
# """
# 3vDetector - VideoVehicleVelocityDetector 根据监控视频检测车辆速度.
# """


import os
import cv2.cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging


image1_path = "session1_center_first_frame.png"
image2_path = "session1_center_no_car.png"
image1 = cv.imread(image1_path, cv.IMREAD_GRAYSCALE)
image2 = cv.imread(image2_path, cv.IMREAD_GRAYSCALE)
diff = cv.absdiff(image1, image2)



cv.imwrite("image1.png", image1)
cv.imwrite("image2.png", image2)
cv.imwrite("diff.png", diff)
cv.waitKey()

# # 导入一些需要用到的全局变量
# from MyGlobalVariables import *
# # 导入自定义的函数 
# from MyFunctions import *




# # 主函数定义
# def main():
#     my_logger.info("Start of program")
    
#     # 各种目录
#     res_dir = os.path.join(".", "res") # 放置资源文件的目录
#     images_dir = os.path.join(res_dir, "images") # 放置图片文件的目录
#     videos_dir = os.path.join(res_dir, "videos") # 放置视频文件的目录
    
#     # 获取图像掩码
#     mask_path = os.path.join(images_dir, "new_mask.png") # 图像掩码的路径
#     mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE) # 读入灰度化的图像掩码
#     # 获取原始背景帧
#     origin_background_path = os.path.join(images_dir, "session1_center_no_car.png") # 原始背景帧的路径       
#     background = cv.imread(origin_background_path, cv.IMREAD_GRAYSCALE) # 读入灰度化的原始背景帧
#     background = cv.medianBlur(background, 3) # 中值滤波去除噪声
#     background &= mask # 利用图像掩码提取主要车道
    
    
#     # 设置视频捕获器
#     video_path = os.path.join(videos_dir, "session1_center_10s-17s.avi")
#     capture = cv.VideoCapture(video_path)
#     # 计算每两帧之间隔多少秒
#     fps = capture.get(cv.CAP_PROP_FPS) # frame per second, 帧每秒
#     spf = 1.0 / fps # second per frame, 秒每帧
    
#     ret, previous_frame = capture.read() # 读取第一帧作为前一帧
#     if not ret: # 读取失败, 退出程序
#         my_logger.error("Error! Read video failed!")
#         exit(1)
#     # 对原始彩色图像进行预处理, 包括灰度化, 中值滤波, 利用图像掩码提取主要车道等
#     previous_frame = PreprocessingImage(previous_frame, mask)
#     # kernel = cv.getStructuringElement(cv.MORPH_RECT, 3)
#     # cv.imwrite("gray_first_frame.png", previous_frame)
#     # exit(0)
#     # 椭圆形的结构元素
#     morph_ellipse_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
#     while True:
#         ret, current_frame = capture.read()
#         if not ret: # 读取失败, 退出循环
#             break
#         # 对原始彩色图像进行预处理, 包括灰度化, 中值滤波, 利用图像掩码提取主要车道等
#         current_frame = PreprocessingImage(current_frame, mask)
        
#         # 更新背景和前景
#         if IsBackground(current_frame, previous_frame):
#             background = current_frame # 更新背景
#         #     my_logger.debug("Background updated!") # 打印提示
#         # else:
#         #     my_logger.debug("Cars moving")
#         foreground = cv.absdiff(current_frame, background) # 前景等于当前帧与背景的差
#         thresholded = cv.threshold(foreground, 30, 255, cv.THRESH_BINARY)[1]
#         medianBlured = cv.medianBlur(thresholded, 3) # 中值滤波去除噪声
#         # 膨胀
#         dilated = cv.dilate(
#             medianBlured,
#             morph_ellipse_kernel,
#             iterations = 6)
#         # opening = cv.morphologyEx(dilated, cv2.MORPH_OPEN, morph_ellipse_kernel)



#         # 用外接矩阵初步标记车辆    
#         contours = cv.findContours(
#             dilated, 
#             cv.RETR_EXTERNAL,  # 只检查最外层轮廓
#             cv.CHAIN_APPROX_SIMPLE)[0] # 只存储水平，垂直，对角直线的起始点
        
#         black_canvas = np.zeros(current_frame.shape, dtype=np.uint8) # 黑色画布
#         for contour in contours:
# 		    if cv.contourArea(contour) > 4000:
# 			    x, y, w, h = cv.boundingRect(contour)
# 			    cv.rectangle(black_canvas, (x, y), (x+w, y+h), COLOR_WHITE, thickness=-1)
#         # 腐蚀
#         eroded = cv.erode(
#             black_canvas,
#             morph_ellipse_kernel,
#             iterations = 10)

        

#         # ShowImage(eroded)
#         # # 合并重叠矩阵
#         # contours = cv.findContours(
#         #     black_canvas, 
#         #     cv.RETR_EXTERNAL,  # 只检查最外层轮廓
#         #     cv.CHAIN_APPROX_SIMPLE)[0] # 只存储水平，垂直，对角直线的起始点
#         # black_canvas = np.zeros(current_frame.shape, dtype=np.uint8)

#         # points = []
#         # for contour in contours:
#         #     x, y, w, h = cv.boundingRect(contour)
#         #     points.append(tuple([x + w/2, h])) # 取车头中点坐标为测量点
#         # if points != []:
#         #     print points



#         # white_canvas = np.full(current_frame.shape, 255, dtype=np.uint8) # 白色画布
#         # cv.circle(current_frame,(200,200),50,(55,255,155),1)

#         # 批量输出多个图像
#         image_list = np.array([[current_frame, background, foreground], [thresholded, medianBlured, dilated]])
#         window_names = np.array([["current_frame", "background", "foreground"], ["thresholded", "medianBlured", "dilated"]])
#         # print image_list.shape[:2]
#         # exit(0)
#         rows, cols = image_list.shape[:2]
#         for i in range(rows):
#             for j in range(cols):
#                 window_name = window_names[i, j] # "WINDOW_NAME_" + str(i * cols + j)
#                 cv.namedWindow(window_name, cv.WINDOW_NORMAL)
#                 cv.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
#                 cv.moveWindow(window_name, (WINDOW_WIDTH + WINDOW_WIDTH_GAP)*j, (WINDOW_HEIGHT + WINDOW_HEIGHT_GAP)*i)
#                 cv.imshow(window_name, image_list[i, j])



        
#         # cv.imshow(WINDOW_NAME_DISPLAY, thresholded)

#         # foreground = cv.blur(foreground, (5, 5))
#         # PlotImage(foreground[::10, ::10])
#         # break # debugging
        
#         # previous_frame = current_frame # 当前帧在下一循环变为前一帧
#         if cv.waitKey(1) & 0xFF == 27: # 等待, 如果按键Esc, 就退出循环
#             #
#             # cv.imwrite("foreground.png", foreground)
#             # cv.imwrite("background.png", background)
#             # exit(0)
#             # debugging
#             break
#     # print sum_all
#     my_logger.info("Finished.")

#     # 等待任意键输入
#     cv.waitKey(0) 
#     # 释放空间
#     capture.release()
#     cv.destroyAllWindows()
#     my_logger.info("End of program")




# # 主函数的执行
# if __name__ == '__main__':
#     main()
