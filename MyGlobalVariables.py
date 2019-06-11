#! py -2
# -*- coding: utf-8 -*-
"""
全局变量, 预定义一些固定尺寸、名称等
"""
# 阈值
MAX_BACKGROUND_CHANGED = 50 # 判断帧间像素是否过大, 如果过大, 就认为有运动物体, 不更新背景
MIN_BACKGROUND_CHANGED = 10 # 判断帧间像素是否过小, 如果过小, 就认为是图片噪声, 不改变
# 地面上车道分界线白色色块的尺寸参数, 单位是m
DASHEDLINE_CONTINOUS_LEN = 6.0 # 白色色块的长度
DASHEDLINE_SEPARATE_LEN = 12.0 # 两个白色色块之间的距离
# 窗口名称
WINDOW_NAME_CURRENT_FRAME = "current frame" # 显示当前帧的窗口名称
WINDOW_NAME_BACKGROUND = "background" # 显示背景的窗口名称
WINDOW_NAME_FOREGROUND = "foreground" # 显示前景的窗口名称
WINDOW_NAME_DISPLAY = "display" # 显示其他

# 窗口尺寸
WINDOW_WIDTH = 413/2 # 413 # 窗口的宽度
WINDOW_HEIGHT = 295/2 # 295 # 窗口的高度
WINDOW_WIDTH_GAP = 0 # 10 # 窗口横向之间的间隙
WINDOW_HEIGHT_GAP = 30 # 窗口纵向之间的间隙
# 颜色
COLOR_WHITE = (255, 255, 255) # 白色
COLOR_GRAY = (127, 127, 127) # 灰色
COLOR_BLACK = (0, 0, 0) # 黑色
# 要注意opencv许多函数中的参数都假设你传入的是(B, G, R)
COLOR_RED = (0, 0, 255) # 红色
COLOR_GREEN = (0, 255, 0) # 绿色
COLOR_BLUE = (255, 0, 0) # 蓝色