# @File  : autoLabelPackages.py
# @Author: Bi cheng 80092691
# @Date  :  2020/04/01

import cv2
import numpy as np


def nothing(x):
    pass


def test():
    img_bg = cv2.imread('./bg.bmp')

    
def get_packages():
    # cap = cv2.VideoCapture(0)
    img = cv2.imread('./1.bmp')
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    # fgbg = cv2.createBackgroundSubtractorKNN()
    # while (cap.isOpened()):
    # img = cv2.imread('./1.bmp')
    # ret, frame = cap.read()
    fgmask = fgbg.apply(img)
    cv2.imshow('frame', fgmask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    img = cv2.imread('./1.bmp')
    cv2.namedWindow('edge')
    cv2.createTrackbar('threshold1', 'edge', 90, 500, nothing)
    cv2.createTrackbar('threshold2', 'edge', 360, 500, nothing)

    # 去除噪声
    img_GaussianBlur = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯模糊
    img_gray = cv2.cvtColor(img_GaussianBlur, cv2.COLOR_BGR2GRAY)    # 灰度图

    kernel = np.ones((20, 20), np.uint8)
    img_opening = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)  # 开运算
    cv2.imshow("123", img_opening)
    img_opening = cv2.addWeighted(img_gray, 1, img_opening, -1, 0)   # 与上一次开运算结果融合

    # 通过bar，初步调整后,得到的参数值
    # threshold1 = 90
    # threshold2 = 360
    while True:
        threshold1 = cv2.getTrackbarPos('threshold1', 'edge')
        threshold2 = cv2.getTrackbarPos('threshold2', 'edge')
        # ret, img_gray = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    # 二值化
        edged = cv2.Canny(img_gray, threshold1, threshold2)  # 边缘检测
        cv2.imshow('edge', edged)
        # 先膨胀，后腐蚀，以此连接未闭合的轮廓
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edged, kernel)     # 膨胀
        dilated = cv2.dilate(dilated, kernel)   # 腐蚀
        # cv2.imshow('dilated', dilated)
        eroded = cv2.erode(dilated, kernel)
        edged = eroded.copy()
        # cv2.imshow('edge2', edged)

        # 找到轮廓
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img2 = img.copy()
        cv2.drawContours(img2, contours, -1, (255, 0, 255))

        cv2.imshow("img1", img2)
        c = cv2.waitKey(1)
        if c == ord('q'):
            break


if __name__ == '__main__':
    # main()
    get_packages()
