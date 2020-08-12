import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def union_image_mask(image_path, mask_path, num):
    # 读取原图
    for img_item in os.listdir(image_path):

        image = cv2.imread(image_path)
        # print(image.shape) # (400, 500, 3)
        # print(image.size) # 600000
        # print(image.dtype) # uint8

        # 读取分割mask，这里本数据集中是白色背景黑色mask
        mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        h, w = 384,384
        # cv2.imshow("2d", mask_2d)


        # 在OpenCV中，查找轮廓是从黑色背景中查找白色对象，所以要转成黑色背景白色mask
        mask_3d = np.ones((h, w, 3), dtype='uint8')*255

        ret, thresh = cv2.threshold(mask_2d, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(image, contours, 0, (0, 255, 0), 1)
        # 打开画了轮廓之后的图像
        # cv2.imshow('mask', image)
        # k = cv2.waitKey(0)
        # if k == 27:
        #     cv2.destroyAllWindows()
        # 保存图像
        cv2.imwrite("../image/result/" + str(num) + ".png", image)

if __name__ == '__main__':
    union_image_mask('/home/zhouyilong/github/YNet-master/stage1/data/valrgb/', '/home/zhouyilong/github/YNet-master/seg_eval/results_ynet_c1/',2019)
