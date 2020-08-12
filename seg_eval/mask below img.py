import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
def add_mask2image_binary(images_path, masks_path, masked_path):
# Add binary masks to images
    for img_item in os.listdir(images_path):
        print(img_item)
        img_path = os.path.join(images_path, img_item)
        img = cv2.imread(img_path)
        mask_path = os.path.join(masks_path, img_item[:-4]+'.png')  # mask是.png格式的，image是.jpg格式的
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取

        # plt.imshow(mask)
        # plt.show()
        masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)  #将image的相素值和mask像素值相加得到结果
        cv2.imwrite(os.path.join(masked_path, img_item), masked)
images_path = '/home/zhouyilong/github/YNet-master/stage1/data/valrgb/'
masks_path = '/home/zhouyilong/github/YNet-master/seg_eval/results_ynet_c1/'
masked_path = '/home/zhouyilong/github/YNet-master/seg_eval/mask_below_img/'
add_mask2image_binary(images_path, masks_path, masked_path)





