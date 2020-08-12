#
#author          :Sachin Mehta
#Description     : This repository contains source code for semantically segmenting WSIs; however, it could be easily
#                   adapted for other domains such as natural image segmentation
#File Description: This file is used to visualize the segmentation masks
#==============================================================================

import numpy as np
import torch

from torch.autograd import Variable
import glob
import cv2
import sys
sys.path.insert(0, '../stage2/')
from stage2.CE_net import CE_Net_YNet
# import Model as Net
import os
from PIL import Image

from stage2.CE_Att2 import CE_Att_Net_YNet
pallete = [ 0, 0, 0,
            # 130, 0, 130,
            # # 0, 0, 130,
            # # 255, 150, 255,
            # # 150 ,150 ,255,
            # # 0 ,255 ,0,
            # 255, 255 ,0,
            255, 255, 255]
model = CE_Net_YNet(2, 5)

model.load_state_dict(torch.load('../stage2/zyl2results_ynet2_CE_/model_300.pth'))
model = model.cuda()
model.eval()

image_list = glob.glob('../stage1/data/valrgb/*.png')
gth_dir = '../stage1/data/valannot/'
out_dir = './results_ynet2_CE/'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
for imgName in image_list:
    print(imgName)
    img = cv2.imread(imgName).astype(np.float32)
    img /=255
    img = img.transpose((2,0,1))
    img_tensor = torch.from_numpy(img)
    img_tensor = torch.unsqueeze(img_tensor, 0) # add a batch dimension
    img_variable = Variable(img_tensor).cuda()
    img_out, sal_out = model(img_variable)
    # # remove the batch dimension
    img_out_norm = torch.squeeze(img_out, 0)
    prob, classMap = torch.max(img_out_norm, 0)
    classMap_numpy = classMap.data.cpu().numpy()

    im_pil = Image.fromarray(np.uint8(classMap_numpy))
    im_pil.putpalette(pallete)
    name = imgName.split('/')[-1]
    im_pil.save(out_dir + name)

    #gth = cv2.imread(gth_dir + os.sep + imgName.split(os.sep)[-1], 0).astype(np.uint8)
    #gth = Image.fromarray(gth)
    #gth.putpalette(pallete)
    #gth.save(out_dir + 'gth_' + name)

