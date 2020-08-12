#
#author: Sachin Mehta
#Project Description: This repository contains source code for semantically segmenting WSIs; however, it could be easily
#                   adapted for other domains such as natural image segmentation
# File Description: This file is used to compute the mean IOU scores and is adapted from
# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py
#==============================================================================

import torch
import numpy as np

# np.seterr(divide='ignore', invalid='ignore')
class iouEval:
    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.reset()

    def reset(self):
        self.overall_acc = 0
        self.per_class_acc = np.zeros(self.nClasses, dtype=np.float32)
        self.iou = np.zeros(self.nClasses, dtype=np.float32)
        self.mIOU = 0
        self.batchCount = 1

    def fast_hist(self, a, b):
        #  找出标签中需要计算的类别，去掉背景
        #  mask = (label_true >= 0) & (label_true < self.num_classes)
        k = (a >= 0) & (a < self.nClasses)
        #  np.bincount 计算0-(n**2)-1 共n**2个数中每个数出现的次数，返回形状(n,n)
        return np.bincount(self.nClasses * a[k].astype(int) + b[k], minlength=self.nClasses ** 2).reshape(self.nClasses, self.nClasses)

    def compute_hist(self, predict, gth):
        hist = self.fast_hist(gth, predict)
        return hist

    def addBatch(self, predict, gth):
        predict = predict.cpu().numpy().flatten()
        gth = gth.cpu().numpy().flatten()

        epsilon = 0.00000001
        hist = self.compute_hist(predict, gth)
        # acc    //np.diag(hist):预测正确的. hist.sum()总数
        overall_acc = np.diag(hist).sum() / (hist.sum())   #np.diag  对角化/反对角化

        per_class_acc = np.diag(hist) / (hist.sum(1))
        # iou
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        # miou
        mIou = np.nanmean(iou)

        self.overall_acc +=overall_acc
        self.per_class_acc += per_class_acc
        self.iou += iou
        self.mIOU += mIou
        self.batchCount += 1

    def getMetric(self):
        overall_acc = self.overall_acc/self.batchCount
        per_class_acc = self.per_class_acc / self.batchCount
        iou = self.iou / self.batchCount
        mIOU = self.mIOU / self.batchCount

        return overall_acc, per_class_acc, iou, mIOU