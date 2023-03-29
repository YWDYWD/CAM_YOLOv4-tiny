import numpy as np
from PIL import Image


import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision.ops import nms
#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[142, 110], [192, 243], [459, 401]
        #   26x26的特征层对应的anchor是[36, 75], [76, 55], [72, 146]
        #   52x52的特征层对应的anchor是[12, 16], [19, 36], [40, 28]
        #-----------------------------------------------------------#
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):
        #-----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size, 255, 13, 13
        #   batch_size, 255, 26, 26
        #   batch_size, 255, 52, 52
        #-----------------------------------------------#
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)

        #-----------------------------------------------#
        #   输入为416x416时
        #   stride_h = stride_w = 32、16、8
        #-----------------------------------------------#
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width
        #-------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #-------------------------------------------------#
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors]

        #-----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size, 3, 13, 13, 85
        #   batch_size, 3, 26, 26, 85
        #   batch_size, 3, 52, 52, 85
        #-----------------------------------------------#
        prediction = input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])  
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]
        h = prediction[..., 3]
        # 获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4])
        # 种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        #----------------------------------------------------------#
        #   生成网格，先验框中心，网格左上角 
        #   batch_size,3,13,13
        #----------------------------------------------------------#
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

        #----------------------------------------------------------#
        #   按照网格格式生成先验框的宽高
        #   batch_size,3,13,13
        #----------------------------------------------------------#
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        #----------------------------------------------------------#
        #   利用预测结果对先验框进行调整
        #   首先调整先验框的中心，从先验框中心向右下角偏移
        #   再调整先验框的宽高。
        #----------------------------------------------------------#
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # fig = plt.figure()
        # ax = fig.add_subplot(121)
        # if input_height==13:
        #     plt.ylim(0,13)
        #     plt.xlim(0,13)
        # elif input_height==26:
        #     plt.ylim(0,26)
        #     plt.xlim(0,26)
        # elif input_height==52:
        #     plt.ylim(0,52)
        #     plt.xlim(0,52)
        # plt.scatter(grid_x.cpu(),grid_y.cpu())

        # anchor_left = grid_x - anchor_w/2 
        # anchor_top = grid_y - anchor_h/2 

        # rect1 = plt.Rectangle([anchor_left[0,0,5,5],anchor_top[0,0,5,5]],anchor_w[0,0,5,5],anchor_h[0,0,5,5],color="r",fill=False)
        # rect2 = plt.Rectangle([anchor_left[0,1,5,5],anchor_top[0,1,5,5]],anchor_w[0,1,5,5],anchor_h[0,1,5,5],color="r",fill=False)
        # rect3 = plt.Rectangle([anchor_left[0,2,5,5],anchor_top[0,2,5,5]],anchor_w[0,2,5,5],anchor_h[0,2,5,5],color="r",fill=False)

        # ax.add_patch(rect1)
        # ax.add_patch(rect2)
        # ax.add_patch(rect3)

        # ax = fig.add_subplot(122)
        # if input_height==13:
        #     plt.ylim(0,13)
        #     plt.xlim(0,13)
        # elif input_height==26:
        #     plt.ylim(0,26)
        #     plt.xlim(0,26)
        # elif input_height==52:
        #     plt.ylim(0,52)
        #     plt.xlim(0,52)
        # plt.scatter(grid_x.cpu(),grid_y.cpu())
        # plt.scatter(pred_boxes[0,:,5,5,0].cpu(),pred_boxes[0,:,5,5,1].cpu(),c='r')

        # pre_left = pred_boxes[...,0] - pred_boxes[...,2]/2 
        # pre_top = pred_boxes[...,1] - pred_boxes[...,3]/2 

        # rect1 = plt.Rectangle([pre_left[0,0,5,5],pre_top[0,0,5,5]],pred_boxes[0,0,5,5,2],pred_boxes[0,0,5,5,3],color="r",fill=False)
        # rect2 = plt.Rectangle([pre_left[0,1,5,5],pre_top[0,1,5,5]],pred_boxes[0,1,5,5,2],pred_boxes[0,1,5,5,3],color="r",fill=False)
        # rect3 = plt.Rectangle([pre_left[0,2,5,5],pre_top[0,2,5,5]],pred_boxes[0,2,5,5,2],pred_boxes[0,2,5,5,3],color="r",fill=False)

        # ax.add_patch(rect1)
        # ax.add_patch(rect2)
        # ax.add_patch(rect3)

        # plt.show()

        #----------------------------------------------------------#
        #   将输出结果调整成相对于输入图像大小
        #----------------------------------------------------------#
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data
#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   获得先验框
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
