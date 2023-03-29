import torch
import torch.nn as nn
from torch.nn import functional as F

from nets.CSPdarknet53_tiny import darknet53_tiny
from nets.attention import cbam_block, eca_block, se_block,CoordAtt

attention_block = [se_block, cbam_block, eca_block,CoordAtt]
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)  

# 整个 ASPP 架构
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, in_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, in_channels))

        self.convs = nn.ModuleList(modules)
        
        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
#-------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + LeakyReLU
#-------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
class BasicConv_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super(BasicConv_2, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m
def yolo_head4(filters_list, in_filters):
    m = nn.Sequential(
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi=0):
        super(YoloBody, self).__init__()
        self.phi            = phi
        self.backbone       = darknet53_tiny(None)

        self.conv_for_P5    = BasicConv(512,256,1)
        self.convp6=BasicConv(256,128,1)
        self.yolo_headP5    = yolo_head([512, len(anchors_mask[0]) * (5 + num_classes)],256)

        self.upsample       = Upsample(256,128)
        self.yolo_headP4    = yolo_head([256, len(anchors_mask[1]) * (5 + num_classes)],384)
        self.aspp=ASPP(512,(3,5,11))
        self.aspp_feat1=ASPP(256,(3,5))
        self.downsampling=nn.Conv2d(384,128,3,2,1)
        self.convp5=BasicConv(256,128,1)
        self.conv_for_P6=nn.Sequential(BasicConv(256,256,3),BasicConv(256,256,1),BasicConv(256,256,1))
    def forward(self, x):
        #---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为26,26,256
        #   feat2的shape为13,13,512
        #---------------------------------------------------#
        feat1, feat2 = self.backbone(x)
        feat2=self.aspp(feat2)
        # 13,13,512 -> 13,13,256
        P5 = self.conv_for_P5(feat2) #考虑P5融合feat1和FEAT2      P6 =touch.cat(feat1_downloading)
        # 13,13,256 -> 13,13,512 -> 13,13,255
        

        # 13,13,256 -> 13,13,128 -> 26,26,128
        P5_Upsample = self.upsample(P5)
        # 26,26,256 + 26,26,128 -> 26,26,384
        P4 = torch.cat([P5_Upsample,feat1],axis=1)
        # 26,26,384 -> 26,26,256 -> 26,26,255
        out1 = self.yolo_headP4(P4)
        P6=self.downsampling(P4)
        P5=self.convp5(P5)
        P6=torch.cat([P6,P5],axis=1)
        P6=self.conv_for_P6(P6)
        out0 = self.yolo_headP5(P6)
        return out0, out1 
'''   aspp pa attention def forward(self, x):
        #---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为26,26,256
        #   feat2的shape为13,13,512
        #---------------------------------------------------#
        feat1, feat2 = self.backbone(x)
        feat2=self.aspp(feat2)
        if 1 <= self.phi and self.phi <= 6:
            feat1 = self.feat1_att(feat1)
            feat2 = self.feat2_att(feat2)

        # 13,13,512 -> 13,13,256
        P5 = self.conv_for_P5(feat2) #考虑P5融合feat1和FEAT2      P6 =touch.cat(feat1_downloading)
        # 13,13,256 -> 13,13,512 -> 13,13,255
        

        # 13,13,256 -> 13,13,128 -> 26,26,128
        P5_Upsample = self.upsample(P5)
        # 26,26,256 + 26,26,128 -> 26,26,384
        if 1 <= self.phi and self.phi <= 6:
            P5_Upsample = self.upsample_att(P5_Upsample)    # ****去掉上采羊的attention
        P4 = torch.cat([P5_Upsample,feat1],axis=1)
        # 26,26,384 -> 26,26,256 -> 26,26,255
        out1 = self.yolo_headP4(P4)
        P6=self.downsampling(P4)
        P6=torch.cat([P6,P5],axis=1)
        P6=self.conv_for_P6(P6)
        out0 = self.yolo_headP5(P6)
        return out0, out1 '''
