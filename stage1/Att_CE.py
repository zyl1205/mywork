import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(F.relu, inplace=True)


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DACblock_without_atrous(nn.Module):
    def __init__(self, channel):
        super(DACblock_without_atrous, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out

        return out


class DACblock_with_inception(nn.Module):
    def __init__(self, channel):
        super(DACblock_with_inception, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)

        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv2d(2 * channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate3(self.dilate1(x)))
        dilate_concat = nonlinearity(self.conv1x1(torch.cat([dilate1_out, dilate2_out], 1)))
        dilate3_out = nonlinearity(self.dilate1(dilate_concat))
        out = x + dilate3_out
        return out


class DACblock_with_inception_blocks(nn.Module):
    def __init__(self, channel):
        super(DACblock_with_inception_blocks, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv5x5 = nn.Conv2d(channel, channel, kernel_size=5, dilation=1, padding=2)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.conv1x1(x))
        dilate2_out = nonlinearity(self.conv3x3(self.conv1x1(x)))
        dilate3_out = nonlinearity(self.conv5x5(self.conv1x1(x)))
        dilate4_out = self.pooling(x)
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(2, 3, 6, 14)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class CBR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class CB(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)

    def forward(self, input):
        output = self.conv(input)
        return output


class DownSampler(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.conv = nn.Conv2d(nIn, nOut - nIn, 3, stride=2, padding=1, bias=False)
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3)
        self.act = nn.ReLU(True)  # nn.PReLU(nOut)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        output = self.act(output)
        return output


class BasicResidualBlock(nn.Module):
    def __init__(self, nIn, nOut, prob=0.03):
        super().__init__()
        self.c1 = CBR(nIn, nOut, 3, 1)
        self.c2 = CB(nOut, nOut, 3, 1)
        self.act = nn.ReLU(True)  # nn.PReLU(nOut)
        # self.drop = nn.Dropout2d(p=prob)

    def forward(self, input):
        output = self.c1(input)
        output = self.c2(output)
        output = input + output
        # output = self.drop(output)
        output = self.act(output)
        return output


class DownSamplerA(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.conv = CBR(nIn, nOut, 3, 2)

    def forward(self, input):
        output = self.conv(input)
        return output


class BR(nn.Module):
    def __init__(self, nOut):
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(True)  # nn.PReLU(nOut)

    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output


class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                              dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output


class CDilated1(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                              dilation=d)
        self.br = BR(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.br(output)


class DilatedParllelResidualBlockB(nn.Module):
    def __init__(self, nIn, nOut, prob=0.03):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3)
        self.act = nn.ReLU(True)  # nn.PReLU(nOut)
        # self.drop = nn.Dropout2d(p=prob)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        combine_in_out = input + combine
        output = self.bn(combine_in_out)
        # output = self.drop(output)
        output = self.act(output)
        return output


class DilatedParllelResidualBlockB1(nn.Module):
    def __init__(self, nIn, nOut, prob=0.03):
        super().__init__()
        k = 4  # we implemented with K=4 only
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n

        self.c1 = C(nIn, n, 3, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3)
        self.act = nn.ReLU(True)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        combine = torch.cat([d1, add1, add2, add3], 1)
        combine_in_out = input + combine
        output = self.bn(combine_in_out)
        output = self.act(output)
        return output


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class PSPDec(nn.Module):
    def __init__(self, nIn, nOut, downSize, upSize=48):
        super().__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(downSize),
            nn.Conv2d(nIn, nOut, 1, bias=False),
            nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3),
            nn.ReLU(True),  # nn.PReLU(nOut),
            nn.Upsample(size=upSize, mode='bilinear')
        )

    def forward(self, x):
        return self.features(x)


class ResNetC1(nn.Module):
    '''
        Segmentation model with ESP as the encoding block.
        This is the same as in stage 1
    '''

    def __init__(self, classes):
        super().__init__()
        self.level1 = CBR(3, 16, 7, 2)  # 0
        self.level1_1 = CBR(16, 64, 3, 1)  # 1
        self.p01 = PSPDec(128, 16, 160, 192)  # 2
        self.p02 = PSPDec(128, 16, 128, 192)  # 3
        self.p03 = PSPDec(128, 16, 96, 192)  # 4
        self.p04 = PSPDec(128, 16, 72, 192)  # 5

        self.class_0 = nn.Sequential(
            nn.Conv2d(128 + 16 * 4, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.95, eps=1e-3),
            nn.ReLU(True),

            nn.Conv2d(32, classes, 7, padding=3, bias=False)
        )  # 6

        self.level2 = DownSamplerA(64, 64)  # 7
        self.level2_0 = DilatedParllelResidualBlockB1(64, 64)  # 8
        self.level2_1 = DilatedParllelResidualBlockB1(64, 64)  # 9

        self.p10 = PSPDec(192, 16, 80, 96)  # 10
        self.p20 = PSPDec(192, 16, 64, 96)  # 11
        self.p30 = PSPDec(192, 16, 48, 96)  # 12
        self.p40 = PSPDec(192, 16, 36, 96)  # 13

        self.class_1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.95, eps=1e-3),
            nn.ReLU(True),

            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64, momentum=0.95, eps=1e-3),
            nn.ReLU(True)
        )  # 14

        self.br_2 = BR(128)  # 15

        self.level3_0 = DownSamplerA(128, 128)  # 16
        self.level3_1 = DilatedParllelResidualBlockB1(128, 128, 0.3)  # 17
        self.level3_2 = DilatedParllelResidualBlockB1(128, 128, 0.3)  # 18

        self.level4_1 = DilatedParllelResidualBlockB1(128, 128, 0.3)  # 19
        self.level4_2 = DilatedParllelResidualBlockB1(128, 128, 0.3)  # 20
        self.level4_3 = DilatedParllelResidualBlockB1(128, 128, 0.3)  # 21

        self.p1 = PSPDec(256, 64, 40, 48)  # 22
        self.p2 = PSPDec(256, 64, 32, 48)  # 23
        self.p3 = PSPDec(256, 64, 24, 48)  # 24
        self.p4 = PSPDec(256, 64, 18, 48)  # 25

        self.br_4 = BR(256)  # 26

        self.classifier = nn.Sequential(
            nn.Conv2d(256 + 4 * 64, 256, 1, padding=0, bias=False),
            nn.BatchNorm2d(256, momentum=0.95, eps=1e-3),
            nn.ReLU(True),

            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.95, eps=1e-3),
            nn.ReLU(True),

            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64, momentum=0.95, eps=1e-3),
            nn.ReLU(True)
        )  # 27

        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 28
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 29
        self.upsample_3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 30

    def forward(self, input1):
        # input1 = self.cmlrn(input)
        output0 = self.level1(input1)
        output0 = self.level1_1(output0)

        output1_0 = self.level2(output0)
        output1 = self.level2_0(output1_0)
        output1 = self.level2_1(output1)

        output1 = self.br_2(torch.cat([output1_0, output1], 1))

        output2_0 = self.level3_0(output1)
        output2 = self.level3_1(output2_0)
        output2 = self.level3_2(output2)

        output3 = self.level4_1(output2)
        output3 = self.level4_2(output3)

        output3 = self.level4_3(output3)
        output3 = self.br_4(torch.cat([output2_0, output3], 1))

        output3 = self.classifier(
            torch.cat([output3, self.p1(output3), self.p2(output3), self.p3(output3), self.p4(output3)], 1))

        output3 = self.upsample_3(output3)

        combine_up_23 = torch.cat([output3, output1], 1)
        output23_hook = self.class_1(torch.cat(
            [combine_up_23, self.p10(combine_up_23), self.p20(combine_up_23), self.p30(combine_up_23),
             self.p40(combine_up_23)], 1))
        output23_hook = self.upsample_2(output23_hook)

        combine_up = torch.cat([output0, output23_hook], 1)

        output0_hook = self.class_0(torch.cat(
            [combine_up, self.p01(combine_up), self.p02(combine_up), self.p03(combine_up), self.p04(combine_up)], 1))
        classifier = self.upsample_1(output0_hook)

        return classifier


class CE_Net_(nn.Module):
    def __init__(self, classes, diagClasses=None):
        super(CE_Net_, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1  # 0
        self.firstbn = resnet.bn1  # 1
        self.firstrelu = resnet.relu  # 2
        self.firstmaxpool = resnet.maxpool  # 3

        self.encoder1 = resnet.layer1  # 4
        self.encoder2 = resnet.layer2  # 5
        self.encoder3 = resnet.layer3  # 6
        self.encoder4 = resnet.layer4  # 7

        self.dblock = DACblock(512)  # 8
        self.spp = SPPblock(512)  # 9

        self.decoder4 = DecoderBlock(516, filters[2])  # 10
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128) # 11
        self.conv4 = nn.Conv2d(512, 256, 3, padding=1)  # 12

        self.decoder3 = DecoderBlock(filters[2], filters[1])  # 13
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)  # 14
        self.conv3 = nn.Conv2d(256, 128, 3, padding=1)  # 15

        self.decoder2 = DecoderBlock(filters[1], filters[0])  # 16
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)

        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 19
        # self.Att1 = Attention_block(F_g=32, F_l=32, F_int=32)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 20
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)  # 22
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, classes, 3, padding=1)  # 24

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # # CE-net Decoder
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)

        # Attention CE-net Decoder
        d4 = self.decoder4(e4)
        x4 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x4, e3), dim=1)
        d4 = self.conv4(d4)

        d3 = self.decoder3(d4)
        x3 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x3, e2), dim=1)
        d3 = self.conv3(d3)

        d2 = self.decoder2(d3)
        x2 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x2, e1), dim=1)
        d2 = self.conv2(d2)

        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        classifier = self.finalconv3(out)

        return classifier
        # return F.sigmoid(out)


class CE_Net_YNet(nn.Module):
    '''
    Jointly learning the segmentation and classification with ESP as encoding blocks
    '''

    def __init__(self, classes, diagClasses, segNetFile=None):
        super().__init__()

        self.level4_0 = DownSamplerA(512, 128)
        self.level4_1 = DilatedParllelResidualBlockB1(128, 128, 0.3)
        self.level4_2 = DilatedParllelResidualBlockB1(128, 128, 0.3)

        self.br_con_4 = BR(256)

        self.level5_0 = DownSamplerA(256, 64)
        self.level5_1 = DilatedParllelResidualBlockB1(64, 64, 0.3)
        self.level5_2 = DilatedParllelResidualBlockB1(64, 64, 0.3)

        self.br_con_5 = BR(128)

        self.global_Avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, diagClasses)

        # segmentation model
        # self.segNet = ResNetC1(classes)
        self.segNet = CE_Net_(classes)
        if segNetFile is not None:
            print('Loading pre-trained CE segmentation model')
            self.segNet.load_state_dict(torch.load(segNetFile))
        self.modules = []
        for i, m in enumerate(self.segNet.children()):
            self.modules.append(m)

    def forward(self, input1):
        output0 = self.modules[0](input1)  # conv (batch_size,64,192,192)
        output0 = self.modules[1](output0)  # BN
        output1_0 = self.modules[2](output0)  # relu
        output1 = self.modules[3](output1_0)  # max pooling (batch_size, 64,96,96)

        output1_1 = self.modules[4](output1)  # resnet_layer1  (8,64,96,96)
        output1_2 = self.modules[5](output1_1)  # resnet_layer2  (8,128,48,48)
        output1_3 = self.modules[6](output1_2)  # resnet_layer3  (8,256,24,24)
        output1_4 = self.modules[7](output1_3)  # resnet_layer4  (8,512,12,12)

        output2_0 = self.modules[8](output1_4)  # DAC  (8,512,12,12)
        output2_1 = self.modules[9](output2_0)  # SPP  (8,516,12,12)



        output3_0 = self.modules[10](output2_1)
        output3_0_1 = self.modules[11](output3_0, output1_3)
        output3_0 = output3_0 + output3_0_1
        output3_0_2 = self.modules[12](output3_0)


        output3_1 = self.modules[13](output3_0_2)
        output3_1_1 = self.modules[14](output3_1, output1_2)
        output3_1 = output3_1 + output3_1_1
        output3_1_2 = self.modules[15](output3_1)
        # output3_1 = self.modules[11](output3_0) + output1_2  # DB  (8,128,48,48)

        output3_2 = self.modules[16](output3_1_2)
        output3_2_1 = self.modules[17](output3_2, output1_1)
        output3_2 = output3_2 + output3_2_1
        output3_2_2 = self.modules[18](output3_2)

        # output3_2 = self.modules[12](output3_1) + output1_1  # DB   (8,64,96,96)


        output3_3 = self.modules[19](output3_2_2)  # DB

        output4_0 = self.modules[20](output3_3)  # ConvTranspose2d (8,64,192,192)
        output4_0 = self.modules[21](output4_0)
        output4_1 = self.modules[22](output4_0)  # conv             (8,32,384,384)
        output4_1 = self.modules[23](output4_1)
        Seg = self.modules[24](output4_1)  # conv            (8,2,384,384)
        # output4_3 = self.modules[17](output4_2)     # relu

        # classifier = self.modules[18](output4_3)    # conv

        # diagnostic branch
        l5_0 = self.level4_0(output1_4)
        l5_1 = self.level4_1(l5_0)
        l5_2 = self.level4_2(l5_1)
        l5_con = self.br_con_4(torch.cat([l5_0, l5_2], 1))

        l6_0 = self.level5_0(l5_con)
        l6_1 = self.level5_1(l6_0)
        l6_2 = self.level5_2(l6_1)
        l6_con = self.br_con_5(torch.cat([l6_0, l6_2], 1))

        glbAvg = self.global_Avg(l6_con)
        flatten = glbAvg.view(glbAvg.size(0), -1)
        fc1 = self.fc1(flatten)
        diagClass = self.fc2(fc1)

        return Seg, diagClass


