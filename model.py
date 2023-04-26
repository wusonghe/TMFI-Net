import torch
from torch import nn
import math
from swin_transformer import *
from collections import OrderedDict


class Gate(nn.Module):
    def __init__(self, in_plane):
        super(Gate, self).__init__()
        self.gate = nn.Conv3d(in_plane, in_plane, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, rgb_fea):
        gate = torch.sigmoid(self.gate(rgb_fea))
        gate_fea = rgb_fea * gate + rgb_fea

        return gate_fea


class VideoSaliencyModel(nn.Module):
    def __init__(self, pretrain=None):
        super(VideoSaliencyModel, self).__init__()

        self.backbone = SwinTransformer3D(pretrained=pretrain)
        self.decoder = DecoderConvUp()

    def forward(self, x):
        x, [y1, y2, y3, y4] = self.backbone(x)

        return self.decoder(x, y3, y2, y1)


class DecoderConvUp(nn.Module):
    def __init__(self):
        super(DecoderConvUp, self).__init__()

        self.upsampling2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.upsampling4 = nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear')
        self.upsampling8 = nn.Upsample(scale_factor=(1, 8, 8), mode='trilinear')

        self.conv1 = nn.Conv3d(96, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv2 = nn.Conv3d(192, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv3 = nn.Conv3d(384, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv4 = nn.Conv3d(768, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.convs1 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.convs2 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.convs3 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)

        self.convtsp1 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.Sigmoid()
        )
        self.convtsp2 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.Sigmoid()
        )
        self.convtsp3 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            self.upsampling2,
            nn.Sigmoid()
        )
        self.convtsp4 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            self.upsampling4,
            nn.Sigmoid()
        )

        self.convout = nn.Sequential(
            nn.Conv3d(4, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.Sigmoid()
        )

        self.gate1 = Gate(192)
        self.gate2 = Gate(192)
        self.gate3 = Gate(192)
        self.gate4 = Gate(192)

    def forward(self, y4, y3, y2, y1):
        y1 = self.conv1(y1)
        y2 = self.conv2(y2)
        y3 = self.conv3(y3)
        y4 = self.conv4(y4)

        t3 = self.upsampling2(y4) + y3
        y3 = self.convs3(t3)
        t2 = self.upsampling2(t3) + y2 + self.upsampling4(y4)
        y2 = self.convs2(t2)
        t1 = self.upsampling2(t2) + y1 + self.upsampling8(y4)
        y1 = self.convs1(t1)


        y1 = self.gate1(y1)
        y2 = self.gate2(y2)
        y3 = self.gate3(y3)
        y4 = self.gate4(y4)

        z1 = self.convtsp1(y1)

        z2 = self.convtsp2(y2)

        z3 = self.convtsp3(y3)

        z4 = self.convtsp4(y4)

        z0 = self.convout(torch.cat((z1, z2, z3, z4), 1))

        z0 = z0.view(z0.size(0), z0.size(3), z0.size(4))
        return z0
