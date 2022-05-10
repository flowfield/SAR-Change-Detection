import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import exponential
import cv2
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler as MMS
from torch.nn.modules.padding import ReplicationPad2d


class CDCNN(nn.Module):
    def __init__(self):
        super(CDCNN, self).__init__()
        # Feature Extractor 1 ~ 5 (?) 두개의 이미지 각각 시행
        # 두 이미지
        self.conv1 = nn.Conv2d(4, 16, kernel_size=(2, 2), stride=(2, 2))
        self.BN1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU(True)
        self.do1 = nn.Dropout(p=0.2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(2, 2), stride=(2, 2))
        self.BN2 = nn.BatchNorm2d(32)
        self.act2 = nn.ReLU(True)
        self.do2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(2, 2), stride=(2, 2))
        self.BN3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU(True)
        self.do3 = nn.Dropout(p=0.2)

        self.convT1 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2))
        self.BN4 = nn.BatchNorm2d(32)
        self.act4 = nn.ReLU(True)
        self.do4 = nn.Dropout(p=0.2)

        self.convT2 = nn.ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2))
        self.BN5 = nn.BatchNorm2d(16)
        self.act5 = nn.ReLU(True)
        self.do5 = nn.Dropout(p=0.2)

        self.convT3 = nn.ConvTranspose2d(16, 1, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        L1_1 = self.conv1((x))
        x1 = self.do1(self.act1(L1_1))

        L2_1 = self.conv2((x1))
        L2_2 = self.BN2(L2_1)
        x2 = self.do2(self.act2(L2_2))

        L3_1 = self.conv3((x2))
        L3_2 = self.BN3(L3_1)
        x3 = self.do3(self.act3(L3_2))

        L4_1 = self.convT1((x3))
        L4_2 = self.BN4(L4_1)
        x4 = self.do4(self.act4(L4_2))

        L5_1 = self.convT2((x4))
        L5_2 = self.BN5(L5_1)
        x5 = self.do5(self.act5(L5_2))

        out = self.convT3(x5)

        return out


class CDCNN_0714(nn.Module):
    def __init__(self):
        super(CDCNN_0714, self).__init__()
        # 두 이미지 # 256,256,2
        self.conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1) # 256,256,2
        self.act1 = nn.ReLU(True)
        # 256,256,2
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1) # 256,256,4
        self.BN2 = nn.BatchNorm2d(4)
        self.act2 = nn.ReLU(True)
        self.do2 = nn.Dropout(p=0.2)
        self.mp2 = nn.MaxPool2d(kernel_size=4, padding=1, stride=2, return_indices=True) # 128,128,4

        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1) # 128,128,8
        self.BN3 = nn.BatchNorm2d(8)
        self.act3 = nn.ReLU(True)
        self.do3 = nn.Dropout(p=0.2)
        self.mp3 = nn.MaxPool2d(kernel_size=4, padding=1, stride=2, return_indices=True) # 64,64,8

        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) # 64,64,16
        self.BN4 = nn.BatchNorm2d(16)
        self.act4 = nn.ReLU(True)
        self.do4 = nn.Dropout(p=0.2)
        self.mp4 = nn.MaxPool2d(kernel_size=4, padding=1, stride=2, return_indices=True) # 32,32,16

        self.conv5 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # 32,32,32
        self.BN5 = nn.BatchNorm2d(32)
        self.act5 = nn.ReLU(True)
        self.do5 = nn.Dropout(p=0.2)
        self.mp5 = nn.MaxPool2d(kernel_size=4, padding=1, stride=2, return_indices=True) # 16,16,32

        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 16,16,32
        self.BN6 = nn.BatchNorm2d(32)
        self.act6 = nn.ReLU(True)
        self.do6 = nn.Dropout(p=0.2)
        self.mp6 = nn.MaxUnpool2d(kernel_size=4, padding=1, stride=2) # mp5 32,32,32

        self.conv7 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1) # 32,32,16
        self.BN7 = nn.BatchNorm2d(16)
        self.act7 = nn.ReLU(True)
        self.do7 = nn.Dropout(p=0.2)
        self.mp7 = nn.MaxUnpool2d(kernel_size=4, padding=1, stride=2) # mp4 64,64,16

        self.conv8 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1) # 64,64,8
        self.BN8 = nn.BatchNorm2d(8)
        self.act8 = nn.ReLU(True)
        self.do8 = nn.Dropout(p=0.2)
        self.mp8 = nn.MaxUnpool2d(kernel_size=4, padding=1, stride=2) # mp3 128,128,8

        self.conv9 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1) # 128,128,4
        self.BN9 = nn.BatchNorm2d(4)
        self.act9 = nn.ReLU(True)
        self.do9 = nn.Dropout(p=0.2)
        self.mp9 = nn.MaxUnpool2d(kernel_size=4, padding=1, stride=2) # mp2 256,256,4

        self.conv10 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1) # 256,256,2
        self.BN10 = nn.BatchNorm2d(2)
        self.act10 = nn.ReLU(True)
        self.do10 = nn.Dropout(p=0.2)

        self.conv11 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        l1_1 = self.conv1(x)
        x1 = self.act1(l1_1)

        l2_1 = self.conv2(x1)
        l2_2 = self.BN2(l2_1)
        l2_3 = self.act2(l2_2)
        l2_4 = self.do2(l2_3)
        x2, id2 = self.mp2(l2_4)

        l3_1 = self.conv3(x2)
        l3_2 = self.BN3(l3_1)
        l3_3 = self.act3(l3_2)
        l3_4 = self.do3(l3_3)
        x3, id3 = self.mp3(l3_4)

        l4_1 = self.conv4(x3)
        l4_2 = self.BN4(l4_1)
        l4_3 = self.act4(l4_2)
        l4_4 = self.do4(l4_3)
        x4, id4 = self.mp4(l4_4)

        l5_1 = self.conv5(x4)
        l5_2 = self.BN5(l5_1)
        l5_3 = self.act5(l5_2)
        l5_4 = self.do5(l5_3)
        x5, id5 = self.mp5(l5_4)

        l6_1 = self.conv6(x5)
        l6_2 = self.BN6(l6_1)
        l6_3 = self.act6(l6_2)
        l6_4 = self.do6(l6_3)
        x6 = self.mp6(l6_4, id5)

        l7_1 = self.conv7(x6)
        l7_2 = self.BN7(l7_1)
        l7_3 = self.act7(l7_2)
        l7_4 = self.do7(l7_3)
        x7 = self.mp7(l7_4, id4)

        l8_1 = self.conv8(x7)
        l8_2 = self.BN8(l8_1)
        l8_3 = self.act8(l8_2)
        l8_4 = self.do8(l8_3)
        x8 = self.mp8(l8_4, id3)

        l9_1 = self.conv9(x8)
        l9_2 = self.BN9(l9_1)
        l9_3 = self.act9(l9_2)
        l9_4 = self.do9(l9_3)
        x9 = self.mp9(l9_4, id2)

        l10_1 = self.conv10(x9)
        l10_2 = self.BN10(l10_1)
        l10_3 = self.act10(l10_2)
        x10 = self.do10(l10_3)

        out = self.conv11(x10)

        return out


class CDCNN_0716(nn.Module):
    def __init__(self):
        super(CDCNN_0716, self).__init__()
        # 두 이미지 # 256,256,2
        self.conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1) # 256,256,2
        self.act1 = nn.ReLU(True)
        # 256,256,2
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1) # 256,256,4
        self.BN2 = nn.BatchNorm2d(4)
        self.act2 = nn.ReLU(True)
        self.do2 = nn.Dropout(p=0.2)
        self.mp2 = nn.MaxPool2d(kernel_size=4, padding=1, stride=2, return_indices=True) # 128,128,4

        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1) # 128,128,8
        self.BN3 = nn.BatchNorm2d(8)
        self.act3 = nn.ReLU(True)
        self.do3 = nn.Dropout(p=0.2)
        self.mp3 = nn.MaxPool2d(kernel_size=4, padding=1, stride=2, return_indices=True) # 64,64,8

        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) # 64,64,16
        self.BN4 = nn.BatchNorm2d(16)
        self.act4 = nn.ReLU(True)
        self.do4 = nn.Dropout(p=0.2)
        self.mp4 = nn.MaxPool2d(kernel_size=4, padding=1, stride=2, return_indices=True) # 32,32,16

        self.conv5 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # 32,32,32
        self.BN5 = nn.BatchNorm2d(32)
        self.act5 = nn.ReLU(True)
        self.do5 = nn.Dropout(p=0.2)
        self.mp5 = nn.MaxPool2d(kernel_size=4, padding=1, stride=2, return_indices=True) # 16,16,32

        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 16,16,32
        self.BN6 = nn.BatchNorm2d(32)
        self.act6 = nn.ReLU(True)
        self.do6 = nn.Dropout(p=0.2)
        self.mp6 = nn.MaxUnpool2d(kernel_size=4, padding=1, stride=2) # mp5 32,32,32

        self.conv7 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1) # 32,32,16
        self.BN7 = nn.BatchNorm2d(16)
        self.act7 = nn.ReLU(True)
        self.do7 = nn.Dropout(p=0.2)
        self.mp7 = nn.MaxUnpool2d(kernel_size=4, padding=1, stride=2) # mp4 64,64,16

        self.conv8 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1) # 64,64,8
        self.BN8 = nn.BatchNorm2d(8)
        self.act8 = nn.ReLU(True)
        self.do8 = nn.Dropout(p=0.2)
        self.mp8 = nn.MaxUnpool2d(kernel_size=4, padding=1, stride=2) # mp3 128,128,8

        self.conv9 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1) # 128,128,4
        self.BN9 = nn.BatchNorm2d(4)
        self.act9 = nn.ReLU(True)
        self.do9 = nn.Dropout(p=0.2)
        self.mp9 = nn.MaxUnpool2d(kernel_size=4, padding=1, stride=2) # mp2 256,256,4

        self.conv10 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1) # 256,256,2
        self.BN10 = nn.BatchNorm2d(2)
        self.act10 = nn.ReLU(True)
        self.do10 = nn.Dropout(p=0.2)

        self.conv11 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        l1_1 = self.conv1(x)
        x1 = self.act1(l1_1)

        l2_1 = self.conv2(x1)
        l2_2 = self.BN2(l2_1)
        l2_3 = self.act2(l2_2)
        l2_4 = self.do2(l2_3)
        x2, id2 = self.mp2(l2_4)

        l3_1 = self.conv3(x2)
        l3_2 = self.BN3(l3_1)
        l3_3 = self.act3(l3_2)
        l3_4 = self.do3(l3_3)
        x3, id3 = self.mp3(l3_4)

        l4_1 = self.conv4(x3)
        l4_2 = self.BN4(l4_1)
        l4_3 = self.act4(l4_2)
        l4_4 = self.do4(l4_3)
        x4, id4 = self.mp4(l4_4)

        l5_1 = self.conv5(x4)
        l5_2 = self.BN5(l5_1)
        l5_3 = self.act5(l5_2)
        l5_4 = self.do5(l5_3)
        x5, id5 = self.mp5(l5_4)

        l6_1 = self.conv6(x5)
        l6_2 = self.BN6(l6_1)
        l6_3 = self.act6(l6_2)
        l6_4 = self.do6(l6_3)
        x6 = self.mp6(l6_4, id5)

        l7_1 = self.conv7(x6)
        l7_2 = self.BN7(l7_1)
        l7_3 = self.act7(l7_2)
        l7_4 = self.do7(l7_3)
        x7 = self.mp7(l7_4, id4)

        l8_1 = self.conv8(x7)
        l8_2 = self.BN8(l8_1)
        l8_3 = self.act8(l8_2)
        l8_4 = self.do8(l8_3)
        x8 = self.mp8(l8_4, id3)

        l9_1 = self.conv9(x8)
        l9_2 = self.BN9(l9_1)
        l9_3 = self.act9(l9_2)
        l9_4 = self.do9(l9_3)
        x9 = self.mp9(l9_4, id2)

        l10_1 = self.conv10(x9)
        l10_2 = self.BN10(l10_1)
        l10_3 = self.act10(l10_2)
        x10 = self.do10(l10_3)

        out = self.conv11(x10)

        return out


class CD_AEResNet_0721(nn.Module):
    def __init__(self):
        super(CD_AEResNet_0721, self).__init__()
        # 두 이미지 # 256,256,2
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1) # 256,256,2
        self.act1 = nn.ReLU(True)

        self.conv1_1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.BN1_1 = nn.BatchNorm2d(8)
        self.act1_1 = nn.ReLU(True)
        self.do1_1 = nn.Dropout(p=0.2)
        self.mp2 = nn.MaxPool2d(kernel_size=4, padding=1, stride=2, return_indices=True) # 128

        # 256,256,2
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) # 256,256,4
        self.BN2 = nn.BatchNorm2d(16)
        self.act2 = nn.ReLU(True)
        self.do2 = nn.Dropout(p=0.2)
        self.mp2 = nn.MaxPool2d(kernel_size=4, padding=1, stride=2, return_indices=True) # 128,128,4

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # 128,128,8
        self.BN3 = nn.BatchNorm2d(32)
        self.act3 = nn.ReLU(True)
        self.do3 = nn.Dropout(p=0.2)
        self.mp3 = nn.MaxPool2d(kernel_size=4, padding=1, stride=2, return_indices=True) # 64,64,8

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 64,64,16
        self.BN4 = nn.BatchNorm2d(64)
        self.act4 = nn.ReLU(True)
        self.do4 = nn.Dropout(p=0.2)
        self.mp4 = nn.MaxPool2d(kernel_size=4, padding=1, stride=2, return_indices=True) # 32,32,16

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # 32,32,32
        self.BN5 = nn.BatchNorm2d(32)
        self.act5 = nn.ReLU(True)
        self.do5 = nn.Dropout(p=0.2)
        self.mp5 = nn.MaxPool2d(kernel_size=4, padding=1, stride=2, return_indices=True) # 16,16,32

        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 16,16,32
        self.BN6 = nn.BatchNorm2d(32)
        self.act6 = nn.ReLU(True)
        self.do6 = nn.Dropout(p=0.2)
        self.mp6 = nn.MaxUnpool2d(kernel_size=4, padding=1, stride=2) # mp5 32,32,32

        self.conv7 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1) # 32,32,16
        self.BN7 = nn.BatchNorm2d(16)
        self.act7 = nn.ReLU(True)
        self.do7 = nn.Dropout(p=0.2)
        self.mp7 = nn.MaxUnpool2d(kernel_size=4, padding=1, stride=2) # mp4 64,64,16

        self.conv8 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1) # 64,64,8
        self.BN8 = nn.BatchNorm2d(8)
        self.act8 = nn.ReLU(True)
        self.do8 = nn.Dropout(p=0.2)
        self.mp8 = nn.MaxUnpool2d(kernel_size=4, padding=1, stride=2) # mp3 128,128,8

        self.conv9 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1) # 128,128,4
        self.BN9 = nn.BatchNorm2d(4)
        self.act9 = nn.ReLU(True)
        self.do9 = nn.Dropout(p=0.2)
        self.mp9 = nn.MaxUnpool2d(kernel_size=4, padding=1, stride=2) # mp2 256,256,4

        self.conv10 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1) # 256,256,2
        self.BN10 = nn.BatchNorm2d(2)
        self.act10 = nn.ReLU(True)
        self.do10 = nn.Dropout(p=0.2)

        self.conv11 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_in = x

        l1_1 = self.conv1(x)
        x1 = self.act1(l1_1)

        l2_1 = self.conv2(x1)
        l2_2 = self.BN2(l2_1)
        l2_3 = self.act2(l2_2)
        l2_4 = self.do2(l2_3)
        x2, id2 = self.mp2(l2_4)

        l3_1 = self.conv3(x2)
        l3_2 = self.BN3(l3_1)
        l3_3 = self.act3(l3_2)
        l3_4 = self.do3(l3_3)
        x3, id3 = self.mp3(l3_4)

        l4_1 = self.conv4(x3)
        l4_2 = self.BN4(l4_1)
        l4_3 = self.act4(l4_2)
        l4_4 = self.do4(l4_3)
        x4, id4 = self.mp4(l4_4)

        l5_1 = self.conv5(x4)
        l5_2 = self.BN5(l5_1)
        l5_3 = self.act5(l5_2)
        l5_4 = self.do5(l5_3)
        x5, id5 = self.mp5(l5_4)

        l6_1 = self.conv6(x5)
        l6_2 = self.BN6(l6_1)
        l6_3 = self.act6(l6_2)
        l6_4 = self.do6(l6_3)
        x6 = self.mp6(l6_4, id5)

        x6 = l5_1 + x6 # 32,32,32

        l7_1 = self.conv7(x6)
        l7_2 = self.BN7(l7_1)
        l7_3 = self.act7(l7_2)
        l7_4 = self.do7(l7_3)
        x7 = self.mp7(l7_4, id4)

        x7 = l4_1 + x7

        l8_1 = self.conv8(x7)
        l8_2 = self.BN8(l8_1)
        l8_3 = self.act8(l8_2)
        l8_4 = self.do8(l8_3)
        x8 = self.mp8(l8_4, id3)

        x8 = l3_1 + x8

        l9_1 = self.conv9(x8)
        l9_2 = self.BN9(l9_1)
        l9_3 = self.act9(l9_2)
        l9_4 = self.do9(l9_3)
        x9 = self.mp9(l9_4, id2)

        x9 = l2_1 + x9 # 256,256,4

        l10_1 = self.conv10(x9)
        l10_2 = self.BN10(l10_1)
        l10_3 = self.act10(l10_2)
        x10 = self.do10(l10_3)

        out = self.conv11(x10)

        return out


class CD_AutoEncoder_0721(nn.Module):
    def __init__(self):
        super(CD_AutoEncoder_0721, self).__init__()
        # 두 이미지 # 256,256,2
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1) # 256,256,2
        self.act1 = nn.ReLU(True)
        # 256,256,2
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1) # 256,256,4
        self.BN2 = nn.BatchNorm2d(4)
        self.act2 = nn.ReLU(True)
        self.do2 = nn.Dropout(p=0.2)
        self.mp2 = nn.MaxPool2d(kernel_size=4, padding=1, stride=2, return_indices=True) # 128,128,4

        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1) # 128,128,8
        self.BN3 = nn.BatchNorm2d(8)
        self.act3 = nn.ReLU(True)
        self.do3 = nn.Dropout(p=0.2)
        self.mp3 = nn.MaxPool2d(kernel_size=4, padding=1, stride=2, return_indices=True) # 64,64,8

        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) # 64,64,16
        self.BN4 = nn.BatchNorm2d(16)
        self.act4 = nn.ReLU(True)
        self.do4 = nn.Dropout(p=0.2)
        self.mp4 = nn.MaxPool2d(kernel_size=4, padding=1, stride=2, return_indices=True) # 32,32,16

        self.conv5 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # 32,32,32
        self.BN5 = nn.BatchNorm2d(32)
        self.act5 = nn.ReLU(True)
        self.do5 = nn.Dropout(p=0.2)
        self.mp5 = nn.MaxPool2d(kernel_size=4, padding=1, stride=2, return_indices=True) # 16,16,32

        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 16,16,32
        self.BN6 = nn.BatchNorm2d(32)
        self.act6 = nn.ReLU(True)
        self.do6 = nn.Dropout(p=0.2)
        self.mp6 = nn.MaxUnpool2d(kernel_size=4, padding=1, stride=2) # mp5 32,32,32

        self.conv7 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1) # 32,32,16
        self.BN7 = nn.BatchNorm2d(16)
        self.act7 = nn.ReLU(True)
        self.do7 = nn.Dropout(p=0.2)
        self.mp7 = nn.MaxUnpool2d(kernel_size=4, padding=1, stride=2) # mp4 64,64,16

        self.conv8 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1) # 64,64,8
        self.BN8 = nn.BatchNorm2d(8)
        self.act8 = nn.ReLU(True)
        self.do8 = nn.Dropout(p=0.2)
        self.mp8 = nn.MaxUnpool2d(kernel_size=4, padding=1, stride=2) # mp3 128,128,8

        self.conv9 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1) # 128,128,4
        self.BN9 = nn.BatchNorm2d(4)
        self.act9 = nn.ReLU(True)
        self.do9 = nn.Dropout(p=0.2)
        self.mp9 = nn.MaxUnpool2d(kernel_size=4, padding=1, stride=2) # mp2 256,256,4

        self.conv10 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1) # 256,256,2
        self.BN10 = nn.BatchNorm2d(2)
        self.act10 = nn.ReLU(True)
        self.do10 = nn.Dropout(p=0.2)

        self.conv11 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        l1_1 = self.conv1(x)
        x1 = self.act1(l1_1)

        l2_1 = self.conv2(x1)
        l2_2 = self.BN2(l2_1)
        l2_3 = self.act2(l2_2)
        l2_4 = self.do2(l2_3)
        x2, id2 = self.mp2(l2_4)

        l3_1 = self.conv3(x2)
        l3_2 = self.BN3(l3_1)
        l3_3 = self.act3(l3_2)
        l3_4 = self.do3(l3_3)
        x3, id3 = self.mp3(l3_4)

        l4_1 = self.conv4(x3)
        l4_2 = self.BN4(l4_1)
        l4_3 = self.act4(l4_2)
        l4_4 = self.do4(l4_3)
        x4, id4 = self.mp4(l4_4)

        l5_1 = self.conv5(x4)
        l5_2 = self.BN5(l5_1)
        l5_3 = self.act5(l5_2)
        l5_4 = self.do5(l5_3)
        x5, id5 = self.mp5(l5_4)

        l6_1 = self.conv6(x5)
        l6_2 = self.BN6(l6_1)
        l6_3 = self.act6(l6_2)
        l6_4 = self.do6(l6_3)
        x6 = self.mp6(l6_4, id5)

        l7_1 = self.conv7(x6)
        l7_2 = self.BN7(l7_1)
        l7_3 = self.act7(l7_2)
        l7_4 = self.do7(l7_3)
        x7 = self.mp7(l7_4, id4)

        l8_1 = self.conv8(x7)
        l8_2 = self.BN8(l8_1)
        l8_3 = self.act8(l8_2)
        l8_4 = self.do8(l8_3)
        x8 = self.mp8(l8_4, id3)

        l9_1 = self.conv9(x8)
        l9_2 = self.BN9(l9_1)
        l9_3 = self.act9(l9_2)
        l9_4 = self.do9(l9_3)
        x9 = self.mp9(l9_4, id2)

        l10_1 = self.conv10(x9)
        l10_2 = self.BN10(l10_1)
        l10_3 = self.act10(l10_2)
        x10 = self.do10(l10_3)

        out = self.conv11(x10)

        return out


class CD_CNN_0722(nn.Module):
    def __init__(self):
        super(CD_CNN_0722, self).__init__()
        # 두 이미지 # 256,256,2
        self.conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1) # 256,256,2
        self.act1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1) # 256,256,4
        self.BN2 = nn.BatchNorm2d(32)
        self.act2 = nn.ReLU(True)
        self.do2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 128,128,8
        self.BN3 = nn.BatchNorm2d(32)
        self.act3 = nn.ReLU(True)
        self.do3 = nn.Dropout(p=0.2)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 64,64,16
        self.BN4 = nn.BatchNorm2d(32)
        self.act4 = nn.ReLU(True)
        self.do4 = nn.Dropout(p=0.2)

        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 32,32,32
        self.BN5 = nn.BatchNorm2d(32)
        self.act5 = nn.ReLU(True)
        self.do5 = nn.Dropout(p=0.2)

        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 16,16,32
        self.BN6 = nn.BatchNorm2d(32)
        self.act6 = nn.ReLU(True)
        self.do6 = nn.Dropout(p=0.2)

        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 32,32,16
        self.BN7 = nn.BatchNorm2d(32)
        self.act7 = nn.ReLU(True)
        self.do7 = nn.Dropout(p=0.2)

        self.conv8 = nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1) # 64,64,8
        self.BN8 = nn.BatchNorm2d(32)
        self.act8 = nn.ReLU(True)
        self.do8 = nn.Dropout(p=0.2)

        self.conv9 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 128,128,4
        self.BN9 = nn.BatchNorm2d(32)
        self.act9 = nn.ReLU(True)
        self.do9 = nn.Dropout(p=0.2)

        self.conv10 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 256,256,2
        self.BN10 = nn.BatchNorm2d(32)
        self.act10 = nn.ReLU(True)
        self.do10 = nn.Dropout(p=0.2)

        self.conv11 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        l1_1 = self.conv1(x)
        x1 = self.act1(l1_1)

        l2_1 = self.conv2(x1)
        l2_2 = self.BN2(l2_1)
        l2_3 = self.act2(l2_2)
        x2 = self.do2(l2_3)

        l3_1 = self.conv3(x2)
        l3_2 = self.BN3(l3_1)
        l3_3 = self.act3(l3_2)
        x3 = self.do3(l3_3)

        l4_1 = self.conv4(x3)
        l4_2 = self.BN4(l4_1)
        l4_3 = self.act4(l4_2)
        x4 = self.do4(l4_3)

        l5_1 = self.conv5(x4)
        l5_2 = self.BN5(l5_1)
        l5_3 = self.act5(l5_2)
        x5 = self.do5(l5_3)

        l6_1 = self.conv6(x5)
        l6_2 = self.BN6(l6_1)
        l6_3 = self.act6(l6_2)
        x6 = self.do6(l6_3)

        # x6 = l5_1 + x6 # 32,32,32

        l7_1 = self.conv7(x6)
        l7_2 = self.BN7(l7_1)
        l7_3 = self.act7(l7_2)
        x7 = self.do7(l7_3)

        # x7 = l4_1 + x7

        l8_1 = self.conv8(x7)
        l8_2 = self.BN8(l8_1)
        l8_3 = self.act8(l8_2)
        x8 = self.do8(l8_3)

        # x8 = l3_1 + x8

        l9_1 = self.conv9(x8)
        l9_2 = self.BN9(l9_1)
        l9_3 = self.act9(l9_2)
        x9 = self.do9(l9_3)

        # x9 = l2_1 + x9 # 256,256,4

        l10_1 = self.conv10(x9)
        l10_2 = self.BN10(l10_1)
        l10_3 = self.act10(l10_2)
        x10 = self.do10(l10_3)

        out = self.conv11(x10)

        return out


class CD_VGG_0724(nn.Module):
    def __init__(self):
        super(CD_VGG_0724, self).__init__()
        # 두 이미지 # 256,256,2
        self.conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1) # 256,256,2
        self.act1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1) # 256,256,4
        self.BN2 = nn.BatchNorm2d(32)
        self.act2 = nn.ReLU(True)
        self.do2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 128,128,8
        self.BN3 = nn.BatchNorm2d(32)
        self.act3 = nn.ReLU(True)
        self.do3 = nn.Dropout(p=0.2)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 64,64,16
        self.BN4 = nn.BatchNorm2d(32)
        self.act4 = nn.ReLU(True)
        self.do4 = nn.Dropout(p=0.2)

        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 32,32,32
        self.BN5 = nn.BatchNorm2d(32)
        self.act5 = nn.ReLU(True)
        self.do5 = nn.Dropout(p=0.2)

        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 16,16,32
        self.BN6 = nn.BatchNorm2d(32)
        self.act6 = nn.ReLU(True)
        self.do6 = nn.Dropout(p=0.2)

        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 32,32,16
        self.BN7 = nn.BatchNorm2d(32)
        self.act7 = nn.ReLU(True)
        self.do7 = nn.Dropout(p=0.2)

        self.conv8 = nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1) # 64,64,8
        self.BN8 = nn.BatchNorm2d(32)
        self.act8 = nn.ReLU(True)
        self.do8 = nn.Dropout(p=0.2)

        self.conv9 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 128,128,4
        self.BN9 = nn.BatchNorm2d(32)
        self.act9 = nn.ReLU(True)
        self.do9 = nn.Dropout(p=0.2)

        self.conv10 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 256,256,2
        self.BN10 = nn.BatchNorm2d(32)
        self.act10 = nn.ReLU(True)
        self.do10 = nn.Dropout(p=0.2)

        self.conv11 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        l1_1 = self.conv1(x)
        x1 = self.act1(l1_1)

        l2_1 = self.conv2(x1)
        l2_2 = self.BN2(l2_1)
        l2_3 = self.act2(l2_2)
        x2 = self.do2(l2_3)

        l3_1 = self.conv3(x2)
        l3_2 = self.BN3(l3_1)
        l3_3 = self.act3(l3_2)
        x3 = self.do3(l3_3)

        l4_1 = self.conv4(x3)
        l4_2 = self.BN4(l4_1)
        l4_3 = self.act4(l4_2)
        x4 = self.do4(l4_3)

        l5_1 = self.conv5(x4)
        l5_2 = self.BN5(l5_1)
        l5_3 = self.act5(l5_2)
        x5 = self.do5(l5_3)

        l6_1 = self.conv6(x5)
        l6_2 = self.BN6(l6_1)
        l6_3 = self.act6(l6_2)
        x6 = self.do6(l6_3)

        # x6 = l5_1 + x6 # 32,32,32

        l7_1 = self.conv7(x6)
        l7_2 = self.BN7(l7_1)
        l7_3 = self.act7(l7_2)
        x7 = self.do7(l7_3)

        # x7 = l4_1 + x7

        l8_1 = self.conv8(x7)
        l8_2 = self.BN8(l8_1)
        l8_3 = self.act8(l8_2)
        x8 = self.do8(l8_3)

        # x8 = l3_1 + x8

        l9_1 = self.conv9(x8)
        l9_2 = self.BN9(l9_1)
        l9_3 = self.act9(l9_2)
        x9 = self.do9(l9_3)

        # x9 = l2_1 + x9 # 256,256,4

        l10_1 = self.conv10(x9)
        l10_2 = self.BN10(l10_1)
        l10_3 = self.act10(l10_2)
        x10 = self.do10(l10_3)

        out = self.conv11(x10)

        return out


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet_VGG(nn.Module):
    def __init__(self, num_classes, input_channels=1, **kwargs):
        super(UNet_VGG, self).__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet_VGG(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, deep_supervision=False, **kwargs):
        super(NestedUNet_VGG, self).__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class NestedUNet_VGGv2(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, deep_supervision=False, **kwargs):
        super(NestedUNet_VGGv2, self).__init__()

        nb_filter = [16, 32, 64, 128, 256]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class Unet(nn.Module):
    """EF segmentation network."""
    def __init__(self, input_nbr=1, label_nbr=16):
        super(Unet, self).__init__()

        self.input_nbr = input_nbr

        self.conv11 = nn.Conv2d(input_nbr, 16, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.do12 = nn.Dropout2d(p=0.2)

        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d(p=0.2)

        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(64)
        self.do33 = nn.Dropout2d(p=0.2)

        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(128)
        self.do43 = nn.Dropout2d(p=0.2)

        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(128)
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(128)
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(64)
        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(64)
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(64)
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(32)
        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(32)
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(16)
        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(16)
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(16, label_nbr, kernel_size=3, padding=1)

        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x1):
        """Forward method."""
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x1))))
        x12 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43, kernel_size=2, stride=2)

        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43.size(3) - x4d.size(3), 0, x43.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), x43), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33.size(3) - x3d.size(3), 0, x33.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), x33), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22.size(3) - x2d.size(3), 0, x22.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), x22), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12.size(3) - x1d.size(3), 0, x12.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), x12), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        return x11d


class convx2(nn.Module):
    def __init__(self, *ch):
        super(convx2, self).__init__()
        self.conv_number = len(ch) - 1
        self.model = nn.Sequential()
        for i in range(self.conv_number):
            self.model.add_module('conv{0}'.format(i), nn.Conv2d(ch[i], ch[i + 1], 3, 1, 1))

    def forward(self, x):
        y = self.model(x)
        return y


class FC_EF(nn.Module):
    def __init__(self, in_ch=1):
        super(FC_EF, self).__init__()
        self.conv1 = convx2(*[in_ch * 2, 16, 16])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = convx2(*[16, 32, 32])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = convx2(*[32, 64, 64, 64])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = convx2(*[64, 128, 128, 128])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv5 = convx2(*[256, 128, 128, 64])
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv6 = convx2(*[128, 64, 64, 32])
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv7 = convx2(*[64, 32, 16])
        self.deconv4 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.conv8 = convx2(*[32, 16, 1])

    def forward(self, x1, x2):
        h1 = self.conv1(torch.cat((x1, x2), 1))
        h = self.pool1(h1)
        h2 = self.conv2(h)
        h = self.pool2(h2)
        h3 = self.conv3(h)
        h = self.pool3(h3)
        h4 = self.conv4(h)
        h = self.pool4(h4)
        h = self.deconv1(h)
        h = self.conv5(torch.cat((h, h4), 1))
        h = self.deconv2(h)
        h = self.conv6(torch.cat((h, h3), 1))
        h = self.deconv3(h)
        h = self.conv7(torch.cat((h, h2), 1))
        h = self.deconv4(h)
        h = self.conv8(torch.cat((h, h1), 1))
        y = h
        return y


class CD_CNN_0728(nn.Module):
    def __init__(self):
        super(CD_CNN_0728, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.BN1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(True)
        self.do1 = nn.Dropout(p=0.2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.BN2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU(True)
        self.do2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.BN3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU(True)
        self.do3 = nn.Dropout(p=0.2)

        self.conv4 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2)
        self.BN4 = nn.BatchNorm2d(32)
        self.act4 = nn.ReLU(True)
        self.do4 = nn.Dropout(p=0.2)

        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.BN5 = nn.BatchNorm2d(32)
        self.act5 = nn.ReLU(True)
        self.do5 = nn.Dropout(p=0.2)

        self.sm = nn.Softmax2d()
        self.conv6 = nn.Conv2d(32, 16, kernel_size=1)

    def forward(self, x1):
        x1_1 = self.do1(self.act1(self.BN1(self.conv1(x1))))

        x1_2 = self.do2(self.act2(self.BN2(self.conv2(x1_1))))

        x1_3 = self.do3(self.act3(self.BN3(self.conv3(x1_2))))

        x1_4 = self.do4(self.act4(self.BN4(self.conv4(x1_3))))

        x1_5 = self.do5(self.act5(self.BN5(self.conv5(x1_4))))

        out = self.conv6(x1_5)

        return out


class FCN_class(nn.Module):
    def __init__(self):
        super(FCN_class, self).__init__()
        self.FCL = nn.Conv2d(1, 1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x1):
        out = self.sig(self.FCL(x1))

        return out


class CD_VGG_0804(nn.Module):
    def __init__(self):
        super(CD_VGG_0804, self).__init__()
        # 두 이미지 # 256,256,1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # 256,256,2
        self.act1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # 256,256,4
        self.BN2 = nn.BatchNorm2d(32)
        self.act2 = nn.ReLU(True)
        self.do2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 128,128,8
        self.BN3 = nn.BatchNorm2d(32)
        self.act3 = nn.ReLU(True)
        self.do3 = nn.Dropout(p=0.2)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 64,64,16
        self.BN4 = nn.BatchNorm2d(32)
        self.act4 = nn.ReLU(True)
        self.do4 = nn.Dropout(p=0.2)

        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 32,32,32
        self.BN5 = nn.BatchNorm2d(32)
        self.act5 = nn.ReLU(True)
        self.do5 = nn.Dropout(p=0.2)

        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 16,16,32
        self.BN6 = nn.BatchNorm2d(32)
        self.act6 = nn.ReLU(True)
        self.do6 = nn.Dropout(p=0.2)

        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 32,32,16
        self.BN7 = nn.BatchNorm2d(32)
        self.act7 = nn.ReLU(True)
        self.do7 = nn.Dropout(p=0.2)

        self.conv8 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 64,64,8
        self.BN8 = nn.BatchNorm2d(32)
        self.act8 = nn.ReLU(True)
        self.do8 = nn.Dropout(p=0.2)

        self.conv9 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 128,128,4
        self.BN9 = nn.BatchNorm2d(32)
        self.act9 = nn.ReLU(True)
        self.do9 = nn.Dropout(p=0.2)

        self.conv10 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 256,256,2
        self.BN10 = nn.BatchNorm2d(32)
        self.act10 = nn.ReLU(True)
        self.do10 = nn.Dropout(p=0.2)

        self.conv11 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        l1_1 = self.conv1(x)
        x1 = self.act1(l1_1)

        l2_1 = self.conv2(x1)
        l2_2 = self.BN2(l2_1)
        l2_3 = self.act2(l2_2)
        x2 = self.do2(l2_3)

        l3_1 = self.conv3(x2)
        l3_2 = self.BN3(l3_1)
        l3_3 = self.act3(l3_2)
        x3 = self.do3(l3_3)

        l4_1 = self.conv4(x3)
        l4_2 = self.BN4(l4_1)
        l4_3 = self.act4(l4_2)
        x4 = self.do4(l4_3)

        l5_1 = self.conv5(x4)
        l5_2 = self.BN5(l5_1)
        l5_3 = self.act5(l5_2)
        x5 = self.do5(l5_3)

        l6_1 = self.conv6(x5)
        l6_2 = self.BN6(l6_1)
        l6_3 = self.act6(l6_2)
        x6 = self.do6(l6_3)

        # x6 = l5_1 + x6 # 32,32,32

        l7_1 = self.conv7(x6)
        l7_2 = self.BN7(l7_1)
        l7_3 = self.act7(l7_2)
        x7 = self.do7(l7_3)

        # x7 = l4_1 + x7

        l8_1 = self.conv8(x7)
        l8_2 = self.BN8(l8_1)
        l8_3 = self.act8(l8_2)
        x8 = self.do8(l8_3)

        # x8 = l3_1 + x8

        l9_1 = self.conv9(x8)
        l9_2 = self.BN9(l9_1)
        l9_3 = self.act9(l9_2)
        x9 = self.do9(l9_3)

        # x9 = l2_1 + x9 # 256,256,4

        l10_1 = self.conv10(x9)
        l10_2 = self.BN10(l10_1)
        l10_3 = self.act10(l10_2)
        x10 = self.do10(l10_3)

        out = self.conv11(x10)

        return out


class CD_CNNcat_0812(nn.Module):
    def __init__(self):
        super(CD_CNNcat_0812, self).__init__()
        # 두 이미지 # 256,256,2
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2) # 256,256,2
        self.BN1 = nn.BatchNorm2d(8)
        self.act1 = nn.ReLU(True)
        self.do1 = nn.Dropout(p=0.2)

        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2) # 256,256,4
        self.BN2 = nn.BatchNorm2d(8)
        self.act2 = nn.ReLU(True)
        self.do2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2) # 128,128,8
        self.BN3 = nn.BatchNorm2d(8)
        self.act3 = nn.ReLU(True)
        self.do3 = nn.Dropout(p=0.2)

        self.conv4 = nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2) # 64,64,16
        self.BN4 = nn.BatchNorm2d(8)
        self.act4 = nn.ReLU(True)
        self.do4 = nn.Dropout(p=0.2)

        self.conv5 = nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2) # 32,32,32
        self.BN5 = nn.BatchNorm2d(8)
        self.act5 = nn.ReLU(True)
        self.do5 = nn.Dropout(p=0.2)

        self.conv6 = nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2)
        self.BN6 = nn.BatchNorm2d(8)
        self.act6 = nn.ReLU(True)
        self.do6 = nn.Dropout(p=0.2)

        self.conv7 = nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2)
        self.BN7 = nn.BatchNorm2d(8)
        self.act7 = nn.ReLU(True)
        self.do7 = nn.Dropout(p=0.2)

        self.conv8 = nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2)
        self.BN8 = nn.BatchNorm2d(8)
        self.act8 = nn.ReLU(True)
        self.do8 = nn.Dropout(p=0.2)

        self.conv9 = nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2)
        self.BN9 = nn.BatchNorm2d(8)
        self.act9 = nn.ReLU(True)
        self.do9 = nn.Dropout(p=0.2)

        self.conv10 = nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2)
        self.BN10 = nn.BatchNorm2d(8)
        self.act10 = nn.ReLU(True)
        self.do10 = nn.Dropout(p=0.2)

    def forward(self, x):
        l1_1 = self.conv1(x)
        l1_2 = self.BN1(l1_1)
        l1_3 = self.act1(l1_2)
        x1 = self.do1(l1_3)

        l2_1 = self.conv2(x1)
        l2_2 = self.BN2(l2_1)
        l2_3 = self.act2(l2_2)
        x2 = self.do2(l2_3)

        l3_1 = self.conv3(x2)
        l3_2 = self.BN3(l3_1)
        l3_3 = self.act3(l3_2)
        x3 = self.do3(l3_3)

        l4_1 = self.conv4(x3)
        l4_2 = self.BN4(l4_1)
        l4_3 = self.act4(l4_2)
        x4 = self.do4(l4_3)

        l5_1 = self.conv5(x4)
        l5_2 = self.BN5(l5_1)
        l5_3 = self.act5(l5_2)
        x5 = self.do5(l5_3)

        l6_1 = self.conv6(x5)
        l6_2 = self.BN6(l6_1)
        l6_3 = self.act6(l6_2)
        x6 = self.do6(l6_3)

        l7_1 = self.conv7(x6)
        l7_2 = self.BN7(l7_1)
        l7_3 = self.act7(l7_2)
        x7 = self.do7(l7_3)

        l8_1 = self.conv8(x7)
        l8_2 = self.BN8(l8_1)
        l8_3 = self.act8(l8_2)
        x8 = self.do8(l8_3)

        l9_1 = self.conv9(x8)
        l9_2 = self.BN9(l9_1)
        l9_3 = self.act9(l9_2)
        x9 = self.do9(l9_3)

        l10_1 = self.conv10(x9)
        l10_2 = self.BN10(l10_1)
        l10_3 = self.act10(l10_2)
        x10 = self.do10(l10_3)

        out = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], 1)

        return out


class CD_CNNcat_0813(nn.Module):
    def __init__(self):
        super(CD_CNNcat_0813, self).__init__()
        # 두 이미지 # 256,256,2
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2) # 256,256,2
        self.BN1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU(True)
        self.do1 = nn.Dropout(p=0.2)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2) # 256,256,4
        self.BN2 = nn.BatchNorm2d(16)
        self.act2 = nn.ReLU(True)
        self.do2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2) # 128,128,8
        self.BN3 = nn.BatchNorm2d(16)
        self.act3 = nn.ReLU(True)
        self.do3 = nn.Dropout(p=0.2)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2) # 64,64,16
        self.BN4 = nn.BatchNorm2d(16)
        self.act4 = nn.ReLU(True)
        self.do4 = nn.Dropout(p=0.2)

        self.conv5 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2) # 32,32,32
        self.BN5 = nn.BatchNorm2d(16)
        self.act5 = nn.ReLU(True)
        self.do5 = nn.Dropout(p=0.2)

        self.conv6 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2)
        self.BN6 = nn.BatchNorm2d(16)
        self.act6 = nn.ReLU(True)
        self.do6 = nn.Dropout(p=0.2)

        self.conv7 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2)
        self.BN7 = nn.BatchNorm2d(16)
        self.act7 = nn.ReLU(True)
        self.do7 = nn.Dropout(p=0.2)

        self.conv8 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2)
        self.BN8 = nn.BatchNorm2d(16)
        self.act8 = nn.ReLU(True)
        self.do8 = nn.Dropout(p=0.2)

        self.conv9 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2)
        self.BN9 = nn.BatchNorm2d(16)
        self.act9 = nn.ReLU(True)
        self.do9 = nn.Dropout(p=0.2)

        self.conv10 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2)
        self.BN10 = nn.BatchNorm2d(16)
        self.act10 = nn.ReLU(True)
        self.do10 = nn.Dropout(p=0.2)

    def forward(self, x):
        l1_1 = self.conv1(x)
        l1_2 = self.BN1(l1_1)
        l1_3 = self.act1(l1_2)
        x1 = self.do1(l1_3)

        l2_1 = self.conv2(x1)
        l2_2 = self.BN2(l2_1)
        l2_3 = self.act2(l2_2)
        x2 = self.do2(l2_3)

        l3_1 = self.conv3(x2)
        l3_2 = self.BN3(l3_1)
        l3_3 = self.act3(l3_2)
        x3 = self.do3(l3_3)

        l4_1 = self.conv4(x3)
        l4_2 = self.BN4(l4_1)
        l4_3 = self.act4(l4_2)
        x4 = self.do4(l4_3)

        l5_1 = self.conv5(x4)
        l5_2 = self.BN5(l5_1)
        l5_3 = self.act5(l5_2)
        x5 = self.do5(l5_3)

        l6_1 = self.conv6(x5)
        l6_2 = self.BN6(l6_1)
        l6_3 = self.act6(l6_2)
        x6 = self.do6(l6_3)

        l7_1 = self.conv7(x6)
        l7_2 = self.BN7(l7_1)
        l7_3 = self.act7(l7_2)
        x7 = self.do7(l7_3)

        l8_1 = self.conv8(x7)
        l8_2 = self.BN8(l8_1)
        l8_3 = self.act8(l8_2)
        x8 = self.do8(l8_3)

        l9_1 = self.conv9(x8)
        l9_2 = self.BN9(l9_1)
        l9_3 = self.act9(l9_2)
        x9 = self.do9(l9_3)

        l10_1 = self.conv10(x9)
        l10_2 = self.BN10(l10_1)
        l10_3 = self.act10(l10_2)
        x10 = self.do10(l10_3)

        out = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], 1)

        return out


class CD_CNNcat_0814(nn.Module):
    def __init__(self):
        super(CD_CNNcat_0814, self).__init__()
        # 두 이미지 # 256,256,2
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # 256,256,2
        self.BN1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU(True)
        self.do1 = nn.Dropout(p=0.2)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1) # 256,256,4
        self.BN2 = nn.BatchNorm2d(16)
        self.act2 = nn.ReLU(True)
        self.do2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1) # 128,128,8
        self.BN3 = nn.BatchNorm2d(16)
        self.act3 = nn.ReLU(True)
        self.do3 = nn.Dropout(p=0.2)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1) # 64,64,16
        self.BN4 = nn.BatchNorm2d(16)
        self.act4 = nn.ReLU(True)
        self.do4 = nn.Dropout(p=0.2)

        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1) # 32,32,32
        self.BN5 = nn.BatchNorm2d(16)
        self.act5 = nn.ReLU(True)
        self.do5 = nn.Dropout(p=0.2)

        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.BN6 = nn.BatchNorm2d(16)
        self.act6 = nn.ReLU(True)
        self.do6 = nn.Dropout(p=0.2)

        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.BN7 = nn.BatchNorm2d(16)
        self.act7 = nn.ReLU(True)
        self.do7 = nn.Dropout(p=0.2)

        self.conv8 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.BN8 = nn.BatchNorm2d(16)
        self.act8 = nn.ReLU(True)
        self.do8 = nn.Dropout(p=0.2)

        self.conv9 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.BN9 = nn.BatchNorm2d(16)
        self.act9 = nn.ReLU(True)
        self.do9 = nn.Dropout(p=0.2)

        self.conv10 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.BN10 = nn.BatchNorm2d(16)
        self.act10 = nn.ReLU(True)
        self.do10 = nn.Dropout(p=0.2)

    def forward(self, x):
        l1_1 = self.conv1(x)
        l1_2 = self.BN1(l1_1)
        l1_3 = self.act1(l1_2)
        x1 = self.do1(l1_3)

        l2_1 = self.conv2(x1)
        l2_2 = self.BN2(l2_1)
        l2_3 = self.act2(l2_2)
        x2 = self.do2(l2_3)

        l3_1 = self.conv3(x2)
        l3_2 = self.BN3(l3_1)
        l3_3 = self.act3(l3_2)
        x3 = self.do3(l3_3)

        l4_1 = self.conv4(x3)
        l4_2 = self.BN4(l4_1)
        l4_3 = self.act4(l4_2)
        x4 = self.do4(l4_3)

        l5_1 = self.conv5(x4)
        l5_2 = self.BN5(l5_1)
        l5_3 = self.act5(l5_2)
        x5 = self.do5(l5_3)

        l6_1 = self.conv6(x5)
        l6_2 = self.BN6(l6_1)
        l6_3 = self.act6(l6_2)
        x6 = self.do6(l6_3)

        l7_1 = self.conv7(x6)
        l7_2 = self.BN7(l7_1)
        l7_3 = self.act7(l7_2)
        x7 = self.do7(l7_3)

        l8_1 = self.conv8(x7)
        l8_2 = self.BN8(l8_1)
        l8_3 = self.act8(l8_2)
        x8 = self.do8(l8_3)

        l9_1 = self.conv9(x8)
        l9_2 = self.BN9(l9_1)
        l9_3 = self.act9(l9_2)
        x9 = self.do9(l9_3)

        l10_1 = self.conv10(x9)
        l10_2 = self.BN10(l10_1)
        l10_3 = self.act10(l10_2)
        x10 = self.do10(l10_3)

        out = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], 1)

        return out


class Total_V8div_UNetV2(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, **kwargs):
        super(Total_V8div_UNetV2, self).__init__()
        # Start Despeckling CNN
        self.pad1 = ReplicationPad2d(2)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(5, 5))
        self.act1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.2)

        self.pad2 = ReplicationPad2d(2)
        self.conv2 = nn.Conv2d(1, 64, kernel_size=(5, 5))
        self.BN2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.2)

        self.pad3 = ReplicationPad2d(2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.BN3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()
        self.do3 = nn.Dropout(p=0.2)

        self.pad4 = ReplicationPad2d(2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.BN4 = nn.BatchNorm2d(64)
        self.act4 = nn.ReLU()
        self.do4 = nn.Dropout(p=0.2)

        self.pad5 = ReplicationPad2d(2)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.BN5 = nn.BatchNorm2d(64)
        self.act5 = nn.ReLU()
        self.do5 = nn.Dropout(p=0.2)

        self.pad6 = ReplicationPad2d(2)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.BN6 = nn.BatchNorm2d(64)
        self.act6 = nn.ReLU()
        self.do6 = nn.Dropout(p=0.2)

        self.pad7 = ReplicationPad2d(2)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.BN7 = nn.BatchNorm2d(64)
        self.act7 = nn.ReLU()
        self.do7 = nn.Dropout(p=0.2)

        self.pad8 = ReplicationPad2d(2)
        self.conv8 = nn.Conv2d(64, 1, kernel_size=(5, 5))
        self.act8 = nn.ReLU()

        self.actf = nn.Tanh()

        # Start Change Detection CNN
        nb_filter = [16, 32, 64, 128, 256]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        x_in = x
        # Layer 1
        L1_1 = self.conv1(self.pad1(x_in))
        x1 = self.act1(L1_1)

        # Layer 2~7
        L2_1 = self.conv2(self.pad2(x1))
        L2_2 = self.BN2(L2_1)
        x2 = self.act2(L2_2)

        L3_1 = self.conv3(self.pad3(x2))
        L3_2 = self.BN3(L3_1)
        x3 = self.act3(L3_2)

        L4_1 = self.conv4(self.pad4(x3))
        L4_2 = self.BN4(L4_1)
        x4 = self.act4(L4_2)

        L5_1 = self.conv5(self.pad5(x4))
        L5_2 = self.BN5(L5_1)
        x5 = self.act5(L5_2)

        L6_1 = self.conv6(self.pad6(x5))
        L6_2 = self.BN6(L6_1)
        x6 = self.act6(L6_2)

        L7_1 = self.conv7(self.pad7(x6))
        L7_2 = self.BN7(L7_1)
        x7 = self.act7(L7_2)

        # Layer 8
        L8_1 = self.conv8(self.pad8(x7))

        out = torch.div(x_in, self.act8(L8_1)+1e-10)
        out = self.actf(out)

        # Start Change Detection
        x0_0 = self.conv0_0(out)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)

        return output


class HRSCD_Unspeckle(Dataset):
    def __init__(self, train=True, transform=None, number=1000):
        super(HRSCD_Unspeckle, self).__init__

        self.transform = transform

        if number == 1000:
            train_test = 1200
        elif number == 2000:
            train_test = 5000
        elif number == 2500:
            train_test = 8000
        else:
            train_test = 31000

        IMAGE06_PATH, IMAGE12_PATH, LABEL_PATH = [], [], []

        BasePath = 'C:/Users/cud01/Desktop/ChangeDetection/DataForCD/'

        path06 = BasePath + '/2006/06_' + str(number) + 'c/'
        path12 = BasePath + '/2012/12_' + str(number) + 'c/'
        path_label = BasePath + '/Label/label_' + str(number) + 'c/'

        FolderPaths06 = [os.path.join(path06, file_name) for file_name in os.listdir(path06)]
        FolderPaths12 = [os.path.join(path12, file_name) for file_name in os.listdir(path12)]
        FolderPaths_label = [os.path.join(path_label, file_name) for file_name in os.listdir(path_label)]

        IMAGE06_PATH.extend(FolderPaths06)
        IMAGE12_PATH.extend(FolderPaths12)
        LABEL_PATH.extend(FolderPaths_label)

        if train is True:
            self.train06 = IMAGE06_PATH[:-train_test]
            self.train12 = IMAGE12_PATH[:-train_test]
            self.label = LABEL_PATH[:-train_test]
        else:
            self.train06 = IMAGE06_PATH[-train_test:]
            self.train12 = IMAGE12_PATH[-train_test:]
            self.label = LABEL_PATH[-train_test:]

    def __getitem__(self, idx):
        train06 = self.train06[idx]
        train12 = self.train12[idx]
        label = self.label[idx]

        img_t06 = Image.open(train06)
        img_t12 = Image.open(train12)
        img_l = Image.open(label)

        if self.transform is not None:
            img_t06 = self.transform(img_t06)
            img_t12 = self.transform(img_t12)
            img_l = self.transform(img_l)

        DI = abs(img_t12 - img_t06)
        DR = img_t12 / img_t06
        DR = DR.type(torch.ByteTensor)

        x = [np.array(img_t06), np.array(img_t12), np.array(DI), np.array(DR)]
        x = torch.tensor(x).squeeze(1)

        return x, img_l

    def __len__(self):
        return len(self.train06)

## 7월에 바꾼 것들
def get_dataloader_0714(bs, num):
    transform = transforms.ToTensor()

    train_dataset = HRSCD_Unspeckle_0714(train=True, transform=transform, number=num)
    test_dataset = HRSCD_Unspeckle_0714(train=False, transform=transform, number=num)

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=1)

    return train_dataloader, test_dataloader

class HRSCD_Unspeckle_0714(Dataset):
    def __init__(self, train=True, transform=None, number=1000):
        super(HRSCD_Unspeckle_0714, self).__init__

        self.transform = transform

        if number == 1000:
            train_test = 1200
        elif number == 2000:
            train_test = 5000
        elif number == 2500:
            train_test = 8500
        else:
            train_test = 32000

        IMAGE06_PATH, IMAGE12_PATH, LABEL_PATH = [], [], []

        BasePath = 'C:/Users/cud01/Desktop/ChangeDetection/DataForCD/'

        path06 = BasePath + '2006/06_' + str(number) + 'c/'
        path12 = BasePath + '2012/12_' + str(number) + 'c/'
        path_label = BasePath + 'Label/label_' + str(number) + 'c/'

        FolderPaths06 = [os.path.join(path06, file_name) for file_name in os.listdir(path06)]
        FolderPaths12 = [os.path.join(path12, file_name) for file_name in os.listdir(path12)]
        FolderPaths_label = [os.path.join(path_label, file_name) for file_name in os.listdir(path_label)]

        IMAGE06_PATH.extend(FolderPaths06)
        IMAGE12_PATH.extend(FolderPaths12)
        LABEL_PATH.extend(FolderPaths_label)

        if train is True:
            self.train06 = IMAGE06_PATH[:train_test]
            self.train12 = IMAGE12_PATH[:train_test]
            self.label = LABEL_PATH[:train_test]
        else:
            self.train06 = IMAGE06_PATH[train_test:]
            self.train12 = IMAGE12_PATH[train_test:]
            self.label = LABEL_PATH[train_test:]

    def __len__(self):
        return len(self.train06)

    def __getitem__(self, idx):
        train06 = self.train06[idx]
        train12 = self.train12[idx]
        label = self.label[idx]

        mms1 = MMS()
        mms2 = MMS()
        mms3 = MMS()

        x06 = Image.open(train06)
        x12 = Image.open(train12)
        img_l = Image.open(label)

        if self.transform is not None:
            x06 = self.transform(x06)
            x12 = self.transform(x12)
            img_l = self.transform(img_l)

        di = abs(x12 - x06)

        mms1.fit(x06)
        mms2.fit(x12)
        x06 = mms1.transform(x06)
        x12 = mms2.transform(x12)

        dr = torch.div(x12, x06)
        mms3.fit(dr)
        dr = mms3.transform(dr) * 255
        dr = dr.type(torch.ByteTensor)

        x = torch.tensor([di, dr]).squeeze(1)
        print(type(x), x.shape)
        print(type(img_l), img_l.shape)

        # x06 = torch.tensor(img_t06).squeeze(1)
        # x12 = torch.tensor(img_t12).squeeze(1)

        return x, img_l

def get_dataloader_0716(bs, num):
    transform = transforms.ToTensor()

    train_dataset = HRSCD_Unspeckle_0716(train=True, transform=transform, number=num)
    test_dataset = HRSCD_Unspeckle_0716(train=False, transform=transform, number=num)

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=8)

    return train_dataloader, test_dataloader

def get_dataloader_0716_prac(bs, num):
    transform = transforms.ToTensor()

    train_dataset = HRSCD_Unspeckle_0716(train=True, transform=transform, number=num)
    test_dataset = HRSCD_Unspeckle_0716(train=False, transform=transform, number=num)

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=1)

    return train_dataloader, test_dataloader

# image diff, image ratio(+eps)
class HRSCD_Unspeckle_0716(Dataset):
    def __init__(self, train=True, transform=None, number=1000):
        super(HRSCD_Unspeckle_0716, self).__init__

        self.transform = transform

        if number == 1000:
            train_test = 1200
        elif number == 2000:
            train_test = 5000
        elif number == 2500:
            train_test = 8000
        else:
            train_test = 30000

        IMAGE06_PATH, IMAGE12_PATH, LABEL_PATH = [], [], []

        BasePath = 'C:/Users/cud01/Desktop/ChangeDetection/DataForCD/'

        path06 = BasePath + '2006/06_' + str(number) + 'c/'
        path12 = BasePath + '2012/12_' + str(number) + 'c/'
        path_label = BasePath + 'Label/label_' + str(number) + 'c/'

        FolderPaths06 = [os.path.join(path06, file_name) for file_name in os.listdir(path06)]
        FolderPaths12 = [os.path.join(path12, file_name) for file_name in os.listdir(path12)]
        FolderPaths_label = [os.path.join(path_label, file_name) for file_name in os.listdir(path_label)]

        IMAGE06_PATH.extend(FolderPaths06)
        IMAGE12_PATH.extend(FolderPaths12)
        LABEL_PATH.extend(FolderPaths_label)

        if train is True:
            self.train06 = IMAGE06_PATH[:train_test]
            self.train12 = IMAGE12_PATH[:train_test]
            self.label = LABEL_PATH[:train_test]
        else:
            self.train06 = IMAGE06_PATH[train_test:]
            self.train12 = IMAGE12_PATH[train_test:]
            self.label = LABEL_PATH[train_test:]

    def __len__(self):
        return len(self.train06)

    def __getitem__(self, idx):
        train06 = self.train06[idx]
        train12 = self.train12[idx]
        label = self.label[idx]

        mms = MMS()

        x06 = Image.open(train06)
        x12 = Image.open(train12)
        img_l = Image.open(label)

        if self.transform is not None:
            x06 = self.transform(x06)
            x12 = self.transform(x12)
            img_l = self.transform(img_l)
        eps = 0.0000000001
        # Difference Image
        di = abs(x12 - x06)

        # Ratio Image
        x06 = x06.squeeze(0)
        x12 = x12.squeeze(0)
        #
        # mms1.fit(x06)
        # mms2.fit(x12)
        # x06 = mms1.transform(x06)
        # x12 = mms2.transform(x12)

        dr = torch.div(torch.as_tensor(x06), torch.as_tensor(x12+eps))
        mms.fit(dr)
        dr = mms.transform(dr) * 255
        dr = torch.tensor(dr).unsqueeze(0).type(torch.ByteTensor)

        # Output
        x = torch.tensor([np.array(di), np.array(dr)]).squeeze(1)
        # print(type(x), x.shape)
        # print(type(img_l), img_l.shape)

        # x06 = torch.tensor(img_t06).squeeze(1)
        # x12 = torch.tensor(img_t12).squeeze(1)

        return x, img_l # x6

def get_dataloader_0721(bs, num):
    transform = transforms.ToTensor()

    train_dataset = HRSCD_Unspeckle_0721(train=True, transform=transform, number=num)
    test_dataset = HRSCD_Unspeckle_0721(train=False, transform=transform, number=num)

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=8)

    return train_dataloader, test_dataloader

def get_dataloader_0721_prac(bs, num):
    transform = transforms.ToTensor()

    train_dataset = HRSCD_Unspeckle_0721(train=True, transform=transform, number=num)
    test_dataset = HRSCD_Unspeckle_0721(train=False, transform=transform, number=num)

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=1)

    return train_dataloader, test_dataloader

# x6, x12
class HRSCD_Unspeckle_0721(Dataset):
    def __init__(self, train=True, transform=None, number=1000):
        super(HRSCD_Unspeckle_0721, self).__init__

        self.transform = transform

        if number == 1000:
            train_test = 1200
        elif number == 2000:
            train_test = 5000
        elif number == 2500:
            train_test = 8000
        else:
            train_test = 30000

        IMAGE06_PATH, IMAGE12_PATH, LABEL_PATH = [], [], []

        BasePath = 'C:/Users/cud01/Desktop/ChangeDetection/DataForCD/'

        path06 = BasePath + '2006/06_' + str(number) + 'c/'
        path12 = BasePath + '2012/12_' + str(number) + 'c/'
        path_label = BasePath + 'Label/label_' + str(number) + 'c/'

        FolderPaths06 = [os.path.join(path06, file_name) for file_name in os.listdir(path06)]
        FolderPaths12 = [os.path.join(path12, file_name) for file_name in os.listdir(path12)]
        FolderPaths_label = [os.path.join(path_label, file_name) for file_name in os.listdir(path_label)]

        IMAGE06_PATH.extend(FolderPaths06)
        IMAGE12_PATH.extend(FolderPaths12)
        LABEL_PATH.extend(FolderPaths_label)

        if train is True:
            self.train06 = IMAGE06_PATH[:train_test]
            self.train12 = IMAGE12_PATH[:train_test]
            self.label = LABEL_PATH[:train_test]
        else:
            self.train06 = IMAGE06_PATH[train_test:]
            self.train12 = IMAGE12_PATH[train_test:]
            self.label = LABEL_PATH[train_test:]

    def __len__(self):
        return len(self.train06)

    def __getitem__(self, idx):
        train06 = self.train06[idx]
        train12 = self.train12[idx]
        label = self.label[idx]

        # mms = MMS()

        x06 = Image.open(train06)
        x12 = Image.open(train12)
        img_l = Image.open(label)

        if self.transform is not None:
            x06 = self.transform(x06)
            x12 = self.transform(x12)
            img_l = self.transform(img_l)

        # eps = 0.0000000001
        # Difference Image
        # di = abs(x12 - x06)

        # Ratio Image
        x06 = x06.squeeze(0)
        x12 = x12.squeeze(0)
        #
        # mms1.fit(x06)
        # mms2.fit(x12)
        # x06 = mms1.transform(x06)
        # x12 = mms2.transform(x12)

        # dr = torch.div(torch.as_tensor(x06), torch.as_tensor(x12+eps))
        # mms.fit(dr)
        # dr = mms.transform(dr) * 255
        # dr = torch.tensor(dr).unsqueeze(0).type(torch.ByteTensor)

        # Output
        x = torch.tensor([np.array(x06), np.array(x12)]).squeeze(1)
        # print(type(x), x.shape)
        # print(type(img_l), img_l.shape)

        # x06 = torch.tensor(img_t06).squeeze(1)
        # x12 = torch.tensor(img_t12).squeeze(1)

        return x, img_l

# image diff, image ratio(+1)
class HRSCD_Unspeckle_0723(Dataset):
    def __init__(self, train=True, transform=None, number=1000):
        super(HRSCD_Unspeckle_0723, self).__init__

        self.transform = transform

        if number == 1000:
            train_test = 1200
        elif number == 2000:
            train_test = 5000
        elif number == 2500:
            train_test = 8000
        else:
            train_test = 30000

        IMAGE06_PATH, IMAGE12_PATH, LABEL_PATH = [], [], []

        BasePath = 'C:/Users/cud01/Desktop/ChangeDetection/DataForCD/'

        path06 = BasePath + '2006/06_' + str(number) + 'c/'
        path12 = BasePath + '2012/12_' + str(number) + 'c/'
        path_label = BasePath + 'Label/label_' + str(number) + 'c/'

        FolderPaths06 = [os.path.join(path06, file_name) for file_name in os.listdir(path06)]
        FolderPaths12 = [os.path.join(path12, file_name) for file_name in os.listdir(path12)]
        FolderPaths_label = [os.path.join(path_label, file_name) for file_name in os.listdir(path_label)]

        IMAGE06_PATH.extend(FolderPaths06)
        IMAGE12_PATH.extend(FolderPaths12)
        LABEL_PATH.extend(FolderPaths_label)

        if train is True:
            self.train06 = IMAGE06_PATH[:train_test]
            self.train12 = IMAGE12_PATH[:train_test]
            self.label = LABEL_PATH[:train_test]
        else:
            self.train06 = IMAGE06_PATH[train_test:]
            self.train12 = IMAGE12_PATH[train_test:]
            self.label = LABEL_PATH[train_test:]

    def __len__(self):
        return len(self.train06)

    def __getitem__(self, idx):
        train06 = self.train06[idx]
        train12 = self.train12[idx]
        label = self.label[idx]

        mms = MMS()

        x06 = Image.open(train06)
        x12 = Image.open(train12)
        img_l = Image.open(label)

        if self.transform is not None:
            x06 = self.transform(x06)
            x12 = self.transform(x12)
            img_l = self.transform(img_l)
        # Difference Image
        di = abs(x12 - x06)

        # Ratio Image
        x06 = x06.squeeze(0)
        x12 = x12.squeeze(0)
        #
        # mms1.fit(x06)
        # mms2.fit(x12)
        # x06 = mms1.transform(x06)
        # x12 = mms2.transform(x12)

        dr = torch.div(torch.as_tensor(x06+1), torch.as_tensor(x12+1))
        mms.fit(dr)
        dr = mms.transform(dr) * 255
        dr = torch.tensor(dr).unsqueeze(0).type(torch.ByteTensor)

        # Output
        x = torch.tensor([np.array(di), np.array(dr)]).squeeze(1)
        # print(type(x), x.shape)
        # print(type(img_l), img_l.shape)

        # x06 = torch.tensor(img_t06).squeeze(1)
        # x12 = torch.tensor(img_t12).squeeze(1)

        return x, img_l # x6

def get_dataloader_0723(bs, num):
    transform = transforms.ToTensor()

    train_dataset = HRSCD_Unspeckle_0723(train=True, transform=transform, number=num)
    test_dataset = HRSCD_Unspeckle_0723(train=False, transform=transform, number=num)

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=8)

    return train_dataloader, test_dataloader

def get_dataloader_0723_prac(bs, num):
    transform = transforms.ToTensor()

    train_dataset = HRSCD_Unspeckle_0723(train=True, transform=transform, number=num)
    test_dataset = HRSCD_Unspeckle_0723(train=False, transform=transform, number=num)

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=1)

    return train_dataloader, test_dataloader

# x06, x12 따로
class HRSCD_x6x12_0724(Dataset):
    def __init__(self, train=True, transform=None, number=1000):
        super(HRSCD_x6x12_0724, self).__init__

        self.transform = transform

        if number == 1000:
            train_test = 1200
        elif number == 2000:
            train_test = 5000
        elif number == 2500:
            train_test = 8000
        else:
            train_test = 30000

        IMAGE06_PATH, IMAGE12_PATH, LABEL_PATH = [], [], []

        BasePath = 'C:/Users/cud01/Desktop/ChangeDetection/DataForCD/'

        path06 = BasePath + '2006/06_' + str(number) + 'c/'
        path12 = BasePath + '2012/12_' + str(number) + 'c/'
        path_label = BasePath + 'Label/label_' + str(number) + 'c/'

        FolderPaths06 = [os.path.join(path06, file_name) for file_name in os.listdir(path06)]
        FolderPaths12 = [os.path.join(path12, file_name) for file_name in os.listdir(path12)]
        FolderPaths_label = [os.path.join(path_label, file_name) for file_name in os.listdir(path_label)]

        IMAGE06_PATH.extend(FolderPaths06)
        IMAGE12_PATH.extend(FolderPaths12)
        LABEL_PATH.extend(FolderPaths_label)

        if train is True:
            self.train06 = IMAGE06_PATH[:train_test]
            self.train12 = IMAGE12_PATH[:train_test]
            self.label = LABEL_PATH[:train_test]
        else:
            self.train06 = IMAGE06_PATH[train_test:]
            self.train12 = IMAGE12_PATH[train_test:]
            self.label = LABEL_PATH[train_test:]

    def __len__(self):
        return len(self.train06)

    def __getitem__(self, idx):
        train06 = self.train06[idx]
        train12 = self.train12[idx]
        label = self.label[idx]

        # mms = MMS()

        x06 = Image.open(train06)
        x12 = Image.open(train12)
        img_l = Image.open(label)

        if self.transform is not None:
            x06 = self.transform(x06)
            x12 = self.transform(x12)
            img_l = self.transform(img_l)

        # eps = 0.0000000001
        # Difference Image
        # di = abs(x12 - x06)

        # Ratio Image
        # x06 = x06.squeeze(0)
        # x12 = x12.squeeze(0)
        #
        # mms1.fit(x06)
        # mms2.fit(x12)
        # x06 = mms1.transform(x06)
        # x12 = mms2.transform(x12)

        # dr = torch.div(torch.as_tensor(x06), torch.as_tensor(x12+eps))
        # mms.fit(dr)
        # dr = mms.transform(dr) * 255
        # dr = torch.tensor(dr).unsqueeze(0).type(torch.ByteTensor)

        # Output
        # x = torch.cat([x06, x12], 1).unsqueeze(0)
        # print(type(x), x.shape)
        # print(type(img_l), img_l.shape)

        # x06 = torch.tensor(img_t06).squeeze(1)
        # x12 = torch.tensor(img_t12).squeeze(1)

        return x06, x12, img_l

def get_dataloader_x6x12_0724(bs, num):
    transform = transforms.ToTensor()

    train_dataset = HRSCD_x6x12_0724(train=True, transform=transform, number=num)
    test_dataset = HRSCD_x6x12_0724(train=False, transform=transform, number=num)

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=8)

    return train_dataloader, test_dataloader

def get_dataloader_x6x12_0724_p(bs, num):
    transform = transforms.ToTensor()

    train_dataset = HRSCD_x6x12_0724(train=True, transform=transform, number=num)
    test_dataset = HRSCD_x6x12_0724(train=False, transform=transform, number=num)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    return train_dataloader, test_dataloader

class PAM_210831(Dataset):
    def __init__(self, train=True, transform=None):
        super(PAM_210831, self).__init__

        self.transform = transform

        A_PATH, B_PATH, L_PATH = [], [], []

        if train is True:
            BasePath = 'C:/Data/CDdata/PAM/train_he_size_crop/'
        else:
            BasePath = 'C:/Data/CDdata/PAM/test_he_size_crop/'

        pathA = BasePath + 'A/'
        pathB = BasePath + 'B/'
        pathL = BasePath + 'label/'

        FolderPathsA = [os.path.join(pathA, file_name) for file_name in os.listdir(pathA)]
        FolderPathsB = [os.path.join(pathB, file_name) for file_name in os.listdir(pathB)]
        FolderPathsL = [os.path.join(pathL, file_name) for file_name in os.listdir(pathL)]

        A_PATH.extend(FolderPathsA)
        B_PATH.extend(FolderPathsB)
        L_PATH.extend(FolderPathsL)

        self.A = A_PATH
        self.B = B_PATH
        self.L = L_PATH

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        train06 = self.A[idx]
        train12 = self.B[idx]
        label = self.L[idx]

        xA = Image.open(train06)
        xB = Image.open(train12)
        L = Image.open(label)
        L = Image2Bin_for_dataloader(L)

        if self.transform is not None:
            xA = self.transform(xA)
            xB = self.transform(xB)
            L = self.transform(L)

        return xA, xB, L


def PAMdataloader_train(bs):
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    train_dataset = PAM_210831(train=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=10)

    return train_dataloader


def PAMdataloader_eval():
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    test_dataset = PAM_210831(train=False, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=10)

    return test_dataloader


def PAMdataloader_test():
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    test_dataset = PAM_210831(train=False, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)

    return test_dataloader

class SARdata(Dataset):
    def __init__(self, transform=None):
        super(SARdata, self).__init__

        self.transform = transform
        basepath = 'C:/Users/user/Desktop/Paper_211025/figs/'
        A_PATH, B_PATH  = [], []

        pathA = basepath + 'SARimg/sub2018/'
        pathB = basepath + 'SARimg/sub2019/'

        FolderPathsA = [os.path.join(pathA, file_name) for file_name in os.listdir(pathA)]
        FolderPathsB = [os.path.join(pathB, file_name) for file_name in os.listdir(pathB)]

        A_PATH.extend(FolderPathsA)
        B_PATH.extend(FolderPathsB)

        self.A = A_PATH
        self.B = B_PATH

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        train06 = self.A[idx]
        train12 = self.B[idx]

        xA = Image.open(train06)
        xB = Image.open(train12)

        if self.transform is not None:
            xA = self.transform(xA)
            xB = self.transform(xB)

        return xA, xB

def SAR_testdataloader():
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    test_dataset = SARdata(transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=5)

    return test_dataloader

def HistogramEq(path, save_path):
    # path1 = 'D:/CD_dataset/images_2006/'
    # path1 = 'D:/CD_dataset/images_2012/'

    imagePaths1 = [os.path.join(path, file_name) for file_name in os.listdir(path)]
    i = 0

    for imagepath1 in imagePaths1:
        img = cv2.imread(imagepath1, 0)

        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        img = clahe.apply(img)

        Image.fromarray(img).save(save_path + str(i) + '.tiff')

        i += 1


def ImageSize(directory, save_dir, wanted_size):
    imagePaths1 = [os.path.join(directory, file_name) for file_name in os.listdir(directory)]
    i = 0

    for imagePath in imagePaths1:
        img = cv2.imread(imagePath, 0)

        resize_img = cv2.resize(img, wanted_size)

        Image.fromarray(resize_img).save(save_dir + str(i) + '.tiff')
        i = i + 1


def ImageCrop(dir_in, dir_out, itera):
    crop_size = (256, 256)
    im = Image.open(dir_in)

    img_size = im.size

    iter_num = (math.ceil(img_size[0] / crop_size[0]), math.ceil(img_size[1] / crop_size[1]))

    start_pt0 = np.linspace(0, img_size[0] - crop_size[0], iter_num[0])
    start_pt1 = np.linspace(0, img_size[1] - crop_size[1], iter_num[1])

    num_v, num_h = 0, 0

    for i in range(iter_num[0]):
        for j in range(iter_num[1]):
            src = Image.fromarray(cv2.imread(dir_in, 0))

            crop_image = im.crop((start_pt0[i], start_pt1[j], start_pt0[i] + crop_size[0], start_pt1[j] + crop_size[1]))
            crop_image.save(dir_out + str(itera) + '.tiff')

            if itera % 500 == 0:
                print(itera)

            itera += 1

    return itera


def train_CD_unspeckled_cat(model, optimizer, criterion, train_dataloader, dev):
    model.train()
    # print("\nEpoch of now: ", epoch)
    global_step, global_step_test = 0, 0
    Loss, Losst = [], []

    print('\n-----START TRAIN-----\n')
    for x, y in tqdm(train_dataloader):
        global_step += 1
        # CPU to GPU
        model = model.to(dev)
        x = x.to(dev)
        y = y.to(dev)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Loss.append(loss.item())

        if global_step % 100 == 0:
            print('\ntrain loss NOW = ', loss.item())
            print('train loss AVE = ', sum(Loss) / len(Loss))


def train_CD_unspeckled(model, model_c, optimizer, criterion1, criterion2, train_dataloader, dev, epoch):
    model.train()

    # print("\nEpoch of now: ", epoch)
    global_step, global_step_test = 0, 0
    Loss = []

    pbar = tqdm(total=len(train_dataloader), desc="EPOCH {}".format(epoch))
    for _, data in enumerate(train_dataloader):
        x06, x12, y = data

        global_step += 1
        # CPU to GPU
        model = model.to(dev)

        x06 = x06.to(dev)
        x12 = x12.to(dev)
        y = y.to(dev)

        pred = model(x06, x12)
        loss1 = criterion1(pred, y)
        loss2 = criterion2(pred, y)

        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Loss.append(loss.item())

        pbar.set_postfix_str("AverageLoss={AveLoss:9.6f}".format(AveLoss=sum(Loss) / len(Loss)))
        pbar.update(1)
    pbar.close()

    # model.eval()
    # with torch.no_grad():
    #     print('\n-----START TEST-----\n')
    #     for xt, yt in tqdm(test_dataloader):
    #
    #         global_step_test += 1
    #
    #         xt = xt.to(dev)
    #         yt = yt.to(dev)
    #
    #         predt = model(xt)
    #
    #         losst = criterion(predt, yt)
    #         Losst.append(losst.item())
    #
    #         if global_step_test % 100 == 0:
    #             print('\nTest loss NOW = ', losst.item())
    #             print('Test loss AVE = ', sum(Losst) / len(Losst))


def train_CD_0728(model, optimizer, scheduler, train_dataloader, dev, epoch, firstloss):
    model.train()
    # model_c.train()
    Loss = []
    beforeLoss = firstloss

    ckpt = {'model': model.state_dict()}

    pbar = tqdm(total=len(train_dataloader), desc="EPOCH {}".format(epoch))
    for _, data in enumerate(train_dataloader):
        x06, x12, y = data

        # CPU to GPU
        model = model.to(dev)
        # model_c = model_c.to(dev)
        x06 = x06.to(dev)
        x12 = x12.to(dev)
        y = y.to(dev)
        # criterion = criterion.to(dev)

        pred1 = model(x06)
        pred2 = model(x12)

        distance = F.pairwise_distance(pred1, pred2, p=2.0, keepdim=True)
        # distance = model_c(distance)

        loss = torch.sum((1 - y) * torch.pow(distance, 2)
                         + y * torch.pow(torch.clamp(2 - distance, min=0.0), 2))
        # loss = loss + criterion(distance, y)

        optimizer.zero_grad()
        loss.backward()
        scheduler.step(loss)
        optimizer.step()

        Loss.append(loss.item())
        averageLoss = sum(Loss) / len(Loss)

        if averageLoss < beforeLoss:
            torch.save(ckpt, './pths/CD_NestedUNet_PAM_Cont2D_210827_1.pth')
        beforeLoss = averageLoss

        pbar.set_postfix_str("AverageLoss={AveLoss:9.6f}".format(AveLoss=averageLoss))
        pbar.update(1)
    pbar.close()

    return model, averageLoss, optimizer, scheduler

    # model.eval()
    # with torch.no_grad():
    #     print('\n-----START TEST-----\n')
    #     for xt, yt in tqdm(test_dataloader):
    #
    #         global_step_test += 1
    #
    #         xt = xt.to(dev)
    #         yt = yt.to(dev)
    #
    #         predt = model(xt)
    #
    #         losst = criterion(predt, yt)
    #         Losst.append(losst.item())
    #
    #         if global_step_test % 100 == 0:
    #             print('\nTest loss NOW = ', losst.item())
    #             print('Test loss AVE = ', sum(Losst) / len(Losst))


def test_CD_unspeckled(model, criterion, test_dataloader):
    model.eval()
    with torch.no_grad():
        for xt, yt in tqdm(test_dataloader):
            M, _ = torch.max(yt, 1)
            m = torch.max(torch.tensor(M))

            if int(m) >= 0.5:
                predt = model(xt)

                losst = criterion(predt, yt)
                plt.imshow(np.array(yt.to('cpu')[0].squeeze(0).squeeze(0)))
                plt.gray()
                plt.show()

                plt.imshow(np.array(predt.to('cpu')[0].squeeze(0).squeeze(0)))
                plt.gray()
                plt.show()

                print(losst.item())


def test_CD_PAM(model, test_dataloader):
    model.eval()
    with torch.no_grad():
        for x1, x2, yt in tqdm(test_dataloader):
            M, _ = torch.max(yt, 1)
            m = torch.max(torch.tensor(M))

            if int(m) >= 0.5:
                pred1 = model(x1)
                pred2 = model(x2)

                distance = F.pairwise_distance(pred1, pred2, p=2.0, keepdim=True)

                losst = torch.sum((1 - yt) * torch.pow(distance, 2)
                                  + yt * torch.pow(torch.clamp(2 - distance, min=0.0), 2))

                plt.imshow(np.array(yt.to('cpu')[0].squeeze(0).squeeze(0)))
                plt.gray()
                plt.show()

                plt.imshow(np.array(pred1.to('cpu')[0].squeeze(0).squeeze(0)))
                plt.gray()
                plt.show()

                plt.imshow(np.array(pred2.to('cpu')[0].squeeze(0).squeeze(0)))
                plt.gray()
                plt.show()

                plt.imshow(np.array(distance.to('cpu')[0].squeeze(0).squeeze(0)))
                plt.gray()
                plt.show()

                print(losst.item())


class BalancedLoss(nn.Module):
    def __init__(self, neg_weight=1.0):
        super(BalancedLoss, self).__init__()
        self.neg_weight = neg_weight

    def forward(self, input, target):
        pos_mask = (target != 0)
        neg_mask = (target == 0)

        pos_num = pos_mask.sum().float()
        neg_num = neg_mask.sum().float()

        weight = target.new_zeros(target.size())

        weight[pos_mask] = 1 / pos_num
        weight[neg_mask] = 1 / neg_num * self.neg_weight

        weight /= weight.sum()
        return F.binary_cross_entropy_with_logits(
            input, target, weight, reduction='sum')


class ConstractiveThresholdHingeLoss(nn.Module):

    def __init__(self, hingethresh=0.0, margin=2.0):
        super(ConstractiveThresholdHingeLoss, self).__init__()
        self.threshold = hingethresh
        self.margin = margin

    def forward(self, out_vec_t0, out_vec_t1, label):
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
        similar_pair = torch.clamp(distance - self.threshold, min=0.0)
        dissimilar_pair = torch.clamp(self.margin - distance, min=0.0)
        # dissimilar_pair = torch.clamp(self.margin-(distance-self.threshold),min=0.0)
        constractive_thresh_loss = torch.sum(
            (1 - label) * torch.pow(similar_pair, 2) + label * torch.pow(dissimilar_pair, 2)
        )
        return constractive_thresh_loss


class ConstractiveLoss(nn.Module):

    def __init__(self, margin=1.0, dist_flag='l2'):
        super(ConstractiveLoss, self).__init__()
        self.margin = margin
        self.dist_flag = dist_flag

    def various_distance(self, out_vec_t0, out_vec_t1):

        if self.dist_flag == 'l2':
            distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
        if self.dist_flag == 'l1':
            distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=1)
        if self.dist_flag == 'cos':
            similarity = F.cosine_similarity(out_vec_t0, out_vec_t1)
            distance = 1 - 2 * similarity / np.pi
        return distance

    def forward(self, out_vec_t0, out_vec_t1, label):

        # distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=2)
        distance = self.various_distance(out_vec_t0, out_vec_t1)
        # distance = 1 - F.cosine_similarity(out_vec_t0,out_vec_t1)
        constractive_loss = torch.sum(
            (1 - label) * torch.pow(distance, 2) + label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return constractive_loss


class ConstractiveMaskLoss(nn.Module):

    def __init__(self, thresh_flag=False, hinge_thresh=0.0, dist_flag='l2'):
        super(ConstractiveMaskLoss, self).__init__()

        if thresh_flag:
            self.sample_constractive_loss = ConstractiveThresholdHingeLoss(margin=2.0, hingethresh=hinge_thresh)
        else:
            self.sample_constractive_loss = ConstractiveLoss(margin=2.0, dist_flag=dist_flag)

    def forward(self, out_t0, out_t1, ground_truth):

        # out_t0 = out_t0.permute(0,2,3,1)
        n, c, h, w = out_t0.data.shape
        out_t0_rz = torch.transpose(out_t0.view(c, h * w), 1, 0)
        out_t1_rz = torch.transpose(out_t1.view(c, h * w), 1, 0)
        gt_tensor = torch.from_numpy(np.array(ground_truth.data.cpu().numpy(), np.float32))
        gt_rz = Variable(torch.transpose(gt_tensor.view(1, h * w), 1, 0)).cuda()
        # gt_rz = Variable(torch.transpose(ground_truth.view(1,h*w),1,0))
        loss = self.sample_constractive_loss(out_t0_rz, out_t1_rz, gt_rz)
        return loss


def CDTraining_ContrastiveLoss(model, optimizer, train_dataloader, savepath, margin,
                               batch_size, dev, epoch, firstloss):
    p = savepath

    Loss = []
    beforeloss = firstloss

    ckpt = {'model': model.state_dict()}

    pbar = tqdm(total=len(train_dataloader), desc="EPOCH {}".format(epoch))
    for _, data in enumerate(train_dataloader):
        model.train()
        x06, x12, y = data

        # CPU to GPU
        model = model.to(dev)
        x06 = x06.to(dev)
        x12 = x12.to(dev)
        y = y.to(dev)

        pred1 = model(x06)
        pred2 = model(x12)

        loss = DistanceLoss(pred1, pred2, y, margin) / batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Loss.append(loss.item())
        averageloss = sum(Loss) / len(Loss)

        if averageloss < beforeloss:
            torch.save(ckpt, p)
        beforeloss = averageloss

        pbar.set_postfix_str("AverageLoss={AveLoss:9.6f}".format(AveLoss=averageloss))
        pbar.update(1)
    pbar.close()

    return model, averageloss


def CrossValidation(model, test_dataloader, savedpath, margin, dev):
    Loss = []

    ckpt = torch.load(savedpath)
    model.load_state_dict(ckpt['model'])
    model.eval()

    for i, data in enumerate(test_dataloader):
        x06, x12, y = data

        # CPU to GPU
        model = model.to(dev)
        x06 = x06.to(dev)
        x12 = x12.to(dev)
        y = y.to(dev)

        pred1 = model(x06)
        pred2 = model(x12)

        loss = DistanceLoss(pred1, pred2, y, margin)

        Loss.append(loss.item())
        averageloss = sum(Loss) / len(Loss)

        if i == 30:
            break

    return averageloss


def DistanceLoss(x1, x2, y, margin):
    dist = F.pairwise_distance(x1, x2, p=2.0, keepdim=True)
    loss = torch.sum((1 - y) * torch.pow(dist, 2)
                     + y * torch.pow(torch.clamp(margin - dist, min=0.0), 2))
    return loss


def CD_Eval(x, y):
    x = Image2Bin(x)
    y = Image2Bin(y)

    tp, fp, tn, fn = PerformanceMeasure(x, y)

    precision = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + fn)
    fpr = float(fp) / float(fp + tn)
    mar = float(fn) / float(fp + fn)
    oa = float(tp + tn) / float(tp + fp + tn + fn)
    f1 = 2 * float(tp) / (2 * float(tp) + float(fp + fn))
    pc = float((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / float((tp + tn + fp + fn) ^ 2)
    pcc = float(tp + tn) / float(tp + tn + fp + fn)
    kappa = (oa - pc) / (1 - pc)

    return precision, recall, oa, kappa, f1


def PerformanceMeasure(x, y_actual):
    # true positive, false positive, true negative, false negative
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(x)):
        if y_actual[i] == x[i] == 255:
            tp += 1
        if x[i] == 255 and (y_actual[i] != x[i]):
            fp += 1
        if y_actual[i] == x[i] == 0:
            tn += 1
        if (x[i] == 0) and (y_actual[i] != x[i]):
            fn += 1

    return tp, fp, tn, fn


def Image2Bin(a):
    a = np.array(a)
    h, w = np.shape(a)

    for i in range(h):
        for j in range(w):
            if a[i][j] > 255*0.7:
                a[i][j] = 255
            else:
                a[i][j] = 0

    a = a.flatten()
    return a


def Image2Bin_for_dataloader(a):
    a = np.array(a)
    h, w = np.shape(a)

    for i in range(h):
        for j in range(w):
            if a[i][j] >= 1:
                a[i][j] = 255
            else:
                a[i][j] = 0
    a = Image.fromarray(a)
    return a


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp