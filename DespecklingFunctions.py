import os
import math
from numpy.random import exponential
import cv2
from tqdm import tqdm
from PIL import Image
from scipy import ndimage
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from math import exp
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance


class VanillaCNN8(nn.Module):
    def __init__(self):
        super(VanillaCNN8, self).__init__()
        # kernel 사이즈 조절해서 maxpooling 거꾸로 과정할 것
        self.pad1 = nn.ReflectionPad2d(2)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(5, 5))
        self.act1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.2)

        self.pad2 = nn.ReflectionPad2d(2)
        self.conv2 = nn.Conv2d(1, 64, kernel_size=(5, 5))
        self.BN2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.2)

        self.pad3 = nn.ReflectionPad2d(2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.BN3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()
        self.do3 = nn.Dropout(p=0.2)

        self.pad4 = nn.ReflectionPad2d(2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.BN4 = nn.BatchNorm2d(64)
        self.act4 = nn.ReLU()
        self.do4 = nn.Dropout(p=0.2)

        self.pad5 = nn.ReflectionPad2d(2)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.BN5 = nn.BatchNorm2d(64)
        self.act5 = nn.ReLU()
        self.do5 = nn.Dropout(p=0.2)

        self.pad6 = nn.ReflectionPad2d(2)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.BN6 = nn.BatchNorm2d(64)
        self.act6 = nn.ReLU()
        self.do6 = nn.Dropout(p=0.2)

        self.pad7 = nn.ReflectionPad2d(2)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.BN7 = nn.BatchNorm2d(64)
        self.act7 = nn.ReLU()
        self.do7 = nn.Dropout(p=0.2)

        self.pad8 = nn.ReflectionPad2d(2)
        self.conv8 = nn.Conv2d(64, 1, kernel_size=(5, 5))
        self.act8 = nn.ReLU()

        self.actf = nn.Tanh()

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

        return out


class VanillaCNN16(nn.Module):
    def __init__(self):
        super(VanillaCNN16, self).__init__()
        # kernel 사이즈 조절해서 maxpooling 거꾸로 과정할 것
        self.pad1 = nn.ReflectionPad2d(2)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(5, 5))
        self.act1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.2)

        self.pad2 = nn.ReflectionPad2d(2)
        self.conv2 = nn.Conv2d(1, 16, kernel_size=(5, 5))
        self.BN2 = nn.BatchNorm2d(16)
        self.act2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.2)

        self.pad3 = nn.ReflectionPad2d(2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(5, 5))
        self.BN3 = nn.BatchNorm2d(32)
        self.act3 = nn.ReLU()
        self.do3 = nn.Dropout(p=0.2)

        self.pad4 = nn.ReflectionPad2d(2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        self.BN4 = nn.BatchNorm2d(64)
        self.act4 = nn.ReLU()
        self.do4 = nn.Dropout(p=0.2)

        self.pad5 = nn.ReflectionPad2d(2)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.BN5 = nn.BatchNorm2d(64)
        self.act5 = nn.ReLU()
        self.do5 = nn.Dropout(p=0.2)

        self.pad6 = nn.ReflectionPad2d(1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.BN6 = nn.BatchNorm2d(64)
        self.act6 = nn.ReLU()
        self.do6 = nn.Dropout(p=0.2)

        self.pad7 = nn.ReflectionPad2d(1)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.BN7 = nn.BatchNorm2d(64)
        self.act7 = nn.ReLU()
        self.do7 = nn.Dropout(p=0.2)

        self.pad8 = nn.ReflectionPad2d(1)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.BN8 = nn.BatchNorm2d(64)
        self.act8 = nn.ReLU()
        self.do8 = nn.Dropout(p=0.2)

        self.pad9 = nn.ReflectionPad2d(1)
        self.conv9 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.BN9 = nn.BatchNorm2d(64)
        self.act9 = nn.ReLU()
        self.do9 = nn.Dropout(p=0.2)

        self.pad10 = nn.ReflectionPad2d(1)
        self.conv10 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.BN10 = nn.BatchNorm2d(64)
        self.act10 = nn.ReLU()
        self.do10 = nn.Dropout(p=0.2)

        self.pad11 = nn.ReflectionPad2d(1)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.BN11 = nn.BatchNorm2d(64)
        self.act11 = nn.ReLU()
        self.do11 = nn.Dropout(p=0.2)

        self.pad12 = nn.ReflectionPad2d(2)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.BN12 = nn.BatchNorm2d(64)
        self.act12 = nn.ReLU()
        self.do12 = nn.Dropout(p=0.2)

        self.pad13 = nn.ReflectionPad2d(2)
        self.conv13 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.BN13 = nn.BatchNorm2d(64)
        self.act13 = nn.ReLU()
        self.do13 = nn.Dropout(p=0.2)

        self.pad14 = nn.ReflectionPad2d(2)
        self.conv14 = nn.Conv2d(64, 32, kernel_size=(5, 5))
        self.BN14 = nn.BatchNorm2d(32)
        self.act14 = nn.ReLU()
        self.do14 = nn.Dropout(p=0.2)

        self.pad15 = nn.ReflectionPad2d(2)
        self.conv15 = nn.Conv2d(32, 16, kernel_size=(5, 5))
        self.BN15 = nn.BatchNorm2d(16)
        self.act15 = nn.ReLU()
        self.do15 = nn.Dropout(p=0.2)

        self.pad16 = nn.ReflectionPad2d(2)
        self.conv16 = nn.Conv2d(16, 1, kernel_size=(5, 5))
        self.act16 = nn.ReLU()

        self.actf = nn.Tanh()

    def forward(self, x):
        x_in = x
        # Layer 1
        L1_1 = self.conv1(self.pad1(x_in))
        x1 = self.do1(self.act1(L1_1))

        # Layer 2~7
        L2_1 = self.conv2(self.pad2(x1))
        L2_2 = self.BN2(L2_1)
        x2 = self.do2(self.act2(L2_2))

        L3_1 = self.conv3(self.pad3(x2))
        L3_2 = self.BN3(L3_1)
        x3 = self.do3(self.act3(L3_2))

        L4_1 = self.conv4(self.pad4(x3))
        L4_2 = self.BN4(L4_1)
        x4 = self.do4(self.act4(L4_2))

        L5_1 = self.conv5(self.pad5(x4))
        L5_2 = self.BN5(L5_1)
        x5 = self.do5(self.act5(L5_2))

        L6_1 = self.conv6(self.pad6(x5))
        L6_2 = self.BN6(L6_1)
        x6 = self.do6(self.act6(L6_2))

        L7_1 = self.conv7(self.pad7(x6))
        L7_2 = self.BN7(L7_1)
        x7 = self.do7(self.act7(L7_2))

        L8_1 = self.conv8(self.pad8(x7))
        L8_2 = self.BN8(L8_1)
        x8 = self.do8(self.act8(L8_2))

        L9_1 = self.conv9(self.pad9(x8))
        L9_2 = self.BN9(L9_1)
        x9 = self.do9(self.act9(L9_2))

        L10_1 = self.conv10(self.pad10(x9))
        L10_2 = self.BN10(L10_1)
        x10 = self.do10(self.act10(L10_2))

        L11_1 = self.conv11(self.pad11(x10))
        L11_2 = self.BN11(L11_1)
        x11 = self.do11(self.act11(L11_2))

        L12_1 = self.conv12(self.pad12(x11))
        L12_2 = self.BN12(L12_1)
        x12 = self.do12(self.act12(L12_2))

        L13_1 = self.conv13(self.pad13(x12))
        L13_2 = self.BN13(L13_1)
        x13 = self.do13(self.act13(L13_2))

        L14_1 = self.conv14(self.pad14(x13))
        L14_2 = self.BN14(L14_1)
        x14 = self.do14(self.act14(L14_2))

        L15_1 = self.conv15(self.pad15(x14))
        L15_2 = self.BN15(L15_1)
        x15 = self.do15(self.act15(L15_2))

        L16_1 = self.conv16(self.pad16(x15))
        out = torch.div(x_in, self.act8(L16_1) + 1e-10)
        out = self.actf(out)

        return out


def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        cs = cs.mean()
        ret = ssim_map.mean()
    else:
        cs = cs.mean(1).mean(1).mean(1)
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


def filename2num(path):
    file_names = os.listdir(path)
    i = 1
    for name in file_names:
        src = os.path.join(path, name)
        dst = str(i) + '.tiff'
        dst = os.path.join(path, dst)
        os.rename(src, dst)
        i += 1


def sar_look(img):
    var = ndimage.variance(img)
    mean = img.mean()

    look = mean / var
    return look


def enl(img):
    mu = torch.mean(img)
    var = torch.var(img)

    ENL = (mu ** 2) / var
    return ENL


def psnr(img1, img2):
    # img1 = torch.Tensor(img1)
    # img2 = torch.Tensor(img2)

    mse = torch.mean((img1 - img2) ** 2)
    return -10 * torch.log10(mse)


def histogram_equalization(path, save_path):
    imagePaths1 = [os.path.join(path, file_name) for file_name in os.listdir(path)]
    i = 0

    for imagepath1 in imagePaths1:
        img = cv2.imread(imagepath1, 0)

        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        img = clahe.apply(img)

        Image.fromarray(img).save(save_path + str(i) + '.tiff', 'TIFF')

        i += 1


def image_sizing(directory, save_dir, wanted_size):
    imagePaths1 = [os.path.join(directory, file_name) for file_name in os.listdir(directory)]
    i = 0

    for imagePath in imagePaths1:
        img = cv2.imread(imagePath, 0)

        resize_img = cv2.resize(img, wanted_size)

        Image.fromarray(resize_img).save(save_dir + str(i) + '.tiff')
        i = i + 1


def image_cropping(dir_in, num_img, dir_out):
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    transform2 = transforms.ToPILImage()

    crop_size = (256, 256)
    it = 0
    for num in range(num_img):
        dir = dir_in
        img = Image.open(dir)
        img = transform(img).squeeze(0)

        img_size = img.size()
        print(img_size)

        iter_num = (math.ceil(img_size[0] / crop_size[0]), math.ceil(img_size[1] / crop_size[1]))

        start_pt0 = np.linspace(0, img_size[0] - crop_size[0], iter_num[0])
        start_pt1 = np.linspace(0, img_size[1] - crop_size[1], iter_num[1])

        for i in range(iter_num[0]):
            for j in range(iter_num[1]):
                crop_image = transform2(img).convert('L')
                crop_image = crop_image.crop((start_pt0[i], start_pt1[j],
                                              start_pt0[i] + crop_size[0],
                                              start_pt1[j] + crop_size[1]))
                crop_image.save(dir_out + str(it) + '.tiff', 'TIFF')
                it += 1


def create_speckle(crop_size):
    speckle = exponential(scale=1, size=crop_size)

    return speckle


def image2bin(a):
    a = np.array(a)
    h, w = np.shape(a)

    for i in range(h):
        for j in range(w):
            if a[i][j] > 255*0.6:
                a[i][j] = 255
            else:
                a[i][j] = 0

    a = a.flatten()
    return a


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class DespecklingData(Dataset):
    def __init__(self, train='train', transform=None):
        super(DespecklingData, self).__init__()

        self.transform = transform
        a_path, l_path = [], []

        if train == 'train':
            base_path = 'C:/Data/IDdata/PAM_speckled/train/'
        elif train == 'eval':
            base_path = 'C:/Data/IDdata/PAM_speckled/cv/'
        elif train == 'test':
            base_path = 'C:/Data/IDdata/PAM_speckled/test/'
        else:
            print('train, cross validation, or test')

        path_a = base_path + 'A/'
        path_l = base_path + 'label/'

        folderpaths_a = [os.path.join(path_a, file_name) for file_name in os.listdir(path_a)]
        folderpaths_l = [os.path.join(path_l, file_name) for file_name in os.listdir(path_l)]

        a_path.extend(folderpaths_a)
        l_path.extend(folderpaths_l)

        self.A = a_path
        self.L = l_path

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        train = self.A[idx]
        label = self.L[idx]

        xa = Image.open(train)
        y = Image.open(label)

        if self.transform is not None:
            xa = self.transform(xa)
            y = self.transform(y)

        return xa, y


class DespecklingDataCOWC(Dataset):
        def __init__(self, train='train', transform=None):
            super(DespecklingDataCOWC, self).__init__()

            self.transform = transform
            a_path, l_path = [], []

            if train == 'train':
                base_path = 'C:/Data/IDdata_new/train/'
            elif train == 'eval':
                base_path = 'C:/Data/IDdata_new/cv/'
            elif train == 'test':
                base_path = 'C:/Data/IDdata_new/test/'
            else:
                print('train, cross validation, or test')

            path_a = base_path + 'A/'
            path_l = base_path + 'label/'

            folderpaths_a = [os.path.join(path_a, file_name) for file_name in os.listdir(path_a)]
            folderpaths_l = [os.path.join(path_l, file_name) for file_name in os.listdir(path_l)]

            a_path.extend(folderpaths_a)
            l_path.extend(folderpaths_l)

            self.A = a_path
            self.L = l_path

        def __len__(self):
            return len(self.A)

        def __getitem__(self, idx):
            train = self.A[idx]
            label = self.L[idx]

            xa = Image.open(train)
            y = Image.open(label)

            if self.transform is not None:
                xa = self.transform(xa)
                y = self.transform(y)

            return xa, y


class DespecklingDataIQA(Dataset):
    def __init__(self, transform=None):
        super(DespecklingDataIQA, self).__init__()

        self.transform = transform
        a_path = []
        path_a = 'C:/Users/user/Desktop/paper_chapter4/crop/'

        folderpaths_a = [os.path.join(path_a, file_name) for file_name in os.listdir(path_a)]
        a_path.extend(folderpaths_a)

        self.A = a_path

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        train = self.A[idx]

        xa = Image.open(train)

        if self.transform is not None:
            xa = self.transform(xa)

        return xa


def dataloader_iqa():
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    train_dataset = DespecklingDataIQA(transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4)

    return train_dataloader


def dataloader_train(bs):
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    train_dataset = DespecklingDataCOWC(train='train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=10)

    return train_dataloader


def dataloader_eval(bs):
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    test_dataset = DespecklingDataCOWC(train='eval', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=10)

    return test_dataloader


def dataloader_test():
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    test_dataset = DespecklingDataCOWC(train='test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)

    return test_dataloader


def train_c1(model, optimizer, criterion, batch_size, dev, epoch):
    model.train()
    Loss = []
    train_loader = dataloader_train(batch_size)

    # Start Training
    pbar = tqdm(total=len(train_loader), desc="EPOCH {}".format(epoch))
    for i, data in enumerate(train_loader):
        x, y = data

        # CPU to GPU
        model = model.to(dev)
        x = x.to(dev)
        y = y.to(dev)

        optimizer.zero_grad()
        pred = model(x)

        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        Loss.append(loss.item())
        averageloss = sum(Loss) / len(Loss)

        pbar.set_postfix_str("AverageLoss={AveLoss:9.8f}".format(AveLoss=averageloss))
        pbar.update(1)
    pbar.close()

    return model


def cross_validation_c1(model, criterion, batch_size, savepath, dev, before_loss, epoch):
    model.eval()
    crossval_loader = dataloader_eval(batch_size)
    ckpt = {'model': model.state_dict()}
    Lossv = []

    with torch.no_grad():
        pbar = tqdm(total=len(crossval_loader), desc="EPOCH {}".format(epoch))
        for _, data in enumerate(crossval_loader):
            xv, yv = data

            model = model.to(dev)
            xv = xv.to(dev)
            yv = yv.to(dev)

            predv = model(xv)

            lossv = criterion(predv, yv)

            Lossv.append(lossv.item())
            average_lossv = sum(Lossv) / len(Lossv)

            pbar.set_postfix_str("CrossValidationLoss={AveLoss:9.8f}".format(AveLoss=average_lossv))
            pbar.update(1)
        pbar.close()

        if before_loss >= average_lossv:
            torch.save(ckpt, savepath)
            print('Model Saved!')
            before_loss = average_lossv
    print(f'Current Before Loss:{before_loss:9.8f}\n')
    return before_loss


def train_c2(model, optimizer, criterion1, criterion2, batch_size, dev, epoch):
    model.train()
    Loss = []
    train_loader = dataloader_train(batch_size)

    # Start Training
    with torch.set_grad_enabled(True):

        pbar = tqdm(total=len(train_loader), desc="EPOCH {}".format(epoch))
        for i, data in enumerate(train_loader):
            x, y = data

            # CPU to GPU
            model = model.to(dev)
            x = x.to(dev)
            y = y.to(dev)

            pred = model(x)

            loss = (1.0 - 0.025) * criterion1(pred, y) + 0.025 * (1 - criterion2(pred, y))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            Loss.append(loss.item())
            averageloss = sum(Loss) / len(Loss)

            pbar.set_postfix_str("AverageLoss={AveLoss:9.8f}".format(AveLoss=averageloss))
            pbar.update(1)
        pbar.close()

    return model


def cross_validation_c2(model, criterion1, criterion2, batch_size, savepath, dev, before_loss, epoch):
    model.eval()
    crossval_loader = dataloader_eval(batch_size)
    ckpt = {'model': model.state_dict()}
    Lossv = []

    with torch.no_grad():
        pbar = tqdm(total=len(crossval_loader), desc="EPOCH {}".format(epoch))
        for _, data in enumerate(crossval_loader):
            xv, yv = data

            model = model.to(dev)
            xv = xv.to(dev)
            yv = yv.to(dev)

            predv = model(xv)

            lossv = (1 - 0.025) * criterion1(predv, yv) + 0.025 * (1 - criterion2(predv, yv))

            Lossv.append(lossv.item())
            average_lossv = sum(Lossv) / len(Lossv)

            pbar.set_postfix_str("CrossValidationLoss={AveLoss:9.8f}".format(AveLoss=average_lossv))
            pbar.update(1)
        pbar.close()

        if before_loss >= average_lossv:
            torch.save(ckpt, savepath)
            print('Model Saved!')
            before_loss = average_lossv

    print(f'Current Before Loss:{before_loss:9.8f}\n')

    return before_loss