import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d

from ChangeDetectionFunctions import Image2Bin_for_dataloader, DistanceLoss, VGGBlock

class Total_V8div_UNetV2(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, **kwargs):
        super(Total_V8div_UNetV2, self).__init__()
        # Start Despeckling CNN
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

class SpeckledCDdata(Dataset):
    def __init__(self, train='train', transform=None):
        super(SpeckledCDdata, self).__init__()

        self.transform = transform
        a_path, b_path, l_path = [], [], []

        if train == 'train':
            base_path = 'C:/Data/SpeckledCD/train/'
        elif train == 'eval':
            base_path = 'C:/Data/SpeckledCD/cv/'
        elif train == 'test':
            base_path = 'C:/Data/SpeckledCD/test/'
        else:
            print('train, cross validation, or test')

        path_a = base_path + 'A/'
        path_b = base_path + 'B/'
        path_l = base_path + 'label/'

        folderpaths_a = [os.path.join(path_a, file_name) for file_name in os.listdir(path_a)]
        folderpaths_b = [os.path.join(path_b, file_name) for file_name in os.listdir(path_b)]
        folderpaths_l = [os.path.join(path_l, file_name) for file_name in os.listdir(path_l)]

        a_path.extend(folderpaths_a)
        b_path.extend(folderpaths_b)
        l_path.extend(folderpaths_l)

        self.A = a_path
        self.B = b_path
        self.L = l_path

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        xA = self.A[idx]
        xB = self.B[idx]
        label = self.L[idx]

        xa = Image.open(xA)
        xb = Image.open(xB)
        y = Image.open(label)
        y = Image2Bin_for_dataloader(y)

        if self.transform is not None:
            xa = self.transform(xa)
            xb = self.transform(xb)
            y = self.transform(y)

        return xa, xb, y


def dataloader_train(bs):
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    train_dataset = SpeckledCDdata(train='train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=6)

    return train_dataloader


def dataloader_eval(bs):
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    test_dataset = SpeckledCDdata(train='eval', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=6)

    return test_dataloader


def dataloader_test():
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    test_dataset = SpeckledCDdata(train='test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=6)

    return test_dataloader


def SpeckledCDtrain(model, optimizer, margin, batch_size, dev, epoch):
    model.train()
    Loss = []
    train_loader = dataloader_train(batch_size)

    pbar = tqdm(total=len(train_loader), desc="EPOCH {}".format(epoch))
    for _, data in enumerate(train_loader):
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

        pbar.set_postfix_str("AverageLoss={AveLoss:9.6f}".format(AveLoss=averageloss))
        pbar.update(1)
    pbar.close()

    return model


def SpeckledCDcv(model, batch_size, savepath, margin, dev, epoch, before_loss):
    model.eval()
    Loss = []

    crossval_loader = dataloader_eval(batch_size)

    ckpt = torch.load(savepath)
    model.load_state_dict(ckpt['model'])

    with torch.no_grad():
        pbar = tqdm(total=len(crossval_loader), desc="EPOCH {}".format(epoch))
        for _, data in enumerate(crossval_loader):
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

            pbar.set_postfix_str("CrossValidationLoss={AveLoss:9.8f}".format(AveLoss=averageloss))
            pbar.update(1)
        pbar.close()

        if before_loss >= averageloss:
            torch.save(ckpt, savepath)
            print('Model Saved!')
            before_loss = averageloss
    print(f'Current Before Loss:{before_loss:9.8f}\n')

    return averageloss
