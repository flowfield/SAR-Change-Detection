import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.init import xavier_normal_
from torch.nn import DataParallel

from DespecklingFunctions import get_n_params, train_c1, cross_validation_c1, \
    train_c2, cross_validation_c2, MSSSIM, SSIM, lee_filter, VanillaCNN8, VanillaCNN16


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == '__main__':
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(dev)
    model = VanillaCNN8()
    model.apply(weights_init)

    batch_size = 16
    epochs = 4
    eval_loss = 1.0

    #
    # loadpath =

    # VanillaCNN 8 Layers MSE + SSIM loss function
    loadpath = './pths/ID_VCNN8_L2MSSSIM_210928_ep15_div.pth'

    if (dev.type == 'cuda') and (torch.cuda.device_count() > 1):
        print('Multi GPU')
        ckpt = torch.load(loadpath)
        model = DataParallel(model, device_ids=list(range(2)))
        model.load_state_dict(ckpt['model'])
        param_num = get_n_params(model)
        print(param_num)

    optimizer = optim.Adam(model.parameters(), lr=5*1e-03, weight_decay=1e-04)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min',
                                                     patience=2,
                                                     verbose=True)
    criterion1 = torch.nn.L1Loss()
    criterion2 = MSSSIM()

    for epoch in range(epochs):
        # Training
        model = train_c2(model, optimizer, criterion1, criterion2, batch_size, dev, epoch)
        # Cross Validation
        eval_loss = cross_validation_c2(model, criterion1, criterion2, batch_size, savepath, dev, eval_loss, epoch)

        scheduler.step(eval_loss)
