import torch
import torch.optim as optim
from torch.nn import DataParallel

from ChangeDetectionFunctions import CDTraining_ContrastiveLoss, CrossValidation, get_n_params, \
    NestedUNet_VGGv2, NestedUNet_VGG, UNet_VGG, PAMdataloader_train, PAMdataloader_eval

if __name__ == '__main__':
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(dev)

    # NestedUNet_VGGv2 until epoch 25
    loadpath = './pths/CD_UNetVGG_PAM_211108_ep10_m20.pth'

    # NestedUNet_VGGv1 until epoch 50, margin 2.0
    savepath = './pths/CD_UNetVGG_PAM_211109_ep10_m20.pth'

    ckpt = torch.load(loadpath)
    CDmodel = UNet_VGG(num_classes=1)
    CDmodel.load_state_dict(ckpt['model'])

    param_num = get_n_params(CDmodel)
    print(param_num)

    loss = 7000
    batch_size = 4
    epochs = 10
    lossmargin = 2.0

    train_dataloader = PAMdataloader_train(batch_size)
    test_dataloader = PAMdataloader_eval()

    optimizer = optim.Adam(CDmodel.parameters(), lr=1e-06)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min',
                                                     patience=3,
                                                     verbose=True)

    for epoch in range(epochs):
        CDmodel, loss = CDTraining_ContrastiveLoss(CDmodel, optimizer, train_dataloader,
                                                   savepath, lossmargin, batch_size,
                                                   dev=dev, epoch=epoch, firstloss=loss)
        CVloss = CrossValidation(CDmodel, test_dataloader, savepath, lossmargin, dev=dev)

        scheduler.step(CVloss)
