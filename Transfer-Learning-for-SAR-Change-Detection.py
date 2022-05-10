import torch
import torch.optim as optim
from torch.nn import DataParallel

from DespecklingFunctions import get_n_params, VanillaCNN8
from ChangeDetectionFunctions import NestedUNet_VGGv2
from TransferLearningFunctions import Total_V8div_UNetV2, SpeckledCDcv, SpeckledCDtrain


if __name__ == '__main__':
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(dev)

    model = Total_V8div_UNetV2()
    if (dev.type == 'cuda') and (torch.cuda.device_count() > 1):
        print('Multi GPU')
        model = DataParallel(model, device_ids=list(range(2)))

        print(get_n_params(model))
    model_dict = model.state_dict()

    # Despeckling Model
    loadpath1 = 'C:/Users/user/PycharmProjects/Despeckling/pths/ID_VCNN8_L2MSSSIM_210928_ep15_div.pth'
    model1 = VanillaCNN8()
    if (dev.type == 'cuda') and (torch.cuda.device_count() > 1):
        print('Multi GPU_IDCNN')
        ckpt1 = torch.load(loadpath1)
        model1 = DataParallel(model1, device_ids=list(range(2)))
        model1.load_state_dict(ckpt1['model'], strict=False)
    pretrained_dict1 = model1.state_dict()
    pretrained_dict1 = {k1: v1 for k1, v1 in pretrained_dict1.items() if k1 in model_dict}

    # Change Detection Model
    loadpath2 = 'C:/Users/user/PycharmProjects/ChangeDectection_py/pths/CD_NUNetVGGv2_PAM_210909_ep50_m20.pth'
    ckpt2 = torch.load(loadpath2)
    model2 = NestedUNet_VGGv2()
    if (dev.type == 'cuda') and (torch.cuda.device_count() > 1):
        print('Multi GPU_CDCNN')
        ckpt2 = torch.load(loadpath2)
        model2 = DataParallel(model2, device_ids=list(range(2)))
        model2.load_state_dict(ckpt2['model'], strict=False)
    pretrained_dict2 = model2.state_dict()
    pretrained_dict2 = {k2: v2 for k2, v2 in pretrained_dict2.items() if k2 in model_dict}

    print('Speckled Model')
    print(pretrained_dict1.items())
    print('CD Model')
    print(pretrained_dict2.items())

    model_dict.update(pretrained_dict1)
    model.load_state_dict(model_dict)
    model_dict.update(pretrained_dict2)
    model.load_state_dict(model_dict)
    print('Total Model')
    print(model.state_dict().items())

    batch_size = 2
    epochs = 25
    eval_loss = 100000.0
    margin = 2.0

    savepath = './pths/ICD_v8l2ms_nunetv2m20_211006_ep25.pth'

    optimizer = optim.Adam(model.parameters(), lr=1e-07, weight_decay=1e-04)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min',
                                                     patience=2,
                                                     verbose=True)

    for epoch in range(epochs):
        # Training
        model = SpeckledCDtrain(model, optimizer, margin, batch_size, dev, epoch)
        # Cross Validation
        eval_loss = SpeckledCDcv(model, batch_size, savepath, margin, dev, epoch, eval_loss)

        scheduler.step(eval_loss)