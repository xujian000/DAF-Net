# -*- coding: utf-8 -*-


import os
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.loss import Fusionloss, cc, infoNCE_loss
import kornia
from model.kernel_loss import kernelLoss
from utils.evaluator import average_similarity
from model.net import (
    Restormer_Encoder,
    Restormer_Decoder,
    BaseFeatureExtractor,
    DetailFeatureExtractor,
)
from utils.dataset import H5Dataset


"""
------------------------------------------------------------------------------
Environment Settings
------------------------------------------------------------------------------
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
GPU_number = os.environ["CUDA_VISIBLE_DEVICES"]


"""
------------------------------------------------------------------------------
Loss Function
------------------------------------------------------------------------------
"""
gaussianLoss = kernelLoss("gaussian")
laplaceLoss = kernelLoss("laplace")
MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
criteria_fusion = Fusionloss()
Loss_ssim = kornia.losses.SSIM(11, reduction="mean")

"""
------------------------------------------------------------------------------
Training HyperParameters
------------------------------------------------------------------------------
"""
batch_size = 8
num_epochs = 5 
windows_size = 11
lr = 1e-4
weight_decay = 0
clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5
"""
------------------------------------------------------------------------------
Loss Function Coefficient
------------------------------------------------------------------------------
"""
coeff_ssim = 5.0
coeff_mse = 1.0
coeff_tv = 5.0
coeff_decomp = 2.0
coeff_nice = 0.1
coeff_cc_basic = 2.0
coeff_gauss = 1.0
coeff_laplace = 1.0
"""
------------------------------------------------------------------------------
Save Format Settings
------------------------------------------------------------------------------
"""

result_name = f"mkmmd_batch{batch_size}_epoch{num_epochs}_WIN{windows_size}_cuda1"

"""
------------------------------------------------------------------------------
Build Model
------------------------------------------------------------------------------
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtractor(dim=64, num_heads=8)).to(
    device
)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtractor(num_layers=1)).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay
)

scheduler1 = torch.optim.lr_scheduler.StepLR(
    optimizer1, step_size=optim_step, gamma=optim_gamma
)
scheduler2 = torch.optim.lr_scheduler.StepLR(
    optimizer2, step_size=optim_step, gamma=optim_gamma
)
scheduler3 = torch.optim.lr_scheduler.StepLR(
    optimizer3, step_size=optim_step, gamma=optim_gamma
)
scheduler4 = torch.optim.lr_scheduler.StepLR(
    optimizer4, step_size=optim_step, gamma=optim_gamma
)

"""
------------------------------------------------------------------------------
DataSet and DataLoader
------------------------------------------------------------------------------
"""
trainloader = DataLoader(
    H5Dataset(r"data/dataSet4Training_imgsize_128_stride_200.h5"),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")


'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

torch.backends.cudnn.benchmark = True
prev_time = time.time()
Encoder.train()
Decoder.train()
BaseFuseLayer.train()
DetailFuseLayer.train()
for epoch in range(num_epochs):
    ''' train '''
    for i, (img_VI, img_IR) in enumerate(loader['train']):

        # Phase I
        img_VI, img_IR = img_VI.cuda(), img_IR.cuda()

        Encoder.zero_grad()
        Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        feature_V_B, feature_V_D, _ = Encoder(img_VI)
        feature_I_B, feature_I_D, _ = Encoder(img_IR)
        data_VI_hat, _ = Decoder(img_VI, feature_V_B, feature_V_D)
        data_IR_hat, _ = Decoder(img_IR, feature_I_B, feature_I_D)

        cc_loss_B = cc(feature_V_B, feature_I_B)
        cc_loss_D = cc(feature_V_D, feature_I_D)

        ssim_loss = coeff_ssim * (
            Loss_ssim(img_IR, data_IR_hat) + Loss_ssim(img_VI, data_VI_hat)
        )

        mse_loss = coeff_mse * (
            MSELoss(img_VI, data_VI_hat) + MSELoss(img_IR, data_IR_hat)
        )

        tv_loss = coeff_tv * (
            L1Loss(
                kornia.filters.SpatialGradient()(img_VI),
                kornia.filters.SpatialGradient()(data_VI_hat),
            )
            + L1Loss(
                kornia.filters.SpatialGradient()(img_IR),
                kornia.filters.SpatialGradient()(data_IR_hat),
            )
        )

        cc_loss = coeff_decomp * (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

        laplace_loss = coeff_laplace * laplaceLoss(feature_V_B, feature_I_B)
        gauss_loss = coeff_gauss * gaussianLoss(feature_V_B, feature_I_B)
        ince_loss = coeff_nice * infoNCE_loss(feature_V_B, feature_I_B)
        basic_cc_loss = coeff_cc_basic * cc_loss_B

        mmd_loss = laplace_loss + gauss_loss + basic_cc_loss + ince_loss

        loss1 = ssim_loss + mse_loss + cc_loss + tv_loss + mmd_loss

        similarity_cos = average_similarity(feature_V_B, feature_I_B, "cosine")
        similarity_pearson = average_similarity(feature_V_B, feature_I_B, "pearson")
        distance_euclidean = average_similarity(feature_V_B, feature_I_B, "euclidean")

        loss1.backward()

        # Phase II

        nn.utils.clip_grad_norm_(
            Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        optimizer1.step()
        optimizer2.step()

        feature_V_B, feature_V_D, feature_V = Encoder(img_VI)
        feature_I_B, feature_I_D, feature_I = Encoder(img_IR)
        feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
        feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)
        data_Fuse, feature_F = Decoder(img_VI, feature_F_B, feature_F_D)

        cc_loss_B = cc(feature_V_B, feature_I_B)
        cc_loss_D = cc(feature_V_D, feature_I_D)
        cc_loss =   coeff_decomp *((cc_loss_D) ** 2 / (1.01 + cc_loss_B))  
        fusionloss, _, _ = criteria_fusion(img_VI, img_IR, data_Fuse)

        loss2 = fusionloss + cc_loss

        loss2.backward()
        nn.utils.clip_grad_norm_(
            Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()

        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        print(
            f"[E:{epoch}/{num_epochs}][B:{i}/{len(loader['train'])}][L1:{loss1.item():.2f},mse:{mse_loss.item():.2f},cc:{cc_loss.item():.2f},tv:{tv_loss.item():.2f},mmd:{mmd_loss.item():.2f},lap:{laplace_loss.item():.2f},gauss:{gauss_loss.item():.2f},ince:{ince_loss.item():.2f},ccb:{basic_cc_loss.item():.2f}][L2:{loss2.item():.2f},f:{fusionloss.item():.2f},cc:{cc_loss.item():.2f}][{similarity_cos:.2f},{similarity_pearson:.2f},{distance_euclidean:.2f}]"
        )

    save_path = os.path.join(f"checkPoints/{result_name}_{epoch}.pth")
    checkpoint = {
        'Encoder': Encoder.state_dict(),
        'Decoder': Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    torch.save(checkpoint, save_path)

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6
