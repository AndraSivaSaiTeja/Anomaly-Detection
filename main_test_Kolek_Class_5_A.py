# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import sys
import os
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation, visualization, test
from torch.nn import functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss

def train(_class_):
    print(_class_)
    epochs = 200
    learning_rate = 0.005
    batch_size = 16
    image_size = 256
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    #train_path = '/home/turing/Ranjeet/pytorch-cutpaste-master/Data/' + _class_ + '/train'
    test_path = '/home/turing/Ranjeet/Code_Anomaly_Detection/RD_Unified/Class_5_A/' + _class_
    ckp_path = './checkpoints/' + 'model.pth'
    #train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    #train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    checkpoint = torch.load(ckp_path)
    bn.load_state_dict(checkpoint['bn'])
    decoder.load_state_dict(checkpoint['decoder'])

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))
    bn.eval()
    decoder.eval()
    encoder.eval()
    pytorch_total_params = sum(p.numel() for p in encoder.parameters()) + sum(q.numel() for q in bn.parameters()) + sum(r.numel() for r in decoder.parameters())
    print("Total number of parameters=", pytorch_total_params)
    #sys.exit()
    auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device)
    print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
    torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, ckp_path)
    return auroc_px, auroc_sp, aupro_px




if __name__ == '__main__':

    setup_seed(111)
    #item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
    #             'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']
    item_list = ["object1"]
    for i in item_list:
        train(i)
