# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
from collections import OrderedDict
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
    
    train_path = 'Data/' + '/train'
    test_path = '/home/turing/Ranjeet/pytorch-cutpaste-master/Data/' + _class_
    ckp_path = './checkpoints/' + 'model.pth'
    
    train_data = ImageFolder(root=train_path, transform=data_transform)
    
    #print("train_data=", train_data)
    #sys.exit()
    
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = resnet18(pretrained=False)
    encoder = encoder.to(device)
    #print("encoder=", encoder)
    #sys.exit()
    
    modelname = '/home/turing/Ranjeet/Code_Anomaly_Detection/Siva/Unified_Code/models/model-2023-03-28_15_46_54.tch'
    weights = torch.load(modelname)
    
    weights_keys = list(weights.keys())
    rem_list = ["resnet18.layer1.0.bn1.num_batches_tracked",\
            "resnet18.layer1.0.bn2.num_batches_tracked",\
            "resnet18.layer1.1.bn1.num_batches_tracked",\
            "resnet18.layer1.1.bn2.num_batches_tracked",\
            "resnet18.layer2.0.bn1.num_batches_tracked",\
            "resnet18.layer2.0.bn2.num_batches_tracked",\
            "resnet18.layer2.1.bn1.num_batches_tracked",\
            "resnet18.layer2.1.bn2.num_batches_tracked",\
            "resnet18.layer3.0.bn1.num_batches_tracked",\
            "resnet18.layer3.0.bn2.num_batches_tracked",\
            "resnet18.layer3.1.bn1.num_batches_tracked",\
            "resnet18.layer3.1.bn2.num_batches_tracked",\
            "resnet18.layer4.0.bn1.num_batches_tracked",\
            "resnet18.layer4.0.bn2.num_batches_tracked",\
            "resnet18.layer4.1.bn1.num_batches_tracked",\
            "resnet18.layer4.1.bn2.num_batches_tracked",\
            "resnet18.bn1.num_batches_tracked",\
            "resnet18.layer2.0.downsample.1.num_batches_tracked",\
            "resnet18.layer3.0.downsample.1.num_batches_tracked",\
            "resnet18.layer4.0.downsample.1.num_batches_tracked",\
            "head.0.weight","head.0.bias", "head.1.weight", "head.1.bias", "head.1.running_mean", "head.1.running_var", "head.1.num_batches_tracked", "head.3.weight", "head.3.bias", "head.4.weight", "head.4.bias", "head.4.running_mean", "head.4.running_var", "head.4.num_batches_tracked", "out.weight", "out.bias", "head.6.weight", "head.6.bias", "head.7.weight", "head.7.bias", "head.7.running_mean", "head.7.running_var", "head.7.num_batches_tracked"]
			
            
    [weights.pop(key) for key in rem_list]
    
    new_keys = ["conv1.weight", "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var", "layer1.0.conv1.weight", "layer1.0.bn1.weight", "layer1.0.bn1.bias", "layer1.0.bn1.running_mean", "layer1.0.bn1.running_var", "layer1.0.conv2.weight", "layer1.0.bn2.weight", "layer1.0.bn2.bias", "layer1.0.bn2.running_mean", "layer1.0.bn2.running_var", "layer1.1.conv1.weight", "layer1.1.bn1.weight", "layer1.1.bn1.bias", "layer1.1.bn1.running_mean", "layer1.1.bn1.running_var", "layer1.1.conv2.weight", "layer1.1.bn2.weight", "layer1.1.bn2.bias", "layer1.1.bn2.running_mean", "layer1.1.bn2.running_var", "layer2.0.conv1.weight", "layer2.0.bn1.weight", "layer2.0.bn1.bias", "layer2.0.bn1.running_mean", "layer2.0.bn1.running_var", "layer2.0.conv2.weight", "layer2.0.bn2.weight", "layer2.0.bn2.bias", "layer2.0.bn2.running_mean", "layer2.0.bn2.running_var", "layer2.0.downsample.0.weight", "layer2.0.downsample.1.weight", "layer2.0.downsample.1.bias", "layer2.0.downsample.1.running_mean", "layer2.0.downsample.1.running_var", "layer2.1.conv1.weight", "layer2.1.bn1.weight", "layer2.1.bn1.bias", "layer2.1.bn1.running_mean", "layer2.1.bn1.running_var", "layer2.1.conv2.weight", "layer2.1.bn2.weight", "layer2.1.bn2.bias", "layer2.1.bn2.running_mean", "layer2.1.bn2.running_var", "layer3.0.conv1.weight", "layer3.0.bn1.weight", "layer3.0.bn1.bias", "layer3.0.bn1.running_mean", "layer3.0.bn1.running_var", "layer3.0.conv2.weight", "layer3.0.bn2.weight", "layer3.0.bn2.bias", "layer3.0.bn2.running_mean", "layer3.0.bn2.running_var", "layer3.0.downsample.0.weight", "layer3.0.downsample.1.weight", "layer3.0.downsample.1.bias", "layer3.0.downsample.1.running_mean", "layer3.0.downsample.1.running_var", "layer3.1.conv1.weight", "layer3.1.bn1.weight", "layer3.1.bn1.bias", "layer3.1.bn1.running_mean", "layer3.1.bn1.running_var", "layer3.1.conv2.weight", "layer3.1.bn2.weight", "layer3.1.bn2.bias", "layer3.1.bn2.running_mean", "layer3.1.bn2.running_var", "layer4.0.conv1.weight", "layer4.0.bn1.weight", "layer4.0.bn1.bias", "layer4.0.bn1.running_mean", "layer4.0.bn1.running_var", "layer4.0.conv2.weight", "layer4.0.bn2.weight", "layer4.0.bn2.bias", "layer4.0.bn2.running_mean", "layer4.0.bn2.running_var", "layer4.0.downsample.0.weight", "layer4.0.downsample.1.weight", "layer4.0.downsample.1.bias", "layer4.0.downsample.1.running_mean", "layer4.0.downsample.1.running_var", "layer4.1.conv1.weight", "layer4.1.bn1.weight", "layer4.1.bn1.bias", "layer4.1.bn1.running_mean", "layer4.1.bn1.running_var", "layer4.1.conv2.weight", "layer4.1.bn2.weight", "layer4.1.bn2.bias", "layer4.1.bn2.running_mean", "layer4.1.bn2.running_var"]
    
    #print("new2", len(new2))
                  
    weights2 = OrderedDict([('a',1)])
    i=0
    #print("new_keys=", len(weights2))
    for key in weights.keys():
    	#print("key=", key)
    	temp = weights[key]
    	#print("temp=", temp)
    	#print("new_keys[i]=", new_keys[i])
    	weights2[new_keys[i]] = temp
    	i = i + 1
    	#if i==99:
    	#	break
    	#print("i=", i)
    
    weights2.pop('a')
    encoder.load_state_dict(weights2)
    #a = torch.Tensor(list(model.parameters())[0])
    #b = torch.Tensor((weights['resnet18.conv1.weight']))
    #print("encoder=", encoder)
    
    
    
    #encoder = torch.nn.Sequential(*(list(encoder.children())[:-1])) #without fc layer
    #encoder = torch.nn.functional(encoder)
    #encoder.fc = torch.nn.Identity()
    #print(encoder)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet18(pretrained=False)
    decoder = decoder.to(device)
    
    #checkpoint = torch.load(ckp_path)
    #bn.load_state_dict(checkpoint['bn'])
    #decoder.load_state_dict(checkpoint['decoder'])

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))


    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            #print("img=", img.shape)
            inputs = encoder(img)
            print("inputs=", inputs[0].shape, inputs[1].shape, inputs[2].shape)
            outputs = decoder(bn(inputs))#bn(inputs))
            loss = loss_fucntion(inputs, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        if (epoch + 1) % 2 == 0:
            auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device)
            print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
            torch.save({'bn': bn.state_dict(),
                        'decoder': decoder.state_dict()}, ckp_path)
    return auroc_px, auroc_sp, aupro_px




if __name__ == '__main__':

    setup_seed(111)
    item_list = ['cable']
    for i in item_list:
        train(i)
#item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
#                 'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']
