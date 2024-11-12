# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import copy
import sys
import os
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from test_1 import evaluation, visualization, test
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
    device = 'cpu'

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = '/home/turing/Ranjeet/pytorch-cutpaste-master/Data/' + _class_ + '/train'
    test_path = '/home/turing/Ranjeet/pytorch-cutpaste-master/Data/' + _class_
    ckp_path = './checkpoints/' + 'wres50_'+_class_+'.pth'
    ckp_path11 = './checkpoints/' + 'wres50_e'+_class_+'.pth'
    ckp_path_1 = './checkpoints/' + 'wres50_n_quan'+_class_+'.pth'
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder = encoder.to("cpu")
    fused_encoder = copy.deepcopy(encoder)
    encoder.eval()
    fused_encoder.eval()
    
    # Fuse the model in place rather manually.
    fused_encoder = torch.quantization.fuse_modules(fused_encoder, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in fused_encoder.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
                        
                       

    assert model_equivalence(model_1=encoder, model_2=fused_encoder, device=cpu_device, rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,32,32)), "Fused model is not equivalent to the original model!"
    
    quantized_model = QuantizedResNet18(model_fp32=fused_encoder)
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    quantized_model.qconfig = quantization_config
    print(quantized_model.qconfig)
    torch.quantization.prepare(quantized_model, inplace=True)
    calibrate_model(model=quantized_model, loader=train_loader, device=cpu_device)
    quantized_model = torch.quantization.convert(quantized_model, inplace=True)
    quantized_model.eval()
    print(quantized_model)
    _, fp32_eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=cpu_device, criterion=None)
    _, int8_eval_accuracy = evaluate_model(model=quantized_jit_model, test_loader=test_loader, device=cpu_device, criterion=None)
    print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_accuracy))
    print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))
    
    save_torchscript_model(model=quantized_model, model_dir=model_dir, model_filename=quantized_model_filename)
    quantized_jit_model = load_torchscript_model(model_filepath=quantized_model_filepath, device=cpu_device)
    
    encoder.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    #model_fp32_fused = torch.quantization.fuse_modules(encoder, [['conv', 'relu']])
    model_fp32_prepared = torch.quantization.prepare(encoder, inplace=True)
    #model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)
    with torch.no_grad():
        for img, label in train_dataloader:
        #for img, gt, label, _ in test_dataloader:
            model_fp32_prepared(img)
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    checkpoint = torch.load(ckp_path, map_location='cpu')
    bn.load_state_dict(checkpoint['bn'])
    decoder.load_state_dict(checkpoint['decoder'])
    torch.save({'encoder': encoder.state_dict(),'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, ckp_path11)
    bn = bn.to("cpu")
    bn.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    #model_fp32_fused = torch.quantization.fuse_modules(encoder, [['conv', 'relu']])
    model_fp32_prepared_bn = torch.quantization.prepare(bn, inplace=True)
    #model_fp32_prepared = torch.quantization.prepare(model_fp32_fused    

    with torch.no_grad():
        for img, label in train_dataloader:
        #for img, gt, label, _ in test_dataloader:
            out1 = model_fp32_prepared(img)
            model_fp32_prepared_bn(out1)

    model_int8_bn = torch.quantization.convert(model_fp32_prepared_bn)
    decoder = decoder.to("cpu")
    decoder.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    #model_fp32_fused = torch.quantization.fuse_modules(encoder, [['conv', 'relu']])
    model_fp32_prepared_decoder = torch.quantization.prepare(decoder, inplace=True)
    #model_fp32_prepared = torch.quantization.prepare(model_fp32_fused    

    with torch.no_grad():
        for img, label in train_dataloader:
        #for img, gt, label, _ in test_dataloader:
            out1 = model_fp32_prepared(img)
            out2 = model_fp32_prepared_bn(out1)
            model_fp32_prepared_decoder(out2)

    model_int8_decoder = torch.quantization.convert(model_fp32_prepared_decoder)
    torch.save({'encoder': model_int8.state_dict(),'bn':  model_int8_bn.state_dict(), 'decoder': model_int8_decoder.state_dict()}, ckp_path_1)

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))
    bn.eval()
    decoder.eval()
    encoder.eval()
    pytorch_total_params = sum(p.numel() for p in encoder.parameters()) + sum(q.numel() for q in bn.parameters()) + sum(r.numel() for r in decoder.parameters())
    print("Total number of parameters=", pytorch_total_params)
    #sys.exit()
    auroc_px, auroc_sp, aupro_px = evaluation(model_int8, bn, decoder, test_dataloader, device,_class_)
    print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
    #torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, ckp_path)
    return auroc_px, auroc_sp, aupro_px




if __name__ == '__main__':

    setup_seed(111)
    item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']
    for i in item_list:
        train(i)

