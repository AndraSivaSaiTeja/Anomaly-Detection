from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from dataset import MVTecAT
from cutpaste import CutPaste
from model import ProjectionNet
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from cutpaste import CutPaste, cut_paste_collate_fn
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import numpy as np
from collections import defaultdict
from density import GaussianDensitySklearn, GaussianDensityTorch
import pandas as pd
from utils import str2bool
from torchvision.models import resnet18
import pandas as pd
from collections import OrderedDict


#orderedDict = collections.OrderedDict()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval models')
    parser.add_argument('--type', default="all",
                        help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')

    parser.add_argument('--model_dir', default="models",
                    help=' directory contating models to evaluate (default: models)')
    
    parser.add_argument('--cuda', default=False, type=str2bool,
                    help='use cuda for model predictions (default: False)')

    parser.add_argument('--head_layer', default=8, type=int,
                    help='number of layers in the projection head (default: 8)')

    parser.add_argument('--density', default="torch", choices=["torch", "sklearn"],
                    help='density implementation to use. See `density.py` for both implementations. (default: torch)')

    parser.add_argument('--save_plots', default=True, type=str2bool,
                    help='save TSNE and roc plots')
    

    args = parser.parse_args()

    args = parser.parse_args()
    #print(args)
    all_types = ['bottle',
             'cable',
             'capsule',
             'carpet',
             'grid',
             'hazelnut',
             'leather',
             'metal_nut',
             'pill',
             'screw',
             'tile',
             'toothbrush',
             'transistor',
             'wood',
             'zipper']
    
    if args.type == "all":
        types = all_types
    else:
        types = args.type.split(",")
    
    device = "cuda" if args.cuda else "cpu"

    density_mapping = {
        "torch": GaussianDensityTorch,
        "sklearn": GaussianDensitySklearn
    }
    density = density_mapping[args.density]

    # find models
    model_names = [list(Path(args.model_dir).glob(f"model-{data_type}*"))[0] for data_type in types if len(list(Path(args.model_dir).glob(f"model-{data_type}*"))) > 0]
    if len(model_names) < len(all_types):
        print("warning: not all types present in folder")

    # for model_name, data_type in zip(model_names, types):
    #     print(model_name,data_type)

    zip = zip(model_names,types)
    zip_list = list(zip)
    #print(zip_list[0][0],zip_list[0][1])
    #print(len(zip_list)) #15

    modelname = zip_list[0][0]
    defecttype = zip_list[0][1]
    weights = torch.load(modelname)
    #print(type(weights)) #len weights = 136

    model = resnet18(weights=None) #with fc layer
    model = torch.nn.Sequential(*(list(model.children())[:-1])) #without fc layer

    #print((list(model.parameters()))[-1]) #len list(model.parameters()) = 60(without fc layers), it is 62 with fc layer
    #net=model.load_state_dict(weights)
    #print(weights['resnet18.conv1.weight'].shape)
    weights_keys = list(weights.keys())
    #print(len(weights_keys)) #136

    # resnet18_weights_keys = list(model.state_dict())
    # print(len(resnet18_weights_keys)) #60
    # print(weights_keys)

    resnet18_weights_keys = []
    for name, param in model.named_parameters(): 
        if (True): #param.requires_grad: 
            resnet18_weights_keys.append(name)
            #print(name)

    #print(len(resnet18_weights_keys)) #60
    #print(weights_keys,resnet18_weights_keys)

    df = pd.DataFrame(weights_keys)
    writer = pd.ExcelWriter('weights_keys.xlsx')
    df.to_excel(writer, sheet_name='weights_keys', index=False)
    writer.save()

    df = pd.DataFrame(resnet18_weights_keys)
    writer = pd.ExcelWriter('resnet18_weights_keys.xlsx')
    df.to_excel(writer, sheet_name='resnet18_weights_keys', index=False)
    writer.save()

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
                "head.0.weight","head.0.bias", "head.1.weight", "head.1.bias", "head.1.running_mean", "head.1.running_var", "head.1.num_batches_tracked", "head.3.weight", "head.3.bias", "head.4.weight", "head.4.bias", "head.4.running_mean", "head.4.running_var", "head.4.num_batches_tracked", "out.weight", "out.bias"]
    [weights.pop(key) for key in rem_list]
    #print(len(weights)) #100
    
    new_keys = ["0.weight", "1.weight", "1.bias", "1.running_mean", "1.running_var", "4.0.conv1.weight", "4.0.bn1.weight", "4.0.bn1.bias", "4.0.bn1.running_mean", "4.0.bn1.running_var", "4.0.conv2.weight", "4.0.bn2.weight", "4.0.bn2.bias", "4.0.bn2.running_mean", "4.0.bn2.running_var", "4.1.conv1.weight", "4.1.bn1.weight", "4.1.bn1.bias", "4.1.bn1.running_mean", "4.1.bn1.running_var", "4.1.conv2.weight", "4.1.bn2.weight", "4.1.bn2.bias", "4.1.bn2.running_mean", "4.1.bn2.running_var", "5.0.conv1.weight", "5.0.bn1.weight", "5.0.bn1.bias", "5.0.bn1.running_mean", "5.0.bn1.running_var", "5.0.conv2.weight", "5.0.bn2.weight", "5.0.bn2.bias", "5.0.bn2.running_mean", "5.0.bn2.running_var", "5.0.downsample.0.weight", "5.0.downsample.1.weight", "5.0.downsample.1.bias", "5.0.downsample.1.running_mean", "5.0.downsample.1.running_var", "5.1.conv1.weight", "5.1.bn1.weight", "5.1.bn1.bias", "5.1.bn1.running_mean", "5.1.bn1.running_var", "5.1.conv2.weight", "5.1.bn2.weight", "5.1.bn2.bias", "5.1.bn2.running_mean", "5.1.bn2.running_var", "6.0.conv1.weight", "6.0.bn1.weight", "6.0.bn1.bias", "6.0.bn1.running_mean", "6.0.bn1.running_var", "6.0.conv2.weight", "6.0.bn2.weight", "6.0.bn2.bias", "6.0.bn2.running_mean", "6.0.bn2.running_var", "6.0.downsample.0.weight", "6.0.downsample.1.weight", "6.0.downsample.1.bias", "6.0.downsample.1.running_mean", "6.0.downsample.1.running_var", "6.1.conv1.weight", "6.1.bn1.weight", "6.1.bn1.bias", "6.1.bn1.running_mean", "6.1.bn1.running_var", "6.1.conv2.weight", "6.1.bn2.weight", "6.1.bn2.bias", "6.1.bn2.running_mean", "6.1.bn2.running_var", "7.0.conv1.weight", "7.0.bn1.weight", "7.0.bn1.bias", "7.0.bn1.running_mean", "7.0.bn1.running_var", "7.0.conv2.weight", "7.0.bn2.weight", "7.0.bn2.bias", "7.0.bn2.running_mean", "7.0.bn2.running_var", "7.0.downsample.0.weight", "7.0.downsample.1.weight", "7.0.downsample.1.bias", "7.0.downsample.1.running_mean", "7.0.downsample.1.running_var", "7.1.conv1.weight", "7.1.bn1.weight", "7.1.bn1.bias", "7.1.bn1.running_mean", "7.1.bn1.running_var", "7.1.conv2.weight", "7.1.bn2.weight", "7.1.bn2.bias", "7.1.bn2.running_mean", "7.1.bn2.running_var"]
    #print(len(new_keys)) #100
    # i=0
    # for k,v in weights.items():
    #     weights[k] = new_keys[i]
    #     i+=1

    weights2 = OrderedDict([('a',1)])
    i=0
    for key in weights.keys():
        temp = weights[key]
        weights2[new_keys[i]] = temp
        #weights.pop(key)
        i+=1

    weights2.pop('a')

    model.load_state_dict(weights2)
    a = torch.Tensor(list(model.parameters())[0])
    b = torch.Tensor((weights['resnet18.conv1.weight']))
    # print(a.shape,b.shape)
    # print(a-b) #getting zeros hence loaded properly
    