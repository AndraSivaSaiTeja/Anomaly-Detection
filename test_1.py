import torch
from dataset import get_data_transforms, load_data
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle
import matplotlib.pyplot as plt
import numpy as np

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i] # Consider one output feature map at a time from encoder
        ft = ft_list[i] # Consider corresponding output feature map at a time from decoder
        #print("ft =", ft .shape)
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        #print("a_map=", a_map.shape)
        a_map = torch.unsqueeze(a_map, dim=1)
        #print("a_map=", a_map.shape)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        #print("a_map=", a_map.shape)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        #print("a_map=", a_map.shape)
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap



def evaluation(encoder, bn, decoder, dataloader,device,_class_=None):
    #_, t_bn = resnet50(pretrained=True)
    #bn.load_state_dict(bn.state_dict())
    bn.eval()
    #print("_class_=", _class_)
    #bn.training = False
    #t_bn.to(device)
    #t_bn.load_state_dict(bn.state_dict())
    decoder.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    tp_total = [] 
    total_r_gt = [] 
    fp_total = []
    fp_r_total = []
    count = 0
    with torch.no_grad():
        for img, gt, label, _ in dataloader:

            #print("img, gt, label", img.shape, gt.shape, label.shape)
            #if (count==3):
            #	break
            count = count + 1
            #print("label.item()=", label.item())
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            #print(img.shape[-1])
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a') # find the anomaly map
            anomaly_map = gaussian_filter(anomaly_map, sigma=4) # Apply the gaussian filter
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item()!=0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis,:,:], tp_total, total_r_gt, fp_total, fp_r_total))
            #gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            #pr_list_px.extend(anomaly_map.ravel())
            #gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            #pr_list_sp.append(np.max(anomaly_map))
            
            print("tp=", len(tp_total))

       #ano_score = (pr_list_sp - np.min(pr_list_sp)) / (np.max(pr_list_sp) - np.min(pr_list_sp))
        #vis_data = {}
        #vis_data['Anomaly Score'] = ano_score
        #vis_data['Ground Truth'] = np.array(gt_list_sp)
        # print(type(vis_data))
        # np.save('vis.npy',vis_data)
        #with open('{}_vis.pkl'.format(_class_), 'wb') as f:
        #    pickle.dump(vis_data, f, pickle.HIGHEST_PROTOCOL)


        #auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        #auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    tp_total = np.asarray(tp_total)
    total_r_gt = np.asarray(total_r_gt)
    fp_total = np.asarray(fp_total)
    fp_r_total = np.asarray(fp_r_total)
    
    tp_total = np.sum(tp_total, axis=0)
    total_r_gt = np.sum(total_r_gt, axis=0)
    fp_total = np.sum(fp_total, axis=0)
    fp_r_total = np.sum(fp_r_total, axis=0)
    fpr = []
    tpr = []
    for i in range(tp_total.shape[0]):
    	tpr.append(tp_total[i]/float(total_r_gt[i]))
    	fpr.append(fp_total[i]/float(fp_r_total[i]))
    	#print("tp_total[i]=", tp_total[i])
    	#print("total_r_gt[i]=", total_r_gt[i])
    #print("tpr=", tpr)
    print("tpr=", max(tpr))
    print("fpr=", max(fpr))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    a_path = _class_ + ".png"
    plt.scatter(fpr, tpr)
    plt.axvline(x=0.01)
    plt.savefig(a_path)
    auroc_px= 0
    auroc_sp=0
    aupro_list = [0]
    	
    return auroc_px, auroc_sp, round(np.mean(aupro_list),3)

def test(_class_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(_class_)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = '../mvtec/' + _class_
    ckp_path = './checkpoints/' + 'rm_1105_wres50_ff_mm_' + _class_ + '.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device,_class_)
    print(_class_,':',auroc_px,',',auroc_sp,',',aupro_px)
    return auroc_px

import os

def visualization(_class_):
    print(_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = '../mvtec/' + _class_
    ckp_path = './checkpoints/' + 'rm_1105_wres50_ff_mm_'+_class_+'.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)

    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    count = 0
    with torch.no_grad():
        for img, gt, label, _ in test_dataloader:
            if (label.item() == 0):
                continue
            #if count <= 10:
            #    count += 1
            #    continue

            decoder.eval()
            bn.eval()

            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            #inputs.append(feature)
            #inputs.append(outputs)
            #t_sne(inputs)


            anomaly_map, amap_list = cal_anomaly_map([inputs[-1]], [outputs[-1]], img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img)*255)
            #if not os.path.exists('./results_all/'+_class_):
            #    os.makedirs('./results_all/'+_class_)
            #cv2.imwrite('./results_all/'+_class_+'/'+str(count)+'_'+'org.png',img)
            #plt.imshow(img)
            #plt.axis('off')
            #plt.savefig('org.png')
            #plt.show()
            ano_map = show_cam_on_image(img, ano_map)
            #cv2.imwrite('./results_all/'+_class_+'/'+str(count)+'_'+'ad.png', ano_map)
            plt.imshow(ano_map)
            plt.axis('off')
            #plt.savefig('ad.png')
            plt.show()

            gt = gt.cpu().numpy().astype(int)[0][0]*255
            #cv2.imwrite('./results/'+_class_+'_'+str(count)+'_'+'gt.png', gt)

            #b, c, h, w = inputs[2].shape
            #t_feat = F.normalize(inputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #s_feat = F.normalize(outputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #c = 1-min_max_norm(cv2.resize(anomaly_map,(h,w))).flatten()
            #print(c.shape)
            #t_sne([t_feat, s_feat], c)
            #assert 1 == 2

            #name = 0
            #for anomaly_map in amap_list:
            #    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            #    ano_map = min_max_norm(anomaly_map)
            #    ano_map = cvt2heatmap(ano_map * 255)
                #ano_map = show_cam_on_image(img, ano_map)
                #cv2.imwrite(str(name) + '.png', ano_map)
                #plt.imshow(ano_map)
                #plt.axis('off')
                #plt.savefig(str(name) + '.png')
                #plt.show()
            #    name+=1
            count += 1
            #if count>20:
            #    return 0
                #assert 1==2


def vis_nd(name, _class_):
    print(name,':',_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    ckp_path = './checkpoints/' + name + '_' + str(_class_) + '.pth'
    train_dataloader, test_dataloader = load_data(name, _class_, batch_size=16)

    encoder, bn = resnet18(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet18(pretrained=False)
    decoder = decoder.to(device)

    ckp = torch.load(ckp_path)

    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    decoder.eval()
    bn.eval()

    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []

    count = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            #if count <= 10:
            #    count += 1
            #    continue
            img = img.to(device)
            inputs = encoder(img)
            #print(inputs[-1].shape)
            outputs = decoder(bn(inputs))


            anomaly_map, amap_list = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            #anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img)*255)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'org.png',img)
            #plt.imshow(img)
            #plt.axis('off')
            #plt.savefig('org.png')
            #plt.show()
            ano_map = show_cam_on_image(img, ano_map)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'ad.png', ano_map)
            #plt.imshow(ano_map)
            #plt.axis('off')
            #plt.savefig('ad.png')
            #plt.show()

            #gt = gt.cpu().numpy().astype(int)[0][0]*255
            #cv2.imwrite('./results/'+_class_+'_'+str(count)+'_'+'gt.png', gt)

            #b, c, h, w = inputs[2].shape
            #t_feat = F.normalize(inputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #s_feat = F.normalize(outputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #c = 1-min_max_norm(cv2.resize(anomaly_map,(h,w))).flatten()
            #print(c.shape)
            #t_sne([t_feat, s_feat], c)
            #assert 1 == 2

            #name = 0
            #for anomaly_map in amap_list:
            #    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            #    ano_map = min_max_norm(anomaly_map)
            #    ano_map = cvt2heatmap(ano_map * 255)
                #ano_map = show_cam_on_image(img, ano_map)
                #cv2.imwrite(str(name) + '.png', ano_map)
                #plt.imshow(ano_map)
                #plt.axis('off')
                #plt.savefig(str(name) + '.png')
                #plt.show()
            #    name+=1
            #count += 1
            #if count>40:
            #    return 0
                #assert 1==2
            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))  # np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1

        ano_score = (prmean_list_sp-np.min(prmean_list_sp))/(np.max(prmean_list_sp)-np.min(prmean_list_sp))
        vis_data = {}
        vis_data['Anomaly Score'] = ano_score
        vis_data['Ground Truth'] = np.array(gt_list_sp)
        #print(type(vis_data))
        #np.save('vis.npy',vis_data)
        with open('vis.pkl','wb') as f:
            pickle.dump(vis_data,f,pickle.HIGHEST_PROTOCOL)


def compute_pro(masks: ndarray, amaps: ndarray, tp_total, total_r_gt, fp_total, fp_r_total, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)
    #print("binary_amaps=", np.max(binary_amaps))
    #print("binary_amaps=", np.min(binary_amaps))
    amaps = amaps.astype('float16')
    
    min_th = 0.0
    max_th = 2.0
    delta = 0.001
    i=0
    #print("min_th=", min_th)
    #print("delta=", delta)
    #print("max_th=", max_th)
    tp_total_loc = []
    total_r_gt_loc = []
    fp_total_loc = []
    fp_r_total_loc = []

    for th in np.arange(min_th, max_th, delta):
        #print("th=", th)
        #print("min_th=", min_th)
        #print("amaps <= th=", amaps <= th)
        #th = float(th)
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1
        #print("binary_amaps=", np.max(binary_amaps))
        #sys.exit()

        pros = []
        pros_tp = 0 
        pros_reg = 0
        for binary_amap, mask in zip(binary_amaps, masks):
            #print("mask=", mask.shape)
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                #print("axes1_ids", axes1_ids)
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)
                pros_tp = pros_tp + tp_pixels
                pros_reg = pros_reg + region.area
                #print("tp_pixels=", tp_pixels)
                #print("region.area=", region.area)
                #sys.exit()
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        #fpr = fp_pixels / inverse_masks.sum()
        fpr = 0.0
        fp_total_loc.append(fp_pixels)
        fp_r_total_loc.append(inverse_masks.sum())
        tp_total_loc.append(pros_tp)
        total_r_gt_loc.append(pros_reg)
        
        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    #df = df[df["fpr"] < 0.3]
    #df["fpr"] = df["fpr"] / df["fpr"].max()
    fp_total_loc = np.asarray(fp_total_loc)
    fp_r_total_loc = np.asarray(fp_r_total_loc)
    tp_total_loc = np.asarray(tp_total_loc)
    total_r_gt_loc = np.asarray(total_r_gt_loc)
    
    tp_total.append(tp_total_loc) 
    total_r_gt.append(total_r_gt_loc) 
    fp_total.append(fp_total_loc) 
    fp_r_total.append(fp_r_total_loc)

    pro_auc = auc(df["fpr"], df["pro"])
    #print("pro_auc=", pro_auc)
    return pro_auc

def detection(encoder, bn, decoder, dataloader,device,_class_):
    #_, t_bn = resnet50(pretrained=True)
    bn.load_state_dict(bn.state_dict())
    bn.eval()
    #t_bn.to(device)
    #t_bn.load_state_dict(bn.state_dict())
    decoder.eval()
    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []
    with torch.no_grad():
        for img, label in dataloader:

            img = img.to(device)
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            label = label.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], 'acc') # Find the anomaly map between inputs and outputs by utilizing cosine distance.
            anomaly_map = gaussian_filter(anomaly_map, sigma=4) # Apply gaussian filter on anomaly map to smooth the same.


            gt_list_sp.extend(label.cpu().data.numpy()) # Add the labels in gt_list_sp
            prmax_list_sp.append(np.max(anomaly_map)) # Add maximum value of anomaly map in prmax_list_sp
            prmean_list_sp.append(np.sum(anomaly_map))#np.sum(anomaly_map.ravel().argsort()[-1:][::-1])) # Add the sum of anomaly map in the prmean_list_sp 

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1


        auroc_sp_max = round(roc_auc_score(gt_list_sp, prmax_list_sp), 4)
        auroc_sp_mean = round(roc_auc_score(gt_list_sp, prmean_list_sp), 4)
    return auroc_sp_max, auroc_sp_mean
