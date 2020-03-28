from __future__ import print_function, division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms as trans
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from reid_network_tmp import AdaptReID_model
from reid_dataset import AdaptReID_Dataset
from reid_loss import *
import config
from reid_evaluate_temp import evaluate
import test1
import random

parser = ArgumentParser()
parser.add_argument("--use_gpu", default=2, type=int)
parser.add_argument("--source_dataset", choices=['CUHK03', 'Duke', 'Market', 'MSMT17_V1'], type=str)
parser.add_argument("--target_dataset", choices=['CUHK03', 'Duke', 'Market', 'MSMT17_V1'], type=str)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--total_epochs", default=1000, type=int)
parser.add_argument("--w_loss_rec", default=0.1, type=float)
parser.add_argument("--w_loss_dif", default=0.1, type=float)
parser.add_argument("--w_loss_mmd", default=0.1, type=float)
parser.add_argument("--w_loss_ctr", default=1.0, type=float)
parser.add_argument("--dist_metric", choices=['L1', 'L2', 'cosine', 'correlation'], type=str)
parser.add_argument("--rank", default=1, type=int)
parser.add_argument("--model_dir", default='model', type=str)
parser.add_argument("--model_name", default='basic_10cls', type=str)
parser.add_argument("--pretrain_model_name", default=None, type=str)

args = parser.parse_args()
total_batch = 4272
#trans.RandomHorizontalFlip(p=0.5),
transform_list = [trans.Resize(size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),\
                  trans.ToTensor(), trans.Normalize(mean=config.MEAN, std=config.STD)]



def setup_gpu():
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    print('Using GPU: {}'.format(args.use_gpu))
    

def tb_visualize(image):
    """Un-normalises the images"""
    data=[]
    mean1=(0.485, 0.456, 0.406)
    std1=(0.229, 0.224, 0.225)
    for tensor in image:
        coun = -1
        
        data1=[]
        for tensor1 in tensor:
            coun = coun+1
            tensor1 = (tensor1*std1[coun]) + mean1[coun]
            data1.append(tensor1)
        t = torch.stack(data1)
        data.append(t)
    final = torch.stack(data)
            
    image = final
    return image


def get_batch(data_iter, data_loader):
    try:
        _, batch = next(data_iter)
    except:
        data_iter = enumerate(data_loader)
        _, batch = next(data_iter)
    if batch['image'].size(0) < args.batch_size or batch['image'].size(0) > args.batch_size:
        batch, data_iter = get_batch(data_iter, data_loader)
    return batch, data_iter

def save_model(model, step):
    model_path = '{}/{}/{}.pth.tar'.format(args.model_dir ,args.model_name ,args.model_name)
    torch.save(model.state_dict(), model_path)

def save_epoch(epoch):
    epoch_path ='{}/{}/{}_{}.pth.tar'.format(args.model_dir ,args.model_name ,args.model_name,epoch)
    torch.save(epoch.state_dict(), epoch_path)

def train():
    print('Model name: {}'.format(args.model_name))
    classifier_output_dim = config.get_dataoutput(args.source_dataset)
    
    model = AdaptReID_model(backbone='resnet-50', classifier_output_dim=classifier_output_dim).cuda()
    # print(model.state_dict())
    
    
    #Load pretrained model after saving it
    if args.pretrain_model_name is not None:
        print("Loading pre-trained model")
        model.load_state_dict(torch.load('{}/{}.pth.tar'.format(args.model_dir, args.pretrain_model_name)))
    
    
    sourceData = AdaptReID_Dataset(dataset_name=args.source_dataset, mode='source', transform=trans.Compose(transform_list),batch_size = args.batch_size)
    sourceDataloader = DataLoader(sourceData, batch_size=args.batch_size, shuffle=True)
    source_iter = enumerate(sourceDataloader)

    """
    #DataLoader for target dataset
    targetData = AdaptReID_Dataset(dataset_name=args.target_dataset, mode='train',
                               transform=trans.Compose(transform_list),batch_size = args.batch_size)

    targetDataloader = DataLoader(targetData, batch_size=args.batch_size)
    target_iter = enumerate(targetDataloader)"""

    unique_list = sourceData.unique_list

    
    model_opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    writer = SummaryWriter('/media/zaid/zaid1/log/{}'.format(args.model_name))
    match, junk = None, None
  

    for epoch in range(args.total_epochs):
        for batch_idx in range(total_batch):
            model.train()

            step = epoch*total_batch + batch_idx
            
            print('{}: epoch: {}/{} step: {}/{} ({:.2f}%)'.format(args.model_name, epoch+1, args.total_epochs, step+1, total_batch, float(batch_idx+1)*100.0/total_batch))

            #Source DataLoader
            sourceDataloader = DataLoader(sourceData, batch_size=args.batch_size, pin_memory= True)  # Creates iterale dataset , shuffle ensures random selection of batches from dataset

            source_iter = enumerate(sourceDataloader) #total images/batch size = number of time to run in each epoch to test all images.
            
            #Get Batch of source image
            source_batch, source_iter = get_batch(source_iter, sourceDataloader)
            source_image, source_label, _,idxx = split_datapack(source_batch)

            
            #Get Batch of source image positive anchor
            source_batch_1, source_iter = get_batch(source_iter, sourceDataloader)
            source_image1, source_label1, _,idxx = split_datapack(source_batch_1)
            
            #Get Batch of source image negative anchor
            source_batch_neg, source_iter = get_batch(source_iter, sourceDataloader)
            source_image_neg, source_label_neg, _,idxx = split_datapack(source_batch_neg)
            
            """target_batch, target_iter = get_batch(target_iter, targetDataloader)
            target_image, target_label, _,idxx = split_datapack(target_batch)"""
            
            #during training only 751 unique id's between 0-1500 are in dataset so alias index has to be obtained for classification loss
            alias_index = source_label
            for ind in range (args.batch_size):
                alias_index[ind] = unique_list.index(source_label[ind].data)
            #alias_index = alias_index%50 #if 50 id's chosen at random
            #print("LL",alias_index)
            
            
            """
                recon_img = reconstructed image having same pose as positive anchor and appearance as source image
                feature = identity embedding corresponding to source image
                feature1 = identity embedding corresponding to positive anchor
                feature2 = identity embedding corresponding to negative image
                feature_or = pose embedding corresponding to positive anchor
                pred_s = predicted labels for the source image by the classifier in the model
                
                """
            
            recon_img,feature,feature1,feature2,pred_s = model(source_img=source_image,target_img = source_image1,negative_img=source_image_neg, flag = 0)
            #loss_rec = loss_rec_func(recon_img,tb_visualize(source_image))
            #loss_dif = loss_dif_func(feature,  feature_or) #orthogonality loss
            #loss_trip = loss_triplet1(feature,feature1,feature2) *5 #Triplet loss with added weight
            loss_cls = loss_cls_func(pred_s, alias_index) #classification loss
            loss_mse_func = torch.nn.MSELoss()
            loss_app = loss_mse_func(feature,feature1) #appearance loss
            loss = loss_cls
            """
            recon_img,feature,feature1,feature2,feature_or,pred_s = model(source_img=source_image,target_img = source_image1,negative_img=source_image_neg, flag = 1)
            loss_rec = loss_rec_func(recon_img,tb_visualize(source_image1))
            loss_dif = loss_dif_func(feature,  feature_or) #orthogonality loss
            #loss_trip = loss_triplet1(feature,feature1,feature2) *10 #Triplet loss with added weight
            loss_cls = loss_cls_func(pred_s, alias_index) #classification loss
            loss_mse_func = torch.nn.MSELoss()
            loss_app = loss_mse_func(feature,feature1) #appearance loss
            loss = loss_cls +loss_rec +loss_dif + loss_app
            """
            
           
            """
                Writing Images and Losses to logs for tensorboard
                """
            
            if (step+1)%3000==0:
                
                
                source_image = tb_visualize(source_image)
                source_image_ = make_grid(source_image)
                writer.add_image('source image', source_image_, step)
                
                
                source_image1 = tb_visualize(source_image1)
                source_image_1 = make_grid(source_image1)
                writer.add_image('source image pos', source_image_1, step)
                #recon_img = tb_visualize(recon_img)
                rec_image_ = make_grid(recon_img)
                writer.add_image('rec image', rec_image_, step)
                """
                target_image1 = tb_visualize(target_image)
                target_image_1 = make_grid(target_image1)
                writer.add_image('target domain image', target_image_1, step)
                """
                source_image_n = tb_visualize(source_image_neg)
                source_image_n = make_grid(source_image_n)
                writer.add_image('source image neg', source_image_n, step)
                writer.add_scalar('loss_cls', loss_cls, step)
                #writer.add_scalar('loss_rec', loss_rec, step)
                #writer.add_scalar('loss_trip', loss_trip, step)
                #writer.add_scalar('loss_app', loss_app, step)
                writer.add_scalar('loss', loss, step)
                
                save_model(model, step)
            #save_epoch(epoch)

            #update model
            
            model_opt.zero_grad()
            loss.backward()
            model_opt.step()
            #evaluate
            if (step+1)%6000==0:
                rank_score, match, junk = evaluate(args, model, transform_list, match, junk)
                writer.add_scalar('rank1_score', rank_score, step)
                print("RANK 5 : ", rank_score)
    #save_model(model, step)



    writer.close()

setup_gpu()
train()
