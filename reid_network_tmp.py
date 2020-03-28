from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import random


class Base_Encoder(nn.Module):                                              #torch.nn the parent class
    def __init__(self, backbone='resnet-50'):                               # Base encoder of the ARN network
        super(Base_Encoder, self).__init__()
        if backbone == 'resnet-50':                                         # Encoder with resnet50 architecture
            self.model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AvgPool2d((7,7))                                  # initialise avgpool
    
    def forward(self, input_feature, use_avg=False):
        feature = self.model(input_feature)                                 #pass input features to model to obtain feature vector
        if use_avg:
            feature = self.avgpool(feature)                                 # average pool this feature
            feature = feature.view(feature.size()[0],-1)
        return feature

class Encoder(nn.Module):                                                   #Encoder setup for the 3 domain variant/ invariant modules
    def __init__(self, backbone='resnet-50'):
        super(Encoder, self).__init__()
        if backbone == 'resnet-50':
            self.model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[7:-2])
        self.avgpool = nn.AvgPool2d((7,7))                                  # initialise avgpool
    
    def forward(self, input_feature, use_avg=False):
        feature = self.model(input_feature)                                 #pass input features to model to obtain feature vector
        if use_avg:
            feature = self.avgpool(feature)                                 # average pool this feature
            feature = feature.view(feature.size()[0],-1)
        return feature




class Base_Decoder(nn.Module):                                                          #Base decoder that takes the domain variant/invariant feature vectors and tries to reconstruct
    def __init__(self, ch_list=None):                                                   #image back in 3 channels
        super(Base_Decoder, self).__init__()
        def get_layers(in_filters, out_filters, stride=1, out_pad=0, last=False,first=False):       # function to construct the decoder setup layer by layer
            
            if first:
                layers = [nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),nn.Conv2d(in_filters, out_filters, kernel_size=3,\
                                             stride=stride, padding=1),
                          nn.BatchNorm2d(out_filters)]
            else:
                layers = [nn.Conv2d(in_filters, out_filters, kernel_size=3,\
                                             stride=stride, padding=1 ),
                          nn.BatchNorm2d(out_filters)]
            
            
            if last==False:
                layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                #if the last layer of decoder we want to use tanh activation else leaky relu
               
            return layers
                          
        def make_blocks(in_ch, out_ch, last=False):                                     #function to use above layer function to form decoder block by block
            
            block = nn.Sequential(*get_layers(in_ch, out_ch,first = True),
                *get_layers(out_ch, out_ch),
                *get_layers(out_ch, out_ch, last=last)
            )
            return block
                                      
        self.block1 = make_blocks(ch_list[0], ch_list[1])                               # block by block making decoder with the in/out sizes from ch_list
        self.block2 = make_blocks(ch_list[1], ch_list[2])
        self.block3 = make_blocks(ch_list[2], ch_list[3])
        self.block4 = make_blocks(ch_list[3], ch_list[4])
        self.block5 = make_blocks(ch_list[4], ch_list[5], last=True)
    def forward(self, input_feature):                                                   #function to run the decoder setup block by block
        feature1 = self.block1(input_feature)
        feature2 = self.block2(feature1)
        feature3 = self.block3(feature2)
        feature4 = self.block4(feature3)
        feature5 = self.block5(feature4)
        return feature5


class Classifier(nn.Module):                                                            #to make output prediction by the decoder
    def __init__(self, input_dim=2048, output_dim=-1):
        super(Classifier, self).__init__()
        #self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, feature):
        #feature = self.dropout(feature)
        prediction = self.linear(feature)
        return prediction



"""
    Add random jitter in brightness , contrast, etc in the image to augment the image passing via pose encoder """

def aug_image(image, rgb_shuf_val):
    if np.random.randint(3) <= 1: # 66% times negative
        rgb_image = 1-np.stack([image[:,:,rgb_shuf_val[0]], image[:,:,rgb_shuf_val[1]], image[:,:,rgb_shuf_val[2]]], 2)
    else:
        rgb_image = np.stack([image[:,:,rgb_shuf_val[0]], image[:,:,rgb_shuf_val[1]], image[:,:,rgb_shuf_val[2]]], 2)
    return rgb_image

def jitter_image(img):
    rgb_shuf_val = np.arange(3)
    random.shuffle(rgb_shuf_val)
    
    rgb_image = aug_image(img, rgb_shuf_val)
    
    return rgb_image


def jitter(image):
    data=[]
    for tensor in image:
        tensor1 = jitter_image(tensor.cpu().numpy().transpose())
        tensor1 = torch.Tensor(tensor1)
        tensor1 = torch.Tensor(tensor1.cpu().numpy().transpose())
        data.append(tensor1)
    #print("DATA",data)
    img = torch.stack(data)
#print("final", img.shape)
    return img


def tb_visualize(image):
    """
        Un-Normalise the image
        """
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



def norm(image):
    """
        Normalise the image
        """
    data=[]
    mean1=(0.485, 0.456, 0.406)
    std1=(0.229, 0.224, 0.225)
    for tensor in image:
        coun = -1
        
        data1=[]
        for tensor1 in tensor:
            coun = coun+1
            tensor1 = (tensor1-mean1[coun]) / std1[coun]
            data1.append(tensor1)
        t = torch.stack(data1)
        data.append(t)
    final = torch.stack(data)

    image = final
    return image



class AdaptReID_model(nn.Module):
    def __init__(self, backbone='resent-50', \
                 classifier_input_dim=2048, classifier_output_dim=-1):
        super(AdaptReID_model, self).__init__()

        self.encoder_base = Base_Encoder(backbone=backbone)
        
        self.encoder_t = Base_Encoder(backbone=backbone)

        
        self.ch_list = [ 2048 ,1024,  512, 256, 64, 3]
        self.decoder = Base_Decoder(ch_list=self.ch_list)
        self.decoder1 = Base_Decoder(ch_list=self.ch_list)
    
        self.classifier = Classifier(input_dim=classifier_input_dim, output_dim=classifier_output_dim)


    def forward(self, source_img, target_img, negative_img , flag):
        if flag == 0:
            """
                flag 0 when testing just 1 branch of the model
                """
            feature = self.encoder_base(source_img)
            feature_avg = self.encoder_base(source_img,use_avg=True)
            #recon_img = self.decoder(feature_1)
            
            pred_s = self.classifier(feature_avg)
            
            feature1 = self.encoder_base(target_img)
            feature_avg_1 = self.encoder_base(target_img,use_avg=True)
            
            feature2 = self.encoder_base(negative_img)
            feature_avg_2 = self.encoder_base(negative_img,use_avg=True)
            

            recon_img1 = self.decoder1(feature)
            
            return recon_img1, feature_avg, feature_avg_1, feature_avg_2,pred_s
        elif flag == 1:
            """
                flag=1 for the entire model
                """
            feature = self.encoder_base(source_img)
            feature_avg = self.encoder_base(source_img,use_avg=True)
            #recon_img = self.decoder(feature_1)
            
            pred_s = self.classifier(feature_avg)
            
            feature1 = self.encoder_base(target_img)
            feature_avg_1 = self.encoder_base(target_img,use_avg=True)
            
            feature2 = self.encoder_base(negative_img)
            feature_avg_2 = self.encoder_base(negative_img,use_avg=True)
            
            """to pass jittered images uncomment and replace target_img by jitter_img.cuda()"""
            #jitter_img = jitter(tb_visualize(target_img))
            #jitter_img = norm(jitter_img)
            
            featuret1 = self.encoder_t(target_img)
            feature_t_avg = self.encoder_t(target_img,use_avg=True)
            
            

            
            feature_concat1 = torch.cat((featuret1,feature),1)


            recon_img1 = self.decoder1(feature_concat1)

            return recon_img1, feature_avg, feature_avg_1, feature_avg_2, feature_t_avg,pred_s


