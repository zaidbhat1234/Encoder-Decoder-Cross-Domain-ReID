from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import numpy as np
from config import get_csv_path
import config
import scipy.io
import h5py

def split_datapack(datapack):
    return datapack['image'].cuda(), datapack['label'].cuda(), datapack['camera_id'].cuda()



class AdaptReID_Dataset(Dataset):
    def __init__(self, dataset_name, mode='source', transform=None):
        self.dataset_name = dataset_name
        self.mode = mode
        self.csv_path = config.get_csv_path(self.dataset_name)
        self.csv = self.get_csv()
        
        self.image_names = self.csv['image_path']
        
        
        if mode == 'source':
            self.image_labels = self.csv['id']
            self.unique_index = self.get_unique_index()
            self.unique_index1 = self.unique_index
        else:
            self.image_labels = -1
        
        self.transform = transform
        self.counter = -1
        self.counter1 = -1
        self.counter_ = []
        
        

        if self.mode == 'test' or self.mode == 'query':
            """self.image_camera_ids = self.csv['camera'].values.astype('int')"""
        
        def __len__(self): #returns length of the dataset which is a necessary function overwritten for a dataloader
            return len(self.csv)

            #overwrite this function to extract data, this is called whenever dataloader tries to access new batch of data
    def __getitem__(self, idx):
        
        if self.mode == 'source':
            self.counter1 = self.counter1 +1
            if (self.counter1)<16:
                #print("HERE1:")
                i = np.random.randint(0, len(self.unique_index1))
                #print("RAND:",i)
                self.counter = i
                self.counter_.append(i)
            elif (self.counter1)>=16 and (self.counter1)<32:
                #print("HERE2:")
                i = self.counter_[(self.counter1)-16]
                if (self.counter1)==31:
                    self.counter1 = -1
                    self.counter_=[]
            i1 = np.random.randint(self.unique_index[i][1],self.unique_index[i][2])
            #print("I",i1)
            idx=i1
        input_image = self.get_image(idx)
        
        
        if self.transform is not None:
            input_image = self.transform(input_image)
        
        datapack = {'image': input_image}
        if self.mode =='train':
             datapack['label'] = -1
        else:
            label = self.get_label(idx)
            datapack['label'] = label
        datapack['camera_id'] = -1
        if self.mode == 'test' or self.mode == 'query':
            camer = self.image_names[idx]
            camer = Variable(torch.tensor(float(camer[-17]), dtype=torch.int64))
            #print("image label:", self.image_names[idx], "Camera:" , camer)
            self.image_camera_ids = camer
            datapack['camera_id'] = camer
        #print("label",datapack['label'])
        return datapack

    def get_csv(self):
        csv_name = '{}_list.csv'.format(self.mode)
        csv_f = os.path.join(self.csv_path, csv_name)
        return pd.read_csv(csv_f)

    def get_image(self, idx):
        dataset_path = config.get_dataset_path(self.dataset_name)
        image_path = os.path.join(dataset_path, self.image_names[idx])
        #print(idx,"ID")
        image = Image.open(image_path).convert('RGB')
        return image

    def get_label(self, idx):
        label = self.image_labels[idx]
        label = Variable(torch.tensor(label, dtype=torch.int64))
        return label



    def get_unique_index(self):
        img = self.image_labels
        unique_list = []
        img_ = np.array(img)
        # traverse for all elements
        for x in img:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
        # print list
        #for x in unique_list:
        #print("Unique:", x)

        array = np.zeros(shape=(len(unique_list),3))
        for i in range (len(unique_list)):
            index = np.where(img==unique_list[i])
            index = np.array(index)
            array[i,1] = index.item(0)
            array[i,2] = index.item(-1)
    
        array[:,0] = unique_list
        return array
