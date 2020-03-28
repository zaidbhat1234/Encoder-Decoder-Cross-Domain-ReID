from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import numpy as np
import config
import scipy.io
import h5py


def split_datapack(datapack):
    return datapack['image'].cuda(), datapack['label'].cuda(), datapack['camera_id'].cuda(), datapack['idxx']

class AdaptReID_Dataset(Dataset):
    def __init__(self, dataset_name, mode='source', transform=None, batch_size=8):
        self.dataset_name = dataset_name
        self.mode= mode
        self.flag=0
        self.id_sal=[] #storing id's of images corresponding to actual images to retrieve corresponding saliency maps
        self.args_batchsize = batch_size
        
        
        if mode=='sal':
            self.counter_sal =-1
            self.flag = 1
            self.mode = 'source'
        self.csv_path = config.get_csv_path(self.dataset_name)
        self.csv = self.get_csv()
        self.image_names = self.csv['image_path']
        
        
        #to get path of saliency images from the train csv file by replacing 'bounding_box_train' with 'binary'
        if mode=='sal':
            self.mode = 'source'
            self.image_names = self.csv['image_path']
            #print(self.image_names)
            cnt = 0
            for row in self.image_names:
                row1 = row[18:]
                row1 = 'binary' + row1
                self.image_names[cnt]=row1
                cnt=cnt+1
    
    #create unique list of id's present in dataset to access a batch of unique id'ed images
        if mode == 'source' or self.mode == 'test' or self.mode == 'query' or mode =='sal':
            self.image_labels = self.csv['id']
            self.unique_index,self.unique_list = self.get_unique_index()
            self.unique_index1 = self.unique_index
        else:
            self.image_labels = -1
        
        self.transform = transform
        self.counter = -1
        self.counter1 = -1
        self.counter_ = []
        self.arr = []
        
        
        

        """ if self.mode == 'test' or self.mode == 'query' :"""
        self.image_camera_ids = self.csv['camera_id'].values.astype('int')
    #print("Camera", self.image_camera_ids)

    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        
        """get_batch function calls this function to get images accessed by idx"""
        
        
        
        """to get batchsize number of unique id's for each iteration, store those in self.counter_ to get the same id's for the positive anchor and different id's for negative anchor"""
        
        if self.mode == 'source' and self.flag==0:
            self.counter1 = self.counter1 +1
            if self.counter1==self.args_batchsize or self.counter1==(self.args_batchsize*2) or self.counter1>((self.args_batchsize*3)-1) :
                self.id_sal = []
            if (self.counter1)<self.args_batchsize:
                i = np.random.randint(0, len(self.unique_index1))
                self.counter = i
                self.counter_.append(i)
            elif (self.counter1)>=self.args_batchsize and (self.counter1)<(self.args_batchsize*2):
                i = self.counter_[(self.counter1)-(self.args_batchsize)]
            
            elif (self.counter1)>=(self.args_batchsize*2) and (self.counter1)<(self.args_batchsize*3):
                i = np.random.randint(0, len(self.unique_index1))
                while(i==self.counter_[self.counter1-(self.args_batchsize*2)]):
                    i = np.random.randint(0, len(self.unique_index1))
                if (self.counter1)==((self.args_batchsize*3)-1):
                    self.counter1 = -1
                    self.counter_=[]

            
            if(self.unique_index[i][1]==self.unique_index[i][2]):
                i1 = self.unique_index[i][1]
                self.id_sal.append(i1)
                    
            else:
                i1 = np.random.randint(self.unique_index[i][1],self.unique_index[i][2])
                """if self.counter1<=7 and self.counter1 > -1:
                    self.id_sal.append(i1)
                if self.counter1>=8 and self.counter1 < 16:
                    self.id_sal.append(i1)
                if (self.counter1>=16 and self.counter1 < 24) or self.counter1==-1:
                    self.id_sal.append(i1)"""
                self.id_sal.append(i1)
                        
            

            idx=i1
                
                

        """code to get saliency maps correponding to the same images chosen for source image, positive and negative anchors
            
            Needed only if using saliency maps, can ignore if original images used
            """
        if self.flag==1 and self.id_sal != [] and self.counter_sal<((self.args_batchsize*2)-1):
            self.counter_sal = self.counter_sal +1
            idx = self.id_sal[(self.counter_sal)%self.args_batchsize]

            if (self.counter_sal)==((self.args_batchsize)-1):
                self.id_sal = []
            elif (self.counter_sal)==((self.args_batchsize*2)-1):
                self.id_sal = []
        if self.flag==1 and self.id_sal != [] and self.counter_sal>=((self.args_batchsize*2)-1):
            self.counter_sal = self.counter_sal +1
            idx = self.id_sal[(self.counter_sal)%self.args_batchsize]
            
            if (self.counter_sal)==((self.args_batchsize*3)-1):
                self.counter_sal = -1
                self.id_sal = []
            
        """Test the code for 50 id's in arr1 for checking small changes in model quickly"""
        """arr1=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]
        
        if(self.counter1)==-1:
            arr = np.random.choice(arr1,self.args_batchsize)
            self.arr = arr
        arr=self.arr
        #print("ARR",arr)
        if self.mode =='source':
            self.counter1 = self.counter1 +1
            if self.counter1 <=self.args_batchsize-1:
                i = arr[self.counter1]
            if self.counter1 >= self.args_batchsize and (self.counter1)<(self.args_batchsize*2):
                i = arr[(self.counter1)-self.args_batchsize]
                
            elif (self.counter1)>=(self.args_batchsize*2) and (self.counter1)<(self.args_batchsize*3):
                i = np.random.randint(0, self.args_batchsize)
                i = arr[i]
                while(i==arr[self.counter1-(self.args_batchsize*2)]):
                    i = np.random.randint(0, self.args_batchsize)
                    i = arr[i]
                if (self.counter1)==((self.args_batchsize*2)-1):
                    self.counter1 = -1
                    
    
            if(self.unique_index[i][1]==self.unique_index[i][2]):
                    i1 = self.unique_index[i][1]
            else:
                i1 = np.random.randint(self.unique_index[i][1],self.unique_index[i][2])
            idx=i1
        
        if self.mode=='query' or self.mode=='test':
            if idx > (self.unique_index[49][2]):
                i = np.random.randint(0,50)
            
                if(self.unique_index[i][1]==self.unique_index[i][2]):
                    i1 = self.unique_index[i][1]
                else:
                    i1 = np.random.randint(self.unique_index[i][1],self.unique_index[i][2])
            else:
                i1 = idx

            idx=i1
            """
                
        idx = int(idx)
        input_image = self.get_image(idx)
        
        if self.transform is not None:
            input_image = self.transform(input_image)
    
        datapack = {'image': input_image}
        if self.mode =='train':
            datapack['label'] = -1
        else:
            label = self.get_label(idx)
            datapack['label'] = label
        datapack['camera_id'] = self.image_camera_ids
        datapack['idxx'] = self.id_sal

        return datapack
    
    def get_csv(self):
        csv_name = '{}_list.csv'.format(self.mode)
        csv_f = os.path.join(self.csv_path, csv_name)
        return pd.read_csv(csv_f)

    def get_image(self, idx):
        dataset_path = config.get_dataset_path(self.dataset_name)
        image_path = os.path.join(dataset_path, self.image_names[idx])
        while (image_path)== "":
            image_path = os.path.join(dataset_path, self.image_names[idx])

        try:
            image = Image.open(image_path).convert('RGB') # open the image file
            image.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            print('Bad file:', image_path)
        
        image = Image.open(image_path).convert('RGB')
        
        return image

    def get_label(self, idx):
        label = self.image_labels[idx]
        label = Variable(torch.tensor(label, dtype=torch.int64))
        return label



    def get_unique_index(self):
        """to get a list containing the unique 1500 id's in our csv file having paths to images in dataset, the list has 3 columns, 1 for the unique label, the other 2 containing starting and ending index of that label in the csv file"""
        img = self.image_labels
        unique_list = []
        img_ = np.array(img)
        # traverse for all elements
        for x in img:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
    
        array = np.zeros(shape=(len(unique_list),3))
        for i in range (len(unique_list)):
            index = np.where(img==unique_list[i])
            index = np.array(index)
            array[i,1] = index.item(0)
            array[i,2] = index.item(-1)

        array[:,0] = unique_list

        
        return array, unique_list

