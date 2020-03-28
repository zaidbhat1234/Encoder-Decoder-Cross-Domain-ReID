
import pandas as pd
import csv
import os
import numpy as np



"""
    #Prepare CSV file from a folder of images
data=[]
with open('/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/cuhk03-np/data/images.csv', 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    for filename in os.listdir("/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/cuhk03-np/detected/bounding_box_train"):
        data.append(filename)
        writer.writerow(data)
        data=[]
writeFile.close()
print("DONE:")
"""

#first above code to create image path , then add 'image' label in csv and run below code


"""
    #Add two columns for image label and camera id
csv_input = pd.read_csv('/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/cuhk03-np/data/images.csv')         #reading my csv file
csv_input['camera_id'] = csv_input['image'].str[:(7)].str[6]         #this would also copy
csv_input['label'] = csv_input['image'].str[:4]
csv_input.to_csv('/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/cuhk03-np/data/images.csv', index=False)
print("DONE")
"""


"""
    #Add complete path of the folder to images by adding prefix to image names
csv_input = pd.read_csv('/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/dataset/cuhk03-np/data/labeled_train.csv')
csv_input['image'] =  'bounding_box_train/' + csv_input['image']
csv_input.to_csv('/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/dataset/cuhk03-np/data/labeled_train.csv', index=False)
print("DONE")


f = open("/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/test/save2 copy.txt", "r")
img1 = f.read()
#img1 = float(img1)
img1 = img1.split()
img1 = float(img1)
print(len(img1))

values = [v for line in img1.split() for v in line.split(' ')]
bstr = ''.join(chr(int(float(v),16)) for v in values)
print(bstr)
"""

"""

from PIL import Image
values = [v for line in data.split() for v in line.split(':')]
bstr = ''.join(chr(int(v, 16)) for v in values)
im = Image.fromstring('L', (80, 60), bstr)
"""





"""
#drop -1 and 0 id'ed values, why??

csv_input = pd.read_csv('/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/dataset1/Market/data/csv/Market/test_list copy.csv')

print(len(csv_input))
#csv_input1 = pd.read_csv('/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/dataset1/Market/data/csv/Market/test_list1 copy.csv')
csv_input1 = csv_input.set_index("id")
csv_input1 = csv_input1.drop("-1",axis=0)
csv_input1 = csv_input1.drop("0",axis=0)

print(csv_input1)

csv_input1['camera_id'] = csv_input1['image_path'].str[24]
print(csv_input1)
#csv_input1['camera_id'] = csv_input1['image_path'].str[25]

#csv_input.to_csv('/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/dataset1/Market/data/csv/Market/test_list copy.csv', index=False)
csv_input1.to_csv('/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/dataset1/Market/data/csv/Market/test_list copy.csv')
"""



"""
#camera id


csv_input = pd.read_csv('/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/dataset1/Market/data/csv/Market/splits/35split1.csv')

csv_input['camera_id'] = csv_input['image_path'].str[25]

csv_input.to_csv('/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/dataset1/Market/data/csv/Market/splits/35split1.csv',index=False)


"""





"""

    #To split the train set into train and validation


import pandas as pd
import csv
import os

#Gets unique list of id's to ensure that validation has atleast 1 image from each unique id
def get_unique_index(img):
    unique_list = []
    img_ = np.array(img)
    # traverse for all elements
    for x in img:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    # for x in unique_list:
    # print("Unique:", x)

    array = np.zeros(shape=(len(unique_list), 3))
    for i in range(len(unique_list)):
        index = np.where(img == unique_list[i])
        index = np.array(index)
        array[i, 1] = index.item(0)
        array[i, 2] = index.item(-1)

    array[:, 0] = unique_list
    return array


csv_input = pd.read_csv('/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/dataset1/Market/data/csv/Market/splits/35split1.csv')
df= pd.DataFrame(csv_input)

csv_input1 = pd.read_csv('/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/dataset1/Market/data/csv/Market/splits/dataset_split.csv')
df1 = pd.DataFrame(csv_input1)

array = get_unique_index(csv_input['id'])


for i in range(len(array[:,0])):
    len = array[i][2]-array[i][1]+1
    counter=0
    while (counter < int((0.1*int(len))) or counter==0 ) and len>1: #0.1 is the split factor, for 80-20 split replace by 0.2

        x = i1 = np.random.randint(array[i][1],array[i][2])

        df1 = df1.append(df.loc[x,:],ignore_index=True,sort=False)
        #print(df1)

        #df.drop(x,inplace=True,axis=0)
        counter = counter+1

print(df.shape[0])
print(df1.shape[0])
df = pd.concat([df, df1, df1]).drop_duplicates(keep=False)

print(df.shape[0])
print(df1.shape[0])
df1.drop_duplicates(keep = 'first')


print(df1.shape[0])


keep_col = ['id','image_path','camera_id']
new_f = df[keep_col]
new_f1 = df1[keep_col]

new_f.to_csv('/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/dataset1/Market/data/csv/Market/splits/35split_25.csv',index=False)
new_f1.to_csv('/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/dataset1/Market/data/csv/Market/splits/35split_10.csv',index=False)

#df.to_csv('/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/dataset1/Market/data/csv/Market/splits/source_list_new.csv',index=False)
#df1.to_csv('/Users/zaidbhat/PycharmProjects/Encoder_Decoder/venv/lib/python3.6/zaid/dataset1/Market/data/csv/Market/splits/dataset_split1.csv',index=False)
print("HERE")

"""
