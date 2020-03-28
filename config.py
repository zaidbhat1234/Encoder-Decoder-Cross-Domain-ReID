import os


"""
    Normalisation and dimensional parameters as mentioned on the pytorch website to be used for the model ResNet
    """
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

MEAN = [0.485, 0.456, 0.406] # ImageNet Pre-trained Mean
STD = [0.229, 0.224, 0.225] # ImageNet Pre-trained STDDEV

#MEAN = [0.5, 0.5, 0.5]
#STD = [0.5, 0.5, 0.5]

MODE = 'train'

if MODE == 'train':
    # Train class number
    DUKE_CLASS_NUM = 702
    MARKET_CLASS_NUM = 751
    MSMT_CLASS_NUM = 1041
    CUHK_CLASS_NUM = 1367

elif MODE == 'all':
    # ALL class number
    DUKE_CLASS_NUM = 1812
    MARKET_CLASS_NUM = 1501
    MSMT_CLASS_NUM = 4101
    CUHK_CLASS_NUM = 1467

CURRENT_DIR = '/media/zaid/dataset/dataset'

DATASET_DIR = '/media/zaid/dataset/dataset'
DATASET_DIR_CUHK = '/media/zaid/dataset/dataset/CUHK03/detected'

GLOBAL_MARGIN = 0.5 #For triplet loss


def get_dataset_path(dataset_name):
    if dataset_name=='CUHK03':
        return os.path.join(DATASET_DIR_CUHK)
    else:
        return os.path.join(DATASET_DIR, dataset_name)



def get_csv_path(dataset_name):
    csv_dir = os.path.join(CURRENT_DIR,dataset_name ,'data/csv')
    return os.path.join(csv_dir, dataset_name)

def get_dataoutput(source_dataset):
    classifier_output_dim = -1
    if source_dataset == 'Duke':
        classifier_output_dim = DUKE_CLASS_NUM
    elif source_dataset == 'Market':
        classifier_output_dim = MARKET_CLASS_NUM
    elif source_dataset == 'MSMT':
        classifier_output_dim = MSMT_CLASS_NUM
    elif source_dataset == 'CUHK':
        classifier_output_dim = CUHK_CLASS_NUM
    return classifier_output_dim

