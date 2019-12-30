import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as trans
from reid_dataset import AdaptReID_Dataset
from reid_loss import *

def extract_feature(model, dataloader): #finds the features of each batch and stores them in a list of lists in features and infos.
    features, infos = [], []
    for batch in dataloader:
        image, label, camera_id = split_datapack(batch)
        feature = model.encoder_base_s(image)
        feature = model.encoder_s(feature, use_avg=True)                                # replace the 2 with one?
        feature = feature.view(feature.size()[0], -1).data.cpu().numpy()
        for i in range(feature.shape[0]):
            features.append(feature[i])
            infos.append((label[i].cpu().numpy(), camera_id[i].cpu().numpy()))
        
    return features, infos

#checks if the matched features lie in match, if yes then rank of that element is 1 else 0, check for each element in the matrix by having nested for.
def rank_func(query_len, test_len, match, junk, matrix_argsort, use_rank):
    CMC = np.zeros([query_len, test_len], dtype=np.float32)    
    for q_idx in range(query_len):
        counter = 0
        for t_idx in range(test_len):
            if matrix_argsort[q_idx][t_idx] in junk[q_idx]: #matrix argsort elements are the id's of sorted elements so compare each of the id of sorted, if in junk id then not include
                continue
            else:
                counter += 1 # used as index memory to store value in the CMC matrix.
                if matrix_argsort[q_idx][t_idx] in match[q_idx]:
                    CMC[q_idx, counter-1:] = 1.0
            if counter == use_rank: # for rank 1 as soon as the minimum first image of a particular id break for that id, rank 5 takes 5 top min dist images of a particualr id
                    break
    rank_score = np.mean(CMC[:,0])
    return rank_score

def dist_func(metric, query_features, test_features): #has 2 lists of lists' having batchwise features and it calculates distance of every i'th feature vector of 1 list with j'th of other
    if metric == 'L1':
        metric = 'hamming'
    elif metric == 'L2':
        metric = 'euclidean'
    # Ma x n dimension with Mb x n dimension input of 2 lists , output Ma x Mb, n is the dimension of the batch of each feature element of lists.
    matrix = cdist(query_features, test_features, metric=metric)
    return matrix
    
def get_match(query_infos, test_infos):
    match, junk = [], []
    for (query_label, query_camera_id) in query_infos:
        tmp_match, tmp_junk = [], []
        for idx, (test_label, test_camera_id) in enumerate(test_infos):
            if test_label == query_label and query_camera_id != test_camera_id: #compare each query image with every test image and stores id of test image which has same label as query but different camera id.
                #print("Q: ", query_camera_id, "T : " , test_camera_id)
                tmp_match.append(idx)
            elif test_label == query_label or test_label < 0:
                #print("JUNK:")
                tmp_junk.append(idx)
        #stores the matched id's from test of each query image in tmp array and appends that list in the list of lists.
        match.append(tmp_match)
        junk.append(tmp_junk)
    return match, junk

def evaluate(args, model, transform_list, match, junk):
    print("Evaluating...")
    model.eval()
    testData = AdaptReID_Dataset(dataset_name=args.target_dataset, mode='test', transform=trans.Compose(transform_list))
    queryData = AdaptReID_Dataset(dataset_name=args.target_dataset, mode='query', transform=trans.Compose(transform_list))
    testDataloader = DataLoader(testData, batch_size=args.batch_size, shuffle=False)
    queryDataloader = DataLoader(queryData, batch_size=args.batch_size, shuffle=False)
    test_features, test_infos = extract_feature(model, testDataloader)
    query_features, query_infos = extract_feature(model, queryDataloader)

    if match == None:
        match, junk = get_match(query_infos, test_infos)
    dist_matrix = dist_func(args.dist_metric, query_features, test_features)
    matrix_argsort = np.argsort(dist_matrix, axis=1)  #returns indices of the sorted list of distances.
    rank_score = rank_func(len(query_features), len(test_features), match, junk, matrix_argsort, args.rank)
    print('Evaluation: Rank score: {}'.format(rank_score))
    return rank_score, match, junk
