import math
import sys
import operator
import networkx as nx
import numpy as np
import scipy.spatial.distance
import skimage
import skimage.io
from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy.optimize import minimize
import cv2
import os
import scipy.misc


############################################################


def S(x1, x2, geodesic, sigma_clr=10):
    return math.exp(-pow(geodesic[x1,x2],2)/(2*sigma_clr*sigma_clr))

    
############################################################


def compute_saliency_cost(smoothness, w_bg, wCtr):
    n = len(w_bg)
    A = np.zeros((n,n))
    b = np.zeros((n))

    for x in range(0,n):
        A[x,x] = 2 * w_bg[x] + 2 * (wCtr[x])
        b[x] = 2 * wCtr[x]
        for y in range(0,n):
            A[x,x] += 2 * smoothness[x,y]
            A[x,y] -= 2 * smoothness[x,y]

    x = np.linalg.solve(A, b)
    return x

    
############################################################

    
def path_length(path, G):
    dist = 0.0
    for i in range(1,len(path)):
        dist += G[path[i - 1]][path[i]]['weight']
    return dist


############################################################


def make_graph(grid, num_segments):

    # get unique labels
    vertices = np.arange(num_segments)

    # create edges
    down = np.c_[grid[:-1, :].ravel(), grid[1:, :].ravel()]
    # print (down.shape)
    right = np.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()]
    # print (right.shape)
    all_edges = np.vstack([right, down])
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]        # takes care of same node edges - self loop/intra connections
    all_edges = np.sort(all_edges, axis=1)
    num_vertices = len(vertices)
    edge_hash = all_edges[:,0] + num_vertices * all_edges[:, 1]         # takes care of duplicate edges - edges between same components
    # find unique connections
    edges = np.unique(edge_hash)
    # undo hashing
    edges = [[vertices[x%num_vertices], vertices[int(x/num_vertices)]] for x in edges] 
    # print (len(edges))
    # print (edges)
    return vertices, edges

    
############################################################


def binarise_saliency_map(saliency_map):

    adaptive_threshold = 0.5 * saliency_map.mean()
    return (saliency_map > adaptive_threshold)


############################################################


def dilation(image):

	m, n = image.shape
	image_pad = np.pad(image, (1,1), 'edge')

	#cv2.imwrite("padded.png", 255 - image_pad.astype(np.uint8)*255)

	kernel = np.ones((3, 3))
	kernel[0][0], kernel[2][0], kernel[0][2], kernel[2][2] = 0, 0, 0, 0

	for i in range(1,m+1):
		for j in range(1,n+1):
			region = image_pad[i-1:i+2, j-1:j+2] * kernel
			if region[0][1] == 1 or region[1][0] == 1 or region[1][2] == 1 or region[2][1] == 1:
				image[i-1][j-1] = 1

	return image


def erosion(image):

	m, n = image.shape
	image_pad = np.pad(image, (1,1), 'edge')

	kernel = np.ones((3, 3))
	kernel[0][0], kernel[2][0], kernel[0][2], kernel[2][2] = 0, 0, 0, 0

	for i in range(1,m+1):
		for j in range(1,n+1):
			region = image_pad[i-1:i+2, j-1:j+2] * kernel
			if region[0][1] == 1 and region[1][0] == 1 and region[1][2] == 1 and region[2][1] == 1:
				image[i-1][j-1] = 1
			else:
				image[i-1][j-1] = 0

	return image
    
    
############################################################


def get_saliency(img_path):

    img = skimage.io.imread(img_path)

    img_lab = img_as_float(skimage.color.rgb2lab(img))
    img_rgb = img_as_float(img)
    img_gray = img_as_float(skimage.color.rgb2gray(img))

    segments_slic = slic(img_rgb, n_segments=250, compactness=10, sigma=1, enforce_connectivity=False)

    num_segments = np.max(segments_slic) + 1

    nrows, ncols = segments_slic.shape
    max_dist = math.sqrt(nrows*nrows + ncols*ncols)

    grid = segments_slic

    (vertices, edges) = make_graph(grid, num_segments)

    gridx, gridy = np.mgrid[:grid.shape[0], :grid.shape[1]]     # counting rows, counting columns

    mean_centers = dict()
    mean_colors = dict()
    distances = dict()
    boundary = dict()

    # vertices -> array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, ......,num_segments-1])
    for v in vertices:
        mean_centers[v] = [gridy[grid == v].mean(), gridx[grid == v].mean()]        # mean position of segment 'v'
        mean_colors[v] = np.mean(img_lab[grid == v],axis=0)         # mean color of segment 'v'

        # these give the coordinates of the segment 'v''s pixels
        x_pix = gridx[grid == v]        # row numbers containing segment 'v'
        y_pix = gridy[grid == v]        # col numbers containing segment 'v'

        # detect if segment lies on boundary
        if np.any(x_pix == 0) or np.any(y_pix == 0) or np.any(x_pix == nrows - 1) or np.any(y_pix == ncols - 1):
            boundary[v] = 1
        else:
            boundary[v] = 0

    G = nx.Graph()

    #buid the graph
    for edge in edges:
        pt1 = edge[0]
        pt2 = edge[1]
        color_distance = scipy.spatial.distance.euclidean(mean_colors[pt1],mean_colors[pt2])
        G.add_edge(pt1, pt2, weight=color_distance )

    #add a new edge in graph if edges are both on boundary
    for v1 in vertices:
        if boundary[v1] == 1:
            for v2 in vertices:
                if boundary[v2] == 1:
                    color_distance = scipy.spatial.distance.euclidean(mean_colors[v1],mean_colors[v2])
                    G.add_edge(v1,v2,weight=color_distance)

    geodesic = np.zeros((len(vertices),len(vertices)),dtype=float)
    spatial = np.zeros((len(vertices),len(vertices)),dtype=float)
    smoothness = np.zeros((len(vertices),len(vertices)),dtype=float)
    adjacency = np.zeros((len(vertices),len(vertices)),dtype=float)

    sigma_clr = 10.0
    sigma_bndcon = 1.0
    sigma_spa = 0.25
    mu = 0.1

    all_shortest_paths_color = nx.shortest_path(G,source=None,target=None,weight='weight')

    for v1 in vertices:
        for v2 in vertices:
            if v1 == v2:
                geodesic[v1,v2] = 0
                spatial[v1,v2] = 0
                smoothness[v1,v2] = 0
            else:
                geodesic[v1,v2] = path_length(all_shortest_paths_color[v1][v2],G)
                spatial[v1,v2] = scipy.spatial.distance.euclidean(mean_centers[v1],mean_centers[v2]) / max_dist
                smoothness[v1,v2] = math.exp( - (geodesic[v1,v2] * geodesic[v1,v2])/(2.0*sigma_clr*sigma_clr)) + mu 

    for edge in edges:
        pt1 = edge[0]
        pt2 = edge[1]
        adjacency[pt1,pt2] = 1
        adjacency[pt2,pt1] = 1

    for v1 in vertices:
        for v2 in vertices:
            smoothness[v1,v2] = adjacency[v1,v2] * smoothness[v1,v2]

    area = dict()
    len_bnd = dict()
    bnd_con = dict()
    w_bg = dict()
    ctr = dict()
    wCtr = dict()

    for v1 in vertices:
        area[v1] = 0
        len_bnd[v1] = 0
        ctr[v1] = 0
        for v2 in vertices:
            d_app = geodesic[v1,v2]
            d_spa = spatial[v1,v2]
            w_spa = math.exp(- ((d_spa)*(d_spa))/(2.0*sigma_spa*sigma_spa))
            area_i = S(v1,v2,geodesic)
            area[v1] += area_i
            len_bnd[v1] += area_i * boundary[v2]
            ctr[v1] += d_app * w_spa
        bnd_con[v1] = len_bnd[v1] / math.sqrt(area[v1])
        w_bg[v1] = 1.0 - math.exp(- (bnd_con[v1]*bnd_con[v1])/(2*sigma_bndcon*sigma_bndcon))

    for v1 in vertices:
        wCtr[v1] = 0
        for v2 in vertices:
            d_app = geodesic[v1,v2]
            d_spa = spatial[v1,v2]
            w_spa = math.exp(- (d_spa*d_spa)/(2.0*sigma_spa*sigma_spa))
            wCtr[v1] += d_app * w_spa *  w_bg[v2]

    min_value = min(wCtr.values())
    max_value = max(wCtr.values())

    minVal = [key for key, value in wCtr.items() if value == min_value]
    maxVal = [key for key, value in wCtr.items() if value == max_value]

    for v in vertices:
        wCtr[v] = (wCtr[v] - min_value)/(max_value - min_value)

    img_disp1 = img_gray.copy()
    img_disp2 = img_gray.copy()

    x = compute_saliency_cost(smoothness,w_bg,wCtr)

    for v in vertices:
        img_disp1[grid == v] = x[v]

    img_disp2 = img_disp1.copy()
    sal = np.zeros((img_disp1.shape[0],img_disp1.shape[1],3))

    sal = img_disp2
    sal_max = np.max(sal)
    sal_min = np.min(sal)
    sal = 255 * ((sal - sal_min) / (sal_max - sal_min))

    return sal

    
############################################################

    
# data_dir_path = 'P:\RBD\mpi\data'
# sal_save_path = 'P:\RBD\mpi\saliency'
# bin_save_path = 'P:\RBD\mpi\binary'

data_dir_path = '/media/zaid/dataset/dataset/Market/bounding_box_train'
sal_save_path = '/media/zaid/dataset/dataset/Market/saliency/'
bin_save_path = '/media/zaid/dataset/dataset/Market/binary/'
count = 0
"""
from filecmp import dircmp

def print_diff_files(dcmp):
     cnt = 0
     for name in dcmp.diff_files:
        print (name)
        cnt = cnt +1
     
     for sub_dcmp in dcmp.subdirs.values():
        print_diff_files(sub_dcmp)
     print(cnt)
dcmp = dircmp('/media/zaid/dataset/dataset/Market/bounding_box_train', '/media/zaid/dataset/dataset/Market/binary/')
print_diff_files(dcmp)
print(cnt)
"""

flag = 0
cnt = 0
filen = []
for file in os.listdir(data_dir_path):
    for file1 in os.listdir(bin_save_path):
        if file == file1:
            #print(file,file1)
            flag=1
    
    if flag != 1:
        filen.append(file)
        
        if file[-4:] == ".jpg": #or filename.endswith(".png"):
        
            filename = os.path.join(data_dir_path,file)
            print(filename)
        
        
        try:
            rbd = get_saliency(filename).astype('uint8')
            binary_sal = binarise_saliency_map(rbd)
            
            sal_path = sal_save_path  + file[:-4] + '.jpg'
            cv2.imwrite(sal_path, rbd)
            
            #openCV cannot display numpy type 0, so convert to uint8 and scale
            bin_path = bin_save_path  + file[:-4] + '.jpg'
            cv2.imwrite(bin_path, 255 * binary_sal.astype('uint8'))
        except:
            print ("Problem with image ", file)
        
        count += 1
        print (count, " files processed")
        
        cnt = cnt+1
    flag = 0
print(cnt,file)

"""

for file in os.listdir(data_dir_path):

    if file[-4:] == ".jpg": #or filename.endswith(".png"):

        filename = os.path.join(data_dir_path,file)
        

        try:
            rbd = get_saliency(filename).astype('uint8')

            binary_sal = binarise_saliency_map(rbd)
            
            sal_path = sal_save_path  + file[:-4] + '.jpg'
            cv2.imwrite(sal_path, rbd)
            
            #openCV cannot display numpy type 0, so convert to uint8 and scale
            bin_path = bin_save_path  + file[:-4] + '.jpg'
            cv2.imwrite(bin_path, 255 * binary_sal.astype('uint8'))
        except:
            print ("Problem with image ", file)
        
        count += 1
        print (count, " files processed")
"""
