import os
import numpy as np
import warnings
import pickle

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch import nn
import cv2

from itertools import product

shapenet_category_to_id = {'chair':'03001627', 'aero':'02691156', 'car':'02958343'}
PNG_FILES = ['render_0.png', 'render_1.png', 'render_2.png', 'render_3.png', 'render_4.png', 'render_5.png', 'render_6.png', 'render_7.png', 'render_8.png', 'render_9.png']
#PNG_FILES = ['render_0.png', 'render_1.png', 'render_2.png', 'render_3.png', 'render_4.png']

pcl_filename = 'pointcloud_1024.npy'

class ShapeNetMultiViewDataset(Dataset):
	def __init__(self, args, split='train'):
		self.data_dir = args.data_dir
		self.batch_size = args.batch_size
		self.category = args.category
		self.category = args.category

		self.models = []

		self.category_id = shapenet_category_to_id[self.category]
		splits_file_path = os.path.join(self.data_dir, 'splits', self.category_id + '_%s_list.txt' % split)

		with open(splits_file_path, 'r') as f:
			for model in f.readlines():
				self.models.append(os.path.join(self.data_dir, self.category_id, model.strip()))

	def __len__(self):
		return len(self.models)

	def __getitem__(self, idx):
		imgs = []
		pcl_gt = []
		filenames = []
		for i in range(self.batch_size*idx, self.batch_size*idx + self.batch_size):
			li = []
			for filename in PNG_FILES:
				img_path = os.path.join(self.models[i], filename)
				ip_image = cv2.imread(img_path)
				ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)
				li.append(ip_image)
			imgs.append(li)
			pcl_path = os.path.join(self.models[i], pcl_filename)
			pcl_gt.append(np.load(pcl_path))
			filenames.append(os.path.basename(self.models[i]))
		return torch.Tensor(np.array(imgs)).cuda(), torch.Tensor(np.array(pcl_gt)).cuda(), filenames



def fetch_pcl(model_path):
	pcl_filename = 'pointcloud_1024.npy'
	pcl_path = os.path.join(model_path, pcl_filename)
	pcl_gt = np.load(pcl_path)
	return pcl_gt


def fetch_image(model_path, index):
	img_path = os.path.join(model_path, PNG_FILES[index])
	ip_image = cv2.imread(img_path)
	ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)
	return ip_image


def fetch_labels(model_path):
	label_path = os.path.join(model_path, 'pointcloud_labels.npy')
	labels_gt = np.load(label_path)
	labels_gt -= 1
	return labels_gt


def get_label_wts(label):
	'''
	Computes weight for every point based on class count of that point
	Args:
		label: class labels for each point in a pcl --> (NUM_POINTS)
	Returns:
		wts: class weights for each point in a pcl --> (NUM_POINTS)
	'''
	cnt = np.bincount(label)
	tot_cnt = np.sum(cnt)
	classes = np.nonzero(cnt)[0]
	wts = []
	cnt_dict = dict(zip(classes,cnt[classes]))
	for lbl in label:
		wts.append(tot_cnt/cnt_dict[lbl])
	wts = np.asarray(wts, dtype=np.float32)
	return wts


def fetch_batch_joint(models, indices, batch_num, batch_size):
	batch_ip = []
	batch_gt = []
	batch_name = []
	for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:
		model_path = models[ind[0]]
		pcl_gt = fetch_pcl(model_path)
		ip_image = fetch_image(model_path, ind[1])
		batch_ip.append(ip_image)
		batch_gt.append(pcl_gt)
		batch_name.append(os.path.basename(model_path))
	batch_ip = np.array(batch_ip)
	batch_gt = np.array(batch_gt)
	return batch_ip, batch_gt, batch_name

def get_data_util(data_dir, category, eval_set):

    models = []

    if category == 'all':
        cats = ['chair', 'car', 'aero']
    else:
        cats = [category]

    for cat in cats:
        category_id = shapenet_category_to_id[cat]
        splits_file_path = os.path.join(data_dir, 'splits', category_id + '_%s_list.txt' % eval_set)

        with open(splits_file_path, 'r') as f:
            for model in f.readlines():
                models.append(os.path.join(data_dir, category_id, model.strip()))

    return models

def get_data_models(data_dir, category, NUM_VIEWS, eval_set):
	models = get_data_util(data_dir, category, eval_set)
	pair_indices = list(product(range(len(models)), range(NUM_VIEWS)))
	print('{}: models={}  samples={}'.format(eval_set, len(models),len(models)*NUM_VIEWS))
	return models, pair_indices

if __name__ == '__main__':
    # data_dir = 'D:\\data\\images\\data'
    # num_views = 10
    # batch_size = 32
    # models, pair_indices = get_data_models(data_dir, 'chair', num_views, 'train')
    # batches = len(pair_indices) // batch_size
    # for b in range(batches):
    #     batch_ip, batch_gt, batch_lbl, batch_lbl_wts = fetch_batch_joint(models, pair_indices, b, batch_size)
    # print(models)

	# test Dataset model
	params = lambda x: 0
	params.data_dir = 'D:\\data\\images\\data'
	params.batch_size = 8
	params.category = 'chair'
	dataset = ShapeNetMultiViewDataset(params)
	imgs, gts = dataset[0]
	print(imgs.size())
