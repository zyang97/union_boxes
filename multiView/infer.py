import os
import torch
import time
from torch.autograd import Variable
from modules.losses import chamfer_loss_img, emd_loss
from modules.cuboid import CuboidSurface
import modules.meshUtils as mUtils
from multiView.model import MultiViewNetwork

from rec.dataloader import ShapeNetMultiViewDataset

import cv2

params = lambda x: 0

# model
params.nSamplesChamfer = 150  # number of points we'll sample per part
params.nz = 3
params.shapeLrDecay = 0.01
params.primTypes = ['Cu']
params.probLrDecay = 0.0001
params.gridBound = 0.5
params.nParts = 10
params.chamferLossWt = 1
params.usePretrain = True

# data
params.data_dir = 'D:\\data\\images\\data'
params.num_views = 1
params.batch_size = 1
params.num_points = 1024
params.model = 'test'

params.category = 'chair'
params.name = params.category + '_multi_view_p10_n10'
params.pretrainNet = params.category + '_multi_view_p10_n10'
params.pretrainIter = 499
params.pretrainDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\multiView\\cachedir\\snapshots')
params.visMeshesDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\multiView\\results\\visualization\\infer', params.name)
params.imgs_path = 'D:\\projects\\experiment\\volumetricPrimitivesPytorch\\third_party\\pix3d\\img\\chair'

if not os.path.exists(params.visMeshesDir):
    os.makedirs(params.visMeshesDir)

cuboid_sampler = CuboidSurface(params.nSamplesChamfer, normFactor='Surf')

netPred = MultiViewNetwork(params).cuda()

if params.usePretrain:
    load_path = os.path.join(params.pretrainDir, params.pretrainNet, 'iter{}.pkl'.format(params.pretrainIter))
    netPretrain = torch.load(load_path)
    netPred.load_state_dict(netPretrain)
    print('Loading pretrained model from {}'.format(load_path))

test_dataloader = ShapeNetMultiViewDataset(params, 'val')

batches = len(test_dataloader) // params.batch_size
# batches = 10

k = 0
chamfer = 0
emd = 0

if __name__ == '__main__':
    filenames = os.listdir(params.imgs_path)
    for k, filename in enumerate(filenames):
        if filename != '1314.jpg':
            continue
        print('Processed object {}'.format(k))
        img_path = os.path.join(params.imgs_path, filename)
        ip_image = cv2.imread(img_path)
        ip_image = cv2.resize(ip_image, (64, 64))
        ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)
        for w in range(64):
            for h in range(64):
                # ip_image[w][h][0] = 255 - ip_image[w][h][0]
                # ip_image[w][h][1] = 255 - ip_image[w][h][1]
                # ip_image[w][h][2] = 255 - ip_image[w][h][2]
                if ip_image[w][h][0] == 255 and ip_image[w][h][1] == 255 and ip_image[w][h][2] == 255:
                    ip_image[w][h][0] = 0
                    ip_image[w][h][1] = 0
                    ip_image[w][h][2] = 0
        # cv2.imshow('image', ip_image)
        # cv2.waitKey(0)
        ip_image = torch.Tensor(ip_image).cuda().unsqueeze(0).unsqueeze(0)
        netPred.eval()
        shapePredParams, _ = netPred.forward(ip_image)
        shapePredParams = shapePredParams.view(params.batch_size, params.nParts, 10)
        netPred.train()
        predParams = shapePredParams

        pred_b = []
        for px in range(params.nParts):
            pred_b.append(predParams[0, px, :].clone().data.cpu())
        mUtils.saveParts(pred_b, '{}/pred_{}.obj'.format(params.visMeshesDir, filename.split('.')[0]))