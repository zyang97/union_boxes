import os
import torch
import time
from torch.autograd import Variable
from modules.losses import chamfer_loss_img, emd_loss
from modules.cuboid import CuboidSurface
import modules.meshUtils as mUtils
from multiView.model import MultiViewNetwork

from rec.dataloader import ShapeNetMultiViewDataset

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
params.num_views = 10
params.batch_size = 1
params.num_points = 1024

params.category = 'bench'
params.name = params.category + '_multi_view_1'
params.pretrainNet = 'bench_multi_view_1'
params.pretrainIter = 499
params.pretrainDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\multiView\\cachedir\\snapshots')
params.visMeshesDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\multiView\\results\\visualization\\tests', params.name)
params.infer = 'ca85baab8740ffa198cd58ee93c42c40'

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
    for b in range(batches):
        print('Processed object {}'.format(k))
        netPred.eval()
        batch_ip, batch_gt, batch_name = test_dataloader[b]
        if batch_name[0] != params.infer:
            continue
        batch_ip = Variable(torch.tensor(batch_ip, dtype=torch.float32).cuda())
        batch_gt = Variable(torch.tensor(batch_gt, dtype=torch.float32).cuda())

        s = time.time()
        shapePredParams, _ = netPred.forward(batch_ip)
        curr_time = 1/(time.time() - s)
        print('Infer time is {}FPS'.format(curr_time))

        shapePredParams = shapePredParams.view(params.batch_size, params.nParts, 10)
        chamfer += chamfer_loss_img(shapePredParams, cuboid_sampler, batch_gt)
        emd += emd_loss(shapePredParams, cuboid_sampler, batch_gt)
        netPred.train()

        predParams = shapePredParams
        for idx in range(0, params.batch_size):
            print('Save batch {} object {}'.format(b, idx))
            pred_b = []
            for px in range(params.nParts):
                pred_b.append(predParams[idx, px, :].clone().data.cpu())

            mUtils.saveParts(pred_b,
                             '{}/pred_{}_{}.obj'.format(params.visMeshesDir, batch_name[idx], k%10))

            k += 1
            print('Chamfer mean is {}'.format(chamfer/k))
            print('EMD mean is {}'.format(emd / k))