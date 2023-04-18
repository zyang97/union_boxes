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
params.model = 'test'
params.nSamplesChamfer = 150  # number of points we'll sample per part
params.nz = 3
params.shapeLrDecay = 0.01
params.primTypes = ['Cu']
params.probLrDecay = 0.0001
params.gridBound = 0.5
params.nParts = 10
params.nSamplesChamfer = 150
params.chamferLossWt = 1
params.usePretrain = True

# data
params.data_dir = 'D:\\data\\images\\data'
params.num_views = 1
params.batch_size = 1
params.num_points = 1024

params.category = 'chair'
params.name = params.category + '_multi_view_p10_n1'
params.pretrainNet = params.category + '_multi_view_p10_n10'
params.pretrainIter = 499
params.pretrainDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\multiView\\cachedir\\snapshots')

params.visMeshesDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\multiView\\results\\visualization\\tests', params.name)

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
                             '{}/pred_{}.obj'.format(params.visMeshesDir, batch_name[idx]))

            k += 1
            print('Chamfer mean is {}'.format(chamfer/k))
            print('EMD mean is {}'.format(emd / k))


def print_all_codes(n, m):

    def print_01_codes(current, num_digits):
        if num_digits == 0:
            print(current)
        else:
            print_01_codes('0' + current, num_digits - 1)
            print_01_codes('1' + current, num_digits - 1)

    upper_bound = 0
    while True:
        for i in range(upper_bound):
            print_01_codes('', n)
        if upper_bound > m:
            break
        upper_bound += 1
    print(upper_bound)

print_all_codes(3,5)

def print_01_codes(current, num_digits):
    if num_digits == 0:
        print(current)
    else:
        print_01_codes('0' + current, num_digits - 1)
        print_01_codes('1' + current, num_digits - 1)

print_01_codes("", 4)