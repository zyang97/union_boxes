import os
import torch
import torch.nn as nn
import modules.netUtils as netUtils
import modules.primitives as primitives
from torch.autograd import Variable
from modules.cuboid import  CuboidSurface
import modules.meshUtils as mUtils

from rec.net import Encoder
from rec.dataloader import get_data_models, fetch_batch_joint

params = lambda x: 0

# model
params.nSamplesChamfer = 150  # number of points we'll sample per part
params.nz = 3
params.shapeLrDecay = 0.01
params.primTypes = ['Cu']
params.probLrDecay = 0.0001
params.gridBound = 0.5
params.nParts = 6
params.chamferLossWt = 1

# data
params.data_dir = 'D:\\data\\images\\data'
params.num_views = 10
params.batch_size = 32
params.category = 'aero'
params.num_points = 1024

# trainer
params.val_iter = 10
params.name = 'aero_img_pretrain_rc'
params.batch_size = 32
params.usePretrain = True
params.pretrainNet = 'aero_img_pretrain_rc'
params.pretrainIter = 100

params.visMeshesDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\results\\visualization\\meshes', params.name)

if not os.path.exists(params.visMeshesDir):
    os.makedirs(params.visMeshesDir)

cuboid_sampler = CuboidSurface(params.nSamplesChamfer, normFactor='Surf')

class Network2(nn.Module):
    def __init__(self, params):
        super(Network2, self).__init__()
        self.backbone = Encoder(params.num_points)
        outChannels = self.outChannels = 256
        layers = []
        for i in range(2):
            layers.append(nn.Conv3d(outChannels, outChannels, kernel_size=1))
            layers.append(nn.BatchNorm3d(outChannels))
            layers.append(nn.LeakyReLU(0.2, True))

        self.fc_layers = nn.Sequential(*layers)
        self.fc_layers.apply(netUtils.weightsInit)
        biasTerms = lambda x: 0

        biasTerms.quat = torch.Tensor([1, 0, 0, 0])
        biasTerms.shape = torch.Tensor(params.nz).fill_(-3) / params.shapeLrDecay
        biasTerms.prob = torch.Tensor(len(params.primTypes)).fill_(0)
        for p in range(len(params.primTypes)):
            if (params.primTypes[p] == 'Cu'):
                biasTerms.prob[p] = 2.5 / params.probLrDecay

        self.primitivesTable = primitives.Primitives(params, outChannels, biasTerms)

    def forward(self, x):
        # input: x (bs, 1, 32, 32,32)
        encoding = self.backbone(x) # encoding (bs, 16, 1, 1, 1)
        for i in range(3):
            encoding = encoding.unsqueeze(-1)
        features = self.fc_layers(encoding) # features (bs, 16, 1, 1, 1)
        primitives = self.primitivesTable(features) # primitives (bs, nParts * 10)
        return primitives

netPred = Network2(params).cuda()

if params.usePretrain:
    # updateShapeWtFunc = netUtils.scaleWeightsFunc(params.pretrainLrShape / params.shapeLrDecay, 'shapePred')
    # updateProbWtFunc = netUtils.scaleWeightsFunc(params.pretrainLrProb / params.probLrDecay, 'probPred')
    # updateBiasWtFunc = netUtils.scaleBiasWeights(params.probLrDecay, 'probPred')
    load_path = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\cachedir\\snapshots', params.pretrainNet, 'iter{}.pkl'.format(params.pretrainIter))
    netPretrain = torch.load(load_path)
    netPred.load_state_dict(netPretrain)
    print('Loading pretrained model from {}'.format(load_path))
    # netPred.primitivesTable.apply(updateShapeWtFunc)
    # netPred.primitivesTable.apply(updateProbWtFunc)
    # netPred.primitivesTable.apply(updateBiasWtFunc)

test_models, test_pair_indices = get_data_models(params.data_dir, params.category, params.num_views, 'test')

batches = len(test_pair_indices) // params.batch_size
# batches = 10

k = 0

if __name__ == '__main__':
    for b in range(batches):
        netPred.eval()
        batch_ip, batch_gt, batch_name = fetch_batch_joint(test_models, test_pair_indices, b, params.batch_size)
        batch_ip = Variable(torch.tensor(batch_ip, dtype=torch.float32).cuda())
        batch_gt = Variable(torch.tensor(batch_gt, dtype=torch.float32).cuda())
        shapePredParams = netPred.forward(batch_ip)
        shapePredParams = shapePredParams.view(params.batch_size, params.nParts, 10)
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





