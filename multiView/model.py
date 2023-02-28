import os
import torch
from torch import nn
from rec.net import Encoder
import modules.netUtils as netUtils
import modules.primitives as primitives
import numpy as np

class InterNetwork(nn.Module):
    def __init__(self, params):
        super(InterNetwork, self).__init__()
        self.num_points = params.num_points
        self.batch_size = params.batch_size
        self.fc = nn.Linear(256, self.num_points * 3)
        self.bn = nn.BatchNorm1d(self.num_points * 3)

    def forward(self, x):
        inter_pcl = self.bn(self.fc(x))
        inter_pcl = inter_pcl.view(self.batch_size, self.num_points, 3)
        return inter_pcl



class MultiViewNetwork(nn.Module):
    def __init__(self, params):
        super(MultiViewNetwork, self).__init__()
        self.backbone = Encoder(params.num_points)
        self.inter_net = InterNetwork(params)

        # loead pretrained part
        if params.model != 'test':
            pretrained_dict = torch.load(params.encoder_pretrain_dir)
            model_dict = self.backbone.state_dict()
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.backbone.load_state_dict(model_dict)

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

    def forward(self, imgs_batch):
        # input: x (bs, 10, 64, 64, 3)
        imgs_batch = imgs_batch.transpose(0, 1)
        num_views = imgs_batch.size(0)
        encoding = 0
        inter_output = 0
        for imgs in imgs_batch:
            x, pcl = self.backbone(imgs)
            encoding += x
            inter_output += pcl

        encoding /= num_views
        inter_output /= num_views
        #inter_output = self.inter_net(encoding)

        for i in range(3):
            encoding = encoding.unsqueeze(-1)
        features = self.fc_layers(encoding)  # features (bs, 16, 1, 1, 1)
        primitives = self.primitivesTable(features)  # primitives (bs, nParts * 10)
        return primitives, inter_output

if __name__ == '__main__':
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
    params.category = 'chair'
    params.num_points = 1024

    # trainer
    params.learning_rate = 0.0005
    params.num_train_iter = 200
    params.val_iter = 10
    params.name = 'chair_multi_view_1'
    params.batchSizeVis = 4
    params.encoder_pretrain_dir = 'D:\\projects\\experiment\\volumetricPrimitivesPytorch\\rec\\cachedir\\snapshots\\chair\\iter150.pkl'

    params.visMeshesDir = os.path.join('/multiView\\cachedir\\visualization\\meshes', params.name)
    params.snapshotDir = os.path.join('/cachedir/snapshots', params.name)

    model = MultiViewNetwork(params).cuda()

    data = torch.rand([4, 10, 64, 64, 3]).cuda()
    pred = model.forward(data)

    print(pred.size())