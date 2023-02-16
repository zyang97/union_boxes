import os
import torch
import torch.nn as nn
import modules.netUtils as netUtils
import modules.primitives as primitives
from torch.autograd import Variable
from modules.losses import get_loss
from modules.cuboid import  CuboidSurface
import modules.meshUtils as mUtils


from rec.net import Encoder
from rec.dataloader import get_data_models, fetch_batch_joint

from tensorboardX import SummaryWriter

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
params.learning_rate = 0.0005
params.num_train_iter = 200
params.val_iter = 10
params.name = 'aero_img_pretrain_rc'
params.batchSizeVis = 32
params.encoder_pretrain_dir = 'D:\\projects\\experiment\\volumetricPrimitivesPytorch\\rec\\cachedir\\snapshots\\aero\\iter150.pkl'

params.visMeshesDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\cachedir\\visualization\\meshes', params.name)
params.snapshotDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\cachedir\\snapshots', params.name)

if not os.path.exists(params.visMeshesDir):
  os.makedirs(params.visMeshesDir)

if not os.path.exists(params.snapshotDir):
  os.makedirs(params.snapshotDir)

cuboid_sampler = CuboidSurface(params.nSamplesChamfer, normFactor='Surf')

writer = SummaryWriter()

def train(netPred, optimizer, iter, batch_ip, batch_gt):
    predParts = netPred.forward(batch_ip)
    predParts = predParts.view(predParts.size(0), -1, 10)
    optimizer.zero_grad()

    loss, coverage, consistency = get_loss(predParts, cuboid_sampler, batch_gt, params.chamferLossWt)

    loss.backward()
    optimizer.step()
    return loss.item(), coverage.item(), consistency.item()



class Network2(nn.Module):
  def __init__(self, params):
    super(Network2, self).__init__()
    self.backbone = Encoder(params.num_points)

    # loead pretrained part
    pretrained_dict = torch.load(params.encoder_pretrain_dir)
    model_dict = self.backbone.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
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

  def forward(self, x):
    # input: x (bs, 1, 32, 32,32)
    encoding = self.backbone(x) # encoding (bs, 16, 1, 1, 1)
    for i in range(3):
      encoding = encoding.unsqueeze(-1)
    features = self.fc_layers(encoding) # features (bs, 16, 1, 1, 1)
    primitives = self.primitivesTable(features) # primitives (bs, nParts * 10)
    return primitives

netPred = Network2(params).cuda()

optimizer = torch.optim.Adam(netPred.parameters(), lr=params.learning_rate)

train_loss = 0
train_coverage = 0
train_consitency = 0

val_loss = 0
val_coverage = 0
val_consitency = 0

train_models, train_pair_indices = get_data_models(params.data_dir, params.category, params.num_views, 'train')
val_models, val_pair_indices = get_data_models(params.data_dir, params.category, params.num_views, 'val')

batches = len(train_pair_indices) // params.batch_size
# batches = 10

if __name__ == '__main__':
    steps = 0
    for iter in range(params.num_train_iter):
        for b in range(batches):
            print("Epoch:{}\tStep:{}:\tLoss:{:10.7f}\tTsdf:{:10.7f}\tChamfer:{:10.7f}".format(iter, b, train_loss, train_coverage, train_consitency))
            batch_ip, batch_gt, batch_name = fetch_batch_joint(train_models, train_pair_indices, b, params.batch_size)
            batch_ip = Variable(torch.tensor(batch_ip, dtype=torch.float32).cuda())
            batch_gt = Variable(torch.tensor(batch_gt, dtype=torch.float32).cuda())
            train_loss, train_coverage, train_consitency = train(netPred, optimizer, iter, batch_ip, batch_gt)

            writer.add_scalar("Loss/train_total_loss", train_loss, steps)
            writer.add_scalar("Loss/train_coverage_loss", train_coverage, steps)
            writer.add_scalar("Loss/train_consitency_loss", train_consitency, steps)

            steps += 1

        if iter % params.val_iter == 0:
            netPred.eval()
            batch_ip, batch_gt, batch_name = fetch_batch_joint(val_models, val_pair_indices, 0, params.batchSizeVis)
            batch_ip = Variable(torch.tensor(batch_ip, dtype=torch.float32).cuda())
            batch_gt = Variable(torch.tensor(batch_gt, dtype=torch.float32).cuda())
            shapePredParams = netPred.forward(batch_ip)
            shapePredParams = shapePredParams.view(params.batchSizeVis, params.nParts, 10)
            netPred.train()

            val_loss, val_coverage, val_consitency = get_loss(shapePredParams, cuboid_sampler, batch_gt, params.chamferLossWt)

            writer.add_scalar("Loss/val_total_loss", val_loss, steps)
            writer.add_scalar("Loss/val_coverage_loss", val_coverage, steps)
            writer.add_scalar("Loss/val_consitency_loss", val_consitency, steps)

            predParams = shapePredParams
            for b in range(0, params.batchSizeVis):

                pred_b = []
                for px in range(params.nParts):
                    pred_b.append(predParams[b, px, :].clone().data.cpu())

                mUtils.saveParts(pred_b,
                                 '{}/iter{}_inst{}_pred_{}.obj'.format(params.visMeshesDir, iter, b, batch_name[b]))

        if (iter % 50) == 0:
            torch.save(netPred.state_dict(), "{}/iter{}.pkl".format(params.snapshotDir, iter))

    torch.save(netPred.state_dict(), "{}/iter{}.pkl".format(params.snapshotDir, iter))
    writer.close()





