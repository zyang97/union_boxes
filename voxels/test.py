import os
import sys
sys.path.insert(0, '/home/nileshk/Research2/volumetricPrimitivesPytorch/')
import torch
import torch.nn as nn
import modules.volumeEncoder as vE
import modules.netUtils as netUtils
import modules.primitives as primitives
from torch.autograd import Variable
from data.cadConfigsChamfer import SimpleCadData
from modules.losses import tsdf_pred, chamfer_loss, chamfer_loss_img, emd_loss
from modules.cuboid import  CuboidSurface
import modules.meshUtils as mUtils

params = lambda x: 0

params.learningRate = 0.001
params.meshSaveIter = 10
params.numTrainIter = 40000
params.batchSize = 1
params.batchSizeVis = 1
params.visPower = 0.25
params.lossPower = 2
params.chamferLossWt = 1
params.symLossWt = 1
params.gridSize = 32
params.gridBound = 0.5
params.useBn = 1
params.nParts = 10
params.disp = 0
params.imsave = 0
params.shapeLrDecay = 0.01
params.probLrDecay = 0.0001
params.gpu = 1
params.visIter = 100
# params.modelIter = 100000  # data loader reloads models after these many iterations
params.modelIter = 2  # data loader reloads models after these many iterations
params.synset = '02828884'  # chair:03001627, aero:2691156, table:4379243 # buildings: `buildings_norm`
# params.synset = '03001628'  # chair:3001627, aero:2691156, table:4379243
params.name = 'bench'
params.bMomentum = 0.9  # baseline momentum for reinforce
params.entropyWt = 0
params.nullReward = 0.000
params.nSamplePoints = 1000
params.nSamplesChamfer = 150  # number of points we'll sample per part
params.useCubOnly = 0
params.usePretrain = True
params.normFactor = 'Surf'
params.pretrainNet = params.name
params.pretrainLrShape = 0.01
params.pretrainLrProb = 0.0001
params.pretrainIter = 39999
params.modelsDataDir = os.path.join('D:\\data\\processed_mat', params.synset)

params.snapshotDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\voxels\\cachedir\\snapshots')
params.visMeshesDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\voxels\\results\\visualization\\tests', params.name)

params.trainDatasetDir = os.path.join('D:\\data\\images\\data\\splits', '{}_train_list.txt'.format(params.synset))
params.valDatasetDir = os.path.join('D:\\data\\images\\data\\splits', '{}_val_list.txt'.format(params.synset))

if not os.path.exists(params.visMeshesDir):
    os.makedirs(params.visMeshesDir)

dataloader = SimpleCadData(params)
params.nz = 3
params.primTypes = ['Cu']
params.nPrimChoices = len(params.primTypes)
params.intrinsicReward = torch.Tensor(len(params.primTypes)).fill_(0)

cuboid_sampler = CuboidSurface(params.nSamplesChamfer, normFactor='Surf')
criterion  = nn.L1Loss()
def train(dataloader, netPred, optimizer, iter):
  inputVol, tsdfGt, sampledPoints, loaded_cps, fileNames, inputPoints = dataloader.forward()
  # voxel as input
  inputVol = Variable(inputVol.clone().cuda())
  sampledPoints = Variable(sampledPoints.cuda()) ## B x np x 3
  predParts = netPred.forward(inputVol) ## B x nPars*10
  predParts = predParts.view(predParts.size(0), -1, 10)

  optimizer.zero_grad()
  tsdfPred= tsdf_pred(sampledPoints, predParts)
  coverage_b = tsdfPred.mean(dim=1)
  coverage = coverage_b.mean()
  consistency = chamfer_loss(predParts, dataloader, cuboid_sampler)
  loss = coverage + params.chamferLossWt*consistency

  loss.backward()
  optimizer.step()
  return loss.item(), coverage.item(), consistency.item()

class Network(nn.Module):
  def __init__(self, params):
    super(Network, self).__init__()
    self.ve = vE.convEncoderSimple3d(3,4,1,params.useBn)
    outChannels = self.outChannels = self.ve.output_channels
    layers = []
    for i in range(2):
      layers.append(nn.Conv3d(outChannels, outChannels,kernel_size=1))
      layers.append(nn.BatchNorm3d(outChannels))
      layers.append(nn.LeakyReLU(0.2,True))

    self.fc_layers = nn.Sequential(*layers)
    self.fc_layers.apply(netUtils.weightsInit)

    biasTerms = lambda x:0

    biasTerms.quat = torch.Tensor([1, 0, 0, 0])
    biasTerms.shape = torch.Tensor(params.nz).fill_(-3) / params.shapeLrDecay
    biasTerms.prob = torch.Tensor(len(params.primTypes)).fill_(0)
    for p in range(len(params.primTypes)):
      if (params.primTypes[p] == 'Cu'):
        biasTerms.prob[p] = 2.5 / params.probLrDecay

    self.primitivesTable = primitives.Primitives(params, outChannels, biasTerms)

  def forward(self, x):
    # input: x (bs, 1, 32, 32,32)
    encoding  = self.ve(x) # encoding (bs, 16, 1, 1, 1)
    features = self.fc_layers(encoding) # features (bs, 16, 1, 1, 1)
    primitives = self.primitivesTable(features) # primitives (bs, nParts * 10)
    return primitives

netPred = Network(params)
netPred.cuda()

if params.usePretrain:
  updateShapeWtFunc = netUtils.scaleWeightsFunc(params.pretrainLrShape / params.shapeLrDecay, 'shapePred')
  updateProbWtFunc = netUtils.scaleWeightsFunc(params.pretrainLrProb / params.probLrDecay, 'probPred')
  updateBiasWtFunc = netUtils.scaleBiasWeights(params.probLrDecay, 'probPred')
  load_path = os.path.join(params.snapshotDir, params.pretrainNet, 'iter{}.pkl'.format(params.pretrainIter))
  netPretrain = torch.load(load_path)
  netPred.load_state_dict(netPretrain)
  print('Loading pretrained model from {}'.format(load_path))
  netPred.primitivesTable.apply(updateShapeWtFunc)
  netPred.primitivesTable.apply(updateProbWtFunc)
  # netPred.primitivesTable.apply(updateBiasWtFunc)

optimizer = torch.optim.Adam(netPred.parameters(), lr=params.learningRate)

nSamplePointsTrain = params.nSamplePoints
nSamplePointsTest = params.gridSize**3

def tsdfSqModTest(x):
  return torch.clamp(x,min=0).pow(2)


sample, fileNames, inputPoints = dataloader.forwardTestResult()

chamfer = 0
emd = 0

for i in range(len(sample)):
  print('Procees object {}'.format(i))
  netPred.eval()
  shapePredParams = netPred.forward(Variable(sample[i].cuda().unsqueeze(0)))
  shapePredParams = shapePredParams.view(params.batchSizeVis, params.nParts, 10)
  netPred.train()

  chamfer += chamfer_loss_img(shapePredParams, cuboid_sampler, inputPoints[i].cuda().unsqueeze(0))
  emd += emd_loss(shapePredParams, cuboid_sampler, inputPoints[i].cuda().unsqueeze(0))

  predParams = shapePredParams
  pred_b = []
  for px in range(params.nParts):
    pred_b.append(predParams[0, px, :].clone().data.cpu())

  mUtils.saveParts(pred_b, '{}/pred_{}.obj'.format(params.visMeshesDir, fileNames[i]))
  print('Chamfer mean is {}'.format(chamfer / (i+1)))
  print('EMD mean is {}'.format(emd / (i+1)))


