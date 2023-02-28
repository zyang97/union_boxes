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
from modules.losses import tsdf_pred, chamfer_loss, shape_loss
from modules.cuboid import  CuboidSurface
import pdb
from modules.plotUtils import  plot3, plot_parts, plot_cuboid
import modules.marching_cubes as mc
import modules.meshUtils as mUtils
from modules.meshUtils import  savePredParts
params = lambda x: 0

params.learningRate = 0.001
params.meshSaveIter = 10
params.numTrainIter = 40000
params.batchSize = 32
params.batchSizeVis = 32
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
params.usePretrain = False
params.normFactor = 'Surf'
params.pretrainNet = params.name
params.pretrainLrShape = 0.01
params.pretrainLrProb = 0.0001
params.pretrainIter = 19999
params.modelsDataDir = os.path.join('D:\\data\\processed_mat', params.synset)

params.visDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\voxels\\cachedir\\visualization', params.name)
params.visMeshesDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\voxels\\cachedir\\visualization\\meshes', params.name)
params.snapshotDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\voxels\\cachedir\\snapshots', params.name)

params.trainDatasetDir = os.path.join('D:\\data\\images\\data\\splits', '{}_train_list.txt'.format(params.synset))
params.valDatasetDir = os.path.join('D:\\data\\images\\data\\splits', '{}_val_list.txt'.format(params.synset))

dataloader = SimpleCadData(params)
params.nz = 3
params.primTypes = ['Cu']
params.nPrimChoices = len(params.primTypes)
params.intrinsicReward = torch.Tensor(len(params.primTypes)).fill_(0)

if not os.path.exists(params.visDir):
  os.makedirs(params.visDir)

if not os.path.exists(params.visMeshesDir):
  os.makedirs(params.visMeshesDir)

if not os.path.exists(params.snapshotDir):
  os.makedirs(params.snapshotDir)

params.primTypesSurface = []
for p in range(len(params.primTypes)):
    params.primTypesSurface.append(params.primTypes[p])


cuboid_sampler = CuboidSurface(params.nSamplesChamfer, normFactor='Surf')
criterion  = nn.L1Loss()
def train(dataloader, netPred, optimizer, iter):
  inputVol, tsdfGt, sampledPoints, loaded_cps, fileNames, inputPoints = dataloader.forward()
  # voxel as input
  inputVol = Variable(inputVol.clone().cuda())
  sampledPoints = Variable(sampledPoints.cuda()) ## B x np x 3
  predParts = netPred.forward(inputVol) ## B x nPars*10
  predParts = predParts.view(predParts.size(0), -1, 10)

  # point cloud as input
  # inputPoints = Variable(inputPoints.clone().cuda())
  # sampledPoints = Variable(sampledPoints.cuda())  ## B x np x 3
  # predParts = netPred.forward(inputPoints)  ## B x nPars*10
  # predParts = predParts.view(predParts.size(0), -1, 10)

  optimizer.zero_grad()
  tsdfPred= tsdf_pred(sampledPoints, predParts)
  # coverage = criterion(tsdfPred, tsdfGt)
  coverage_b = tsdfPred.mean(dim=1)
  coverage = coverage_b.mean()
  consistency = chamfer_loss(predParts, dataloader, cuboid_sampler)
  # consistency = torch.tensor([0.0]).cuda()
  # shape_ = shape_loss(predParts)
  loss = coverage + params.chamferLossWt*consistency

  # if iter % 200 == 0:
  #   # pdb.set_trace()
  #   # plot3(sampledPoints[0].data.cpu())
  #   for i in range(4):
  #     savePredParts(predParts[i], 'train_preds/train_{}_{}.obj'.format(i, fileNames[i]))

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

loss = 0
coverage = 0
consitency = 0
shape = 0

import torch.nn.functional as F

def tsdfSqModTest(x):
  return torch.clamp(x,min=0).pow(2)




print("Iter\tErr\tTSDF\tChamf")
for iter  in range(params.numTrainIter):
  print("{:10.7f}\t{:10.7f}\t{:10.7f}\t{:10.7f}".format(iter, loss, coverage, consitency))
  loss, coverage, consitency = train(dataloader, netPred, optimizer, iter)

  if iter % params.visIter ==0:
    reshapeSize = torch.Size([params.batchSizeVis, 1, params.gridSize, params.gridSize, params.gridSize])

    sample, fileNames, inputPoints = dataloader.forwardTest()

    sample = sample[0:params.batchSizeVis].cuda()
    fileNames = fileNames[0:params.batchSizeVis]

    netPred.eval()
    shapePredParams = netPred.forward(Variable(sample))
    shapePredParams = shapePredParams.view(params.batchSizeVis, params.nParts, 10)
    netPred.train()

    if iter % params.meshSaveIter == 0:

      predParams = shapePredParams
      for b in range(0, params.batchSizeVis):

        # visTriSurf = mc.march(tsdfGt[b][0].cpu().numpy())
        # mc.writeObj('{}/iter{}_inst{}_gt.obj'.format(params.visMeshesDir ,iter, b), visTriSurf)


        pred_b = []
        for px in range(params.nParts):
          pred_b.append(predParams[b,px,:].clone().data.cpu())

        mUtils.saveParts(pred_b, '{}/iter{}_inst{}_pred_{}.obj'.format(params.visMeshesDir, iter, b, fileNames[b]))

    if (iter % 1000) == 0 :
      torch.save(netPred.state_dict() ,"{}/iter{}.pkl".format(params.snapshotDir,iter))

torch.save(netPred.state_dict() ,"{}/iter{}.pkl".format(params.snapshotDir,iter))