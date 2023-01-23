import torch
import torch.nn as nn
import modules.netUtils as nUtils
import pdb

def meshGrid(minVal, maxVal, gridSize):
  # Xs, Ys, Zs = MeshGrid
  pointsX = torch.linspace(minVal[0], maxVal[0], gridSize[0])
  pointsY = torch.linspace(minVal[1], maxVal[1], gridSize[1])
  pointsZ = torch.linspace(minVal[2], maxVal[2], gridSize[2])
  # xs = torch.repeat(pointsX.view(-1, 1, 1, 1), 1, gridSize[1], gridSize[2], 1)
  # ys = torch.repeat(pointsY.view(1, -1, 1, 1), gridSize[0], 1, gridSize[2], 1)
  # zs = torch.repeat(pointsZ.view(1, 1, -1, 1), gridSize[0], gridSize[1], 1, 1)

  xs = pointsX.view(-1, 1, 1, 1).repeat(1, gridSize[1], gridSize[2], 1)
  ys = pointsY.view(1, -1, 1, 1).repeat(gridSize[0], 1, gridSize[2], 1)
  zs = pointsZ.view(1, 1, -1, 1).repeat(gridSize[0], gridSize[1], 1, 1)
  return torch.cat([xs, ys, zs],dim=3)



class ShapePredModule(nn.Module):
  def __init__(self, params):
    self.shapeModule = nn.Sequential()


class ShapePredModule(nn.Module):
  def __init__(self, params, outChannelsV, biasTerms=None):
    super(ShapePredModule, self).__init__()
    shapeLayer = nn.Conv3d(outChannelsV, params.nz, kernel_size=1)
    self.shapeLrDecay = params.shapeLrDecay
    shapeLayer.apply(nUtils.weightsInit)
    shapeLayer.note = 'shapePred'
    self.gridBound = params.gridBound
    biasTerms = biasTerms
    if biasTerms is not None:
      shapeLayer.bias.data = biasTerms.shape.clone()
    self.shapeLayer = shapeLayer
    self.sigmoid = nn.Sigmoid()

  def forward(self, feature):
    x = self.shapeLayer(feature)
    x = x * self.shapeLrDecay
    x = self.sigmoid(x)
    x = x * self.gridBound
    x = x.view(feature.size(0), -1)
    return x

class QuatPredModule(nn.Module):
  def __init__(self, params, outChannelsV, biasTerms=None):
    super(QuatPredModule, self).__init__()
    quatLayer = nn.Conv3d(outChannelsV, 4, kernel_size=1)
    self.shapeLrDecay = params.shapeLrDecay or 1
    quatLayer.apply(nUtils.weightsInit)
    quatLayer.note = 'quatLayer'

    biasTerms = biasTerms
    try:
      quatLayer.bias.data = biasTerms.quat.clone()
    except AttributeError:
      print("No bias init for qaut")
    self.quatLayer = quatLayer

  def forward(self, feature):
    x = self.quatLayer(feature)
    x = x.view(x.size(0), -1)
    x = normalize(x)
    return x


def normalize(x):
  x  = x/ (1E-12 + torch.norm(x, dim=1, p=2, keepdim=True).expand(x.size()))
  return x

class TransPredModule(nn.Module):
  def __init__(self, params, outChannelsV, biasTerms=None):
    super(TransPredModule, self).__init__()
    transLayer = nn.Conv3d(outChannelsV, params.nz, kernel_size=1)
    # transLayer.apply(nUtils.weightsInit)
    transLayer.note = 'transPred'
    self.gridBound = params.gridBound
    biasTerms = biasTerms
    try:
      transLayer.bias.data = biasTerms.trans.clone()
    except AttributeError:
      assert True
      # print("No bias init for trans")
    self.transLayer = transLayer
    self.tanh = nn.Tanh()

  def forward(self, feature):
    x = self.transLayer(feature)
    x = self.tanh(x)
    x = x * self.gridBound
    x = x.view(x.size(0), -1)
    return x

class PrimitivePredModule(nn.Module):
  def __init__(self, params, outChannelsV, biasTerm):
    super(PrimitivePredModule, self).__init__()
    self.shapePred = ShapePredModule(params, outChannelsV, biasTerm)
    self.quatPred = QuatPredModule(params, outChannelsV, biasTerm)
    self.transPred = TransPredModule(params, outChannelsV, biasTerm)

  def forward(self, feature):
    shape = self.shapePred(feature)
    quat = self.quatPred(feature)
    transPred = self.transPred(feature)
    output = torch.cat([shape, transPred, quat], dim = 1)
    return output


class Primitives(nn.Module):
  def __init__(self, params, outChannelsV, biasTerms):
    super(Primitives, self).__init__()
    self.primitivePredModules = nn.ModuleList([PrimitivePredModule(params, outChannelsV, biasTerms)
                                               for i in range(params.nParts)])

  def forward(self, feature):
    primitives = []
    for i, l in enumerate(self.primitivePredModules):
      primitives.append(l(feature))

    return torch.cat(primitives, dim=1)  ## BatchSize x nParts*10