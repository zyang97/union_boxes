from scipy.io import loadmat
import glob
import os
import pdb
import torch
import modules.primitives as primitives
import numpy as np
from torch.autograd import Variable

## Dataloader for Volumetric Primitives

class OBJ:
  def __init__(self, vertices, lines_index):

    self.vertices = vertices
    self.lines = [[line[0], line[1]] for line in lines_index]

  def save_obj(self, output_dir):
    with open(output_dir, 'w') as f:
      f.write('# OBJ file\n')
      for v in self.vertices:
        f.write('v {0} {1} {2}\n'.format(v[0], v[1], v[2]))
      for l in self.lines:
        f.write('l {0} {1}\n'.format(int(l[0]), int(l[1])))

def load_file_name(source):
  files = os.listdir(source)
  fileNames = []
  for file in files:
    fileNames.append(file.split('.')[0])
  return fileNames

class SimpleCadData(object):
  def __init__(self, params):
    self.gridSize = params.gridSize
    self.modelsDir = params.modelsDataDir
    self.modelIter = params.modelIter
    self.batchSize = params.batchSize
    self.modelSize = params.gridSize
    self.nSamplePoints = params.nSamplePoints
    self.gridBound = params.gridBound

    self.iter = 0
    self.startModelIndex = 0


    self.modelNames = []
    self.fileNames = []

    # for filename in glob.iglob(self.modelsDir + '/*.mat'):
    #   self.modelNames.append(filename)
    #   self.fileNames.append(os.path.splitext(os.path.basename(filename))[0])
    # self.fileNames = np.array(self.fileNames)

    self.fileNames = load_file_name('D:\\projects\\UnionBox2\\test2\\imgs')
    # self.fileNames = load_file_name('D:\\projects\\volumetricPrimitivesPytorch\\cachedir\\buildings_norm')
    for fileName in self.fileNames:
      modelName = os.path.join(self.modelsDir, fileName + '.mat')
      self.modelNames.append(modelName)
    self.fileNames = np.array(self.fileNames)

    ## Limit to 200 chairs
    # self.modelNames = self.modelNames[0:200]

    self.loadedVoxels = torch.Tensor(self.batchSize, 1, self.modelSize, self.modelSize, self.modelSize).fill_(0)
    self.loadedTsdfs = torch.Tensor(self.batchSize, 1, self.modelSize, self.modelSize, self.modelSize).fill_(0)
    self.loadedCPs = torch.Tensor(self.batchSize, 1, self.modelSize, self.modelSize, self.modelSize, 3).fill_(0)
    self.loadedSurfaceSamples = {}

    gridMin = -params.gridBound + params.gridBound / params.gridSize
    gridMax = params.gridBound - params.gridBound / params.gridSize

    meshGridInit = primitives.meshGrid([gridMin, gridMin, gridMin], [gridMax, gridMax, gridMax],
                                       [params.gridSize, params.gridSize, params.gridSize])
    meshGrid = meshGridInit.repeat(params.batchSize, 1, 1, 1, 1)
    self.gridPoints = meshGrid.view(params.batchSize, params.gridSize**3, 3).clone()
    self.shape_mats = dict()
    self.load_all_mats()

    self.load_torch_tensors()
    self.outSampleTsfds = torch.Tensor(self.batchSize, self.nSamplePoints).fill_(0).cuda()
    self.global_index = 0
    # for i in range(self.all_surfaceSamples.size(0)):
    #   obj = OBJ(self.all_surfaceSamples[i], [])
    #   path = "D:\\projects\\experiment\\volumetricPrimitivesPytorch\\cachedir\\chairs_point_clouds"
    #   obj.save_obj(os.path.join(path, self.fileNames[i] + ".obj"))
    #
    # print('load data done.')

  def load_all_mats(self):
    for ix in range(len(self.modelNames)):
      model_name = self.modelNames[ix]
      self.shape_mats[model_name] = loadmat(model_name)

  def load_torch_tensors(self):
    self.all_tsdfs = []
    self.all_volumes = []
    self.all_closetPoints = []
    self.all_surfaceSamples = []

    for model_file, shape in self.shape_mats.items():
      self.all_tsdfs.append(torch.from_numpy(shape['tsdf']).float())
      self.all_volumes.append(torch.from_numpy(shape['Volume']).float())
      self.all_closetPoints.append(torch.from_numpy(shape['closestPoints']).float())
      self.all_surfaceSamples.append(torch.from_numpy(shape['surfaceSamples']).float())

    self.all_tsdfs = torch.stack(self.all_tsdfs).cuda()
    self.all_volumes = torch.stack(self.all_volumes).unsqueeze(1).cuda()
    self.all_closetPoints = torch.stack(self.all_closetPoints).unsqueeze(1).cuda()
    self.all_surfaceSamples = torch.stack(self.all_surfaceSamples).cuda()


  # def reloadShapes(self):
  #   for ix in range(self.batchSize):
  #     self.startModelIndex = np.random.randint(0, len(self.modelNames))
  #     shape_dict = self.shape_mats[self.modelNames[self.startModelIndex]]
  #     # shape_dict = loadmat(self.modelNames[self.startModelIndex])#,
  #       # {'Volume', 'tsdf', 'surfaceSamples', 'closestPoints'})
  #     shape = lambda x:0
  #     shape.tsdf = torch.from_numpy(shape_dict['tsdf']).float()
  #     shape.Volume = torch.from_numpy(shape_dict['Volume']).float()
  #     shape.closestPoints = torch.from_numpy(shape_dict['closestPoints']).float()
  #     shape.surfaceSamples = torch.from_numpy(shape_dict['surfaceSamples']).float()
  #     shape.vertices = shape_dict['vertices']
  #     shape.faces = shape_dict['faces']
  #     self.loadedVoxels[ix][0].copy_(shape.Volume)
  #     self.loadedTsdfs[ix][0].copy_(shape.tsdf)
  #     self.loadedCPs[ix][0].copy_(shape.closestPoints)
  #     self.loadedSurfaceSamples[ix] = shape.surfaceSamples.clone()
  #   self.loadedShapes = self.loadedVoxels.clone()

  def reloadShapes(self):
    ids = []
    for ix in range(self.batchSize):
      self.startModelIndex = np.random.randint(0, len(self.modelNames))
      shape_dict = self.shape_mats[self.modelNames[self.startModelIndex]]
      ids.append(self.startModelIndex)

    ids = torch.LongTensor(ids).cuda()
    # pdb.set_trace()
    self.loadedVoxels = self.all_volumes[ids]
    self.loadedTsdfs = self.all_tsdfs[ids]
    self.loadedCPs = self.all_closetPoints[ids]
    self.loadedSurfaceSamples = self.all_surfaceSamples[ids]
    self.loadedShapes = self.loadedVoxels
    self.loadedFileNames = self.fileNames[ids.cpu()]

  # def forward(self):
  #   if (self.iter % self.modelIter == 0):
  #     self.reloadShapes()
  #   self.iter = self.iter + 1
  #   outSampleTsfds = torch.Tensor(self.batchSize, self.nSamplePoints).fill_(0)
  #   outSamplePoints = torch.Tensor(self.batchSize, self.nSamplePoints, 3)
  #
  #   for b in range(self.batchSize):
  #     nPointsTot = self.loadedSurfaceSamples[b].size(1)
  #     for ns in range(self.nSamplePoints):
  #       pId = np.random.randint(nPointsTot)
  #       outSamplePoints[b][ns] = self.loadedSurfaceSamples[b][pId]
  #   output = [self.loadedShapes.clone(), outSampleTsfds, outSamplePoints, self.loadedCPs]
  #   return output

  def forward(self):
    if (self.iter % self.modelIter == 0):
      self.reloadShapes()
    self.iter = self.iter + 1

    outSamplePoints = []
    # pdb.set_trace()
    for b in range(self.batchSize):
      nPointsTot = self.loadedSurfaceSamples[b].size(0)
      sample_ids = torch.LongTensor(np.random.randint(0, nPointsTot, self.nSamplePoints)).cuda()
      outSamplePoints.append(self.loadedSurfaceSamples[b][sample_ids])
    outSamplePoints = torch.stack(outSamplePoints)
    output = [self.loadedShapes, self.outSampleTsfds, outSamplePoints, self.loadedCPs, self.loadedFileNames, self.loadedSurfaceSamples]
    return output

  def forwardTest(self):
    if self.iter % self.modelIter == 0:
      self.reloadShapes()
    self.iter  = self.iter + 1
    outTsfds = self.loadedTsdfs.view(self.batchSize, self.gridSize**3)
    outPoints = self.gridPoints.clone()
    return self.loadedShapes, outTsfds, outPoints, self.loadedFileNames, self.loadedSurfaceSamples

  def reloadShapesSequential(self):
    ids = []
    max_samples = len(self.all_volumes)
    if max_samples < self.global_index:
      return False

    ids = torch.LongTensor([i for i in range(self.global_index, self.global_index+self.batchSize)])
    ids = torch.clamp(ids, 0, max_samples-1).long().cuda()
    self.global_index += self.batchSize
    self.loadedVoxels = self.all_volumes[ids]
    self.loadedTsdfs = self.all_tsdfs[ids]
    self.loadedCPs = self.all_closetPoints[ids]
    self.loadedSurfaceSamples = self.all_surfaceSamples[ids]
    self.loadedShapes = self.loadedVoxels
    return True

  def forwardTestSequential(self):
    status = self.reloadShapesSequential()
    if not status:
      return  None, None, None
    outTsfds = self.loadedTsdfs.view(self.batchSize, self.gridSize**3)
    outPoints = self.gridPoints.clone()
    return self.loadedShapes, outTsfds, outPoints


  def chamfer_forward(self, queryPoints):
    #query points is B x nQ x 3
    neighbourIds = self.pointClosestCellIndex(queryPoints).data
    # queryDiffs = queryPoints.clone()
    # queryDiffs.data.fill_(0)
    loadedCPs = Variable(self.loadedCPs.cuda())
    queryDiffs = []
    cps = []
    batch_size = queryPoints.size(0)

    for b in range(queryPoints.size(0)):
      inds = neighbourIds[b]
      inds = self.gridSize*self.gridSize*inds[:,0]  + self.gridSize*inds[:,1] + inds[:,2]
      cp = loadedCPs[b,0].view(-1,3)
      cp = cp[inds]
      voxels = Variable(self.loadedVoxels[b][0].view(-1))
      voxels = voxels[inds]
      diff = (cp - queryPoints[b].view(-1,3)).pow(2).sum(1)
      queryDiffs.append((-voxels+1) * diff)
    queryDiffs = torch.stack(queryDiffs)

    # for b in range(queryPoints.size(0)):
    #   for np in range(queryPoints.size(1)):
    #     ind = neighbourIds[b][np]
    #     if self.loadedVoxels[b][0][ind[0]][ind[1]][ind[2]] == 0:
    #       cp = loadedCPs[b][0][ind[0]][ind[1]][ind[2]]
    #       queryDiffs.append(queryPoints[b][np] - cp)
    #     else:
    #       queryDiffs.append(Variable(torch.zeros(3)).cuda())
    # queryDiffs = torch.stack(queryDiffs)
    # queryDiffs = queryDiffs.view(queryPoints.size())

    # queryDiffs = Variable(torch.FloatTensor(queryPoints.size()).type_as(queryPoints.data).fill_(0))
    # outDists = queryDiffs.pow(2).sum(2)
    outDists = queryDiffs
    self.queryDiffs = queryDiffs
    return outDists

  def pointClosestCellIndex(self, points):
    gridMin = -self.gridBound + self.gridBound / self.gridSize
    gridMax = self.gridBound - self.gridBound / self.gridSize
    inds = (points - gridMin) * self.gridSize / (2 * self.gridBound)
    inds = torch.round(torch.clamp(inds, min=0, max=self.gridSize-1)).long()
    return inds

# if __name__ == '__main__':
#   import open3d as o3d
#   mesh_path = 'D:\\data\\chairs\\03001627\\1a6f615e8b1b5ae4dbbc9440457e303e\\model.obj'
#   mesh = o3d.io.read_triangle_mesh(mesh_path)
#   print(mesh)