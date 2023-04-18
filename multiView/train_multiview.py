import os
import torch
from torch.autograd import Variable
from modules.losses import get_loss, get_aux_loss
from modules.cuboid import CuboidSurface
import modules.meshUtils as mUtils
from multiView.model import MultiViewNetwork

from rec.dataloader import ShapeNetMultiViewDataset

from tensorboardX import SummaryWriter


params = lambda x: 0

# model
params.model = 'train'
params.nSamplesChamfer = 150  # number of points we'll sample per part
params.nz = 3
params.shapeLrDecay = 0.01
params.primTypes = ['Cu']
params.probLrDecay = 0.0001
params.gridBound = 0.5
params.nParts = 10
params.chamferLossWt = 1

# data
params.data_dir = 'D:\\data\\images\\data'
params.num_views = 10
params.batch_size = 32
params.category = 'aero'
params.num_points = 1024

# trainer
params.learning_rate = 1e-4
params.num_train_iter = 500
params.val_iter = 10
params.name = params.category + '_multi_view_p10_n10_sep_no_aux'
params.encoder_pretrain_dir = 'D:\\projects\\experiment\\volumetricPrimitivesPytorch\\rec\\cachedir\\snapshots\\' + params.category + '\\iter150.pkl'

params.visMeshesDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\multiView\\cachedir\\visualization\\meshes', params.name)
params.visPCLDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\multiView\\cachedir\\visualization\\inter_pcls', params.name)
params.snapshotDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\multiView\\cachedir\\snapshots', params.name)

if not os.path.exists(params.visMeshesDir):
    os.makedirs(params.visMeshesDir)

if not os.path.exists(params.visPCLDir):
    os.makedirs(params.visPCLDir)

if not os.path.exists(params.snapshotDir):
    os.makedirs(params.snapshotDir)

cuboid_sampler = CuboidSurface(params.nSamplesChamfer, normFactor='Surf')

writer = SummaryWriter(comment=params.name)

def train(netPred, optimizer, batch_ip, batch_gt):
    predParts, inter_output = netPred.forward(batch_ip)
    predParts = predParts.view(predParts.size(0), -1, 10)
    optimizer.zero_grad()
    aux_loss = get_aux_loss(inter_output, batch_gt)
    loss, coverage, consistency = get_loss(predParts, cuboid_sampler, batch_gt, params.chamferLossWt)
    # comments aux_loss for ablation study
    #loss += aux_loss

    loss.backward()
    optimizer.step()
    return loss.item(), coverage.item(), consistency.item(), aux_loss.item()

netPred = MultiViewNetwork(params).cuda()
optimizer = torch.optim.Adam(netPred.parameters(), lr=params.learning_rate)

train_loss = 0
train_coverage = 0
train_consitency = 0
train_aux = 0

val_loss = 0
val_coverage = 0
val_consitency = 0
val_aux = 0

train_dataloader = ShapeNetMultiViewDataset(params, 'train')
val_dataloader = ShapeNetMultiViewDataset(params, 'val')

batches = len(train_dataloader) // params.batch_size

if __name__ == '__main__':
    steps = 0
    for iter in range(params.num_train_iter):
        for b in range(batches):
            print("Epoch:{}\tStep:{}:\tLoss:{:10.7f}\tTsdf:{:10.7f}\tChamfer:{:10.7f}\tAux:{:10.7f}".format(iter, b, train_loss, train_coverage, train_consitency, train_aux))
            batch_ip, batch_gt, batch_name = train_dataloader[b]
            batch_ip = Variable(batch_ip)
            batch_gt = Variable(batch_gt)
            train_loss, train_coverage, train_consitency, train_aux = train(netPred, optimizer, batch_ip, batch_gt)

            writer.add_scalar("Loss/train_total_loss", train_loss, steps)
            writer.add_scalar("Loss/train_coverage_loss", train_coverage, steps)
            writer.add_scalar("Loss/train_consitency_loss", train_consitency, steps)
            writer.add_scalar("Loss/train_aux_loss", train_aux, steps)

            steps += 1

        if iter % params.val_iter == 0:
            netPred.eval()
            batch_ip, batch_gt, batch_name = val_dataloader[0]
            batch_ip = Variable(batch_ip)
            batch_gt = Variable(batch_gt)
            shapePredParams, inter_output = netPred.forward(batch_ip)
            shapePredParams = shapePredParams.view(shapePredParams.size(0), -1, 10)
            netPred.train()

            val_aux_loss = get_aux_loss(inter_output, batch_gt)
            val_loss, val_coverage, val_consitency = get_loss(shapePredParams, cuboid_sampler, batch_gt, params.chamferLossWt)

            writer.add_scalar("Loss/val_total_loss", val_loss, steps)
            writer.add_scalar("Loss/val_coverage_loss", val_coverage, steps)
            writer.add_scalar("Loss/val_consitency_loss", val_consitency, steps)
            writer.add_scalar("Loss/val_aux_loss", val_aux_loss, steps)

            predParams = shapePredParams
            for b in range(0, predParams.size(0)):

                # Save intermediate output point cloud
                # from data.cadConfigsChamfer import OBJ
                # obj = OBJ(inter_output[b], [])
                # obj.save_obj(os.path.join(params.visPCLDir,
                #                           'iter{}_inst{}_pred_{}.obj'.format(iter, b, batch_name[b])))

                # Save predicted result
                pred_b = []
                for px in range(params.nParts):
                    pred_b.append(predParams[b, px, :].clone().data.cpu())

                mUtils.saveParts(pred_b,
                                 '{}/iter{}_inst{}_pred_{}.obj'.format(params.visMeshesDir, iter, b, batch_name[b]))

        # Save weights
        if (iter % 20) == 0:
            torch.save(netPred.state_dict(), "{}/iter{}.pkl".format(params.snapshotDir, iter))

    # Save weights
    torch.save(netPred.state_dict(), "{}/iter{}.pkl".format(params.snapshotDir, iter))
    writer.close()