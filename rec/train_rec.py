import os
import torch
from torch.autograd import Variable
from third_party.chamfer_distance.chamfer_distance import ChamferDistance

from net import Encoder
from dataloader import get_data_models, fetch_batch_joint
from tensorboardX import SummaryWriter

# data
params = lambda x: 0
params.data_dir = 'D:\\data\\images\\data'
params.num_views = 10
params.batch_size = 32
params.category = 'table'
params.num_points = 1024

# trainer
params.learning_rate = 0.0005
params.num_train_iter = 500
params.vis_iter = 10
params.name = 'table'
params.snapshotDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\rec\\cachedir\\snapshots', params.name)
params.visMeshesDir = os.path.join('D:\\projects\\experiment\\volumetricPrimitivesPytorch\\rec\\cachedir\\visualization\\meshes', params.name)
if not os.path.exists(params.snapshotDir):
    os.makedirs(params.snapshotDir)

if not os.path.exists(params.visMeshesDir):
    os.makedirs(params.visMeshesDir)

writer = SummaryWriter(comment=params.name)

def train(netPred, optimizer, iter, batch_ip, batch_gt):
    _, out_pcl = netPred.forward(batch_ip)
    optimizer.zero_grad()

    chd = ChamferDistance()
    dist1, dist2, idx1, idx2 = chd(batch_gt, out_pcl)
    loss = (torch.mean(dist1)) + (torch.mean(dist2))

    loss.backward()
    optimizer.step()
    return loss.item()

netPred = Encoder(params.num_points).cuda()

optimizer = torch.optim.Adam(netPred.parameters(), lr=params.learning_rate)

loss = 0

train_models, train_pair_indices = get_data_models(params.data_dir, params.category, params.num_views, 'train')
val_models, val_pair_indices = get_data_models(params.data_dir, params.category, params.num_views, 'val')

batches = len(train_pair_indices) // params.batch_size


if __name__ == '__main__':
    steps = 0
    for iter in range(params.num_train_iter):
        for b in range(batches):
            print("Epoch:{}\tStep:{}:\tLoss:{:10.7f}".format(iter, b, loss))
            batch_ip, batch_gt, batch_name = fetch_batch_joint(train_models, train_pair_indices, b, params.batch_size)
            batch_ip = Variable(torch.tensor(batch_ip, dtype=torch.float32).cuda())
            batch_gt = Variable(torch.tensor(batch_gt, dtype=torch.float32).cuda())
            loss = train(netPred, optimizer, iter, batch_ip, batch_gt)

            writer.add_scalar("Loss/train_loss", loss, steps)

        if iter % params.vis_iter == 0:
            netPred.eval()
            batch_ip, batch_gt, batch_name = fetch_batch_joint(val_models, val_pair_indices, 0, 32)
            batch_ip = Variable(torch.tensor(batch_ip, dtype=torch.float32).cuda())
            batch_gt = Variable(torch.tensor(batch_gt, dtype=torch.float32).cuda())
            _, out_pcl = netPred.forward(batch_ip)
            netPred.train()

            chd = ChamferDistance()
            dist1, dist2, idx1, idx2 = chd(batch_gt, out_pcl)
            loss = (torch.mean(dist1)) + (torch.mean(dist2))
            writer.add_scalar("Loss/val_loss", loss, steps)


            from data.cadConfigsChamfer import OBJ
            for idx in range(3):
                idx_ = 10*idx
                obj = OBJ(out_pcl[idx], [])
                obj.save_obj(os.path.join(params.visMeshesDir, "pred_iter_{}_{}.obj".format(iter, batch_name[idx_])))

        if (iter % 50) == 0:
            torch.save(netPred.state_dict(), "{}/iter{}.pkl".format(params.snapshotDir, iter))

        steps += 1

    torch.save(netPred.state_dict(), "{}/iter{}.pkl".format(params.snapshotDir, iter))
    writer.close()





