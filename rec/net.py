import torch
from torch import nn

def conv2d_block(in_c, out_c, kernel):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel, stride=2, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )

def linear_block(in_c, out_c):
    return nn.Sequential(
        nn.Linear(in_c, out_c),
        nn.BatchNorm1d(out_c),
        nn.ReLU(),
    )

class Encoder(nn.Module):
    def __init__(self, num_points):
        super(Encoder, self).__init__()
        self.num_points = num_points
        self.conv1 = conv2d_block(3, 32, 3)
        self.conv2 = conv2d_block(32, 64, 3)
        self.conv3 = conv2d_block(64, 128, 3)
        self.conv4 = conv2d_block(128, 256, 5)
        self.fc1 = nn.Linear(2304, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=0.4)
        # self.fc1 = linear_block(2304, 128)
        # self.fc2 = linear_block(128, 128)
        # self.fc3 = linear_block(128, 128)
        # self.fc4 = linear_block(128, num_points*3)
        self.fc4 = nn.Linear(256, self.num_points*3)
        self.bn4 = nn.BatchNorm1d(self.num_points*3)


    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # [bsize, 3, 64, 64]
        x = self.conv1(x) # [bsize, 32, 32, 32]
        x = self.conv2(x) # [bsize, 64, 16, 16]
        x = self.conv3(x) # [bsize, 128, 8, 8]
        x = self.conv4(x) # [bsize, 256, 3, 3]
        x = x.reshape(x.size(0), -1) # [32, 256, 3, 3]
        x = self.relu(self.bn1(self.fc1(x))) # [bsize, 1024]
        x = self.relu(self.bn2(self.fc2(x))) # [bsize, 512]
        x = self.relu(self.bn3(self.fc3(x))) # [bsize, 256]
        inter_pcl = self.bn4(self.fc4(x))
        inter_pcl = inter_pcl.view(x.size(0), self.num_points, 3)
        return x, inter_pcl

if __name__ == '__main__':
    model = Encoder(1024).cuda()
    image = torch.rand(32, 64, 64, 3).cuda()
    output = model(image)
