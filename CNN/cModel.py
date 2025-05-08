import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # 定义每个特征的卷积操作序列
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3),
            nn.ReLU(),
            # nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3),
            # nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3),
            nn.ReLU(),
            # nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3),
            # nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3),
            nn.ReLU(),
            # nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3),
            # nn.ReLU()
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3),
            nn.ReLU(),
            # nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3),
            # nn.ReLU(),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3),
            nn.ReLU(),
            # nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3),
            # nn.ReLU(),
        )

        self.conv_block6 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3),
            nn.ReLU(),
            # nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3),
            # nn.ReLU()

        )

        self.batchnorm = nn.BatchNorm1d(256)

        self.fc = nn.Sequential(
            nn.Linear(3396, 1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        (mirna, gene_expression, CNV, meth, phy, fp, _) = data
        # 对每个特征应用相应的卷积操作序列
        mirna = self.conv_block1(mirna)
        mirna = torch.flatten(mirna, start_dim=1, end_dim=2)

        gene_expression = self.conv_block2(gene_expression)
        gene_expression = torch.flatten(gene_expression, start_dim=1, end_dim=2)

        CNV = self.conv_block3(CNV)
        CNV = torch.flatten(CNV, start_dim=1, end_dim=2)

        meth = self.conv_block4(meth)
        meth = torch.flatten(meth, start_dim=1, end_dim=2)

        phy = self.conv_block5(phy)
        phy = torch.flatten(phy, start_dim=1, end_dim=2)

        fp = self.conv_block6(fp)
        fp = torch.flatten(fp, start_dim=1, end_dim=2)

        # 将四个特征的卷积结果拼接
        x = torch.cat((mirna, gene_expression, CNV, meth, phy, fp), dim=1)
        x = self.fc(x)
        return x
