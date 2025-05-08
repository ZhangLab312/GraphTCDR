import torch
import torch.nn as nn


# 定义多层感知器模型
class MLP(nn.Module):
    def __init__(self, output_dim=64):
        super(MLP, self).__init__()

        # 分子指纹
        self.fpFC = nn.Sequential(
            nn.Linear(881, 1024),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        self.phyFC = nn.Sequential(
            nn.Linear(269, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        self.geneExpFC = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        self.CnvFC = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        self.MethFC = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        self.MiRnaFC = nn.Sequential(
            nn.Linear(734, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        self.fc = nn.Sequential(
            nn.Linear(output_dim*6, output_dim*8),
            nn.Dropout(0.2),
            nn.Linear(output_dim*8, output_dim*2),
            nn.Dropout(0.2),
            nn.Linear(output_dim*2, output_dim*1),
            nn.Dropout(0.2),
            nn.Linear(output_dim*1, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        (mirna, gene_expression, CNV, meth, phy, fp, _) = data
        # 对每个特征应用相应的卷积操作序列
        mirna = self.MiRnaFC(mirna)
        gene_expression = self.geneExpFC(gene_expression)
        CNV = self.CnvFC(CNV)
        meth = self.MethFC(meth)
        phy = self.phyFC(phy)
        fp = self.fpFC(fp)
        # 将四个特征的卷积结果拼接
        x = torch.cat((mirna, gene_expression, CNV, meth, phy, fp), dim=1)
        x = self.fc(x)
        return x
