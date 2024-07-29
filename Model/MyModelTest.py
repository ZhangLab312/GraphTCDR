import os.path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GCN, GATConv, TransformerConv, HGTConv, Linear, HANConv

output_dim = 256

data = HeteroData()
data['cellLine'].x = ...
data['drug'].x = ...
data['cellLine', 'to', 'cellLine'].edge_index = ...
data['drug', 'to', 'drug'].edge_index = ...
data['cellLine', 'to', 'drug'].edge_index = ...


class MyModelTest(torch.nn.Module):
    def __init__(self, device, n_filters=2, output_dim=256, AEDim=256, num_layers=2, num_heads=16):  # num_layers=2
        super(MyModelTest, self).__init__()
        self.device = device
        self.output_dim = output_dim
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.cellLineSeq = nn.Sequential(
            # nn.Linear(AEDim * 2, AEDim * 4),
            nn.Linear(AEDim * 2, AEDim * 4),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.BatchNorm1d(AEDim * 4),
            nn.Linear(AEDim * 4, output_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim * 1), 
        )

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, output_dim)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv({'cellLine': output_dim, 'drug': output_dim}, output_dim, data.metadata(),
                           num_heads)
            self.convs.append(conv)

        # drug fingerprint
        self.fingerprintSeq1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=3),
            nn.Dropout(0.2),  # 0.1-》0.2
            nn.ReLU(),
            nn.BatchNorm1d(n_filters),  
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=3),
            nn.Dropout(0.2),  # 0.1-》0.2
            nn.ReLU(),
            nn.BatchNorm1d(n_filters * 2),  
            nn.MaxPool1d(3),
        )
        self.fingerprintSeq2 = nn.Sequential(
            nn.Linear(388, 800), 
            nn.Dropout(0.2),  # 0.1-》0.2
            nn.ReLU(),
            nn.BatchNorm1d(800),
            nn.Linear(800, int(output_dim / 1)),
            nn.Dropout(0.2), 
            nn.ReLU(),
            nn.BatchNorm1d(int(output_dim / 1))
        )

        # fusion layers
        self.combPredSeq = nn.Sequential(
            nn.Linear(2 * output_dim, 1 * output_dim),
            nn.Dropout(0.21), 
            nn.ReLU(),
            nn.Linear(1 * output_dim, int(output_dim / 2)),
            nn.Dropout(0.21),  
            nn.ReLU(),
            nn.Linear(int(output_dim / 2), 1),
            nn.Dropout(0.1),
            nn.Sigmoid()
        )

    def forward(self, gene_expression_feature, CNV_feature, cell_edge_idx,
                fingerprint_feature, drug_edge_index, train_pair_mask, cell_drug_edge_idx):
        cell_line_feature = torch.cat((gene_expression_feature, CNV_feature), 1)

        cell_line_feature = gene_expression_feature

        cell_line_feature = self.cellLineSeq(cell_line_feature)
        fingerprint_feature = torch.unsqueeze(fingerprint_feature, 1)
        fingerprint_feature = self.fingerprintSeq1(fingerprint_feature)
        fingerprint_feature = fingerprint_feature.view(-1, fingerprint_feature.shape[1] * fingerprint_feature.shape[2])
        fingerprint_feature = self.fingerprintSeq2(fingerprint_feature)
        drug_feature = fingerprint_feature

        MyData = HeteroData()
        MyData['cellLine'].x = cell_line_feature
        MyData['drug'].x = drug_feature
        MyData['cellLine', 'to', 'cellLine'].edge_index = cell_edge_idx
        MyData['drug', 'to', 'drug'].edge_index = drug_edge_index
        MyData['cellLine', 'to', 'drug'].edge_index = cell_drug_edge_idx
        MyData.to(self.device)

        for conv in self.convs:
            x_dict = conv(MyData.x_dict, MyData.edge_index_dict)

        drug_index, cell_index = train_pair_mask[:, 0], train_pair_mask[:, 1]
        cell_drug_pair = torch.cat((x_dict['cellLine'][cell_index.int()], x_dict['drug'][drug_index.int()]), 1)
        output = self.combPredSeq(cell_drug_pair)
        return output[:, 0]
