import datetime
import os
from random import random

import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import argparse
import torch.nn.functional as F


from torch.utils import data
from tqdm import tqdm
from Model.MyModelTest2 import MyModelTest2
from ToolsFun import rmse, mse
from early_stop import EarlyStopping

device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
SaveResultAddress = 'SaveResult/TrainTest2WeightResult.csv'
saveResultAndModel = True


def loadData():
    mirna_feature = torch.load('data/cell_line/miRNAFeature402.pt', map_location=device)
    mirna_feature = torch.unsqueeze(mirna_feature, 1)
    gene_expression_feature = torch.load('data/cell_line/512dim_geneExpression_CNV_MiRNA_AE.pt', map_location=device)
    CNV_feature = torch.load('data/cell_line/256dim_CNV_AE.pt', map_location=device)
    cell_edge_idx = torch.load('data/cell_line/cell_line_edge_index_0.75_Meth.pt', map_location=device)
    cell_drug_edge_idx = torch.load('data/cell_drug/cell_line_drug_edge_index_0.5.pt', map_location=device)
    DrugSMILESGraph = torch.load('data/drug/DrugSMILESGraph1353.pt', map_location=device)
    for index in range(1353):
        DrugSMILESGraph.iloc[index][1] = torch.tensor(np.array(DrugSMILESGraph.iloc[index][1])).to(device)
        DrugSMILESGraph.iloc[index][2] = torch.tensor(np.array(DrugSMILESGraph.iloc[index][2])).to(device)

    fingerprint_feature = torch.load('data/drug/fingerPrintFeature1353.pt')
    # fingerprint_feature = torch.unsqueeze(fingerprint_feature, 1)
    fingerprint_feature = fingerprint_feature.to(torch.float)  # 转化类型
    # drug_edge_index = torch.load('data/drug/drug_edge_index_0.75_phy.pt').to(device)
    drug_edge_index = torch.load('data/drug/drug_edge_index_0.75_fp.pt').to(device)
    return (gene_expression_feature, CNV_feature,
            fingerprint_feature, cell_edge_idx, drug_edge_index, cell_drug_edge_idx)


def loadResponseMask(N, i):
    drug_response_data = torch.tensor(pd.read_csv('data/drug_response_data_number_log_4_5.csv').to_numpy())
    drug_response_data = drug_response_data[torch.randperm(drug_response_data.size(0))]


    MaskList = torch.chunk(drug_response_data, N, 0)
    train_dataset = torch.tensor([])
    test_val = torch.tensor([])
    for k in range(N):
        if k == i:
            test_val = MaskList[k]
        else:
            train_dataset = torch.cat([train_dataset, MaskList[k]], 0)
    train_mask = train_dataset[:, 0:2].to(device)
    train_label = train_dataset[:, 2].to(device)
    tvLen = int(test_val.shape[0] / 2)
    test_label = test_val[:tvLen, 2].to(device)
    test_mask = test_val[:tvLen, 0:2].to(device)
    val_label = test_val[tvLen:, 2].to(device)
    val_mask = test_val[tvLen:, 0:2].to(device)
    trainDataset = data.TensorDataset(train_mask, train_label)
    valDataset = data.TensorDataset(val_mask, val_label)
    testDataset = data.TensorDataset(test_mask, test_label)
    return trainDataset, valDataset, testDataset


def train(model, Lr, epochs, batchSize, i):
    model = model(device).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=Lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Lr)
    best_rmse = 1
    best_pearson = 0
    best_SCC = 0
    best_r2_score = 0
    best_epoch = -1
    (gene_expression_feature, CNV_feature,
     fingerprint_feature, cell_edge_idx, drug_edge_index, cell_drug_edge_idx) = loadData()
    # 加载到device
    gene_expression_feature = gene_expression_feature.to(device)
    CNV_feature = CNV_feature.to(device)
    fingerprint_feature = fingerprint_feature.to(device)
    cell_edge_idx = cell_edge_idx.to(device)
    drug_edge_index = drug_edge_index.to(device)

    trainDataset, valDataset, testDataset = loadResponseMask(5, i)
    trainDataLoader = data.DataLoader(trainDataset, batchSize, shuffle=True)
    valDataLoader = data.DataLoader(valDataset, batchSize, shuffle=True)
    testDataLoader = data.DataLoader(testDataset, batchSize, shuffle=True)
    loss_fun = nn.MSELoss()
    test_save_metric = []
    val_save_metric = []

    early_stop = EarlyStopping(40)
    # train model
    for epoch in tqdm(range(epochs)):
        model.train()
        for batch_index, (X, y) in enumerate(trainDataLoader):
            optimizer.zero_grad()
            output = model(gene_expression_feature, CNV_feature,
                           cell_edge_idx, fingerprint_feature, drug_edge_index, X, cell_drug_edge_idx)

            weight = torch.ones(y.shape[0])
            weight[(y >= 0) & (y < 0.2)] = 6
            weight[(y >= 0.2) & (y < 0.5)] = 4
            weight[(y >= 0.5) & (y < 0.6)] = 3
            weight[(y >= 0.6) & (y < 0.9)] = 1
            weight[(y >= 0.9) & (y <= 1)] = 4

            weight = weight.to(device)

  
            alpha = 0.4
            weighted_l1_loss = F.l1_loss(output.float(), y.float(), reduction='none') * weight
            weighted_mse_loss = F.mse_loss(output.float(), y.float(), reduction='none') * weight
            train_loss = torch.mean(weighted_mse_loss*alpha + (1-alpha) * weighted_l1_loss)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            if batch_index % 30 == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_index * batchSize,
                                                                               len(trainDataset),
                                                                               100. * batch_index / len(trainDataLoader)
                                                                               , train_loss.item()))
        val_total_pred = torch.Tensor()
        val_total_labels = torch.Tensor()
        test_total_pred = torch.Tensor()
        test_total_labels = torch.Tensor()

        # eval model
        model.eval()
        with torch.no_grad():
            for X, y in valDataLoader:
                val_output = model(gene_expression_feature, CNV_feature,
                                   cell_edge_idx, fingerprint_feature, drug_edge_index, X, cell_drug_edge_idx)
                val_total_pred = torch.cat((val_total_pred, val_output.cpu()), 0)
                val_total_labels = torch.cat((val_total_labels, y.cpu()), 0)

        val_metric = [rmse(val_total_labels, val_total_pred),
                      (pearsonr(val_total_labels.tolist(), val_total_pred.tolist()))[0],
                      (spearmanr(val_total_labels.tolist(), val_total_pred.tolist()))[0],
                      r2_score(val_total_labels.tolist(), val_total_pred.tolist())]
        early_stop(val_metric[0], model=model, metric=False)
        if val_metric[0] < best_rmse:
            if saveResultAndModel:
                torch.save(model.state_dict(), 'SaveModel/model_test2_weight_fold{}.pth'.format(i))
            val_save_metric = val_metric
            # 目前最好效果进行test
            with torch.no_grad():
                for X, y in testDataLoader:
                    test_output = model(gene_expression_feature, CNV_feature,
                                        cell_edge_idx, fingerprint_feature, drug_edge_index, X, cell_drug_edge_idx)
                    test_total_pred = torch.cat((test_total_pred, test_output.cpu()), 0)
                    test_total_labels = torch.cat((test_total_labels, y.cpu()), 0)

            test_metric = [rmse(test_total_labels, test_total_pred),
                           (pearsonr(test_total_labels.tolist(), test_total_pred.tolist()))[0],
                           (spearmanr(test_total_labels.tolist(), test_total_pred.tolist()))[0],
                           r2_score(test_total_labels.tolist(), test_total_pred.tolist())]
            test_save_metric = test_metric
            best_epoch = epoch
            best_rmse = val_metric[0]
            best_pearson = val_metric[1]
            best_SCC = val_metric[2]
            best_r2_score = val_metric[3]
            print('improvement epoch {}; best_rmse:{:.4f}, best PCC:{:.4f}, best SCC:{:.4f},'
                  ' best_r2_score:{:.4f}'.format(best_epoch, best_rmse, best_pearson, best_SCC, best_r2_score))
            print('test_metric:', test_metric)
        else:
            print('no improvement since epoch {};best_rmse:{:.4f}'.format(best_epoch, best_rmse))
        if early_stop.early_stop:
            break
    val_save_metric.append('Val')
    val_save_metric.append('fold{}'.format(i))
    test_save_metric.append('Test')
    test_save_metric.append('fold{}'.format(i))
    if saveResultAndModel:
        with open(SaveResultAddress, 'a') as f:
            f.write(','.join(map(str, val_save_metric)))
            f.write('\n')
            f.write(','.join(map(str, test_save_metric)))
            f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--lr', type=float, required=False, default=8.0*1e-4, help='Learning rate') # 8
    parser.add_argument('--num_epoch', type=int, required=False, default=200, help='Number of epoch')
    parser.add_argument('--batch_size', type=int, required=False, default=1024, help='batch_size')
    args = parser.parse_args()
    modeling = MyModelTest
    lr = args.lr
    num_epoch = args.num_epoch
    batch_size = args.batch_size

    os.environ['PYTHONHASHSEED'] = str(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    np.random.seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


    timeStr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if saveResultAndModel:
        with open(SaveResultAddress, 'a') as f:
            f.write("begin Train datetime: {}\n".format(timeStr))

    for fold in range(5):
        train(modeling, lr, num_epoch, batch_size, fold)
