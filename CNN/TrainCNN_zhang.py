import datetime
import os

import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import argparse
from torch.utils import data
from tqdm import tqdm

from CNN.cModel import ConvNet
from Model.MyModel import MyModel
from ToolsFun import rmse, mse

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
SaveResultAddress = '../SaveResult/compared/CNNResult.csv'
saveResultAndModel = True


def loadData(N, i):
    mirna_feature = torch.load('../data/cell_line/miRNAFeature402.pt')
    gene_expression_feature = torch.load('../data/cell_line/512dim_geneExpression_AE.pt')
    CNV_feature = torch.load('../data/cell_line/512dim_CNV_AE.pt')
    meth = torch.load('../data/cell_line/512dim_Methylation_AE.pt')

    phy = torch.load('../data/drug/physicochemicalFeature1353.pt')
    phy = torch.tensor(phy, dtype=torch.float)
    fingerprint_feature = torch.load('../data/drug/fingerPrintFeature1353.pt')
    fingerprint_feature = torch.tensor(fingerprint_feature, dtype=torch.float)

    mirna_feature = torch.unsqueeze(mirna_feature, 1).to(device)
    gene_expression_feature = torch.unsqueeze(gene_expression_feature, 1).to(device)
    CNV_feature = torch.unsqueeze(CNV_feature, 1).to(device)
    fingerprint_feature = torch.unsqueeze(fingerprint_feature, 1).to(device)
    meth = torch.unsqueeze(meth, 1).to(device)
    phy = torch.unsqueeze(phy, 1).to(device)

    drug_response_data = torch.tensor(pd.read_csv('../data/drug_response_data_number_log_4_5.csv').to_numpy())
    drug_response_data = drug_response_data[torch.randperm(drug_response_data.size(0))]

    MaskList = torch.chunk(drug_response_data, N, 0)
    train_dataset = torch.tensor([])
    test_val = torch.tensor([])
    for k in range(N):
        if k == i:
            test_val = MaskList[k]
        else:
            train_dataset = torch.cat([train_dataset, MaskList[k]], 0)
    train_mask = torch.tensor(train_dataset[:, 0:2], dtype=torch.int)
    train_label = train_dataset[:, 2].to(device)
    tvLen = int(test_val.shape[0] / 2)
    test_label = test_val[:tvLen, 2].to(device)
    test_mask = torch.tensor(test_val[:tvLen, 0:2], dtype=torch.int)
    val_label = test_val[tvLen:, 2].to(device)
    val_mask = torch.tensor(test_val[tvLen:, 0:2], dtype=torch.int)
    trainDataset = data.TensorDataset(mirna_feature[train_mask[:, 1]], gene_expression_feature[train_mask[:, 1]],
                                      CNV_feature[train_mask[:, 1]], meth[train_mask[:, 1]],
                                      phy[train_mask[:, 0]], fingerprint_feature[train_mask[:, 0]], train_label)

    valDataset = data.TensorDataset(mirna_feature[val_mask[:, 1]], gene_expression_feature[val_mask[:, 1]],
                                    CNV_feature[val_mask[:, 1]], meth[val_mask[:, 1]],
                                    phy[val_mask[:, 0]], fingerprint_feature[val_mask[:, 0]], val_label)

    testDataset = data.TensorDataset(mirna_feature[test_mask[:, 1]], gene_expression_feature[test_mask[:, 1]],
                                     CNV_feature[test_mask[:, 1]], meth[test_mask[:, 1]],
                                     phy[test_mask[:, 0]], fingerprint_feature[test_mask[:, 0]], test_label)

    return trainDataset, valDataset, testDataset


def train(model, Lr, epochs, batchSize, i):
    model = model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Lr)
    best_rmse = 1
    best_pearson = 0
    best_SCC = 0
    best_r2_score = 0
    best_epoch = -1
    # 加载到device
    trainDataset, valDataset, testDataset = loadData(5, i)
    trainDataLoader = data.DataLoader(trainDataset, batchSize, shuffle=True)
    valDataLoader = data.DataLoader(valDataset, batchSize, shuffle=True)
    testDataLoader = data.DataLoader(testDataset, batchSize, shuffle=True)
    loss_fun = nn.MSELoss()
    test_save_metric = []
    val_save_metric = []

    # train model
    for epoch in tqdm(range(epochs)):
        model.train()
        for batch_index, TrainData in enumerate(trainDataLoader):
            optimizer.zero_grad()
            output = model(TrainData)
            train_loss = loss_fun(output.view(-1, 1).to(torch.float), TrainData[6].view(-1, 1).to(torch.float))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            if batch_index % 60 == 0:
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
            for ValData in valDataLoader:
                val_output = model(ValData)
                val_total_pred = torch.cat((val_total_pred, val_output.cpu()), 0)
                val_total_labels = torch.cat((val_total_labels, ValData[6].cpu()), 0)

        val_metric = [rmse(val_total_labels, val_total_pred[:, 0]),
                      (pearsonr(val_total_labels.tolist(), val_total_pred[:, 0].tolist()))[0],
                      (spearmanr(val_total_labels.tolist(), val_total_pred[:, 0].tolist()))[0],
                      r2_score(val_total_labels.tolist(), val_total_pred[:, 0].tolist())]
        if val_metric[0] < best_rmse:
            # if saveResultAndModel:
            #     torch.save(model.state_dict(), 'SaveModel/model_fold{}.pth'.format(i))
            val_save_metric = val_metric
            # 目前最好效果进行test
            with torch.no_grad():
                for Testdata in testDataLoader:
                    test_output = model(Testdata)
                    test_total_pred = torch.cat((test_total_pred, test_output.cpu()), 0)
                    test_total_labels = torch.cat((test_total_labels, Testdata[6].cpu()), 0)

            test_metric = [rmse(test_total_labels, test_total_pred[:, 0]),
                           (pearsonr(test_total_labels.tolist(), test_total_pred[:, 0].tolist()))[0],
                           (spearmanr(test_total_labels.tolist(), test_total_pred[:, 0].tolist()))[0],
                           r2_score(test_total_labels.tolist(), test_total_pred[:, 0].tolist())]
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
    parser.add_argument('--lr', type=float, required=False, default=5*1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=100, help='Number of epoch')
    parser.add_argument('--batch_size', type=int, required=False, default=1024, help='batch_size')
    args = parser.parse_args()
    modeling = ConvNet
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
