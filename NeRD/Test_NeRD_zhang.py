import datetime

import pandas as pd
import torch.nn as nn
import torch
from torch_geometric.data import DataLoader
from functions import *
from sklearn.metrics import r2_score
import argparse
from model.NeRD_Net import NeRD_Net

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def predict(model, data_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    with torch.no_grad():
        for index, data in enumerate(data_loader):
            print("Batch size : {}".format(index+1))
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
    return total_preds.numpy().flatten()


def train(modeling, test_batch, i):

    test_data = TestbedDataset(root='data', dataset='test_set_unknown')
    test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)

    # training the model
    print(device)
    model = modeling().to(device)
    model.load_state_dict(torch.load('saveModel/NeRD_fold0.pth'))
    P_test = predict(model, test_loader)
    data = pd.read_csv('../data/unknownPairs/unknownPairsIndexData.csv')
    data['min_max_log-ic50'] = P_test
    data.to_csv('result/unKnown/NeRDUnknownPredLabel.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--test_batch', type=int, required=False, default=1024, help='Batch size test set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=200, help='Number of epoch')
    args = parser.parse_args()
    modeling = NeRD_Net
    test_batch = args.test_batch
    lr = args.lr
    num_epoch = args.num_epoch
    timeStr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i in range(1):
        train(modeling, test_batch, i)
