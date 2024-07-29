import csv

import pandas as pd
import torch.nn as nn
from sklearn import preprocessing
import torch
import torch.utils.data as Data
from scipy.stats import pearsonr
import pickle


EPOCH = 500
BATCH_SIZE = 402
LR = 5*1e-5
hiddenDim = [2048, 1024]
inDim = 20587  # 甲基化 20588
outDim = 128
cell_line_number = 402

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inDim, hiddenDim[0]),
            nn.BatchNorm1d(hiddenDim[0]),
            nn.ReLU(),
            nn.Linear(hiddenDim[0], hiddenDim[0]),
            nn.BatchNorm1d(hiddenDim[0]),
            nn.ReLU(),
            nn.Linear(hiddenDim[0], outDim),
            nn.BatchNorm1d(outDim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(outDim, hiddenDim[0]),
            nn.BatchNorm1d(hiddenDim[0]),
            nn.ReLU(),
            nn.Linear(hiddenDim[0], hiddenDim[0]),
            nn.BatchNorm1d(hiddenDim[0]),
            nn.ReLU(),
            nn.Linear(hiddenDim[0], inDim),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def load_data():
    all_data = torch.load('../data/cell_line/MethylationFeature402.pt')
    train_size = int(all_data.shape[0] * 0.8)
    data_train = all_data[:train_size]
    data_test = all_data[train_size:]

    return data_train, data_test, all_data


def train(train_data, test_data, data_all):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
    autoencoder = AutoEncoder().to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    best_loss = 1
    res = [[0 for col in range(outDim)] for row in range(cell_line_number)]
    for epoch in range(EPOCH):
        for step, data in enumerate(train_loader):
            data = data.to(device)
            encoded, decoded = autoencoder(data)
            loss = loss_func(decoded, data)  # mean square error
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy())

        test_data = test_data.to(device)
        test_en, test_de = autoencoder(test_data)
        test_loss = loss_func(test_de, test_data).cpu()
        pearson = pearsonr(test_de.view(-1).cpu().tolist(), test_data.view(-1).cpu())[0]
        best_pearson = pearson
        data_all = data_all.to(device)
        if test_loss < best_loss:
            best_loss = test_loss
            res, _ = autoencoder(data_all)
            torch.save(res.data, '../data/cell_line/{}dim_Methylation_AE.pt'.format(outDim))
            print("best_loss: ", best_loss.data.numpy())
            print("pearson: ", pearson)
        if  epoch == (EPOCH-1):
            print("训练结果--------")
            print("best_loss: ", best_loss.data.numpy())
            print("pearson: ", best_pearson)
    return


if __name__ == "__main__":
    train_data, test_data, all_data = load_data()
    train(train_data, test_data, all_data)
