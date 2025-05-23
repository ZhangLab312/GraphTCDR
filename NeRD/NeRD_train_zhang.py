import datetime

import torch.nn as nn
import torch
from torch_geometric.data import DataLoader
from functions import *
from sklearn.metrics import r2_score
import argparse
from model.NeRD_Net import NeRD_Net


SaveResultAddress = '../SaveResult/compared/NerD_result.csv'


def train_step(model, device, train_loader, optimizer, epoch):
    print(epoch)
    model.train()
    loss_fun = nn.MSELoss()
    avg_loss = []

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fun(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % 40 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.y),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

    return sum(avg_loss)/len(avg_loss)


def predict(model, device, data_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def train(modeling, train_batch, val_batch, test_batch, lr, epoch_num, cuda_name, i):

    model_st = modeling.__name__
    train_losses = []
    val_losses = []
    val_pearsons = []
    val_r2 = []
    train_data = TestbedDataset(root='data', dataset='train_set{num}'.format(num=i))
    val_data = TestbedDataset(root='data', dataset='val_set{num}'.format(num=i))
    test_data = TestbedDataset(root='data', dataset='test_set{num}'.format(num=i))

    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(device)
    model = modeling().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_mse = 1
    best_epoch = -1
    test_save_metric = []

    for epoch in range(epoch_num):
        train_loss = train_step(model, device, train_loader, optimizer, epoch + 1)
        G, P = predict(model, device, val_loader)
        ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), r2_score(G, P)]

        G_test, P_test = predict(model, device, test_loader)
        ret_test = [rmse(G_test, P_test), mse(G_test, P_test), pearson(G_test, P_test), spearman(G_test, P_test), r2_score(G_test, P_test)]

        train_losses.append(train_loss)
        val_losses.append(ret[1])
        val_pearsons.append(ret[2])
        val_r2.append(ret[4])

        if ret[1] < best_mse:
            torch.save(model.state_dict(), "saveModel/NeRD_fold{}.pth".format(i))
            test_save_metric = ret_test
            best_epoch = epoch + 1
            best_rmse = ret[0]
            best_mse = ret[1]
            best_pearson = ret[2]
            print('improvement in epoch {} best_rmse:{}, best pearson:{}'.format(best_epoch, best_rmse, best_pearson))
        else:
            print(' no improvement since epoch ', best_epoch)

    test_save_metric.append('Test')
    test_save_metric.append('fold{}'.format(i))
    with open(SaveResultAddress.format(i), 'a') as f:
        f.write(','.join(map(str, test_save_metric)))
        f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--train_batch', type=int, required=False, default=1024, help='Batch size training set')
    parser.add_argument('--val_batch', type=int, required=False, default=1024, help='Batch size validation set')
    parser.add_argument('--test_batch', type=int, required=False, default=1024, help='Batch size test set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=200, help='Number of epoch')
    parser.add_argument('--cuda_name', type=str, required=False, default="cuda:0", help='Cuda')

    args = parser.parse_args()

    modeling = NeRD_Net
    train_batch = args.train_batch
    val_batch = args.val_batch
    test_batch = args.test_batch
    lr = args.lr
    num_epoch = args.num_epoch
    cuda_name = args.cuda_name
    timeStr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(SaveResultAddress, 'a') as f:
        f.write("begin Train datetime: {}\n".format(timeStr))
    for i in range(5):
        train(modeling, train_batch, val_batch, test_batch, lr, num_epoch, cuda_name, i)
